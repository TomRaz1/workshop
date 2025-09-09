#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch SVM runner over all fractional_occupancy_* files inside combination_results.

Filename assumptions (unchanged from dwell_time version)
-------------------------------------------------------
Subject label files in your data/labels directory are named like:
    l_b_sub-2_bibiindictment_bn246.npy
    l_n_sub-3_neutral_bn246.npy
    r_b_sub-7_bibiindictment_bn246.npy
    r_n_sub-11_neutral_bn246.npy

i.e., they all match:
    ^(?P<label>[lr])_(?:b|n)_sub-(?P<id>\d+).*\.npy$

Where:
  - [lr] is the hand/direction label ('l' or 'r') that becomes y ∈ {'l','r'}
  - (b|n) indicates the condition (e.g., bibi/neutral) but is ignored for y
  - sub-<ID> is a 1-based subject ID
  - any extra tokens after the ID are allowed and ignored by the loader

Row alignment convention for fractional_occupancy matrices
----------------------------------------------------------
We DO NOT assume that subject ID k is stored at row (k-1).
Instead, we build a stable mapping ID -> row index (0..N-1) from the
sorted list of discovered subject IDs, and require that the fractional_occupancy
file contains exactly one row per discovered subject (order can differ).
This avoids errors when IDs are sparse, e.g., {1,4,7}.

What the script does
--------------------
1) Scans the labels directory to build (ids, y, id2idx), aligned by ascending subject IDs.
2) Iterates over each combination folder under combination_results, finds any
   fractional_occupancy_*.npy files (e.g., fractional_occupancy_pca.npy / fractional_occupancy_umap.npy / fractional_occupancy_ae.npy),
   and for each: runs an SVM with GridSearchCV (Stratified 5-fold, scoring=ROC-AUC).
3) Aggregates metrics (Accuracy, F1(pos='r'), ROC-AUC, best params) into a single CSV.

How to run
----------
You can run with defaults (set below) simply as:
    python run_svm_batch_fractional.py

Or override paths/seed from CLI:
    python run_svm_batch_fractional.py       --data_dir   /path/to/data_or_labels       --comb_dir   /path/to/combination_results       --out_csv    /path/to/all_svm_results_fractional.csv       --labels_dir /path/to/labels_if_separate       --random_state 42
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ------------------------- Defaults (edit to your environment) -------------------------

DEFAULTS = {
    # Folder with per-subject label files like: l_b_sub-2_*.npy / l_n_sub-3_*.npy / r_b_sub-7_*.npy / r_n_sub-11_*.npy
    "DATA_DIR": "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/data/bibi_246_flat",

    # Root folder that contains the combination subfolders
    "COMB_DIR": "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/combination_results",

    # Where to save the aggregated CSV
    "OUT_CSV": "/home/yandex/0368352201_BrainWS2025b/tomraz/all_svm_results_fractional.csv",

    # If labels live elsewhere, set a path string; otherwise leave as None to use DATA_DIR
    "LABELS_DIR": None,

    # CV shuffle seed
    "RANDOM_STATE": 42,
}


# ------------------------- Label utilities -------------------------

# Accept l_b / l_n / r_b / r_n; capture label = l|r, and sub-<ID>
LABEL_PAT = re.compile(r'^(?P<label>[lr])_(?:b|n)_sub-(?P<id>\d+).*\.npy$', re.IGNORECASE)


def find_label_files(base_dir: str) -> List[str]:
    """Return list of full paths to subject label files matching LABEL_PAT."""
    return [
        fp for fp in glob.glob(os.path.join(base_dir, "*.npy"))
        if LABEL_PAT.match(os.path.basename(fp))
    ]


def collect_ids_and_labels(labels_dir: str) -> Tuple[List[int], np.ndarray, Dict[int, int]]:
    """
    Scan labels_dir for files named like l_b_sub-<ID>_*.npy / l_n_sub-<ID>_*.npy / r_b_... / r_n_...
    and return:
      - ids: sorted list of subject IDs (ints, ascending)
      - y:   numpy array of labels ('l'/'r') aligned to ids
      - id2idx: mapping from subject ID to a stable row index (0..N-1),
                where index is the position in the sorted ids list.

    We DO NOT assume that row index == (ID - 1).
    """
    rows: List[Tuple[int, str]] = []
    for fp in find_label_files(labels_dir):
        m = LABEL_PAT.match(os.path.basename(fp))
        lab = m.group("label").lower()   # 'l' or 'r'
        sid = int(m.group("id"))         # 1-based subject ID
        rows.append((sid, lab))

    if not rows:
        raise RuntimeError(
            f"No subject label files found in '{labels_dir}'. "
            "Expected names like l_b_sub-2_*.npy, l_n_sub-3_*.npy, r_b_sub-7_*.npy, r_n_sub-11_*.npy"
        )

    # Sort by numeric ID to define a stable subject order and mapping
    rows.sort(key=lambda x: x[0])
    ids = [sid for (sid, _) in rows]
    y = np.array([lab for (_, lab) in rows], dtype=object)
    id2idx = {sid: i for i, sid in enumerate(ids)}
    return ids, y, id2idx


def build_X_by_ids(frac_path: str, ids: List[int], id2idx: Dict[int, int]) -> np.ndarray:
    """
    Load fractional_occupancy matrix and stack rows in the order of provided IDs using id2idx.
    This does NOT assume that row == (ID-1).

    Requirements:
      - F has shape [num_subjects x num_states]
      - num_subjects must equal len(ids)

    If there's a mismatch in the number of rows, raise a clear error so the
    caller can verify the file corresponds to the same subject set.
    """
    F = np.load(frac_path)
    if F.ndim != 2:
        raise ValueError(f"'{frac_path}' must be 2D, got shape {F.shape}")

    if F.shape[0] != len(ids):
        raise ValueError(
            f"Row mismatch for '{os.path.basename(frac_path)}': "
            f'fractional_occupancy has {F.shape[0]} rows but discovered {len(ids)} subjects from labels. '
            "Make sure the fractional_occupancy file was computed on the same subject set (order can differ)."
        )

    # Reorder rows to match the stable subject order (ids)
    X = np.vstack([F[id2idx[sid]] for sid in ids])
    return X


# ------------------------- SVM runner -------------------------

def run_svm_cv(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Dict[str, Any]:
    """
    Train an SVM with GridSearchCV (5-fold, scoring=ROC-AUC).
    Returns a dict with metrics and best params.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, class_weight="balanced")),
    ])

    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10, 100]},
        {"svc__kernel": ["rbf"],    "svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]},
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)

    best = grid.best_estimator_
    # Cross-validated predictions with the best hyperparameters (no leakage).
    y_pred  = cross_val_predict(best, X, y, cv=cv, n_jobs=-1, method="predict")
    y_proba = cross_val_predict(best, X, y, cv=cv, n_jobs=-1, method="predict_proba")[:, 1]

    metrics = {
        "cv_best_mean_roc_auc": float(grid.best_score_),
        "cv_acc": float(accuracy_score(y, y_pred)),
        "cv_f1_pos_r": float(f1_score(y, y_pred, pos_label="r")),
        "cv_roc_auc_pos_r": float(roc_auc_score((y == "r").astype(int), y_proba)),
        "best_params": grid.best_params_,
    }
    return metrics


# ------------------------- Combination parsing -------------------------

def parse_combination_name(name: str) -> Dict[str, Optional[str]]:
    """
    Parse a combination folder name like:
       2025-08-11_data_40_hamming_1_k3_pca_32
    into a dict of useful fields. The format may vary; we keep it robust.
    """
    info = {
        "date": None,
        "data_tag": None,
        "win_size": None,
        "win_func": None,
        "step": None,
        "k": None,
        "method": None,
        "method_params": None,
        "raw_folder": name,
    }
    parts = name.split("_")
    # Try to extract a date at the beginning (YYYY-MM-DD)
    if parts and re.match(r"^\d{4}-\d{2}-\d{2}$", parts[0]):
        info["date"] = parts[0]
        parts = parts[1:]

    # Heuristic parsing for the remaining tokens
    # e.g., data 40 hamming 1 k3 pca 32
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok == "data" and i + 1 < len(parts):
            info["data_tag"] = parts[i+1]
            i += 2
            continue
        if tok.isdigit() and info["win_size"] is None:
            info["win_size"] = tok
            i += 1
            continue
        if tok in ("hamming", "hann", "boxcar", "rect", "rectangular") and info["win_func"] is None:
            info["win_func"] = tok
            i += 1
            continue
        if tok.isdigit() and info["step"] is None:
            info["step"] = tok
            i += 1
            continue
        m = re.match(r"^k(?P<k>\d+)$", tok, re.IGNORECASE)
        if m and info["k"] is None:
            info["k"] = m.group("k")
            i += 1
            continue
        # method + optional params (like ae_512_256_32 or pca_32 or umap_1000_64_30)
        if tok in ("ae", "pca", "umap", "kmeans", "kmeans-l2"):
            info["method"] = tok
            # collect trailing numeric tokens as method params
            params = []
            j = i + 1
            while j < len(parts) and re.match(r"^\d+$", parts[j]):
                params.append(parts[j])
                j += 1
            info["method_params"] = "_".join(params) if params else None
            i = j
            continue
        i += 1

    return info


# ------------------------- Fractional file discovery -------------------------

def find_fractional_files(comb_dir: str) -> List[str]:
    """Return all fractional_occupancy*.npy files inside a combination folder (non-recursive)."""
    candidates = glob.glob(os.path.join(comb_dir, "fractional_occupancy*.npy"))
    return [fp for fp in candidates if os.path.basename(fp).startswith("fractional_occupancy")]


# ------------------------- Main loop -------------------------

def main():
    # Args are optional; they default to DEFAULTS above
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULTS["DATA_DIR"], help="Folder containing per-subject label files.")
    ap.add_argument("--comb_dir", default=DEFAULTS["COMB_DIR"], help="Root folder of combination_results.")
    ap.add_argument("--out_csv",  default=DEFAULTS["OUT_CSV"],  help="Path to save the aggregated SVM results CSV.")
    ap.add_argument("--labels_dir", default=DEFAULTS["LABELS_DIR"], help="Optional separate labels folder (else use data_dir).")
    ap.add_argument("--random_state", type=int, default=DEFAULTS["RANDOM_STATE"])
    args = ap.parse_args()

    labels_dir = args.labels_dir or args.data_dir

    # Sanity checks + echo config
    for pth, desc in [(args.data_dir, "data_dir/labels_dir (for subject files)"),
                      (args.comb_dir, "comb_dir (combination_results root)")]:
        if not os.path.isdir(pth):
            raise FileNotFoundError(f"Path not found: {pth}  <-- {desc}")

    print("[Config]")
    print(f"  data_dir   = {args.data_dir}")
    print(f"  labels_dir = {labels_dir}")
    print(f"  comb_dir   = {args.comb_dir}")
    print(f"  out_csv    = {args.out_csv}")
    print(f"  seed       = {args.random_state}")

    # 1) Build y, IDs, and stable ID->row mapping
    ids, y, id2idx = collect_ids_and_labels(labels_dir)

    # 2) For each combination subfolder, find fractional_occupancy_*.npy and run SVM.
    rows: List[Dict[str, Any]] = []
    comb_subdirs = sorted(
        [d for d in glob.glob(os.path.join(args.comb_dir, "*")) if os.path.isdir(d)]
    )

    # Counters for summary
    num_combos_with_frac = 0
    total_frac_attempts = 0
    total_success = 0

    for comb_path in comb_subdirs:
        comb_name = os.path.basename(comb_path.rstrip(os.sep))
        meta = parse_combination_name(comb_name)

        fractional_files = find_fractional_files(comb_path)
        if not fractional_files:
            continue

        num_combos_with_frac += 1

        for frac_fp in sorted(fractional_files):
            total_frac_attempts += 1
            try:
                # Build X aligned to the stable subject order (ids) using id2idx
                X = build_X_by_ids(frac_fp, ids, id2idx)
                # Run SVM CV
                res = run_svm_cv(X, y, random_state=args.random_state)
                F = np.load(frac_fp)

                rows.append({
                    **meta,
                    "frac_file": os.path.basename(frac_fp),
                    "num_subjects": int(F.shape[0]),
                    "num_states": int(F.shape[1]),
                    "cv_best_mean_roc_auc": float(res["cv_best_mean_roc_auc"]),
                    "cv_acc": float(res["cv_acc"]),
                    "cv_f1_pos_r": float(res["cv_f1_pos_r"]),
                    "cv_roc_auc_pos_r": float(res["cv_roc_auc_pos_r"]),
                    # store best params as compact JSON
                    "best_params": json.dumps(res["best_params"], ensure_ascii=False),
                    "error": None,
                })
                total_success += 1

            except Exception as e:
                # Keep going even if one combo fails; record the error.
                rows.append({
                    **meta,
                    "frac_file": os.path.basename(frac_fp),
                    "num_subjects": None,
                    "num_states": None,
                    "cv_best_mean_roc_auc": None,
                    "cv_acc": None,
                    "cv_f1_pos_r": None,
                    "cv_roc_auc_pos_r": None,
                    "best_params": None,
                    "error": str(e),
                })
                continue

    # 3) Save results table
    df = pd.DataFrame(
        rows,
        columns=[
            "date","data_tag","win_size","win_func","step","k","method","method_params",
            "raw_folder","frac_file","num_subjects","num_states",
            "cv_best_mean_roc_auc","cv_acc","cv_f1_pos_r","cv_roc_auc_pos_r",
            "best_params","error"
        ]
    )
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved results to: {args.out_csv}")

    # 4) Final summary printout
    print("\n[Summary]")
    print(f"  combinations discovered: {len(comb_subdirs)}")
    print(f"  combinations with fractional_occupancy files: {num_combos_with_frac}")
    print(f"  fractional_occupancy files attempted: {total_frac_attempts}")
    print(f"  successful SVM runs: {total_success}")
    print("Done.")


if __name__ == "__main__":
    main()
