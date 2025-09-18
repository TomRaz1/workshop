#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch SVM runner over all fractional_occupancy_* files inside sub-directories combination_results.
Adapted to use labels.xlsx with columns: person_id, group.
Row 0 in fractional_occupancy = subject_id = FIRST_SUBJECT_ID (default 0).
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


# ------------------------- Defaults -------------------------

DEFAULTS = {
    "COMB_DIR": "path to combination directory",
    "OUT_CSV":  "./all_svm_results_fractional.csv",
    "RANDOM_STATE": 42,
    "LABELS_XLSX": "./labels.xlsx",
    "FIRST_SUBJECT_ID": 1, 
    "POS_CLASS": "A",
}


# ------------------------- Label utilities -------------------------

def collect_ids_and_labels_from_xlsx(xlsx_path: str) -> Tuple[List[int], np.ndarray, Dict[int, int]]:
    """
    Read labels.xlsx with columns: person_id, group.
    Returns ids (sorted), y (np.array of group strings), and id2idx mapping.
    """
    df = pd.read_excel(xlsx_path)
    if not {"person_id", "group"}.issubset(df.columns):
        raise RuntimeError(f"'{xlsx_path}' must have columns: person_id, group")

    df = df.dropna(subset=["person_id", "group"])
    df["person_id"] = df["person_id"].astype(int)
    df["group"] = df["group"].astype(str)
    df = df.sort_values("person_id")

    ids = df["person_id"].tolist()
    y = df["group"].to_numpy(dtype=object)
    id2idx = {sid: i for i, sid in enumerate(ids)}
    return ids, y, id2idx


def build_X_by_ids_with_offset(frac_path: str, ids: List[int], first_subject_id: int) -> Tuple[np.ndarray, List[int]]:
    """
    Map row i in fractional_occupancy to subject_id = first_subject_id + i.
    Stack rows in the order of 'ids' (filtering any IDs outside the file's row range).
    """
    F = np.load(frac_path)
    if F.ndim != 2:
        raise ValueError(f"'{frac_path}' must be 2D, got shape {F.shape}")

    min_id = first_subject_id
    max_id = first_subject_id + F.shape[0] - 1
    in_range_ids = [sid for sid in ids if min_id <= sid <= max_id]
    if not in_range_ids:
        raise ValueError(
            f"No overlapping subject IDs between labels and '{os.path.basename(frac_path)}' "
            f"(file rows cover IDs [{min_id}..{max_id}])."
        )

    def id_to_row(sid: int) -> int:
        return sid - first_subject_id

    rows = [F[id_to_row(sid)] for sid in in_range_ids]
    X = np.vstack(rows)
    return X, in_range_ids


# ------------------------- SVM runner -------------------------

def run_svm_cv(X: np.ndarray, y: np.ndarray, random_state: int = 42, pos_label: Optional[str] = None) -> Dict[str, Any]:
    """
    SVM + GridSearchCV. scoring='accuracy' to support multi-class;
    if there are exactly 2 classes and pos_label is provided — also compute ROC-AUC.
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
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)

    best = grid.best_estimator_
    y_pred  = cross_val_predict(best, X, y, cv=cv, n_jobs=-1, method="predict")

    metrics = {
        "cv_best_mean_acc": float(grid.best_score_),
        "cv_acc": float(accuracy_score(y, y_pred)),
        "cv_f1_macro": float(f1_score(y, y_pred, average="macro")),
        "cv_roc_auc": None,
        "best_params": grid.best_params_,
    }

    classes = np.unique(y)
    if len(classes) == 2 and pos_label is not None and pos_label in classes:
        y_proba = cross_val_predict(best, X, y, cv=cv, n_jobs=-1, method="predict_proba")[:, list(best.classes_).index(pos_label)]
        metrics["cv_roc_auc"] = float(roc_auc_score((y == pos_label).astype(int), y_proba))

    return metrics


# ------------------------- Combination parsing -------------------------

def parse_combination_name(name: str) -> Dict[str, Optional[str]]:
    info = {
        "date": None, "data_tag": None, "win_size": None, "win_func": None,
        "step": None, "k": None, "method": None, "method_params": None, "raw_folder": name,
    }
    parts = name.split("_")
    if parts and re.match(r"^\d{4}-\d{2}-\d{2}$", parts[0]):
        info["date"] = parts[0]
        parts = parts[1:]
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok == "data" and i + 1 < len(parts):
            info["data_tag"] = parts[i+1]; i += 2; continue
        if tok.isdigit() and info["win_size"] is None:
            info["win_size"] = tok; i += 1; continue
        if tok in ("hamming", "hann", "boxcar", "rect", "rectangular") and info["win_func"] is None:
            info["win_func"] = tok; i += 1; continue
        if tok.isdigit() and info["step"] is None:
            info["step"] = tok; i += 1; continue
        m = re.match(r"^k(?P<k>\d+)$", tok, re.IGNORECASE)
        if m and info["k"] is None:
            info["k"] = m.group("k"); i += 1; continue
        if tok in ("ae", "pca", "umap", "kmeans", "kmeans-l2"):
            info["method"] = tok
            params = []
            j = i + 1
            while j < len(parts) and re.match(r"^\d+$", parts[j]):
                params.append(parts[j]); j += 1
            info["method_params"] = "_".join(params) if params else None
            i = j; continue
        i += 1
    return info


# ------------------------- Fractional file discovery -------------------------

def find_fractional_files(comb_dir: str) -> List[str]:
    return [fp for fp in glob.glob(os.path.join(comb_dir, "fractional_occupancy*.npy"))
            if os.path.basename(fp).startswith("fractional_occupancy")]


# ------------------------- Main loop -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comb_dir", default=DEFAULTS["COMB_DIR"])
    ap.add_argument("--out_csv",  default=DEFAULTS["OUT_CSV"])
    ap.add_argument("--random_state", type=int, default=DEFAULTS["RANDOM_STATE"])
    ap.add_argument("--labels_xlsx", default=DEFAULTS["LABELS_XLSX"])
    ap.add_argument("--first_subject_id", type=int, default=DEFAULTS["FIRST_SUBJECT_ID"])
    ap.add_argument("--pos_class", default=DEFAULTS["POS_CLASS"])
    args = ap.parse_args()

    print("[Config]")
    print(f"  comb_dir   = {args.comb_dir}")
    print(f"  out_csv    = {args.out_csv}")
    print(f"  labels_xlsx= {args.labels_xlsx}")
    print(f"  first_id   = {args.first_subject_id}")
    print(f"  seed       = {args.random_state}")

    ids, y, id2idx = collect_ids_and_labels_from_xlsx(args.labels_xlsx)

    rows: List[Dict[str, Any]] = []
    comb_subdirs = sorted([d for d in glob.glob(os.path.join(args.comb_dir, "*")) if os.path.isdir(d)])

    for comb_path in comb_subdirs:
        comb_name = os.path.basename(comb_path.rstrip(os.sep))
        meta = parse_combination_name(comb_name)
        fractional_files = find_fractional_files(comb_path)
        if not fractional_files:
            continue
        for frac_fp in sorted(fractional_files):
            try:
                X, used_ids = build_X_by_ids_with_offset(frac_fp, ids, args.first_subject_id)
                y_used = np.array([id2idx[sid] for sid in used_ids])
                y_used = y[y_used]
                res = run_svm_cv(X, y_used, random_state=args.random_state, pos_label=args.pos_class)
                F = np.load(frac_fp)
                rows.append({
                    **meta,
                    "frac_file": os.path.basename(frac_fp),
                    "num_subjects": int(F.shape[0]),
                    "num_states": int(F.shape[1]),
                    "used_subjects": len(used_ids),
                    "cv_best_mean_acc": float(res["cv_best_mean_acc"]),
                    "cv_acc": float(res["cv_acc"]),
                    "cv_f1_macro": float(res["cv_f1_macro"]),
                    "cv_roc_auc": None if res["cv_roc_auc"] is None else float(res["cv_roc_auc"]),
                    "best_params": json.dumps(res["best_params"], ensure_ascii=False),
                    "error": None,
                })
            except Exception as e:
                rows.append({
                    **meta, "frac_file": os.path.basename(frac_fp),
                    "num_subjects": None, "num_states": None, "used_subjects": None,
                    "cv_best_mean_acc": None, "cv_acc": None, "cv_f1_macro": None, "cv_roc_auc": None,
                    "best_params": None, "error": str(e)})
                continue

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved results to: {args.out_csv}")


if __name__ == "__main__":
    main()
