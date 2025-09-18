#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train SVM on Religious vs Secular and report ONLY ExRe agreement accuracy
vs a pseudo-label chosen at the TOP of this file.
"""

# =============== CHOOSE HOW TO TREAT ExRe (Ex-Religious) =====================
# Set to either "Religious" or "Secular"
AGREEMENT_TARGET = "Religious"
# =============================================================================

import os, re, glob, json, argparse
from typing import Dict, Any, List, Optional
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_combination_name(name: str) -> Dict[str, Optional[str]]:
    info = {
        "date": None, "data_tag": None, "win_size": None, "win_func": None,
        "step": None, "k": None, "method": None, "method_params": None,
        "raw_folder": name,
    }
    parts = name.split("_")
    if parts and re.match(r"^\d{4}-\d{2}-\d{2}$", parts[0]):
        info["date"] = parts[0]; parts = parts[1:]
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok == "data" and i+1 < len(parts): info["data_tag"]=parts[i+1]; i+=2; continue
        if tok.isdigit() and info["win_size"] is None: info["win_size"]=tok; i+=1; continue
        if tok in ("hamming","hann","boxcar","rect","rectangular") and info["win_func"] is None:
            info["win_func"]=tok; i+=1; continue
        if tok.isdigit() and info["step"] is None: info["step"]=tok; i+=1; continue
        m=re.match(r"^k(?P<k>\d+)$", tok, re.IGNORECASE)
        if m and info["k"] is None: info["k"]=m.group("k"); i+=1; continue
        if tok in ("ae","pca","umap","kmeans","kmeans-l2"):
            info["method"]=tok; params=[]; j=i+1
            while j < len(parts) and re.match(r"^\d+$", parts[j]): params.append(parts[j]); j+=1
            info["method_params"] = "_".join(params) if params else None; i=j; continue
        i+=1
    return info


def find_fractional_files(comb_dir: str) -> List[str]:
    cands = glob.glob(os.path.join(comb_dir, "fractional_occupancy*.npy"))
    return sorted([fp for fp in cands if os.path.basename(fp).startswith("fractional_occupancy")])


def build_row_order_mapping(n_rows: int) -> Dict[int,int]:
    # Assumes row i corresponds to subject_id (i+1). Adjust here if needed.
    return {sid: sid-1 for sid in range(1, n_rows+1)}


def subset_by_ids(F: np.ndarray, ids: List[int], id2row: Dict[int,int]) -> np.ndarray:
    return F[[id2row[i] for i in ids], :]


def pick_model(X: np.ndarray, y: np.ndarray, random_state: int=42):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, class_weight="balanced"))
    ])
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.01,0.1,1,10,100]},
        {"svc__kernel": ["rbf"],    "svc__C": [0.1,1,10,100], "svc__gamma": ["scale",0.01,0.1,1]},
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc",
                        cv=cv, n_jobs=-1, refit=True)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def main():
    ap = argparse.ArgumentParser(description="Report ONLY ExRe agreement accuracy vs pseudo-label.")
    ap.add_argument("--comb_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--labels_csv", required=True,
                    help="CSV with columns: subject_id,group in {Religious,Secular,ExRe}")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    if AGREEMENT_TARGET not in ("Religious", "Secular"):
        raise ValueError("AGREEMENT_TARGET must be 'Religious' or 'Secular'")

    df_lab = pd.read_csv(args.labels_csv)
    if not {"subject_id","group"}.issubset(df_lab.columns):
        raise ValueError("labels_csv must have columns: subject_id, group")

    norm = {"religious":"Religious","secular":"Secular","exre":"ExRe"}
    df_lab["group"] = df_lab["group"].astype(str).map(lambda s: norm.get(s.strip(), s.strip()))

    ids_rel = df_lab.loc[df_lab["group"]=="Religious","subject_id"].astype(int).tolist()
    ids_sec = df_lab.loc[df_lab["group"]=="Secular","subject_id"].astype(int).tolist()
    ids_exr = df_lab.loc[df_lab["group"]=="ExRe","subject_id"].astype(int).tolist()

    rows = []
    for comb in sorted([d for d in glob.glob(os.path.join(args.comb_dir,"*")) if os.path.isdir(d)]):
        meta = parse_combination_name(os.path.basename(comb))
        for frac_fp in find_fractional_files(comb):
            try:
                F = np.load(frac_fp)
                if F.ndim != 2: raise ValueError(f"{os.path.basename(frac_fp)} must be 2D, got {F.shape}")
                id2row = build_row_order_mapping(F.shape[0])
                X_rel = subset_by_ids(F, ids_rel, id2row)
                X_sec = subset_by_ids(F, ids_sec, id2row)
                X_exr = subset_by_ids(F, ids_exr, id2row)

                Xtr = np.vstack([X_rel, X_sec])
                ytr = np.array(["Religious"]*len(ids_rel) + ["Secular"]*len(ids_sec), dtype=object)

                model, best_params = pick_model(Xtr, ytr, random_state=args.random_state)
                proba = model.predict_proba(X_exr)
                classes = list(model.classes_)
                idx_rel = classes.index("Religious")
                idx_sec = classes.index("Secular")
                p_rel = proba[:, idx_rel]
                p_sec = proba[:, idx_sec]
                pred = model.predict(X_exr)

                pseudo = np.array([AGREEMENT_TARGET]*len(ids_exr), dtype=object)
                agree = (pred == pseudo)
                agree_acc = float(np.mean(agree))

                # write per-subject predictions
                per_path = os.path.join(
                    comb,
                    f"exre_agreementONLY_{AGREEMENT_TARGET}_{os.path.splitext(os.path.basename(frac_fp))[0]}.csv"
                )
                pd.DataFrame({
                    "subject_id": ids_exr,
                    "predicted_class": pred,
                    "p_religious": p_rel,
                    "p_secular": p_sec,
                    "pseudo_label": pseudo,
                    "agree": agree.astype(int),
                }).to_csv(per_path, index=False)

                rows.append({
                    **meta,
                    "frac_file": os.path.basename(frac_fp),
                    "num_states": int(F.shape[1]),
                    "agreement_target": AGREEMENT_TARGET,
                    "agreement_accuracy": agree_acc,
                    "exre_pred_relig_frac": float(np.mean(pred=="Religious")),
                    "exre_pred_secul_frac": float(np.mean(pred=="Secular")),
                    "exre_mean_p_relig": float(np.mean(p_rel)),
                    "exre_mean_p_secul": float(np.mean(p_sec)),
                    "per_subject_csv": per_path,
                    "best_params": json.dumps(best_params, ensure_ascii=False),
                    "error": None,
                })

            except Exception as e:
                rows.append({
                    **meta,
                    "frac_file": os.path.basename(frac_fp) if frac_fp else None,
                    "num_states": None,
                    "agreement_target": AGREEMENT_TARGET,
                    "agreement_accuracy": None,
                    "exre_pred_relig_frac": None,
                    "exre_pred_secul_frac": None,
                    "exre_mean_p_relig": None,
                    "exre_mean_p_secul": None,
                    "per_subject_csv": None,
                    "best_params": None,
                    "error": str(e),
                })

    out_df = pd.DataFrame(rows, columns=[
        "date","data_tag","win_size","win_func","step","k","method","method_params","raw_folder",
        "frac_file","num_states","agreement_target","agreement_accuracy",
        "exre_pred_relig_frac","exre_pred_secul_frac","exre_mean_p_relig","exre_mean_p_secul",
        "per_subject_csv","best_params","error"
    ])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, float_format="%.6f")
    print(f"[OK] Saved scores to: {args.out_csv}")


if __name__ == "__main__":
    main()
