"""
ASSUMPTIONS ABOUT FILE STRUCTURE AND DATA
-----------------------------------------
1) Per-subject label files live in one folder and are named:
      l_b_<ID>.npy  or  r_b_<ID>.npy or l_n_<ID>.npy  or  r_n_<ID>.npy )
   where:
      - the first letter is the class label: 'l' or 'r'
      - <ID> is a 1-based integer subject ID (e.g., 1, 2, 3, ...)

2) A single dwell-time matrix exists at dwell_time.npy (or a path you provide),
   with shape [num_subjects X num_states].
   Row (ID-1) corresponds to subject ID = <ID>. 
   Example: subject ID 1 is at row index 0.

3) Dwell-time values are ALREADY proportions per subject (i.e., rows sum to ~1).
   Therefore, no additional conversion to proportions is needed.

4) We keep labels as strings 'l' and 'r' (not converting to 0/1).
   For metrics that require a positive class, we treat 'r' as positive.

Edit the paths below before running.
"""

import os
import glob
import re
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


# ======= INPUT PATHS (EDIT THESE) =======
LABELS_FOLDER = "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/data/bibi_246_flat" #or neutral_246_flat
#r"path\to\per_subject_label_files"   # folder containing l_b_<ID>.npy / r_b_<ID>.npy
DWELL_PATH = DWELL_PATH = "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/dwells_for_svm/dwell_time_pca.npy"

#r"path\to\dwell_time.npy"            # matrix [num_subjects × num_states]


def collect_ids_and_labels(labels_folder: str):
    """
    Scan 'labels_folder' for files like:
      l_b_sub-2_bibiindictment_yeo17... .npy
      r_n_sub-15_something.npy
    Extract:
      - label: first char 'l' or 'r'
      - id:    digits after 'sub-'
    Returns:
      ids: sorted list of IDs (ints)
      y:   numpy array of labels aligned to ids
    """
    pattern = re.compile(r'^(?P<label>[lr])_(?P<grp>[bn])_sub-(?P<id>\d+).*\.npy$', re.IGNORECASE)
    rows = []  # (id_int, label_char)

    for name in os.listdir(labels_folder):
        if not name.lower().endswith(".npy"):
            continue
        m = pattern.match(name)
        if not m:
            continue
        sid = int(m.group("id"))
        label = m.group("label").lower()  # 'l' or 'r'
        rows.append((sid, label))

    if not rows:
        raise RuntimeError(
            f"No matching .npy files found in '{labels_folder}'. "
            "Expected names like l_b_sub-12_*.npy or r_n_sub-7_*.npy."
        )

    rows.sort(key=lambda x: x[0])
    ids = [sid for (sid, _) in rows]
    y   = np.array([label for (_, label) in rows], dtype=object)
    return ids, y

def build_X_by_ids(dwell_path: str, ids: list[int]):
    """
    Load dwell_time (2D) and align rows to given 1-based IDs.

    Modes:
    - Direct (ID-1): row for subject ID == ID-1 (original assumption).
    - Compact order: if max(ID) > num_rows but len(ids) == num_rows,
      assume rows are in the order of sorted(ids); i.e., row index is the rank
      of the subject ID within the sorted ID list.

    If neither fits, we drop IDs that don't map and warn.
    """
    D = np.load(dwell_path)
    if D.ndim != 2:
        raise ValueError(f"'dwell_time.npy' must be 2D, got shape={D.shape}.")

    num_subjects, num_states = D.shape
    ids_sorted = sorted(ids)
    max_id = max(ids)

    # Case A: original assumption holds
    if max_id <= num_subjects:
        X = np.vstack([D[sid - 1] for sid in ids])
        return X

    # Case B: compact order likely (same count)
    if len(ids) == num_subjects:
        id_to_row = {sid: i for i, sid in enumerate(ids_sorted)}
        X = np.vstack([D[id_to_row[sid]] for sid in ids])
        return X

    # Case C: mixed — drop IDs that cannot map, with a clear warning
    valid_ids_direct = [sid for sid in ids if 1 <= sid <= num_subjects]
    dropped = sorted(set(ids) - set(valid_ids_direct))
    print(f"[WARN] Dropping IDs without rows in dwell file: {dropped} "
          f"(dwell rows={num_subjects})")
    if not valid_ids_direct:
        raise IndexError("No IDs can be mapped to dwell rows. Check alignment or rebuild dwell file.")
    X = np.vstack([D[sid - 1] for sid in valid_ids_direct])
    return X


def run_svm_cv(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Fit an SVM using a Pipeline(StandardScaler -> SVC) with GridSearchCV.
    Keep labels as 'l'/'r'. We treat 'r' as the positive class for metrics.
    Returns the best estimator and prints CV performance.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, class_weight="balanced"))
    ])

    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10, 100]},
        {"svc__kernel": ["rbf"],    "svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]},
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # GridSearch optimizing ROC-AUC ('r' treated as positive)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    grid.fit(X, y)
    print("\n=== GridSearchCV Results ===")
    print("Best params:", grid.best_params_)
    print("Best mean ROC-AUC (CV):", grid.best_score_)

    best_pipe = grid.best_estimator_

    # Cross-validated predictions with the best params (no train leakage)
    y_pred  = cross_val_predict(best_pipe, X, y, cv=cv, n_jobs=-1, method="predict")
    y_proba = cross_val_predict(best_pipe, X, y, cv=cv, n_jobs=-1, method="predict_proba")[:, 1]

    print("\n=== Cross-validated Performance (using best params) ===")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("F1 (positive='r'):", f1_score(y, y_pred, pos_label='r'))
    print("ROC-AUC (positive='r'):", roc_auc_score((y == 'r').astype(int), y_proba))
    print("\nClassification report (pos='r'):\n",
          classification_report(y, y_pred, target_names=['l', 'r']))

    return best_pipe


def main():
    # 1) Collect subject IDs and string labels
    ids, y = collect_ids_and_labels(LABELS_FOLDER)

    # 2) Build X (already proportions) by aligning rows to the gathered IDs
    X = build_X_by_ids(DWELL_PATH, ids)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First 5 (ID -> label):", list(zip(ids[:5], y[:5])))

    # 3) Run SVM with CV & report metrics
    _ = run_svm_cv(X, y, random_state=42)


if __name__ == "__main__":
    main()
