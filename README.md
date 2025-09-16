# dFC DimReduction Pipeline (TAU — workshop/code)

An adaptation of **\[apcspencer/dFC\_DimReduction]** with convenience scripts for running dynamic functional connectivity (dFC) on fMRI time‑series, optional dimensionality reduction (UMAP/PCA/AE), state clustering (k‑means), feature extraction, and downstream **group comparison** (e.g., Relogious vs Secular). The code produces state/time‑in‑state features and evaluation reports.

---

## What this repo does (high level)

1. **Compute dynamic FC** from sliding‑window correlations on node‑averaged time‑series.
2. **Reduce dimensionality** of windowed FC (optional): UMAP / PCA / autoencoder.
3. **Cluster** reduced (or full) representations into brain "states" using k‑means.
4. **Extract temporal features** per subject: dwell time, fraction time, etc.
5. **Compare groups** (e.g., *Religious* vs *Secular*, or any two label sets) using the extracted features and SVM classifier.

> Designed for **between‑group comparisons** under naturalistic viewing paradigms but is task‑agnostic.

---

## Repo layout (key pieces)


* `main.py` — single‑run pipeline over one dataset/config.
* `try_combs.py` — run a grid of combinations (UMAP params, k, window sizes, etc.). especially for exploratory analyses

---

## Requirements

### Python

* **Python 3.10+** 

### Python packages

Install via `pip` (recommended in a fresh virtualenv/conda env):

```
numpy
scipy
pandas
scikit-learn
umap-learn
matplotlib
joblib
h5py            # if reading .mat (v7.3) / HDF5
nibabel          # if reading NIfTI at any step
pyyaml           # if using YAML configs
tqdm
# optional depending on your settings
scikit-image     # if you use any image utils
tensorflow       # only if you enable autoencoder path
pyclustering     # used upstream; optional here unless you call it
skggm            # only if you explicitly switch to SKGGM Graphical Lasso
```

Optional GPU. (Recommended). 

---

## Data expectations

* **Input:** node‑averaged time‑series per subject (CSV/NPY/MAT), shaped roughly `[time × nodes]`. We used 246/17 nodes.
* **Directory structure:**  single folder for **both groups** (e.g., `Religious/`, `Secular/`, `ExReligious/`).

### Example dataset for users to try

We included a  synthetic dataset you can run end‑to‑end:

* Folder name: `synthetic_tseries_to_try/`
* Files: `1.csv, 2.csv, ..., 50.csv` (50 subjects; any consistent `[T×nodes]` format your loader supports)
* **Ad‑hoc groups for demo**: subjects 1–25 → *GroupA*; 26–50 → *GroupB*.

Run on the synthetic data:

```
python3 /path/to/code/main.py umap 1000,64,30 /path/to/data \
    -w hamming -l 30 -s 1 -k 3 

```

Demo labels CSV for SVM:

labels_50subjects_demo.csv
```

Evaluate:

Using run_svm.py code. 

---

## Exploratory Use:
We reccomend running Exploratory run using try_combs.py and run_svm_batch.py.

It will:

* pick the input path according to `data_set`.
* build dated subfolders.
* call `main_tamar.py` per combo with the correct signature.

### 3) Recommended workflow for **group comparison**

1. **Produce features** with either a single run or `try_combs.py` (see above). The run saves `fractional_occupancy_*.npy`, `dwell_time_*.npy`, etc., inside each combo folder. fileciteturn0file0
2. **Evaluate with SVM** (after features exist):

   ```bash
   # Example; adapt script/flags to your repo
   python3 code/run_svm.py \
     --results_root /home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/code/results_246_flattened_neutral \
     --labels_csv    /path/to/labels.csv \
     --metric roc_auc
   ```

   *The idea*: point `run_svm` at the combo subfolders and supply a labels CSV (Subject → Group). (may vary by script)

* `--data_dir` / `--data_root` — input folder for a single task or a root containing multiple tasks.
* `--out_dir` / `--out_root` — where results are written. Scripts make dated subfolders.
* `--window_size`, `--window_step`, `--window_shape` — sliding window params (e.g., `hamming`).
* `--method` — `none | pca | umap | ae`.
* `--umap_n_neighbors`, `--umap_min_dist`, `--umap_n_components` — UMAP hyper‑params.
* `--k` — number of k‑means states.
* `--tr` — sampling interval in seconds.
* `--alpha` — Graphical Lasso strength (if enabled).
* `--seed` — RNG seed for reproducibility.

---

## Outputs

Each run creates a folder like:

```
code/combination_results/2025-09-15_<task>_<win>_k<k>_<method>_<params>/
  ├─ logs.txt                 # CONFIG, PATH, SUMMARY lines
  ├─ state_centroids.npy      # k × features
  ├─ assignments.csv          # per-window state labels per subject
  ├─ features.csv             # dwell time, frac time, transitions, etc.
  ├─ plots/                   # optional figures
  └─ model.joblib             # saved k‑means / reducer, if enabled
```

---

## Reproducibility & versions

* Prefer pinned versions via `requirements.txt` (see below template).
* Set `--seed` everywhere you can (UMAP, k‑means, splitters).
* Log shapes and parameter summaries at the start of each run.

---

## Example `requirements.txt`

> Use this as a starting point and tweak to your environment.

```
numpy>=1.22
scipy>=1.8
pandas>=1.4
scikit-learn>=1.1
umap-learn>=0.5
matplotlib>=3.5
joblib>=1.1
h5py>=3.7
nibabel>=5.0
pyyaml>=6.0
tqdm>=4.64
# Optional
pyclustering>=0.10.1
tensorflow>=2.9   # only if using AE
skggm==0.2.8      # only if you switch to SKGGM GLasso
```

To generate a frozen spec from your working env:

```bash
pip freeze > requirements-lock.txt
```

---

## Notes vs. upstream

* We follow the pipeline described in **apcspencer/dFC\_DimReduction** and keep interface‑compatible choices where possible (UMAP/PCA/AE; k‑means; sliding window DFC). If you can run their `main.py`, you can run ours with similar arguments.
* Some helpers prefer **`sklearn.covariance.graphical_lasso`** over SKGGM by default; install `skggm` only if you switch.
* UMAP defaults may differ; pass explicit params for consistency.

---

## Runtime environment

These experiments were executed on **Tel‑Aviv University’s HPC cluster** (Linux) with access to **NVIDIA GPUs**. While the code runs on CPU, using GPU acceleration (for UMAP or TensorFlow autoencoder variants) can significantly reduce runtime for large datasets.

---

## Citation

If you use this code in a publication, please cite:

* Spencer, A.P.C., & Goodfellow, M. (2022). *Using Deep Clustering to Improve fMRI Dynamic Functional Connectivity Analysis*. NeuroImage, 119288. [https://doi.org/10.1016/j.neuroimage.2022.119288](https://doi.org/10.1016/j.neuroimage.2022.119288)

---

## Troubleshooting

* **Long runtimes**: reduce `n_neighbors`/`n_components` (UMAP), use PCA, lower `k`, or skip Graphical Lasso.
* **ConvergenceWarnings** in Graphical Lasso: increase `alpha`, lower max iterations, or use sample correlations.
* **Memory**: ensure DFC caching is enabled and reuse `dfc.npy` across runs.

---

## License

This repo inherits the upstream academic spirit. Check/add a LICENSE file appropriate for your use.
