# dFC DimReduction Pipeline (TAU — workshop/code)

An adaptation of **\[apcspencer/dFC\_DimReduction]** with convenience scripts for running dynamic functional connectivity (dFC) on fMRI time‑series, optional dimensionality reduction (UMAP/PCA/AE), state clustering (k‑means), feature extraction, and downstream **group comparison** (e.g., Relogious vs Secular). The code produces state/time‑in‑state features and evaluation reports.

---

## What this repo does (high level)

1. **Compute dynamic FC** from sliding‑window correlations on node‑averaged time‑series (or load it from cache if it already exists).  
2. **Reduce dimensionality** of windowed FC (optional): UMAP / PCA / autoencoder.
3. **Cluster** reduced (or full) representations into brain "states" using k‑means.
4. **Extract temporal features** per subject: dwell time, fraction time, etc.
5. **Compare groups** (e.g., *Religious* vs *Secular*, or any two label sets) using the extracted features and SVM classifier.

> Designed for **between‑group comparisons** under naturalistic viewing paradigms but is task‑agnostic.

> **Note: Please be aware that running the code may take up to 1.5 hours, depending on the dimensions of the data (i.e., the number of nodes) and the length of the videos.**

---

## Repo layout (key pieces)


* `main.py` — single‑run pipeline over one dataset/config.
* `try_combs.py` — run a grid of combinations (UMAP params, k, window sizes, etc.). especially for exploratory analyses

---

## Requirements

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
h5py           
nibabel         
pyyaml           
tqdm
# optional depending on your settings
tensorflow       
pyclustering     
skggm  
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
**Explenation**

Dimensionality reduction: umap

UMAP parameters: 1000,64,30 (interpreted in the code as number of neighbors = 1000, output dimensions = 64, and minimum distance = 30)

Input data path: /path/to/data (folder with node-averaged fMRI time series)

Window shape: hamming (applied for the sliding-window correlation)

Window length: 30 TRs (each dFC window covers 30 time points)

Step size: 1 TR (windows slide forward one TR at a time, resulting in maximal overlap)

Number of clusters: 3 (k-means will partition the dFC windows into three brain states)


You can also choose different parameters depending on your needs — for example, replacing umap with pca or ae, changing the window length or step size, or selecting a different number of clusters.

```
Demo labels CSV for SVM:

labels_50subjects_demo.csv
```

Evaluate:

Using run_svm.py. 

---

## Exploratory Use:
We reccomend running Exploratory run using try_combs.py and run_svm_batch.py.

It will:

* pick the input path according to `data_set`.
* build dated subfolders.
* call `main.py` per combo with the correct signature.

### 3) Recommended workflow for **group comparison**

1. **Produce features** with either a single run or `try_combs.py` (see above). The run saves `fractional_occupancy_*.npy`, `dwell_time_*.npy`, etc., inside each combo folder.
2. **Evaluate with SVM** (after features exist):

A single run is executed in the same way as with the demo dataset.
When using try_combs, the run is triggered by a simple command, but the specific parameter combinations are defined and adjusted directly in the code.
```

### Output files

At the end of a run, the pipeline automatically creates the output and saves four key files:

- **`clusters_umap.npy`** – assignment of each window to a brain state (list of indices per state).  
- **`dwell_time_umap.npy`** – average dwell time (in windows) each subject spent in each state before switching.  
- **`fractional_occupancy_umap.npy`** – fraction of total time each subject spent in each state.  
- **`states_umap.npy`** – centroid representation of each state in the reduced feature space (e.g., UMAP space).  

---

## Runtime environment

These experiments were executed on **Tel‑Aviv University’s HPC cluster** (Linux) with access to **NVIDIA GPUs**. While the code runs on CPU, using GPU acceleration (for UMAP or TensorFlow autoencoder variants) can significantly reduce runtime for large datasets.

---

## Citation

If you use this code in a publication, please cite:

* Spencer, A.P.C., & Goodfellow, M. (2022). *Using Deep Clustering to Improve fMRI Dynamic Functional Connectivity Analysis*. NeuroImage, 119288. [https://doi.org/10.1016/j.neuroimage.2022.119288](https://doi.org/10.1016/j.neuroimage.2022.119288)

---

**Good luck! :)**
