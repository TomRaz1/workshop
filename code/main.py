# MIT License
#
# Copyright (c) 2021 Arthur Spencer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import fcCluster
import fcPreproc
import fcUtils as utils
from sklearn import metrics
from scipy import stats
import argparse
from argparse import RawTextHelpFormatter

# parallel preproc
run_parallel_preproc = True
preproc_only = False
if run_parallel_preproc:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import multiprocessing as proc
    n_jobs = proc.cpu_count()
    n_jobs = min(n_jobs, int(os.environ.get("SLURM_CPUS_PER_TASK","4")))
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["OPENBLAS_NUM_THREADS"]="1"


def log(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def run_preproc(path_, datadir, window_shape, window_size, step, hcp=False):
    """
    Run sliding window correlations using fcPreproc.get_dfc
    """
    log("[STAGE] Preprocessing: load raw time series")
    subjs = fcPreproc.load_data(path_, hcp=hcp)

    nsubjs = len(subjs)
    n_nodes = len(subjs[0])
    batch_size_preproc = 256
    batch_dfc_preproc = False

    log(f"[INFO] Subjects: {nsubjs}")
    log(f"[INFO] Nodes: {n_nodes}")
    log(f"[INFO] Time points per node: {len(subjs[0,0])}")

    if nsubjs > batch_size_preproc:
        log(f"[INFO] Running DFC in batches of {batch_size_preproc}")
        batch_dfc_preproc = True

    t0 = time.time()
    if run_parallel_preproc:
        p = proc.Pool(processes=n_jobs)
        if batch_dfc_preproc:
            batch_inds = np.arange(0, nsubjs, batch_size_preproc)
            for i, ind in enumerate(batch_inds):
                batch_subjs = subjs[ind:min([ind+batch_size_preproc, nsubjs])]
                log(f"[INFO] Batch {i}: {ind} to {min([ind+batch_size_preproc, nsubjs])-1}")
                subjs_arr = [(subj, window_size, window_shape, n_nodes, step, '-') for subj in batch_subjs]
                dfc_ = p.map(fcPreproc.get_dfc, subjs_arr)
                np.save(os.path.join(datadir, f'dfc_{i}.npy'), np.array(dfc_))
            dfc_ = np.concatenate([np.load(os.path.join(datadir, f'dfc_{i}.npy')) for i in range(len(batch_inds))], axis=0)
        else:
            log("[INFO] Parallel DFC (one process per subject)")
            subjs_arr = [(subjs[i], window_size, window_shape, n_nodes, step, '-') for i in range(nsubjs)]
            dfc_ = p.map(fcPreproc.get_dfc, subjs_arr)
    else:
        dfc_ = [None]*nsubjs
        for s in range(nsubjs):
            dfc_[s] = fcPreproc.get_dfc((subjs[s], window_size, window_shape, n_nodes, step))

    log(f"[DONE] Preprocessing took {datetime.timedelta(seconds=time.time()-t0)}")
    return np.array(dfc_)

def run_clustering(dfc, D, method, n_clusters=5, n_epochs=None, batch_size=None, embedded_path=None, run_elbow=False, kmax=12, show_plots=False):
    """
    Optional dimensionality reduction followed by clustering
    """
    log("[STAGE] Clustering / Dimensionality reduction")
    exemplar_inds = utils.get_exemplars(dfc)

    if method == 'ae':
        import autoEncFC as fcae
        if embedded_path is None:
            log("[INFO] Training AutoEncoder")
            autoencoder = fcae.AutoEncoder(D)
            dfc_embedded = autoencoder.encode(dfc, maximum_epoch=n_epochs, batch_size=batch_size, outdir=outdir, show_plots=show_plots)
            np.save(os.path.join(outdir, 'dfc_embedded.npy'), dfc_embedded)
        else:
            log(f"[INFO] Loading embedded data from {embedded_path}")
            dfc_embedded = np.load(os.path.join(embedded_path, 'dfc_embedded.npy'))
        dist = 'euclidean'
    else:
        dfc_embedded = utils.concatenate_windows(dfc)
        if method == 'pca':
            from sklearn.decomposition import PCA
            log(f"[INFO] PCA to {D[-1]} components")
            dfc_embedded = PCA(n_components=D[-1]).fit_transform(dfc_embedded)
            dist = 'euclidean'
        elif method == 'umap':
            import umap
            min_dist = D[0] * 1e-3
            n_comp = D[1]
            n_neighbors = D[2]
            dfc_embedded = umap.UMAP(min_dist=min_dist, n_components=n_comp,
                             n_neighbors=n_neighbors, verbose=True).fit_transform(dfc_embedded)
            dist = 'euclidean'
        elif method == 'kmeans':
            dist = 'l1'
        elif method == 'kmeans-l2':
            dist = 'euclidean'

    if run_elbow:
        log("[INFO] Running elbow criterion")
        cluster_krange(dfc_embedded[exemplar_inds], kmax, dist=dist)
        clusters = None
    else:
        log(f"[INFO] KMeans with k={n_clusters}, dist={dist}")
        clusters, centroids, score = fcCluster.kmeans(dfc_embedded, exemplar_inds, n_clusters=n_clusters, dist=dist)

    log("[DONE] Clustering stage finished")
    return clusters, dfc_embedded

def cluster_krange(dfc, kmax, dist='euclidean'):
    """
    Runs k-means with a range of k on the embedded data to allow plot for elbow criterion
    """
    cvi_krange = np.zeros(kmax-1)
    for k in range(2, kmax+1):
        clusters, centroids, score = fcCluster.kmeans(dfc, np.arange(len(dfc)), n_clusters=k, dist=dist)

        labels = np.zeros(len(dfc))
        for i, clus in enumerate(clusters):
            for c in clus:
                labels[c] = i

        states = np.array([np.mean(dfc[labels==i], axis=0) for i in range(k)])
        wcd = np.zeros(k)
        bcd = 0.
        for i in range(k):
            wcd[i] = np.sum(metrics.pairwise.euclidean_distances(dfc[labels==i], [states[i]]))
            if i < k-1:
                bcd += np.sum(metrics.pairwise.euclidean_distances(states[np.arange(i+1, k)], [states[i]]))
        bcd /= float(k*(k-1))/2
        cvi = np.sum(wcd) / (bcd*len(dfc))

        log(f"[ELBOW] k={k} score={cvi:.6f}")
        cvi_krange[k-2] = cvi

    out_csv = '/home/yandex/0368352201_BrainWS2025b/tomraz/results_tamar/cvi.csv'
    with open(out_csv, 'a') as resfile:
        resfile.write(','.join(str(x) for x in cvi_krange))
        resfile.write('\n')
    log(f"[SAVED] Elbow CVI to {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dFC analysis with SWC, dimensionality reduction and clustering.', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-k','--nclusters', help='Number of clusters. Default: 5', default=5, type=int)
    parser.add_argument('-c','--elbow', help='Run elbow criterion', action='store_true')
    parser.add_argument('-K','--kmax', help='Max k if running elbow criterion. Default: 12', default=12, type=int)
    parser.add_argument('-b','--batch_size', help='Batch size for autoencoder. Default: 50', default=50, type=int)
    parser.add_argument('-e','--n_epochs', help='Epochs for autoencoder training. Default: 100', default=100, type=int)
    parser.add_argument('-p','--plots', help='Show plots', action='store_true')
    parser.add_argument('-m','--model_data', help='Data generated by the SimTB model. If true, checks clustering performance against ground truth', action='store_true')
    parser.add_argument('-hcp','--hcp_data', help='HCP data (has a different file structure to SimTB data)', action='store_true')
    parser.add_argument('-t','--TR', help='Repetition time (TR) of the data, in seconds. Default: 1', default=1., type=float)
    parser.add_argument('-w','--window_shape', help='Window shape for sliding-window correlations. Default: rectangle', default='rectangle', type=str)
    parser.add_argument('-l','--window_size', help='Window size for sliding-window correlations, in TR. Default: 20', default=20, type=int)
    parser.add_argument('-s','--step', help='Step size for sliding-window correlations. Default: 1', default=1, type=int)
    parser.add_argument('-E','--embedded_path', help='Path to embedded dfc data', default=None, type=str)
    parser.add_argument('method', help='Clustering/dimensionality reduction method.\n\t Options: kmeans, kmeans-l2, ae, pca, umap', type=str)
    parser.add_argument('params', help=str('Parameters. \n\t Method : parameters \n\t ae : d1,d2,d3 (e.g. 512,256,32)\n\t umap : v*10^3,u,m (e.g. 1000,64,30)\n\t pca : 1,1,p (e.g. 1,1,32)\n\t kmeans or kmeans-l2 : 1,1,1 (just a place-holder).'), type=str)
    parser.add_argument('func_path', help='Path to raw data directory.', type=str)

    args = parser.parse_args()
    D = [int(d) for d in args.params.split(',')]

    if args.method not in ['kmeans', 'kmeans-l2', 'pca', 'umap', 'ae']:
        raise Exception('Method not recognised - choose from kmeans, kmeans-l2, pca, umap, ae')

    base_folder = os.path.basename(os.path.dirname(args.func_path.rstrip('/')))

    # === paths (changed) ===
    datadir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Demo",
    f"{base_folder}_{args.window_size}_{args.window_shape}_{args.step}"
)
    datadir = os.path.abspath(datadir) + os.sep
    today_str = datetime.datetime.today().strftime("%Y-%m-%d")
    outdir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "combination_results",
    f"{today_str}_{base_folder}b_{args.window_size}_{args.window_shape}_{args.step}_k{args.nclusters}_{args.method}"
)
    outdir = os.path.abspath(outdir)

    if args.method in ['ae','umap']:
        for d in D:
            outdir += f'_{d}'
    elif args.method == 'pca':
        outdir += f'_{D[2]}'
    outdir += '/'

    log("=== START MAIN ===")
    log(f"[CONFIG] Method={args.method} Params={args.params} K={args.nclusters} TR={args.TR}")
    log(f"[CONFIG] Window shape={args.window_shape} size={args.window_size} step={args.step}")
    log(f"[PATH] Data folder: {datadir}")
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    log(f"[PATH] Output folder: {outdir}")
    os.makedirs(outdir, exist_ok=True)

    # preprocessing
    if os.path.isfile(os.path.join(datadir, 'dfc.npy')):
        log('[INFO] Found existing DFC windows. Loading...')
        dfc = np.array(np.load(os.path.join(datadir, 'dfc.npy')))
    else:
        log('[INFO] No saved DFC found. Running sliding-window correlations...')
        t_start = time.time()
        try:
            dfc = run_preproc(args.func_path, datadir, args.window_shape, args.window_size, args.step, hcp=args.hcp_data)
        except Exception as e:
            log(f"[ERROR] Preprocessing failed: {e}")
            raise
        np.save(os.path.join(datadir, 'dfc.npy'), dfc)
        log('[SAVED] dfc.npy')
        runtime = time.time() - t_start
        log(f"[TIMER] SWC runtime: {datetime.timedelta(seconds=runtime)}")
        if preproc_only:
            log("[INFO] Preproc-only mode. Exiting.")
            raise Exception('Preproc only')

    nsubjs = len(dfc)
    nwindows = len(dfc[0,:,0,0])
    n_nodes = len(dfc[0,0,:,0])
    log(f"[SUMMARY] Subjects={nsubjs} | Windows/subject={nwindows} | Nodes={n_nodes} | Features={int(n_nodes*(n_nodes-1)/2)}")

    # clustering / dim-red
    log("[INFO] Starting clustering step")
    try:
        clusters, dfc_embedded = run_clustering(
            dfc, D, args.method, n_clusters=args.nclusters,
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            embedded_path=args.embedded_path, run_elbow=args.elbow,
            kmax=args.kmax, show_plots=args.plots
        )
    except Exception as e:
        log(f"[ERROR] Clustering step failed: {e}")
        raise

    if args.elbow:
        log('[DONE] Elbow completed. Choose k and rerun without --elbow.')
    else:
        log("[STAGE] Computing temporal properties (fractional occupancy, dwell time, states)")
        frac_occ, dwell_time, states = utils.state_properties(clusters, dfc, args.TR, args.step)
        np.save(os.path.join(outdir, f'clusters_{args.method}.npy'), np.array(clusters, dtype=object))
        np.save(os.path.join(outdir, f'fractional_occupancy_{args.method}.npy'), frac_occ)
        np.save(os.path.join(outdir, f'dwell_time_{args.method}.npy'), dwell_time)
        np.save(os.path.join(outdir, f'states_{args.method}.npy'), states)
        log("[SAVED] clusters / fractional_occupancy / dwell_time / states")

    log("=== END MAIN SUCCESS ===") 
