import os, itertools, datetime, time, json, subprocess, numpy as np, math

# === CONFIGURATION ===
USE_PARAMETER_RANGES = False
dim_red_methods = ["pca","umap"]
k_list = [2,3,4]
time_window_lengths = [30,40,60]
steps = [1,15,30,45,60]
n_nodes_options = [246]
TR = 1.0
TIMEOUT_SEC = 420          # 7 minutes per combo
MIN_TW_RATIO = 1/10        # skip if tw_len < n_nodes/10
data_set = "bibi"          # or "neutral"

if data_set == "bibi":
    duration_sec = 235
elif data_set == "neutral":
    duration_sec = 151
else:
    raise ValueError("DATASET must be either 'bibi' or 'neutral'")

# === DIRECTORIES ===
base_dir = "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/code"

if data_set == "bibi":
    input_path = "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/data/bibi_246_flat"
    output_dir = os.path.join(base_dir, "results_246_flattened_bibi")
elif data_set == "neutral":
    input_path = "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/data/neutral_246_flat"
    output_dir = os.path.join(base_dir, "results_246_flattened_neutral")

progress_file     = os.path.join(base_dir, "progress.json")
combinations_file = os.path.join(base_dir, "combinations.json")
logfile           = os.path.join(base_dir, "run_log.txt")

os.makedirs(output_dir, exist_ok=True)

# --- reset tracking files each run ---
for fp in (progress_file, combinations_file, logfile):
    try:
        os.remove(fp)
    except FileNotFoundError:
        pass

# === METHOD PARAMETERS ===
method_params_fixed = {"ae":"512,256,32","pca":"1,1,32","umap":"1000,64,30","kmeans":"1,1,1"}
method_param_ranges = {
    "ae":{"layer1":[256,512,1024],"layer2":[128,256,512],"layer3":[16,32,64]},
    "pca":{"n_components":[16,32,64,128]},
    "umap":{"min_dist_x1000":[100,500,1000],"n_components":[32,64,128],"n_neighbors":[15,30,50]},
    "kmeans":{"placeholder":[1]}
}

# === UTILITY FUNCTIONS ===
def compute_n_time_windows(duration_sec, TR, window_size, step):
    n_timepoints = duration_sec / TR
    return max(math.floor((n_timepoints - window_size) / step) + 1, 0)

def log(message):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] {message}"
    print(msg)
    with open(logfile, 'a') as f:
        f.write(msg + "\n")

def combo_id(method, method_params, k, tw_len, step, n_nodes):
    return f"{method}|{method_params}|k{k}|tw{tw_len}|s{step}|n{n_nodes}"

def run_main_script(dim_red, k, input_path, window_length, step_size, params, log_path):
    script_path = os.path.join(base_dir, "main_tamar.py")
    cmd = ["python3", "-u", script_path, dim_red, params, input_path,
           "-w","hamming","-l",str(window_length),"-s",str(step_size),"-k",str(k)]
    log("Running command: " + " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        proc.wait(timeout=TIMEOUT_SEC)
        return proc.returncode
    except subprocess.TimeoutExpired:
        f.write(f"\n[TIMEOUT after {TIMEOUT_SEC}s]\n"); f.flush()
        proc.kill()
        raise
    finally:
        f.flush(); f.close()

# === PARAMETER COMBINATIONS ===
def generate_fixed_method_combinations():
    return [(m, method_params_fixed.get(m, "1,1,1")) for m in dim_red_methods]

def generate_range_method_combinations():
    all_combos = []
    for method in dim_red_methods:
        if method not in method_param_ranges:
            all_combos.append((method, "1,1,1")); continue
        pd = method_param_ranges[method]
        if method == "ae":
            for L1,L2,L3 in itertools.product(pd["layer1"], pd["layer2"], pd["layer3"]):
                all_combos.append((method, f"{L1},{L2},{L3}"))
        elif method == "pca":
            for nc in pd["n_components"]:
                all_combos.append((method, f"1,1,{nc}"))
        elif method == "umap":
            for md,nc,nn in itertools.product(pd["min_dist_x1000"], pd["n_components"], pd["n_neighbors"]):
                all_combos.append((method, f"{md},{nc},{nn}"))
        elif method in ("kmeans","kmeans-l2"):
            all_combos.append((method, "1,1,1"))
    return all_combos

method_param_combinations = generate_range_method_combinations() if USE_PARAMETER_RANGES else generate_fixed_method_combinations()
log(f"Using {'RANGES' if USE_PARAMETER_RANGES else 'FIXED'} mode - {len(method_param_combinations)} method-parameter combinations")
for method, params in method_param_combinations:
    log(f"  {method}: {params}")

# === FULL GRID ===
full_combinations = [(n_nodes, method, method_params, k, tw_len, step)
                     for n_nodes in n_nodes_options
                     for method, method_params in method_param_combinations
                     for k in k_list
                     for tw_len in time_window_lengths
                     for step in steps]
total = len(full_combinations)
log(f"Total combinations to run: {total}")

# Save combinations snapshot
comb_data = []
for (n_nodes, method, method_params, k, tw_len, step) in full_combinations:
    n_tw = compute_n_time_windows(duration_sec, TR, tw_len, step)
    comb_data.append({"n_nodes":n_nodes,"dim_red":method,"method_params":method_params,
                      "k":k,"tw_len":tw_len,"step":step,"n_tw":n_tw})
with open(combinations_file, 'w') as f:
    json.dump(comb_data, f, indent=2)

# === MAIN LOOP ===
progress = {}
start_time = time.time()
today_str = datetime.datetime.today().strftime("%Y-%m-%d")

for idx, (n_nodes, method, method_params, k, tw_len, step) in enumerate(full_combinations, start=1):
    n_tw = compute_n_time_windows(duration_sec, TR, tw_len, step)
    subdir = f"{method}_params{method_params.replace(',', '-')}_k{k}_tw{tw_len}_step{step}_nTW{n_tw}_{today_str}"
    method_output_dir = os.path.join(output_dir, subdir)
    os.makedirs(method_output_dir, exist_ok=True)

    cid = combo_id(method, method_params, k, tw_len, step, n_nodes)
    params_dict = {"n_nodes":n_nodes,"dim_red":method,"method_params":method_params,
                   "k":k,"tw_len":tw_len,"step":step,"n_tw":n_tw}

    # skip if tw_len too small vs nodes
    threshold = n_nodes * MIN_TW_RATIO
    if tw_len < threshold:
        log(f"Skip {cid} (tw_len {tw_len} < n_nodes/10 {threshold:.1f})")
        progress[cid] = {"status":"skipped",
                         "reason":"tw_len < n_nodes/10",
                         "params":params_dict,
                         "skipped_at":datetime.datetime.now().isoformat(),
                         "threshold":threshold}
        with open(progress_file, "w") as f: json.dump(progress, f, indent=2)
        continue

    # per-combo live log path
    log_fname = (f"res_n{params_dict['n_nodes']}_dim{params_dict['dim_red']}"
                 f"_k{params_dict['k']}_tw{params_dict['tw_len']}_step{params_dict['step']}"
                 f"_nTW{params_dict['n_tw']}_params{params_dict['method_params'].replace(',', '-')}.txt")
    combo_log_path = os.path.join(method_output_dir, log_fname)

    iter_start = time.time()
    progress[cid] = {"status":"started","started_at":datetime.datetime.now().isoformat(),
                     "params":params_dict,"output_dir":method_output_dir}
    with open(progress_file, "w") as f: json.dump(progress, f, indent=2)

    log(f"Start iteration {idx}/{total} | {cid}")
    try:
        rc = run_main_script(method, k, input_path, tw_len, step, method_params, combo_log_path)
        rt = round(time.time() - iter_start, 3)

        if rc == 0:
            progress[cid].update({"status":"finished","finished_at":datetime.datetime.now().isoformat(),
                                  "returncode":0,"stdout_stderr_file":combo_log_path,"runtime_sec":rt})
        else:
            log(f"ERROR non-zero exit ({rc}) for {cid}")
            progress[cid].update({"status":"error","finished_at":datetime.datetime.now().isoformat(),
                                  "returncode":rc,"stdout_stderr_file":combo_log_path,
                                  "runtime_sec":rt,"error":"non-zero return code"})
    except subprocess.TimeoutExpired:
        progress[cid].update({"status":"error","error":f"timeout {TIMEOUT_SEC}s",
                              "finished_at":datetime.datetime.now().isoformat(),
                              "runtime_sec":round(time.time() - iter_start, 3),
                              "stdout_stderr_file":combo_log_path})
        log(f"TIMEOUT at {cid} after {TIMEOUT_SEC}s")
    except Exception as e:
        progress[cid].update({"status":"error","error":str(e),
                              "finished_at":datetime.datetime.now().isoformat(),
                              "runtime_sec":round(time.time() - iter_start, 3),
                              "stdout_stderr_file":combo_log_path})
        log(f"ERROR at {cid}: {e}")
    finally:
        with open(progress_file, "w") as f: json.dump(progress, f, indent=2)

    elapsed = time.time() - start_time
    done = sum(1 for v in progress.values() if v.get("status") == "finished")
    avg_time = elapsed / max(done, 1)
    remaining = avg_time * (total - done)
    log(f"Finished {cid} | Iteration time: {progress[cid].get('runtime_sec', 0):.2f}s | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")

log("All combinations looped.")

# === SUMMARY ===
total_time = time.time() - start_time
done = sum(1 for v in progress.values() if v.get("status") == "finished")
log("SUMMARY:")
log(f"  Mode: {'RANGES' if USE_PARAMETER_RANGES else 'FIXED'}")
log(f"  Finished: {done}/{total}")
log(f"  Total runtime: {total_time/3600:.2f} hours")
log(f"  Avg time per finished combo: {total_time/max(done,1):.1f} seconds")
