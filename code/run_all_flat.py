import os
import itertools
import datetime
import time
import subprocess
import numpy as np
import math

# === Parameters and Paths ===
input_dir = "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/data/flattened_output"
output_dir = "./results_17"
os.makedirs(output_dir, exist_ok=True)
logfile = "run_log.txt"

# === Analysis Parameters ===
dim_red_methods = ["umap", "ae", "pca"]
k_list = [2, 3, 4]
time_window_lengths = [20]
steps = [1]
n_nodes = 246
duration_sec = 150
TR = 1.0

# === Utility Functions ===
def compute_n_time_windows(duration_sec, TR, window_size, step):
    n_timepoints = duration_sec / TR
    return math.floor((n_timepoints - window_size) / step) + 1

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(logfile, 'a') as f:
        f.write(msg + "\n")

def run_main_script(dim_red, k, input_path, win_length, step_size):
    if dim_red == "ae":
        params = "512,256,32"
    elif dim_red == "pca":
        params = "1,1,32"
    elif dim_red == "umap":
        params = "1000,64,30"
    else:
        params = "1,1,1"

    cmd = [
        "python3", "/home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/code/main_tamar.py",
        dim_red,
        params,
        input_path,
        "-w", "hamming",
        "-l", str(win_length),
        "-s", str(step_size),
        "-k", str(k)
    ]

    print("[DEBUG] Running command:", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)

def save_result(result_text, params, output_dir):
    filename = (
        f"res_dim{params['dim_red']}_k{params['k']}_tw{params['tw_len']}"
        f"_step{params['step']}_nTW{params['n_tw']}_subj_{params['subject']}.txt"
    )
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        f.write(result_text)

# === Main Execution ===
files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
param_combinations = list(itertools.product(dim_red_methods, k_list, time_window_lengths, steps))
total = len(files) * len(param_combinations)

idx = 1
start_time = time.time()

for fname in files:
    input_path = os.path.join(input_dir, fname)
    subject_id = fname.replace(".npy", "")

    for dim_red, k, tw_len, step in param_combinations:
        n_tw = compute_n_time_windows(duration_sec, TR, tw_len, step)
        params = {
            "n_nodes": n_nodes,
            "dim_red": dim_red,
            "k": k,
            "tw_len": tw_len,
            "step": step,
            "n_tw": n_tw,
            "subject": subject_id
        }

        log(f"▶ Running {idx}/{total} | Subject: {subject_id} | Method: {dim_red} | k={k}")

        try:
            result = run_main_script(dim_red, k, input_path, tw_len, step)
            save_result(result.stdout + result.stderr, params, output_dir)
        except Exception as e:
            log(f"ERROR at {idx}: {e}")
            continue

        idx += 1

log("✔ All combinations finished.")
