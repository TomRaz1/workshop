"""
This script generates a mapping table linking each dynamic functional connectivity (dFC) window
to its corresponding subject ID and time range (in time points and seconds), based on the sliding
window parameters used in the dFC pipeline.
"""

import pandas as pd

def create_window_mapping(n_subjects, n_windows_per_subject, window_size, step, TR=2.0):
    rows = []
    global_index = 0
    for subj in range(n_subjects):
        for i in range(n_windows_per_subject):
            start_time = i * step
            end_time = start_time + window_size
            row = {
                "window_index": global_index,
                "subject_id": subj,
                "window_start_time": start_time,
                "window_end_time": end_time,
                "window_start_sec": start_time * TR,
                "window_end_sec": end_time * TR
            }
            rows.append(row)
            global_index += 1
    return pd.DataFrame(rows)

# main
if __name__ == "__main__":
    #parameters from the run
    n_subjects = 50
    n_windows_per_subject = 10
    window_size = 60
    step = 30
    TR = 2.0

    df = create_window_mapping(n_subjects, n_windows_per_subject, window_size, step, TR)
    df.to_csv("window_mapping.csv", index=False)
    print("Saved mapping to window_mapping.csv")
    # run with python map_windows.py
