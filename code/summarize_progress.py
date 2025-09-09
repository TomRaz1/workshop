import json
import os
from datetime import datetime

progress_file = "progress.json"

def load_progress():
    if not os.path.exists(progress_file):
        print("No progress.json file found.")
        return {}
    with open(progress_file, "r") as f:
        return json.load(f)

def summarize():
    progress = load_progress()
    if not progress:
        return

    finished = [cid for cid, info in progress.items() if info.get("status") == "finished"]
    errors   = [cid for cid, info in progress.items() if info.get("status") == "error"]
    started  = [cid for cid, info in progress.items() if info.get("status") == "started"]

    if errors:
        print("---- Errors ----")
        for cid in errors:
            info = progress[cid]
            err = info.get("error", "Unknown error")
            out = info.get("stdout_stderr_file", "no file")
            t   = info.get("finished_at", "unknown time")
            print(f"* {cid}")
            print(f"    Error: {err}")
            print(f"    Log file: {out}")
            print(f"    Time: {t}")
        print()

    if finished:
        print("---- Finished ----")
        for cid in finished:
            info = progress[cid]
            rc   = info.get("returncode", 0)
            out  = info.get("stdout_stderr_file", "no file")
            rt   = info.get("runtime_sec", "n/a")
            t    = info.get("finished_at", "unknown time")
            print(f"* {cid}")
            print(f"    Return code: {rc}")
            print(f"    Runtime: {rt} sec")
            print(f"    Log file: {out}")
            print(f"    Time: {t}")
        print()
        print("===== PROGRESS SUMMARY =====")
    print(f"Total combinations tracked: {len(progress)}")
    print(f"  Finished: {len(finished)}")
    print(f"  Errors:   {len(errors)}")
    print(f"  Started (but not finished): {len(started)}")
    print()
if __name__ == "__main__":
    summarize()
