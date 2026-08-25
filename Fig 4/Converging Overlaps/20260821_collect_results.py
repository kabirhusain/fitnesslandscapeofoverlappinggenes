"""
Collect all per-index csv files from "Data/" into a single pandas DataFrame
saved as "20260821_all_results.csv".

Columns: pf1, pf2, overlap_nuc, within_pareto, zscore_dist

Also reports which array indices are missing, and prints the sbatch line that
reruns just those.
"""

import csv
import importlib
import os

import pandas as pd

DATA_DIR = "Data"
OUT_FILE = "20260821_all_results.csv"

scan = importlib.import_module("20260821_scan_overlaps_array")

rows = []
seen = set()

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.startswith("row_") or not fname.endswith(".csv"):
        continue
    with open(os.path.join(DATA_DIR, fname), newline="") as f:
        for r in csv.reader(f):
            if not r:
                continue
            index, pf1, pf2, frame, overlap_nuc, within_pareto, zscore_dist = r[:7]
            seen.add(int(index))
            rows.append({
                "pf1": pf1,
                "pf2": pf2,
                "overlap_nuc": int(overlap_nuc),
                "within_pareto": bool(int(within_pareto)),
                "zscore_dist": float(zscore_dist),
            })

df = pd.DataFrame(rows)
df["pair"] = df["pf1"] + "_" + df["pf2"]
df["frame"] = df["overlap_nuc"] % 3

df.to_csv(OUT_FILE, index=False)
print(f"Saved {len(df):,} rows ({df['pair'].nunique()} pairs) -> {OUT_FILE}")
print(df.head())


def ranges(idx):
    """Compress a sorted index list into slurm's comma/dash --array syntax."""
    out, i = [], 0
    idx = sorted(idx)
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(str(idx[i]) if i == j else f"{idx[i]}-{idx[j]}")
        i = j + 1
    return ",".join(out)


# Indices are complete when their .done marker is present -- see the scan script.
done = {int(f[4:-5]) for f in os.listdir(DATA_DIR) if f.endswith(".done")}
missing = [i for i in range(len(scan.TASKS)) if i not in done]

if missing:
    print(f"\n{len(missing)} of {len(scan.TASKS)} indices incomplete")
    print(f"  sbatch --array={ranges(missing)} 20260821_scan_overlaps_array.sbatch")
else:
    print(f"\nAll {len(scan.TASKS)} indices complete.")
