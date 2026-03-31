"""
Collect all 66 per-pair pickle files from "20260303 Data/" into a single
pandas DataFrame saved as "20260303_all_results.pkl".

Columns: pf1, pf2, overlap_nuc, within_pareto, zscore_dist
"""

import os
import pickle
import re

import pandas as pd

DATA_DIR = "20260327 Corrected Data"
OUT_FILE = "Pickles/20260327_all_results.pkl"

rows = []
pattern = re.compile(r"20260303_overlap_scan_results_(PF\d+)_(PF\d+)\.pkl")

files = sorted(os.listdir(DATA_DIR))
for fname in files:
    m = pattern.match(fname)
    if not m:
        continue
    pf1, pf2 = m.group(1), m.group(2)
    with open(os.path.join(DATA_DIR, fname), "rb") as f:
        data = pickle.load(f)
    for overlap_nuc, within_pareto, zscore_dist in data:
        rows.append({
            "pf1": pf1,
            "pf2": pf2,
            "overlap_nuc": overlap_nuc,
            "within_pareto": bool(within_pareto),
            "zscore_dist": zscore_dist,
        })

df = pd.DataFrame(rows)
df["pair"] = df["pf1"] + "_" + df["pf2"]
df["frame"] = df["overlap_nuc"] % 3

df.to_pickle(OUT_FILE)
print(f"Saved {len(df):,} rows ({df['pair'].nunique()} pairs) -> {OUT_FILE}")
print(df.head())
