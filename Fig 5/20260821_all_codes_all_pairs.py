"""
Fig 5 redone: all three synthetic code types, all five pairs, in one array job,
under the codon-degeneracy correction.

This replaces the three near-identical scripts of
`../../20260817 Shuffled Codes/20260820 Type I II II/Cluster Code`
(20260817_shuffle_and_permute_frames.py, 20260818_shuffle_and_permute_pairs.py,
20260819_type3_pairs.py) with one task list.  The sampling, the temperature
grid, RE_KWARGS, N_total, the one-process-per-code rule and the row format are
theirs, unchanged, so rows produced here pool with each other directly.

    python 20260821_all_codes_all_pairs.py            # print the task count
    python 20260821_all_codes_all_pairs.py 0          # run task 0
    SLURM_ARRAY_TASK_ID=0 python 20260821_all_codes_all_pairs.py

Adding seeds later
------------------
SEED_FROM/SEED_TO pick the block of code seeds this submission covers, and row
files are named by what they contain rather than by array index, so a later
block cannot overwrite an earlier one and `cat Data/*.csv` still pools the lot:

    SEED_FROM=15 SEED_TO=30 sbatch --export=ALL --array=0-674 20260821_all_codes_all_pairs.sbatch

The standard-code baseline is emitted only when SEED_FROM == 0, so it is
measured once and never re-run.  A task whose row file already exists skips
itself, so a resubmission of a partly-finished array is free.

One process per code
--------------------
`set_genetic_code` mutates a module-level array that numba bakes into a compiled
kernel as a constant at FIRST COMPILATION.  Change the code afterwards and numba
silently keeps using the old one -- verified on numba 0.62.1: the Python-level
table changes, the kernel does not, and nothing warns.  So the code is installed
at the top, before anything can compile, the process handles exactly one code,
and the assertion below checks it on all 64 codons rather than trusting it.

The same hazard applies to the correction: ln_n is read from CODON_TABLE, and
Type II and Type III both change the per-amino-acid codon counts (Type I does
not).  It is therefore computed AFTER set_genetic_code, and passed to
replica_exchange as an argument -- never read from a global inside a kernel.

- Kabir Husain, with assistance from Claude Code (Anthropic)
"""

import csv
import importlib
import os
import sys
import time
import zlib

import numpy as np

import overlappingGenes as og
from overlappingGenes import (
    codon_degeneracy_ln_n, extract_params, load_natural_energies,
    make_aa_permuted_genetic_code, make_shuffled_genetic_code,
    set_genetic_code, set_seed,
)
from replica_exchange import make_temperature_grid, replica_exchange, analyze_replicas

# A module name cannot start with a digit, so this one import has to go through
# importlib.  Pure Python dict manipulation, no numba, so importing it cannot
# compile a kernel against the wrong code -- see the one-process-per-code note.
make_scrambled_genetic_code = importlib.import_module(
    "20260819_type3_code").make_scrambled_genetic_code


# --- Parameters ---
# In priority order.  The array is laid out pair-major, so trimming --array from
# the top drops whole pairs rather than leaving five half-finished ones.
PAIRS = [
    ("PF00072", "PF00009"),   # Fig 5's own pair, at its three longest overlaps
    ("PF00595", "PF00013"),   # replicates Fig 5's pair: frames 0,1 fail, 2 works
    ("PF00271", "PF00018"),   # only frame 1 fails -- is the Type I effect frame-specific?
    ("PF00076", "PF00017"),   # all three frames fail -- is there a ceiling?
    ("PF00153", "PF00011"),   # second replicate, and both genes almost fully overlapped
]

# The three largest overlaps each pair allows, covering all three reading frames,
# written as offsets below that pair's own maximum because the maximum differs
# per pair.  min(L1, L2)*3 - 6 is always a multiple of 3, so (max-2, max-1, max)
# is (frame 1, frame 2, frame 0) for every pair -- and for PF00072 x PF00009
# that is 316/317/318, exactly the overlaps the 20260817 frames run used.
OVERLAPS = [2, 1, 0]

# Code seeds covered by THIS submission.  See "Adding seeds later" above.
SEED_FROM = int(os.environ.get("SEED_FROM", 0))
SEED_TO = int(os.environ.get("SEED_TO", 15))

bmDCA_dir = os.environ.get(
    "BMDCA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "0 bmDCA")) + os.sep

# REMC settings.  N_swap, N_equil, discard_frac and the N_thin rule are Fig 5's;
# N_thin = N_total // 1000 keeps Fig 5's 800 retained samples per replica at any
# budget, which matters because z is an extremum over those samples -- a
# different sample count is a different statistic, not just a noisier one.
# The grid is 121 replicas over [0.3, 1.0]: Fig 5's 20x20 over [0.1, 1.0] puts
# only 10 of 20 points per axis at T >= 0.3, so ~100 of its 400 replicas do the
# work, and 1e7 is where z stops moving (measured: 0.098 residual against a
# run-to-run scatter of 0.128).
N_TOTAL = 10_000_000
T1_vals, T2_vals = make_temperature_grid(T_min=0.3, T_max=1.0, M1=11, M2=11)
RE_KWARGS = dict(
    N_swap=1000, N_total=N_TOTAL,
    N_equil=100_000, N_thin=N_TOTAL // 1000,
    discard_frac=0.2, quiet=True,
)

# --- The task list: one (pair, code, overlap) per array index ---
# "shuffled" = Type I, "permuted" = Type II, "scrambled" = Type III.  The
# standard code carries seed -1 and appears only in the first seed block.
CODE_MAKERS = {
    "standard": None,
    "shuffled": make_shuffled_genetic_code,
    "permuted": make_aa_permuted_genetic_code,
    "scrambled": make_scrambled_genetic_code,
}
CODES = ([("standard", -1)] if SEED_FROM == 0 else []) + [
    (kind, seed)
    for kind in ("shuffled", "permuted", "scrambled")
    for seed in range(SEED_FROM, SEED_TO)
]
TASKS = [(pf1, pf2, kind, seed, overlap)
         for pf1, pf2 in PAIRS
         for kind, seed in CODES
         for overlap in OVERLAPS]

if len(sys.argv) > 1:
    INDEX = int(sys.argv[1])
elif "SLURM_ARRAY_TASK_ID" in os.environ:
    INDEX = int(os.environ["SLURM_ARRAY_TASK_ID"])
else:
    per_pair = len(CODES) * len(OVERLAPS)
    print(f"{len(TASKS)} tasks  ->  sbatch --array=0-{len(TASKS) - 1}")
    print(f"  seeds {SEED_FROM}-{SEED_TO - 1}: {len(PAIRS)} pairs x {len(CODES)} codes "
          f"x {len(OVERLAPS)} overlaps")
    print(f"  codes: {'1 standard + ' if SEED_FROM == 0 else ''}"
          f"{SEED_TO - SEED_FROM} each of Type I, II, III")
    for i, (pf1, pf2) in enumerate(PAIRS):
        print(f"    {i * per_pair:5d}-{(i + 1) * per_pair - 1:5d}  {pf1} x {pf2}")
    sys.exit(0)

PF1, PF2, CODE_TYPE, CODE_SEED, BELOW_MAX = TASKS[INDEX]

# --- The genetic code, installed before any numba kernel compiles ---
if CODE_TYPE != "standard":
    set_genetic_code(*CODE_MAKERS[CODE_TYPE](seed=CODE_SEED))

_codons = np.array([[i, j, k] for i in range(4) for j in range(4)
                    for k in range(4)], dtype=np.uint8).ravel()
_got = np.empty(64, dtype=np.int32)
og.translate_numeric_out(_codons, _got)
assert np.array_equal(_got, og.CODON_TABLE_NUMERIC.ravel()), \
    "numba is using a stale genetic code"

# --- Codon-degeneracy correction, read from the code just installed ---
# Sampling only: the E1/E2 that replica_exchange reports, and so the Pareto front
# and z below, are the original DCA energies.
ln_n = codon_degeneracy_ln_n()


def within_and_distance(pareto_front, mu1, sig1, mu2, sig2):
    """Returns (within_pareto: bool, zscore_distance: float)."""
    within = False
    for pf_point in pareto_front:
        fz = np.array([(pf_point[0] - mu1) / sig1, (pf_point[1] - mu2) / sig2])
        if (fz[0] <= 0 and fz[1] <= 0 and (fz[0] < 0 or fz[1] < 0)):
            within = True
            break
    if not within:
        for pf_point in pareto_front:
            fz = np.array([(pf_point[0] - mu1) / sig1, (pf_point[1] - mu2) / sig2])
            if abs(fz[0]) < 1e-6 and abs(fz[1]) < 1e-6:
                within = True
                break
    pf_z = np.array([[(p[0] - mu1) / sig1, (p[1] - mu2) / sig2] for p in pareto_front])
    distances = np.sqrt(np.sum(pf_z**2, axis=1))
    zscore_dist = -float(np.min(distances)) if within else float(np.min(distances))
    return within, zscore_dist


# --- Load DCA params + natural energies ---
J1, h1 = extract_params(f"{bmDCA_dir}{PF1}/{PF1}_params.dat")
J2, h2 = extract_params(f"{bmDCA_dir}{PF2}/{PF2}_params.dat")
DCA_1, DCA_2 = [J1, h1], [J2, h2]

L1 = len(h1) // 21
L2 = len(h2) // 21

ne1 = load_natural_energies(f"{bmDCA_dir}{PF1}/{PF1}_naturalenergies.txt")
ne2 = load_natural_energies(f"{bmDCA_dir}{PF2}/{PF2}_naturalenergies.txt")
mu1, sig1 = np.mean(ne1), np.std(ne1)
mu2, sig2 = np.mean(ne2), np.std(ne2)

max_overlap = min(L1, L2) * 3 - 6
OVERLAP = max_overlap - BELOW_MAX
assert OVERLAP >= 12, \
    f"overlap {OVERLAP} shorter than the MIN_OVERLAP of every earlier scan"

# --- Save path: named by content, not by array index ---
# A later seed block gets different indices for the same tasks, so an
# index-named file would collide.  This name cannot, and it also lets a task
# skip itself when its row is already on disk.
os.makedirs("Data", exist_ok=True)
seed_tag = "std" if CODE_TYPE == "standard" else f"{CODE_SEED:03d}"
row_path = f"Data/{PF1}_{PF2}_{CODE_TYPE}_{seed_tag}_ov{OVERLAP}.csv"

print(f"[{INDEX}] {CODE_TYPE}:{CODE_SEED}  {PF1}({L1}) x {PF2}({L2})  "
      f"overlap {OVERLAP} of {max_overlap} nt (frame {OVERLAP % 3})  "
      f"{len(T1_vals) * len(T2_vals)} replicas  N_total {N_TOTAL:.0e}  "
      f"ATG->{og.CODON_TABLE['ATG']}", flush=True)
print(f"  {PF1}: <E> = {mu1:.1f} +/- {sig1:.1f}   "
      f"{PF2}: <E> = {mu2:.1f} +/- {sig2:.1f}", flush=True)
print(f"  -> {row_path}", flush=True)

if os.path.exists(row_path):
    print("  already done, skipping", flush=True)
    sys.exit(0)

# Seeded from the task itself, not the array index, so the same (pair, code,
# overlap) reproduces whichever seed block it was submitted in.  (Only up to
# overlappingGenes.initial_seq_no_stops, which draws from an unseeded
# default_rng -- inherited from Fig 5, not introduced here.)
seed = zlib.crc32(row_path.encode()) % (2 ** 31)
set_seed(seed)
np.random.seed(seed)

# --- Run ---
t0 = time.time()
re_results = replica_exchange(DCA_1, DCA_2, L1, L2, OVERLAP,
                              T1_vals, T2_vals, n_workers=1, ln_n=ln_n,
                              **RE_KWARGS)
analysis = analyze_replicas(re_results, mu1, sig1, mu2, sig2)
within, zscore_dist = within_and_distance(analysis["pareto_front"],
                                          mu1, sig1, mu2, sig2)
elapsed = time.time() - t0

# --- Save: one row per task, one file per task ---
# One file per task, not one shared csv: hundreds of array tasks appending to
# the same file over a shared filesystem is a corruption risk.  Headerless, so
# the whole run is just `cat Data/*.csv` -- see the README.
with open(row_path, "w", newline="") as f:
    csv.writer(f).writerow(
        [CODE_TYPE, CODE_SEED, PF1, PF2, OVERLAP, OVERLAP % 3,
         len(T1_vals) * len(T2_vals), N_TOTAL, RE_KWARGS["N_thin"], seed,
         int(within), zscore_dist, elapsed])

print(f"  within={str(bool(within)):5s}  z={zscore_dist:+.3f}   "
      f"{elapsed / 60:.1f} min", flush=True)
