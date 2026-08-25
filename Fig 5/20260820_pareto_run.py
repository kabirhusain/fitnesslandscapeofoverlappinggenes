"""
One Pareto example: one genetic code, one (pair, overlap), one process.

Companion to `20260820 Pareto Examples.ipynb`, which shells out to this script
once per code.  A separate process per code is not a convenience here, it is the
same correctness requirement the cluster scripts document: `set_genetic_code`
mutates a module-level array that numba bakes into the compiled kernel at FIRST
COMPILATION, so a single interpreter that runs four codes in turn silently runs
the first one four times.  The notebook therefore never imports numba itself.
The assertion below checks the installed code on all 64 codons rather than
trusting that.

Sampling is `20260821_all_codes_all_pairs.py`'s -- the cluster scan that produced
`20260821_all_rows.csv` -- unchanged and including the codon-degeneracy
correction:

    121 replicas, make_temperature_grid(T_min=0.3, T_max=1.0, M1=11, M2=11)
    N_swap 1000, N_equil 1e5, N_thin N_total//1000, discard_frac 0.2
    ln_n = codon_degeneracy_ln_n()          <- the correction, see below

so the z printed here is directly comparable with that csv's rows.  N_total and
n_workers are command-line arguments: N_total because the notebook may want a
cheaper preview than the cluster's 1e7 (the default), n_workers because it is a
local-machine choice with no bearing on the result.

The correction has the same stale-code hazard as the code itself: ln_n is read
from CODON_TABLE, and Type II and Type III both change the per-amino-acid codon
counts (Type I does not).  It is therefore computed AFTER set_genetic_code, and
passed to replica_exchange as an argument -- never read from a global inside a
kernel.  It enters the local MC acceptance only: the E1/E2 stored below, and so
the Pareto front and z, are the original DCA energies.

n_workers 5, not the cluster's 1: the cluster ran one task per core and had
nothing to gain, whereas here four codes share one 20-core machine.  Measured on
this pair, all four codes, N_total 2e5, wall for the whole set:

    4 concurrent x 1 worker       4 cores    23.0 s
    4 sequential x 20 workers    20 cores    23.9 s   <- Fig 5's shape
    4 concurrent x 5 workers     20 cores    12.1 s   <- this
    2 concurrent x 10 workers    20 cores    15.4 s

Fig 5's shape wins nothing: `executor.map` over 121 replicas once per swap round
is a barrier plus IPC every 1000 steps, so speedup saturates near 6.8x by 10
workers, and each sequential process pays the ~3 s numba compile alone.  Running
the four codes at once overlaps those compiles and gives each pool coarser work.

    python 20260820_pareto_run.py <code_type> <seed> <N_total> <out.npz> [n_workers]
    python 20260820_pareto_run.py standard -1 1000000 out/standard.npz 5

- Kabir Husain, with assistance from Claude Code (Anthropic)
"""

import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import overlappingGenes as og
from overlappingGenes import (
    codon_degeneracy_ln_n, extract_params, load_natural_energies,
    make_aa_permuted_genetic_code, make_shuffled_genetic_code,
    set_genetic_code, set_seed,
)
from replica_exchange import make_temperature_grid, replica_exchange, analyze_replicas

make_scrambled_genetic_code = importlib.import_module(
    "20260819_type3_code").make_scrambled_genetic_code


# --- Parameters ---
PF1, PF2 = "PF00595", "PF00013"
OVERLAP  = 166                     # frame 1, one below this pair's maximum of 168

bmDCA_dir = os.environ.get(
    "BMDCA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "0 bmDCA")) + os.sep

CODE_TYPE, CODE_SEED, N_TOTAL, OUTFILE = (sys.argv[1], int(sys.argv[2]),
                                          int(float(sys.argv[3])), sys.argv[4])
N_WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 5

# REMC settings, identical to 20260821_all_codes_all_pairs.py.  The one departure
# is the max(1, ...) guard on N_thin, which only ever binds below N_total 1000 --
# a preview budget the cluster never used.
T1_vals, T2_vals = make_temperature_grid(T_min=0.3, T_max=1.0, M1=11, M2=11)
RE_KWARGS = dict(
    N_swap=1000, N_total=N_TOTAL,
    N_equil=100_000, N_thin=max(1, N_TOTAL // 1000),
    discard_frac=0.2, quiet=True,
)

# --- The genetic code, installed before any numba kernel compiles ---
if CODE_TYPE == "shuffled":                                          # Type I
    set_genetic_code(*make_shuffled_genetic_code(seed=CODE_SEED))
elif CODE_TYPE == "permuted":                                        # Type II
    set_genetic_code(*make_aa_permuted_genetic_code(seed=CODE_SEED))
elif CODE_TYPE == "scrambled":                                       # Type III
    set_genetic_code(*make_scrambled_genetic_code(seed=CODE_SEED))
elif CODE_TYPE != "standard":
    raise ValueError(f"unknown code type {CODE_TYPE!r}")

_codons = np.array([[i, j, k] for i in range(4) for j in range(4)
                    for k in range(4)], dtype=np.uint8).ravel()
_got = np.empty(64, dtype=np.int32)
og.translate_numeric_out(_codons, _got)
assert np.array_equal(_got, og.CODON_TABLE_NUMERIC.ravel()), \
    "numba is using a stale genetic code"

# --- Codon-degeneracy correction, read from the code just installed ---
# Sampling only: the E1/E2 stored below, and so the Pareto front and z, are the
# original DCA energies.
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

L1, L2 = len(h1) // 21, len(h2) // 21

ne1 = load_natural_energies(f"{bmDCA_dir}{PF1}/{PF1}_naturalenergies.txt")
ne2 = load_natural_energies(f"{bmDCA_dir}{PF2}/{PF2}_naturalenergies.txt")
mu1, sig1 = np.mean(ne1), np.std(ne1)
mu2, sig2 = np.mean(ne2), np.std(ne2)

# Off both earlier runs' seed streams (they used INDEX+1 and INDEX+10001, with
# INDEX at most 731 and 449), and off the cluster scan's, which seeds from
# crc32 of the row filename.
seed = (1000003 * (20001 + abs(CODE_SEED) + 100 * ["standard", "shuffled",
        "permuted", "scrambled"].index(CODE_TYPE))) % (2 ** 31)
set_seed(seed)
np.random.seed(seed)

print(f"{CODE_TYPE}:{CODE_SEED}  {PF1}({L1}) x {PF2}({L2})  overlap {OVERLAP} nt "
      f"(frame {OVERLAP % 3})  N_total {N_TOTAL:.0e}  n_workers {N_WORKERS}  "
      f"degeneracy-corrected", flush=True)

# --- Run ---
t0 = time.time()
re_results = replica_exchange(DCA_1, DCA_2, L1, L2, OVERLAP,
                              T1_vals, T2_vals, n_workers=N_WORKERS, ln_n=ln_n,
                              **RE_KWARGS)
analysis = analyze_replicas(re_results, mu1, sig1, mu2, sig2)
within, zscore_dist = within_and_distance(analysis["pareto_front"],
                                          mu1, sig1, mu2, sig2)
elapsed = time.time() - t0

# --- Save: energies of every retained sample, plus the front ---
# The sequences are dropped; the Pareto panel only ever plots (E1, E2).
E1 = np.concatenate([re_results["samples"][(a, b)]["E1"]
                     for a in range(len(T1_vals)) for b in range(len(T2_vals))])
E2 = np.concatenate([re_results["samples"][(a, b)]["E2"]
                     for a in range(len(T1_vals)) for b in range(len(T2_vals))])

os.makedirs(os.path.dirname(OUTFILE) or ".", exist_ok=True)
np.savez_compressed(
    OUTFILE, E1=E1, E2=E2, pareto_front=analysis["pareto_front"],
    mu1=mu1, sig1=sig1, mu2=mu2, sig2=sig2,
    code_type=CODE_TYPE, code_seed=CODE_SEED, pf1=PF1, pf2=PF2,
    overlap_nuc=OVERLAP, frame=OVERLAP % 3, n_total=N_TOTAL,
    n_replicas=len(T1_vals) * len(T2_vals), n_thin=RE_KWARGS["N_thin"],
    rng_seed=seed, within_pareto=int(within), zscore_dist=zscore_dist,
    seconds=elapsed, n_workers=N_WORKERS,
    # Provenance: the correction actually used, so a cached npz says on its face
    # whether it predates this change.
    ln_n=ln_n, degeneracy_corrected=1)

print(f"  within={str(bool(within)):5s}  z={zscore_dist:+.3f}   "
      f"{len(E1)} samples   {elapsed / 60:.2f} min", flush=True)
