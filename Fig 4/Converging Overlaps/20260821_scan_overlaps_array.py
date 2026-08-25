"""
Scan all overlaps with replica-exchange Monte Carlo for a given Pfam pair.

Array index is (pair, frame): 136 pairs x 3 frames.  One index walks the
overlaps of its own frame -- 12 to min(L1,L2)*3-6, step 3 -- and writes one row
per overlap.  All 136 pairs are the C(17,2) combinations of the 17 Pfam families.

For each overlap length:
  - Runs replica exchange (matching notebook cells 3-4 parameters)
  - Saves a Pareto front plot to {date}_Pareto_front_plots_{pf1}_{pf2}/
  - Records:
      (a) whether the natural-energy mean lies within the Pareto front
      (b) z-score distance to the Pareto front (0 if within)

Results saved to Data/row_{index:05d}.csv, one row per overlap.

One CPU per array index, no worker pool: the previous 36-cpu run of this scan
(../../20260729 Replica Exchange Codon Corrected/From Cluster) measured ~6%
efficiency -- 847 CPU-h billed for ~49 CPU-h of work -- because every swap round
is a barrier over 121 replicas.  With n_workers=1 there is no pool, no barrier,
no IPC, and the numba compile is paid once per index instead of once per overlap.

    python 20260821_scan_overlaps_array.py            # print the task count
    python 20260821_scan_overlaps_array.py 0          # run index 0
    SLURM_ARRAY_TASK_ID=0 python 20260821_scan_overlaps_array.py
"""

import argparse
import csv
import itertools
import os
import time

import numpy as np

from overlappingGenes import (
    extract_params, load_natural_energies, codon_degeneracy_ln_n,
)
from replica_exchange import make_temperature_grid, replica_exchange, analyze_replicas

outdir = "Data"

bmDCA_dir = os.environ.get(
    "BMDCA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "0 bmDCA")) + os.sep

# ---------------------------------------------------------------------------
# All Pfam families and pairs
# ---------------------------------------------------------------------------

PFAM_FAMILIES = [
    "PF00004", "PF00041", "PF00072", "PF00076", "PF00096", "PF00153",
    "PF00271", "PF00397", "PF00512", "PF00595", "PF02518", "PF07679",
    "PF00009", "PF00011", "PF00013", "PF00017", "PF00018", 
]
ALL_PAIRS = list(itertools.combinations(PFAM_FAMILIES, 2))  # 66 pairs


# ---------------------------------------------------------------------------
# Helper functions (explicit mu/sig args, no module globals)
# ---------------------------------------------------------------------------

def _is_dominated(point, pareto_front, mu1, sig1, mu2, sig2):
    """True if any Pareto-front point dominates `point` in z-score space."""
    pz = np.array([(point[0] - mu1) / sig1, (point[1] - mu2) / sig2])
    for pf_point in pareto_front:
        fz = np.array([(pf_point[0] - mu1) / sig1, (pf_point[1] - mu2) / sig2])
        if (fz[0] <= pz[0] and fz[1] <= pz[1] and
                (fz[0] < pz[0] or fz[1] < pz[1])):
            return True
    return False


def _is_on_front(point, pareto_front, mu1, sig1, mu2, sig2, tol=1e-6):
    """True if `point` coincides with a Pareto-front point in z-score space."""
    pz = np.array([(point[0] - mu1) / sig1, (point[1] - mu2) / sig2])
    for pf_point in pareto_front:
        fz = np.array([(pf_point[0] - mu1) / sig1, (pf_point[1] - mu2) / sig2])
        if abs(fz[0] - pz[0]) < tol and abs(fz[1] - pz[1]) < tol:
            return True
    return False


def within_and_distance(pareto_front, mu1, sig1, mu2, sig2):
    """
    Returns (within_pareto: bool, zscore_distance: float).

    within_pareto is True if the natural mean (mu1, mu2) lies on or is
    dominated by the Pareto front (i.e. the overlap can match or beat
    natural-sequence energies in both dimensions simultaneously).

    zscore_distance is the minimum Euclidean distance from the natural mean
    to the Pareto front in z-score space. Negative if within_pareto is True,
    positive otherwise.
    """
    natural_mean = np.array([mu1, mu2])
    within = (_is_on_front(natural_mean, pareto_front, mu1, sig1, mu2, sig2) or
              _is_dominated(natural_mean, pareto_front, mu1, sig1, mu2, sig2))

    pf_z = np.array([[(p[0] - mu1) / sig1, (p[1] - mu2) / sig2] for p in pareto_front])
    distances = np.sqrt(np.sum(pf_z**2, axis=1))   # distance from (0,0) in z-space
    zscore_dist = -float(np.min(distances)) if within else float(np.min(distances))

    return within, zscore_dist


# ---------------------------------------------------------------------------
# The task list: one (pair, frame) per array index
# ---------------------------------------------------------------------------

FRAMES = [0, 1, 2]
TASKS = [(pf1, pf2, frame) for pf1, pf2 in ALL_PAIRS for frame in FRAMES]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replica-exchange overlap scan for one (Pfam pair, frame).")
    parser.add_argument("index", type=int, nargs="?", default=None,
                        help="Array index. Defaults to SLURM_ARRAY_TASK_ID; "
                             "with neither, prints the task count and exits.")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: run subset of overlaps, print timing extrapolation.")
    parser.add_argument("--n-test-overlaps", type=int, default=5,
                        help="Number of evenly-spaced overlaps in test mode (default 5).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Determine array index ---
    if args.index is not None:
        index = args.index
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        print(f"{len(TASKS)} tasks  ->  sbatch --array=0-{len(TASKS) - 1}")
        print(f"  {len(ALL_PAIRS)} pairs x {len(FRAMES)} frames")
        return

    pf1, pf2, frame = TASKS[index]
    print(f"Index {index}: {pf1} x {pf2}, frame {frame}")

    # --- Codon-degeneracy correction ---
    # Sampling only.  The E1/E2 that replica_exchange reports are the original
    # DCA energies, so the z-scores and Pareto fronts below use the original hvec.
    ln_n = codon_degeneracy_ln_n()

    # --- Load DCA parameters and natural-energy statistics (notebook cell 2) ---
    J1, h1 = extract_params(f"{bmDCA_dir}{pf1}/{pf1}_params.dat")
    J2, h2 = extract_params(f"{bmDCA_dir}{pf2}/{pf2}_params.dat")

    DCA_1 = [J1, h1]
    DCA_2 = [J2, h2]

    L1 = len(h1) // 21   # protein 1 length in AA (no stop)
    L2 = len(h2) // 21   # protein 2 length in AA (no stop)

    ne1 = load_natural_energies(f"{bmDCA_dir}{pf1}/{pf1}_naturalenergies.txt")
    ne2 = load_natural_energies(f"{bmDCA_dir}{pf2}/{pf2}_naturalenergies.txt")
    mu1, sig1 = np.mean(ne1), np.std(ne1)
    mu2, sig2 = np.mean(ne2), np.std(ne2)

    max_overlap = min(L1, L2) * 3 - 6

    print(f"{pf1}: L = {L1} AA,  <E> = {mu1:.1f} +/- {sig1:.1f}")
    print(f"{pf2}: L = {L2} AA,  <E> = {mu2:.1f} +/- {sig2:.1f}")

    # --- Temperature grid (notebook cell 3) ---
    T1_vals, T2_vals = make_temperature_grid(T_min=0.3, T_max=1.0, M1=11, M2=11)
    M1_grid, M2_grid = len(T1_vals), len(T2_vals)

    # --- Determine overlaps to scan ---
    # Step 1 over the whole range, split three ways by frame, so the union of
    # the three indices of a pair is the original step-1 scan.
    min_overlap = 12
    all_overlaps = [ov for ov in range(min_overlap, max_overlap + 1)
                    if ov % 3 == frame]

    if args.test:
        # Evenly-spaced subset
        n_test = min(args.n_test_overlaps, len(all_overlaps))
        indices = np.linspace(0, len(all_overlaps) - 1, n_test, dtype=int)
        overlaps = [all_overlaps[i] for i in indices]
        print(f"TEST MODE: {n_test} overlaps out of {len(all_overlaps)}: {overlaps}")
    else:
        overlaps = all_overlaps
        print(f"Overlap range: {min_overlap} to {max_overlap} nt "
              f"(frame {frame}, {len(all_overlaps)} values)")

    # --- Output directory for plots ---
    # plot_dir = f"{date_str}_Pareto_front_plots_{run_tag}"
    # os.makedirs(plot_dir, exist_ok=True)

    # --- Resume: skip overlaps already in this index's csv ---
    # One file per array index, not one shared csv: hundreds of array tasks
    # appending to the same file over a shared filesystem is a corruption risk.
    # Headerless, so the whole run is just `cat Data/row_*.csv`.
    os.makedirs(outdir, exist_ok=True)
    csv_path = f"{outdir}/row_{index:05d}.csv"
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            done = {int(r[4]) for r in csv.reader(f) if r}
        print(f"Resuming: {len(done)} of {len(overlaps)} overlaps already done")

    # --- Main scan loop ---
    timings = []
    fh = open(csv_path, "a", newline="")
    writer = csv.writer(fh)

    for overlap_nuc in overlaps:
        if overlap_nuc in done:
            continue
        t0 = time.time()
        print(f"\n=== overlap = {overlap_nuc} nt ===")

        # Seeded from the index and overlap so a rerun of the same task
        # reproduces it.  (Only up to overlappingGenes.initial_seq_no_stops,
        # which draws from an unseeded default_rng.)
        np.random.seed((1000003 * (index + 1) + overlap_nuc) % (2 ** 31))

        # --- Replica exchange (notebook cell 3) ---
        re_results = replica_exchange(
            DCA_1, DCA_2, L1, L2, overlap_nuc,
            T1_vals, T2_vals,
            N_swap=500,
            N_total=100_000,
            N_equil=10_000,
            N_thin=500,
            discard_frac=0.2,
            quiet=True,
            n_workers=1,
            ln_n=ln_n,
        )

        # --- Analysis (notebook cell 4) ---
        analysis = analyze_replicas(re_results, mu1, sig1, mu2, sig2)
        pf = analysis["pareto_front"]
        samples = re_results["samples"]

        # # --- Pareto front plot (notebook cell 4 style) ---
        # fig, ax = plt.subplots(figsize=(6, 5))

        # all_E1, all_E2 = [], []
        # for a in range(M1_grid):
        #     for b in range(M2_grid):
        #         all_E1.extend(samples[(a, b)]["E1"].tolist())
        #         all_E2.extend(samples[(a, b)]["E2"].tolist())

        # ax.scatter(all_E1, all_E2, s=1, alpha=0.1, color="gray", rasterized=True)
        # ax.plot(pf[:, 0], pf[:, 1], "-", color="gray", lw=2, label="Pareto front")

        # # Natural-mean cross-hair (1 std dev arms)
        # ax.plot([mu1 - sig1, mu1 + sig1], [mu2, mu2], "-", color="k", alpha=1)
        # ax.plot([mu1, mu1], [mu2 - sig2, mu2 + sig2], "-", color="k", alpha=1)
        # ax.plot(mu1, mu2, "o", color="k", ms=10, mew=2, label="natural mean")

        # ax.set_xlabel(f"$E_1$ ({pf1})")
        # ax.set_ylabel(f"$E_2$ ({pf2})")
        # ax.set_title(f"Pareto front  (overlap = {overlap_nuc} nt)")
        # ax.legend(fontsize=11)
        # plt.tight_layout()
        # plt.savefig(os.path.join(plot_dir, f"pareto_{overlap_nuc:04d}nt.png"), dpi=150)
        # plt.close(fig)

        # --- Metrics (notebook cell 5) ---
        within, zscore_dist = within_and_distance(pf, mu1, sig1, mu2, sig2)

        elapsed = time.time() - t0
        print(f"  within Pareto front : {within}")
        print(f"  z-score distance    : {zscore_dist:.3f}")
        print(f"  elapsed time        : {elapsed:.1f} s")

        # --- Save this row ---
        writer.writerow([index, pf1, pf2, frame, overlap_nuc,
                         int(within), zscore_dist, elapsed])
        fh.flush()
        timings.append(elapsed)

    fh.close()

    # Marker file: written only once every overlap of this index is in the csv,
    # so the collector can tell "finished" from "killed part way" without
    # needing to know how many overlaps each index owns.
    if not args.test:
        open(f"{outdir}/row_{index:05d}.done", "w").close()

    # --- Test mode: timing extrapolation ---
    if args.test:
        total_time = sum(timings)
        mean_time = total_time / len(timings)
        n_all = len(all_overlaps)
        extrapolated = mean_time * n_all
        print(f"\n--- TEST MODE TIMING SUMMARY ---")
        print(f"  Overlaps tested      : {len(timings)}")
        print(f"  Total time           : {total_time:.1f} s  ({total_time / 60:.1f} min)")
        print(f"  Mean per overlap     : {mean_time:.1f} s")
        print(f"  Total overlaps       : {n_all}")
        print(f"  Extrapolated total   : {extrapolated:.0f} s  ({extrapolated / 3600:.1f} h)")


if __name__ == "__main__":
    main()
