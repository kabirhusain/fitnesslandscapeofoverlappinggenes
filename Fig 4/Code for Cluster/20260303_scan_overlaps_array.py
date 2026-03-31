"""
Scan all overlaps with replica-exchange Monte Carlo for a given Pfam pair.

Pair index is taken from SLURM_ARRAY_TASK_ID (0-65) or --pair-index CLI arg.
All 66 pairs are the C(12,2) combinations of the 12 Pfam families.

For each overlap length:
  - Runs replica exchange (matching notebook cells 3-4 parameters)
  - Saves a Pareto front plot to {date}_Pareto_front_plots_{pf1}_{pf2}/
  - Records:
      (a) whether the natural-energy mean lies within the Pareto front
      (b) z-score distance to the Pareto front (0 if within)

Results saved to {date}_overlap_scan_results_{pf1}_{pf2}.pkl
"""

import argparse
import itertools
import os
import pickle
import time

import numpy as np

from overlappingGenes import extract_params, load_natural_energies
from replica_exchange import make_temperature_grid, replica_exchange, analyze_replicas

outdir = "Data"

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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replica-exchange overlap scan for a single Pfam pair.")
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Pair index (0-65). Overrides SLURM_ARRAY_TASK_ID.")
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

    # --- Determine pair index ---
    if args.pair_index is not None:
        pair_idx = args.pair_index
    else:
        pair_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

    pf1, pf2 = ALL_PAIRS[pair_idx]
    print(f"Pair index {pair_idx}: {pf1} x {pf2}")

    # --- Skip if results already exist ---
    date_str = "20260303"
    run_tag = f"{pf1}_{pf2}"
    pkl_path = f"{outdir}/{date_str}_overlap_scan_results_{run_tag}.pkl"
    if os.path.exists(pkl_path):
        print(f"Results already exist at {pkl_path}, skipping.")
        return

    # --- Load DCA parameters and natural-energy statistics (notebook cell 2) ---
    bmDCA_dir = "../0 bmDCA/"

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
    min_overlap = 12
    all_overlaps = list(range(min_overlap, max_overlap + 1))

    if args.test:
        # Evenly-spaced subset
        n_test = min(args.n_test_overlaps, len(all_overlaps))
        indices = np.linspace(0, len(all_overlaps) - 1, n_test, dtype=int)
        overlaps = [all_overlaps[i] for i in indices]
        print(f"TEST MODE: {n_test} overlaps out of {len(all_overlaps)}: {overlaps}")
    else:
        overlaps = all_overlaps
        print(f"Overlap range: {min_overlap} to {max_overlap} nt (step 1)")

    # --- Output directory for plots ---
    # plot_dir = f"{date_str}_Pareto_front_plots_{run_tag}"
    # os.makedirs(plot_dir, exist_ok=True)

    # --- Main scan loop ---
    rows = []
    timings = []

    for overlap_nuc in overlaps:
        t0 = time.time()
        print(f"\n=== overlap = {overlap_nuc} nt ===")

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

        rows.append((overlap_nuc, int(within), zscore_dist))
        timings.append(elapsed)

    # --- Save results ---
    with open(pkl_path, "wb") as f:
        pickle.dump(rows, f)

    # print(f"\nDone.  Results -> {pkl_path}  /")

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
