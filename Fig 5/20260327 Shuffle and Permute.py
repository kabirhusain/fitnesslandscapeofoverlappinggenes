import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

sns.set_context("talk", rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15})
sns.set_style("whitegrid", {"grid.color": ".9", "grid.linestyle": "--",
                             "axes.edgecolor": ".6", "xtick.bottom": True, "ytick.left": True})

import overlappingGenes as og
from overlappingGenes import (
    extract_params, load_natural_energies, make_shuffled_genetic_code, make_aa_permuted_genetic_code, set_genetic_code,
)
from replica_exchange import make_temperature_grid, replica_exchange, analyze_replicas

import pickle

# Snapshot standard code before any modification
_STANDARD_TABLE   = dict(og.CODON_TABLE)
_STANDARD_NUMERIC = og.CODON_TABLE_NUMERIC.copy()

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


# --- Parameters ---
PF1      = "PF00072"
PF2      = "PF00009"
OVERLAPS  = [211, 317] # 317 is longest -2, 211 is potentially interesting Frame -1
N_CODES  = 100
SEEDS    = list(range(N_CODES))

bmDCA_dir = "../0 bmDCA/"

# REMC settings
T1_vals, T2_vals = make_temperature_grid(T_min=0.1, T_max=1.0, M1=20, M2=20)
RE_KWARGS = dict(
    N_swap=1000, N_total=1_000_000,
    N_equil=100_000, N_thin=1000,
    discard_frac=0.2, quiet=True,
)

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

print(f"{PF1}: L = {L1} AA,  <E> = {mu1:.1f} ± {sig1:.1f}")
print(f"{PF2}: L = {L2} AA,  <E> = {mu2:.1f} ± {sig2:.1f}")
print(f"Temperature grid: {len(T1_vals)}×{len(T2_vals)} = {len(T1_vals)*len(T2_vals)} replicas")


for OVERLAP in OVERLAPS:
    # --- Standard code ---
    set_genetic_code(_STANDARD_TABLE, _STANDARD_NUMERIC)

    re_std = replica_exchange(DCA_1, DCA_2, L1, L2, OVERLAP, T1_vals, T2_vals, **RE_KWARGS)
    analysis_std = analyze_replicas(re_std, mu1, sig1, mu2, sig2)
    within_std, z_std = within_and_distance(analysis_std["pareto_front"], mu1, sig1, mu2, sig2)

    print(f"{OVERLAP}: Standard code:  within={within_std},  z={z_std:+.3f}")

    # --- Shuffled codes ---

    shuffled = []

    for seed in SEEDS:
        # Always reset to standard before generating a new code
        set_genetic_code(_STANDARD_TABLE, _STANDARD_NUMERIC)
        code_dict, code_numeric = make_shuffled_genetic_code(seed=seed)
        set_genetic_code(code_dict, code_numeric)

        re = replica_exchange(DCA_1, DCA_2, L1, L2, OVERLAP, T1_vals, T2_vals, **RE_KWARGS)
        analysis = analyze_replicas(re, mu1, sig1, mu2, sig2)
        within, z = within_and_distance(analysis["pareto_front"], mu1, sig1, mu2, sig2)

        shuffled.append({"seed": seed,
                        "within": within, "z": z})
        print(f"  {OVERLAP}: shuffle seed {seed:2d}:  within={str(within):5s}  z={z:+.3f}")

    # Reset to standard
    set_genetic_code(_STANDARD_TABLE, _STANDARD_NUMERIC)

    # --- AA Permuted codes ---

    permuted = []

    for seed in SEEDS:
        # Always reset to standard before generating a new code
        set_genetic_code(_STANDARD_TABLE, _STANDARD_NUMERIC)
        code_dict, code_numeric = make_aa_permuted_genetic_code(seed=seed)
        set_genetic_code(code_dict, code_numeric)

        re = replica_exchange(DCA_1, DCA_2, L1, L2, OVERLAP, T1_vals, T2_vals, **RE_KWARGS)
        analysis = analyze_replicas(re, mu1, sig1, mu2, sig2)
        within, z = within_and_distance(analysis["pareto_front"], mu1, sig1, mu2, sig2)

        permuted.append({"seed": seed,
                        "within": within, "z": z})
        print(f"  {OVERLAP}: permuted seed {seed:2d}:  within={str(within):5s}  z={z:+.3f}")

    # Reset to standard
    set_genetic_code(_STANDARD_TABLE, _STANDARD_NUMERIC)

    # Save results
    results = {
        "standard": {"within": within_std, "z": z_std},
        "shuffled": shuffled,
        "permuted": permuted,
    }
    with open(f"Pickles/20260327_{PF1}_{PF2}_{OVERLAP}_shuffled_permuted_codes_results.pkl", "wb") as f:
        pickle.dump(results, f)