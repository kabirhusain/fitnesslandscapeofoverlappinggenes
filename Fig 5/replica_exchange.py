"""
2D Replica Exchange Monte Carlo for overlapping gene design.

Implements a grid of replicas at different (T1, T2) values with periodic
swap attempts between neighbors, enabling MBAR-based analysis.

The local MC steps (the expensive part) run in parallel across replicas
using ProcessPoolExecutor.  Swap attempts remain sequential.

Uses functions from overlappingGenes.py for energy calculations and
sequence manipulation.

- Kabir Husain and Orson Kirsch, with assistance from Claude Code (Anthropic)

"""

import os
import numpy as np
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from overlappingGenes import (
    calculate_Energy,
    calculate_Delta_Energy,
    split_sequence_and_to_numeric_out,
    seq_str_to_int_array,
    int_array_to_seq_str,
    CODON_TABLE_NUMERIC,
    initial_seq_no_stops,
    extract_params,
    load_natural_energies,
    set_seed,
)


# ---------------------------------------------------------------------------
# Temperature grid
# ---------------------------------------------------------------------------

def make_temperature_grid(T_min=0.3, T_max=3.0, M1=10, M2=10):
    """Log-spaced 2D temperature grid.

    T_alpha^(k) = T_min * (T_max / T_min) ** ((k-1) / (M-1))

    Returns
    -------
    T1_values : ndarray (M1,)
    T2_values : ndarray (M2,)
    """
    T1_values = T_min * (T_max / T_min) ** (np.arange(M1) / max(M1 - 1, 1))
    T2_values = T_min * (T_max / T_min) ** (np.arange(M2) / max(M2 - 1, 1))
    
    return T1_values, T2_values


# ---------------------------------------------------------------------------
# Core MC at fixed temperatures  (@njit)
# ---------------------------------------------------------------------------

@njit
def _run_mc_steps(Jvec1, hvec1, Jvec2, hvec2,
                  seq, aa_seq_1, aa_seq_2,
                  T1, T2, n_steps, E1, E2):
    """Run *n_steps* of standard overlap MC at fixed (T1, T2).

    Mirrors the Metropolis logic of ``overlapped_sequence_generator_int``
    but without history tracking or best-energy tracking.

    Parameters
    ----------
    seq : uint8 array – nucleotide sequence (modified in-place on accept)
    aa_seq_1, aa_seq_2 : int32 arrays – current AA translations (modified)
    E1, E2 : current energies for protein 1 and 2

    Returns
    -------
    seq, aa_seq_1, aa_seq_2, E1, E2, n_accepted
    """
    sequence_L = len(seq)
    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    len_aa_1 = len_seq_1_n // 3
    len_aa_2 = len_seq_2_n // 3

    # Working buffers
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)

    n_accepted = 0

    for _step in range(n_steps):
        # 1. Propose single-nucleotide mutation
        pos = np.random.randint(0, sequence_L)
        old_nt = seq[pos]
        idx = np.random.randint(0, 3)
        if idx >= old_nt:
            idx += 1
        new_nt = idx
        seq[pos] = new_nt

        # 2. Translate
        split_sequence_and_to_numeric_out(
            seq, len_seq_1_n, len_seq_2_n,
            aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # 3. Stop-codon check
        stop_err = False
        if aa_seq_1_new[len_aa_1 - 1] != 21 or aa_seq_2_new[len_aa_2 - 1] != 21:
            stop_err = True
        else:
            for i in range(len_aa_1 - 1):
                if aa_seq_1_new[i] == 21:
                    stop_err = True
                    break
            if not stop_err:
                for i in range(len_aa_2 - 1):
                    if aa_seq_2_new[i] == 21:
                        stop_err = True
                        break

        if stop_err:
            seq[pos] = old_nt
            continue

        # 4. Delta-E for each protein
        delta_H_1 = 0.0
        delta_H_2 = 0.0

        aa_pos_1 = -1
        new_aa_1 = -1
        for i in range(len_aa_1 - 1):
            if aa_seq_1[i] != aa_seq_1_new[i]:
                aa_pos_1 = i
                new_aa_1 = aa_seq_1_new[i]
                break
        if aa_pos_1 != -1:
            delta_H_1 = calculate_Delta_Energy(
                aa_seq_1, Jvec1, hvec1, aa_pos_1, new_aa_1)

        aa_pos_2 = -1
        new_aa_2 = -1
        for i in range(len_aa_2 - 1):
            if aa_seq_2[i] != aa_seq_2_new[i]:
                aa_pos_2 = i
                new_aa_2 = aa_seq_2_new[i]
                break
        if aa_pos_2 != -1:
            delta_H_2 = calculate_Delta_Energy(
                aa_seq_2, Jvec2, hvec2, aa_pos_2, new_aa_2)

        # 5. Metropolis
        delta_H = delta_H_1 / T1 + delta_H_2 / T2
        accept = False
        if delta_H <= 0:
            accept = True
        elif np.random.rand() < np.exp(-delta_H):
            accept = True

        if accept:
            for i in range(len_aa_1):
                aa_seq_1[i] = aa_seq_1_new[i]
            for i in range(len_aa_2):
                aa_seq_2[i] = aa_seq_2_new[i]
            E1 += delta_H_1
            E2 += delta_H_2
            n_accepted += 1
        else:
            seq[pos] = old_nt

    return seq, aa_seq_1, aa_seq_2, E1, E2, n_accepted


# ---------------------------------------------------------------------------
# Swap acceptance  (@njit)
# ---------------------------------------------------------------------------

@njit
def _attempt_swap(E1_i, E2_i, E1_j, E2_j,
                  inv_T1_i, inv_T2_i, inv_T1_j, inv_T2_j):
    """Metropolis acceptance for swapping configurations of two replicas.

    p_swap = min(1, exp[(1/T1_i - 1/T1_j)*(E1_i - E1_j)
                      + (1/T2_i - 1/T2_j)*(E2_i - E2_j)])

    Returns True if accepted.
    """
    delta = ((inv_T1_i - inv_T1_j) * (E1_i - E1_j)
             + (inv_T2_i - inv_T2_j) * (E2_i - E2_j))
    if delta >= 0:
        return True
    if np.random.rand() < np.exp(delta):
        return True
    return False


# ---------------------------------------------------------------------------
# Worker functions for ProcessPoolExecutor
# ---------------------------------------------------------------------------

# Per-worker-process cache (populated once by _init_worker, reused every task)
_worker_state = {}


def _init_worker(Jvec1, hvec1, Jvec2, hvec2):
    """Initializer run once in each worker process.

    Stores the (large) DCA parameter arrays so they are pickled only once
    per worker rather than once per task.  Also seeds the numba RNG from
    the worker's PID to avoid identical streams in forked processes.
    """
    _worker_state["Jvec1"] = Jvec1
    _worker_state["hvec1"] = hvec1
    _worker_state["Jvec2"] = Jvec2
    _worker_state["hvec2"] = hvec2
    set_seed(os.getpid())


def _worker_init_replica(args):
    """Worker task: generate a random overlapping DNA sequence, translate,
    compute initial energies, and equilibrate for N_equil MC steps.
    """
    L1_aa, L2_aa, overlap_nuc, T1, T2, N_equil, seed = args
    Jvec1 = _worker_state["Jvec1"]
    hvec1 = _worker_state["hvec1"]
    Jvec2 = _worker_state["Jvec2"]
    hvec2 = _worker_state["hvec2"]

    set_seed(seed)

    init_str = initial_seq_no_stops(L1_aa, L2_aa, overlap_nuc, quiet=True)
    seq_int = seq_str_to_int_array(init_str)

    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    aa1 = np.empty(len_seq_1_n // 3, dtype=np.int32)
    aa2 = np.empty(len_seq_2_n // 3, dtype=np.int32)
    rc_buf = np.empty(len_seq_2_n, dtype=np.uint8)
    split_sequence_and_to_numeric_out(
        seq_int, len_seq_1_n, len_seq_2_n, aa1, aa2, rc_buf)

    E1 = calculate_Energy(aa1[:-1], Jvec1, hvec1)
    E2 = calculate_Energy(aa2[:-1], Jvec2, hvec2)

    seq_int, aa1, aa2, E1, E2, _ = _run_mc_steps(
        Jvec1, hvec1, Jvec2, hvec2,
        seq_int, aa1, aa2, T1, T2, N_equil, E1, E2)

    return seq_int, aa1, aa2, E1, E2


def _worker_mc(args):
    """Worker task: run N_swap MC steps for a single replica."""
    seq, aa1, aa2, T1, T2, n_steps, E1, E2, seed = args
    set_seed(seed)
    seq, aa1, aa2, E1, E2, nacc = _run_mc_steps(
        _worker_state["Jvec1"], _worker_state["hvec1"],
        _worker_state["Jvec2"], _worker_state["hvec2"],
        seq, aa1, aa2, T1, T2, n_steps, E1, E2)
    return seq, aa1, aa2, E1, E2, nacc


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def replica_exchange(DCA_params_1, DCA_params_2,
                     L1_aa, L2_aa, overlap_nuc,
                     T1_values, T2_values,
                     N_swap=100, N_total=1_000_000,
                     N_equil=10_000, N_thin=100,
                     discard_frac=0.2, n_workers=None, quiet=False):
    """2D replica-exchange Monte Carlo.

    Parameters
    ----------
    DCA_params_1, DCA_params_2 : list  [Jvec, hvec] for each protein
    L1_aa, L2_aa : int  protein lengths in AA (without stop)
    overlap_nuc : int  overlap in nucleotides
    T1_values, T2_values : 1D arrays  temperature grids
    N_swap : int  MC steps between swap attempts per replica
    N_total : int  total MC steps per replica (approx)
    N_equil : int  equilibration steps before main loop
    N_thin : int  thinning interval for sample collection
    discard_frac : float  fraction of main-loop samples to discard (burn-in)
    n_workers : int or None
        Number of parallel worker processes.  ``None`` (default) uses
        ``min(cpu_count, n_replicas)``.  Set to 1 to disable parallelism.

    Returns
    -------
    dict with keys:
        samples  – dict[(a,b)] -> dict(E1=array, E2=array, seqs=list)
        swap_rates – dict[(a1,b1,a2,b2)] -> acceptance rate
        T1_values, T2_values
    """
    Jvec1, hvec1 = DCA_params_1[0], DCA_params_1[1]
    Jvec2, hvec2 = DCA_params_2[0], DCA_params_2[1]

    M1 = len(T1_values)
    M2 = len(T2_values)
    n_replicas = M1 * M2

    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    len_aa_1 = len_seq_1_n // 3
    len_aa_2 = len_seq_2_n // 3

    # Decide parallelism
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, n_replicas)
    use_parallel = n_workers > 1

    # Helper: flat index order for the grid
    grid_indices = [(a, b) for a in range(M1) for b in range(M2)]

    # ----- Initialise replicas -----
    replicas = {}

    if use_parallel:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(Jvec1, hvec1, Jvec2, hvec2),
        )

    try:
        if use_parallel:
            init_seeds = np.random.randint(0, 2**31, size=n_replicas)
            init_args = [
                (L1_aa, L2_aa, overlap_nuc,
                 T1_values[a], T2_values[b], N_equil,
                 int(init_seeds[idx]))
                for idx, (a, b) in enumerate(grid_indices)
            ]
            init_results = list(executor.map(_worker_init_replica, init_args))
            for idx, (a, b) in enumerate(grid_indices):
                replicas[(a, b)] = init_results[idx]
        else:
            for a in range(M1):
                for b in range(M2):
                    init_str = initial_seq_no_stops(
                        L1_aa, L2_aa, overlap_nuc, quiet=True)
                    seq_int = seq_str_to_int_array(init_str)

                    aa1 = np.empty(len_aa_1, dtype=np.int32)
                    aa2 = np.empty(len_aa_2, dtype=np.int32)
                    rc_buf = np.empty(len_seq_2_n, dtype=np.uint8)
                    split_sequence_and_to_numeric_out(
                        seq_int, len_seq_1_n, len_seq_2_n, aa1, aa2, rc_buf)

                    E1 = calculate_Energy(aa1[:-1], Jvec1, hvec1)
                    E2 = calculate_Energy(aa2[:-1], Jvec2, hvec2)

                    seq_int, aa1, aa2, E1, E2, _ = _run_mc_steps(
                        Jvec1, hvec1, Jvec2, hvec2,
                        seq_int, aa1, aa2,
                        T1_values[a], T2_values[b],
                        N_equil, E1, E2)

                    replicas[(a, b)] = (seq_int, aa1, aa2, E1, E2)

        if not quiet:
            mode = f"{n_workers} workers" if use_parallel else "sequential"
            print(f"Initialised {n_replicas} replicas, each equilibrated "
                  f"for {N_equil} steps ({mode})")

        # ----- Main loop -----
        n_rounds = max(1, N_total // N_swap)
        n_collect_start = int(n_rounds * discard_frac)

        # Sample storage
        samples = {(a, b): {"E1": [], "E2": [], "seqs": []}
                   for a in range(M1) for b in range(M2)}

        # Swap counters
        swap_attempts = {}
        swap_accepts = {}

        step_counter = 0

        for rnd in range(n_rounds):
            # --- Local MC for each replica (parallel or sequential) ---
            if use_parallel:
                mc_seeds = np.random.randint(0, 2**31, size=n_replicas)
                mc_args = []
                for idx, (a, b) in enumerate(grid_indices):
                    state = replicas[(a, b)]
                    mc_args.append((
                        state[0], state[1], state[2],
                        T1_values[a], T2_values[b], N_swap,
                        state[3], state[4], int(mc_seeds[idx]),
                    ))
                mc_results = list(executor.map(_worker_mc, mc_args))
                for idx, (a, b) in enumerate(grid_indices):
                    seq, aa1, aa2, E1, E2, _nacc = mc_results[idx]
                    replicas[(a, b)] = (seq, aa1, aa2, E1, E2)
            else:
                for a in range(M1):
                    for b in range(M2):
                        seq_int, aa1, aa2, E1, E2 = replicas[(a, b)]
                        seq_int, aa1, aa2, E1, E2, _nacc = _run_mc_steps(
                            Jvec1, hvec1, Jvec2, hvec2,
                            seq_int, aa1, aa2,
                            T1_values[a], T2_values[b],
                            N_swap, E1, E2)
                        replicas[(a, b)] = (seq_int, aa1, aa2, E1, E2)

            step_counter += N_swap

            # --- Collect samples ---
            if rnd >= n_collect_start and step_counter % N_thin < N_swap:
                for a in range(M1):
                    for b in range(M2):
                        _, _, _, E1, E2 = replicas[(a, b)]
                        samples[(a, b)]["E1"].append(E1)
                        samples[(a, b)]["E2"].append(E2)
                        samples[(a, b)]["seqs"].append(
                            replicas[(a, b)][0].copy())

            # --- Swap attempts (checkerboard) ---
            parity = rnd % 2

            # Horizontal swaps: same b, adjacent a
            for a in range(parity, M1 - 1, 2):
                for b in range(M2):
                    key = (a, b, a + 1, b)
                    swap_attempts[key] = swap_attempts.get(key, 0) + 1

                    _, _, _, E1_i, E2_i = replicas[(a, b)]
                    _, _, _, E1_j, E2_j = replicas[(a + 1, b)]

                    if _attempt_swap(
                            E1_i, E2_i, E1_j, E2_j,
                            1.0 / T1_values[a], 1.0 / T2_values[b],
                            1.0 / T1_values[a + 1], 1.0 / T2_values[b]):
                        replicas[(a, b)], replicas[(a + 1, b)] = (
                            replicas[(a + 1, b)], replicas[(a, b)])
                        swap_accepts[key] = swap_accepts.get(key, 0) + 1

            # Vertical swaps: same a, adjacent b
            for a in range(M1):
                for b in range(parity, M2 - 1, 2):
                    key = (a, b, a, b + 1)
                    swap_attempts[key] = swap_attempts.get(key, 0) + 1

                    _, _, _, E1_i, E2_i = replicas[(a, b)]
                    _, _, _, E1_j, E2_j = replicas[(a, b + 1)]

                    if _attempt_swap(
                            E1_i, E2_i, E1_j, E2_j,
                            1.0 / T1_values[a], 1.0 / T2_values[b],
                            1.0 / T1_values[a], 1.0 / T2_values[b + 1]):
                        replicas[(a, b)], replicas[(a, b + 1)] = (
                            replicas[(a, b + 1)], replicas[(a, b)])
                        swap_accepts[key] = swap_accepts.get(key, 0) + 1

            if not quiet and (rnd + 1) % max(1, n_rounds // 10) == 0:
                print(f"  Round {rnd+1}/{n_rounds}")

    finally:
        if use_parallel:
            executor.shutdown(wait=True)

    # ----- Convert sample lists to arrays -----
    for key in samples:
        samples[key]["E1"] = np.array(samples[key]["E1"])
        samples[key]["E2"] = np.array(samples[key]["E2"])

    # ----- Swap rates -----
    swap_rates = {}
    for key in swap_attempts:
        swap_rates[key] = (swap_accepts.get(key, 0)
                           / max(swap_attempts[key], 1))

    return {
        "samples": samples,
        "swap_rates": swap_rates,
        "T1_values": T1_values,
        "T2_values": T2_values,
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def analyze_replicas(results, nat_mean1, nat_std1, nat_mean2, nat_std2):
    """Compute z-score grids, find optimal (T1*, T2*), extract Pareto front.

    Parameters
    ----------
    results : dict  output of ``replica_exchange``
    nat_mean1, nat_std1 : natural energy statistics for protein 1
    nat_mean2, nat_std2 : natural energy statistics for protein 2

    Returns
    -------
    dict with z1_grid, z2_grid, T1_opt, T2_opt, pareto_front
    """
    T1_values = results["T1_values"]
    T2_values = results["T2_values"]
    samples = results["samples"]
    M1, M2 = len(T1_values), len(T2_values)

    z1_grid = np.full((M1, M2), np.nan)
    z2_grid = np.full((M1, M2), np.nan)

    for a in range(M1):
        for b in range(M2):
            E1_arr = samples[(a, b)]["E1"]
            E2_arr = samples[(a, b)]["E2"]
            if len(E1_arr) == 0:
                continue
            z1_grid[a, b] = (np.mean(E1_arr) - nat_mean1) / nat_std1
            z2_grid[a, b] = (np.mean(E2_arr) - nat_mean2) / nat_std2

    # Find optimal temperatures by minimising |z1| + |z2|
    combined = np.abs(z1_grid) + np.abs(z2_grid)
    best_idx = np.unravel_index(np.nanargmin(combined), combined.shape)
    T1_opt = T1_values[best_idx[0]]
    T2_opt = T2_values[best_idx[1]]

    # Pareto front: collect all (E1, E2) across grid, find non-dominated
    all_E1 = []
    all_E2 = []
    for a in range(M1):
        for b in range(M2):
            E1_arr = samples[(a, b)]["E1"]
            E2_arr = samples[(a, b)]["E2"]
            all_E1.extend(E1_arr.tolist())
            all_E2.extend(E2_arr.tolist())

    all_E1 = np.array(all_E1)
    all_E2 = np.array(all_E2)

    # Non-dominated: lower energy is better
    pareto_mask = np.ones(len(all_E1), dtype=np.bool_)
    for i in range(len(all_E1)):
        if not pareto_mask[i]:
            continue
        dominated = ((all_E1 <= all_E1[i]) & (all_E2 <= all_E2[i])
                     & ((all_E1 < all_E1[i]) | (all_E2 < all_E2[i])))
        if np.any(dominated):
            pareto_mask[i] = False

    pareto_front = np.column_stack([all_E1[pareto_mask],
                                     all_E2[pareto_mask]])
    # Sort by E1
    order = np.argsort(pareto_front[:, 0])
    pareto_front = pareto_front[order]

    return {
        "z1_grid": z1_grid,
        "z2_grid": z2_grid,
        "T1_opt": T1_opt,
        "T2_opt": T2_opt,
        "pareto_front": pareto_front,
    }


# ---------------------------------------------------------------------------
# Optional MBAR analysis
# ---------------------------------------------------------------------------

def mbar_analysis(results, nat_mean1, nat_std1, nat_mean2, nat_std2):
    """MBAR reweighting across the temperature grid.

    Requires ``pymbar`` to be installed.  Falls back gracefully if absent.

    Parameters
    ----------
    results : dict  output of ``replica_exchange``
    nat_mean1, nat_std1, nat_mean2, nat_std2 : natural energy statistics

    Returns
    -------
    dict with f_k, z1_mbar, z2_mbar, T1_opt, T2_opt  or None if pymbar
    is unavailable / too few samples.
    """
    try:
        import pymbar
    except ImportError:
        print("pymbar not available; skipping MBAR analysis")
        return None

    T1_values = results["T1_values"]
    T2_values = results["T2_values"]
    samples = results["samples"]
    M1, M2 = len(T1_values), len(T2_values)
    K = M1 * M2  # number of thermodynamic states

    # Collect all samples and their state indices
    E1_all = []
    E2_all = []
    state_indices = []  # which state each sample came from
    N_k = np.zeros(K, dtype=np.int64)

    for a in range(M1):
        for b in range(M2):
            k = a * M2 + b
            E1_arr = samples[(a, b)]["E1"]
            E2_arr = samples[(a, b)]["E2"]
            n = len(E1_arr)
            N_k[k] = n
            E1_all.extend(E1_arr.tolist())
            E2_all.extend(E2_arr.tolist())
            state_indices.extend([k] * n)

    E1_all = np.array(E1_all)
    E2_all = np.array(E2_all)
    N_total = len(E1_all)

    if N_total < K:
        print("Too few samples for MBAR analysis")
        return None

    # Build reduced energy matrix u_kn  (K x N_total)
    u_kn = np.zeros((K, N_total))
    for a in range(M1):
        for b in range(M2):
            k = a * M2 + b
            u_kn[k, :] = E1_all / T1_values[a] + E2_all / T2_values[b]

    # Run MBAR
    mbar_obj = pymbar.MBAR(u_kn, N_k)
    f_k = mbar_obj.f_k  # dimensionless free energies

    # Compute z-scores via MBAR expectations
    z1_mbar = np.full((M1, M2), np.nan)
    z2_mbar = np.full((M1, M2), np.nan)

    for a in range(M1):
        for b in range(M2):
            k = a * M2 + b
            # Reweight E1 to state k
            results_E1 = mbar_obj.compute_expectations(E1_all, state_dependent=False)
            results_E2 = mbar_obj.compute_expectations(E2_all, state_dependent=False)
            z1_mbar[a, b] = (results_E1["mu"][k] - nat_mean1) / nat_std1
            z2_mbar[a, b] = (results_E2["mu"][k] - nat_mean2) / nat_std2

    combined = np.abs(z1_mbar) + np.abs(z2_mbar)
    best_idx = np.unravel_index(np.nanargmin(combined), combined.shape)
    T1_opt = T1_values[best_idx[0]]
    T2_opt = T2_values[best_idx[1]]

    return {
        "f_k": f_k,
        "z1_mbar": z1_mbar,
        "z2_mbar": z2_mbar,
        "T1_opt": T1_opt,
        "T2_opt": T2_opt,
    }
