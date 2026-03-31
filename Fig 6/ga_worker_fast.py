"""
Fast GA Worker Module - Optimized

Key optimizations over ga_worker.py:
1. evaluate_path_fitness_delta: O(L) per mutation step instead of O(L^2)
2. evaluate_population_sequential: no prange to avoid thread contention with ProcessPoolExecutor
3. get_path_energies_separate_fast: numba-compiled array-based path energy extraction
4. breed_generation / initialize_population_numba: full GA loop in numba (no Python overhead)

- Kabir Husain and Orson Kirsch, with assistance from Claude Code (Anthropic)
"""

import numpy as np
from numba import njit
import overlappingGenes as og
from ga_worker import (
    seq_to_array, array_to_seq, get_mutations_arrays, get_mutation_path,
    path_has_stop_codons
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# OPTIMIZED FITNESS EVALUATION (DELTA ENERGY)
# =============================================================================

@njit
def evaluate_path_fitness_delta(path_order, seq_arr, mut_positions, mut_new_nts,
                                Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2,
                                nat_mean_1, nat_mean_2, nat_std_1, nat_std_2, use_z_score):
    """
    Evaluate fitness using O(L) delta energy per mutation step.
    One full O(L^2) energy calc at the start, then delta updates.
    """
    current_seq = seq_arr.copy()
    len_seq_1_n = len_aa_1 * 3
    len_seq_2_n = len_aa_2 * 3

    # Pre-allocate buffers
    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)

    # Initial full energy calculation O(L^2)
    og.split_sequence_and_to_numeric_out(current_seq, len_seq_1_n, len_seq_2_n,
                                          aa_seq_1, aa_seq_2, rc_buffer)

    # Check for internal stop codons in initial sequence
    for i in range(len_aa_1 - 1):
        if aa_seq_1[i] == 21:
            return np.inf
    for i in range(len_aa_2 - 1):
        if aa_seq_2[i] == 21:
            return np.inf

    E1 = og.calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = og.calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)

    # Initial distance
    if use_z_score and nat_std_1 > 0 and nat_std_2 > 0:
        max_distance = abs(E1 - nat_mean_1) / nat_std_1 + abs(E2 - nat_mean_2) / nat_std_2
    else:
        max_distance = abs(E1 - nat_mean_1) + abs(E2 - nat_mean_2)

    for i in range(len(path_order)):
        idx = path_order[i]
        pos = mut_positions[idx]
        new_nt = mut_new_nts[idx]
        current_seq[pos] = new_nt

        # Translate to find new AAs
        og.split_sequence_and_to_numeric_out(current_seq, len_seq_1_n, len_seq_2_n,
                                              aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # Check for internal stop codons
        has_stop = False
        for j in range(len_aa_1 - 1):
            if aa_seq_1_new[j] == 21:
                has_stop = True
                break
        if not has_stop:
            for j in range(len_aa_2 - 1):
                if aa_seq_2_new[j] == 21:
                    has_stop = True
                    break
        if has_stop:
            return np.inf

        # Delta energy for protein 1 - O(L)
        delta_E1 = 0.0
        for j in range(len_aa_1 - 1):
            if aa_seq_1[j] != aa_seq_1_new[j]:
                delta_E1 = og.calculate_Delta_Energy(aa_seq_1, Jvec1, hvec1, j, aa_seq_1_new[j])
                break

        # Delta energy for protein 2 - O(L)
        delta_E2 = 0.0
        for j in range(len_aa_2 - 1):
            if aa_seq_2[j] != aa_seq_2_new[j]:
                delta_E2 = og.calculate_Delta_Energy(aa_seq_2, Jvec2, hvec2, j, aa_seq_2_new[j])
                break

        E1 += delta_E1
        E2 += delta_E2

        # Update AA buffers
        for j in range(len_aa_1):
            aa_seq_1[j] = aa_seq_1_new[j]
        for j in range(len_aa_2):
            aa_seq_2[j] = aa_seq_2_new[j]

        # Track max distance
        if use_z_score and nat_std_1 > 0 and nat_std_2 > 0:
            distance = abs(E1 - nat_mean_1) / nat_std_1 + abs(E2 - nat_mean_2) / nat_std_2
        else:
            distance = abs(E1 - nat_mean_1) + abs(E2 - nat_mean_2)
        if distance > max_distance:
            max_distance = distance

    return max_distance


@njit
def evaluate_population_sequential(population, seq_arr, mut_positions, mut_new_nts,
                                    Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2,
                                    nat_mean_1, nat_mean_2, nat_std_1, nat_std_2, use_z_score):
    """
    Sequential population evaluation (no prange) to avoid thread contention
    when used with ProcessPoolExecutor.
    """
    pop_size = population.shape[0]
    fitnesses = np.empty(pop_size, dtype=np.float64)
    for i in range(pop_size):
        fitnesses[i] = evaluate_path_fitness_delta(
            population[i], seq_arr, mut_positions, mut_new_nts,
            Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2,
            nat_mean_1, nat_mean_2, nat_std_1, nat_std_2, use_z_score)
    return fitnesses


# =============================================================================
# FAST PATH ENERGY EXTRACTION
# =============================================================================

@njit
def get_path_energies_separate_fast(path_order, seq_arr, mut_positions, mut_new_nts,
                                     Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2):
    """
    Get E1 and E2 separately at each step along the mutation path.
    Returns (e1_array, e2_array) each of length n_mutations + 1.
    Uses delta energy for O(L) per step instead of O(L^2).
    """
    n_steps = len(path_order)
    e1_out = np.empty(n_steps + 1, dtype=np.float64)
    e2_out = np.empty(n_steps + 1, dtype=np.float64)

    current_seq = seq_arr.copy()
    len_seq_1_n = len_aa_1 * 3
    len_seq_2_n = len_aa_2 * 3

    # Pre-allocate buffers
    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)

    # Initial full energy
    og.split_sequence_and_to_numeric_out(current_seq, len_seq_1_n, len_seq_2_n,
                                          aa_seq_1, aa_seq_2, rc_buffer)

    E1 = og.calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = og.calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)
    e1_out[0] = E1
    e2_out[0] = E2

    for i in range(n_steps):
        idx = path_order[i]
        pos = mut_positions[idx]
        new_nt = mut_new_nts[idx]
        current_seq[pos] = new_nt

        # Translate
        og.split_sequence_and_to_numeric_out(current_seq, len_seq_1_n, len_seq_2_n,
                                              aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # Delta energy for protein 1
        delta_E1 = 0.0
        for j in range(len_aa_1 - 1):
            if aa_seq_1[j] != aa_seq_1_new[j]:
                delta_E1 = og.calculate_Delta_Energy(aa_seq_1, Jvec1, hvec1, j, aa_seq_1_new[j])
                break

        # Delta energy for protein 2
        delta_E2 = 0.0
        for j in range(len_aa_2 - 1):
            if aa_seq_2[j] != aa_seq_2_new[j]:
                delta_E2 = og.calculate_Delta_Energy(aa_seq_2, Jvec2, hvec2, j, aa_seq_2_new[j])
                break

        E1 += delta_E1
        E2 += delta_E2
        e1_out[i + 1] = E1
        e2_out[i + 1] = E2

        # Update AA buffers
        for j in range(len_aa_1):
            aa_seq_1[j] = aa_seq_1_new[j]
        for j in range(len_aa_2):
            aa_seq_2[j] = aa_seq_2_new[j]

    return e1_out, e2_out


# =============================================================================
# NUMBA-COMPILED GA OPERATORS
# =============================================================================

@njit
def _crossover_ox1(parent1, parent2, child, in_child_buf):
    """OX1 crossover for permutations. Writes result into child buffer."""
    size = len(parent1)
    if size < 2:
        for i in range(size):
            child[i] = parent1[i]
        return

    # Pick two distinct random indices
    a = np.random.randint(0, size)
    b = np.random.randint(0, size - 1)
    if b >= a:
        b += 1
    if a < b:
        start, end = a, b
    else:
        start, end = b, a

    # Reset in_child buffer (indexed by mutation value, not position)
    for i in range(size):
        in_child_buf[i] = False

    # Copy segment from parent1
    for i in range(start, end):
        child[i] = parent1[i]
        in_child_buf[parent1[i]] = True

    # Fill remaining positions from parent2 in order
    p2_idx = 0
    for i in range(size):
        if start <= i < end:
            continue
        while in_child_buf[parent2[p2_idx]]:
            p2_idx += 1
        child[i] = parent2[p2_idx]
        p2_idx += 1


@njit
def initialize_population_numba(pop_size, n_mutations, seq_arr, mut_positions,
                                 mut_new_nts, len_aa_1, len_aa_2):
    """Numba-compiled population initialization with stop-codon rejection."""
    population = np.empty((pop_size, n_mutations), dtype=np.int32)
    max_attempts = 100000
    for i in range(pop_size):
        for attempt in range(max_attempts):
            perm = np.random.permutation(n_mutations)
            for k in range(n_mutations):
                population[i, k] = perm[k]
            if not path_has_stop_codons(population[i], seq_arr, mut_positions,
                                         mut_new_nts, len_aa_1, len_aa_2):
                break
    return population


@njit
def breed_generation(population, fitnesses, new_population, seq_arr, mut_positions,
                     mut_new_nts, len_aa_1, len_aa_2, elitism_count, mutation_rate):
    """
    Numba-compiled breeding: sort, elitism, crossover+mutate+stop-check.
    Writes new generation into new_population.
    """
    pop_size = population.shape[0]
    n_mutations = population.shape[1]

    sorted_idx = np.argsort(fitnesses)

    # Elitism
    for i in range(elitism_count):
        for j in range(n_mutations):
            new_population[i, j] = population[sorted_idx[i], j]

    # Scratch buffers
    child = np.empty(n_mutations, dtype=np.int32)
    in_child_buf = np.empty(n_mutations, dtype=np.bool_)
    half_pop = pop_size // 2

    for idx in range(elitism_count, pop_size):
        for _ in range(100):  # max resample
            p1_idx = sorted_idx[np.random.randint(0, half_pop)]
            p2_idx = sorted_idx[np.random.randint(0, half_pop)]

            # OX1 crossover
            _crossover_ox1(population[p1_idx], population[p2_idx], child, in_child_buf)

            # Swap mutation
            if np.random.rand() < mutation_rate and n_mutations >= 2:
                si = np.random.randint(0, n_mutations)
                sj = np.random.randint(0, n_mutations - 1)
                if sj >= si:
                    sj += 1
                child[si], child[sj] = child[sj], child[si]

            # Stop codon check
            if not path_has_stop_codons(child, seq_arr, mut_positions, mut_new_nts,
                                         len_aa_1, len_aa_2):
                break

        for j in range(n_mutations):
            new_population[idx, j] = child[j]

    return sorted_idx


# =============================================================================
# FAST GENETIC ALGORITHM PATH FINDER
# =============================================================================

class GeneticPathFinderFast:
    """
    Optimized GA path finder using delta energy evaluation and
    numba-compiled breeding loop.
    """

    def __init__(self, seq_start, seq_end,
                 Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2,
                 nat_mean_1, nat_mean_2, nat_std_1=None, nat_std_2=None, z_score=False,
                 pop_size=50, n_generations=100, mutation_rate=0.15, elitism=0.1):
        self.seq_start = seq_start
        self.seq_end = seq_end
        self.Jvec1 = Jvec1
        self.hvec1 = hvec1
        self.Jvec2 = Jvec2
        self.hvec2 = hvec2
        self.len_aa_1 = len_aa_1
        self.len_aa_2 = len_aa_2
        self.nat_mean_1 = nat_mean_1
        self.nat_mean_2 = nat_mean_2
        self.nat_std_1 = nat_std_1 if nat_std_1 is not None else 1.0
        self.nat_std_2 = nat_std_2 if nat_std_2 is not None else 1.0
        self.z_score = z_score

        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elitism_count = max(1, int(elitism * pop_size))

        # Convert sequences to arrays for numba
        self.seq_arr = seq_to_array(seq_start)
        self.mut_positions, self.mut_old_nts, self.mut_new_nts = get_mutations_arrays(seq_start, seq_end)
        self.n_mutations = len(self.mut_positions)

        self.fitness_history = []

    def initialize_population(self):
        """Create initial random population (wrapper for numba function)."""
        return initialize_population_numba(
            self.pop_size, self.n_mutations, self.seq_arr,
            self.mut_positions, self.mut_new_nts, self.len_aa_1, self.len_aa_2
        )

    def run(self, verbose=False):
        if self.n_mutations == 0:
            return [], 0.0, np.array([]), np.array([])

        population = initialize_population_numba(
            self.pop_size, self.n_mutations, self.seq_arr,
            self.mut_positions, self.mut_new_nts, self.len_aa_1, self.len_aa_2
        )

        # Initialize best to first individual
        best_individual = population[0].copy()
        best_fitness = evaluate_path_fitness_delta(
            best_individual, self.seq_arr, self.mut_positions, self.mut_new_nts,
            self.Jvec1, self.hvec1, self.Jvec2, self.hvec2, self.len_aa_1, self.len_aa_2,
            self.nat_mean_1, self.nat_mean_2, self.nat_std_1, self.nat_std_2, self.z_score
        )

        new_population = np.empty_like(population)

        iterator = tqdm(range(self.n_generations), desc="GA", leave=False) if verbose else range(self.n_generations)

        for gen in iterator:
            fitnesses = evaluate_population_sequential(
                population, self.seq_arr, self.mut_positions, self.mut_new_nts,
                self.Jvec1, self.hvec1, self.Jvec2, self.hvec2, self.len_aa_1, self.len_aa_2,
                self.nat_mean_1, self.nat_mean_2, self.nat_std_1, self.nat_std_2, self.z_score
            )

            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            self.fitness_history.append(best_fitness)

            breed_generation(
                population, fitnesses, new_population, self.seq_arr,
                self.mut_positions, self.mut_new_nts, self.len_aa_1, self.len_aa_2,
                self.elitism_count, self.mutation_rate
            )

            population, new_population = new_population, population

        # Get final path energies using fast delta-based extraction
        e1_path, e2_path = get_path_energies_separate_fast(
            best_individual, self.seq_arr, self.mut_positions, self.mut_new_nts,
            self.Jvec1, self.hvec1, self.Jvec2, self.hvec2, self.len_aa_1, self.len_aa_2
        )

        # Build combined energy and distance arrays for compatibility
        path_energies = e1_path + e2_path

        if self.z_score and self.nat_std_1 > 0 and self.nat_std_2 > 0:
            path_distances = np.abs(e1_path - self.nat_mean_1) / self.nat_std_1 + \
                             np.abs(e2_path - self.nat_mean_2) / self.nat_std_2
        else:
            path_distances = np.abs(e1_path - self.nat_mean_1) + \
                             np.abs(e2_path - self.nat_mean_2)

        return best_individual.tolist(), best_fitness, path_energies, path_distances
