"""
GA Worker Module for Multiprocessing

This module contains all functions needed for parallel GA optimization of
overlapping gene mutation paths. Designed for Windows multiprocessing compatibility.

Worker functions are in a separate .py file because Windows uses 'spawn' for
multiprocessing, which requires functions to be importable from a module.

Includes:
- generate_single_sequence: Worker for parallel sequence generation
- GeneticPathFinder: GA optimizer for mutation path ordering
- process_overlap_trial: Combined worker for full trial processing

- Kabir Husain and Orson Kirsch, with assistance from Claude Code (Anthropic)
"""

import numpy as np
from numba import njit, prange
import overlappingGenes as og

# Use standard tqdm (not tqdm.auto) for subprocess compatibility
try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no progress bar if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# NUCLEOTIDE ENCODING
# =============================================================================

NT_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INT_TO_NT = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


def seq_to_array(seq_str):
    """Convert string sequence to numpy uint8 array."""
    return np.array([NT_TO_INT[c] for c in seq_str], dtype=np.uint8)


def array_to_seq(seq_arr):
    """Convert numpy uint8 array back to string."""
    return ''.join(INT_TO_NT[i] for i in seq_arr)


def get_mutations_arrays(seq1, seq2):
    """Get mutations as numpy arrays for numba compatibility."""
    positions = []
    old_nts = []
    new_nts = []
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            positions.append(i)
            old_nts.append(NT_TO_INT[seq1[i]])
            new_nts.append(NT_TO_INT[seq2[i]])
    return (np.array(positions, dtype=np.int32),
            np.array(old_nts, dtype=np.uint8),
            np.array(new_nts, dtype=np.uint8))


# =============================================================================
# SEQUENCE UTILITIES
# =============================================================================

def hamming_distance(seq1, seq2):
    """Calculate Hamming distance between two sequences."""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def get_mutation_path(seq1, seq2):
    """Get list of positions where sequences differ."""
    return [(i, seq1[i], seq2[i]) for i in range(len(seq1)) if seq1[i] != seq2[i]]


def find_closest_pair(sequences):
    """Find the pair of sequences with minimum Hamming distance."""
    n = len(sequences)
    min_dist = float('inf')
    best_pair = (0, 1)

    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(sequences[i], sequences[j])
            if d < min_dist:
                min_dist = d
                best_pair = (i, j)

    return best_pair, min_dist


def find_pair_with_target_hamming(sequences, target_hamming, tolerance):
    """
    Find a pair of sequences with Hamming distance close to target.

    Returns (i, j, hamming_dist) tuple, or None if no sequences.
    Returns best match found, even if outside tolerance.
    """
    n = len(sequences)
    if n < 2:
        return None

    best_pair = None
    best_diff = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(sequences[i], sequences[j])
            diff = abs(d - target_hamming)
            if diff < best_diff:
                best_diff = diff
                best_pair = (i, j, d)
                if diff <= tolerance:
                    return best_pair  # Found acceptable pair, return early

    return best_pair  # Return best found even if outside tolerance


# =============================================================================
# PARALLEL SEQUENCE GENERATION WORKER
# =============================================================================

def generate_single_sequence(args):
    """
    Worker function to generate ONE functional overlapping sequence.
    Designed for Pool.map() - all params passed explicitly.

    Args:
        args: tuple containing:
            (overlap, trial_idx, seq_idx, seed,
             Jvec1, hvec1, Jvec2, hvec2,
             prot1_len, prot2_len,
             mc_iterations, mc_temp_1, mc_temp_2,
             mean_e1, mean_e2, std_e1, std_e2, z_score)

    Returns:
        dict with keys: overlap, trial, seq_idx, success, sequence, energies (or error)
    """
    (overlap, trial_idx, seq_idx, seed,
     Jvec1, hvec1, Jvec2, hvec2,
     prot1_len, prot2_len,
     mc_iterations, mc_temp_1, mc_temp_2,
     mean_e1, mean_e2, std_e1, std_e2, z_score) = args

    np.random.seed(seed)

    try:
        initial_seq = og.initial_seq_no_stops(prot1_len, prot2_len, overlap, quiet=True)
        result = og.overlapped_sequence_generator_int(
            (Jvec1, hvec1), (Jvec2, hvec2), initial_seq,
            T1=mc_temp_1, T2=mc_temp_2,
            numberofiterations=mc_iterations,
            quiet=True, whentosave=100.0,
            nat_mean1=mean_e1, nat_mean2=mean_e2,
            nat_std1=std_e1, nat_std2=std_e2,
            use_z_score=z_score
        )
        return {
            'overlap': overlap,
            'trial': trial_idx,
            'seq_idx': seq_idx,
            'success': True,
            'sequence': result[6],
            'energies': result[5]
        }
    except Exception as e:
        return {
            'overlap': overlap,
            'trial': trial_idx,
            'seq_idx': seq_idx,
            'success': False,
            'error': str(e)
        }


# =============================================================================
# NUMBA-OPTIMIZED ENERGY FUNCTIONS
# =============================================================================

@njit
def calculate_energies_from_array(seq_arr, Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2):
    """
    Calculate E1 and E2 from a uint8 array sequence.
    Returns (E1, E2) tuple. Returns (inf, inf) if stop codons are present.

    Note: DCA params passed explicitly (no globals) for multiprocessing compatibility.
    """
    len_seq_1_n = len_aa_1 * 3
    len_seq_2_n = len_aa_2 * 3

    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)

    og.split_sequence_and_to_numeric_out(seq_arr, len_seq_1_n, len_seq_2_n,
                                          aa_seq_1, aa_seq_2, rc_buffer)

    # Check for internal stop codons
    for i in range(len_aa_1 - 1):
        if aa_seq_1[i] == 21:
            return np.inf, np.inf
    for i in range(len_aa_2 - 1):
        if aa_seq_2[i] == 21:
            return np.inf, np.inf

    E1 = og.calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = og.calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)

    return E1, E2


@njit
def evaluate_path_fitness_numba(path_order, seq_arr, mut_positions, mut_new_nts,
                                 Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2,
                                 nat_mean_1, nat_mean_2, nat_std_1, nat_std_2, use_z_score):
    """
    Numba-optimized fitness evaluation.
    Works with numpy arrays instead of Python strings/lists.

    If use_z_score=True and nat_std values > 0, uses z-score normalized distance.
    """
    current_seq = seq_arr.copy()

    E1, E2 = calculate_energies_from_array(current_seq, Jvec1, hvec1, Jvec2, hvec2,
                                            len_aa_1, len_aa_2)

    # Calculate distance - optionally z-score normalized
    if use_z_score and nat_std_1 > 0 and nat_std_2 > 0:
        max_distance = abs(E1 - nat_mean_1)/nat_std_1 + abs(E2 - nat_mean_2)/nat_std_2
    else:
        max_distance = abs(E1 - nat_mean_1) + abs(E2 - nat_mean_2)

    for i in range(len(path_order)):
        idx = path_order[i]
        pos = mut_positions[idx]
        new_nt = mut_new_nts[idx]
        current_seq[pos] = new_nt

        E1, E2 = calculate_energies_from_array(current_seq, Jvec1, hvec1, Jvec2, hvec2,
                                                len_aa_1, len_aa_2)

        if use_z_score and nat_std_1 > 0 and nat_std_2 > 0:
            distance = abs(E1 - nat_mean_1)/nat_std_1 + abs(E2 - nat_mean_2)/nat_std_2
        else:
            distance = abs(E1 - nat_mean_1) + abs(E2 - nat_mean_2)

        if distance > max_distance:
            max_distance = distance

    return max_distance


@njit(parallel=True)
def evaluate_population_parallel(population, seq_arr, mut_positions, mut_new_nts,
                                  Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2,
                                  nat_mean_1, nat_mean_2, nat_std_1, nat_std_2, use_z_score):
    """
    Parallel evaluation of entire population using numba prange.
    population: 2D array of shape (pop_size, n_mutations)
    """
    pop_size = population.shape[0]
    fitnesses = np.empty(pop_size, dtype=np.float64)

    for i in prange(pop_size):
        fitnesses[i] = evaluate_path_fitness_numba(
            population[i], seq_arr, mut_positions, mut_new_nts,
            Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2,
            nat_mean_1, nat_mean_2, nat_std_1, nat_std_2, use_z_score
        )

    return fitnesses


@njit
def path_has_stop_codons(path_order, seq_arr, mut_positions, mut_new_nts,
                          len_aa_1, len_aa_2):
    """
    Check if any intermediate sequence along a mutation path has internal stop codons.
    Returns True if any stop codon is found, False if the entire path is clean.
    More efficient than full fitness evaluation since it skips energy calculations
    and exits early on the first stop codon encountered.
    """
    current_seq = seq_arr.copy()
    len_seq_1_n = len_aa_1 * 3
    len_seq_2_n = len_aa_2 * 3

    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)

    for i in range(len(path_order)):
        idx = path_order[i]
        pos = mut_positions[idx]
        new_nt = mut_new_nts[idx]
        current_seq[pos] = new_nt

        og.split_sequence_and_to_numeric_out(current_seq, len_seq_1_n, len_seq_2_n,
                                              aa_seq_1, aa_seq_2, rc_buffer)

        for j in range(len_aa_1 - 1):
            if aa_seq_1[j] == 21:
                return True
        for j in range(len_aa_2 - 1):
            if aa_seq_2[j] == 21:
                return True

    return False


@njit
def calculate_energies_separate(seq_str_arr, Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2):
    """
    Calculate E1 and E2 separately for an overlapping sequence (from string converted to array).
    Returns (E1, E2) tuple. Returns (inf, inf) if stop codons are present.
    """
    len_seq_1_n = len_aa_1 * 3
    len_seq_2_n = len_aa_2 * 3

    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)

    og.split_sequence_and_to_numeric_out(seq_str_arr, len_seq_1_n, len_seq_2_n,
                                          aa_seq_1, aa_seq_2, rc_buffer)

    for i in range(len_aa_1 - 1):
        if aa_seq_1[i] == 21:
            return np.inf, np.inf
    for i in range(len_aa_2 - 1):
        if aa_seq_2[i] == 21:
            return np.inf, np.inf

    E1 = og.calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = og.calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)

    return E1, E2


# =============================================================================
# PATH ENERGY CALCULATION
# =============================================================================

def get_path_energies(path_order, seq_start, mutations,
                      Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2,
                      nat_mean_1, nat_mean_2, nat_std_1=None, nat_std_2=None, z_score=False):
    """Get energies and distances at each step along the path."""
    current_seq = seq_start
    seq_arr = og.seq_str_to_int_array(current_seq)

    E1, E2 = calculate_energies_separate(seq_arr, Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2)

    path_energies = [E1 + E2]

    # Calculate distance - optionally z-score normalized
    if z_score and nat_std_1 is not None and nat_std_2 is not None:
        path_distances = [abs(E1 - nat_mean_1)/nat_std_1 + abs(E2 - nat_mean_2)/nat_std_2]
    else:
        path_distances = [abs(E1 - nat_mean_1) + abs(E2 - nat_mean_2)]

    for idx in path_order:
        pos, old_nt, new_nt = mutations[idx]
        current_seq = current_seq[:pos] + new_nt + current_seq[pos+1:]
        seq_arr = og.seq_str_to_int_array(current_seq)

        E1, E2 = calculate_energies_separate(seq_arr, Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2)
        path_energies.append(E1 + E2)

        if z_score and nat_std_1 is not None and nat_std_2 is not None:
            path_distances.append(abs(E1 - nat_mean_1)/nat_std_1 + abs(E2 - nat_mean_2)/nat_std_2)
        else:
            path_distances.append(abs(E1 - nat_mean_1) + abs(E2 - nat_mean_2))

    return np.array(path_energies), np.array(path_distances)


# =============================================================================
# GENETIC ALGORITHM PATH FINDER
# =============================================================================

class GeneticPathFinder:
    """
    GA to find optimal mutation ordering that minimizes max distance from natural.
    Uses numba-accelerated parallel fitness evaluation for speed.

    All DCA parameters passed explicitly (no globals) for multiprocessing compatibility.
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

        # Keep string mutations for final path extraction
        self.mutations = get_mutation_path(seq_start, seq_end)
        self.fitness_history = []

    def initialize_population(self):
        """Create initial random population as 2D numpy array.
        All paths are guaranteed to be stop-codon free (resampled if necessary)."""
        population = np.zeros((self.pop_size, self.n_mutations), dtype=np.int32)
        max_attempts = 100000
        for i in range(self.pop_size):
            found = False
            for attempt in range(max_attempts):
                population[i] = np.random.permutation(self.n_mutations)
                if not path_has_stop_codons(population[i], self.seq_arr,
                                             self.mut_positions, self.mut_new_nts,
                                             self.len_aa_1, self.len_aa_2):
                    found = True
                    break
            if not found:
                print(f"Warning: Could not find stop-codon-free path for individual {i} "
                      f"after {max_attempts} attempts")
        return population

    def evaluate_population(self, population):
        """Parallel fitness evaluation using numba."""
        return evaluate_population_parallel(
            population, self.seq_arr, self.mut_positions, self.mut_new_nts,
            self.Jvec1, self.hvec1, self.Jvec2, self.hvec2, self.len_aa_1, self.len_aa_2,
            self.nat_mean_1, self.nat_mean_2, self.nat_std_1, self.nat_std_2, self.z_score
        )

    def select_parents(self, population, fitnesses):
        """Tournament selection."""
        parents = []
        for _ in range(2):
            indices = np.random.choice(len(population), 3, replace=False)
            best_idx = indices[np.argmin(fitnesses[indices])]
            parents.append(population[best_idx].copy())
        return parents

    def crossover(self, parent1, parent2):
        """Order crossover (OX1) for permutations."""
        size = len(parent1)
        if size < 2:
            return parent1.copy()
        start, end = sorted(np.random.choice(size, 2, replace=False))

        child = np.full(size, -1, dtype=np.int32)
        child[start:end] = parent1[start:end]

        in_child = set(child[start:end])
        p2_remaining = [x for x in parent2 if x not in in_child]
        idx = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = p2_remaining[idx]
                idx += 1

        return child

    def mutate(self, individual):
        """Swap mutation for permutations."""
        if np.random.rand() < self.mutation_rate and len(individual) >= 2:
            i, j = np.random.choice(len(individual), 2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def run(self, verbose=False):
        if self.n_mutations == 0:
            return [], 0.0, np.array([]), np.array([])

        population = self.initialize_population()

        # Initialize best to first individual
        best_individual = population[0].copy()
        best_fitness = evaluate_path_fitness_numba(
            best_individual, self.seq_arr, self.mut_positions, self.mut_new_nts,
            self.Jvec1, self.hvec1, self.Jvec2, self.hvec2, self.len_aa_1, self.len_aa_2,
            self.nat_mean_1, self.nat_mean_2, self.nat_std_1, self.nat_std_2, self.z_score
        )

        iterator = tqdm(range(self.n_generations), desc="GA", leave=False) if verbose else range(self.n_generations)

        for gen in iterator:
            fitnesses = self.evaluate_population(population)

            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            self.fitness_history.append(best_fitness)

            sorted_idx = np.argsort(fitnesses)
            new_population = np.zeros_like(population)

            for i in range(self.elitism_count):
                new_population[i] = population[sorted_idx[i]]

            idx = self.elitism_count
            while idx < self.pop_size:
                max_resample = 100
                child = None
                for _ in range(max_resample):
                    p1_idx = sorted_idx[np.random.choice(self.pop_size // 2)]
                    p2_idx = sorted_idx[np.random.choice(self.pop_size // 2)]
                    parent1, parent2 = population[p1_idx], population[p2_idx]

                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    if not path_has_stop_codons(child, self.seq_arr,
                                                 self.mut_positions, self.mut_new_nts,
                                                 self.len_aa_1, self.len_aa_2):
                        break
                new_population[idx] = child
                idx += 1

            population = new_population

        # Get final path energies
        path_energies, path_distances = get_path_energies(
            best_individual.tolist(), self.seq_start, self.mutations,
            self.Jvec1, self.hvec1, self.Jvec2, self.hvec2, self.len_aa_1, self.len_aa_2,
            self.nat_mean_1, self.nat_mean_2,
            nat_std_1=self.nat_std_1, nat_std_2=self.nat_std_2, z_score=self.z_score
        )

        return best_individual.tolist(), best_fitness, path_energies, path_distances


# =============================================================================
# MAIN WORKER FUNCTION
# =============================================================================

def process_overlap_trial(args, DCA_params_1, DCA_params_2, stats, config):
    """
    Worker function for processing a single (overlap, trial) work unit.

    This function is designed for multiprocessing - all parameters are passed
    explicitly with no reliance on global variables.

    Args:
        args: tuple of (overlap, trial_idx, random_seed)
        DCA_params_1: tuple of (Jvec_1, hvec_1)
        DCA_params_2: tuple of (Jvec_2, hvec_2)
        stats: tuple of (mean_e1, std_e1, mean_e2, std_e2, prot1_len, prot2_len)
        config: dict with keys:
            - N_SEQUENCES: number of sequences to generate
            - MC_ITERATIONS: Monte Carlo iterations per sequence
            - MC_TEMP_1, MC_TEMP_2: MC temperatures
            - GA_POPULATION: GA population size
            - GA_GENERATIONS: GA generations
            - Z_SCORE: whether to use z-score normalization

    Returns:
        dict with trial results
    """
    overlap, trial_idx, seed = args
    Jvec1, hvec1 = DCA_params_1
    Jvec2, hvec2 = DCA_params_2
    mean_e1, std_e1, mean_e2, std_e2, prot1_len, prot2_len = stats

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Extract config
    n_sequences = config['N_SEQUENCES']
    mc_iterations = config['MC_ITERATIONS']
    mc_temp_1 = config['MC_TEMP_1']
    mc_temp_2 = config['MC_TEMP_2']
    ga_population = config['GA_POPULATION']
    ga_generations = config['GA_GENERATIONS']
    z_score = config['Z_SCORE']

    # Target Hamming distance control (None = use closest pair)
    target_hamming = config.get('TARGET_HAMMING', None)
    hamming_tolerance = config.get('HAMMING_TOLERANCE', 10)
    max_sequences = config.get('MAX_SEQUENCES', 100)

    # Calculate amino acid lengths
    len_aa_1 = prot1_len + 1  # +1 for stop codon
    len_aa_2 = prot2_len + 1

    # Step 1: Generate functional sequences
    # If target Hamming specified, keep generating until valid pair found
    sequences = []
    energies = []
    found_valid_pair = False

    while len(sequences) < max_sequences:
        # Generate a batch of sequences
        batch_size = n_sequences if len(sequences) == 0 else min(n_sequences, max_sequences - len(sequences))

        for i in range(batch_size):
            try:
                initial_seq = og.initial_seq_no_stops(prot1_len, prot2_len, overlap, quiet=True)

                result = og.overlapped_sequence_generator_int(
                    DCA_params_1, DCA_params_2, initial_seq,
                    T1=mc_temp_1, T2=mc_temp_2,
                    numberofiterations=mc_iterations,
                    quiet=True,
                    whentosave=100.0,
                    nat_mean1=mean_e1,
                    nat_mean2=mean_e2,
                    nat_std1=std_e1,
                    nat_std2=std_e2,
                    use_z_score=z_score
                )

                best_seq = result[6]
                best_energies = result[5]

                sequences.append(best_seq)
                energies.append(best_energies)
            except Exception:
                continue

        # Check if we have enough sequences and a valid pair
        if len(sequences) < 2:
            continue

        if target_hamming is not None:
            # Check if we have a pair within tolerance of target
            pair_result = find_pair_with_target_hamming(sequences, target_hamming, hamming_tolerance)
            if pair_result and abs(pair_result[2] - target_hamming) <= hamming_tolerance:
                found_valid_pair = True
                break
        else:
            # No target specified, any pair works (original behavior)
            found_valid_pair = True
            break

    # Check if enough sequences generated
    if len(sequences) < 2:
        return {
            'overlap': overlap,
            'trial': trial_idx,
            'seed': seed,
            'error': 'Not enough sequences generated',
            'hamming_distance': np.nan,
            'n_mutations': np.nan,
            'max_distance_from_natural': np.nan,
            'start_distance': np.nan,
            'end_distance': np.nan,
            'start_energy': np.nan,
            'end_energy': np.nan,
            'mean_e1_generated': np.nan,
            'mean_e2_generated': np.nan,
            'ga_converged': False
        }

    energies = np.array(energies)

    # Step 2: Select pair based on target Hamming or closest
    if target_hamming is not None:
        pair_result = find_pair_with_target_hamming(sequences, target_hamming, hamming_tolerance)
        p1, p2, ham_dist = pair_result
    else:
        (p1, p2), ham_dist = find_closest_pair(sequences)

    seq_start = sequences[p1]
    seq_end = sequences[p2]

    # Step 3: Run Genetic Algorithm
    ga = GeneticPathFinder(
        seq_start, seq_end,
        Jvec1, hvec1, Jvec2, hvec2, len_aa_1, len_aa_2,
        nat_mean_1=mean_e1, nat_mean_2=mean_e2,
        nat_std_1=std_e1, nat_std_2=std_e2, z_score=z_score,
        pop_size=ga_population, n_generations=ga_generations
    )

    best_order, max_distance, path_energies, path_distances = ga.run(verbose=False)

    # Determine if GA converged
    ga_converged = (len(ga.fitness_history) > 1 and
                    ga.fitness_history[-1] < ga.fitness_history[0])

    return {
        'overlap': overlap,
        'trial': trial_idx,
        'seed': seed,
        'error': None,
        'hamming_distance': ham_dist,
        'n_mutations': len(ga.mutations),
        'max_distance_from_natural': max_distance,
        'start_distance': path_distances[0] if len(path_distances) > 0 else np.nan,
        'end_distance': path_distances[-1] if len(path_distances) > 0 else np.nan,
        'start_energy': path_energies[0] if len(path_energies) > 0 else np.nan,
        'end_energy': path_energies[-1] if len(path_energies) > 0 else np.nan,
        'mean_e1_generated': energies[:, 0].mean(),
        'mean_e2_generated': energies[:, 1].mean(),
        'ga_converged': ga_converged
    }
