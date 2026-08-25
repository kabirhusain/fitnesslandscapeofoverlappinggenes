#!/usr/bin/env python
"""
Overlap scan for DIVERGING and SAME-STRAND gene pairs, run locally.

The existing scans (20260330 Code Collection/Fig 4, and the local reduced-set
version in ../20260810 Finding a reduced set) only ever consider CONVERGENT
overlaps: gene 1 forward at the 5' end, gene 2 on the reverse strand at the
3' end, so the genes point at each other and their 3' ends share the overlap.

This script adds the other two topologies:

    convergent (existing)        diverging (new)            same strand (new)
     g1 5'------->3'                   5'------->3' g1       g1 5'------->3'
           3'<-------5' g2    g2 3'<-------5'                      5'------->3' g2
       [both 3' ends]              [both 5' ends]            [g1 3' end inside g2]

    geometry      stop codons inside the overlap
    convergent    2   gene 1's at the right edge, gene 2's (reverse strand) at the left
    same strand   1   gene 1's, read out of frame by gene 2
    diverging     0   both stops sit in single-coding flanks

so diverging overlaps are the least constrained of the three.

Only two things in overlappingGenes.py / replica_exchange.py encode the topology:
``split_sequence_and_to_numeric_out`` and ``initial_seq_no_stops``.  Replacements
for both are defined below and bound into ``replica_exchange`` by
``use_geometry``; neither file is edited.  Everything else -- the Metropolis
loop, the delta-energy trick, the swap logic, the temperature grid,
``analyze_replicas``, the Pareto metrics, the reduced sets and the cost model --
is the convergent scan's, unchanged, so the geometries are comparable.

Subcommands
-----------
  check   self-tests: geometry layout, initial conditions, energy bookkeeping,
          and the copied scan machinery against the convergent run
  run     scan a preset (or a custom set) and append results to ONE csv
  plot    draw the overlap-fraction figure, over the convergent curve

Typical use:
    python 20260810_strand_scan.py check
    python 20260810_strand_scan.py run  --geometry diverging --preset quick
    python 20260810_strand_scan.py plot --geometry diverging --preset quick

Only overlappingGenes.py and replica_exchange.py (both in this folder) are
needed; no other project code is imported.

- Kabir Husain, with assistance from Claude Code (Anthropic)
"""

import argparse
import csv
import itertools
import os
import time
import zlib
from functools import partial
from multiprocessing import Pool
from typing import NamedTuple

import numpy as np
from numba import njit

import replica_exchange as RX
from overlappingGenes import (CODON_TABLE, calculate_Delta_Energy,
                              calculate_Energy, codon_degeneracy_ln_n,
                              extract_params, get_rc_seq_out,
                              initial_seq_no_stops, load_natural_energies,
                              seq_str_to_int_array, set_seed,
                              split_sequence_and_to_numeric_out,
                              translate_numeric_out)
from replica_exchange import analyze_replicas, make_temperature_grid, replica_exchange

# ---------------------------------------------------------------------------
# Configuration  (identical to ../20260810 Finding a reduced set)
# ---------------------------------------------------------------------------

DATE = "20260821"
BMDCA_DIR = os.environ.get(
    "BMDCA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "0 bmDCA")) + os.sep
CONVERGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                              "20260810 Finding a reduced set") + os.sep
CONVERGENT_DATE = "20260810"   # that run's files keep their own date

# Codon-degeneracy correction, applied to the Metropolis acceptance only; the
# energies replica_exchange reports stay the raw DCA energies.  The standard
# code is never swapped in this script, so one vector at import is enough.
LN_N = codon_degeneracy_ln_n()

# Protein length in AA (without stop) for each family, from the bmDCA h-vectors.
# `run` re-derives them and asserts agreement.
FAMILY_LENGTHS = {
    "PF00096": 23, "PF00397": 28, "PF00018": 43, "PF00013": 58, "PF00076": 59,
    "PF00512": 62, "PF00595": 71, "PF00041": 74, "PF00017": 75, "PF07679": 76,
    "PF00011": 82, "PF00153": 85, "PF00271": 95, "PF02518": 101, "PF00072": 108,
    "PF00004": 110, "PF00009": 162,
}
ALL_FAMILIES = sorted(FAMILY_LENGTHS, key=FAMILY_LENGTHS.get)

MIN_OVERLAP = 12

# Replica-exchange parameters -- identical to 20260303_scan_overlaps_array.py
RE_KWARGS = dict(N_swap=500, N_total=100_000, N_equil=10_000, N_thin=500,
                 discard_frac=0.2, quiet=True)
T_GRID = dict(T_min=0.3, T_max=1.0, M1=11, M2=11)

PRESETS = {
    "quick": dict(
        families=["PF00018", "PF00017", "PF02518", "PF00072", "PF00004",
                  "PF00009"],
        step=10,
        note="~15 min -- pipeline check + rough curve",
    ),
    "hour": dict(
        families=["PF00397", "PF00013", "PF00512", "PF00017", "PF07679",
                  "PF00011", "PF00153", "PF00271", "PF02518", "PF00072",
                  "PF00004", "PF00009"],
        step=7,
        note="~1 h -- the figure, recognisably",
    ),
    "full": dict(
        families=ALL_FAMILIES,
        step=1,
        note="the whole scan -- cluster only, see the sbatch files",
    ),
    "long": dict(
        families=["PF00018", "PF00076", "PF00595", "PF00041", "PF00017",
                  "PF07679", "PF00011", "PF00153", "PF00271", "PF02518",
                  "PF00072", "PF00004", "PF00009"],
        step=5,
        note="~2 h -- as close to the full scan as subsampling gets",
    ),
}

# 20 cores, but only 8 are P-cores: measured throughput is x4.8 at 8 workers,
# x5.2 at 12 and x5.5 at 16.  12 buys 96% of the maximum and leaves 8 cores free.
DEFAULT_WORKERS = 12

# Single-core seconds per task ~= a + b*(L1+L2) + c*(L1^2 + L2^2), measured on
# this machine (20260810_timing_model.json, 2026-08-10).  The quadratic term is
# not algorithmic: the coupling matrices are tens of MB, so per-step cost grows
# once J stops fitting in cache.  Per-task cost is geometry-independent.
MODEL = dict(a=2.1259, b=0.0068216, c=4.49199e-4, speedup=5.2457)


# ---------------------------------------------------------------------------
# Geometry:  the two topology-dependent pieces
# ---------------------------------------------------------------------------

@njit
def _split_diverging(sequence, len_1_n, len_2_n, aa_out_1, aa_out_2, rc_buffer):
    """Gene 1 forward at the 3' end, gene 2 reverse-complement at the 5' end.

    Same signature as overlappingGenes.split_sequence_and_to_numeric_out.
    """
    translate_numeric_out(sequence[len(sequence) - len_1_n:], aa_out_1)
    get_rc_seq_out(sequence[:len_2_n], rc_buffer)
    translate_numeric_out(rc_buffer, aa_out_2)


@njit
def _split_same_strand(sequence, len_1_n, len_2_n, aa_out_1, aa_out_2, rc_buffer):
    """Both genes forward; gene 2 starts inside gene 1.  rc_buffer unused."""
    translate_numeric_out(sequence[:len_1_n], aa_out_1)
    translate_numeric_out(sequence[len(sequence) - len_2_n:], aa_out_2)


class Geometry(NamedTuple):
    """Everything that distinguishes one overlap topology from another.

    layout            (l1, l2, overlap) -> [(start, nt_length, reverse), ...]
                      in nucleotide coordinates of the construct, gene 1 first
    frame_sign        frame label is (frame_sign * overlap) % 3; see `frame_of`
    frames            frame labels the topology admits
    ordered_pairs     whether (A, B) and (B, A) are different constructs
    stops_in_overlap  deliberate stop codons inside the dual-coding region
    splitter          njit translation, with the signature of
                      overlappingGenes.split_sequence_and_to_numeric_out
    generator         starting-sequence builder, or None to use `initial_seq`
    """
    layout: object
    frame_sign: int
    frames: tuple
    ordered_pairs: bool
    stops_in_overlap: int
    splitter: object
    generator: object


GEOMETRY = {
    # gene 1 forward at the 3' end, gene 2 reverse-complement at the 5' end
    "diverging": Geometry(
        layout=lambda l1, l2, ov: [(l2 - ov, l1, False), (0, l2, True)],
        frame_sign=-1, frames=(0, 1, 2), ordered_pairs=False,
        stops_in_overlap=0, splitter=_split_diverging, generator=None),
    # both genes forward, gene 1 upstream; its stop codon lies inside gene 2
    "same": Geometry(
        layout=lambda l1, l2, ov: [(0, l1, False), (l1 - ov, l2, False)],
        frame_sign=1, frames=(1, 2), ordered_pairs=True,
        stops_in_overlap=1, splitter=_split_same_strand, generator=None),
    # the original: gene 1 forward at the 5' end, gene 2 reverse at the 3' end
    "convergent": Geometry(
        layout=lambda l1, l2, ov: [(0, l1, False), (l1 - ov, l2, True)],
        frame_sign=1, frames=(0, 1, 2), ordered_pairs=False,
        stops_in_overlap=2, splitter=split_sequence_and_to_numeric_out,
        generator=initial_seq_no_stops),
}
GEOMETRIES = tuple(GEOMETRY)


class Gene(NamedTuple):
    """One gene's footprint on the construct, in nucleotide coordinates."""
    start: int
    length: int
    reverse: bool
    stop: int        # first base of this gene's own stop codon


_NT = "ACGT"
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _revcomp(s):
    """Reverse complement, str -> str.  overlappingGenes has list -> list
    versions (fast_reverse_complement, plus a third copy nested inside
    initial_seq_no_stops); the repair loop below wants strings."""
    return s.translate(_COMPLEMENT)[::-1]


def _stop_codons():
    """Stop codons of the genetic code currently in force.

    Read from CODON_TABLE rather than hard-coded, because `set_genetic_code`
    rewrites it in place and the MC kernel judges stops by its numeric twin
    CODON_TABLE_NUMERIC.  Hard-coding TAA/TAG/TGA would let this generator and
    that kernel disagree under a shuffled code.
    """
    return [codon for codon, aa in CODON_TABLE.items() if aa == "*"]


def gene_spans(prot1, prot2, overlap, geometry):
    """Layout of the two genes on the construct: (seqL, [gene1, gene2]).

    For same strand gene 1 is the UPSTREAM gene: its C-terminus (and its stop
    codon) sits inside gene 2's N-terminal region.  See `family_pairs`.
    """
    l1, l2 = 3 * prot1 + 3, 3 * prot2 + 3
    genes = [Gene(start, ln, reverse, start if reverse else start + ln - 3)
             for start, ln, reverse
             in GEOMETRY[geometry].layout(l1, l2, overlap)]
    return l1 + l2 - overlap, genes


def _gene_seq(s, gene):
    """One gene's nucleotides as a string, 5'->3' along its own strand.
    `s` may be a string or a list of characters."""
    sub = "".join(s[gene.start:gene.start + gene.length])
    return _revcomp(sub) if gene.reverse else sub


def _internal_stops(s, genes):
    """Positions of every stop codon that is not a gene's own final codon."""
    stops = set(_stop_codons())
    bad = []
    for gene in genes:
        sub = _gene_seq(s, gene)
        for k in range(0, gene.length - 3, 3):
            if sub[k:k + 3] in stops:
                bad.append(gene.start + gene.length - k - 3 if gene.reverse
                           else gene.start + k)
    return bad


def initial_seq(prot1, prot2, overlap, quiet=False, *, geometry,
                max_repairs=200):
    """Random dual-coding sequence with no internal stops -- the replacement for
    overlappingGenes.initial_seq_no_stops, returning the same kind of string.

    That function needs ~160 lines of frame-by-frame case analysis because a
    convergent overlap has to carry a stop codon that the *opposite strand*
    reads through, which is a real constraint.  Neither new geometry has one:
    diverging puts no stop in the overlap at all, and same strand puts gene 1's
    stop inside gene 2, but a same-strand frame shift makes gene 2 read it as a
    codon ending in TA/TG or starting with A/G, none of which are stops (true
    for all 3 stop codons x 2 offsets x 64 flanking contexts).  So filling at
    random and repairing whatever stops appear is enough, and converges in ~2
    passes.

    The rng is seeded from the legacy global, which `_run_task` seeds per task,
    so initial conditions are reproducible (the original is not).
    """
    if frame_of(overlap, geometry) not in GEOMETRY[geometry].frames:
        raise ValueError(f"{geometry} overlaps cannot be in frame "
                         f"{frame_of(overlap, geometry)} (overlap={overlap})")
    seqL, genes = gene_spans(prot1, prot2, overlap, geometry)
    protected = {i for gene in genes for i in range(gene.stop, gene.stop + 3)}
    rng = np.random.default_rng(np.random.randint(2 ** 31))

    s = list(rng.choice(list(_NT), seqL))
    for gene in genes:
        stop = rng.choice(_stop_codons())
        s[gene.stop:gene.stop + 3] = list(
            _revcomp(stop) if gene.reverse else stop)

    for _ in range(max_repairs):
        bad = _internal_stops(s, genes)
        if not bad:
            return "".join(s)
        # Re-randomise one unprotected nucleotide of each offending codon.
        # There is always at least one: an offending codon can only collide
        # with the *other* gene's stop, and no geometry puts the two genes in
        # the same frame, so it can share at most 2 of its 3 nucleotides.
        for g in bad:
            p = int(rng.choice([g + j for j in range(3)
                                if g + j not in protected]))
            s[p] = rng.choice([n for n in _NT if n != s[p]])

    raise RuntimeError(f"could not build a {geometry} sequence for "
                       f"prot1={prot1} prot2={prot2} overlap={overlap}")


def _make_mc_kernel(split):
    """Build the fixed-(T1,T2) Metropolis kernel for one geometry.

    The body is replica_exchange._run_mc_steps verbatim except for the marked
    line: the hard-coded convergent translation becomes the closed-over
    `split`.  A factory rather than one copy per geometry.
    """

    @njit
    def _run_mc_steps(Jvec1, hvec1, Jvec2, hvec2,
                      seq, aa_seq_1, aa_seq_2,
                      T1, T2, n_steps, E1, E2, ln_n):
        sequence_L = len(seq)
        len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
        len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
        len_aa_1 = len_seq_1_n // 3
        len_aa_2 = len_seq_2_n // 3

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
            seq[pos] = idx

            # 2. Translate  <-- the only geometry-dependent line
            split(seq, len_seq_1_n, len_seq_2_n,
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

            # Codon-degeneracy correction.  Equivalent to sampling with
            # h -> h - T * ln n_codons, but with no factor of T: it cancels
            # against the 1/T already dividing the energy difference.  aa_seq_*
            # still holds the OLD residue here, since it is only updated on accept.
            if aa_pos_1 != -1:
                delta_H += ln_n[new_aa_1] - ln_n[aa_seq_1[aa_pos_1]]
            if aa_pos_2 != -1:
                delta_H += ln_n[new_aa_2] - ln_n[aa_seq_2[aa_pos_2]]

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

    return _run_mc_steps


_KERNELS = {"convergent": RX._run_mc_steps}   # captured before any patching


def kernel_for(name):
    """The Metropolis kernel for one geometry, compiled once and cached."""
    if name not in _KERNELS:
        _KERNELS[name] = _make_mc_kernel(GEOMETRY[name].splitter)
    return _KERNELS[name]


def use_geometry(name):
    """Point replica_exchange at one geometry, for this process and its forks.

    replica_exchange() and its pool workers are plain Python and look these
    three names up at call time, so rebinding them redirects the whole
    orchestrator without duplicating any of it.  "convergent" rebinds the
    originals, so this is reversible.

    Relies on `_run_task` passing n_workers=1, so replica_exchange never starts
    its own ProcessPoolExecutor.  If that ever changes, a spawn-based inner
    pool would get pristine convergent functions and silently write convergent
    numbers to a file named for another geometry.
    """
    if name not in GEOMETRY:
        raise SystemExit(f"unknown geometry {name!r}; pick from {GEOMETRIES}")
    geo = GEOMETRY[name]
    RX._run_mc_steps = kernel_for(name)
    RX.split_sequence_and_to_numeric_out = geo.splitter
    RX.initial_seq_no_stops = (geo.generator
                               or partial(initial_seq, geometry=name))


# ---------------------------------------------------------------------------
# Pareto-front metrics (verbatim from 20260303_scan_overlaps_array.py)
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
    """Returns (within_pareto: bool, zscore_distance: float)."""
    natural_mean = np.array([mu1, mu2])
    within = (_is_on_front(natural_mean, pareto_front, mu1, sig1, mu2, sig2) or
              _is_dominated(natural_mean, pareto_front, mu1, sig1, mu2, sig2))

    pf_z = np.array([[(p[0] - mu1) / sig1, (p[1] - mu2) / sig2]
                     for p in pareto_front])
    distances = np.sqrt(np.sum(pf_z ** 2, axis=1))
    zscore_dist = -float(np.min(distances)) if within else float(np.min(distances))

    return within, zscore_dist


# ---------------------------------------------------------------------------
# Task enumeration and cost model
# ---------------------------------------------------------------------------

def max_overlap(pf1, pf2):
    """Longest overlap in nt for a pair, as in the cluster script."""
    return min(FAMILY_LENGTHS[pf1], FAMILY_LENGTHS[pf2]) * 3 - 6


def frame_of(overlap_nuc, geometry):
    """Frame label, defined so the same number means the same thing everywhere.

    What is physical is the offset between the two genes' codon grids: a gene's
    codon boundaries sit at positions == its start (mod 3) whichever strand it
    is on, since its nt length is a multiple of 3.  That offset is
    (-overlap) % 3 for convergent and same strand, but (+overlap) % 3 for
    diverging -- the two genes swap which one carries the offset, so the frame
    relationship runs the other way with overlap length.

    The convergent scan labels its panels `overlap % 3`, so that convention is
    kept and diverging is labelled `(-overlap) % 3`.  Panel n then holds the
    same codon-grid offset in every geometry, which is what makes the figures
    comparable; a convergent 13 nt overlap pairs with a diverging 14 nt one.

    Arithmetic only, so this also works elementwise on a pandas Series.
    """
    return (GEOMETRY[geometry].frame_sign * overlap_nuc) % 3


def codon_pairing(geometry, frame, prot1=43, prot2=101):
    """Which codon position of gene 1 sits opposite which of gene 2.

    Returns {pos1: pos2} 0-indexed (so 2 is the wobble base).  The mapping is
    the same for every overlap in a panel, so the first one is representative.

    The two geometry classes differ in kind here, which is why panels do NOT
    correspond across them:

    * antiparallel (convergent, diverging) -- the genes read opposite strands,
      so the pairing is a REFLECTION and is symmetric.  Frame 1 is
      wobble-to-wobble, both ways round.
    * same strand -- the genes read the same strand at an offset, so the
      pairing is a CYCLIC SHIFT and is never symmetric.  In frame 1 the
      upstream wobble meets the downstream 1st position while the downstream
      wobble meets the upstream 2nd.  It also means (A upstream, frame 1) has
      the same A-to-B pairing as (B upstream, frame 2).
    """
    overlap = next(ov for ov in range(MIN_OVERLAP, MIN_OVERLAP + 9)
                   if frame_of(ov, geometry) == frame)
    _, genes = gene_spans(prot1, prot2, overlap, geometry)

    def position(i, gene):
        if not (gene.start <= i < gene.start + gene.length):
            return None
        return ((gene.start + gene.length - 1 - i) % 3 if gene.reverse
                else (i - gene.start) % 3)

    dual = range(max(g.start for g in genes),
                 min(g.start + g.length for g in genes))
    return dict(sorted({(position(i, genes[0]), position(i, genes[1]))
                        for i in dual}))


def frame_title(geometry, frame):
    """Panel title naming the codon-position relation, not just its number."""
    nth = {0: "1st", 1: "2nd", 2: "3rd"}
    pairing = codon_pairing(geometry, frame)
    forward = nth[pairing[2]]                                  # g1 wobble ->
    backward = nth[{v: k for k, v in pairing.items()}[2]]      # g2 wobble ->
    if forward == backward:                                    # symmetric
        return f"frame {frame}\nwobble $\\leftrightarrow$ {forward}"
    return (f"frame {frame}\nup wobble $\\rightarrow$ dn {forward}, "
            f"dn $\\rightarrow$ up {backward}")


def overlaps_for(pf1, pf2, step, geometry):
    """Overlap lengths to scan for one pair.

    Same strand has no frame 0 (see `GEOMETRY["same"].frames`): there the genes
    share a frame, gene 1's stop codon is an in-frame stop for gene 2, and the
    layout does not exist.
    """
    frames = GEOMETRY[geometry].frames
    return [ov for ov in range(MIN_OVERLAP, max_overlap(pf1, pf2) + 1, step)
            if frame_of(ov, geometry) in frames]


def family_pairs(families, geometry):
    """Pairs to scan.  Ordered for same strand, unordered for the rest.

    The antiparallel geometries couple like ends -- C-to-C for convergent,
    N-to-N for diverging -- so swapping the two families gives the same
    construct read off the other strand, and unordered pairs are complete.
    Same strand couples the upstream gene's C-terminus to the downstream
    gene's N-terminus, which is not symmetric: A upstream of B and B upstream
    of A are different configurations of different residues.  So it takes
    ordered pairs, with pf1 the UPSTREAM gene, and twice as many of them.
    """
    order = (itertools.permutations if GEOMETRY[geometry].ordered_pairs
             else itertools.combinations)
    return list(order(sorted(families), 2))


def enumerate_tasks(families, step, geometry):
    """[(pf1, pf2, overlap_nuc), ...]; for same strand pf1 is upstream."""
    return [(pf1, pf2, ov)
            for pf1, pf2 in family_pairs(families, geometry)
            for ov in overlaps_for(pf1, pf2, step, geometry)]


def task_seconds(pf1, pf2):
    L1, L2 = FAMILY_LENGTHS[pf1], FAMILY_LENGTHS[pf2]
    return MODEL["a"] + MODEL["b"] * (L1 + L2) + MODEL["c"] * (L1 ** 2 + L2 ** 2)


def predict_wall(tasks):
    """Predicted wall-clock seconds for a list of tasks."""
    return sum(task_seconds(pf1, pf2) for pf1, pf2, _ in tasks) / MODEL["speedup"]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_W = {}   # per-worker cache: parameters for the pair currently being scanned


def task_seed(pf1, pf2, overlap_nuc, run_seed):
    """Deterministic per-task seed (Python's hash() is salted per process)."""
    return zlib.crc32(f"{pf1}|{pf2}|{overlap_nuc}|{run_seed}".encode()) % (2 ** 31)


def _load_pair(pf1, pf2):
    """Load (and cache) DCA parameters + natural-energy stats for one pair."""
    if _W.get("pair") == (pf1, pf2):
        return
    _W.clear()
    J1, h1 = extract_params(f"{BMDCA_DIR}{pf1}/{pf1}_params.dat")
    J2, h2 = extract_params(f"{BMDCA_DIR}{pf2}/{pf2}_params.dat")
    ne1 = np.asarray(load_natural_energies(f"{BMDCA_DIR}{pf1}/{pf1}_naturalenergies.txt"),
                     dtype=float)
    ne2 = np.asarray(load_natural_energies(f"{BMDCA_DIR}{pf2}/{pf2}_naturalenergies.txt"),
                     dtype=float)
    _W.update(pair=(pf1, pf2), DCA1=[J1, h1], DCA2=[J2, h2],
              L1=len(h1) // 21, L2=len(h2) // 21,
              mu1=ne1.mean(), sig1=ne1.std(), mu2=ne2.mean(), sig2=ne2.std())
    assert _W["L1"] == FAMILY_LENGTHS[pf1] and _W["L2"] == FAMILY_LENGTHS[pf2], \
        f"length table disagrees with bmDCA files for {pf1}/{pf2}"


def _run_task(task):
    """One (pair, overlap): replica exchange -> Pareto front -> metrics."""
    pf1, pf2, overlap_nuc, seed = task
    t0 = time.time()
    _load_pair(pf1, pf2)
    set_seed(seed)
    np.random.seed(seed % (2 ** 31))

    T1_vals, T2_vals = make_temperature_grid(**T_GRID)
    re_results = replica_exchange(
        _W["DCA1"], _W["DCA2"], _W["L1"], _W["L2"], overlap_nuc,
        T1_vals, T2_vals, n_workers=1, ln_n=LN_N, **RE_KWARGS)

    analysis = analyze_replicas(re_results, _W["mu1"], _W["sig1"],
                                _W["mu2"], _W["sig2"])
    within, zscore_dist = within_and_distance(
        analysis["pareto_front"], _W["mu1"], _W["sig1"], _W["mu2"], _W["sig2"])

    return (pf1, pf2, overlap_nuc, int(within), zscore_dist, time.time() - t0)


def _warm_up_jit():
    """Compile the numba kernels in the parent so forked workers inherit them.

    Without this every worker pays the ~10 s compile cost on its first task.
    Call after use_geometry(), so the geometry's own kernel is the one compiled.
    """
    pf1, pf2 = "PF00096", "PF00397"
    J1, h1 = extract_params(f"{BMDCA_DIR}{pf1}/{pf1}_params.dat")
    J2, h2 = extract_params(f"{BMDCA_DIR}{pf2}/{pf2}_params.dat")
    T1_vals, T2_vals = make_temperature_grid(T_min=0.3, T_max=1.0, M1=2, M2=2)
    kwargs = dict(RE_KWARGS)
    kwargs.update(N_total=200, N_equil=50, N_swap=100, N_thin=100)
    res = replica_exchange([J1, h1], [J2, h2], len(h1) // 21, len(h2) // 21,
                           31, T1_vals, T2_vals, n_workers=1, ln_n=LN_N, **kwargs)
    analyze_replicas(res, 0.0, 1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

CSV_HEADER = ["pf1", "pf2", "overlap_nuc", "within_pareto", "zscore_dist", "seconds"]

# Dark2, matching the project's plotting style
FRAME_COLOURS = {0: (0, 0, 0),
                 1: (27 / 255, 158 / 255, 119 / 255),
                 2: (217 / 255, 95 / 255, 2 / 255)}


def csv_path(geometry, tag):
    return f"{DATE}_strand_scan_{geometry}_{tag}.csv"


def convergent_csv(tag):
    """The matching convergent run, for the plot overlay: same families, same
    step, so the same (pair, overlap) points."""
    return f"{CONVERGENT_DIR}{CONVERGENT_DATE}_reduced_scan_{tag}.csv"


def load_done(path):
    """Set of (pf1, pf2, overlap) already in the csv -- makes runs resumable."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for row in csv.DictReader(f):
            done.add((row["pf1"], row["pf2"], int(row["overlap_nuc"])))
    return done


def cmd_run(args):
    families, step, tag = resolve_set(args)

    tasks = enumerate_tasks(families, step, args.geometry)
    path = csv_path(args.geometry, tag)
    done = load_done(path)
    todo = [t for t in tasks if t not in done]

    pairs = family_pairs(families, args.geometry)
    print(f"{args.geometry} overlaps, set '{tag}': {len(families)} families, "
          f"{len(pairs)} pairs"
          f"{' (ordered, pf1 upstream)' if args.geometry == 'same' else ''}, "
          f"overlap step {step}, frames {GEOMETRY[args.geometry].frames}")
    print(f"  {len(tasks)} tasks total, {len(done)} already done, {len(todo)} to run")
    print(f"  predicted wall time: {predict_wall(tasks) / 3600:.2f} h "
          f"(whole set, {args.workers} workers)")
    print(f"  output -> {path}")
    if args.dry_run:
        return

    # Pair-sorted so each worker keeps one pair's parameters cached; a
    # deterministic seed per task keeps reruns reproducible.
    todo.sort()
    seeded = [(pf1, pf2, ov, task_seed(pf1, pf2, ov, args.seed))
              for pf1, pf2, ov in todo]

    use_geometry(args.geometry)
    print("  compiling numba kernels in parent ...", flush=True)
    t_warm = time.time()
    _warm_up_jit()
    print(f"  ... {time.time() - t_warm:.0f} s", flush=True)

    new_file = not os.path.exists(path)
    t_start = time.time()
    n_done = 0
    busy_seconds = 0.0   # wall time inside workers (contended)
    solo_seconds = 0.0   # what those tasks would cost on an idle single core
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(CSV_HEADER)
            fh.flush()
        with Pool(args.workers, initializer=use_geometry,
                  initargs=(args.geometry,)) as pool:
            for row in pool.imap(_run_task, seeded, chunksize=1):
                writer.writerow(row)
                fh.flush()
                busy_seconds += row[5]
                solo_seconds += task_seconds(row[0], row[1])
                n_done += 1
                if n_done % 10 == 0 or n_done == len(seeded):
                    el = time.time() - t_start
                    rate = n_done / el
                    eta = (len(seeded) - n_done) / rate if rate > 0 else 0
                    print(f"  {n_done}/{len(seeded)}  "
                          f"{el / 60:.1f} min elapsed, {eta / 60:.1f} min left  "
                          f"({rate * 60:.1f} tasks/min)", flush=True)

    wall = time.time() - t_start
    print(f"\nDone in {wall / 60:.1f} min -> {path}")

    # Speedup is throughput relative to an IDLE single core.  Per-task `seconds`
    # are contended (a task that takes 20 s alone takes ~40 s with 12 running),
    # so busy_seconds/wall counts busy workers, not speedup -- don't confuse them.
    if n_done:
        print(f"  realised speedup x{solo_seconds / wall:.2f} "
              f"(model assumed x{MODEL['speedup']:.2f}); "
              f"{busy_seconds / wall:.1f} workers busy on average")

    print(f"Now plot with:  python {os.path.basename(__file__)} plot "
          f"--geometry {args.geometry} --preset {tag}")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def _fractions(df, geometry):
    """Fraction of pairs beating natural energies, by overlap, per frame."""
    df = df.assign(frame=frame_of(df["overlap_nuc"], geometry),
                   better=df["zscore_dist"] < 0)
    return {frame: sub.groupby("overlap_nuc")["better"].mean().sort_index()
            for frame, sub in df.groupby("frame")}


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    families, step, tag = resolve_set(args)
    path = csv_path(args.geometry, tag)
    if not os.path.exists(path):
        raise SystemExit(f"No results at {path} -- run the scan first.")

    df = pd.read_csv(path)
    npairs = df.groupby(["pf1", "pf2"]).ngroups
    print(f"{len(df)} runs, {npairs} pairs, "
          f"overlaps {df.overlap_nuc.min()}-{df.overlap_nuc.max()}")

    expected = len(enumerate_tasks(families, step, args.geometry))
    if len(df) < expected:
        # Tasks run in pair order, so a partial csv is a biased sample of pairs,
        # not a random one -- the curve will shift as the rest land.
        print(f"  WARNING: {expected - len(df)} of {expected} tasks are still "
              f"missing; this curve is a biased partial sample.")

    curves = _fractions(df, args.geometry)
    frames = GEOMETRY[args.geometry].frames
    ref = None
    ref_path = convergent_csv(tag)
    # Panels only correspond across geometries when the codon-position pairing
    # is the same kind of relation.  Antiparallel pairings are reflections
    # (symmetric); same strand's are cyclic shifts (directed).  Overlaying
    # convergent on a same-strand panel would imply a correspondence that is
    # not there -- matching frame numbers, different relations.  See
    # `codon_pairing`.  --force-compare draws it anyway.
    comparable = all(codon_pairing(args.geometry, f)
                     == codon_pairing("convergent", f) for f in frames)
    if args.compare and not comparable and not args.force_compare:
        print(f"  NOT overlaying convergent: frame {frames[0]} means "
              f"{codon_pairing('convergent', frames[0])} there but "
              f"{codon_pairing(args.geometry, frames[0])} here (gene1 pos -> "
              f"gene2 pos, 0-indexed).  Same number, different relation.  "
              f"--force-compare to draw it anyway.")
    elif args.compare and os.path.exists(ref_path):
        ref = _fractions(pd.read_csv(ref_path), "convergent")
        if GEOMETRY[args.geometry].ordered_pairs:
            print(f"  NOTE: the convergent reference has "
                  f"{len(family_pairs(families, 'convergent'))} unordered pairs "
                  f"against this run's {npairs} ordered ones; the curves share "
                  f"overlap lengths but not a denominator.")
    elif args.compare:
        print(f"  (no convergent run at {ref_path}; drawing without the overlay)")

    sns.set_context("talk", rc={"font.size": 15, "axes.titlesize": 15,
                               "axes.labelsize": 15})
    sns.set_style("whitegrid", {"grid.color": ".9", "grid.linestyle": "--",
                                "axes.edgecolor": ".6", "xtick.bottom": True,
                                "ytick.left": True})

    if args.combine_frames:
        # Every frame on one axes, coloured -- for asking how overlappable the
        # geometry is overall rather than comparing panel to panel.
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        panels = [(frame, ax, FRAME_COLOURS[frame],
                   frame_title(args.geometry, frame).replace("\n", ": "))
                  for frame in frames]
    else:
        fig, axs = plt.subplots(1, len(frames), figsize=(4.3 * len(frames), 4.2),
                                sharey=True, squeeze=False)
        panels = [(frame, axis, "k",
                   f"{args.geometry} ({npairs} pairs, step {step})")
                  for frame, axis in zip(frames, axs[0])]

    for frame, ax, colour, label in panels:
        if ref is not None and frame in ref:
            r = ref[frame]
            ax.plot(r.index, r.rolling(args.smooth, min_periods=1,
                                       center=True).mean().values,
                    "-", color=(.6, .6, .6), lw=2, zorder=1, label="convergent")
        g = curves.get(frame)
        if g is not None:
            ax.plot(g.index, g.values, "o", color=colour, markersize=4,
                    zorder=3, label=label)
            ax.plot(g.index, g.rolling(args.smooth, min_periods=1,
                                       center=True).mean().values,
                    "-", color=colour, lw=2, zorder=2)
        ax.set_xticks(np.arange(0, 351, 50))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_yticklabels([f"{int(x * 100)}%" for x in np.arange(0, 1.01, 0.2)])
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, 335)
        ax.set_xlabel("overlap (nt)")
        if not args.combine_frames:
            ax.set_title(frame_title(args.geometry, frame), fontsize=12)

    first = panels[0][1]
    first.set_ylabel("pairs beating natural energies")
    if args.combine_frames:
        first.set_title(f"{args.geometry}  ({npairs} pairs, step {step})",
                        fontsize=13)
        first.legend(fontsize=9, loc="lower left")
    else:
        first.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    suffix = "_combined" if args.combine_frames else ""
    for ext in args.formats:
        out = f"{DATE}_overlap_fractions_{args.geometry}_{tag}{suffix}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=200)
        print(f"-> {out}")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def _check_layout_and_init():
    """Geometry layout, initial conditions, and the stop-codon bookkeeping."""
    print("frame labels")
    for geometry, geo in GEOMETRY.items():
        for ov in range(MIN_OVERLAP, MIN_OVERLAP + 12):
            frame = frame_of(ov, geometry)
            if frame not in geo.frames:
                continue
            _, genes = gene_spans(43, 101, ov, geometry)
            # a gene's codon boundaries sit at positions == its start (mod 3)
            offset = (genes[1].start - genes[0].start) % 3
            assert offset == (-frame) % 3, (geometry, ov, frame, offset)
        print(f"  {geometry:11s} panel n holds codon-grid offset (-n) % 3  OK")

    print("initial conditions")
    stops = _stop_codons()
    for geometry, geo in GEOMETRY.items():
        if geo.generator is not None:
            continue          # convergent keeps the original generator
        for prot1, prot2 in [(23, 28), (43, 101), (110, 162)]:
            overlaps = [ov for ov in range(MIN_OVERLAP, 3 * min(prot1, prot2) - 5)
                        if frame_of(ov, geometry) in geo.frames]
            for ov in overlaps[:3] + overlaps[-3:]:
                seqL, genes = gene_spans(prot1, prot2, ov, geometry)
                # the overlap is where the two gene spans intersect
                dual = range(max(g.start for g in genes),
                             min(g.start + g.length for g in genes))
                assert len(dual) == ov, (geometry, ov, len(dual))
                n_in = sum(g.stop in dual for g in genes)
                assert n_in == geo.stops_in_overlap, (geometry, ov, n_in)

                s = initial_seq(prot1, prot2, ov, geometry=geometry)
                assert len(s) == seqL == 3 * prot1 + 3 * prot2 + 6 - ov
                assert not _internal_stops(s, genes)
                for gene in genes:
                    last = _gene_seq(s, gene)[-3:]
                    assert last in stops, (geometry, ov, last)
            print(f"  {geometry:10s} prot1={prot1:3d} prot2={prot2:3d}: "
                  f"{len(overlaps)} overlaps, frames {geo.frames}, "
                  f"{geo.stops_in_overlap} stop(s) in the overlap  OK")


def _check_kernel():
    """Energy bookkeeping: the incremental E1/E2 must match a recomputation.

    Uses random DCA parameters -- this tests the kernel and the splitter, not
    the models, and needs no 67 MB parameter file.
    """
    print("kernel")
    rng = np.random.default_rng(0)
    prot1, prot2, ov = 15, 20, 31   # frame 1, the one both geometries allow
    hs, Js = [], []
    for L in (prot1, prot2):
        hs.append(rng.normal(0, 1, 21 * L))
        Js.append(rng.normal(0, 0.1, (L * (L - 1) // 2) * 441))

    for geometry, geo in GEOMETRY.items():
        if geo.generator is not None:
            continue          # convergent keeps the original generator
        split = geo.splitter
        len_1_n, len_2_n = 3 * prot1 + 3, 3 * prot2 + 3
        aa1 = np.empty(len_1_n // 3, dtype=np.int32)
        aa2 = np.empty(len_2_n // 3, dtype=np.int32)
        rc_buffer = np.empty(len_2_n, dtype=np.uint8)

        seq = seq_str_to_int_array(initial_seq(prot1, prot2, ov,
                                               geometry=geometry))
        split(seq, len_1_n, len_2_n, aa1, aa2, rc_buffer)
        E1 = calculate_Energy(aa1[:-1], Js[0], hs[0])
        E2 = calculate_Energy(aa2[:-1], Js[1], hs[1])

        set_seed(3)
        n_steps = 20_000
        seq, aa1, aa2, E1, E2, n_acc = kernel_for(geometry)(
            Js[0], hs[0], Js[1], hs[1], seq, aa1, aa2, 1.0, 1.0, n_steps, E1, E2)

        # A mis-wired splitter rejects essentially every move, so this catches a
        # kernel that is translating the wrong geometry.
        assert n_acc > n_steps // 100, f"{geometry}: {n_acc}/{n_steps} accepted"
        split(seq, len_1_n, len_2_n, aa1, aa2, rc_buffer)
        assert not (aa1[:-1] == 21).any() and not (aa2[:-1] == 21).any()
        for E, aa, J, h in ((E1, aa1, Js[0], hs[0]), (E2, aa2, Js[1], hs[1])):
            assert abs(E - calculate_Energy(aa[:-1], J, h)) < 1e-6, geometry
        print(f"  {geometry:10s} {n_acc}/{n_steps} moves accepted, "
              f"incremental energies match a full recomputation  OK")

    # The seam itself: rebinding must take for every geometry, and convergent
    # must put the originals back.  Everything else here bypasses use_geometry.
    for geometry, geo in GEOMETRY.items():
        use_geometry(geometry)
        assert RX.split_sequence_and_to_numeric_out is geo.splitter, geometry
        assert RX._run_mc_steps is kernel_for(geometry), geometry
        assert (RX.initial_seq_no_stops is geo.generator
                or RX.initial_seq_no_stops.keywords == {"geometry": geometry})
    use_geometry("convergent")
    assert RX.initial_seq_no_stops is initial_seq_no_stops
    print(f"  use_geometry rebinds all three names for "
          f"{'/'.join(GEOMETRY)}, convergent restores  OK")


def _check_machinery(args):
    """The scan machinery copied from the convergent script still agrees with it.

    Task enumeration is checked exactly against the convergent run's csv.  The
    numbers can only be checked statistically: the initial sequences there come
    from an unseeded rng, so the convergent run is not bit-reproducible.
    """
    import pandas as pd

    ref_path = convergent_csv("quick")
    if not os.path.exists(ref_path):
        print(f"machinery: no convergent run at {ref_path}, skipping")
        return
    ref = pd.read_csv(ref_path)

    print("machinery")
    families, step = PRESETS["quick"]["families"], PRESETS["quick"]["step"]
    mine = set(enumerate_tasks(families, step, "convergent"))
    theirs = set(zip(ref.pf1, ref.pf2, ref.overlap_nuc))
    assert mine == theirs, f"{len(mine ^ theirs)} tasks differ from the convergent run"
    print(f"  task enumeration: all {len(mine)} quick-preset tasks match "
          f"{os.path.basename(ref_path)}  OK")

    assert (ref.within_pareto.astype(bool) == (ref.zscore_dist < 0)).all()
    print("  within_pareto == (zscore_dist < 0) on every convergent row  OK")

    same_pairs = set(family_pairs(families, "same"))
    assert same_pairs == {p[::-1] for p in same_pairs}
    assert len(same_pairs) == 2 * len(family_pairs(families, "diverging"))
    print(f"  same strand enumerates both orderings of all "
          f"{len(same_pairs) // 2} pairs  OK")

    if args.no_rerun:
        return
    use_geometry("convergent")
    tasks = sorted(t for t in mine if {t[0], t[1]} == {"PF00017", "PF00018"})[:4]
    print(f"  re-running {len(tasks)} convergent tasks ({tasks[0][0]} x "
          f"{tasks[0][1]}) to compare with the recorded run ...", flush=True)
    _warm_up_jit()
    for pf1, pf2, ov in tasks:
        row = _run_task((pf1, pf2, ov, task_seed(pf1, pf2, ov, args.seed)))
        was = ref[(ref.pf1 == pf1) & (ref.pf2 == pf2) & (ref.overlap_nuc == ov)]
        d = row[4] - float(was.zscore_dist.iloc[0])
        flag = "OK" if abs(d) < 1.0 else "CHECK"
        print(f"    ov={ov:3d}  z {row[4]:+.3f} vs {float(was.zscore_dist.iloc[0]):+.3f}"
              f"  (d={d:+.3f}, MC sd 0.32)  {flag}")


def cmd_check(args):
    _check_layout_and_init()
    _check_kernel()
    _check_machinery(args)
    print("\nall checks passed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_set(args):
    """(families, step, tag) from --preset or explicit --families/--step."""
    if args.families:
        unknown = [f for f in args.families if f not in FAMILY_LENGTHS]
        if unknown:
            raise SystemExit(f"unknown families: {unknown}\n"
                             f"known: {' '.join(ALL_FAMILIES)}")
        if args.step is None:
            raise SystemExit("--step is required with --families")
        return (sorted(args.families, key=FAMILY_LENGTHS.get), args.step,
                args.tag or "custom")
    p = PRESETS[args.preset]
    step = args.step if args.step is not None else p["step"]
    return sorted(p["families"], key=FAMILY_LENGTHS.get), step, args.tag or args.preset


# ---------------------------------------------------------------------------
# cluster:  the same scan as `run`, one slurm array index at a time
# ---------------------------------------------------------------------------

def cluster_indices(families, geometry):
    """[(pf1, pf2, frame), ...] -- one slurm array index each.

    Splitting a pair by frame rather than giving it a whole index keeps the
    longest index to about a third of the pair, which packs better into a fixed
    number of concurrent slots.  Same strand has no frame 0, so it gets two
    indices per (ordered) pair and the others get three.
    """
    return [(pf1, pf2, frame)
            for pf1, pf2 in family_pairs(families, geometry)
            for frame in GEOMETRY[geometry].frames]


def cluster_dir(geometry, tag):
    return f"Data_{geometry}_{tag}"


def cmd_cluster(args):
    families, step, tag = resolve_set(args)
    indices = cluster_indices(families, args.geometry)

    if args.index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        args.index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if args.index is None:
        tasks = enumerate_tasks(families, step, args.geometry)
        print(f"{len(indices)} indices  ->  sbatch --array=0-{len(indices) - 1}")
        print(f"  {args.geometry}, set '{tag}', step {step}: "
              f"{len(tasks)} (pair, overlap) units")
        print(f"  predicted {predict_wall(tasks) * MODEL['speedup'] / 3600:.0f} "
              f"single-core CPU-h on this machine")
        return

    pf1, pf2, frame = indices[args.index]
    todo = [(pf1, pf2, ov) for ov in overlaps_for(pf1, pf2, step, args.geometry)
            if frame_of(ov, args.geometry) == frame]

    outdir = cluster_dir(args.geometry, tag)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"row_{args.index:05d}.csv")
    done = {ov for _, _, ov in load_done(path)} if os.path.exists(path) else set()

    print(f"[{args.index}] {args.geometry}  {pf1} x {pf2}  frame {frame}  "
          f"{len(todo)} overlaps, {len(done)} already done", flush=True)

    use_geometry(args.geometry)
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not done:
            writer.writerow(CSV_HEADER)
            fh.flush()
        for pf1_, pf2_, ov in todo:
            if ov in done:
                continue
            row = _run_task((pf1_, pf2_, ov, task_seed(pf1_, pf2_, ov, args.seed)))
            writer.writerow(row)
            fh.flush()
            print(f"  {ov} nt: within={bool(row[3])} z={row[4]:+.3f} "
                  f"{row[5] / 60:.1f} min", flush=True)

    open(os.path.join(outdir, f"row_{args.index:05d}.done"), "w").close()


def cmd_collect(args):
    """Pool the per-index csvs into the single csv that `plot` reads."""
    families, step, tag = resolve_set(args)
    indices = cluster_indices(families, args.geometry)
    outdir = cluster_dir(args.geometry, tag)
    path = csv_path(args.geometry, tag)

    rows = []
    for i in range(len(indices)):
        f = os.path.join(outdir, f"row_{i:05d}.csv")
        if not os.path.exists(f):
            continue
        with open(f, newline="") as fh:
            rows.extend(list(csv.DictReader(fh)))

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for r in rows:
            writer.writerow([r[c] for c in CSV_HEADER])
    print(f"{len(rows)} rows from {outdir}/ -> {path}")

    missing = [i for i in range(len(indices))
               if not os.path.exists(os.path.join(outdir, f"row_{i:05d}.done"))]
    if missing:
        out, i = [], 0
        while i < len(missing):
            j = i
            while j + 1 < len(missing) and missing[j + 1] == missing[j] + 1:
                j += 1
            out.append(str(missing[i]) if i == j else f"{missing[i]}-{missing[j]}")
            i = j + 1
        print(f"{len(missing)} of {len(indices)} indices incomplete")
        print(f"  resubmit with --array={','.join(out)}")
    else:
        print(f"All {len(indices)} indices complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="self-tests")
    c.add_argument("--seed", type=int, default=20260810)
    c.add_argument("--no-rerun", action="store_true",
                   help="skip the convergent re-run (which needs the bmDCA files)")
    c.set_defaults(func=cmd_check)

    for name, fn, helptext in [("run", cmd_run, "run a scan"),
                               ("cluster", cmd_cluster, "run one slurm array index"),
                               ("collect", cmd_collect, "pool the per-index csvs"),
                               ("plot", cmd_plot, "plot results from the csv")]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--geometry", choices=GEOMETRIES, default="diverging")
        p.add_argument("--preset", choices=sorted(PRESETS), default="quick")
        p.add_argument("--families", nargs="+", help="override the preset family list")
        p.add_argument("--step", type=int, default=None, help="overlap step in nt")
        p.add_argument("--tag", default=None, help="output filename tag")
        if name == "run":
            p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
            p.add_argument("--seed", type=int, default=20260810)
            p.add_argument("--dry-run", action="store_true",
                           help="print the plan and predicted time, run nothing")
        elif name == "cluster":
            p.add_argument("--index", type=int, default=None,
                           help="array index; with none, print the task count")
            p.add_argument("--seed", type=int, default=20260810)
        elif name == "collect":
            pass
        else:
            p.add_argument("--formats", nargs="+", default=["svg"],
                           metavar="EXT",
                           help="output formats, e.g. --formats svg png")
            p.add_argument("--smooth", type=int, default=5,
                           help="rolling window in sampled points (default 5)")
            p.add_argument("--no-compare", dest="compare", action="store_false",
                           help="do not overlay the convergent run")
            p.add_argument("--combine-frames", action="store_true",
                           help="all frames on one axes, coloured, instead of "
                                "one panel each")
            p.add_argument("--force-compare", action="store_true",
                           help="overlay convergent even when its panels mean "
                                "a different codon-position relation")
        p.set_defaults(func=fn)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
