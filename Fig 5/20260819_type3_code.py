"""
Type III genetic codes: the missing cell of the 2x2.

Fig 5 uses two randomisations, which sit at two corners of a 2x2 over the two
things the standard code could be getting right:

                        | aa<->degeneracy intact | aa<->degeneracy scrambled
    --------------------+------------------------+--------------------------
    wobble intact       | standard               | Type II (permuted)
    wobble destroyed    | Type I (shuffled)      | Type III  <- this file

`make_shuffled_genetic_code` (Type I) scatters the 61 sense codons at random but
hands each amino acid its own standard degeneracy back -- Leu still gets 6
codons, Trp still gets 1.  `make_aa_permuted_genetic_code` (Type II) does the
opposite: the synonymous blocks are left exactly where they are and only the 20
amino-acid labels move.

Type III does both: the codons are scattered as in Type I, AND the degeneracy
counts are dealt out to amino acids at random, so Trp may end up 6-fold and Leu
1-fold.  What it does NOT do is change the degeneracy *spectrum* -- the multiset
{6,6,6,4,4,4,4,4,3,2,...} is still the standard code's, so total coding capacity
is matched and every amino acid still has at least one codon.  That is what
makes this one cell of the 2x2 rather than a move along three axes at once;
drawing each codon's amino acid i.i.d. uniform would also flatten the spectrum
and would leave ~1 amino acid per code with no codon at all (4.4% per amino
acid), which is a different experiment.

Stop codons are TAA/TAG/TGA here exactly as in the standard code and in both of
Fig 5's randomisations, so nothing in any arm is a stop-codon effect.

Two independent RNG streams, so that Type III seed s scatters the codons into
precisely the sequence Type I seed s does -- only the count-to-amino-acid deal
differs.  (The synonymous blocks are still cut at different points, because the
counts arrive in a different order; the pairing is a tidiness measure, not an
exact matched control.)

Everything here is pure Python dict/array manipulation -- no numba is touched,
so importing this module cannot compile a kernel against the wrong code.  See
the one-process-per-code note in the run script.

    python 20260819_type3_code.py        # self-test: assert the 2x2 cell is what it claims

- Kabir Husain, with assistance from Claude Code (Anthropic)
"""

from collections import Counter

import numpy as np

import overlappingGenes as og

# The standard code, snapshotted AT IMPORT.  `set_genetic_code` clears and
# refills og.CODON_TABLE in place, so a generator that read the live global
# would build from whatever code happened to be installed.  Nothing here ever
# looks at og.CODON_TABLE again.
_STANDARD = dict(og.CODON_TABLE)


def make_scrambled_genetic_code(seed=None):
    """Return (new_table_dict, new_table_numeric) with the sense codons randomly
    scattered AND the degeneracy counts randomly reassigned among the 20 amino
    acids.  The degeneracy spectrum and the stop codons are the standard code's."""
    sense_codons = [c for c, aa in _STANDARD.items() if aa != '*']
    aa_counts = Counter(_STANDARD[c] for c in sense_codons)

    # Separate streams: the codon scatter matches Type I's for the same seed.
    rng_codon = np.random.default_rng(seed)
    rng_count = np.random.default_rng(None if seed is None else seed + 1_000_000)

    # Deal the standard degeneracy counts out to amino acids at random.  This is
    # the single line that separates Type III from Type I, whose equivalent is
    # `aa_index_list.extend([aa] * aa_counts[aa])` in standard-count order.
    aas = sorted(aa_counts.keys())
    counts = [int(n) for n in rng_count.permutation([aa_counts[a] for a in aas])]

    aa_index_list = []
    for aa_char, n in zip(aas, counts):
        aa_index_list.extend([og._AA_CHAR_TO_INT[aa_char]] * n)

    shuffled_codons = list(rng_codon.permutation(sense_codons))

    new_table = {}
    new_numeric = np.full((4, 4, 4), 0, dtype=np.uint8)

    for codon, aa_int in zip(shuffled_codons, aa_index_list):
        new_table[codon] = og._INT_TO_AA_CHAR[aa_int]
        i, j, k = og.NUC_TO_INT[codon[0]], og.NUC_TO_INT[codon[1]], og.NUC_TO_INT[codon[2]]
        new_numeric[i, j, k] = aa_int

    for stop_codon in ('TAA', 'TAG', 'TGA'):
        new_table[stop_codon] = '*'
        i, j, k = og.NUC_TO_INT[stop_codon[0]], og.NUC_TO_INT[stop_codon[1]], og.NUC_TO_INT[stop_codon[2]]
        new_numeric[i, j, k] = 21

    return new_table, new_numeric


# --- Self-test: does this code actually sit in the cell it claims? -----------

def _degeneracy(table):
    return dict(Counter(aa for aa in table.values() if aa != '*'))


def _wobble(table):
    """Fraction of codon pairs sharing positions 1 and 2 that are synonymous.
    Standard code 0.716; a code with the wobble destroyed sits near 0.05."""
    import itertools
    n_pairs = n_syn = 0
    for first in "ACGT":
        for second in "ACGT":
            box = [first + second + third for third in "ACGT"
                   if table.get(first + second + third, '*') != '*']
            for x, y in itertools.combinations(box, 2):
                n_pairs += 1
                n_syn += (table[x] == table[y])
    return n_syn / n_pairs


if __name__ == "__main__":
    from overlappingGenes import make_shuffled_genetic_code

    std_deg = _degeneracy(_STANDARD)
    std_wob = _wobble(_STANDARD)
    print(f"standard code: wobble {std_wob:.3f}, degeneracy spectrum "
          f"{sorted(std_deg.values(), reverse=True)}")

    n_map_differs = 0
    wobbles = []
    for seed in range(30):
        table, numeric = make_scrambled_genetic_code(seed=seed)

        assert len(table) == 64, f"seed {seed}: {len(table)} codons, expected 64"
        assert sum(1 for aa in table.values() if aa == '*') == 3, "stop count changed"
        assert all(table[c] == '*' for c in ('TAA', 'TAG', 'TGA')), "stops moved"

        deg = _degeneracy(table)
        assert len(deg) == 20, f"seed {seed}: {len(deg)} amino acids encoded, expected 20"
        assert sorted(deg.values()) == sorted(std_deg.values()), \
            f"seed {seed}: degeneracy spectrum changed"
        n_map_differs += (deg != std_deg)

        # the numeric table must agree with the dict on all 64 codons
        for codon, aa in table.items():
            i, j, k = (og.NUC_TO_INT[codon[0]], og.NUC_TO_INT[codon[1]],
                       og.NUC_TO_INT[codon[2]])
            expect = 21 if aa == '*' else og._AA_CHAR_TO_INT[aa]
            assert numeric[i, j, k] == expect, f"seed {seed}: {codon} dict/numeric disagree"

        wobbles.append(_wobble(table))

    print(f"Type III, 30 seeds: wobble {np.mean(wobbles):.3f} +/- {np.std(wobbles):.3f} "
          f"(standard {std_wob:.3f})  -- wobble destroyed")
    print(f"                    degeneracy spectrum preserved for 30/30 seeds")
    print(f"                    per-amino-acid degeneracy differs from standard "
          f"for {n_map_differs}/30 seeds  -- map scrambled")

    # The cell above it in the 2x2, for contrast.
    t1_deg = [_degeneracy(make_shuffled_genetic_code(seed=s)[0]) == std_deg for s in range(30)]
    print(f"\nType I, same 30 seeds: per-amino-acid degeneracy identical to standard "
          f"for {sum(t1_deg)}/30 seeds  -- map intact, which is the difference")
