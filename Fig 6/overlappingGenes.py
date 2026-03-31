"""

Functions to implement overlap genes using bmDCA models.

- Nicole Wood, Kabir Husain, Orson Kirsch, with assistance from Claude Code (Anthropic)

"""

######## Import Functions

# Usual suspects
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit, prange, njit
import time  # Add import for timing

colorTable = {}
colorTable['k'] = [0,0,0]
colorTable['g'] = [27/255,158/255,119/255]
colorTable['o'] = [217/255,95/255,2/255]

######## Data input, and converting between types

#define stop codons
amber = ['T','A','G']
amber_rc = ['C','T','A']
ochre = ['T','A','A']
ochre_rc = ['T','T','A']
opal = ['T','G','A']
opal_rc = ['T','C','A']
stops = [amber, ochre, opal]
stops_rc = [amber_rc, ochre_rc, opal_rc]

def load_natural_energies(filename):
    """
    Load the natural energies for a given protein family from a file.
    File format is expected to be a text file with one energy value per line.
    
    INPUTS:
    pfname: Protein family name (string)
    
    OUTPUTS:
    energies: List of natural energies (float)
    """
    
    with open(filename, "r") as f:
        energies = [float(line.strip()) for line in f]
    return energies

def extract_params(params_data):
    """"
    Formats the parameter data from bmDCA
    
    INPUTS: numerical_data = numerical data
            params_data = parameter data
    
    OUTPUTS: 
    J_data = J values (last column only)
    h_data = h values (last column only)
    """
    # Read file once and separate J and h parameters, extracting only the values
    J_lines = []
    h_lines = []
    j_section = True
    
    with open(params_data, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:  # Skip empty lines
                continue
                
            if j_section and len(parts) == 6:
                J_lines.append(float(parts[5]))  # Only extract the value
            elif len(parts) == 4:
                j_section = False
                h_lines.append(float(parts[3]))  # Only extract the value
            elif not j_section and len(parts) >= 3:
                h_lines.append(float(parts[3]))  # Only extract the value
    
    # Convert directly to numpy arrays without intermediate steps
    J_final = np.array(J_lines)
    h_final = np.array(h_lines)
    
    return J_final, h_final

def aa_to_n(aa_sequence):
    """
    Takes input numerical amino acid sequence and outputs it 
    into its nucleotide constituents
    """
    amino_acid = [('-','-','-'),('G','C','T'),('T','G','T'),('G','A','T'),('G','A','A'),('T','T','T'),('G','G','T'),('C','A','T'),('A','T','T'),('A','A','A'),('T','T','G'),('A','T','G'),('A','A','T'),('C','C','T'),('C','A','A'),('C','G','T'),('T','C','T'),('A','C','T'),('G','T','T'),('T','G','G'),('T','A','T')]
    n_seq = []
    for i in aa_sequence:
        n_seq.extend(amino_acid[i])
    return n_seq
    #sequence_string = "".join(n_seq)
    #return Seq(sequence_string)

def to_numeric(n_sequence):
    """
    Takes input amino acid sequence and turns it into a numeric sequence,
    n_sequence must be an array, and have no stop codons in it.
    """

    numerical_sequence = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

    return np.array([numerical_sequence[aa] for aa in n_sequence])

#NEW: Numba char-to-int mapper
@njit
def aa_char_to_int(aa_char):
    """
    char-to-int converter for AA codes.
    """
    if aa_char == '-': return 0
    if aa_char == 'A': return 1
    if aa_char == 'C': return 2
    if aa_char == 'D': return 3
    if aa_char == 'E': return 4
    if aa_char == 'F': return 5
    if aa_char == 'G': return 6
    if aa_char == 'H': return 7
    if aa_char == 'I': return 8
    if aa_char == 'K': return 9
    if aa_char == 'L': return 10
    if aa_char == 'M': return 11
    if aa_char == 'N': return 12
    if aa_char == 'P': return 13
    if aa_char == 'Q': return 14
    if aa_char == 'R': return 15
    if aa_char == 'S': return 16
    if aa_char == 'T': return 17
    if aa_char == 'V': return 18
    if aa_char == 'W': return 19
    if aa_char == 'Y': return 20
    if aa_char == '*': return 21 # STOP codon
    return 0 # Default/gap

@njit
def count_matches(seq1, seq2):
    """
    Counts number of matching positions between two integer arrays.
    """
    matches = 0
    for i in range(len(seq1)):
        if seq1[i] == seq2[i]:
            matches += 1
    return matches

@njit
def run_ffs_shoot(DCA_params_1, DCA_params_2, initial_seq_int, target_seq_int, 
                  target_lambda, fail_lambda, 
                  T1=1.0, T2=1.0, max_steps=100000):
    """
    Runs a single FFS shoot/trial.
    
    Arguments:
    - initial_seq_int: Starting sequence (int array)
    - target_seq_int: Target sequence (int array) - used for lambda calc
    - target_lambda: Success threshold (number of matches >= this)
    - fail_lambda: Failure threshold (number of matches < this)
    
    Returns:
    - code (int): 1 = Success, -1 = Failure, 0 = Timeout
    - final_seq (int array): The sequence at termination
    """
    # Unpack params
    Jvec1, hvec1 = DCA_params_1[0], DCA_params_1[1]
    Jvec2, hvec2 = DCA_params_2[0], DCA_params_2[1]

    seq = initial_seq_int.copy()
    sequence_L = len(seq)
    
    # Lengths
    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    len_aa_1 = len_seq_1_n // 3
    len_aa_2 = len_seq_2_n // 3
    
    # Buffers
    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32) 
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)

    # Initial States
    split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1, aa_seq_2, rc_buffer)
    
    # Initial Check (Optimization: Assume caller checked initial lambda?)
    # But let's check just in case
    current_lambda = count_matches(seq, target_seq_int)
    if current_lambda >= target_lambda:
        return 1, seq
    if current_lambda < fail_lambda:
        return -1, seq
        
    # Initial Energy (Optimized: we might not need E if only mutations matter? 
    # No, we need Delta E for Metropolis)
    # But we don't need full E, just Delta E. 
    # However, to maintain current state, it's easier to just track changes.
    # Wait, we usually don't need absolute E for updates, but we need current AA seqs.
    
    itera = 0
    while itera < max_steps:
        # 1. Mutate
        new_position = np.random.randint(0, sequence_L)
        old_nt = seq[new_position]
        idx = np.random.randint(0, 3)
        if idx >= old_nt: idx += 1
        new_nt = idx
        
        seq[new_position] = new_nt
        
        # 2. Translate
        split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # 3. Stop Codon Check
        stop_codon_error = False
        if aa_seq_1_new[len_aa_1 - 1] != 21 or aa_seq_2_new[len_aa_2 - 1] != 21:
            stop_codon_error = True
        else:
            for i in range(len_aa_1 - 1):
                if aa_seq_1_new[i] == 21: stop_codon_error = True; break
            if not stop_codon_error:
                for i in range(len_aa_2 - 1):
                    if aa_seq_2_new[i] == 21: stop_codon_error = True; break
        
        if stop_codon_error:
            # Revert
            seq[new_position] = old_nt
            itera += 1
            continue

        # 4. Energy Delta
        delta_H = 0.0 # Initialize?
        
        # We need calculate_Delta_Energy.
        # But we need to know WHERE changes are in AA.
        # Re-use logic from generator.
        
        delta_H_1 = 0.0
        aa_pos_1 = -1; new_aa_1 = -1
        for i in range(len_aa_1 - 1):
            if aa_seq_1[i] != aa_seq_1_new[i]:
                aa_pos_1 = i; new_aa_1 = aa_seq_1_new[i]; break
        if aa_pos_1 != -1:
            delta_H_1 = calculate_Delta_Energy(aa_seq_1, Jvec1, hvec1, aa_pos_1, new_aa_1)
            
        delta_H_2 = 0.0
        aa_pos_2 = -1; new_aa_2 = -1
        for i in range(len_aa_2 - 1):
            if aa_seq_2[i] != aa_seq_2_new[i]:
                aa_pos_2 = i; new_aa_2 = aa_seq_2_new[i]; break
        if aa_pos_2 != -1:
            delta_H_2 = calculate_Delta_Energy(aa_seq_2, Jvec2, hvec2, aa_pos_2, new_aa_2)
            
        delta_H = (delta_H_1 / T1) + (delta_H_2 / T2)
        
        # 5. Metropolis
        accept = False
        if delta_H <= 0: accept = True
        elif np.random.rand() < np.exp(-delta_H): accept = True
        
        if accept:
            # Update AAs
            for i in range(len_aa_1): aa_seq_1[i] = aa_seq_1_new[i]
            for i in range(len_aa_2): aa_seq_2[i] = aa_seq_2_new[i]
            # Seq is already updated.
            
            # CHECK LAMBDA
            current_lambda = count_matches(seq, target_seq_int)
            if current_lambda >= target_lambda:
                return 1, seq
            if current_lambda < fail_lambda:
                return -1, seq
        else:
            # Revert
            seq[new_position] = old_nt
            
        itera += 1
        
    return 0, seq # Timeout

#NEW: numba list-to-array converter
@njit
def to_numeric_int(n_sequence):
    """
    Takes a list of AA chars (from fast_translate_int) and returns a
    integer array for energy calculations.
    """
    l = len(n_sequence)
    arr = np.empty(l, dtype=np.int32)
    for i in range(l):
        arr[i] = aa_char_to_int(n_sequence[i])
    return arr

#NEW: helper function to find the *one* changed AA
@njit
def find_changed_aa(aa_seq_old, aa_seq_new):
    """
    Compares two AA sequences (lists of chars) and finds the *first*
    position that differs. Returns (position, old_aa_int, new_aa_int).
    Returns (-1, -1, -1) if no change.
    """
    L = len(aa_seq_old)
    for i in range(L):
        if aa_seq_old[i] != aa_seq_new[i]:
            # Found the change
            return i, aa_char_to_int(aa_seq_old[i]), aa_char_to_int(aa_seq_new[i])
    return -1, -1, -1 # No change found

# Precompute codon table as a dict
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

def fast_translate(seq):
    aa_seq = []
    for i in range(0, len(seq)-2, 3):
        codon = ''.join(seq[i:i+3])
        aa_seq.append(CODON_TABLE.get(codon, 'X'))  # 'X' for unknown
    return aa_seq

def split_sequence_and_to_aa(sequence, len_1, len_2):
    aa_sequence_1 = fast_translate(sequence[:len_1])
    rc_seq = sequence[-len_2:][::-1]  # reverse
    rc_seq = [complement_base(nt) for nt in rc_seq]
    aa_sequence_2 = fast_translate(rc_seq)
    return aa_sequence_1, aa_sequence_2

def complement_base(nt):
    return {'A':'T', 'T':'A', 'G':'C', 'C':'G'}[nt]

def fast_reverse_complement(seq):
    # seq: list or np.array of 'A','T','G','C'
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    # Use list comprehension for speed
    return [comp[nt] for nt in reversed(seq)]

def split_sequence_and_to_aa(sequence, len_1, len_2):
    """
    Fast version: avoids BioPython, uses fast_translate and fast_reverse_complement.
    sequence: list or np.array of nucleotides (characters)
    len_1: length in nucleotides for first protein
    len_2: length in nucleotides for second protein
    """
    aa_sequence_1 = fast_translate(sequence[:len_1])
    rc_seq = fast_reverse_complement(sequence[-len_2:])
    aa_sequence_2 = fast_translate(rc_seq)
    return aa_sequence_1, aa_sequence_2




######## Energy calculations

@jit(nopython = True)
def calculate_Energy(thisseq, Jvec, hvec):
    """
    Written by Kabir Husain and Nicole Wood

    Given an amino acid sequence converted to numerical form,
    computes its DCA energy
    """
    thisenergy = 0
    seqL = int(len(hvec)/21) # length of the sequence
    Jfirstterm = 0
    for i in np.arange(seqL):
        a = thisseq[i]

        # add the field energies
        hlookup = int(21*i + a)
        thisenergy += hvec[hlookup]

        for j in np.arange(i+1,seqL):
            b = thisseq[j]

            # add the coupling energies
            Jlookup = int(Jfirstterm + 21*21*(j-i-1) + 21*a + b)
            thisenergy += Jvec[Jlookup]

        Jfirstterm += (seqL-i-1)*21*21

    return -1*thisenergy

# --- NEW: O(L) Delta Energy Calculation ---
@njit
def calculate_Delta_Energy(numeric_seq_old, Jvec, hvec, aa_pos, new_aa_int):
    """
    Calculates the *change* in energy from a single AA mutation (O(L)).
    This is the core optimization.
    """
    seqL = int(len(hvec) / 21)
    old_aa_int = numeric_seq_old[aa_pos]
    
    delta_E_terms = 0.0
    
    # 1. Field (h) term change
    delta_E_terms += hvec[21 * aa_pos + new_aa_int] #adds energy contribution from new AA
    delta_E_terms -= hvec[21 * aa_pos + old_aa_int] #subtracts energy contribution from old AA
    
    # 2. Coupling (J) term changes
    # We must iterate through the J matrix *exactly* as calculate_Energy does
    # to find all couplings involving aa_pos.
    Jfirstterm = 0
    for i in range(seqL):
        if i == aa_pos:
            # i *is* the changed position. Loop over all its partners j > i.
            a_old = old_aa_int
            a_new = new_aa_int
            for j in range(i + 1, seqL): # loops for the remaining positions
                b = numeric_seq_old[j]
                Jlookup_old = int(Jfirstterm + 21*21*(j-i-1) + 21*a_old + b)
                Jlookup_new = int(Jfirstterm + 21*21*(j-i-1) + 21*a_new + b)
                delta_E_terms += Jvec[Jlookup_new] - Jvec[Jlookup_old]
        
        elif i < aa_pos:
            # i is a partner *before* the changed position (j = aa_pos)
            j = aa_pos
            a = numeric_seq_old[i]
            b_old = old_aa_int
            b_new = new_aa_int
            Jlookup_old = int(Jfirstterm + 21*21*(j-i-1) + 21*a + b_old)
            Jlookup_new = int(Jfirstterm + 21*21*(j-i-1) + 21*a + b_new)
            delta_E_terms += Jvec[Jlookup_new] - Jvec[Jlookup_old]
        
        # else (i > aa_pos):
        # The changed position `aa_pos` would be the *first* index (i)
        # in the pair, which is handled by the `i == aa_pos` case.
        
        # Update Jfirstterm for the next i
        Jfirstterm += (seqL - i - 1) * 21 * 21
            
    # The energy is -1 * (sum of terms), so
    # E_new - E_old = -1*(terms_new) - (-1*(terms_old))
    #               = -1 * (terms_new - terms_old)
    #               = -1 * delta_E_terms
    return -1.0 * delta_E_terms

def calculate_energies(aa_sequence_1, aa_sequence_2, Jvec_1, hvec_1, Jvec_2, hvec_2):
    """
    Compute the energies of two sequences, and the total energy
    """
    energy_1 = calculate_Energy(to_numeric(aa_sequence_1[:-1]), Jvec_1, hvec_1)     #have to convert to numeric for the energy calculation
    energy_2 = calculate_Energy(to_numeric(aa_sequence_2[:-1]), Jvec_2, hvec_2)     #index to last one as sstop codon has no defined energy in bmDCA 
    energy_total = energy_1 + energy_2
    return energy_1, energy_2, energy_total


######## Sampling and initial conditions


#code to genetrate an initial sequence - cannot do any overlap less than 3 or overlap = 5 for some reason
def initial_seq_no_stops(prot1, prot2, overlap, quiet=False):
    """
    Function written by Kabir on 5/3/2025

    Generates a sequence that has 2 sequences,
    one of prot1 overlapped with reverse complement of prot2, 
    removes stop codons from the sequences and adds them to the end of each sequence. 

    Arguments:
    - prot1: Length of protein 1 in amino acids (without stop codon)
    - prot2: Length of protein 2 in amino acids (without stop codon)
    - overlap: Length of overlap region (in nucleotides). Must be >=6

    Returns:
    - seq: A string representing the nucleotide sequence of the two proteins

    Defintions:
    - Reading frame "3-0" is a perfect overlap
    - Reading frame "2-1" is of the type:
       AAABBB
        XXXYYY
    - Reading frame "1-2" is of the type:
       AAABBB
         YYYXXX

    """

    ## Utility functions
    def revcomp(seq):
        return seq.replace("A", "t").replace("T", "a").replace("C", "g").replace("G","c")[::-1].upper()

    def randnt():
        return rng.choice(nts)

    ### Utility variables
    nts = ["A", "T", "G", "C"]
    stopcodons = ["TAA", "TAG", "TGA"]
    rng = np.random.default_rng()

    codonsNoStop = ["".join([n1,n2,n3]) 
                    for n1 in nts 
                    for n2 in nts 
                    for n3 in nts 
                    if "".join([n1,n2,n3]) not in stopcodons]
    
    codonsNoStopEitherFrame = [a for a in codonsNoStop if revcomp(a) not in stopcodons]

    codonsNoStopNoA1 = ["".join([n1,n2,n3]) 
                    for n1 in ["T", "G", "C"] 
                    for n2 in nts 
                    for n3 in nts 
                    if "".join([n1,n2,n3]) not in stopcodons]
    
    diCodonsStop12 = [a + b for a in codonsNoStop for b in codonsNoStop 
                      if a not in stopcodons and b not in stopcodons and a[2] + b[:2] in ["CTA","TTA","TCA"]]

    ## The 1-2 frame requires some work
    noTAs = [codon for codon in codonsNoStop if codon[:2] != "TA"]
    noCAsOrTAs = [codon for codon in codonsNoStop if codon[:2] != "CA" and codon[:2] != "TA"]

    codonChain12 = {}
    for n1 in nts:
        for n2 in nts:
            for n3 in nts:
                thiscodon = n1 + n2 + n3
                if n3 == "T":
                    # cannot allow TA or CA in the first two 2nt
                    codonChain12[thiscodon] = noCAsOrTAs
                elif n3 == "C":
                    # cannot allow a TA in the first two 2nt
                    codonChain12[thiscodon] = noTAs
                else:
                    codonChain12[thiscodon] = codonsNoStop

    ### Generation code
    # Basic lengths -- in nucleotides, including stop codons
    l1 = 3*prot1 + 3
    l2 = 3*prot2 + 3
    seqL = l1 + l2 - overlap

    # Determine reading frame
    if (l1-overlap)%3 == 0:
        readingframe = "3-0"
    elif (l1-overlap - 1)%3 == 0:
        readingframe = "2-1"
    elif (l1-overlap - 2)%3 == 0:
        readingframe = "1-2"

    if not quiet:
        print("Reading frame: ", readingframe)

    # Step 1: fill in codons before the overlap
    beforeoverlap = int(np.floor((l1-overlap)/3))    # Codons before overlap

    seq = "".join(rng.choice(codonsNoStop, beforeoverlap))

    # Step 2: Pick the next two codons such that the frame opposite has a stop codon
    if readingframe == "3-0":
        next2 = revcomp(rng.choice(stopcodons)) + rng.choice(codonsNoStopEitherFrame)
    if readingframe == "2-1":
        next2 = randnt() + rng.choice(["TT", "CT", "TC"]) + "A" + randnt() + randnt()
    if readingframe == "1-2":
        next2 = randnt() + randnt() + rng.choice(["TT", "CT", "TC"]) + "A"
        if next2[3] == "T":
            next2 += rng.choice(["T","C"])
        else:
            next2 += randnt()

    seq += next2

    # Step 3: Fill out the bulk of the overlapping region
    lengthmidoverlap = int((l1 - 3*beforeoverlap - 12)/3) # not including the dicodons around each stop

    if readingframe == "3-0":
        seq += "".join(rng.choice(codonsNoStopEitherFrame, lengthmidoverlap))
        
    elif readingframe == "2-1":
        # one-by-one
        for codonsadded in np.arange(lengthmidoverlap):
            # Avoid a stop codon
            if seq[-2:] in ["TC", "CT", "TT"]:
                nextcodon = rng.choice(codonsNoStopNoA1)
            else:
                nextcodon = rng.choice(codonsNoStop)

            seq += nextcodon

    elif readingframe == "1-2":
        for codonsadded in np.arange(lengthmidoverlap):
            seq += rng.choice(codonChain12[seq[-3:]])


    # Step 4: Pick the last two codons -- the last one being a stop
    if readingframe == "3-0":
        nexttwocodons = rng.choice(codonsNoStopEitherFrame) + rng.choice(stopcodons)
    elif readingframe == "2-1":
        if seq[-2:] in ["TC", "CT", "TT"]:
            nexttwocodons = rng.choice(codonsNoStopNoA1) + rng.choice(stopcodons)
        else:
            nexttwocodons = rng.choice(codonsNoStop) + rng.choice(stopcodons)
    elif readingframe == "1-2":
        nexttwocodons = rng.choice(codonChain12[seq[-3:]])
        if nexttwocodons[-1] == "C" or nexttwocodons[-1] == "T":
            nexttwocodons += "TGA"
        else:
            nexttwocodons += rng.choice(stopcodons)

    seq += nexttwocodons

    # Step 5: If in a 2-1 or 1-2 frame, fill in nucleotides to complete a codon in frame 2
    if readingframe == "2-1":
        seq += randnt()
    elif readingframe == "1-2":
        seq+= randnt() + randnt()

    # Step 6: generate random codons in remainder of frame 2
    remainingcodons = int( np.floor((l2-overlap)/3) )

    seq += revcomp("".join(rng.choice(codonsNoStop, remainingcodons)))

    return seq


# Integer encoding for nucleotides
NUC_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INT_TO_NUC = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

# Build integer codon table for fast lookup
CODON_TABLE_INT = np.full((4, 4, 4), ord('X'), dtype=np.uint8)
CODON_TABLE_NUMERIC = np.full((4, 4, 4), 0, dtype=np.uint8) # Default to 0 ('-')

for codon, aa in CODON_TABLE.items():
    i, j, k = NUC_TO_INT[codon[0]], NUC_TO_INT[codon[1]], NUC_TO_INT[codon[2]]
    CODON_TABLE_INT[i, j, k] = ord(aa)
    # Convert char to our numeric int
    val = 0
    if aa == '-': val = 0
    elif aa == 'A': val = 1
    elif aa == 'C': val = 2
    elif aa == 'D': val = 3
    elif aa == 'E': val = 4
    elif aa == 'F': val = 5
    elif aa == 'G': val = 6
    elif aa == 'H': val = 7
    elif aa == 'I': val = 8
    elif aa == 'K': val = 9
    elif aa == 'L': val = 10
    elif aa == 'M': val = 11
    elif aa == 'N': val = 12
    elif aa == 'P': val = 13
    elif aa == 'Q': val = 14
    elif aa == 'R': val = 15
    elif aa == 'S': val = 16
    elif aa == 'T': val = 17
    elif aa == 'V': val = 18
    elif aa == 'W': val = 19
    elif aa == 'Y': val = 20
    elif aa == '*': val = 21
    CODON_TABLE_NUMERIC[i, j, k] = val


# ---------------------------------------------------------------------------
# Shuffled genetic codes (for synthetic-code controls)
# ---------------------------------------------------------------------------

# Mapping from amino-acid character to numeric index (matches the loop above)
_AA_CHAR_TO_INT = {
    '-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14,
    'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '*': 21,
}
_INT_TO_AA_CHAR = {v: k for k, v in _AA_CHAR_TO_INT.items()}


def make_shuffled_genetic_code(seed=None):
    """Return (new_table_dict, new_table_numeric) with amino acids randomly
    reassigned to sense codons, preserving degeneracy counts.
    Stop codons (TAA, TAG, TGA -> '*') are unchanged."""
    from collections import Counter

    # Extract degeneracy counts from the standard genetic code
    sense_codons = [c for c, aa in CODON_TABLE.items() if aa != '*']
    sense_aas = [CODON_TABLE[c] for c in sense_codons]
    aa_counts = Counter(sense_aas)  # e.g. {'L': 6, 'S': 6, 'R': 6, ...}

    # Build a list of 61 AA indices, repeated by degeneracy count, sorted
    aa_index_list = []
    for aa_char in sorted(aa_counts.keys()):
        aa_index_list.extend([_AA_CHAR_TO_INT[aa_char]] * aa_counts[aa_char])

    # Shuffle the 61 sense codons
    rng = np.random.default_rng(seed)
    shuffled_codons = list(rng.permutation(sense_codons))

    # Build new tables
    new_table = {}
    new_numeric = np.full((4, 4, 4), 0, dtype=np.uint8)

    for codon, aa_int in zip(shuffled_codons, aa_index_list):
        aa_char = _INT_TO_AA_CHAR[aa_int]
        new_table[codon] = aa_char
        i, j, k = NUC_TO_INT[codon[0]], NUC_TO_INT[codon[1]], NUC_TO_INT[codon[2]]
        new_numeric[i, j, k] = aa_int

    # Add stop codons unchanged
    for stop_codon in ('TAA', 'TAG', 'TGA'):
        new_table[stop_codon] = '*'
        i, j, k = NUC_TO_INT[stop_codon[0]], NUC_TO_INT[stop_codon[1]], NUC_TO_INT[stop_codon[2]]
        new_numeric[i, j, k] = 21

    return new_table, new_numeric


def make_aa_permuted_genetic_code(seed=None):
    """Return (new_table_dict, new_table_numeric) with amino acid *labels*
    randomly permuted across the 20 sense AAs.  All codons that coded for
    a given AA in the standard code now code for the same new AA, so the
    synonymous codon group structure is preserved.  Stops unchanged."""
    from collections import defaultdict

    # Map each sense AA to its codons in the standard code
    aa_to_codons = defaultdict(list)
    for codon, aa in CODON_TABLE.items():
        if aa != '*':
            aa_to_codons[aa].append(codon)

    original_aas = sorted(aa_to_codons.keys())  # 20 AAs
    rng = np.random.default_rng(seed)
    permuted_aas = list(rng.permutation(original_aas))

    new_table = {}
    new_numeric = np.full((4, 4, 4), 0, dtype=np.uint8)

    for orig_aa, new_aa in zip(original_aas, permuted_aas):
        new_int = _AA_CHAR_TO_INT[new_aa]
        for codon in aa_to_codons[orig_aa]:
            new_table[codon] = new_aa
            i, j, k = NUC_TO_INT[codon[0]], NUC_TO_INT[codon[1]], NUC_TO_INT[codon[2]]
            new_numeric[i, j, k] = new_int

    # Add stop codons unchanged
    for stop_codon in ('TAA', 'TAG', 'TGA'):
        new_table[stop_codon] = '*'
        i, j, k = NUC_TO_INT[stop_codon[0]], NUC_TO_INT[stop_codon[1]], NUC_TO_INT[stop_codon[2]]
        new_numeric[i, j, k] = 21

    return new_table, new_numeric


def set_genetic_code(new_table_dict, new_table_numeric):
    """Swap the global genetic code tables in-place so all downstream code
    (including numba-compiled functions) sees the new mapping."""
    CODON_TABLE.clear()
    CODON_TABLE.update(new_table_dict)
    CODON_TABLE_NUMERIC[:] = new_table_numeric
    # Rebuild CODON_TABLE_INT to match
    for codon, aa in new_table_dict.items():
        i, j, k = NUC_TO_INT[codon[0]], NUC_TO_INT[codon[1]], NUC_TO_INT[codon[2]]
        CODON_TABLE_INT[i, j, k] = ord(aa)


@njit
def seq_str_to_int_array(seq):
    arr = np.empty(len(seq), dtype=np.uint8)
    for i in range(len(seq)):
        nt = seq[i]
        if nt == 'A':
            arr[i] = 0
        elif nt == 'C':
            arr[i] = 1
        elif nt == 'G':
            arr[i] = 2
        else:  # 'T'
            arr[i] = 3
    return arr

@njit
def int_array_to_seq_str(arr):
    out = []
    for i in range(len(arr)):
        nt = arr[i]
        if nt == 0:
            out.append('A')
        elif nt == 1:
            out.append('C')
        elif nt == 2:
            out.append('G')
        else:
            out.append('T')
    return ''.join(out)

@njit
def get_rc_seq_out(seq, out):
    # seq: np.array of ints (0,1,2,3)
    # out: pre-allocated array of same length
    comp = np.array([3, 2, 1, 0], dtype=np.uint8)  # A<->T, C<->G
    n = len(seq)
    for i in range(n):
        out[i] = comp[seq[n - 1 - i]]

@njit
def translate_numeric_out(seq, out):
    # seq: np.array of ints (0,1,2,3)
    # out: pre-allocated array of ints (length = len(seq)//3)
    n_codons = len(seq) // 3
    for i in range(n_codons):
        a, b, c = seq[3*i], seq[3*i+1], seq[3*i+2]
        out[i] = CODON_TABLE_NUMERIC[a, b, c]

@njit
def split_sequence_and_to_numeric_out(sequence, len_1_n, len_2_n, aa_out_1, aa_out_2, rc_buffer):
    #
    # sequence: np.array of ints
    # len_1_n: length in nucleotides for seq 1
    # len_2_n: length in nucleotides for seq 2
    # aa_out_1: pre-allocated output for AA 1
    # aa_out_2: pre-allocated output for AA 2
    # rc_buffer: pre-allocated buffer for RC sequence (length len_2_n)
    
    # 1. Translate Seq 1
    translate_numeric_out(sequence[:len_1_n], aa_out_1)
    
    # 2. Get RC of Seq 2 part
    # sequence[-len_2_n:]
    start_2 = len(sequence) - len_2_n
    get_rc_seq_out(sequence[start_2:], rc_buffer)
    
    # 3. Translate Seq 2
    translate_numeric_out(rc_buffer, aa_out_2)

# --- MODIFIED: Main simulation loop (Optimized) ---
@njit
def overlapped_sequence_generator_int(DCA_params_1, DCA_params_2, initialsequence, T1=1.0, T2=1.0, numberofiterations=100000, quiet=False, whentosave=0.1, nat_mean1=None, nat_mean2=None, std_mean1=None, std_mean2=None):
    # Unpack params
    Jvec1, hvec1 = DCA_params_1[0], DCA_params_1[1]
    Jvec2, hvec2 = DCA_params_2[0], DCA_params_2[1]

    # Convert initial sequence to int array if it's a string
    seq = seq_str_to_int_array(initialsequence)
    sequence_L = len(seq)
    
    # Lengths in nucleotides
    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    
    # Lengths in AA (including stop)
    len_aa_1 = len_seq_1_n // 3
    len_aa_2 = len_seq_2_n // 3

    accepted = 0.0
    prob_accepted = 0.0
    not_accepted = 0.0
    
    # Pre-allocate history arrays
    max_saves = int(100.0 / whentosave) + 10 # Buffer
    energy_history_seq_1 = np.empty(max_saves, dtype=np.float64)
    energy_history_seq_2 = np.empty(max_saves, dtype=np.float64)
    save_idx = 0

    # Pre-allocate working arrays
    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32) # Numeric AA
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)
    
    # Buffers for "new" sequences (to avoid allocation)
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)

    # Initial Translation
    split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1, aa_seq_2, rc_buffer)
    
    # Calculate Initial Energy (Full O(L^2))
    E1 = calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)
    E = E1 + E2

    energy_history_seq_1[save_idx] = E1
    energy_history_seq_2[save_idx] = E2
    save_idx += 1

    # Initialize "Best" tracking (Closest to Natural Mean)
    best_E1 = E1
    best_E2 = E2
    min_dist = 1e99 # Infinity
    best_save_idx = -1 # Index of best save
    best_seq = seq.copy() # Store best sequence
    
    if nat_mean1 is not None and nat_mean2 is not None:
        # Initial check
        dist = abs((E1 - nat_mean1) / std_mean1) + abs((E2 - nat_mean2) / std_mean2)
        min_dist = dist
        best_save_idx = save_idx - 1

    itera = 1
    nextmessage = 100 * whentosave 
    
    # --- Main Monte Carlo Loop ---
    while itera < numberofiterations:
        if 100 * (itera / numberofiterations) > nextmessage:
            nextmessage += 100 * whentosave
            if save_idx < max_saves:
                energy_history_seq_1[save_idx] = E1
                energy_history_seq_2[save_idx] = E2
                save_idx += 1

        # 1. Mutate (In-place)
        new_position = np.random.randint(0, sequence_L)
        old_nt = seq[new_position]
        idx = np.random.randint(0, 3)
        if idx >= old_nt:
            idx += 1
        new_nt = idx
        
        # Apply mutation
        seq[new_position] = new_nt
        
        # 2. Translate to "New" buffers
        split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # 3. Check for invalid stop codons
        stop_codon_error = False
        # Check last positions are stops (21)
        if aa_seq_1_new[len_aa_1 - 1] != 21 or aa_seq_2_new[len_aa_2 - 1] != 21:
            stop_codon_error = True
        else:
            # Check internal positions for stops
            for i in range(len_aa_1 - 1):
                if aa_seq_1_new[i] == 21: stop_codon_error = True; break
            if not stop_codon_error:
                for i in range(len_aa_2 - 1):
                    if aa_seq_2_new[i] == 21: stop_codon_error = True; break
        
        if stop_codon_error:
            not_accepted += 1
            # Revert mutation
            seq[new_position] = old_nt
            # itera += 1
            continue

        # 4. Find changed AAs and calculate Delta_E
        delta_H_1 = 0.0
        delta_H_2 = 0.0
        
        # Find change in Seq 1
        aa_pos_1 = -1
        new_aa_1 = -1
        
        for i in range(len_aa_1 - 1): # Ignore stop codon at end
            if aa_seq_1[i] != aa_seq_1_new[i]:
                aa_pos_1 = i
                new_aa_1 = aa_seq_1_new[i]
                break
        
        if aa_pos_1 != -1:
            delta_H_1 = calculate_Delta_Energy(aa_seq_1, Jvec1, hvec1, aa_pos_1, new_aa_1)

        # Frame 2
        aa_pos_2 = -1
        new_aa_2 = -1
        for i in range(len_aa_2 - 1):
            if aa_seq_2[i] != aa_seq_2_new[i]:
                aa_pos_2 = i
                new_aa_2 = aa_seq_2_new[i]
                break
        
        if aa_pos_2 != -1:
            delta_H_2 = calculate_Delta_Energy(aa_seq_2, Jvec2, hvec2, aa_pos_2, new_aa_2)

        # 5. Metropolis Step
        delta_H = (delta_H_1 / T1) + (delta_H_2 / T2)

        accept = False
        if delta_H <= 0:
            accept = True
        else:
            if np.random.rand() < np.exp(-delta_H):
                accept = True
        
        if accept:
            # Accept: Update State
            for i in range(len_aa_1): aa_seq_1[i] = aa_seq_1_new[i]
            for i in range(len_aa_2): aa_seq_2[i] = aa_seq_2_new[i]
            
            E1 += delta_H_1
            E2 += delta_H_2
            E = E1 + E2
            
            if delta_H <= 0:
                accepted += 1
            else:
                prob_accepted += 1
            
            # --- Track Best Energy (Closest to Natural) ---
            if nat_mean1 is not None and nat_mean2 is not None:
                current_dist = dist = abs((E1 - nat_mean1) / std_mean1) + abs((E2 - nat_mean2) / std_mean2)
                if current_dist < min_dist:
                    min_dist = current_dist
                    best_E1 = E1
                    best_E2 = E2
                    best_seq[:] = seq[:] # Copy current sequence to best
                    best_save_idx = save_idx - 1
        else:
            # Reject: Revert State
            seq[new_position] = old_nt
            not_accepted += 1

        itera += 1
        
        # Sanity Check every 1000 iterations
        if itera % 1000 == 0:
            E1_check = calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
            E2_check = calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)
            E_check = E1_check + E2_check
            
            if abs(E_check - E) > 1e-4:
                E1 = E1_check
                E2 = E2_check
                E = E_check

    finalenergies = np.array([E1, E2])
    acceptedornot = np.array([accepted, prob_accepted, not_accepted])
    
    # Return string sequence, and also return the BEST energies found
    best_energies = np.array([best_E1, best_E2])
    final_seq_str = int_array_to_seq_str(seq)
    best_seq_str = int_array_to_seq_str(best_seq)
    
    return final_seq_str, acceptedornot, energy_history_seq_1[:save_idx], energy_history_seq_2[:save_idx], finalenergies, best_energies, best_seq_str, best_save_idx

@njit
def overlapped_sequence_generator_best(DCA_params_1, DCA_params_2, initialsequence, target_E1, target_E2, T1=1.0, T2=1.0, numberofiterations=100000, quiet=False, whentosave=0.1):
    """
    Same as overlapped_sequence_generator_int, but returns the sequence that was Closest
    to the target energies (Euclidean distance in E1-E2 space), not the final sequence.
    """
    # Unpack params
    Jvec1, hvec1 = DCA_params_1[0], DCA_params_1[1]
    Jvec2, hvec2 = DCA_params_2[0], DCA_params_2[1]

    # Convert initial sequence to int array if it's a string
    seq = seq_str_to_int_array(initialsequence)
    sequence_L = len(seq)
    
    # Lengths in nucleotides
    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    
    # Lengths in AA (including stop)
    len_aa_1 = len_seq_1_n // 3
    len_aa_2 = len_seq_2_n // 3

    accepted = 0.0
    prob_accepted = 0.0
    not_accepted = 0.0
    
    # Pre-allocate history arrays
    max_saves = int(100.0 / whentosave) + 10 # Buffer
    energy_history_seq_1 = np.empty(max_saves, dtype=np.float64)
    energy_history_seq_2 = np.empty(max_saves, dtype=np.float64)
    save_idx = 0

    # Pre-allocate working arrays
    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32) # Numeric AA
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)
    
    # Buffers for "new" sequences (to avoid allocation)
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)

    # Best Sequence Tracking
    best_seq = seq.copy()
    min_dist_sq = 1e20 # Large number

    # Initial Translation
    split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1, aa_seq_2, rc_buffer)
    
    # Calculate Initial Energy (Full O(L^2))
    E1 = calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)
    E = E1 + E2

    # Initial Best Check
    dist_sq = (E1 - target_E1)**2 + (E2 - target_E2)**2
    if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq
        best_seq[:] = seq[:]

    energy_history_seq_1[save_idx] = E1
    energy_history_seq_2[save_idx] = E2
    save_idx += 1

    itera = 1
    nextmessage = 100 * whentosave 
    
    # --- Main Monte Carlo Loop ---
    while itera < numberofiterations:
        if 100 * (itera / numberofiterations) > nextmessage:
            nextmessage += 100 * whentosave
            if save_idx < max_saves:
                energy_history_seq_1[save_idx] = E1
                energy_history_seq_2[save_idx] = E2
                save_idx += 1

        # 1. Mutate (In-place)
        new_position = np.random.randint(0, sequence_L)
        old_nt = seq[new_position]
        idx = np.random.randint(0, 3)
        if idx >= old_nt:
            idx += 1
        new_nt = idx
        
        # Apply mutation
        seq[new_position] = new_nt
        
        # 2. Translate to "New" buffers
        split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # 3. Check for invalid stop codons
        stop_codon_error = False
        # Check last positions are stops (21)
        if aa_seq_1_new[len_aa_1 - 1] != 21 or aa_seq_2_new[len_aa_2 - 1] != 21:
            stop_codon_error = True
        else:
            # Check internal positions for stops
            for i in range(len_aa_1 - 1):
                if aa_seq_1_new[i] == 21: stop_codon_error = True; break
            if not stop_codon_error:
                for i in range(len_aa_2 - 1):
                    if aa_seq_2_new[i] == 21: stop_codon_error = True; break
        
        if stop_codon_error:
            not_accepted += 1
            # Revert mutation
            seq[new_position] = old_nt
            # itera += 1
            continue

        # 4. Find changed AAs and calculate Delta_E
        delta_H_1 = 0.0
        delta_H_2 = 0.0
        
        # Find change in Seq 1
        aa_pos_1 = -1
        new_aa_1 = -1
        
        for i in range(len_aa_1 - 1): # Ignore stop codon at end
            if aa_seq_1[i] != aa_seq_1_new[i]:
                aa_pos_1 = i
                new_aa_1 = aa_seq_1_new[i]
                break
        
        if aa_pos_1 != -1:
            delta_H_1 = calculate_Delta_Energy(aa_seq_1, Jvec1, hvec1, aa_pos_1, new_aa_1)

        # Frame 2
        aa_pos_2 = -1
        new_aa_2 = -1
        for i in range(len_aa_2 - 1):
            if aa_seq_2[i] != aa_seq_2_new[i]:
                aa_pos_2 = i
                new_aa_2 = aa_seq_2_new[i]
                break
        
        if aa_pos_2 != -1:
            delta_H_2 = calculate_Delta_Energy(aa_seq_2, Jvec2, hvec2, aa_pos_2, new_aa_2)

        # 5. Metropolis Step
        delta_H = (delta_H_1 / T1) + (delta_H_2 / T2)

        accept = False
        if delta_H <= 0:
            accept = True
        else:
            if np.random.rand() < np.exp(-delta_H):
                accept = True
        
        if accept:
            # Accept: Update State
            for i in range(len_aa_1): aa_seq_1[i] = aa_seq_1_new[i]
            for i in range(len_aa_2): aa_seq_2[i] = aa_seq_2_new[i]
            
            E1 += delta_H_1
            E2 += delta_H_2
            E = E1 + E2
            
            # Update Best Sequence if this accepted state is better
            dist_sq = (E1 - target_E1)**2 + (E2 - target_E2)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_seq[:] = seq[:]
            
            if delta_H <= 0:
                accepted += 1
            else:
                prob_accepted += 1
        else:
            # Reject: Revert State
            seq[new_position] = old_nt
            not_accepted += 1

        itera += 1
        
        # Sanity Check every 1000 iterations
        if itera % 1000 == 0:
            E1_check = calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
            E2_check = calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)
            E_check = E1_check + E2_check
            
            if abs(E_check - E) > 1e-4:
                E1 = E1_check
                E2 = E2_check
                E = E_check
                # Re-check best (just in case)
                dist_sq = (E1 - target_E1)**2 + (E2 - target_E2)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_seq[:] = seq[:]

    finalenergies = np.array([E1, E2])
    acceptedornot = np.array([accepted, prob_accepted, not_accepted])
    
    # Return BEST sequence string
    best_seq_str = int_array_to_seq_str(best_seq)
    
    return best_seq_str, acceptedornot, energy_history_seq_1[:save_idx], energy_history_seq_2[:save_idx], finalenergies

# --- NEW: Seeding helper ---
@njit
def set_seed(value):
    np.random.seed(value)

# --- NEW: Slow Generator for Verification ---
@njit
def overlapped_sequence_generator_slow(DCA_params_1, DCA_params_2, initialsequence, T1=1.0, T2=1.0, numberofiterations=100000, quiet=False, whentosave=0.1):
    # Unpack params
    Jvec1, hvec1 = DCA_params_1[0], DCA_params_1[1]
    Jvec2, hvec2 = DCA_params_2[0], DCA_params_2[1]

    # Convert initial sequence to int array if it's a string
    seq = seq_str_to_int_array(initialsequence)
    sequence_L = len(seq)
    
    # Lengths in nucleotides
    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    
    # Lengths in AA (including stop)
    len_aa_1 = len_seq_1_n // 3
    len_aa_2 = len_seq_2_n // 3

    accepted = 0.0
    prob_accepted = 0.0
    not_accepted = 0.0
    
    # Pre-allocate history arrays
    max_saves = int(100.0 / whentosave) + 10 # Buffer
    energy_history_seq_1 = np.empty(max_saves, dtype=np.float64)
    energy_history_seq_2 = np.empty(max_saves, dtype=np.float64)
    save_idx = 0

    # Pre-allocate working arrays
    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32) # Numeric AA
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)
    
    # Buffers for "new" sequences (to avoid allocation)
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)

    # Initial Translation
    split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1, aa_seq_2, rc_buffer)
    
    # Calculate Initial Energy (Full O(L^2))
    E1 = calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)
    E = E1 + E2

    energy_history_seq_1[save_idx] = E1
    energy_history_seq_2[save_idx] = E2
    save_idx += 1

    itera = 1
    nextmessage = 100 * whentosave 
    
    # --- Main Monte Carlo Loop ---
    while itera < numberofiterations:
        if 100 * (itera / numberofiterations) > nextmessage:
            nextmessage += 100 * whentosave
            if save_idx < max_saves:
                energy_history_seq_1[save_idx] = E1
                energy_history_seq_2[save_idx] = E2
                save_idx += 1

        # 1. Mutate (In-place)
        new_position = np.random.randint(0, sequence_L)
        old_nt = seq[new_position]
        idx = np.random.randint(0, 3)
        if idx >= old_nt:
            idx += 1
        new_nt = idx
        
        # Apply mutation
        seq[new_position] = new_nt
        
        # 2. Translate to "New" buffers
        split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # 3. Check for invalid stop codons
        stop_codon_error = False
        # Check last positions are stops (21)
        if aa_seq_1_new[len_aa_1 - 1] != 21 or aa_seq_2_new[len_aa_2 - 1] != 21:
            stop_codon_error = True
        else:
            # Check internal positions for stops
            for i in range(len_aa_1 - 1):
                if aa_seq_1_new[i] == 21: stop_codon_error = True; break
            if not stop_codon_error:
                for i in range(len_aa_2 - 1):
                    if aa_seq_2_new[i] == 21: stop_codon_error = True; break
        
        if stop_codon_error:
            not_accepted += 1
            # Revert mutation
            seq[new_position] = old_nt
            # itera += 1
            continue

        # 4. Calculate Delta_E using FULL ENERGY CALCULATION (Slow)
        E1_new = calculate_Energy(aa_seq_1_new[:-1], Jvec1, hvec1)
        E2_new = calculate_Energy(aa_seq_2_new[:-1], Jvec2, hvec2)
        
        delta_H_1 = E1_new - E1
        delta_H_2 = E2_new - E2

        # 5. Metropolis Step
        delta_H = (delta_H_1 / T1) + (delta_H_2 / T2)

        accept = False
        if delta_H <= 0:
            accept = True
        else:
            if np.random.rand() < np.exp(-delta_H):
                accept = True
        
        if accept:
            # Accept: Update State
            for i in range(len_aa_1): aa_seq_1[i] = aa_seq_1_new[i]
            for i in range(len_aa_2): aa_seq_2[i] = aa_seq_2_new[i]
            
            # Update Energies directly
            E1 = E1_new
            E2 = E2_new
            E = E1 + E2
            
            if delta_H <= 0:
                accepted += 1
            else:
                prob_accepted += 1
        else:
            # Reject: Revert State
            seq[new_position] = old_nt
            not_accepted += 1

        itera += 1
        
    finalenergies = np.array([E1, E2])
    acceptedornot = np.array([accepted, prob_accepted, not_accepted])
    
    # Return string sequence
    final_seq_str = int_array_to_seq_str(seq)
    
    return final_seq_str, acceptedornot, energy_history_seq_1[:save_idx], energy_history_seq_2[:save_idx], finalenergies

# --- NEW: Convergence Generator ---
@njit
def overlapped_sequence_generator_convergence(DCA_params_1, DCA_params_2, initialsequence, mean_e1, std_e1, mean_e2, std_e2, max_iterations=10000000, T1=1.0, T2=1.0):
    # Unpack params
    Jvec1, hvec1 = DCA_params_1[0], DCA_params_1[1]
    Jvec2, hvec2 = DCA_params_2[0], DCA_params_2[1]

    # Convert initial sequence to int array if it's a string
    seq = seq_str_to_int_array(initialsequence)
    sequence_L = len(seq)
    
    # Lengths in nucleotides
    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    
    # Lengths in AA (including stop)
    len_aa_1 = len_seq_1_n // 3
    len_aa_2 = len_seq_2_n // 3

    # Pre-allocate working arrays
    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32) 
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)
    
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)

    # Initial Translation
    split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1, aa_seq_2, rc_buffer)
    
    # Calculate Initial Energy
    E1 = calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)
    E = E1 + E2

    itera = 0
    converged = False
    
    # Check if within 1 SD of mean
    if (mean_e1 - std_e1 <= E1 <= mean_e1 + std_e1) and (mean_e2 - std_e2 <= E2 <= mean_e2 + std_e2):
        return itera, True, E1, E2

    while itera < max_iterations:
        # 1. Mutate
        new_position = np.random.randint(0, sequence_L)
        old_nt = seq[new_position]
        idx = np.random.randint(0, 3)
        if idx >= old_nt:
            idx += 1
        new_nt = idx
        
        seq[new_position] = new_nt
        
        # 2. Translate
        split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # 3. Check Stops
        stop_codon_error = False
        if aa_seq_1_new[len_aa_1 - 1] != 21 or aa_seq_2_new[len_aa_2 - 1] != 21:
            stop_codon_error = True
        else:
            for i in range(len_aa_1 - 1):
                if aa_seq_1_new[i] == 21: stop_codon_error = True; break
            if not stop_codon_error:
                for i in range(len_aa_2 - 1):
                    if aa_seq_2_new[i] == 21: stop_codon_error = True; break
        
        if stop_codon_error:
            seq[new_position] = old_nt
            itera += 1
            continue

        # 4. Delta E
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
            delta_H_1 = calculate_Delta_Energy(aa_seq_1, Jvec1, hvec1, aa_pos_1, new_aa_1)

        aa_pos_2 = -1
        new_aa_2 = -1
        for i in range(len_aa_2 - 1):
            if aa_seq_2[i] != aa_seq_2_new[i]:
                aa_pos_2 = i
                new_aa_2 = aa_seq_2_new[i]
                break
        if aa_pos_2 != -1:
            delta_H_2 = calculate_Delta_Energy(aa_seq_2, Jvec2, hvec2, aa_pos_2, new_aa_2)

        # 5. Metropolis
        delta_H = (delta_H_1 / T1) + (delta_H_2 / T2)

        accept = False
        if delta_H <= 0:
            accept = True
        else:
            if np.random.rand() < np.exp(-delta_H):
                accept = True
        
        if accept:
            for i in range(len_aa_1): aa_seq_1[i] = aa_seq_1_new[i]
            for i in range(len_aa_2): aa_seq_2[i] = aa_seq_2_new[i]
            E1 += delta_H_1
            E2 += delta_H_2
            E = E1 + E2
        else:
            seq[new_position] = old_nt

        itera += 1
        
        # Check if within 1 SD of mean
        if (mean_e1 - std_e1 <= E1 <= mean_e1 + std_e1) and (mean_e2 - std_e2 <= E2 <= mean_e2 + std_e2):
            converged = True
            break

    return itera, converged, E1, E2


# --- NEW: Selective Convergence Generator ---
@njit
def overlapped_sequence_generator_selective(DCA_params_1, DCA_params_2, initialsequence, mean_e1, std_e1, mean_e2, std_e2, max_iterations=10000000, T1=1.0, T2=1.0, check_1=True, check_2=True):
    # Unpack params
    Jvec1, hvec1 = DCA_params_1[0], DCA_params_1[1]
    Jvec2, hvec2 = DCA_params_2[0], DCA_params_2[1]

    # Convert initial sequence to int array if it's a string
    seq = seq_str_to_int_array(initialsequence)
    sequence_L = len(seq)
    
    # Lengths in nucleotides
    len_seq_1_n = int(3 * len(hvec1) / 21 + 3)
    len_seq_2_n = int(3 * len(hvec2) / 21 + 3)
    
    # Lengths in AA (including stop)
    len_aa_1 = len_seq_1_n // 3
    len_aa_2 = len_seq_2_n // 3

    # Pre-allocate working arrays
    aa_seq_1 = np.empty(len_aa_1, dtype=np.int32) 
    aa_seq_2 = np.empty(len_aa_2, dtype=np.int32)
    rc_buffer = np.empty(len_seq_2_n, dtype=np.uint8)
    
    aa_seq_1_new = np.empty(len_aa_1, dtype=np.int32)
    aa_seq_2_new = np.empty(len_aa_2, dtype=np.int32)

    # Initial Translation
    split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1, aa_seq_2, rc_buffer)
    
    # Calculate Initial Energy
    E1 = calculate_Energy(aa_seq_1[:-1], Jvec1, hvec1)
    E2 = calculate_Energy(aa_seq_2[:-1], Jvec2, hvec2)
    E = E1 + E2

    itera = 0
    converged = False
    
    # Helper to check convergence
    # We define convergence as being within 1 SD of the mean natural energy
    # Note: Logic inside loop to allow checking per iteration

    while itera < max_iterations:
        # Check convergence at start of loop (or end, doesn't matter much)
        is_conv_1 = True
        if check_1:
            if not (mean_e1 - std_e1 <= E1 <= mean_e1 + std_e1):
                is_conv_1 = False
        
        is_conv_2 = True
        if check_2:
            if not (mean_e2 - std_e2 <= E2 <= mean_e2 + std_e2):
                is_conv_2 = False
        
        if is_conv_1 and is_conv_2:
            converged = True
            break

        # 1. Mutate
        new_position = np.random.randint(0, sequence_L)
        old_nt = seq[new_position]
        idx = np.random.randint(0, 3)
        if idx >= old_nt:
            idx += 1
        new_nt = idx
        
        seq[new_position] = new_nt
        
        # 2. Translate
        split_sequence_and_to_numeric_out(seq, len_seq_1_n, len_seq_2_n, aa_seq_1_new, aa_seq_2_new, rc_buffer)

        # 3. Check Stops
        stop_codon_error = False
        # Relaxed check: Only check for internal stops (premature truncation)
        # We generally assume the last codon is intended to be a stop or is ignored by energy calc ([:-1])
        
        for i in range(len_aa_1 - 1):
            if aa_seq_1_new[i] == 21: stop_codon_error = True; break
        
        if not stop_codon_error:
            for i in range(len_aa_2 - 1):
                if aa_seq_2_new[i] == 21: stop_codon_error = True; break
        
        if stop_codon_error:
            seq[new_position] = old_nt
            itera += 1
            continue

        # 4. Delta E
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
            delta_H_1 = calculate_Delta_Energy(aa_seq_1, Jvec1, hvec1, aa_pos_1, new_aa_1)

        aa_pos_2 = -1
        new_aa_2 = -1
        for i in range(len_aa_2 - 1):
            if aa_seq_2[i] != aa_seq_2_new[i]:
                aa_pos_2 = i
                new_aa_2 = aa_seq_2_new[i]
                break
        if aa_pos_2 != -1:
            delta_H_2 = calculate_Delta_Energy(aa_seq_2, Jvec2, hvec2, aa_pos_2, new_aa_2)

        # 5. Metropolis
        delta_H = (delta_H_1 / T1) + (delta_H_2 / T2)

        accept = False
        if delta_H <= 0:
            accept = True
        else:
            if np.random.rand() < np.exp(-delta_H):
                accept = True
        
        if accept:
            for i in range(len_aa_1): aa_seq_1[i] = aa_seq_1_new[i]
            for i in range(len_aa_2): aa_seq_2[i] = aa_seq_2_new[i]
            E1 += delta_H_1
            E2 += delta_H_2
            E = E1 + E2
        else:
            seq[new_position] = old_nt

        itera += 1
        
    final_seq_str = int_array_to_seq_str(seq)
    return itera, converged, E1, E2, final_seq_str

def main():

    overlapLen = 62

    #### Read in input data
    dcaparams1 = "PF00004/PF00004_params.dat"
    dcaparams2 = "PF00041/PF00041_params.dat"

    naturalenergies1_file = "PF00004/PF00004_naturalenergies.txt"
    naturalenergies2_file = "PF00041/PF00041_naturalenergies.txt"

    Js_1, hs_1 = extract_params(dcaparams1)
    Js_2, hs_2 = extract_params(dcaparams2)

    # Length in amino acids, not including stop codons
    lenprot1 = len(hs_1)/21  # 21 is the number of amino acids incl blank
    lenprot2 = len(hs_2)/21  # 21 is the number of amino acids incl blank

    print(f"Length of protein 1 w/out stop: {int(lenprot1)} amino acids or {int(3*lenprot1)} nucleotides")
    print(f"Length of protein 2 w/out stop: {int(lenprot2)} amino acids or {int(3*lenprot2)} nucleotides")
    print(f"Overlap length (nucleotides): {overlapLen}")

    # Combining parameters
    DCA_params_1 = [Js_1, hs_1]
    DCA_params_2 = [Js_2, hs_2]

    # The energies of natural sequences
    naturalenergies1 = load_natural_energies(naturalenergies1_file)
    naturalenergies2 = load_natural_energies(naturalenergies2_file)
    mean_1 = np.mean(naturalenergies1)
    mean_2 = np.mean(naturalenergies2)
    sd_1 = np.std(naturalenergies1)
    sd_2 = np.std(naturalenergies2)

    print("Data loaded")

    #### Overlap

    # Generate initial sequence
    initialCondition = initial_seq_no_stops(lenprot1, lenprot2, overlapLen)

    print("\n")

    # Time the sequence generator
    start_time = time.time()
    overlapOutput = overlapped_sequence_generator_int(DCA_params_1, DCA_params_2, 
                                                  initialCondition, numberofiterations=10000, 
                                                  T1=0.4, T2=0.5)
    end_time = time.time()

    # Print the execution time
    print(f"\n Overlapped sequence generation completed in {end_time - start_time:.2f} seconds.")

    seq = overlapOutput[0]
    acceptedornot = overlapOutput[1]
    E1_vec = overlapOutput[2]
    E2_vec = overlapOutput[3]
    hamming_vec = overlapOutput[4]

    # Print the final nucleotide sequence
    print("Final nucleotide sequence:")
    print("".join(seq))

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    axs[0].plot(E1_vec, "-k")
    axs[0].axhline(y=mean_1, linestyle='-', color=colorTable["o"])
    axs[0].axhline(y=mean_1 + sd_1, linestyle='--', color=colorTable["o"], label=r'$+\sigma$ seq 1')
    axs[0].axhline(y=mean_1 - sd_1, linestyle='--', color=colorTable["o"], label=r'$-\sigma$ seq 1')

    axs[1].plot(E2_vec, "-b")
    axs[1].axhline(y=mean_2, linestyle='-', color=colorTable["g"])
    axs[1].axhline(y=mean_2 + sd_2, linestyle='--', color=colorTable["g"], label=r'$+\sigma$ seq 2')
    axs[1].axhline(y=mean_2 - sd_2, linestyle='--', color=colorTable["g"], label=r'$-\sigma$ seq 2')
    plt.show()


def get_optimal_overlaps(min_len, max_len, step=3):
    """
    Generates a list of overlap lengths that result in the '3-0' (easiest)
    reading frame to reduce benchmarking noise.
    
    In overlappingGenes.py, l1 is a multiple of 3.
    The 3-0 frame occurs when (l1 - overlap) % 3 == 0.
    Therefore, overlap must be a multiple of 3.
    """
    # Create basic range
    overlaps = []
    
    # Ensure start is a multiple of 3
    current = min_len
    while current % 3 != 0:
        current += 1
        
    while current <= max_len:
        overlaps.append(current)
        # Ensure step keeps us on multiples of 3
        # If user passes step=5, we force it to next multiple of 3
        current += step
        while current % 3 != 0:
            current += 1
            
    return overlaps



if __name__ == "__main__":
    # Cprofile to see how long it takes to run
    # import cProfile
    # cProfile.run('main()')
    main()
