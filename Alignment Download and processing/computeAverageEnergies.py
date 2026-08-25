import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import palettable as pal
import time
import os

from overlappingGenes import extract_params, to_numeric, calculate_Energy

from Bio import SeqIO

pfname = "PF00004" 

pfnames = ["PF00009",
"PF00011",
"PF00013",
"PF00017",
"PF00018"]

for pfname in pfnames:
    paramfile = "/home/kabir/Work/UCL/Projects/2024 Overlapping Genes/0 bmDCA/" + pfname + "/" + pfname + "_params.dat"
    fastafile = "../Data/Trimmed Alignments/" + pfname + "-trimmed.fasta"

    numerical_sequence = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

    Jvec, hvec = extract_params(paramfile)

    energies = []
    rand_energies = []
    with open(fastafile) as handle:
        numseq = 0
        numseq_invalid = 0
        for record in SeqIO.parse(handle, "fasta"):
            # print(record.seq)
            numseq += 1
            # Check if any characters in the sequence are not in the numerical_sequence dictionary
            if any(char not in numerical_sequence for char in record.seq):
                # print(f"Invalid character found in sequence {numseq}: {record.id}")
                numseq_invalid += 1
                continue

            # Convert the sequence to numeric representation
            numeric_sequence = to_numeric(str(record.seq))
            energies.append(calculate_Energy(numeric_sequence, Jvec, hvec))

            randseq = np.random.choice(list(numerical_sequence.values()), size=len(numeric_sequence), replace=True)
            rand_energies.append(calculate_Energy(randseq, Jvec, hvec))

    plt.figure(figsize=(10, 6))
    plt.hist(energies, bins=30, color=pal.colorbrewer.qualitative.Set1_9.hex_colors[0], edgecolor='black', alpha=0.7)
    plt.hist(rand_energies, bins=30, color=pal.colorbrewer.qualitative.Set1_9.hex_colors[1], edgecolor='black', alpha=0.7)
    plt.title('Distribution of Energies (blue is random seqs, orange is real seqs)')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f"Figures/{pfname}_energy_distribution.png")
    plt.close()

    # Write energies to file
    with open(f"/home/kabir/Work/UCL/Projects/2024 Overlapping Genes/0 bmDCA/{pfname}/{pfname}_naturalenergies.txt", "w") as f:
        for energy in energies:
            f.write(f"{energy}\n")

    print(f"Finished processing {pfname}: {numseq} sequences with {numseq_invalid} invalid sequences.")