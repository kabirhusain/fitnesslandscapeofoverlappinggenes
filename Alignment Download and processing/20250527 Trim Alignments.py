import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import palettable as pal
import time
import os
import gzip
from io import TextIOWrapper

sns.set_context("talk",rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":15})
sns.set_style("whitegrid",{"grid.color": '.9', 'grid.linestyle': '--','axes.edgecolor': '.6', 'xtick.bottom': True,'ytick.left': True})

colorTable = {}
colorTable['k'] = [0,0,0]
colorTable['g'] = [27/255,158/255,119/255]
colorTable['o'] = [217/255,95/255,2/255]


import Bio
from Bio import AlignIO

from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
from numba import njit, prange
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio import Align

def count_gaps_per_position(alignment, gapchars = [".", "-"]):
    """
    Take an alignment in array format

    Returns an array of number of gaps per position in the alignment
    """
    num_gaps = np.zeros(alignment.shape[1])
    for i in range(alignment.shape[1]):
      num_gaps[i] = np.sum([np.sum(alignment[:,i] == gapchar) for gapchar in gapchars])

    return num_gaps

def count_gaps_per_sequence(alignment, gapchars = [".", "-"]):
    """
    Take an alignment in array format

    Returns an array of number of gaps per sequence in the alignment
    """
    num_gaps = np.zeros(alignment.shape[0])    #go along sequence
    for i in range(alignment.shape[0]):
        num_gaps[i] = np.sum([np.sum(alignment[i,:] == gapchar) for gapchar in gapchars])

    return num_gaps

def remove_gapped_positions(alignment, gapthreshold = 0.2):
    """
    Function that gets rid of highly gapped positions given an alignment and data on gaps at each position

    Takes an alignment in np array format (first index is number of sequences, second index is number of residues)
        
    Returns:
    new_alignment - MSA but with highly gapped positions removed, 
    lost_indices - list of indices for positions which have been removed. 
    """

    # count number of spaces
    num_spaces = count_gaps_per_position(alignment)
    
    # Determine the number of columns to keep
    keep_columns = np.zeros(alignment.shape[1], dtype=np.bool_)    
    for i in range(len(num_spaces)):
      if num_spaces[i]/alignment.shape[0] <= gapthreshold:
          keep_columns[i] = True
    
    # Create a new array for the updated alignment
    #number of colums is number of kept columns, sums over only true values
    new_alignment = np.empty((alignment.shape[0], np.sum(keep_columns)), dtype=alignment.dtype)
    
    # Fill the new alignment array, update alignment according to old one if the column is the same as a keep one
    new_index = 0
    lost_indices = []
    for i in range(alignment.shape[1]):
      if keep_columns[i] == True:
          new_alignment[:, new_index] = alignment[:, i]
          new_index += 1
      if keep_columns[i] == False:
          lost_indices.append(i)

    return new_alignment, lost_indices

def remove_gapped_sequences(alignment, ID_array, gapthreshold = 0.2):
    """
    Function that gets rid of highly gapped sequences given an alignment and array of IDs
    
    Takes an alignment in np array format (first index is number of sequences, second index is number of residues), as well as a list
    of sequence IDs

    Returns:
    new_alignment - MSA but with highly gapped sequences removed, 
    newIDs - the ID's of each kept sequence. 
    """

    gaps_per_seq = count_gaps_per_sequence(alignment)
    
    # Determine sequences to keep
    keep_rows = np.zeros(alignment.shape[0], dtype=np.bool_)
    for i in range(len(gaps_per_seq)):
        if gaps_per_seq[i]/alignment.shape[1] <= gapthreshold:
            keep_rows[i] = True

    # Create a new array for the updated alignment
    new_alignment = np.empty((np.sum(keep_rows), alignment.shape[1]), dtype=alignment.dtype)
    new_ID_array = np.empty(np.sum(keep_rows), dtype = ID_array.dtype)
    new_index = 0
    for i in range(alignment.shape[0]):
        if keep_rows[i]:
          new_alignment[new_index,:] = alignment[i,:]
          new_ID_array[new_index] = ID_array[i]
          new_index += 1

    return new_alignment, new_ID_array

def plotGapStatistics(alignment, fnam, suffix):  
    """
    Plot statistics of gaps in an alignment (np array format)
    """
    gaps_pos = np.array(count_gaps_per_position(alignment))
    gaps_seq = np.array(count_gaps_per_sequence(alignment))
    
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(10,3)
    
    axs[0].hist(gaps_pos/alignment.shape[0], color = "k", bins = np.linspace(0,1,50))
    axs[0].set_xlabel("gaps per position")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("counts")
    
    axs[1].hist(gaps_seq/alignment.shape[1], color = "k", bins = np.linspace(0,1,50))
    axs[1].set_xlabel("gaps per sequence")
    axs[1].set_yscale("log")
    
    # print("After width trimming:")
    print("Number of positions is", alignment.shape[1])
    print("Number of sequences is" , alignment.shape[0])
    
    print()
    print("Average position has", np.mean(100*gaps_pos/alignment.shape[0]), "percent gaps")
    print("Average sequence has", np.mean(100*gaps_seq/alignment.shape[1]), "percent gaps")

    plt.savefig(f"../Data/Trimmed Alignments/Plots/{fnam}-{suffix}-gap_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()


def saveAlignment(alignment, ID_array, output_file):
    # Generate SeqRecord objects
    seq_records = [ SeqRecord(Seq("".join(list(seq))).upper(), id=idname, description=idname) 
                   for idname, seq in zip(ID_array,alignment) ]
    
    # Write to a FASTA file
    SeqIO.write(seq_records, output_file, "fasta")
    
    print(f"FASTA file '{output_file}' has been created.")

def inputAlignment(filename):
    """
    Looks for a Stockholm-format alignment file, extracts the alignment and returns:
    1. The alignment as a numpy array
    2. The list of sequence IDs
    """
    # Open the file, handling gzipped files if necessary
    if filename.endswith(".gz"):
        with gzip.open(filename, "rt") as file:
            alignment = AlignIO.read(file, "stockholm")
    else:
        with open(filename, "r") as file:
            alignment = AlignIO.read(file, "stockholm")
    
    length = alignment.get_alignment_length()
    
    ID_list = []
    for record in alignment:
        ID_list.append(record.id)
    ID_array = np.array(ID_list)
    
    arr_alig = np.array(alignment)

    return arr_alig, ID_array


def trimAlignment(filename, outputfile, pfam_id):
    print(f"Processing {pfam_id} alignment...")
    arr_alig, ID_array = inputAlignment(filename)
    
    # print("Original alignment:")
    plotGapStatistics(arr_alig, pfam_id, "original")
    
    width_trimmed_alignment, removed_positions = remove_gapped_positions(arr_alig)
    
    # print()
    # print("After trimming positions:")
    plotGapStatistics(width_trimmed_alignment, pfam_id, "trimmedPositions")
    
    final_trimmed_alignment, trimmed_IDs = remove_gapped_sequences(width_trimmed_alignment, ID_array)
    
    # print()
    # print("After trimming sequences:")
    plotGapStatistics(final_trimmed_alignment, pfam_id, "trimmedSequences")
    
    # print()
    saveAlignment(final_trimmed_alignment, trimmed_IDs, outputfile)
    print(f"Trimmed alignment saved to {outputfile}")
    print("\n")


# # Left out: "PF00005",
# pfam_ids = [
#     "PF00004", "PF00041", "PF00072", "PF00076", "PF00096",
#     "PF00153", "PF00271", "PF00397", "PF00512", "PF00595", "PF02518", "PF07679"
# ]

pfam_ids = ["PF00009", "PF00011", "PF00013", "PF00012", "PF00017", "PF00018"]

for pfam_id in pfam_ids:
    trimAlignment(f"../Data/20250527_Pfam_alignments/{pfam_id}.alignment.full.gz",
                  f"../Data/Trimmed Alignments/{pfam_id}-trimmed.fasta", pfam_id)