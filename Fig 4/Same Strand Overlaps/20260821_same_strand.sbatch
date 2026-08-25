#!/bin/bash
# Same-strand overlaps, all 272 ordered pairs, step 1, under the codon-degeneracy
# correction.  ONE CPU per array index; one index = one (ordered pair, frame).
#
# --array must match the index count.  Get it from:
#     python 20260821_strand_scan.py cluster --geometry same --preset full --tag full
# 272 ordered pairs x 2 frames = 544 indices -> --array=0-543
# (same strand has no frame 0: gene 1's stop is an in-frame stop for gene 2)
#
#SBATCH -p BIOP
#SBATCH --time 24:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-543
#SBATCH -J re_same

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

eval "$(/home/xucapusa/miniconda3/bin/conda shell.bash hook)"
conda activate jupy

export BMDCA_DIR="${BMDCA_DIR:-/home/xucapusa/0 bmDCA}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

python -u 20260821_strand_scan.py cluster --geometry same --preset full --tag full
