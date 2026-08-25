#!/bin/bash
# Diverging overlaps, all 136 pairs, step 10, under the codon-degeneracy correction.
# ONE CPU per array index; one index = one (pair, frame) = one csv of rows.
#
# --array must match the index count.  Get it from:
#     python 20260821_strand_scan.py cluster --geometry diverging --preset full --step 10 --tag step10
# 136 pairs x 3 frames = 408 indices -> --array=0-407
#
#SBATCH -p BIOP
#SBATCH --time 24:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-407
#SBATCH -J re_diverging

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

eval "$(/home/xucapusa/miniconda3/bin/conda shell.bash hook)"
conda activate jupy

export BMDCA_DIR="${BMDCA_DIR:-/home/xucapusa/0 bmDCA}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

python -u 20260821_strand_scan.py cluster \
    --geometry diverging --preset full --step 10 --tag step10
