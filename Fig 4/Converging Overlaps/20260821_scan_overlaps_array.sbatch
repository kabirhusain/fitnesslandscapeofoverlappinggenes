#!/bin/bash
# Fig 4's overlap scan, all 136 pairs, step 1, under the codon-degeneracy correction.
# ONE CPU per array index; one index = one (pair, frame) = one csv of rows.
#
# --array must match the task count.  Get it from:
#     python 20260821_scan_overlaps_array.py
# 136 pairs x 3 frames = 408 tasks -> --array=0-407
#
# One cpu, not 36: the previous 36-cpu run of this scan
# (../../20260729 Replica Exchange Codon Corrected/From Cluster) measured ~6%
# efficiency -- 847 CPU-h billed for ~49 CPU-h of work -- because every swap
# round is a barrier over 121 replicas.
#
#SBATCH -p BIOP
#SBATCH --time 24:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-407
#SBATCH -J re_scan

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p Data

eval "$(/home/xucapusa/miniconda3/bin/conda shell.bash hook)"
conda activate jupy

export BMDCA_DIR="${BMDCA_DIR:-/home/xucapusa/0 bmDCA}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

python -u 20260821_scan_overlaps_array.py
