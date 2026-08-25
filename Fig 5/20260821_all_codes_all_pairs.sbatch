#!/bin/bash
# Fig 5 redone: Type I, II and III codes, all five pairs, three overlaps each,
# under the codon-degeneracy correction.
# ONE CPU per array index; one index = one (pair, code, overlap) = one csv row.
#
# --array must match the task count.  Get it from:
#     python 20260821_all_codes_all_pairs.py
# 5 pairs x 46 codes x 3 overlaps = 690 tasks -> --array=0-689
# SEED_FROM/SEED_TO change that count -- see 20260821_README.md.
# No %N throttle: the partition already caps concurrency by what is free, so a
# throttle can only stop the array expanding if slots open up overnight.
# The array is pair-major, so a truncated one drops whole pairs rather than
# leaving five half-finished:  sbatch --array=0-275  is the first two pairs.
#
# One cpu, not 36: the 36-cpu run of the Fig 4 scan
# (../../20260729 Replica Exchange Codon Corrected/From Cluster) parallelised
# across replicas inside each run and measured ~6% efficiency -- 847 CPU-h
# billed for ~49 CPU-h of work -- because every swap round is a barrier over 121
# replicas.  With n_workers=1 there is no pool, no barrier, no IPC, and the
# numba compile is paid once per index instead of once per overlap.
#
#SBATCH -p BIOP
#SBATCH --time 24:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-689
#SBATCH -J re_fig5

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p Data

eval "$(/home/xucapusa/miniconda3/bin/conda shell.bash hook)"
conda activate jupy

export BMDCA_DIR="${BMDCA_DIR:-/home/xucapusa/0 bmDCA}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

python -u 20260821_all_codes_all_pairs.py
