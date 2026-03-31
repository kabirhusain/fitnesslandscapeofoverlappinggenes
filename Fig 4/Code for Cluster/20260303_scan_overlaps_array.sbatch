#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 96:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=36
#SBATCH --array=0-135
#SBATCH -J re_scan

cd "/home/xucapusa/20260303 Replica Exchange"

eval "$(/home/xucapusa/miniconda3/bin/conda shell.bash hook)"
conda activate jupy

srun python 20260303_scan_overlaps_array.py
