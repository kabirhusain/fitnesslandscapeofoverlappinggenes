#!/bin/bash -l
#$ -l h_rt=48:00:0
#$ -l gpu=1
#$ -N bmDCA
#$ -wd /home/ucapusa/20250526_eaDCA/

cd /home/ucapusa/20250526_eaDCA/

eval "$(/home/ucapusa/miniconda3/bin/conda shell.bash hook)"
conda activate dca

adabmDCA train -m bmDCA -d Trimmed\ Alignments/PF00009-trimmed.fasta -o bmDCA/PF00009 -l PF00009
adabmDCA train -m bmDCA -d Trimmed\ Alignments/PF00011-trimmed.fasta -o bmDCA/PF00011 -l PF00011
adabmDCA train -m bmDCA -d Trimmed\ Alignments/PF00013-trimmed.fasta -o bmDCA/PF00013 -l PF00013
adabmDCA train -m bmDCA -d Trimmed\ Alignments/PF00017-trimmed.fasta -o bmDCA/PF00017 -l PF00017
adabmDCA train -m bmDCA -d Trimmed\ Alignments/PF00018-trimmed.fasta -o bmDCA/PF00018 -l PF00018
