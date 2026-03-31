# Overlapping Genes Code Collection

Code and analysis for generating and studying overlapping genes using Potts models trained on multiple-sequence-alignments. This repository contains reproducible analysis scripts and notebooks for the associated manuscript, *The fitness landscape of overlapping genes*, by Orson Kirsch, Nicole Wood, Steven A. Redford, and Kabir Husain.

## Project Overview

This codebase implements computational methods to analyze and design overlapping genetic sequences using Direct Coupling Analysis (DCA) models, specifically training bmDCA models on protein family alignments. The project explores how genetic codes can be shuffled and permuted while maintaining fitness in overlapping gene configurations.

## Repository Structure

### Core Analysis Folders

- **Fig 2** - Analysis code for Figure 2 of the manuscript (constructing joint Potts models)
- **Fig 3** - Analysis code for Figure 3 of the manuscript (replica exchange Monte Carlo)
- **Fig 4** - Analysis code for Figure 4 of the manuscript (comparision of protein pairs)
- **Fig 5** - Analysis code for Figure 5 of the manuscript (shuffled codes and Pareto optimization)
- **Fig 6** - Analysis code for Figure 6 of the manuscript (mutational paths)

### Data Processing

- **Alignment Download and processing** - Scripts for downloading PFam protein family alignments and computing energy values for training DCA models
  - `download_pfam_alignments.py` - Download protein family alignments from InterPro/PFam
  - `20250527 Trim Alignments.py` - Trim and process alignments
  - `computeAverageEnergies.py` - Compute average energies from alignments

- The "bmDCA Training" subfolder has an example script for training a bmDCA model using adabmDCA 2.0.

### Pre-trained Models

- **0 bmDCA** - Contains pre-trained bmDCA models (required for code execution)
  - These models must be placed in this folder for the analysis scripts to run

## Setup & Requirements

### Prerequisites
- Python 3.x
- NumPy
- Matplotlib
- Seaborn
- Jupyter (for notebook analysis)
- Numba (for compiled functions)

### Installing bmDCA Models

The analysis scripts require trained bmDCA models to be present in the `0 bmDCA/` folder. These models should be placed there before running any analysis code.

To generate models from scratch:
1. Run scripts in `Alignment Download and processing/` to download and process PFam alignments
2. Train bmDCA models using your preferred DCA implementation (code assumes adabmDCA 2.0 output, but could be adapted to others)
3. Place trained models in the `0 bmDCA/` folder with appropriate naming convention
