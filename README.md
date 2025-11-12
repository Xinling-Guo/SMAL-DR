# SMAL-DR: Structural fold mining-and deep learning-guided domain recombination

SMAL-DR is an integrated pipeline designed to collect candidate domains related to a target protein from large-scale domain resources, refine their structural boundaries, and predict the potential activity of recombined variants using RecombRank, a pairwise ranking model for few-shot learning.
The workflow includes database mining and preparation (Task 1), DALI-based domain boundary refinement (Task 2), and ESM-2 embedding–based RecombRank training and inference (Tasks 3–6).

Terminology

FSE: Full-Sequence Embedding — averaged ESM-2 embeddings of the full protein sequence (typically 1280 dimensions).

CSE: Catalytic-Site Embedding — ESM-2 embeddings extracted from aligned catalytic site positions.

RecombRank: Pairwise ranking model predicting the relative activity of domain recombination variants (implemented in code as PairNet_*).


## Table of Contents

  - Overview
  - System Requirements
  - Installation
  - Data and Directory Structure
  - Configuration File (config.json)
  - Pipeline Tasks (1–6)
  - Demo Execution
  - Output Files
  - Optional: Task Control
  - Reproducibility and Best Practices
  - Troubleshooting
  - License and Citation

## Overview

Candidate Domain Collection (Task 1):
Based on the input configuration, SMAL-DR retrieves and organizes candidate domains related to the target HNH-like domain from TED, CATH, and cluster-based databases. It downloads corresponding structure and sequence files, organizes them for subsequent analysis, and performs structural similarity searches using Foldseek.

Boundary Refinement (Task 2):
Structural alignments are conducted using DALI between candidate and reference domains to refine boundary positions. The outputs include refined domain boundaries, split structures, and similarity result files.

Potential Activity Ranking (Tasks 3–6):
Using ESM-2 embeddings (FSE + CSE), RecombRank performs pairwise ranking training and inference:

Tasks 3–4: MLP-based FSE+CSE fusion (training and inference).

Tasks 5–6: Transformer-based FSE+CSE fusion (training and inference).

Note: RecombRank ranks variants by potential activity, not by recombination compatibility. The structural compatibility aspect is primarily addressed in Task 2 via DALI boundary refinement.

## System Requirements

OS: Linux
Python: 3.9

Hardware:
  CPU capable of all steps
  GPU recommended (e.g., A100 / RTX3090) for RecombRank Transformer training/inference
  External Tools (must be available and executable):
  DALI (DaliLite v5)
  Foldseek
  (Optional) TM-align
  ESM-2 (embeddings can be precomputed and stored as .npy)

Validated on Ubuntu 22.04, Python 3.9, CUDA 12.6.

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Optional: check external tool availability
foldseek --version
# DALI check (depending on your installation)
# e.g. ensure /path/to/DaliLite.v5/bin contains executable binaries
````
It is recommended to precompute ESM-2 embeddings and specify the corresponding directories in config.json.

## Data and Directory Structure

```bash
SMAL-DR/
├─ src/
│  ├─ pipeline.py
│  ├─ PairNet_MLP_train.py              # RecombRank-MLP (training)
│  ├─ PairNet_MLP_inference.py          # RecombRank-MLP (inference)
│  ├─ PairNet_Transformer_train.py      # RecombRank-Transformer (training)
│  ├─ PairNet_Transformer_inference.py  # RecombRank-Transformer (inference)
│  ├─ utils1.py                         # Task1/structure-related utilities
│  └─ utils2.py                         # Task2 (DALI) utilities
├─ data/
│  └─ Cas9_submit/
│     ├─ config.json                    # Configuration example
│     ├─ test.csv                       # Sample input
│     └─ ... (input/intermediate files for Task1/2)
├─ requirements.txt
└─ README.md
````
For RecombRank inference (task4_config / task6_config)
  - main_dir: directory containing per-variant .npy ESM-2 embeddings.
  - embeddings_dir: aggregated ESM-2 embedding directory (e.g., Esm2Embedding-sp-1280).
  - json_path: JSON mapping variant IDs and feature positions (e.g., active_numbers).
  - cache_path: cache file (embedding_cache.pt) automatically generated during first inference.

## Configuration File (config.json)

Key fields are summarized below:
### Global and Input Settings
  - work_dir: root working directory (absolute path recommended).
  - proteins: list defining the target protein (e.g., Cas9/HNH).
  - input_files: paths to required input files for Tasks 1–2 (e.g., true_hnh_ted_info, ted_domain_cath_info, wt_domain_dir).
### Control Flags (Task 1)
  - Boolean switches: run_step0, run_step1, run_step2_*, run_step3_*, run_data_processing.
### External Tools
  - foldseek_binary_path
  - Under task2_config: dali_bin_path, dali_work_dir, import/export directories, and filenames.
### RecombRank (Training & Inference)
  - Tasks 3/4 (MLP)
    - task3_config: data paths, hyperparameters, and model file names.
    - task4_config: inference paths (main_dir, output_dir, model_weight, etc.).
  - Tasks 5/6 (Transformer)
    - task5_config: data paths, hyperparameters, model save directory, and filename.
    - task6_config: inference paths and output file.
Ensure Task5 best_model_filename matches Task6 model_weight (e.g., best_model_transformer.pth).







SMAL-DR operates in six main stages:

- **Task 1 – Structural Fold Mining (Database Exploration)**  
  In this stage, the pipeline mines structural folds from the TED database to identify and classify diverse HNH-like domains relevant to Cas9 engineering.  
  This process based TED (The Encyclopedia of Domains,Lau AMC et al. (2024) Exploring structural diversity across the protein universe with The Encyclopedia of Domains. Science 386:e adq4946.) to collect, filter, and preprocess protein domain information, preparing it for downstream structural and functional analyses.

- **Task 2 – Domain Boundary Refinement (DALI Alignment)**  
  Structural boundaries of candidate domains are refined using DALI structural alignment.  
  This step ensures recombinational compatibility between domain fragments and improves the precision of subsequent modeling by removing poorly aligned or structurally inconsistent regions.

- **Task 3 – MLP Model Training (Supervised Learning)**  
  A model based multilayer perceptron (PairNet-MLP) is trained on curated wet-lab datasets and sequence embeddings (from ESM-2).  
  The model learns to capture sequence–activity relationships and predict activity outcomes for engineered variants.

- **Task 4 – MLP Model Inference**  
  The trained MLP model is applied to unseen protein variants to infer potential activity and assess the recombinational compatibility of newly designed domains.

- **Task 5 – Transformer Model Training (Sequence Representation Learning)**  
  A model based transformer (PairNet-Transformer) is trained on curated wet-lab datasets and sequence embeddings (from ESM-2).  
  The model learns to capture sequence–activity relationships and predict activity outcomes for engineered variants.

- **Task 6 – Transformer Model Inference**  
  The trained Transformer modelis applied to unseen protein variants to infer potential activity and assess the recombinational compatibility of newly designed domains.

---

Overall, the SMAL-DR pipeline provides an end-to-end computational solution for **protein domain discovery, boundary refinement, and activity prediction**.  
All stages can be executed seamlessly through the integrated entry point:

```bash
python src/pipeline.py
````

## System Requirements

- **Operating System:** Linux  
- **Python:** 3.9  
- **Dependencies:** Listed in `requirements.txt`  
- **External Tools:**  
  The SMAL-DR pipeline integrates several external bioinformatics and structural analysis tools:  
  - **[DALI](http://ekhidna2.biocenter.helsinki.fi/dali/)** — for structural alignment and domain boundary refinement  
  - **[Foldseek](https://github.com/steineggerlab/foldseek)** — for rapid structural comparison (optional but recommended)  
  - **[TM-align](https://zhanggroup.org/TM-align/)** — for fold-level structural similarity evaluation  
  - **[ESM-2](https://github.com/facebookresearch/esm)** — pretrained protein language model for embedding generation  
  - **BLAST+** (optional) — for initial sequence-based similarity search  
- **Hardware Requirements:**  
  - GPU recommended for Transformer model training and inference (e.g., NVIDIA RTX 3090 / A100)  
  - CPU-only mode is supported for feature extraction and MLP inference  
- **Tested Environment:**  
  - Ubuntu 22.04  
  - Python 3.9  
  - CUDA 12.3  
  - PyTorch 2.1  
  - Foldseek 9.0, TM-align (Nov 2022 build), DALI v5

## Installation Guide

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/SMAL-DR.git
cd SMAL-DR
pip install -r requirements.txt
````

## Demo Instructions

The full SMAL-DR pipeline can be run directly by executing the following command:

```bash
python src/pipeline.py
````

What the demo will do:
Input data:
The pipeline will automatically read the example input data from data/Cas9_submit/test.csv.
This file contains pre-collected protein domain information for demonstration purposes. 

Pipeline Execution:
Running the command will sequentially execute the following tasks:
Task 1: Structural fold mining to identify diverse HNH-like domains for Cas9 engineering.
Task 2: Refining domain boundaries using DALI structural alignment.
Task 3: MLP-based model training using wet-lab data.
Task 4: MLP-based model inference.
Task 5: Transformer-based model training using wet-lab data.
Task 6: Transformer-based model inference.

Output:
After running the pipeline, the following output will be generated:
Log files:
smal_dr_phase1.log: Logs for Task 1 execution.
smal_dr_phase2.log: Logs for Task 2 execution.

Resulting Processed Data:
Files will be saved in your working directory as defined in the config.json. This includes various processed files from tasks such as PDB data, domain information, and refined structural data.

Expected Output:
The following results will be available in your specified work_dir:
Log files: Detailed logs for each task, saved as .log files, indicating the progress and results of each pipeline step.

Processed Data:

Processed protein and domain data files from the structural alignment and domain identification steps.

Example: If you process protein sequences, results will be saved in specific subdirectories for each task, including processed PDB files, domain information, and results of structural similarity analysis.
















