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

Estimated Time: 1-2 hours

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
│  ├─ All_wet_ID.xlsx
│  ├─ config.json     
│  ├─ HNH_TED_info.xlsx
│  └─ HNH_trueSpcas9/
│     └─ Q99ZW2_774-900_HNH.pdb
│  
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

## Pipeline Tasks (1–6)

### Task 1 — Candidate Domain Preparation
  - Retrieves and processes candidate domains from TED/CATH/cluster-related sources, downloads structure/sequence files, organizes output directories, and performs structure searches via Foldseek.

### Task 2 — DALI Boundary Refinement
  - Uses DALI for structure-based alignments between candidate and reference domains, outputs refined domains and structural similarity files.

### Task 3 — RecombRank (MLP Fusion) Training
  - Trains a pairwise ranking model using ESM-2 FSE and CSE embeddings with an MLP-based fusion network and MarginRankingLoss.

###  Task 4 — RecombRank (MLP Fusion) Inference
  - Loads pretrained MLP weights, caches embeddings, and produces activity ranking results for unseen variants.

### Task 5 — RecombRank (Transformer Fusion) Training
  -Uses a Transformer Encoder to fuse FSE and CSE features for ranking-based training.

### Task 6 — RecombRank (Transformer Fusion) Inference
  - Loads Transformer weights to infer ranking scores and outputs final predictions.

Tasks 3–6 share the same model logic (RecombRank) and differ only in the FSE+CSE fusion approach (MLP vs Transformer).

## Demo Execution
Run the integrated pipeline:

```bash
python src/pipeline.py --config data/Cas9_submit/config.json
````
Estimated Time: 3-5 hours

## Output Files

Typical outputs (may vary based on configuration):

  - Task1
    - work_dir/protein_relation_v3/domain_cath_teddb/cytoscape_network/cas9_fs_edge.csv
    - work_dir/protein_relation_v3/domain_cath_teddb/cytoscape_network/domain_cath_teddb_SpCas9.csv
    - work_dir/protein_relation_v3/domain_All_teddb_SpCas9.csv
    - work_dir/protein_relation_v3/domain_All_teddb_SpCas9_filter.csv
      
  - Task2
    - Dali_results/dali_results.csv
    - Dali_results/dali_10_results_pro.csv
    - Dali_results/Target_Dali_domain_10/, Dali_results/Query_Dali_domain_10/
    - Dali_results/FS_results/All_fstm_results.csv
    - Dali_results/dali_10_refined_results.csv
      
  - Task4 (MLP Inference)
    - .../output/mlp_inference/final_sorted_winrates.csv
      
  - Task5 (Transformer Training)
    - .../model_weights/transformer_train/<best_model_filename>
    
  - Task6 (Transformer Inference)
    - .../output/transformer_inference/final_sorted_winrates.csv
      
## Optional: Task Control

  - Enable/disable Task1 phases using Boolean flags in config.json (e.g., run_step2_cath_teddb = true/false).
    
  - Control Task2 substeps under task2_config (e.g., run_step1_download, run_step2_dali).
    
  - RecombRank training and inference are configured through task3_config–task6_config.

## Reproducibility and Best Practices

  - ESM-2 embeddings: Use consistent model and dimension (e.g., 1280-dim), with standardized pooling and site alignment.

  - Random seed: Set a fixed seed (e.g., set_seed(42)) for reproducible results.
 
  - Naming consistency: Match best_model_filename and model_weight between training/inference.
 
  - Cache reuse: Reuse embedding_cache.pt for faster subsequent inference.

  - External tool paths: Always provide absolute paths if not available in PATH.

## Troubleshooting

  - File not found: Check --config path, file existence, mount points, and permissions.

  - External tool errors: Verify DALI/Foldseek binaries exist and are executable.

  - Module import errors: Ensure execution from project root or add src/ to PYTHONPATH.

  - Empty inference output: Verify .npy embeddings and JSON index (active_numbers) match, and that embedding directories are consistent between training and inference.

## License and Citation

  - License: This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

  - Citation (example placeholder):

SMAL-DR & RecombRank: Structural fold mining and deep learning-guided domain recombination.
Authors, Journal/Year, DOI

  - Third-Party Resources:
    -ESM-2 (Meta AI)
    -Foldseek
    -DALI (DaliLite v5)
    -TED (The Encyclopedia of Domains)






















