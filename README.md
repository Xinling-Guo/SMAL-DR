# SMAL-DR: Structural fold mining-and deep learning-guided domain recombination

SMAL-DR is an integrated pipeline designed to collect candidate domains related to a target protein from large-scale domain resources, refine their structural boundaries, and predict the potential activity of recombined variants using RecombRank, a pairwise ranking model for few-shot learning.
The workflow includes database mining and preparation (Task 1), DALI-based domain boundary refinement (Task 2), and ESM-2 embedding–based RecombRank training and inference (Tasks 3–6).

Terminology

FSE: Full-Sequence Embedding — averaged ESM-2 embeddings of the full protein sequence (typically 1280 dimensions).

CSE: Catalytic-Site Embedding — ESM-2 embeddings extracted from aligned catalytic site positions.

RecombRank: Pairwise ranking model predicting the relative activity of domain recombination variants (implemented in code as PairNet_*).

## Pipeline Overview

The **SMAL-DR (Structural fold Mining And deep Learning-guided Domain Recombination)** pipeline is a modular and extensible framework designed for protein domain analysis and engineering.  
It integrates large-scale structural data mining, alignment-based domain refinement, and deep learning-driven activity prediction into a unified workflow.

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












