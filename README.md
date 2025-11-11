# SMAL-DR: Structural fold mining-and deep learning-guided domain recombination

This repository contains the code for a comprehensive pipeline designed to process protein domain data using various tasks. The pipeline integrates sequence analysis, domain boundary detection, and deep learning models (MLP and Transformer) for protein data training, and inference.

## Pipeline Overview

The SMAL-DR pipeline consists of the following tasks:
- **Task 1**: Structural fold mining from the TED database identifies diverse HNH-like domains for Cas9 engineering.
- **Task 2**: Refining domain boundaries using DALI structural alignment to enhance recombinational compatibility.
- **Task 3**: MLP-based model training for protein data.
- **Task 4**: MLP-based model inference for protein data.
- **Task 5**: Transformer-based model training for protein data.
- **Task 6**: Transformer-based model inference for protein data.
- 
The pipeline can be executed using the `pipeline.py` script.


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

### Installation Guide

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/SMAL-DR.git
cd SMAL-DR
pip install -r requirements.txt


## Demo Instructions

The full SMAL-DR pipeline can be run directly by executing the following command:

```bash
python src/pipeline.py

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






