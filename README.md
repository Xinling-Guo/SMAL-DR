# SMAL-DR: Structural fold mining-and deep learning-guided domain recombination

This repository contains the code for a comprehensive pipeline designed to process protein domain data using various tasks. The pipeline integrates sequence analysis, domain boundary detection, and deep learning models (MLP and Transformer) for protein data training, and inference.

## Pipeline Overview

The SMAL-DR pipeline consists of the following tasks:
- **Task 1**: Protein sequence data processing and domain identification.
- **Task 2**: Domain boundary refinement using DALI structural alignment.
- **Task 3**: MLP-based model training for protein data.
- **Task 4**: MLP-based model inference for protein data.
- **Task 5**: Transformer-based model training for protein data.
- **Task 6**: Transformer-based model inference for protein data.
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



