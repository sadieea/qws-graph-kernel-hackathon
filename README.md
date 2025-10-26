# Quantum Walk Signatures (QWS) for Molecular Classification

**Author:** Sadiya Ansari  
**Hackathon:** QunaSys Quantum Hackathon Challenge, October 2025

---

## Overview

This project proposes a **Quantum Walk Signature (QWS)** — a quantum-inspired graph feature map for molecular classification.  
QWS leverages **Continuous-Time Quantum Walks (CTQW)** on the graph Laplacian to encode global structural information of molecular graphs into low-dimensional feature vectors.  
These vectors are then used with **Support Vector Machines (SVMs)** for classification.

### Benchmark Methods
- **Extended Feature Vector (EFV):** Classical topological indices (Wiener, Randić, Estrada)  
- **Weisfeiler–Lehman (WL) Subtree Kernel:** State-of-the-art graph kernel

**Datasets:** MUTAG, PROTEINS, PTC-MR, NCI1, AIDS (TU Dortmund repository)

---

## Repository Structure

Quantum-Walk-Signature/
├── QuantumWalkSignature.py # Main Python file (single entry point)
├── report/
│ └── Quantum_Walk_Signature_Hackathon.pdf
├── datasets/
│ ├── MUTAG/
│ ├── PROTEINS/
│ ├── PTC-MR/
│ ├── NCI1/
│ └── AIDS/
├── requirements.txt # Python dependencies
└── README.md

text

---

## Installation

Clone the repository and install dependencies:

git clone <repo_url>
cd Quantum-Walk-Signature
pip install -r requirements.txt

text

---

## Running the Pipeline

Run the full pipeline for all datasets:

python QuantumWalkSignature.py

text

The script will:

- Load each dataset from the `datasets/` folder  
- Compute Quantum Walk Signatures (QWS) for each graph  
- Compute EFV features  
- Compute WL kernel matrices  
- Train SVM classifiers with 10-fold cross-validation  
- Print Accuracy and F1-score for all methods  

---

## Sample Results

| Dataset  | Method | Accuracy | F1-Score |
|-----------|---------|-----------|-----------|
| MUTAG     | QWS     | 0.86      | 0.85      |
|           | EFV     | 0.85      | 0.85      |
|           | WL      | 0.85      | 0.85      |
| PROTEINS  | QWS     | 0.74      | 0.72      |
|           | EFV     | 0.72      | 0.72      |
|           | WL      | 0.71      | 0.71      |
| PTC-MR    | QWS     | 0.65      | 0.63      |
|           | EFV     | 0.61      | 0.61      |
|           | WL      | 0.63      | 0.62      |
| NCI1      | QWS     | TBD       | TBD       |
| AIDS      | QWS     | TBD       | TBD       |

*Note:* “TBD” indicates datasets where results are pending.

---

## Requirements

All dependencies are listed in `requirements.txt`:

- Python 3.8+  
- numpy  
- scipy  
- networkx  
- scikit-learn  
- matplotlib *(optional, for visualization)*

---

## References

- *Graph Kernels: State-of-the-Art and Future Challenges*  
- *Benchmark of the Full and Reduced Effective Resistance Kernel for Molecular Classification*  
- *Kernel Methods for Protein Function Prediction*  
- *Fast, Accurate and Interpretable Graph Classification with Topological Kernels*  

---

## Notes

- Ensure datasets are placed in the correct folder structure.  
- The QWS computation involves eigendecomposition, which scales as **O(n³)** per graph.  
- EFV features are classical baselines for comparison.  
- WL kernel is manually implemented and normalized.
