# KomlosEncode

**An open-source Python package for efficient DNA/RNA sequence encoding using the Komlós–Hadamard transform**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Software Impacts](https://img.shields.io/badge/Software%20Impacts-Published-green)](https://doi.org/10.1016/XXXXXX)

---

## Overview

KomlosEncode implements the **Komlós–Hadamard encoding scheme** for biological sequence representation. Unlike conventional One-Hot encoding, which produces high-dimensional sparse matrices whose size grows exponentially with k-mer length, KomlosEncode assigns each nucleotide a compact **±1 vector** whose dimension grows only as **O(log N)** with alphabet size.

Key properties:
- **Dimensionality reduction**: 3k features per k-mer vs. 4k for One-Hot — an exponential saving
- **Equidistant codes**: all nucleotide pairs are equally separated under Euclidean, Minkowski, and Chebyshev metrics, eliminating ordering bias
- **Theoretically grounded**: based on the Komlós conjecture from combinatorial discrepancy theory
- **Broad applicability**: supports DNA, RNA, dinucleotide, and amino acid (protein) sequences
- **Hilbert curve integration**: sequences are transformed into 2D spatial images (Komlós-encoded Hilbert curve images) for CNN-based classification workflows

---

## Repository Structure

```
KomlosEncode/
├── final_komlos.py         # Core training pipeline with Optuna hyperparameter optimisation (CPU)
├── final_komlos_gpu.py     # GPU-accelerated training pipeline with resource monitoring
├── final_komlos_kf.py      # K-fold cross-validation pipeline with stability analysis
├── README.md
└── LICENSE
```

### File descriptions

| File | Description |
|------|-------------|
| `final_komlos.py` | Main training script. Loads Komlós-encoded sequence images, runs Optuna hyperparameter search, trains InceptionV3, and saves results, metrics, and visualisations. |
| `final_komlos_gpu.py` | GPU-optimised version of the training pipeline with extended resource usage logging (CPU/GPU memory tracking per epoch). |
| `final_komlos_kf.py` | K-fold cross-validation pipeline. Runs stratified k-fold training, evaluates model stability across folds, and identifies the most stable trial configuration. |

---

## Requirements

```
python >= 3.8
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
opencv-python (cv2)
optuna
psutil
```

Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow opencv-python optuna psutil
```

---

## Data Requirements

Each script expects two inputs:

1. **Komlós-encoded sequence images** — pre-generated 2D Hilbert curve image representations of the encoded sequences, stored as `.png` files named `Zoom_Komlos_<index>.png`
2. **A JSON dataset file** — a `.json` file (one record per line) containing sequence labels and metadata

Update the data paths at the top of each script before running:

```python
# In final_komlos.py / final_komlos_gpu.py / final_komlos_kf.py
image_path = os.path.expanduser("~/Zoom_Images/KomlosEncoding")   # folder of .png images
data_path  = os.path.expanduser("~/Datasets/Small_Data_Human_InP.json")  # JSON labels file
```

---

## Usage

### Standard training with Optuna hyperparameter search

```bash
python final_komlos.py
```

Runs Optuna-based hyperparameter optimisation over InceptionV3 architecture. Outputs are saved to:
```
~/experiment_results_komlos_secOptuna/
├── optuna_study_runs_secOptuna/   # Per-trial logs
├── best_model_secOptuna/          # Best model checkpoint
├── configs_secOptuna/             # Best hyperparameter config (JSON)
├── metrics_secOptuna/             # Training/validation metrics
├── visualisations_secOptuna/      # Optuna plots, confusion matrices, ROC curves
└── processed_images_secOptuna/    # Sample preprocessed images
```

### GPU-accelerated training

```bash
python final_komlos_gpu.py
```

Same pipeline as above, optimised for GPU execution with per-epoch CPU and GPU memory monitoring.

### K-fold cross-validation

```bash
python final_komlos_kf.py
```

Runs stratified k-fold cross-validation. Evaluates performance stability across folds and identifies the most stable trial. Outputs confusion matrices, ROC curves, and per-fold metrics.

---

## Nucleotide Encoding

Each DNA nucleotide is assigned to a vertex of a regular tetrahedron in ℝ³ with all coordinates in {−1, +1}:

| Nucleotide | Komlós Code   |
|-----------|---------------|
| A         | (+1, +1, −1) |
| T         | (+1, −1, +1) |
| C         | (−1, +1, +1) |
| G         | (−1, −1, −1) |

All pairwise distances are equal (Euclidean = 2√2, Chebyshev = 2), ensuring no nucleotide is implicitly favoured during model training. For k-mers, individual nucleotide vectors are concatenated, yielding vectors of length **3k** versus **4k** for One-Hot encoding.

---

## Benchmarks

KomlosEncode was evaluated on three public genomic datasets against One-Hot, Binary, NCP, ENAC, NCPAFE, and FastDNA encodings:

| Encoding     | Accuracy (%) | Precision (%) | Recall (%) | AUC (%) |
|-------------|-------------|--------------|-----------|--------|
| ENAC        | 84.20       | 88.30        | 82.15     | 90.10  |
| Binary      | 85.83       | 91.58        | 79.98     | 92.35  |
| NCP         | 86.10       | 89.50        | 84.75     | 92.50  |
| NCPAFE      | 87.45       | 90.22        | 86.33     | 93.80  |
| One-Hot     | 88.99       | 91.01        | 79.96     | 93.43  |
| FastDNA     | 89.92       | 93.10        | 90.50     | 96.25  |
| **Komlós**  | **90.46**   | **94.45**    | **91.71** | **97.17** |

*CNN classifier on ChIP Atlas enhancer dataset. See the paper for full results including FANTOM5 and MODOMICS datasets.*

---

## Citation

If you use KomlosEncode in your research, please cite:

```bibtex
@article{kabbani2025komlosencode,
  title     = {KomlosEncode: An open-source Python package for efficient 
               DNA/RNA sequence encoding using the Komlós–Hadamard transform},
  author    = {Kabbani, Kareem and Belhaouari, Samir B. and Aupetit, Michaël 
               and Al-Qahtani, Aisha and Halabi, Ahmad and Haoudi, Sophia L. 
               and Bensmail, Halima},
  journal   = {Software Impacts},
  year      = {2025},
  publisher = {Elsevier},
  doi       = {10.1016/XXXXXX}
}
```

---

## Related Work

This package is based on the encoding theory developed in:

> Belhaouari, S.B., AlQudah, R. *Evaluation of the Komlós Conjecture Using Multi-Objective Optimization.* Contemporary Mathematics, 3484–3516 (2024).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact the corresponding author:  
**Halima Bensmail** — hbensmail@hbku.edu.qa  
Qatar Computing Research Institute, Hamad Bin Khalifa University, Doha, Qatar
