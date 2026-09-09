# KomlosEncode

**An open-source Python package for efficient DNA/RNA sequence encoding using the Komlós–Hadamard transform**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Compact `±1` encoding of biological sequences, with native Hilbert-curve image
support for spatial deep learning. Each nucleotide is placed at a vertex of a
regular tetrahedron in `R³`, preserving equidistant pairwise geometry while
keeping the encoded vector length at `3k` per k-mer versus `4^k` for one-hot.

---

## Installation

From source / GitHub:

```bash
pip install git+https://github.com/aisha-alqahtani/KomlosEncode.git
```

For local development:

```bash
git clone https://github.com/aisha-alqahtani/KomlosEncode.git
cd KomlosEncode
pip install -e ".[test]"
pytest            # run the test suite
```

The core library depends only on NumPy, SciPy, scikit-learn, and Matplotlib.
The training pipelines under `experiments/` need a heavier stack (see below).

---

## Quickstart

```python
from komlos_encode.komlos import encode_sequence
from komlos_encode.hilbert import to_hilbert_image

seq = "ATCGATCGATCG"
enc = encode_sequence(seq, k=3, mode="dna")   # shape (10, 9)
img = to_hilbert_image(enc, order=4)          # shape (16, 16, 9)
```

Supported `mode` values: `dna`, `rna`, `dinucleotide`, `amino`.

A complete, self-contained classification example (no external data, core deps
only):

```bash
python examples/quickstart.py
```

---

## Repository structure

```
KomlosEncode/
├── src/komlos_encode/       # installable library
│   ├── komlos.py            #   core ±1 code tables + encode_sequence
│   ├── utils.py             #   k-mer helpers, batch encoding
│   ├── hilbert.py           #   sequence → 2D Hilbert image
│   └── viz.py               #   plotting helpers
├── examples/quickstart.py   # end-to-end demo (encode → classify)
├── experiments/             # benchmark pipelines + image generation
│   ├── generate_images.py   #   sequences → Zoom_Komlos_<index>.png
│   ├── final_komlos.py      #   Optuna + InceptionV3 (CPU)
│   ├── final_komlos_gpu.py  #   GPU version
│   └── final_komlos_kf.py   #   k-fold cross-validation
├── tests/                   # pytest suite
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Modules

| Module | Purpose |
| --- | --- |
| `komlos_encode.komlos` | Core `±1` code tables and `encode_sequence` |
| `komlos_encode.utils` | k-mer helpers and batch encoding |
| `komlos_encode.hilbert` | Map an encoded sequence to a 2-D Hilbert image |
| `komlos_encode.viz` | Plot code geometry and Hilbert images |

---

## Reproducing the benchmarks

The training pipelines consume Hilbert-curve images generated from sequences.
First produce the images with the package, then train:

```bash
python experiments/generate_images.py \
    --input data/sequences.jsonl \
    --outdir Zoom_Images/KomlosEncoding --labels-out labels.csv
python experiments/final_komlos.py        # or _gpu.py / _kf.py
```

The training scripts additionally require:

```
torch torchvision pandas seaborn Pillow opencv-python optuna psutil
```

See `experiments/README.md` for details.

---

## Nucleotide encoding

Each DNA nucleotide is a vertex of a regular tetrahedron in `R³`, all
coordinates in `{−1, +1}`:

| Nucleotide | Komlós code |
| --- | --- |
| A | (+1, +1, −1) |
| T | (+1, −1, +1) |
| C | (−1, +1, +1) |
| G | (−1, −1, −1) |

All six pairwise distances are equal (Euclidean = 2√2, Chebyshev = 2), so no
nucleotide is implicitly favoured. For k-mers the nucleotide vectors are
concatenated, giving length `3k` versus `4^k` for one-hot. RNA shares the same
tetrahedron with U in place of T. Larger alphabets (dinucleotide, amino acid)
use compact `±1` codes of dimension `⌈log₂ n⌉`.

---

## Benchmarks

CNN classifier on the ChIP Atlas enhancer dataset (see the paper for FANTOM5
and MODOMICS):

| Encoding | Accuracy (%) | Precision (%) | Recall (%) | AUC (%) |
| --- | --- | --- | --- | --- |
| ENAC | 84.20 | 88.30 | 82.15 | 90.10 |
| Binary | 85.83 | 91.58 | 79.98 | 92.35 |
| NCP | 86.10 | 89.50 | 84.75 | 92.50 |
| NCPAFE | 87.45 | 90.22 | 86.33 | 93.80 |
| One-Hot | 88.99 | 91.01 | 79.96 | 93.43 |
| FastDNA | 89.92 | 93.10 | 90.50 | 96.25 |
| **Komlós** | **90.46** | **94.45** | **91.71** | **97.17** |

---

## Citation

If you use KomlosEncode, please cite the method paper and this software.

Method paper (fill in once finalised):

```
@article{komlos_method,
  title   = {<BMC Bioinformatics method paper — add title>},
  author  = {<authors>},
  journal = {BMC Bioinformatics},
  year    = {<year>}
}
```

Software:

```
@article{kabbani_komlosencode,
  title     = {KomlosEncode: An open-source Python package for efficient
               DNA/RNA sequence encoding using the Komlós–Hadamard transform},
  author    = {Kabbani, Kareem and Belhaouari, Samir B. and Aupetit, Michaël
               and Al-Qahtani, Aisha and Halabi, Ahmad and Haoudi, Sophia L.
               and Bensmail, Halima},
  year      = {2026}
}
```

Encoding theory:

> Belhaouari, S.B., AlQudah, R. *Evaluation of the Komlós Conjecture Using
> Multi-Objective Optimization.* Contemporary Mathematics, 3484–3516 (2024).

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

Corresponding author: **Halima Bensmail** — <hbensmail@hbku.edu.qa>
Qatar Computing Research Institute, Hamad Bin Khalifa University, Doha, Qatar.
