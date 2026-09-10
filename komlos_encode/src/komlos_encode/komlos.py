"""Core Komlos-Hadamard encoding.

Each symbol of a biological alphabet is mapped to a compact ``+/-1`` code
vector. For the 4-letter nucleotide alphabets the codes are the vertices of a
regular tetrahedron in R^3 (Belhaouari & AlQudah, 2024), which are exactly
equidistant under the Euclidean, Minkowski and Chebyshev metrics. For larger
alphabets (dinucleotides, amino acids) compact ``+/-1`` codes of dimension
``d = ceil(log2(n))`` are generated; see ``build_codebook``.

Public API
----------
encode_sequence(seq, k=1, mode="dna") -> np.ndarray
    Encode a string into an array of shape (n_kmers, code_dim).
get_codebook(mode) -> dict[str, np.ndarray]
    Return the symbol -> code mapping for a given alphabet.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Fixed nucleotide code tables (regular tetrahedron in R^3, +/-1 space).
# All six pairwise squared Euclidean distances equal 8 -> exactly equidistant.
# ---------------------------------------------------------------------------
_DNA_CODES: Dict[str, np.ndarray] = {
    "A": np.array([+1, +1, -1], dtype=np.int8),
    "T": np.array([+1, -1, +1], dtype=np.int8),
    "C": np.array([-1, +1, +1], dtype=np.int8),
    "G": np.array([-1, -1, -1], dtype=np.int8),
}

# RNA shares the tetrahedron; U takes T's vertex.
_RNA_CODES: Dict[str, np.ndarray] = {
    "A": _DNA_CODES["A"],
    "U": _DNA_CODES["T"],
    "C": _DNA_CODES["C"],
    "G": _DNA_CODES["G"],
}

# Standard 20 amino acids.
_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _compact_pm1_codes(n: int) -> np.ndarray:
    """Return ``n`` distinct compact ``+/-1`` codes of dimension ceil(log2 n).

    Codes are drawn from the binary-reflected (Gray-ordered) vertices of the
    ``d``-cube so that consecutive symbols differ in a single coordinate. This
    keeps dimensionality at ``O(log n)``; unlike the tetrahedron it is not
    exactly equidistant for ``n > 4``, which is the standard compactness/
    equidistance trade-off for large alphabets.
    """
    d = max(1, math.ceil(math.log2(n)))
    codes = np.empty((n, d), dtype=np.int8)
    for i in range(n):
        gray = i ^ (i >> 1)  # binary-reflected Gray code
        bits = [(gray >> b) & 1 for b in range(d)]
        codes[i] = np.array([1 if b else -1 for b in bits], dtype=np.int8)
    return codes


def build_codebook(mode: str = "dna") -> Dict[str, np.ndarray]:
    """Return the symbol -> +/-1 code mapping for the requested alphabet.

    Parameters
    ----------
    mode : {"dna", "rna", "dinucleotide", "amino", "protein"}
    """
    mode = mode.lower()
    if mode == "dna":
        return dict(_DNA_CODES)
    if mode == "rna":
        return dict(_RNA_CODES)
    if mode in ("amino", "protein", "aa"):
        codes = _compact_pm1_codes(len(_AMINO_ACIDS))
        return {aa: codes[i] for i, aa in enumerate(_AMINO_ACIDS)}
    if mode in ("dinucleotide", "dinuc"):
        # A dinucleotide is a DNA 2-mer: concatenate the two nucleotide codes.
        alphabet = "ACGT"
        book = {}
        for a in alphabet:
            for b in alphabet:
                book[a + b] = np.concatenate([_DNA_CODES[a], _DNA_CODES[b]])
        return book
    raise ValueError(f"unknown mode {mode!r}")


# Public alias used in the paper's docstring/examples.
def get_codebook(mode: str = "dna") -> Dict[str, np.ndarray]:
    return build_codebook(mode)


def encode_sequence(seq: str, k: int = 1, mode: str = "dna") -> np.ndarray:
    """Encode ``seq`` into an array of shape ``(n_kmers, code_dim)``.

    For ``k == 1`` each symbol becomes one code vector. For ``k > 1`` the codes
    of the ``k`` symbols in each sliding window are concatenated, giving a
    per-k-mer vector of length ``k * base_code_dim`` (e.g. ``3k`` for DNA)
    versus ``4**k`` for one-hot.

    Parameters
    ----------
    seq : str
        Input sequence. Case-insensitive; whitespace ignored.
    k : int, default 1
        k-mer length (sliding window, step 1).
    mode : str, default "dna"
        Alphabet; see :func:`build_codebook`.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    seq = "".join(seq.split()).upper()
    book = build_codebook(mode)
    base_dim = next(iter(book.values())).shape[0]

    unknown = set(seq) - set(book)
    if unknown:
        raise ValueError(f"symbols {sorted(unknown)} not in {mode!r} alphabet")

    n_kmers = len(seq) - k + 1
    if n_kmers <= 0:
        raise ValueError(f"sequence length {len(seq)} shorter than k={k}")

    out = np.empty((n_kmers, k * base_dim), dtype=np.int8)
    for i in range(n_kmers):
        window = seq[i : i + k]
        out[i] = np.concatenate([book[s] for s in window])
    return out


def code_dim(mode: str = "dna", k: int = 1) -> int:
    """Length of the encoded vector per k-mer."""
    base = next(iter(build_codebook(mode).values())).shape[0]
    return k * base
