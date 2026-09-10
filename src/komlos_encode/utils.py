"""k-mer and batch utilities."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .komlos import encode_sequence, code_dim


def kmers(seq: str, k: int) -> List[str]:
    """Return the list of overlapping k-mers (step 1)."""
    seq = "".join(seq.split()).upper()
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def encode_batch(
    seqs: Iterable[str], k: int = 1, mode: str = "dna", pad: bool = True
) -> np.ndarray:
    """Encode many sequences at once.

    Returns an array of shape ``(n_seqs, max_kmers, code_dim)``. Shorter
    sequences are zero-padded when ``pad=True``; otherwise all sequences must
    share the same length.
    """
    encoded = [encode_sequence(s, k=k, mode=mode) for s in seqs]
    dim = code_dim(mode, k)
    lengths = [e.shape[0] for e in encoded]
    if not pad and len(set(lengths)) > 1:
        raise ValueError("sequences differ in length; set pad=True")
    max_len = max(lengths)
    out = np.zeros((len(encoded), max_len, dim), dtype=np.int8)
    for i, e in enumerate(encoded):
        out[i, : e.shape[0]] = e
    return out


def flatten(encoded: np.ndarray) -> np.ndarray:
    """Flatten a (n_kmers, code_dim) encoding to a 1-D feature vector."""
    return np.asarray(encoded).reshape(-1)
