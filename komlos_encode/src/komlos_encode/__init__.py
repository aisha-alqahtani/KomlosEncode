"""KomlosEncode: compact +/-1 encoding of biological sequences.

Grounded in the Komlos conjecture from combinatorial discrepancy theory, each
symbol is mapped to a compact +/-1 code whose dimension grows as O(log N) with
alphabet size, with native Hilbert-curve image support for spatial deep
learning.
"""

from .komlos import encode_sequence, build_codebook, get_codebook, code_dim
from .hilbert import to_hilbert_image, min_order
from .utils import kmers, encode_batch, flatten

__all__ = [
    "encode_sequence",
    "build_codebook",
    "get_codebook",
    "code_dim",
    "to_hilbert_image",
    "min_order",
    "kmers",
    "encode_batch",
    "flatten",
]

__version__ = "0.1.0"
