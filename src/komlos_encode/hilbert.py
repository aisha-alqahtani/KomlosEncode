"""Map an encoded 1-D sequence onto a 2-D Hilbert curve image.

The encoded array of shape ``(n_kmers, code_dim)`` is laid out along a Hilbert
space-filling curve of order ``order`` (side ``2**order``), producing an image
of shape ``(2**order, 2**order, code_dim)`` suitable as CNN input. The Hilbert
layout preserves 1-D locality in 2-D, which is what lets convolutional kernels
see neighbouring k-mers as spatial neighbours.
"""

from __future__ import annotations

import math

import numpy as np


def _d2xy(side: int, d: int):
    """Convert Hilbert distance ``d`` to (x, y) on a ``side`` x ``side`` grid."""
    x = y = 0
    t = d
    s = 1
    while s < side:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        # rotate the quadrant
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def min_order(n_points: int) -> int:
    """Smallest order ``l`` with ``2**(2l) >= n_points`` (paper's rule)."""
    if n_points <= 1:
        return 1
    return max(1, math.ceil(math.log2(n_points) / 2))


def to_hilbert_image(encoded: np.ndarray, order: int | None = None) -> np.ndarray:
    """Arrange ``encoded`` k-mer vectors along a Hilbert curve.

    Parameters
    ----------
    encoded : np.ndarray
        Array of shape ``(n_kmers, code_dim)`` from
        :func:`komlos_encode.komlos.encode_sequence`.
    order : int, optional
        Curve order; side length is ``2**order``. If ``None`` the smallest
        order that fits every k-mer is chosen.

    Returns
    -------
    np.ndarray
        Image of shape ``(2**order, 2**order, code_dim)`` (float32). Cells
        beyond the sequence length are left at zero.
    """
    encoded = np.atleast_2d(np.asarray(encoded))
    n_points, code_dim = encoded.shape
    if order is None:
        order = min_order(n_points)
    side = 2 ** order
    capacity = side * side
    if n_points > capacity:
        raise ValueError(
            f"order={order} gives {capacity} cells but sequence has "
            f"{n_points} k-mers; increase order"
        )

    img = np.zeros((side, side, code_dim), dtype=np.float32)
    for d in range(n_points):
        x, y = _d2xy(side, d)
        img[y, x] = encoded[d]
    return img
