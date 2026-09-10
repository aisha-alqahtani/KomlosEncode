"""Lightweight visualisation helpers (matplotlib)."""

from __future__ import annotations

import numpy as np

from .komlos import build_codebook


def plot_codes(mode: str = "dna", ax=None):
    """Scatter the code vectors. For DNA/RNA this draws the R^3 tetrahedron."""
    import matplotlib.pyplot as plt  # local import keeps import time low

    book = build_codebook(mode)
    labels = list(book)
    pts = np.stack([book[s] for s in labels]).astype(float)

    if pts.shape[1] == 3:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
        for (x, y, z), s in zip(pts, labels):
            ax.text(x, y, z, s)
    else:
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(pts[:, 0], pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts)))
        for p, s in zip(pts, labels):
            ax.text(p[0], p[1] if pts.shape[1] > 1 else 0, s)
    return ax


def plot_hilbert_image(img: np.ndarray, channel: int = 0, ax=None):
    """Display one channel of a Hilbert-curve image."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(np.asarray(img)[..., channel], interpolation="nearest")
    ax.set_axis_off()
    return ax
