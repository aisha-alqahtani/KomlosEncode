#!/usr/bin/env python
"""Generate Komlos-encoded Hilbert-curve images from sequences.

This is the preprocessing step that turns raw sequences into the
``Zoom_Komlos_<index>.png`` images consumed by the training pipelines
(``final_komlos.py``, ``final_komlos_gpu.py``, ``final_komlos_kf.py``).

For DNA at ``k=1`` each nucleotide maps to a 3-component +/-1 code, which is
written directly as the R, G, B channels of the image -- so a sequence becomes
an RGB Hilbert image with no extra colour mapping.

Input
-----
A JSON-lines file (one record per line) with at least a ``sequence`` field and
optionally a ``label`` field, e.g.::

    {"sequence": "ATCG...", "label": 1}
    {"sequence": "GGCA...", "label": 0}

Usage
-----
    python experiments/generate_images.py \
        --input data/sequences.jsonl \
        --outdir Zoom_Images/KomlosEncoding \
        --labels-out labels.csv \
        --mode dna --k 1

Note
----
If your published figures used the FocusFFT / "Zoom" preprocessing (Amer et
al., 2025), apply that transform to the encoded array before saving; this
script produces the plain Hilbert layout, which is fully reproducible from the
package alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from matplotlib import image as mpimg

from komlos_encode.komlos import encode_sequence
from komlos_encode.hilbert import to_hilbert_image


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Map a Hilbert image (H, W, C) in {-1,+1} to a uint8 RGB array.

    Uses the first three channels (exactly the 3 tetrahedron coordinates for
    DNA/RNA at k=1). Channels are padded or truncated to 3.
    """
    img = np.asarray(img, dtype=np.float32)
    h, w, c = img.shape
    if c < 3:
        img = np.concatenate([img, np.zeros((h, w, 3 - c), np.float32)], axis=2)
    rgb = img[:, :, :3]
    rgb = ((rgb + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)  # {-1,+1}->{0,255}
    return rgb


def generate(input_path, outdir, labels_out=None, mode="dna", k=1, order=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(input_path) as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = rec["sequence"]
            enc = encode_sequence(seq, k=k, mode=mode)
            img = to_hilbert_image(enc, order=order)
            rgb = _to_rgb(img)
            fname = f"Zoom_Komlos_{idx}.png"
            mpimg.imsave(outdir / fname, rgb)
            rows.append({"index": idx, "image": fname, "label": rec.get("label", "")})

    if labels_out:
        with open(labels_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["index", "image", "label"])
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSON-lines file of sequences")
    p.add_argument("--outdir", required=True, help="output folder for PNG images")
    p.add_argument("--labels-out", default=None, help="optional CSV of index,image,label")
    p.add_argument("--mode", default="dna", choices=["dna", "rna", "dinucleotide", "amino"])
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--order", type=int, default=None, help="Hilbert order (auto if omitted)")
    args = p.parse_args()
    n = generate(args.input, args.outdir, args.labels_out, args.mode, args.k, args.order)
    print(f"wrote {n} images to {args.outdir}")


if __name__ == "__main__":
    main()
