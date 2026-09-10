#!/usr/bin/env python
"""End-to-end example: encode sequences and train a simple classifier.

Runs with only the package's core dependencies (no torch, no external data),
so it works as a smoke test that the encoder feeds a real ML workflow.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from komlos_encode.komlos import encode_sequence
from komlos_encode.utils import flatten


def make_toy_dataset(n=400, length=60, seed=0):
    """Two classes: class 1 is GC-enriched, class 0 is AT-enriched."""
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        if rng.random() < 0.5:
            probs = [0.15, 0.15, 0.35, 0.35]  # A T C G  -> GC rich
            y = 1
        else:
            probs = [0.35, 0.35, 0.15, 0.15]  # AT rich
            y = 0
        seqs.append("".join(rng.choice(list("ATCG"), size=length, p=probs)))
        labels.append(y)
    return seqs, np.array(labels)


def main():
    seqs, y = make_toy_dataset()

    # Encode each sequence as a flat Komlos feature vector (k=3 DNA).
    X = np.stack([flatten(encode_sequence(s, k=3, mode="dna")) for s in seqs])
    print(f"feature matrix: {X.shape}  (one-hot k=3 would be {len(seqs[0]) - 2} x 64)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"held-out accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
