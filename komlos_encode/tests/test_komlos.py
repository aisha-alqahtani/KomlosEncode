import numpy as np
import pytest

from komlos_encode.komlos import encode_sequence, build_codebook, code_dim
from komlos_encode.hilbert import to_hilbert_image, min_order


def test_dna_codes_are_pm1_and_equidistant():
    book = build_codebook("dna")
    codes = np.stack(list(book.values())).astype(float)
    assert set(np.unique(codes)) <= {-1.0, 1.0}
    # all six pairwise squared distances equal (regular tetrahedron)
    dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            dists.append(np.sum((codes[i] - codes[j]) ** 2))
    assert len(set(dists)) == 1
    assert dists[0] == 8


def test_encode_shape_dna_kmer():
    enc = encode_sequence("ATCGATCGATCG", k=3, mode="dna")
    # 12 - 3 + 1 = 10 k-mers, 3 * 3 = 9 dims
    assert enc.shape == (10, 9)
    assert set(np.unique(enc)) <= {-1, 1}


def test_dimensionality_beats_one_hot():
    k = 4
    assert code_dim("dna", k) == 3 * k          # 12
    assert code_dim("dna", k) < 4 ** k          # vs 256 for one-hot


def test_rna_uses_u_not_t():
    book = build_codebook("rna")
    assert "U" in book and "T" not in book


def test_amino_acid_codes_distinct_and_compact():
    book = build_codebook("amino")
    assert len(book) == 20
    dim = next(iter(book.values())).shape[0]
    assert dim == 5  # ceil(log2(20))
    codes = {tuple(v.tolist()) for v in book.values()}
    assert len(codes) == 20  # all distinct


def test_hilbert_image_shape_and_order_rule():
    enc = encode_sequence("ATCGATCGATCG", k=3, mode="dna")
    img = to_hilbert_image(enc, order=4)
    assert img.shape == (16, 16, 9)
    # auto order fits every k-mer
    auto = to_hilbert_image(enc)
    side = 2 ** min_order(enc.shape[0])
    assert auto.shape[0] == side


def test_unknown_symbol_raises():
    with pytest.raises(ValueError):
        encode_sequence("ATZX", k=1, mode="dna")


def test_paper_snippet_runs():
    from komlos_encode.komlos import encode_sequence
    from komlos_encode.hilbert import to_hilbert_image

    seq = "ATCGATCGATCG"
    enc = encode_sequence(seq, k=3, mode="dna")
    img = to_hilbert_image(enc, order=4)
    assert enc.ndim == 2 and img.ndim == 3
