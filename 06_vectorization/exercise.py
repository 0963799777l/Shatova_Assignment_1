"""
Exercise Set 6 — Vectorization (easy → hard)

Instructions:
- Fill in the TODO sections.
- Run: python exercise.py
"""
import numpy as np


def rowwise_softmax(X: np.ndarray) -> np.ndarray:
    """Stable softmax over axis=1 for X shape (N,D)."""
    # TODO
    return None


def rolling_mean_1d(x: np.ndarray, k: int) -> np.ndarray:
    """Rolling mean, output length len(x)-k+1. Use cumsum trick."""
    # TODO
    return None


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode labels shape (N,) -> (N,C)."""
    # TODO
    return None


def pairwise_cosine_similarity(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity for all row pairs, returns (N,M)."""
    # TODO
    return None


def segment_sum(values: np.ndarray, segment_ids: np.ndarray, num_segments: int) -> np.ndarray:
    """Per-segment sums, like a tiny groupby-sum."""
    # TODO
    return None


def conv2d_valid(X: np.ndarray, K: np.ndarray) -> np.ndarray:
    """VALID 2D convolution using sliding_window_view (no output loops)."""
    # TODO
    return None


def _tests():
    rng = np.random.default_rng(0)

    X = rng.normal(size=(5, 7))
    S = rowwise_softmax(X)
    assert S.shape == X.shape
    assert np.allclose(S.sum(axis=1), 1.0)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(rolling_mean_1d(x, 2), np.array([1.5, 2.5, 3.5]))

    labels = np.array([2, 0, 1, 2])
    OH = one_hot(labels, 3)
    assert OH.shape == (4, 3)
    assert np.array_equal(OH.sum(axis=1), np.ones(4))
    assert np.array_equal(OH.argmax(axis=1), labels)

    A = rng.normal(size=(6, 4))
    B = rng.normal(size=(3, 4))
    C = pairwise_cosine_similarity(A, B)
    assert C.shape == (6, 3)
    i, j = 2, 1
    ref = float(np.dot(A[i], B[j]) / ((np.linalg.norm(A[i]) * np.linalg.norm(B[j])) + 1e-12))
    assert np.allclose(C[i, j], ref)

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    seg = np.array([0, 1, 0, 2, 1])
    out = segment_sum(values, seg, 3)
    assert np.allclose(out, np.array([4.0, 7.0, 4.0]))

    X2 = np.arange(1, 17, dtype=float).reshape(4, 4)
    K = np.array([[1, 0], [0, -1]], dtype=float)
    Y = conv2d_valid(X2, K)
    Y_ref = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            Y_ref[r, c] = np.sum(X2[r:r+2, c:c+2] * K)
    assert np.allclose(Y, Y_ref)

    print("All tests passed ✅")


if __name__ == "__main__":
    _tests()
