"""
Exercise Set 3 — Broadcasting

Instructions:
- Fill in the TODO sections.
- Run: python exercise.py
"""
import numpy as np


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-normalize X to have L2 norm = 1 per row.

    X: shape (N, D) float
    Return: shape (N, D)

    Requirements:
    - Use broadcasting (no loops).
    - Avoid division by 0 by adding eps inside the sqrt.
    """
    # TODO
    return None


def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances between rows of A and rows of B.

    A: shape (N, D)
    B: shape (M, D)
    Return: shape (N, M) where out[i, j] = ||A[i] - B[j]||^2

    Requirements:
    - Use broadcasting (no explicit Python loops).
    """
    # TODO
    return None


def _tests():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5))
    Y = normalize_rows(X)
    norms = np.sqrt((Y * Y).sum(axis=1))
    assert np.allclose(norms, 1.0, atol=1e-7)

    A = rng.normal(size=(7, 3))
    B = rng.normal(size=(4, 3))
    D = pairwise_sq_dists(A, B)
    assert D.shape == (7, 4)

    D_ref = np.empty((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            diff = A[i] - B[j]
            D_ref[i, j] = float(np.dot(diff, diff))
    assert np.allclose(D, D_ref)

    print("All tests passed ✅")


if __name__ == "__main__":
    _tests()
