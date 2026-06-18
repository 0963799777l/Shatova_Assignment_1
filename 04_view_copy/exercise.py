"""
Exercise Set 4 — View and copy

Instructions:
- Fill in the TODO sections.
- Run: python exercise.py
"""

from __future__ import annotations

def as_contiguous_float32(X: np.ndarray) -> np.ndarray:
    """
    Return X as a C-contiguous float32 array.

    Requirements:
    - If X is already float32 and C-contiguous, return it WITHOUT copying.
    - Otherwise, return a copy with dtype float32 and C order.
    """
    # TODO
    return None


def split_view(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given X shape (N, 2D) where D>0, return (A, B) where:
      A: first D columns of X
      B: last  D columns of X

    Requirements:
    - A and B must both be VIEWS into X (no copies).
    """
    # TODO
    return None, None


def _tests():
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    Y = as_contiguous_float32(X)
    assert Y is X

    Xf = np.asfortranarray(X)
    Y2 = as_contiguous_float32(Xf)
    assert Y2.dtype == np.float32 and Y2.flags["C_CONTIGUOUS"]
    assert np.array_equal(Y2, Xf)
    assert Y2 is not Xf

    Z = np.arange(60).reshape(5, 12)
    A, B = split_view(Z)
    assert A.shape == (5, 6) and B.shape == (5, 6)
    A[0, 0] = -123
    assert Z[0, 0] == -123
    B[0, 0] = -456
    assert Z[0, 6] == -456

    print("All tests passed ✅")


if __name__ == "__main__":
    _tests()
