"""
Exercise Set 4 — View and copy

Instructions:
- Fill in the TODO sections.
- Run: python exercise.py
"""

from __future__ import annotations
import numpy as np


def as_contiguous_float32(X: np.ndarray) -> np.ndarray:
    """
    Return X as a C-contiguous float32 array.

    Requirements:
    - If X is already float32 and C-contiguous, return it WITHOUT copying.
    - Otherwise, return a copy with dtype float32 and C order.
    """
    # Перевіряємо, чи масив уже має тип float32 і C-contiguous порядок зберігання.
    # Якщо так, повертаємо той самий масив без копіювання.
    if X.dtype == np.float32 and X.flags["C_CONTIGUOUS"]:
        return X

    # Якщо масив не відповідає вимогам, створюємо C-contiguous копію з типом float32.
    return np.ascontiguousarray(X, dtype=np.float32)


def split_view(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given X shape (N, 2D) where D>0, return (A, B) where:
      A: first D columns of X
      B: last  D columns of X

    Requirements:
    - A and B must both be VIEWS into X (no copies).
    """
    # Визначаємо кількість стовпців у половині масиву.
    # Якщо X має форму (N, 2D), то D дорівнює половині кількості стовпців.
    D = X.shape[1] // 2

    # A — перша половина стовпців.
    # B — друга половина стовпців.
    # Зрізи повертають view, тобто представлення вихідного масиву без копіювання.
    A = X[:, :D]
    B = X[:, D:]

    return A, B


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

    print("All tests passed")


if __name__ == "__main__":
    _tests()
