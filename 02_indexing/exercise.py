"""
Exercise Set 2 — Array indexing

Instructions:
- Fill in the TODO sections.
- Run: python exercise.py
"""
import numpy as np


def slice_and_mask(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given X with shape (8, 8) containing integers, return:
      S1: a view of the center 4x4 block (rows 2..5, cols 2..5)
      S2: a 1D array of the elements on the anti-diagonal (top-right to bottom-left)
      S3: all elements of X that are prime numbers (in increasing order), as a 1D array

    Notes:
    - S1 must be a VIEW, not a copy.
    - S2 can be view or copy (either is OK).
    - For S3 you must use boolean masking (no Python loops over elements).
    """
    assert X.shape == (8, 8)

    # S1: центральний блок 4x4 як view
    S1 = X[2:6, 2:6]

    # S2: побічна діагональ, тобто елементи (0,7), (1,6), ..., (7,0)
    n = X.shape[0]
    idx = np.arange(n)
    S2 = X[idx, n - 1 - idx]

    # S3: прості числа з X через булеву маску
    is_prime = _is_prime_sieve(int(X.max()))
    S3 = np.sort(X[is_prime[X]])

    return S1, S2, S3


def _is_prime_sieve(n: int) -> np.ndarray:
    """Return boolean array is_prime[0..n] using a sieve."""
    if n < 1:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p:n+1:p] = False
    return is_prime


def _tests():
    X = np.arange(64, dtype=int).reshape(8, 8)
    S1, S2, S3 = slice_and_mask(X)

    assert S1.shape == (4, 4)
    old = X[2, 2]
    S1[0, 0] = -999
    assert X[2, 2] == -999, "S1 must be a view"
    X[2, 2] = old

    expected_S2 = np.array([7, 14, 21, 28, 35, 42, 49, 56], dtype=int)
    assert np.array_equal(S2, expected_S2)

    sieve = _is_prime_sieve(int(X.max()))
    expected_S3 = np.sort(X[sieve[X]])
    assert np.array_equal(S3, expected_S3)

    print("All tests passed")


if __name__ == "__main__":
    _tests()
