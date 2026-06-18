"""
Exercise Set 1 — Array creation examples

Instructions:
- Fill in the TODO sections.
- Run: python exercise.py
"""
import numpy as np


def make_matrices(seed: int = 0) -> dict[str, np.ndarray]:
    """
    Return a dict of arrays with exact shapes/dtypes/values:
      A: shape (3, 4) float64, values 0..11 row-major
      B: shape (4, 3) int32, values 1..12 column-major (Fortran order)
      C: shape (5,) complex128, values [0+0j, 1+1j, 2+2j, 3+3j, 4+4j]
      D: shape (2, 3, 4) uint8, random integers in [0, 256)
      E: shape (6,) float32, 6 linearly spaced values from -1 to 1 inclusive
      F: shape (4, 4) bool, True on the main diagonal, False elsewhere
    """
    rng = np.random.default_rng(seed)

    # TODO: create A
    A = None

    # TODO: create B (int32) in Fortran order. Tip: reshape(..., order="F")
    B = None

    # TODO: create C
    C = None

    # TODO: create D using rng.integers(..., dtype=np.uint8)
    D = None

    # TODO: create E
    E = None

    # TODO: create F
    F = None

    return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}


def _tests():
    out = make_matrices(seed=123)
    A, B, C, D, E, F = (out[k] for k in ["A", "B", "C", "D", "E", "F"])

    assert A.shape == (3, 4) and A.dtype == np.float64
    assert np.array_equal(A, np.arange(12, dtype=np.float64).reshape(3, 4))

    assert B.shape == (4, 3) and B.dtype == np.int32
    expected_B = np.arange(1, 13, dtype=np.int32).reshape(4, 3, order="F")
    assert np.array_equal(B, expected_B)
    assert B.flags["F_CONTIGUOUS"], "B must be Fortran-contiguous (column-major)"

    assert C.shape == (5,) and C.dtype == np.complex128
    assert np.array_equal(C, (np.arange(5) + 1j*np.arange(5)).astype(np.complex128))

    assert D.shape == (2, 3, 4) and D.dtype == np.uint8
    assert D.min() >= 0 and D.max() <= 255

    assert E.shape == (6,) and E.dtype == np.float32
    assert np.allclose(E, np.linspace(-1, 1, 6, dtype=np.float32))

    assert F.shape == (4, 4) and F.dtype == np.bool_
    assert np.array_equal(F, np.eye(4, dtype=bool))

    print("All tests passed ✅")


if __name__ == "__main__":
    _tests()
