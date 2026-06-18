"""
Exercise Set 5 — Structured arrays

Instructions:
- Fill in the TODO sections.
- Run: python exercise.py
"""
import numpy as np


DTYPE = np.dtype([
    ("id", np.int32),
    ("x",  np.float64),
    ("y",  np.float64),
    ("label", "U10"),
])

def build_points() -> np.ndarray:
    """
    Build and return a structured array with dtype DTYPE and shape (6,).

    Data:
      id:    [10, 11, 12, 13, 14, 15]
      x:     [0.0, 1.5, -2.0, 3.0, 1.0, -1.0]
      y:     [0.0, -0.5, 2.0, 1.0, -2.0, 1.0]
      label: ["A", "B", "A", "C", "B", "A"]
    """
    # TODO
    return None


def group_centroids(points: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return dict: label -> centroid [mean_x, mean_y].

    Requirements:
    - No Python loops over points.
    - You may loop over unique labels.
    """
    # TODO
    return None


def _tests():
    pts = build_points()
    assert pts.dtype == DTYPE and pts.shape == (6,)
    cent = group_centroids(pts)
    assert np.allclose(cent["A"], np.array([-1.0, 1.0]))
    assert np.allclose(cent["B"], np.array([1.25, -1.25]))
    assert np.allclose(cent["C"], np.array([3.0, 1.0]))
    print("All tests passed ✅")


if __name__ == "__main__":
    _tests()
