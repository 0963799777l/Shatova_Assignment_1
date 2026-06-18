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
    # Створюємо структурований масив.
    # Кожен запис містить поля: id, x, y та label.
    points = np.array([
        (10,  0.0,  0.0, "A"),
        (11,  1.5, -0.5, "B"),
        (12, -2.0,  2.0, "A"),
        (13,  3.0,  1.0, "C"),
        (14,  1.0, -2.0, "B"),
        (15, -1.0,  1.0, "A"),
    ], dtype=DTYPE)

    return points


def group_centroids(points: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return dict: label -> centroid [mean_x, mean_y].

    Requirements:
    - No Python loops over points.
    - You may loop over unique labels.
    """
    # Знаходимо всі унікальні мітки класів.
    labels = np.unique(points["label"])

    # Створюємо словник, де ключ — мітка, а значення — центроїд [mean_x, mean_y].
    centroids = {}

    # Цикл по унікальних мітках дозволений умовою завдання.
    for label in labels:
        # Створюємо булеву маску для вибору точок з поточною міткою.
        mask = points["label"] == label

        # Обчислюємо середнє значення координат x та y для цієї групи.
        mean_x = points["x"][mask].mean()
        mean_y = points["y"][mask].mean()

        # Записуємо центроїд у словник.
        centroids[label] = np.array([mean_x, mean_y])

    return centroids


def _tests():
    pts = build_points()
    assert pts.dtype == DTYPE and pts.shape == (6,)
    cent = group_centroids(pts)
    assert np.allclose(cent["A"], np.array([-1.0, 1.0]))
    assert np.allclose(cent["B"], np.array([1.25, -1.25]))
    assert np.allclose(cent["C"], np.array([3.0, 1.0]))
    print("All tests passed")


if __name__ == "__main__":
    _tests()
