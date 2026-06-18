"""
Exercise Set 6 — Vectorization (easy → hard)

Instructions:
- Fill in the TODO sections.
- Run: python exercise.py
"""
import numpy as np


def rowwise_softmax(X: np.ndarray) -> np.ndarray:
    """Stable softmax over axis=1 for X shape (N,D)."""
    # Для числової стабільності віднімаємо максимум у кожному рядку.
    # keepdims=True залишає форму (N, 1), щоб далі працював broadcasting.
    X_shifted = X - np.max(X, axis=1, keepdims=True)

    # Обчислюємо експоненти для всіх елементів.
    exp_X = np.exp(X_shifted)

    # Ділимо кожен рядок на суму експонент у цьому рядку.
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


def rolling_mean_1d(x: np.ndarray, k: int) -> np.ndarray:
    """Rolling mean, output length len(x)-k+1. Use cumsum trick."""
    # Використовуємо накопичувальну суму для швидкого обчислення сум у вікнах.
    # На початок додаємо 0, щоб зручно рахувати різницю сум.
    cumsum = np.cumsum(np.insert(x, 0, 0.0))

    # Сума кожного вікна довжини k дорівнює різниці двох елементів cumsum.
    window_sums = cumsum[k:] - cumsum[:-k]

    # Середнє значення — це сума вікна, поділена на k.
    return window_sums / k


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode labels shape (N,) -> (N,C)."""
    # Створюємо нульову матрицю розміру N x C.
    out = np.zeros((labels.shape[0], num_classes), dtype=int)

    # За допомогою fancy indexing ставимо 1 у відповідний клас для кожної мітки.
    out[np.arange(labels.shape[0]), labels] = 1

    return out


def pairwise_cosine_similarity(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity for all row pairs, returns (N,M)."""
    # Обчислюємо скалярні добутки між усіма рядками A та B.
    dot_products = A @ B.T

    # Обчислюємо L2-норми рядків A і B.
    norm_A = np.sqrt(np.sum(A * A, axis=1, keepdims=True))
    norm_B = np.sqrt(np.sum(B * B, axis=1, keepdims=True))

    # Завдяки broadcasting отримуємо знаменник для кожної пари рядків.
    denominator = norm_A @ norm_B.T

    # Додаємо eps, щоб уникнути ділення на нуль.
    return dot_products / (denominator + eps)


def segment_sum(values: np.ndarray, segment_ids: np.ndarray, num_segments: int) -> np.ndarray:
    """Per-segment sums, like a tiny groupby-sum."""
    # Створюємо масив для сум кожного сегмента.
    out = np.zeros(num_segments, dtype=values.dtype)

    # np.add.at додає значення у відповідні позиції за segment_ids.
    # Це аналог групування і сумування.
    np.add.at(out, segment_ids, values)

    return out


def conv2d_valid(X: np.ndarray, K: np.ndarray) -> np.ndarray:
    """VALID 2D convolution using sliding_window_view (no output loops)."""
    # Отримуємо розміри ядра згортки.
    kh, kw = K.shape

    # Створюємо всі ковзні вікна розміру kh x kw у матриці X.
    # Для X форми (H, W) результат матиме форму (H-kh+1, W-kw+1, kh, kw).
    windows = np.lib.stride_tricks.sliding_window_view(X, (kh, kw))

    # Множимо кожне вікно на ядро K і сумуємо по останніх двох осях.
    # Отримуємо результат valid-згортки без явних циклів.
    return np.sum(windows * K, axis=(2, 3))


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

    print("All tests passed")


if __name__ == "__main__":
    _tests()
