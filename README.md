# NumPy Exercise Packs (6 sets)

Each folder contains:
- `exercise.py`
- `README.md`

Run any set:
```bash
python exercise.py
```

Sets:
- `01_array_creation` — Array creation examples
- `02_indexing` — Array indexing
- `03_broadcasting` — Broadcasting
- `04_view_copy` — View and copy
- `05_structured_arrays` — Structured arrays
- `06_vectorization` — Vectorization (easy → hard)

Опис виконання українською

У цьому завданні реалізовано шість векторизованих операцій NumPy без використання явних циклів для основних обчислень.

У функції `rowwise_softmax` обчислено стабільний softmax по рядках із попереднім відніманням максимального значення в кожному рядку.

У функції `rolling_mean_1d` реалізовано ковзне середнє за допомогою накопичувальної суми `np.cumsum`, що дозволяє швидко обчислювати середні значення для вікон заданої довжини.

У функції `one_hot` виконано one-hot кодування міток класів за допомогою fancy indexing.

У функції `pairwise_cosine_similarity` обчислено косинусну схожість між усіма парами рядків двох матриць з використанням матричного множення та broadcasting.

У функції `segment_sum` реалізовано сумування значень за сегментами за допомогою `np.add.at`.

У функції `conv2d_valid` виконано valid-згортку двовимірного масиву з використанням `sliding_window_view` і подальшим векторизованим множенням та сумуванням.

Коментарі в коді додано українською мовою для пояснення основних етапів виконання.
