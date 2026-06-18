# Set 4 — View and copy


Views share data with the original array; copies own their own data.

## Tasks
1) Return a C-contiguous float32 array, copying only if needed.  
2) Split `(N, 2D)` into two **views** `(N, D)`.

## Reference docs
- Copies vs views: https://numpy.org/doc/stable/user/basics.copies.html
- `ndarray.flags`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
- `np.ascontiguousarray`: https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html


## Опис виконання українською

У цьому завданні реалізовано роботу з представленнями та копіями масивів NumPy.

У функції `as_contiguous_float32` перевіряється, чи масив уже має тип `float32` і C-contiguous порядок зберігання. Якщо ці умови виконуються, масив повертається без копіювання. Якщо ні — створюється новий C-contiguous масив з типом `float32`.

У функції `split_view` масив форми `(N, 2D)` розділено на дві частини: перші `D` стовпців і останні `D` стовпців. Для цього використано зрізи, які повертають представлення вихідного масиву без створення копій.

Коментарі в коді додано українською мовою для пояснення основних етапів виконання.
