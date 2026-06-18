# Set 5 — Structured arrays


Structured arrays store record-like data inside a NumPy array.

## Tasks
1) Create a structured array `DTYPE` and fill all fields.  
2) Compute centroids per label using boolean masks.

## Reference docs
- Structured arrays: https://numpy.org/doc/stable/user/basics.rec.html
- Dtypes: https://numpy.org/doc/stable/reference/arrays.dtypes.html
- `np.unique`: https://numpy.org/doc/stable/reference/generated/numpy.unique.html


## Опис виконання українською

У цьому завданні реалізовано роботу зі структурованими масивами NumPy.

У функції `build_points` створено структурований масив із полями `id`, `x`, `y` та `label`. Кожен елемент такого масиву є записом, що містить ідентифікатор точки, її координати та мітку групи.

У функції `group_centroids` обчислено центроїди для кожної унікальної мітки. Для цього використано `np.unique` для отримання списку міток і булеві маски для вибору точок певної групи. Для кожної групи обчислено середні значення координат `x` та `y`.

Коментарі в коді додано українською мовою для пояснення основних етапів виконання.
