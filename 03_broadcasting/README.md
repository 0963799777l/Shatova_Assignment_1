# Set 3 — Broadcasting


Broadcasting lets NumPy “stretch” arrays of different shapes to make elementwise operations work without copying.

## Tasks
1) Row-normalize using `keepdims=True`  
2) Pairwise squared distances using `A[:, None, :] - B[None, :, :]`

## Reference docs
- Broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
- `np.sum`: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
- `np.newaxis`: https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis


## Опис виконання українською

У цьому завданні реалізовано використання broadcasting у NumPy для роботи з масивами різних форм.

У функції `normalize_rows` виконано нормалізацію рядків матриці за L2-нормою. Для цього обчислюється норма кожного рядка з параметром `keepdims=True`, що дозволяє зберегти правильну форму масиву для подальшого ділення через broadcasting.

У функції `pairwise_sq_dists` обчислено попарні квадрати евклідових відстаней між рядками двох матриць. Для цього використано додаткові осі `A[:, None, :]` та `B[None, :, :]`, що дозволяє отримати різниці для всіх пар рядків без явних циклів.

Коментарі в коді додано українською мовою для пояснення основних етапів виконання.
