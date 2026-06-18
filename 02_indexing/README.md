# Set 2 — Array indexing


This set focuses on slicing, fancy indexing, and boolean masking.

## Files
- `exercise.py` — implement `slice_and_mask`

## Tasks
Given `X` (shape `(8, 8)`), return:

1) **S1** — center 4×4 block as a **view**  
2) **S2** — anti-diagonal values: `(0,7), (1,6), …, (7,0)`  
3) **S3** — all **prime** values from `X`, using boolean masking (no loops)

## Hints
- Center slice: `X[2:6, 2:6]` (view).
- Anti-diagonal: use `np.arange(n)` and `n-1-np.arange(n)`.
- Primes: build `is_prime[0..X.max()]`, then mask with `is_prime[X]`.

## Reference docs (NumPy)
- Indexing overview: https://numpy.org/doc/stable/user/basics.indexing.html
- Boolean indexing: https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing
- Advanced indexing: https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
