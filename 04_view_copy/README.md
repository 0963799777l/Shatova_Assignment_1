# Set 4 — View and copy


Views share data with the original array; copies own their own data.

## Tasks
1) Return a C-contiguous float32 array, copying only if needed.  
2) Split `(N, 2D)` into two **views** `(N, D)`.

## Reference docs
- Copies vs views: https://numpy.org/doc/stable/user/basics.copies.html
- `ndarray.flags`: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
- `np.ascontiguousarray`: https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html
