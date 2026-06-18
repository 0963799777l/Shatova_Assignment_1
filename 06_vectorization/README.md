# Set 6 — Vectorization (easy → hard)


6 vectorization tasks, from “classic interview” to “mini-numerical kernel”.

## Tasks
1) Stable rowwise softmax  
2) Rolling mean via `cumsum`  
3) One-hot encoding via fancy indexing  
4) Pairwise cosine similarity via matrix multiply + broadcasting  
5) Segment sum via `np.add.at` (or `bincount`)  
6) Valid 2D convolution via `sliding_window_view` + reduction

## Reference docs
- `np.exp`: https://numpy.org/doc/stable/reference/generated/numpy.exp.html
- `np.cumsum`: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
- `np.add.at`: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
- `np.bincount`: https://numpy.org/doc/stable/reference/generated/numpy.bincount.html
- `sliding_window_view`: https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
- `np.einsum`: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
