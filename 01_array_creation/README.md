# Set 1 — Array creation examples


You will practice creating NumPy arrays with specific **shapes**, **dtypes**, and **memory layouts**.

## Files
- `exercise.py` — implement `make_matrices`
- Run: `python exercise.py`

## What you’ll learn
- Creating ranges and reshaping (`arange`, `reshape`)
- Choosing dtypes (`dtype=...`)
- Fortran (column-major) vs C (row-major) memory order
- Random integer generation with `default_rng().integers`
- Common constructors (`linspace`, `eye`)

## Tasks
Implement `make_matrices()` to return arrays **A..F** exactly as specified in the docstring.

## Hints
- Use `np.arange(n, dtype=...)` then `.reshape(...)`.
- For Fortran order: `reshape(..., order="F")`.
- Complex values can be created with `np.arange(...) + 1j*np.arange(...)`.
- RNG: `rng = np.random.default_rng(seed)` then `rng.integers(low, high, size=..., dtype=...)`.

## Reference docs (NumPy)
- `np.arange`: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
- `np.reshape`: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
- `np.random.default_rng`: https://numpy.org/doc/stable/reference/random/generator.html
- `Generator.integers`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
- `np.linspace`: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
- `np.eye`: https://numpy.org/doc/stable/reference/generated/numpy.eye.html
