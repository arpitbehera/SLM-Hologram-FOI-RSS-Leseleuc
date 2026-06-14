# Triangular Lattice Target Support — Design

**Date:** 2026-06-14
**Branch:** `feat/triangular-lattice`

## Goal

Add a `--lattice {square,triangular}` CLI argument across all `scripts/`, selecting
the target lattice geometry. `square` stays the default so every existing workflow
and command line behaves identically. `triangular` generates atom/tweezer positions
on a triangular lattice with a hexagonal boundary, following the conventions already
established in `src/foitweezers/targets.py`.

## Constraints / Decisions

1. **Default `square`.** No behavior change for any existing invocation. Square
   target generation, metrics, and outputs stay byte-identical.
2. **`--n-spots` is reused, not replaced.** Dual meaning:
   - `square`: side length `N` of an `N x N` array (existing semantics).
   - `triangular`: `radius = n_spots // 2` in the axial-hex sense of the reference
     code. So `--n-spots 5 -> radius 2 -> 19 sites`. `--n-spots 7 -> radius 3 -> 37`.
   - Validation: `n_spots >= 1`, else error. No magic-number rounding/erroring —
     the `// 2` mapping is total and deterministic.
3. **`--spacing` / `--spacings` unchanged.** Meaning stays "nearest-neighbour
   distance in coarse target-plane pixels" for both lattices. The reference code's
   `s` is the NN distance, consistent with the square `spacing_fine_px`.
4. **`vp_ratio` is square-specific** (it reshapes the flat position list into an
   `n_spots x n_spots` grid with axis-aligned neighbours). The square path is left
   untouched. Triangular gets a **separate** NN-midpoint `vp_ratio` path; no shared
   reshape.
5. **Centralized lattice generation.** All geometry lives in `targets.py`; the
   `n_spots -> radius` mapping lives at the single `build_target` call site. No
   per-script duplication.

## Architecture

All four scripts (`fig1b`, `fig3`, `optimize_mask`, `table1`) route through
`scripts/_runner.py`: `add_common_args` -> `resolve` -> `build_target`. Wiring
`--lattice` there propagates it everywhere automatically.

### 1. `src/foitweezers/targets.py`

Add, mirroring the existing square API (same return contract):

- `triangular_lattice_positions(m, spacing_fine_px, radius=2) -> (positions, s)`
  - `s = int(round(spacing_fine_px))`; guard `s < 1` -> `ValueError` (same message
    style as square).
  - `c = m // 2`.
  - For `q, r in [-radius, radius]`, keep where `max(|q|, |r|, |q+r|) <= radius`.
  - `x = s * (q + 0.5*r)`, `y = s * (sqrt(3)/2) * r`.
  - `row = int(round(c + y))`, `col = int(round(c + x))`.
  - Return `sorted(set(positions)), s`.
- `triangular_lattice_target(m, spacing_fine_px, radius=2, xp=np) -> (T, positions, spacing_int)`
  - Build `T` of zeros, set `T[r, c] = 1.0` at each position. Same contract as
    `square_lattice_target`.
- `n_triangular_sites(radius) -> 1 + 3*radius*(radius + 1)` (reporting / tests).

### 2. `scripts/_runner.py`

- `add_common_args`: add
  `--lattice`, `choices=["square","triangular"]`, `default="square"`.
  Update `--n-spots` help text to document the dual meaning.
- `resolve`: `cfg["lattice"] = args.lattice`.
- `build_target(cfg, spacing_coarse, n_spots=None)`: dispatch on `cfg["lattice"]`.
  - `n_spots < 1` -> `ValueError`.
  - `square`: unchanged call to `square_lattice_target`.
  - `triangular`: `radius = n_spots // 2`; call `triangular_lattice_target`.
  - Returns the same `(T, pos, sint)` 3-tuple in both cases -> zero caller breakage.

### 3. `src/foitweezers/metrics.py`

- `vp_ratio` (square, `n_spots x n_spots` reshape): **untouched**.
- New `vp_ratio_triangular(I, positions, spacing_int)`:
  - NN pairs identified by Euclidean distance `< 1.5 * spacing_int`.
  - peak = mean site intensity; valley = mean of pair-midpoint intensities.
  - Returns `valley / peak` (`NaN` if peak == 0). No grid reshape.
- `evaluate(I, positions, spacing_int, n_spots=5, half_window=None, lattice="square")`:
  - New `lattice` kwarg defaults to `"square"` -> existing call sites unaffected.
  - Dispatch the vp computation on `lattice`; `uniformity` and `efficiency`
    unchanged for both.

### 4. Call sites

- `fig3.py`, `optimize_mask.py`, `table1.py`: pass `lattice=cfg["lattice"]` into
  `evaluate(...)`.
- `fig1b.py`: no `evaluate` call; only needs the `build_target` dispatch (already
  routed via `_runner`). No direct change beyond inheriting `--lattice`.

### 5. `src/foitweezers/__init__.py`

- Export `triangular_lattice_target` and `triangular_lattice_positions` alongside
  the square versions.

## Testing (`tests/test_triangular.py`)

- **positions:** `len(triangular_lattice_positions(...)[0]) == n_triangular_sites(R)`
  for `R = 0, 1, 2, 3`; centered on `m // 2`; deterministic from spacing.
- **build_target dispatch:** square unchanged; `triangular` with `--n-spots 5`
  yields 19 sites (`radius 2`).
- **evaluate triangular:** vp path runs without the square reshape; an ideal target
  gives `vp_ratio approx 0`.
- **CLI end-to-end:** subprocess run with `--lattice triangular` (mirror existing
  `test_optimize_mask_cli_end_to_end`).
- **square regression:** existing square behavior unchanged (covered by current
  `test_core.py`; add an explicit assert that `build_target` default == square).
- **validation:** `n_spots < 1` raises.

## Out of scope

- Kagome / other Fig-5 lattices (future phase).
- Changing spacing semantics or physical-unit conversions.
- Refactoring the square `vp_ratio` grid logic.
