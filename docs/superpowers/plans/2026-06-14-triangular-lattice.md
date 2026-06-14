# Triangular Lattice Target Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--lattice {square,triangular}` CLI argument (default `square`) across all `scripts/`, with centralized triangular-lattice target generation and triangular-aware metrics.

**Architecture:** Lattice geometry lives in `src/foitweezers/targets.py` (mirrors the existing square API). All four scripts route through `scripts/_runner.py` (`add_common_args` → `resolve` → `build_target`), so wiring `--lattice` there propagates everywhere. `metrics.evaluate` gains a `lattice` kwarg dispatching the square vs. triangular `vp_ratio` path.

**Tech Stack:** Python, NumPy, pytest. Tests run from repo root; `tests/` inserts `src/` and `scripts/` onto `sys.path`.

**Spec:** `docs/superpowers/specs/2026-06-14-triangular-lattice-design.md`

**Conventions (match existing code):**
- Run tests from repo root: `cd <repo>` then `pytest ...`.
- Commit message style: Conventional Commits (e.g. `feat(targets): ...`).
- End each commit body with the Co-Authored-By trailer if your harness requires it.

---

### Task 1: Triangular lattice geometry in `targets.py`

**Files:**
- Modify: `src/foitweezers/targets.py`
- Test: `tests/test_triangular.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_triangular.py`:

```python
"""Unit tests for triangular-lattice targets and triangular metrics."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from foitweezers.targets import (
    triangular_lattice_positions,
    triangular_lattice_target,
    n_triangular_sites,
)


@pytest.mark.parametrize("radius,expected", [(0, 1), (1, 7), (2, 19), (3, 37)])
def test_triangular_site_count(radius, expected):
    m = 400
    positions, s = triangular_lattice_positions(m, spacing_fine_px=20, radius=radius)
    assert n_triangular_sites(radius) == expected
    assert len(positions) == expected


def test_triangular_centered_and_deterministic():
    m = 400
    pos_a, s_a = triangular_lattice_positions(m, spacing_fine_px=20, radius=2)
    pos_b, s_b = triangular_lattice_positions(m, spacing_fine_px=20, radius=2)
    assert pos_a == pos_b  # deterministic
    assert s_a == 20
    assert (m // 2, m // 2) in pos_a  # center site present


def test_triangular_spacing_guard():
    with pytest.raises(ValueError):
        triangular_lattice_positions(400, spacing_fine_px=0.4, radius=2)


def test_triangular_target_shape_and_peaks():
    m = 400
    T, positions, s = triangular_lattice_target(m, spacing_fine_px=20, radius=2)
    assert T.shape == (m, m)
    assert float(T.sum()) == pytest.approx(len(positions))
    for r, c in positions:
        assert T[r, c] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_triangular.py -v`
Expected: FAIL — `ImportError: cannot import name 'triangular_lattice_positions'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/foitweezers/targets.py` (after `square_lattice_target`):

```python
def triangular_lattice_positions(m, spacing_fine_px, radius=2):
    """Integer (row, col) pixel positions of a triangular lattice with a hexagonal
    boundary, centered on the grid.

    ``radius`` is the axial-hex shell count; total sites = 1 + 3*radius*(radius+1)
    (radius=2 -> 19, radius=3 -> 37). Spacing is the nearest-neighbour distance,
    rounded to the nearest integer pixel so spots land on grid points.
    """
    s = int(round(spacing_fine_px))
    if s < 1:
        raise ValueError("spacing rounds to < 1 fine pixel; increase oversample.")
    c = m // 2
    positions = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if max(abs(q), abs(r), abs(q + r)) <= radius:
                x = s * (q + 0.5 * r)
                y = s * (np.sqrt(3) / 2.0) * r
                row = int(round(c + y))
                col = int(round(c + x))
                positions.append((row, col))
    return sorted(set(positions)), s


def triangular_lattice_target(m, spacing_fine_px, radius=2, xp=np):
    """Build the triangular-lattice target intensity ``T`` (M x M) and its spot list.

    Same return contract as :func:`square_lattice_target`.
    """
    positions, spacing_int = triangular_lattice_positions(m, spacing_fine_px, radius)
    T = xp.zeros((m, m), dtype=xp.float64)
    for (r, c) in positions:
        T[r, c] = 1.0
    return T, positions, spacing_int


def n_triangular_sites(radius):
    """Total sites in a complete ``radius``-shell hexagonal triangular lattice."""
    return 1 + 3 * radius * (radius + 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_triangular.py -v`
Expected: PASS (all parametrized + 3 others).

- [ ] **Step 5: Commit**

```bash
git add src/foitweezers/targets.py tests/test_triangular.py
git commit -m "feat(targets): triangular lattice with hexagonal boundary"
```

---

### Task 2: Export triangular API from package `__init__`

**Files:**
- Modify: `src/foitweezers/__init__.py`
- Test: `tests/test_triangular.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_triangular.py`:

```python
def test_triangular_exported_from_package():
    import foitweezers
    assert hasattr(foitweezers, "triangular_lattice_target")
    assert hasattr(foitweezers, "triangular_lattice_positions")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_triangular.py::test_triangular_exported_from_package -v`
Expected: FAIL — `AssertionError`.

- [ ] **Step 3: Write minimal implementation**

In `src/foitweezers/__init__.py`, update the targets import and `__all__`. The existing line is:

```python
from .targets import square_lattice_target  # noqa: F401
```

Replace with:

```python
from .targets import (  # noqa: F401
    square_lattice_target,
    triangular_lattice_target,
    triangular_lattice_positions,
)
```

And add `"triangular_lattice_target"` and `"triangular_lattice_positions"` to the
`__all__` list (next to the existing `"square_lattice_target"` entry).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_triangular.py::test_triangular_exported_from_package -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foitweezers/__init__.py tests/test_triangular.py
git commit -m "feat(targets): export triangular lattice API from package"
```

---

### Task 3: Triangular `vp_ratio` + `evaluate` dispatch in `metrics.py`

**Files:**
- Modify: `src/foitweezers/metrics.py`
- Test: `tests/test_triangular.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_triangular.py`:

```python
def test_vp_ratio_triangular_ideal_is_zero():
    from foitweezers.metrics import vp_ratio_triangular
    m = 400
    T, positions, s = triangular_lattice_target(m, spacing_fine_px=20, radius=2)
    # Ideal delta peaks: midpoints between neighbours are empty -> vp == 0.
    assert vp_ratio_triangular(T, positions, s) == pytest.approx(0.0, abs=1e-12)


def test_evaluate_triangular_path_runs():
    from foitweezers.metrics import evaluate
    m = 400
    T, positions, s = triangular_lattice_target(m, spacing_fine_px=20, radius=2)
    met = evaluate(T, positions, s, n_spots=5, lattice="triangular")
    assert met["uniformity"] < 1e-9
    assert met["efficiency"] == pytest.approx(1.0, abs=1e-9)
    assert met["vp_ratio"] == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_triangular.py::test_vp_ratio_triangular_ideal_is_zero tests/test_triangular.py::test_evaluate_triangular_path_runs -v`
Expected: FAIL — `ImportError: cannot import name 'vp_ratio_triangular'`.

- [ ] **Step 3: Write minimal implementation**

In `src/foitweezers/metrics.py`, add `vp_ratio_triangular` after the existing
`vp_ratio` function:

```python
def vp_ratio_triangular(I, positions, spacing_int):
    """Valley-to-peak ratio for a triangular lattice.

    Nearest-neighbour pairs are those at Euclidean distance < 1.5 * spacing_int
    (NN distance is ~spacing_int; the next shell is ~sqrt(3)*spacing_int). Valleys
    are the pair-midpoint intensities; peaks are the site intensities.
    """
    s = int(spacing_int)
    peaks = [float(I[r, c]) for (r, c) in positions]
    peak = float(np.mean(peaks)) if peaks else 0.0
    thr2 = (1.5 * s) ** 2
    valleys = []
    n = len(positions)
    for i in range(n):
        r1, c1 = positions[i]
        for j in range(i + 1, n):
            r2, c2 = positions[j]
            if (r1 - r2) ** 2 + (c1 - c2) ** 2 <= thr2:
                valleys.append(float(I[(r1 + r2) // 2, (c1 + c2) // 2]))
    if peak == 0 or not valleys:
        return np.nan
    return float(np.mean(valleys)) / peak
```

Then update `evaluate` to dispatch on `lattice`. The current function is:

```python
def evaluate(I, positions, spacing_int, n_spots=5, half_window=None):
    """Convenience: return all Table-I / Fig-4 metrics for one reproduced image."""
    if half_window is None:
        half_window = default_half_window(spacing_int)
    p = spot_powers(I, positions, half_window)
    return {
        "uniformity": uniformity(p),
        "efficiency": efficiency(I, positions, half_window),
        "vp_ratio": vp_ratio(I, positions, n_spots),
        "half_window": half_window,
        "spot_powers": p,
    }
```

Replace it with:

```python
def evaluate(I, positions, spacing_int, n_spots=5, half_window=None, lattice="square"):
    """Convenience: return all Table-I / Fig-4 metrics for one reproduced image.

    ``lattice`` selects the valley-to-peak computation: "square" uses the
    axis-aligned N x N grid path; "triangular" uses nearest-neighbour midpoints.
    """
    if half_window is None:
        half_window = default_half_window(spacing_int)
    p = spot_powers(I, positions, half_window)
    if lattice == "triangular":
        vp = vp_ratio_triangular(I, positions, spacing_int)
    else:
        vp = vp_ratio(I, positions, n_spots)
    return {
        "uniformity": uniformity(p),
        "efficiency": efficiency(I, positions, half_window),
        "vp_ratio": vp,
        "half_window": half_window,
        "spot_powers": p,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_triangular.py -v`
Expected: PASS (all triangular tests).

- [ ] **Step 5: Commit**

```bash
git add src/foitweezers/metrics.py tests/test_triangular.py
git commit -m "feat(metrics): triangular nearest-neighbour vp_ratio + evaluate dispatch"
```

---

### Task 4: `--lattice` CLI arg + `build_target` dispatch in `_runner.py`

**Files:**
- Modify: `scripts/_runner.py`
- Test: `tests/test_triangular.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_triangular.py`:

```python
import argparse

from scripts._runner import add_common_args, resolve, build_target


def _resolve_cli(*cli):
    args = add_common_args(argparse.ArgumentParser()).parse_args(list(cli))
    return resolve(args)


def test_lattice_defaults_to_square():
    cfg = _resolve_cli("--preset", "tiny")
    assert cfg["lattice"] == "square"
    T, pos, sint = build_target(cfg, spacing_coarse=2.0)
    assert len(pos) == cfg["n_spots"] ** 2  # 5x5 = 25 by default


def test_build_target_triangular_19_sites():
    cfg = _resolve_cli("--preset", "tiny", "--lattice", "triangular", "--n-spots", "5")
    assert cfg["lattice"] == "triangular"
    T, pos, sint = build_target(cfg, spacing_coarse=2.0)
    assert len(pos) == 19  # radius = 5 // 2 = 2


def test_build_target_triangular_rejects_even_n_spots():
    cfg = _resolve_cli("--preset", "tiny", "--lattice", "triangular", "--n-spots", "4")
    with pytest.raises(ValueError, match="odd"):
        build_target(cfg, spacing_coarse=2.0)


def test_build_target_rejects_zero_n_spots():
    cfg = _resolve_cli("--preset", "tiny", "--n-spots", "0")
    with pytest.raises(ValueError):
        build_target(cfg, spacing_coarse=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_triangular.py -k "lattice or build_target" -v`
Expected: FAIL — `KeyError: 'lattice'` (resolve does not set it yet).

- [ ] **Step 3: Write minimal implementation**

In `scripts/_runner.py`:

1. Update the targets import. The current line is:

```python
from foitweezers.targets import square_lattice_target  # noqa: E402
```

Replace with:

```python
from foitweezers.targets import (  # noqa: E402
    square_lattice_target,
    triangular_lattice_target,
)
```

2. In `add_common_args`, add the `--lattice` argument and update the `--n-spots`
help. The current line is:

```python
    p.add_argument("--n-spots", type=int, default=5, help="array side length (N x N)")
```

Replace with:

```python
    p.add_argument(
        "--lattice", choices=["square", "triangular"], default="square",
        help="target lattice geometry (default: square)",
    )
    p.add_argument(
        "--n-spots", type=int, default=5,
        help="square: array side length (N x N); "
             "triangular: must be odd, radius = n_spots // 2 "
             "(5 -> 19 sites, 7 -> 37 sites)",
    )
```

3. In `resolve`, set the lattice on the config. After the existing line
`cfg["n_spots"] = args.n_spots`, add:

```python
    cfg["lattice"] = args.lattice
```

4. Replace `build_target` with the dispatching version. The current function is:

```python
def build_target(cfg, spacing_coarse, n_spots=None):
    m = cfg["n"] * cfg["oversample"]
    if n_spots is None:
        n_spots = cfg.get("n_spots", 5)
    T, pos, sint = square_lattice_target(
        m, spacing_fine_px=spacing_coarse * cfg["oversample"], n_spots=n_spots
    )
    return T, pos, sint
```

Replace with:

```python
def build_target(cfg, spacing_coarse, n_spots=None):
    m = cfg["n"] * cfg["oversample"]
    if n_spots is None:
        n_spots = cfg.get("n_spots", 5)
    if n_spots < 1:
        raise ValueError(f"--n-spots must be >= 1; got {n_spots}")
    spacing_fine = spacing_coarse * cfg["oversample"]
    lattice = cfg.get("lattice", "square")
    if lattice == "triangular":
        if n_spots % 2 == 0:
            raise ValueError(
                f"triangular lattice requires an odd --n-spots "
                f"(1, 3, 5, ...); got {n_spots}"
            )
        radius = n_spots // 2
        T, pos, sint = triangular_lattice_target(
            m, spacing_fine_px=spacing_fine, radius=radius
        )
    else:
        T, pos, sint = square_lattice_target(
            m, spacing_fine_px=spacing_fine, n_spots=n_spots
        )
    return T, pos, sint
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_triangular.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add scripts/_runner.py tests/test_triangular.py
git commit -m "feat(runner): --lattice arg with triangular build_target dispatch"
```

---

### Task 5: Pass `lattice` into `evaluate` at all call sites + fix labels

**Files:**
- Modify: `scripts/fig3.py`, `scripts/optimize_mask.py`, `scripts/table1.py`

> **Labeling (grill decision Q1):** `--n-spots` has a dual meaning, so square-only
> labels mislabel triangular runs. `fig3.py` suptitle is hardcoded `{ns}x{ns}` and
> `optimize_mask` reports only `n_spots` (a `--n-spots 5` triangular run is actually
> 19 sites). Fix both below. `table1.py` has **no** lattice label (its `label` is the
> aberration case), so it needs only the `evaluate` change.

- [ ] **Step 1: Update `scripts/fig3.py` — `evaluate` call + suptitle**

(a) Find the `evaluate(...)` call (currently `met = evaluate(I, pos, sint, n_spots=ns)`)
and add `lattice=cfg["lattice"]`:

```python
            met = evaluate(I, pos, sint, n_spots=ns, lattice=cfg["lattice"])
```

(b) Fix the hardcoded square suptitle (currently
`fig.suptitle(f"Fig 3: numerically reproduced {ns}x{ns} arrays (top FOI, bottom RSS)")`).
`pos` from the last `build_target` is in scope; branch on the lattice:

```python
    if cfg["lattice"] == "triangular":
        array_label = f"{len(pos)}-site triangular (R={ns // 2})"
    else:
        array_label = f"{ns}x{ns} square"
    fig.suptitle(f"Fig 3: numerically reproduced {array_label} arrays (top FOI, bottom RSS)")
```

- [ ] **Step 2: Update `scripts/optimize_mask.py` — `evaluate` calls + `n_sites` in meta**

(a) There are two `evaluate(...)` calls (currently
`met = evaluate(I, pos, sint, n_spots=cfg["n_spots"])`). Update **both** to:

```python
    met = evaluate(I, pos, sint, n_spots=cfg["n_spots"], lattice=cfg["lattice"])
```

(b) Add the actual site count to the reported results so a triangular run is
self-describing (it reports `n_spots=5` but has 19 sites). In the `results` dict
written to `{stem}_meta.json` (the block around `"n_spots": ...`), add the real
count from the built target (`pos`/`T` from `build_target` at optimize_mask.py:221
is in scope):

```python
        "n_sites": len(pos),
```

Keep the existing `n_spots` field unchanged (it records the CLI arg).

- [ ] **Step 3: Update `scripts/table1.py`**

Find the `evaluate(...)` call (currently `met = evaluate(I, pos, sint, n_spots=ns)`)
and update to:

```python
                met = evaluate(I, pos, sint, n_spots=ns, lattice=cfg["lattice"])
```

- [ ] **Step 4: Verify nothing broke**

Run: `pytest tests/ -q`
Expected: PASS (full suite, including existing `test_core.py`).

- [ ] **Step 5: Commit**

```bash
git add scripts/fig3.py scripts/optimize_mask.py scripts/table1.py
git commit -m "feat(scripts): pass lattice into evaluate at all call sites"
```

---

### Task 6: CLI end-to-end test for `--lattice triangular`

**Files:**
- Test: `tests/test_triangular.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_triangular.py`:

```python
def test_optimize_mask_triangular_cli_end_to_end(tmp_path):
    import json
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..")
    subprocess.run(
        [sys.executable, "scripts/optimize_mask.py", "--method", "FOI",
         "--preset", "tiny", "--seeds", "1", "--iters", "15",
         "--lattice", "triangular", "--n-spots", "5", "--tag", "tri1"],
        cwd=repo, check=True,
    )
    stem = os.path.join(repo, "outputs", "optmask_FOI_sp1.8_tri1")
    assert os.path.exists(stem + "_meta.json")
    meta = json.load(open(stem + "_meta.json"))["results"]
    assert meta["n_spots"] == 5
    assert meta["n_sites"] == 19  # real triangular discriminator (Task 5b)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_triangular.py::test_optimize_mask_triangular_cli_end_to_end -v`
Expected: PASS (the subprocess runs the full triangular pipeline end-to-end).

Note: this is a no-new-production-code test — it validates the wiring from Tasks 4–5
through the real CLI. If it fails, the failure points to a wiring gap to fix in those
tasks, not new code here.

- [ ] **Step 3: Commit**

```bash
git add tests/test_triangular.py
git commit -m "test: triangular lattice CLI end-to-end"
```

---

### Task 7: Final regression + docs

**Files:**
- Modify: `README.md` (if it documents CLI flags — check first)

- [ ] **Step 1: Full suite green**

Run: `pytest tests/ -q`
Expected: PASS, no regressions in `test_core.py` / `test_chained.py`.

- [ ] **Step 2: Document the flag (if applicable)**

Check whether `README.md` lists CLI flags (`grep -n "n-spots\|--spacing\|--method" README.md`).
If it documents flags, add a short `--lattice {square,triangular}` entry describing
the dual `--n-spots` meaning and the odd-only triangular constraint, matching the
surrounding doc style. If `README.md` does not document flags, skip this step.

- [ ] **Step 3: Commit (only if README changed)**

```bash
git add README.md
git commit -m "docs(readme): document --lattice square/triangular"
```

---

## Self-Review Notes

- **Spec coverage:** targets (Task 1), exports (Task 2), metrics/vp dispatch (Task 3),
  CLI arg + build_target dispatch + odd validation (Task 4), call sites (Task 5),
  CLI e2e (Task 6), regression + docs (Task 7). All spec sections mapped.
- **Square regression:** Task 4 `test_lattice_defaults_to_square` asserts 25 sites
  for default; Task 5/7 run the full existing suite.
- **Labeling (grill Q1):** Task 5 branches the `fig3.py` suptitle on lattice and adds
  `n_sites=len(pos)` to `optimize_mask` meta — triangular runs no longer mislabel as
  `NxN` / under-report site count. `table1.py` has no lattice label (its `label` is the
  aberration case), so no change there. Task 6 asserts `meta["n_sites"]==19` so the
  e2e test actually discriminates triangular from square (the `n_spots==5` assert
  alone cannot).
- **Validation:** even-`n_spots` and `n_spots < 1` covered in Task 4.
- **Naming consistency:** `triangular_lattice_positions`, `triangular_lattice_target`,
  `n_triangular_sites`, `vp_ratio_triangular`, `cfg["lattice"]`, `evaluate(..., lattice=)`
  used identically across all tasks.
