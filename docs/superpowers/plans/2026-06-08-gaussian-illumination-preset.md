# Gaussian Illumination Preset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable Gaussian illumination preset for the script pipeline and make it the default illumination without changing the existing `tiny`, `cpu`, or `paper` runtime presets.

**Architecture:** Keep grid/runtime presets separate from illumination presets. Reuse `foitweezers.forward.make_aperture`, which already supports `profile="gaussian"`, and centralize aperture construction in `scripts/_runner.py` so every figure/table script gets the same default and the old flat-top path remains selectable.

**Tech Stack:** Python, NumPy, SciPy, argparse, pytest.

---

## File Map

- Modify `scripts/_runner.py`: add illumination preset config, CLI choice, `make_illumination()` helper, and resolved config fields.
- Modify `scripts/fig1b.py`: use `make_illumination(cfg)` instead of direct flat-top aperture construction.
- Modify `scripts/fig3.py`: use `make_illumination(cfg)` instead of direct flat-top aperture construction.
- Modify `scripts/table1.py`: use `make_illumination(cfg)` for design/reproduction and keep aberration radius tied to aperture radius.
- Modify `tests/test_core.py`: add focused tests for Gaussian aperture normalization and runner preset resolution/default behavior.
- Modify `README.md`: document illumination presets, Gaussian default, and flat-top compatibility flag.

## Task 1: Runner Preset Tests

**Files:**
- Modify: `tests/test_core.py`

- [ ] **Step 1: Add failing tests**

Add imports:

```python
from scripts._runner import DEFAULT_ILLUMINATION, resolve, make_illumination, APERTURE_FRAC
```

Add tests:

```python
def test_gaussian_aperture_is_l2_normalized_and_truncated():
    n = 40
    radius = APERTURE_FRAC * n
    amp = make_aperture(n, radius_px=radius, profile="gaussian", gauss_radius_px=radius)
    c = (n - 1) / 2.0
    y, x = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    outside = x * x + y * y > radius ** 2

    assert np.isclose(np.sum(amp ** 2), 1.0)
    assert np.all(amp[outside] == 0.0)
    assert amp[n // 2, n // 2] > amp[n // 2, int(n // 2 + radius * 0.8)]


def test_resolve_defaults_to_gaussian_illumination():
    parser = add_common_args(argparse.ArgumentParser())
    cfg = resolve(parser.parse_args([]))

    assert DEFAULT_ILLUMINATION == "gaussian"
    assert cfg["illumination"] == "gaussian"
    assert cfg["illumination_profile"] == "gaussian"
    assert cfg["gauss_radius_px"] == pytest.approx(APERTURE_FRAC * cfg["n"])


def test_tophat_illumination_remains_selectable():
    parser = add_common_args(argparse.ArgumentParser())
    cfg = resolve(parser.parse_args(["--illumination", "tophat"]))
    amp = make_illumination(cfg)

    assert cfg["illumination"] == "tophat"
    assert cfg["illumination_profile"] == "tophat"
    assert cfg["gauss_radius_px"] is None
    assert np.isclose(np.sum(amp ** 2), 1.0)
    assert len(np.unique(amp[amp > 0])) == 1
```

- [ ] **Step 2: Run RED**

Run:

```bash
rtk pytest tests/test_core.py::test_resolve_defaults_to_gaussian_illumination tests/test_core.py::test_tophat_illumination_remains_selectable -q
```

Expected: import failure or missing CLI/helper failure because `DEFAULT_ILLUMINATION`, `make_illumination`, or `--illumination` does not exist yet.

## Task 2: Runner Implementation

**Files:**
- Modify: `scripts/_runner.py`
- Modify: `scripts/fig1b.py`
- Modify: `scripts/fig3.py`
- Modify: `scripts/table1.py`

- [ ] **Step 1: Add illumination constants and helper in `scripts/_runner.py`**

Add after `APERTURE_FRAC = 0.45`:

```python
ILLUMINATION_PRESETS = {
    "gaussian": {"profile": "gaussian"},
    "tophat": {"profile": "tophat"},
}
DEFAULT_ILLUMINATION = "gaussian"
```

Add to `add_common_args`:

```python
p.add_argument(
    "--illumination",
    choices=list(ILLUMINATION_PRESETS),
    default=DEFAULT_ILLUMINATION,
    help="nearfield illumination amplitude preset (default: gaussian)",
)
```

Add to `resolve(args)` after `cfg["aperture_radius_px"]`:

```python
illum = ILLUMINATION_PRESETS[args.illumination]
cfg["illumination"] = args.illumination
cfg["illumination_profile"] = illum["profile"]
cfg["gauss_radius_px"] = (
    cfg["aperture_radius_px"] if illum["profile"] == "gaussian" else None
)
```

Add helper:

```python
def make_illumination(cfg):
    """Build the resolved nearfield illumination amplitude."""
    return make_aperture(
        cfg["n"],
        radius_px=cfg["aperture_radius_px"],
        profile=cfg["illumination_profile"],
        gauss_radius_px=cfg["gauss_radius_px"],
    )
```

Update `design_and_reproduce` default aperture creation:

```python
if amp is None:
    amp = make_illumination(cfg)
```

Update `__all__` to include `make_illumination`, `ILLUMINATION_PRESETS`, and `DEFAULT_ILLUMINATION`.

- [ ] **Step 2: Update scripts to use helper**

In `scripts/fig1b.py`, `scripts/fig3.py`, and `scripts/table1.py`, import `make_illumination` from `_runner` and replace:

```python
amp = make_aperture(cfg["n"], radius_px=cfg["aperture_radius_px"])
```

with:

```python
amp = make_illumination(cfg)
```

Keep `make_aperture` exported from `_runner.py` for backwards compatibility.

- [ ] **Step 3: Run GREEN**

Run:

```bash
rtk pytest tests/test_core.py::test_gaussian_aperture_is_l2_normalized_and_truncated tests/test_core.py::test_resolve_defaults_to_gaussian_illumination tests/test_core.py::test_tophat_illumination_remains_selectable -q
```

Expected: all selected tests pass.

## Task 3: Documentation And Validation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document default and compatibility flag**

Update run examples to show:

```bash
python scripts/fig3.py --preset tiny
python scripts/fig3.py --preset tiny --illumination tophat
```

Add a short section under presets:

```markdown
### Illumination presets

`--illumination gaussian` is the default. It creates a truncated Gaussian nearfield amplitude inside the same circular aperture as the flat-top model, with 1/e^2 waist `APERTURE_FRAC * n` (`0.45 * n`, so `540 px` for `n=1200`). The amplitude is L2-normalized to unit incident power, matching the existing flat-top normalization.

Use `--illumination tophat` to reproduce the previous flat-top default.
```

- [ ] **Step 2: Run full test suite**

Run:

```bash
rtk pytest -q
```

Expected: all tests pass. If optional torch is unavailable, `tests/test_torch.py` is skipped by `pytest.importorskip("torch")`.

- [ ] **Step 3: Run script smoke validation**

Run:

```bash
rtk python scripts/fig3.py --preset tiny --iters 1 --spacings 1.8 --n-spots 2
rtk python scripts/fig3.py --preset tiny --iters 1 --spacings 1.8 --n-spots 2 --illumination tophat
```

Expected: both commands complete and write `outputs/fig3.png`; console prints FOI/RSS metrics for each run.

- [ ] **Step 4: Commit**

Run:

```bash
rtk git status --short
rtk git add docs/superpowers/plans/2026-06-08-gaussian-illumination-preset.md README.md scripts/_runner.py scripts/fig1b.py scripts/fig3.py scripts/table1.py tests/test_core.py
rtk git commit -m "feat: add gaussian illumination preset"
rtk git rev-parse HEAD
```

Expected: commit succeeds on `feat/gaussian-illumination-preset`; final hash printed.

## Self-Review

- Requirement coverage: plan adds selectable Gaussian preset, waist `APERTURE_FRAC * n`, default Gaussian behavior, old flat-top selectable behavior, tests, docs, validation, branch/commit.
- Placeholder scan: no `TBD`, `TODO`, or unresolved implementation blanks.
- Type consistency: runner config keys are `illumination`, `illumination_profile`, `gauss_radius_px`, and `aperture_radius_px`; tests and scripts use same names.
