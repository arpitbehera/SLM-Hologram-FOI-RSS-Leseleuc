# foitweezers — reproducing Nishimura/Ando/de Léséleuc, *PRA* 113, 013119 (2026)

Python simulation of **optimization-based (FOI) hologram design** for fine optical
tweezer arrays near the Rayleigh limit, reproducing the numerical results of

> K. Nishimura, H. Sakai, T. Ando, T. Tomita, S. de Léséleuc,
> "Optimization-based hologram design for fine optical tweezer arrays and extension
> of super-resolution criteria," *Phys. Rev. A* **113**, 013119 (2026).

Method and pseudocode follow the paper plus the companion Hamamatsu note
(`docs/refs/papers/`). The plan lives at
`~/.claude-personal/plans/before-doing-anything-tell-jiggly-sunrise.md`.

## What it does (Phase 1)

Reproduces the paper's central numerical result — **FOI vs RSS** cost functions on
a 5×5 array:

- **Fig 1(b)** — the FOI and RSS hologram phase patterns (`scripts/fig1b.py`).
- **Fig 3** — numerically reproduced 5×5 images at 1.6/1.8/2.1/2.5 px; FOI keeps
  spots separated where RSS merges (`scripts/fig3.py`).
- **Table I** — spot uniformity (relative SD) and light-utilization efficiency,
  ± an added reproduction aberration (`scripts/table1.py`).

Phase 2 (planned): vector-Debye VP-ratio super-resolution criterion (Fig 4),
non-square lattices (Fig 5), truncated-Gaussian illumination sweep (Table II).

## Algorithm

Scalar Fraunhofer forward model `Ψ = FFT(a·e^{iθ})`, `I = |Ψ|²` (unitary FFT,
energy-conserving). Minimize the **FOI** cost `f = −Σ Ĩ_i T̃_i` (Euclidean-normed)
— or the **RSS** baseline `Σ(I_i−T_i)²` — over the nearfield phase. Faithful
conjugate-gradient via `scipy.optimize.minimize(method="CG")` with the analytic
gradient

```
df/dθ_j = 2·Im[ conj(ψ_j) · crop_N( IFFT( (df/dI)·Ψ ) )_j ]
```

(`df/dI = (−T̃ − f·Ĩ)/‖I‖₂` for FOI; verified to 4e-12 vs finite differences —
note the sign of the `f` term differs from the paper's transcribed Eq. 4).

## Relationship to slmsuite

`slmsuite` (vendored at `docs/refs/slmsuite`) is the framework for the GPU path,
Zernike aberrations, spot fitting, and analysis. Its `Hologram.optimize(
method="CG", loss=...)` accepts a custom torch loss — we plug in `FOILoss`
(`src/foitweezers/losses.py`). Note slmsuite's `"CG"` is Adam gradient descent,
not strict CGM, so the **`scipy` backend is the faithful reproduction**; the
torch/slmsuite backends are the GPU-accelerated approximations.

## Install & run

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .            # core (numpy/scipy/matplotlib/slmsuite)
uv pip install -e '.[torch]'   # optional: torch backend
# On a CUDA machine, also: uv pip install cupy-cuda12x   (match local CUDA)

pytest -q                                  # unit + gradient + autodiff tests
python scripts/fig3.py   --preset tiny     # quick CPU smoke (minutes)
python scripts/fig3.py   --preset tiny --illumination tophat  # previous flat-top default
python scripts/run_all.py --preset tiny    # Fig1b + Fig3 + Table I -> outputs/
```

### Presets (runtime vs fidelity)

| preset | grid N | oversample | iters | seeds | where |
|--------|--------|-----------|-------|-------|-------|
| `tiny` | 96  | 6  | 120  | 1  | CPU smoke (default) |
| `cpu`  | 160 | 10 | 300  | 3  | CPU overnight |
| `paper`| 1200 | 10 | 1000 | 20 | **GPU** (12000² FFT; paper quotes 10–20 min/hologram on GPU) |

Spacings are *designed* in coarse target-plane pixels (1.6–2.5) but the final
figures/tables quote them in physical units only (`r_A`, µm, and λ at a 780 nm
reference). See **Spacing units** below.

### CLI reference

The figure and table scripts all share a common parser from `scripts/_runner.py`.
Use `python scripts/<script>.py --help` for the argparse summary, or use the
tables below to choose parameters without reading code.

| script | purpose | arguments |
|--------|---------|-----------|
| `scripts/fig1b.py` | FOI/RSS CGH phase patterns at the primary spacing | Common args only. Uses the first resolved seed and fixed spacing `1.8` coarse px. |
| `scripts/fig3.py` | FOI/RSS reproduced images over multiple spacings | Common args. Uses the first resolved seed and `--spacings` for columns. |
| `scripts/table1.py` | Uniformity, efficiency, and VP metrics with/without aberration | Common args plus `--spacing` and `--aberration-rms`. Aggregates over all resolved seeds. |
| `scripts/run_all.py` | Runs `fig1b.py`, `fig3.py`, then `table1.py` | Forwards CLI args to all three scripts. Use common args only; `--spacing` and `--aberration-rms` are `table1.py`-only and will be rejected by `fig1b.py`/`fig3.py`. |
| `scripts/optimize_mask.py` | Best single FOI/RSS mask + predicted image for one spacing, over N seeds | Common args plus `--method` (required), `--spacing`, `--tag`. `--seeds` defaults to `1` here. |

Common arguments:

| argument | default | choices / type | controls |
|----------|---------|----------------|----------|
| `--preset` | `tiny` | `tiny`, `cpu`, `paper` | Runtime/fidelity bundle: grid size `n`, far-field oversample, optimizer iterations, and seed list. `tiny` = quick CPU smoke (`n=96`, oversample `6`, `120` iterations, seed `0`); `cpu` = slower CPU run (`n=160`, oversample `10`, `300` iterations, seeds `0..2`); `paper` = full-scale GPU run (`n=1200`, oversample `10`, `1000` iterations, seeds `0..19`). |
| `--illumination` | `gaussian` | `gaussian`, `tophat` | Nearfield amplitude profile inside the circular aperture. `gaussian` uses a truncated Gaussian with 1/e² radius equal to `0.45 * n`; `tophat` uses the previous flat-top aperture. Both are L2-normalized to unit incident power. |
| `--backend` | `scipy` | `scipy`, `torch`, `slmsuite` | Optimization backend passed to `design_cgh`. `scipy` is the faithful conjugate-gradient reproduction; `torch`/`slmsuite` are GPU-capable approximations and need optional torch/GPU dependencies. |
| `--iters` | preset value | integer | Overrides the preset optimizer iteration count. Leave unset to use `120`, `300`, or `1000` from `--preset`. |
| `--seeds` | preset list | positive integer | Overrides the preset seed list with `range(N)`. `fig1b.py` and `fig3.py` use only the first resolved seed; `table1.py` uses all resolved seeds for mean +/- SE. |
| `--spacings` | `1.8 2.1 2.4 2.7` | zero or more floats | Spacing list in coarse target-plane pixels. Used by `fig3.py` to choose montage columns. Accepted but not used by `fig1b.py` and `table1.py`; use `--spacing` for `table1.py`. |
| `--n-spots` | `5` | integer | Square array side length. `5` means a `5x5` target; `7` means `7x7`, etc. Affects target construction, crops, labels, and metrics. |

`table1.py` extra arguments:

| argument | default | type | controls |
|----------|---------|------|----------|
| `--spacing` | `1.8` | float | Single Table I spacing in coarse target-plane pixels. This controls both numerical and aberrated table rows. |
| `--aberration-rms` | `0.02` | float | Added reproduction aberration RMS in waves for the `numerical+aberration` rows. Set `0` to make the aberrated reproduction identical to the numerical reproduction. |

`optimize_mask.py` extra arguments:

| argument | default | type | controls |
|----------|---------|------|----------|
| `--method` | *(required)* | one or more of `FOI` / `RSS` | Cost function(s). Two or more (e.g. `--method RSS FOI`, also `RSS FOI RSS`) run as warm-started stages in order: each stage seeds the next from its continuous phase. `--iters` applies uniformly to every stage. |
| `--spacing` | `1.8` | float | Single spacing in coarse target-plane px. |
| `--tag` | none | str | Optional suffix appended to output filenames. |

Note: unlike the other scripts, `--seeds` defaults to `1` (a single design). With
`--seeds N > 1` the script designs `N` independent holograms and saves only the
single **best** one — the seed with the **lowest final optimizer cost**. Metadata
records aggregate `*_mean` (and `*_std` when `N>1`) plus a `best` block (the chosen
seed's own metrics, never std). Outputs to `outputs/` (stem uses the joined method
label, e.g. `optmask_RSS-FOI_sp{spacing}`):
`optmask_{label}_sp{spacing}[_{tag}]_{phase,image}.{png,npz}`, `_convergence.png`
(cost vs iteration per seed), and `_meta.json`. For chained runs the single
`_convergence.png` is replaced by `_convergence_RSS.png` / `_convergence_FOI.png`
(continuous-axis RSS and FOI cost across all stages, with stage-transition markers).
The mask `.npz` stores `phase_uint8` (dtype uint8, code `k` -> phase `k*2pi/256`).

Convenient examples:

```bash
python scripts/fig1b.py --preset tiny
python scripts/fig3.py --preset tiny --spacings 1.6 1.8 2.1 2.5
python scripts/table1.py --preset cpu --seeds 3 --spacing 1.8 --aberration-rms 0.02
python scripts/run_all.py --preset tiny --backend scipy --illumination gaussian
python scripts/run_all.py --preset tiny --illumination tophat
python scripts/optimize_mask.py --method FOI --preset tiny --seeds 5 --spacing 1.8
```

## Spacing units

A "coarse target-plane pixel" is one pixel of the discretized far-field (focal)
plane in the numerical reproduction — *not* an SLM pixel. The Rayleigh radius in
those pixels is fixed by the aperture fill alone (pure FFT geometry), so it is
grid/preset-independent:

```
r_A[px] = 1.22 · M / D_aperture / oversample = 1.22 / (2 · APERTURE_FRAC)
        = 1.22 / (2 · 0.45) = 1.356 coarse px      (M = n·oversample, D = 2·frac·n)
```

The absolute scale comes from the optics in `config.py` (λ=852 nm, NA=0.7):

```
r_A[µm]    = 0.61 · λ / NA = 0.61 · 0.852 / 0.7 = 0.742 µm
µm/px      = r_A[µm] / r_A[px] = 0.742 / 1.356  = 0.548 µm
λ-units    = d[µm] / 0.780 µm                    (780 nm reference, ≠ design λ)
```

So for a coarse spacing `d` (px): `d/r_A = d/1.356`, `d_µm = 0.548·d`,
`d_λ = d_µm/0.780`. The conversion lives in `scripts/_runner.py`
(`spacing_units` / `spacing_label`); `r_A` and `λ` units track the design optics,
the 780 nm λ-reference is a fixed display convention.

| coarse px | r_A | µm | λ (780 nm) |
|-----------|------|------|------------|
| 1.6 | 1.18 | 0.88 | 1.12 |
| 1.8 | 1.33 | 0.99 | 1.26 |
| 2.1 | 1.55 | 1.15 | 1.47 |
| 2.5 | 1.84 | 1.37 | 1.76 |

`d/r_A` ratios are preset-independent; absolute Table I numbers only converge to
the paper at `--preset paper` on a GPU.

### Illumination presets

`--illumination gaussian` is the default. It creates a truncated Gaussian
nearfield amplitude inside the same circular aperture as the flat-top model, with
1/e² waist `APERTURE_FRAC * n` (`0.45 * n`, so `540 px` for `n=1200`). The
amplitude is L2-normalized to unit incident power, matching the existing flat-top
normalization.

Use `--illumination tophat` to reproduce the previous flat-top default.

## Optimal mask generator (`optimize_mask.py`)

Designs a single optimal FOI- or RSS-CGH for one spacing, optionally over several
random seeds, and saves only the **best** hologram plus its predicted image and
metadata. Use this when you want one deliverable mask (e.g. to upload to an SLM),
rather than a multi-panel paper figure.

```bash
source .venv/bin/activate                    # or prefix commands with .venv/bin/python

# quickest smoke (single seed, ~minutes on CPU)
python scripts/optimize_mask.py --method FOI --preset tiny

# 5 seeds, keep only the best; tag the output files
python scripts/optimize_mask.py --method RSS --preset tiny --seeds 5 --tag run1

# faster smoke: fewer iterations
python scripts/optimize_mask.py --method FOI --preset tiny --seeds 3 --iters 30

# chained: warm-start RSS, then refine with FOI (continuous-axis dual plots)
python scripts/optimize_mask.py --method RSS FOI --preset tiny --iters 30
```

### Supported presets

Same `--preset` bundles as the figure scripts (grid `n` / oversample / iters /
seed list). `optimize_mask.py` overrides the preset's seed *count* with `--seeds`
(default `1`); the other preset values (`n`, oversample, iters) still apply.

| preset | grid N | oversample | iters | where |
|--------|--------|-----------|-------|-------|
| `tiny` | 96  | 6  | 120  | CPU smoke (default) |
| `cpu`  | 160 | 10 | 300  | CPU overnight |
| `paper`| 1200 | 10 | 1000 | **GPU** (use `--backend torch`/`slmsuite`) |

### CLI arguments

All [common arguments](#cli-reference) apply (`--preset`, `--illumination`,
`--backend`, `--iters`, `--n-spots`, `--seeds`), plus:

| argument | default | type | controls |
|----------|---------|------|----------|
| `--method` | *(required)* | one or more of `FOI` / `RSS` | Cost function(s). A single method is the classic run. Two **or more** (e.g. `--method RSS FOI`, also `RSS FOI RSS`) run as warm-started stages **in order**: each stage starts from the previous stage's converged continuous phase. `--iters` applies uniformly to every stage. |
| `--spacing` | `1.8` | float | Single spacing in coarse target-plane px. |
| `--seeds` | `1` | int | Number of independent designs; only the best is saved. |
| `--tag` | none | str | Optional suffix appended to output filenames. |

**Best-seed criterion:** the saved hologram is the seed with the **lowest final
optimizer cost** (`design_cgh`'s `final_cost`). Since `--method` is fixed per run,
all seeds minimize the same objective, so their costs are directly comparable.

### Outputs

Written to `outputs/` with stem `optmask_{label}_sp{spacing}[_{tag}]`, where
`{label}` is the joined method label (single-method usage is unchanged, e.g.
`optmask_FOI_sp1.8`; chained becomes `optmask_RSS-FOI_sp1.8`):

| file | content |
|------|---------|
| `{stem}_phase.png` / `.npz` | best phase mask; `.npz` key `phase_uint8` (dtype `uint8`, code `k` -> phase `k*2pi/256`) |
| `{stem}_image.png` / `.npz` | best predicted intensity image; `.npz` key `image` |
| `{stem}_convergence.png` | (single-method) optimizer cost vs iteration, one line per seed (best highlighted) |
| `{stem}_convergence_RSS.png` / `_convergence_FOI.png` | (chained) continuous-axis RSS and FOI cost across all stages, with dashed stage-transition markers |
| `{stem}_meta.json` | run params + results |

For chained runs, `results.method` is the ordered method list, `results.stages`
records each stage's `method`/`final_cost`/`nit`, and the `best` block gains
`rss_cost` / `foi_cost` (the best seed's final per-method cost).

Metadata `results` block records `method`, `spacing_px/rA/um/lambda`,
`uniformity_mean`, `efficiency_mean`, `vp_mean`, `n_seeds`, the `*_std` fields
(**only when `--seeds > 1`**), and a `best` block with the chosen seed's own
`seed`, `final_cost`, `uniformity`, `efficiency`, `vp_ratio` (never std). Inspect it:

```bash
python -c "import json;print(json.dumps(json.load(open('outputs/optmask_FOI_sp1.8_meta.json'))['results'],indent=2))"
```

Note: the `scipy` backend (default) populates the per-iteration convergence
history; with `torch`/`slmsuite` only the final cost is available, so the
convergence plot is skipped. Chained warm-starting works on every backend, but the
dual continuous-axis plots are scipy-only (only scipy exposes a per-iteration
callback); chained `torch`/`slmsuite` runs still stage correctly but emit no plots.

## Layout

```
src/foitweezers/   config, forward, targets, losses, design, metrics, aberration, io
scripts/           fig1b.py, fig3.py, table1.py, run_all.py, optimize_mask.py
tests/             gradient/forward/metrics + torch autodiff checks
outputs/           generated figures, tables, arrays
docs/refs/         papers + slmsuite reference checkout
```
