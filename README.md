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
python scripts/run_all.py --preset tiny    # Fig1b + Fig3 + Table I -> outputs/
```

### Presets (runtime vs fidelity)

| preset | grid N | oversample | iters | seeds | where |
|--------|--------|-----------|-------|-------|-------|
| `tiny` | 96  | 6  | 120  | 1  | CPU smoke (default) |
| `cpu`  | 160 | 10 | 300  | 3  | CPU overnight |
| `paper`| 800 | 10 | 1000 | 20 | **GPU** (8000² FFT; paper quotes 10–20 min/hologram on GPU) |

Spacings are *designed* in coarse target-plane pixels (1.6–2.5) but the final
figures/tables quote them in physical units only (`r_A`, µm, and λ at a 780 nm
reference). See **Spacing units** below.

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

## Layout

```
src/foitweezers/   config, forward, targets, losses, design, metrics, aberration, io
scripts/           fig1b.py, fig3.py, table1.py, run_all.py
tests/             gradient/forward/metrics + torch autodiff checks
outputs/           generated figures, tables, arrays
docs/refs/         papers + slmsuite reference checkout
```
