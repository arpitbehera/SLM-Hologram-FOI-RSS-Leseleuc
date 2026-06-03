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

Spacings are quoted in coarse target-plane pixels (1.6–2.5), so d/r_A ratios are
preset-independent; absolute Table I numbers only converge to the paper at
`--preset paper` on a GPU.

## Layout

```
src/foitweezers/   config, forward, targets, losses, design, metrics, aberration, io
scripts/           fig1b.py, fig3.py, table1.py, run_all.py
tests/             gradient/forward/metrics + torch autodiff checks
outputs/           generated figures, tables, arrays
docs/refs/         papers + slmsuite reference checkout
```
