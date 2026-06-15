# Insights: How the CGH Phase Optimization Works

This document explains the core algorithms behind `foitweezers`: the two cost
functions (FOI and RSS), the three optimization backends, the shared Fourier
forward/backward model, and a from-scratch crashcourse on phase-only hologram
optimization. It is written to be read top-to-bottom by someone new to the
codebase.

Source files referenced throughout:

- `src/foitweezers/forward.py` — the Fourier forward model.
- `src/foitweezers/losses.py` — FOI/RSS cost functions and gradients.
- `src/foitweezers/design.py` — the optimization drivers (scipy / torch / slmsuite).

---

## 0. Crashcourse: phase-only hologram optimization

Read this first if you are new. It frames everything below.

### The physical problem

A Spatial Light Modulator (SLM) can change only the **phase** of incoming light,
pixel by pixel — it cannot freely set the amplitude. We illuminate it with a
laser (fixed amplitude `a`, a circular top-hat or Gaussian aperture) and we get
to choose the phase pattern `θ` (one value per pixel, an `n × n` grid).

A lens performs a **Fourier transform** of the SLM field on its way to the focal
(image) plane. So the intensity we see at the focal plane is

```
I = | FFT( a · exp(iθ) ) |²
```

Goal: pick `θ` so that `I` matches a desired target intensity `T` — for example,
an array of bright spots that become optical tweezers for trapping atoms.

The catch: we control phase, not amplitude. There is no closed-form inverse, the
problem is **non-convex**, and we solve it by iterative optimization.

### The optimization loop (method-agnostic)

```
1. θ₀ ← random uniform [0, 2π)          # design.py: initial_phase()
2. repeat:
   a. forward:   ψ = a·exp(iθ)  →  Ψ = FFT(ψ)  →  I = |Ψ|²
   b. cost:      f = how far I is from T   (FOI or RSS — see §1)
   c. gradient:  df/dθ                     (analytic or autodiff — see §2)
   d. step:      θ ← θ − α · df/dθ         (CG / Adam / LBFGS)
3. θ_final ← mod(θ, 2π)                  # wrap to a physical phase
```

### Why gradient descent works through a `|·|²`

The cost `f` is a smooth function of the real phase `θ`. The FFT is linear and
its adjoint is the inverse FFT, so the chain rule gives a clean analytic
gradient (derived in §3):

```
df/dθ_j = 2 · Im[ conj(ψ_j) · crop_NN( IFFT( (df/dI) · Ψ ) )_j ]
```

Intuition: take the intensity error signal `df/dI`, **backpropagate it through
the inverse FFT** to the SLM plane, then project onto the phase direction. The
factor `Im[ conj(ψ) · (…) ]` is exactly "how much does `|field|²` move when I
nudge the phase here."

### Why it is hard (and the practical tricks)

- **Phase-only ⇒ non-convex.** Different random seeds land in different local
  minima. The production tool (`scripts/optimize_mask.py`) runs several seeds and
  keeps the best mask.
- **Oversampling.** The `n × n` SLM field is zero-padded to `m = n · oversample`
  before the FFT, giving a finer-sampled focal plane (sinc interpolation). This
  is the Kim 2019 upsampling (paper ref [29]).
- **Unitary FFT** (`norm="ortho"`) is used everywhere so energy is conserved
  between planes and the gradient math stays clean (§3).
- **8-bit quantization** at the end (`forward.quantize_phase`) emulates a real
  SLM's discrete phase levels.

Everything below is detail on the two pieces that vary — the cost function (§1)
and the backend that drives the loop (§2) — plus the Fourier model they share
(§3).

---

## 1. The two cost functions: FOI and RSS

Both measure "how far is reproduced intensity `I` from target `T`," but they
disagree on what "far" means. Only `df/dI` differs between them; the whole FFT
forward/backward pipeline is identical (§3).

All arrays are in **natural FFT ordering** (not `fftshift`-ed). The caller builds
a centered target and `ifftshift`s it before handing it to these functions
(`design.py: _normalize_target_natural`).

### RSS — Residual Sum of Squares

`losses.py: rss_cost_grad`

```
f_RSS = Σ_i (I_i − T_i)²
```

- Both `I` and `T` are **sum-normalized** (Σ = 1). `I` sums to 1 automatically:
  the ortho FFT conserves energy and `amp` is L2-normalized. `T` is sum-normalized
  by the caller.
- Gradient term: `df/dI = 2 (I − T)`.
- Character: penalizes the **absolute** per-pixel intensity error. Sensitive to
  the overall brightness/scale of the reconstruction.

### FOI — Fidelity Of Intensity

`losses.py: foi_cost_grad`

```
f_FOI = − Σ_i Ĩ_i · T̃_i ,   with  Ĩ = I / ‖I‖₂ ,  T̃ = T / ‖T‖₂
```

- This is the **negative cosine similarity** between the `I` and `T` vectors
  (L2 / Euclidean normalization). `f ∈ [−1, 0]`, with `−1` a perfect match.
- Because both vectors are normalized, FOI is **scale-invariant**: it cares about
  the *shape* of the intensity pattern, not its absolute brightness.
- Gradient term: `df/dI = (−T̃ − f · Ĩ) / N_I`, where `N_I = ‖I‖₂`.

> **Sign subtlety (code corrects a paper typo).** The transcribed paper /
> Hamamatsu pseudocode writes `+ f · Ĩ`. The rigorous derivative (carrying the
> `N_I = ‖I‖₂` dependence) is `− f · Ĩ`. The code uses `− f · Ĩ`, verified to
> `4e-12` against finite differences. See the comment at `losses.py:82`.

### Why FOI is preferred

Scale-invariance means FOI tolerates uniform efficiency loss and produces cleaner,
more uniform traps for Rayleigh tweezers — the central thesis of the paper this
repo reproduces. RSS, by contrast, fights to match absolute intensity and is more
fragile to global scaling.

### Side-by-side

| | RSS | FOI |
|---|---|---|
| Cost | `Σ (I−T)²` | `−Σ Ĩ·T̃` (neg. cosine sim) |
| Normalization | sum (Σ = 1) | L2 (‖·‖₂ = 1) |
| `df/dI` | `2(I − T)` | `(−T̃ − f·Ĩ)/N_I` |
| Scale behavior | scale-sensitive | scale-invariant |
| Range | `≥ 0`, lower better | `[−1, 0]`, lower better |

`COST_GRAD = {"FOI": foi_cost_grad, "RSS": rss_cost_grad}` (`losses.py`) selects
between them by name.

---

## 2. The three backends: scipy vs torch vs slmsuite

`design.py: design_cgh(..., backend=...)` dispatches the *same cost math* to one
of three optimization engines. They differ in optimizer, how the gradient is
obtained, target device, and how faithful they are to the paper.

| | `scipy` (default) | `torch` | `slmsuite` |
|---|---|---|---|
| Driver | `_design_scipy` | `_design_torch` | `_design_slmsuite` |
| Optimizer | true nonlinear CG (`minimize(method="CG")`) | Adam / LBFGS | slmsuite `Hologram.optimize(method="CG")` |
| Gradient | **analytic** (`COST_GRAD`) | autodiff (`loss.backward()`) | autodiff (`TORCH_LOSSES`) |
| Device | CPU | CPU / CUDA (auto-detect) | GPU-oriented |
| Faithful to paper? | **yes** (strict CGM) | approximate | approximate |

### `scipy` — faithful conjugate-gradient (default)

`_design_scipy`. Feeds the **analytic** cost+gradient (`foi_cost_grad` /
`rss_cost_grad`) into `scipy.optimize.minimize(method="CG", jac=True)`. This is a
true nonlinear conjugate-gradient method. Runs on CPU, deterministic per seed.
Use this to reproduce the paper exactly.

### `torch` — autodiff gradient descent

`_design_torch`. Self-contained: rebuilds the field, FFT, and cost **inline**
(`losses` not reused) and lets autograd compute the gradient via
`loss.backward()`. Optimizer is Adam (default, `lr=0.1`) or LBFGS (a single
`opt.step(closure)`). Auto-detects CUDA (`dev = "cuda" if torch.cuda.is_available()`).
Use this for GPU runs where you control the loop. The gradient is autodiff, not
the analytic form, so results are close but not bit-identical to scipy.

### `slmsuite` — adapter onto an external GPU library

`_design_slmsuite`. Delegates to the external `slmsuite` package's experimental CG
path (`Hologram.optimize(method="CG", loss=...)`). The target is converted to an
**amplitude** `√T` on the centered `m × m` grid (slmsuite's convention) and the
loss is supplied as a `torch.nn.Module` (`FOILoss` / `RSSLoss` from `losses.py`).

Caveats:

- slmsuite's `"CG"` is actually **gradient descent (Adam)**, *not* strict
  conjugate gradient — so it is not a faithful CGM reproduction.
- slmsuite's `optimize_cg` calls `optimizer.step()` **without a closure**, so
  closure-only optimizers (LBFGS) are unsupported here; Adam is the default and
  matches slmsuite's own configuration.

### Bottom line

- Reproduce the paper exactly → **scipy**.
- GPU run you control end-to-end → **torch**.
- Drive the external library / compare against it → **slmsuite** (experimental,
  least faithful).

---

## 3. The Fourier transform: shared by FOI and RSS

**The FFT is not method-specific.** It is the forward physical model, identical
for FOI and RSS. The two methods diverge only at the single `df/dI` step that
sits *between* the forward FFT and the backward IFFT.

### Forward model

`forward.py: farfield` and `losses.py: _forward_natural`:

```
ψ = amp · exp(iθ)              # n×n nearfield (SLM plane)
P = embed(ψ, m)               # center zero-pad to m×m, m = n·oversample
Ψ = fft2(P, norm="ortho")     # m×m farfield, natural ordering
I = |Ψ|²                       # reproduced intensity
```

### Backward (gradient) model

`losses.py: _grad_from_dfdI`:

```
back   = ifft2(dfdI · Ψ, norm="ortho")    # backpropagate error through the IFFT
back_nn = crop_NN(back)                    # take the center n×n block
df/dθ  = 2 · Im[ conj(ψ) · back_nn ]
```

Here `dfdI` is the **only** thing that differs between FOI and RSS (§1). Same FFT,
same IFFT, same projection — different scalar field in the middle.

### Why these specific choices

- **`norm="ortho"` (unitary FFT) everywhere.** Energy is conserved between
  planes, so `Σ I = 1` for free (this is what lets RSS skip re-normalizing `I`).
  It also makes the FFT's adjoint equal to the IFFT, so the analytic identity
  `Σ_i u*_ij X_i = IFFT(X)_j` holds exactly — that identity *is* the gradient.
- **Zero-pad oversampling** (`_embed`): `n×n → m×m`, `m = n·oversample`. This is
  the Kim 2019 upsampling that gives ~10× finer focal-plane sampling than the
  coarse target grid (sinc interpolation of the farfield).
- **Natural FFT ordering** in the gradient math (no `fftshift`). The target is
  built centered and `ifftshift`-ed before use (`design.py`); display/metrics use
  `fftshift` (`forward.reproduce_intensity`).

### A note on the torch backend's FFT

The `torch` backend re-implements the *same* forward FFT inline
(`design.py: loss_fn`) — same `embed`, same `fft2(norm="ortho")`, same `|Ψ|²`.
The difference is only that it never writes the explicit backward IFFT: autograd
derives the equivalent gradient automatically from `loss.backward()`.

---

## 4. How target intensities are handled

Short answer: **yes — the spot locations are pre-declared.** You specify the
lattice geometry up front; the target `T` is a hand-built intensity map with
peaks at exactly those sites. The optimizer's job is to make the reproduced
farfield `I` resemble `T`. It never invents or discovers spot positions.

### Building the target

`targets.py: square_lattice_target(m, spacing_fine_px, n_spots=5)`:

1. Compute the integer `(row, col)` positions of an `n_spots × n_spots` lattice,
   **centered** on the `m × m` grid (`square_lattice_positions`).
2. `T = zeros(m, m)`; set `T[r, c] = 1.0` at each spot, **0 everywhere else**.

So `T` is a set of **single-pixel, unit-intensity peaks** at the lattice sites on
an otherwise-zero background — a delta-like map. The function returns
`(T, positions, spacing_int)`.

### Key facts

- **Pre-declared, not discovered.** You declare *where* the spots go; the
  optimizer finds a phase that steers light to those declared maxima.
- **Grid-snapped.** The spacing is rounded to the nearest integer fine pixel so
  spots land on grid points (the paper's targets are likewise on grid points).
  It requires `spacing_fine_px = spacing_coarse_px × oversample ≥ 1`; if it
  rounds below 1, you get an error telling you to increase `oversample`.
- **Centered (fftshift) ordering.** `T` is defined on the centered grid (matching
  `reproduce_intensity(..., shift=True)`). `design._normalize_target_natural` then
  `ifftshift`s it into natural FFT ordering and normalizes it (sum-normalized for
  RSS; left as-is for FOI, which L2-normalizes internally) before it reaches the
  loss function (§1, §3).
- **Binary peaks, not soft blobs.** The target is exact-zero background plus
  isolated `1.0` peaks. The achievable `I` is `|FFT|²` of a finite aperture, so
  real spots have finite width — they will never be perfect deltas. That is
  expected; the trap is the local intensity maximum at each declared site.

### The same positions drive the metrics

The pre-declared `positions` list is reused as ground truth for scoring the result
(`metrics.py`):

- **spot_powers** — sum of `I` inside a `(2·hw+1)²` box around each declared
  position.
- **uniformity** = `std / mean` of those spot powers (lower = more equal traps).
- **efficiency** = in-spot power / total power.
- **vp_ratio** = mean valley intensity (midpoints between adjacent declared spots)
  / mean peak intensity.

So the declared spot positions serve double duty: they define the target `T` *and*
they define where the metrics measure success.

### Scope

Phase 1 implements the `5 × 5` square lattice (Fig 1b, Fig 3, Table I).
Non-square lattices (hexagonal / kagome / triangular, Fig 5) are Phase 2 — see the
module docstring in `targets.py`.

---

## Large-mask numerical stability (RSS on 1200×1200 and up)

### Symptom

Under `--preset paper` (n=1200, oversample=10 → M=12000 farfield grid), RSS
optimization "fails to converge": with `--backend scipy` it returns the initial
random phase (a blank hologram); with `--backend torch` the spot quality degrades
as the mask grows. The same code is fine on the small/cpu presets.

### Root cause: a size-dependent gradient meeting size-blind constants

The reproduction intensity `I = |FFT(ψ)|²` is energy-normalised, so its total is 1
spread over `M² = (n·oversample)²` farfield pixels — each pixel carries `~1/M²`.
The **RSS cost itself stays O(1/K)** (K = number of spots; spot-mismatch
dominated), but the **RSS gradient inherits the tiny per-pixel intensity scale**.
Measured (and matching the analytic chain rule):

| quantity            | scaling in n | at n=256, ov=8 | extrapolated n=1200 |
|---------------------|--------------|----------------|---------------------|
| RSS cost `f`        | `n^0`        | 4.0e-2         | 4.0e-2              |
| `‖grad‖₂`           | `n^-2`       | 1.6e-7         | ~7e-9               |
| `max\|grad\|`       | `n^-3`       | 2.3e-9         | ~1e-11              |

Two fixed *absolute* constants are blind to this shrinkage and break:

1. **scipy CG `gtol=1e-9`** (`_design_scipy`). CG stops when `max|grad| < gtol`.
   For n ≳ 300 the gradient *starts* below 1e-9, so CG declares convergence at
   **iteration 0** and returns the random phase. Reproduced at n=384: `nit=0`,
   efficiency `6e-4`.
2. **torch Adam `eps=1e-8`** (`_design_torch`). Adam scales the step by
   `1/(√v + eps)`. Once `√v ~ 1e-11 ≪ eps`, the step collapses to `~grad/eps` and
   Adam loses its (intended) scale-invariance, so quality decays with n. Proof:
   re-running with `eps=1e-16` gives **bit-identical** results to multiplying the
   loss by `M²` — confirming the loss/gradient magnitude, not the math, is the
   lever.

FOI is the *same class* of problem but ~2 orders safer: its L2-normalised
objective has `‖grad‖₂ ~ n^-1`, `max|grad| ~ n^-2`, so it never trips these
constants in the paper regime. The fixes below cover it anyway (shared code path).

### Fix: scale-aware criteria (`design.py`)

* **Relative CG gtol** — `_relative_gtol(g0_inf) = max(g0_inf, floor) · 1e-6`,
  i.e. stop once the gradient has dropped ~6 orders **below its own starting
  value**. Size-independent. Applied in `_design_scipy` and `design_cgh_dual`.
* **Adam eps at the float64 noise floor** — `ADAM_EPS = 1e-16`, passed to any
  Adam-family optimizer in `_design_torch` (and the slmsuite path). On small masks
  (`√v ~ 1e-3 ≫ 1e-8 ≫ 1e-16`) this changes nothing; it only matters once the
  gradient sinks into the `1e-11` regime, so existing small-mask behaviour is
  preserved exactly.

### Validation

* `tests/test_numerical_stability.py`: the distilled bug (`gtol=1e-9` → `nit=0` in
  a tiny-gradient quadratic), the `_relative_gtol` scaling, the Adam-eps floor, a
  fast small-mask regression, and a `@pytest.mark.slow` n=384 end-to-end (was
  `nit=0`/eff 6e-4, now converges).
* `scripts/bench_stability.py` (torch, fixed code) — RSS efficiency is **flat
  0.86–0.88** across n=128→1024 (pre-fix `eps=1e-8`: 0.866→0.813), uniformity ~0.08.
* Paper-faithful n=1200, oversample=10 (M=12000) on GPU: **RSS eff 0.875, unif
  0.082, vp 0.27** (peak 10.5 GB); FOI eff 0.072, unif 0.034, vp 0.002.
* `scripts/diag_scaling.py` reproduces the `n^-2`/`n^-3` gradient scaling table.

---

## Quick reference

- Forward model + aperture + quantization → `src/foitweezers/forward.py`
- Cost functions + analytic gradients + torch loss modules → `src/foitweezers/losses.py`
- Optimization drivers (scipy / torch / slmsuite) → `src/foitweezers/design.py`
- Target intensity construction → `src/foitweezers/targets.py`
- Spot-array metrics (uniformity / efficiency / VP ratio) → `src/foitweezers/metrics.py`
- Best-of-N mask generator CLI → `scripts/optimize_mask.py`
- Paper references → `docs/refs/papers/`
