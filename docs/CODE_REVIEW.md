# Code Review: Initial Commit `16a4b54`

## Verdict

Qualified pass for the core scalar FOI/RSS simulator. The commit implements the main optimization model described in `docs/refs/papers/Leseleuc_PRA_2026.pdf` and the Hamamatsu method note: phase-only input field, Fourier propagation, RSS and FOI costs, analytical gradients, random phase initialization, and a SciPy conjugate-gradient optimizer.

Not a full faithful reproduction of every numerical quantity claimed in the README. Table I/Fig. 3 scripts currently verify the qualitative FOI-vs-RSS trend, but metric definitions and reproduction details drift from the paper enough that I would not treat `outputs/table1.csv` as paper-faithful without fixes below.

## Findings

1. `scripts/_runner.py:78-82`: reproduced images use continuous optimized phase, not the 8-bit discretized CGH. The paper states optimized floating-point phases are discretized into 256 levels and calls those discretized patterns "CGHs". Fix: quantize phase before numerical reproduction for Fig. 1(b), Fig. 3, and Table I, or explicitly label current outputs as ideal continuous-phase simulations.

2. `src/foitweezers/metrics.py:49-60`: utilization efficiency is measured only inside spot integration windows. Table I defines numerical utilization efficiency as the sum of `I_i` over the displayed Fig. 3 area, not the sum of per-spot boxes. Fix: add a Fig. 3/display-region efficiency metric for Table I; keep spot-window power only for uniformity.

3. `src/foitweezers/metrics.py:24-37` and `src/foitweezers/metrics.py:93-100`: spot uniformity uses fixed integer windows derived from lattice spacing. The paper derives spot powers after fitting `{x_k, y_k, w0}` and integrates each spot over `[x_k - w0/2, x_k + w0/2] x [y_k - w0/2, y_k + w0/2]`. Fix: implement the Gaussian/spot fitting workflow or downgrade Table I claims to an approximate windowed metric.

4. `README.md` and `scripts/table1.py`: claims say Table I is reproduced, but verification only covers small/tiny qualitative trends. No committed test asserts the paper values `FOI uniformity 1.21e-2`, `RSS uniformity 5.6e-2`, `FOI efficiency 0.120`, `RSS efficiency 0.928` at paper scale. Fix: add a paper-scale benchmark artifact plus tolerance, or change wording from "reproduces Table I" to "smoke-tests expected trend".

5. `src/foitweezers/losses.py:18`: docstring says FOI `df/dI = (-Ttilde + f * Itilde) / N_I`, while implementation at `src/foitweezers/losses.py:84` uses `(-Ttilde - f * Itilde) / N_I`. The implementation matches autodiff/finite difference for the coded FOI objective; the docstring still mirrors the paper/Hamamatsu sign. Fix: update the docstring and call out that this is a mathematical correction to the transcribed equation.

6. `src/foitweezers/forward.py:57-61`: Gaussian illumination uses field amplitude `exp(-2 r^2 / w^2)` while documenting `w` as a 1/e^2 radius. For a 1/e^2 intensity radius, field amplitude should be `exp(-r^2 / w^2)`. Not used in Phase 1, but it will break Table II-style illumination sweeps. Fix before claiming Appendix/Table II support.

7. `scripts/run_all.py:6` and `src/foitweezers/design.py:107-157`: `--backend torch` is Adam/LBFGS gradient descent, not the CGM described in the papers. The code documents this in `design.py`, but `run_all.py --preset paper --backend torch` can still produce outputs that look paper-faithful while using a different optimizer. Fix: restrict "paper-faithful" runs to `backend=scipy`, or rename GPU outputs as approximate.

## Paper Match

Confirmed matches:

- Forward model: `psi_j = a_j exp(i theta_j)`, Fourier propagation, `I_i = |Psi_i|^2`.
- FOI objective: negative L2-normalized intensity correlation.
- RSS objective: sum-normalized residual sum of squares.
- Analytical gradient path is tested against finite differences and Torch autodiff.
- Random initialization in `[0, 2pi)`.
- Paper preset parameters exist: `N=800`, oversample `10`, `8000 x 8000` reproduction grid, `1000` iterations, `20` seeds, circular top-hat aperture radius `360 px`.
- Qualitative trend matches smoke run: FOI lower uniformity error and lower efficiency than RSS.

Confirmed limitations:

- No vector-Debye focal model for Fig. 4/super-resolution VP analysis.
- No measured plan-fluorite aberration; synthetic Zernike aberration only.
- No Gaussian-fit peak extraction for Table I.
- No paper-scale numerical run was executed in this review.

## Verification

Commands run:

```bash
rtk .venv/bin/python -m pytest -q
rtk .venv/bin/python scripts/table1.py --preset tiny --iters 5 --seeds 1
```

Results:

- Unit tests passed: 9 tests.
- Tiny smoke: FOI `sigma=1.608e-01`, efficiency `0.306`; RSS `sigma=3.774e-01`, efficiency `0.609`.
- System Python lacked `numpy`/`pytest`; verification used repo `.venv` Python 3.12.13 with `numpy 2.4.6`, `scipy 1.17.1`, `pytest 9.0.3`.

## Final Review Verdict

Core FOI/RSS optimization: true scalar Python simulation of the papers' central algorithm, with the FOI gradient sign corrected to match the actual objective derivative.

Paper reproduction claims: not fully supported yet. Fix quantization, Table I metrics, paper-scale benchmarks, and wording around non-CGM GPU backends before treating the repository as a faithful reproduction of the published numerical results.
