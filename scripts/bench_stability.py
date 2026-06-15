"""Benchmark the numerical-stability fix across mask sizes via the REAL pipeline.

Uses the fixed design_cgh (relative CG gtol + scale-aware Adam eps). Reports
efficiency + uniformity for RSS and FOI so we can confirm:
  * RSS converges reliably up to the paper's 1200x1200 mask;
  * FOI (already better-scaled) is preserved and shows no regression.

Run: .venv/bin/python -u scripts/bench_stability.py [oversample] [nmax]
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foitweezers.forward import make_aperture, reproduce_intensity
from foitweezers.targets import square_lattice_target
from foitweezers.design import design_cgh
from foitweezers.metrics import evaluate

APERTURE_FRAC = 0.45
SPACING_COARSE = 1.8
N_SPOTS = 5
SEED = 0


def run(n, oversample, method, backend, iters):
    amp = make_aperture(n, radius_px=APERTURE_FRAC * n, profile="tophat")
    m = n * oversample
    T, pos, sint = square_lattice_target(m, spacing_fine_px=SPACING_COARSE * oversample, n_spots=N_SPOTS)
    t0 = time.time()
    phase, info = design_cgh(T, amp, oversample, method=method, seed=SEED,
                             iters=iters, backend=backend, return_info=True)
    I = reproduce_intensity(phase, amp, oversample, shift=True)
    ev = evaluate(I, pos, sint, n_spots=N_SPOTS)
    return ev["efficiency"], ev["uniformity"], info.get("nit"), time.time() - t0


def main():
    oversample = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    iters = 500
    sizes = [n for n in (128, 256, 512, 768, 1024, 1200) if n <= nmax]
    print(f"oversample={oversample} iters={iters} backend=torch")
    print(f"{'method':>6} {'n':>5} {'m':>6} {'eff':>9} {'unif':>9} {'sec':>7}")
    for method in ("RSS", "FOI"):
        for n in sizes:
            try:
                eff, unif, nit, dt = run(n, oversample, method, "torch", iters)
                print(f"{method:>6} {n:5d} {n*oversample:6d} {eff:9.4f} {unif:9.4f} {dt:7.1f}", flush=True)
            except Exception as e:
                print(f"{method:>6} {n:5d} {n*oversample:6d}  ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
