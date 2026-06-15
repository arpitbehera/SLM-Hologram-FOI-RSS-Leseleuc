"""Diagnostic: how do RSS/FOI cost + gradient magnitudes scale with mask size n?

Evidence-gathering for the large-mask convergence failure. Measures, at the
initial random phase (fixed seed), for a sweep of n at fixed oversample:

  * cost f
  * ||grad||_2 and ||grad||_inf
  * for scipy CG: nit actually taken (vs requested) -> premature stop?
  * for torch Adam: loss decrease ratio over a few hundred iters -> learning?

Run:  .venv/bin/python scripts/diag_scaling.py
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foitweezers.forward import make_aperture
from foitweezers.targets import square_lattice_target
from foitweezers.design import design_cgh, _normalize_target_natural, initial_phase
from foitweezers.losses import rss_cost_grad, foi_cost_grad

OVERSAMPLE = 8
APERTURE_FRAC = 0.45
SPACING_COARSE = 1.8
N_SPOTS = 5
SEED = 0


def measure(n, method):
    amp = make_aperture(n, radius_px=APERTURE_FRAC * n, profile="tophat")
    m = n * OVERSAMPLE
    T_centered, _, _ = square_lattice_target(m, spacing_fine_px=SPACING_COARSE * OVERSAMPLE, n_spots=N_SPOTS)
    T_nat = _normalize_target_natural(T_centered, method)
    phase0 = initial_phase(n, SEED)
    cg = rss_cost_grad if method == "RSS" else foi_cost_grad
    f, g = cg(phase0, amp, T_nat, OVERSAMPLE)
    return dict(n=n, m=m, f=f, g2=float(np.linalg.norm(g)), ginf=float(np.abs(g).max()),
                amp_max=float(amp.max()), T_nat=T_nat, amp=amp, phase0=phase0)


def torch_run(n, method, iters=300):
    """Return (final_loss, init_loss) for the torch Adam path."""
    import torch
    amp = make_aperture(n, radius_px=APERTURE_FRAC * n, profile="tophat")
    m = n * OVERSAMPLE
    T_centered, _, _ = square_lattice_target(m, spacing_fine_px=SPACING_COARSE * OVERSAMPLE, n_spots=N_SPOTS)
    T_nat = _normalize_target_natural(T_centered, method)
    phase0 = initial_phase(n, SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_t = torch.as_tensor(amp, dtype=torch.float64, device=dev)
    T_t = torch.as_tensor(np.asarray(T_nat), dtype=torch.float64, device=dev)
    ph = torch.as_tensor(phase0, dtype=torch.float64, device=dev).clone().requires_grad_(True)
    off = (m - n) // 2

    def loss_fn():
        psi = amp_t * torch.exp(1j * ph)
        P = torch.zeros((m, m), dtype=torch.complex128, device=dev)
        P[off:off + n, off:off + n] = psi
        Psi = torch.fft.fft2(P, norm="ortho")
        I = torch.abs(Psi) ** 2
        if method == "FOI":
            return -torch.sum((I / torch.linalg.vector_norm(I)) * (T_t / torch.linalg.vector_norm(T_t)))
        return torch.sum((I - T_t) ** 2)

    opt = torch.optim.Adam([ph], lr=0.1)
    l0 = float(loss_fn().detach())
    last = l0
    for _ in range(iters):
        opt.zero_grad(); l = loss_fn(); l.backward(); opt.step(); last = float(l.detach())
    return last, l0


def main():
    for method in ("RSS", "FOI"):
        print(f"\n==== {method} : cost & gradient at phase0 (oversample={OVERSAMPLE}) ====")
        print(f"{'n':>6} {'m':>7} {'f':>14} {'||g||2':>12} {'||g||inf':>12}")
        rows = []
        for n in (32, 64, 96, 128, 192, 256):
            r = measure(n, method)
            rows.append(r)
            print(f"{r['n']:6d} {r['m']:7d} {r['f']:14.4e} {r['g2']:12.4e} {r['ginf']:12.4e}")
        # fit scaling exponents in n
        ns = np.array([r["n"] for r in rows], float)
        for key in ("g2", "ginf", "f"):
            vals = np.array([abs(r[key]) for r in rows], float)
            p = np.polyfit(np.log(ns), np.log(vals), 1)[0]
            print(f"  scaling {key} ~ n^{p:+.2f}")

    print("\n==== torch Adam (lr=0.1, eps=1e-8 default): loss decrease over 300 iters ====")
    print(f"{'method':>6} {'n':>6} {'init':>14} {'final':>14} {'ratio f/i':>12}")
    for method in ("RSS", "FOI"):
        for n in (64, 128, 256):
            last, l0 = torch_run(n, method)
            print(f"{method:>6} {n:6d} {l0:14.4e} {last:14.4e} {last/l0:12.4f}")


if __name__ == "__main__":
    main()
