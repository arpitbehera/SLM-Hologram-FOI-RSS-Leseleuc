"""Torch-path tests: autodiff gradient must match the analytic FOI/RSS gradient."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")

from foitweezers.forward import make_aperture
from foitweezers.targets import square_lattice_target
from foitweezers.losses import foi_cost_grad, rss_cost_grad


def _autodiff_grad(phase, amp, T_nat, ov, method):
    n = amp.shape[0]
    m = n * ov
    off = (m - n) // 2
    amp_t = torch.as_tensor(amp, dtype=torch.float64)
    T_t = torch.as_tensor(T_nat, dtype=torch.float64)
    ph = torch.as_tensor(phase, dtype=torch.float64).clone().requires_grad_(True)
    psi = amp_t * torch.exp(1j * ph)
    P = torch.zeros((m, m), dtype=torch.complex128)
    P[off:off + n, off:off + n] = psi
    Psi = torch.fft.fft2(P, norm="ortho")
    I = torch.abs(Psi) ** 2
    if method == "FOI":
        loss = -torch.sum((I / torch.linalg.vector_norm(I)) * (T_t / torch.linalg.vector_norm(T_t)))
    else:
        loss = torch.sum((I - T_t) ** 2)
    loss.backward()
    return float(loss.detach()), ph.grad.numpy()


@pytest.mark.parametrize("method,cost_grad", [("FOI", foi_cost_grad), ("RSS", rss_cost_grad)])
def test_autodiff_matches_analytic(method, cost_grad):
    n, ov = 24, 4
    amp = make_aperture(n, radius_px=0.45 * n)
    m = n * ov
    T, _, _ = square_lattice_target(m, spacing_fine_px=2.0 * ov, n_spots=2)
    T_nat = np.fft.ifftshift(T)
    rng = np.random.default_rng(0)
    phase = rng.uniform(0, 2 * np.pi, (n, n))

    f_ana, g_ana = cost_grad(phase, amp, T_nat, ov)
    f_ad, g_ad = _autodiff_grad(phase, amp, T_nat, ov, method)

    assert abs(f_ana - f_ad) < 1e-9
    assert np.max(np.abs(g_ana - g_ad)) < 1e-8
