"""Numerical-stability regressions for large SLM masks.

Root cause (see docs/INSIGHTS.md): the RSS gradient magnitude shrinks with the
farfield grid size M = n*oversample (||grad||_2 ~ 1/n^2, max|grad| ~ 1/n^3),
because the unit total intensity is spread over M^2 pixels. Two *absolute*
numerical constants are blind to that scaling and break on large masks:

  * scipy CG ``gtol=1e-9`` -> ||grad||_inf starts below it for n >~ 300, so CG
    declares convergence at iteration 0 and returns the initial random phase.
  * torch Adam ``eps=1e-8`` -> once sqrt(v) ~ eps the optimizer loses its
    scale-invariance and the step collapses, degrading quality as n grows.

The fixes are scale-aware: a relative CG gtol anchored to the initial gradient,
and an Adam eps at the float64 noise floor.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foitweezers.forward import make_aperture, reproduce_intensity
from foitweezers.targets import square_lattice_target
from foitweezers.metrics import efficiency
from foitweezers.design import _relative_gtol, design_cgh


def test_relative_gtol_scales_with_initial_gradient():
    # gtol tracks the initial gradient infinity-norm (factor `rel` below it).
    assert _relative_gtol(1e-3, rel=1e-6) == pytest.approx(1e-9)
    assert _relative_gtol(2e-11, rel=1e-6) == pytest.approx(2e-17)
    # A vanishing initial gradient must not produce a zero (or negative) tolerance.
    assert _relative_gtol(0.0) > 0


def test_relative_gtol_iterates_when_gradient_is_tiny():
    """The core bug, distilled: an absolute gtol=1e-9 stops at nit=0 in the
    tiny-gradient regime that large RSS masks fall into; a relative gtol keeps
    optimizing. Uses a 2-D quadratic so it runs in microseconds."""
    from scipy.optimize import minimize

    scale = 1e-11                      # gradient scale comparable to a ~1200px mask
    x_star = np.array([3.0, -2.0])

    def obj(x):
        d = x - x_star
        return float(0.5 * scale * d @ d), scale * d

    x0 = np.zeros(2)
    g0 = float(np.abs(obj(x0)[1]).max())

    res_abs = minimize(obj, x0, jac=True, method="CG", options={"gtol": 1e-9})
    res_rel = minimize(obj, x0, jac=True, method="CG", options={"gtol": _relative_gtol(g0)})

    assert res_abs.nit == 0                         # absolute tol trips immediately
    assert res_rel.nit > 0                          # relative tol keeps optimizing
    assert np.allclose(res_rel.x, x_star, atol=1e-4)


def test_torch_design_default_adam_eps_below_large_mask_floor():
    """White-box guard: the torch path's default Adam eps must sit well below the
    1e-8 gradient floor that large masks reach (otherwise eps gates the step)."""
    import inspect
    from foitweezers.design import _design_torch

    eps_default = inspect.signature(_design_torch).parameters["adam_eps"].default
    assert eps_default <= 1e-12


def test_scipy_rss_small_mask_still_converges():
    """Regression guard for small masks: the relative gtol must not break the
    normal (already-working) small-mask path -- CG should still take real steps
    and deliver a usable hologram."""
    n, ov = 64, 6
    amp = make_aperture(n, radius_px=0.45 * n)
    m = n * ov
    T, pos, sint = square_lattice_target(m, spacing_fine_px=1.8 * ov, n_spots=5)
    phase, info = design_cgh(T, amp, ov, method="RSS", seed=0, iters=120,
                             backend="scipy", return_info=True)
    I = reproduce_intensity(phase, amp, ov, shift=True)
    assert info["nit"] > 0
    assert efficiency(I, pos, sint // 2) > 0.5


@pytest.mark.slow
def test_scipy_rss_converges_on_large_mask():
    """End-to-end reproduction: at n=384 the OLD absolute gtol=1e-9 gave nit=0 and
    efficiency ~ 6e-4 (initial random phase). The scale-aware gtol must converge."""
    n, ov = 384, 8
    amp = make_aperture(n, radius_px=0.45 * n)
    m = n * ov
    T, pos, sint = square_lattice_target(m, spacing_fine_px=1.8 * ov, n_spots=5)
    phase, info = design_cgh(T, amp, ov, method="RSS", seed=0, iters=300,
                             backend="scipy", return_info=True)
    I = reproduce_intensity(phase, amp, ov, shift=True)
    assert info["nit"] > 0
    assert efficiency(I, pos, sint // 2) > 0.5
