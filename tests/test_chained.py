"""Unit + regression tests for chained optimization methods."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from foitweezers.forward import make_aperture
from foitweezers.targets import square_lattice_target


def _setup(n=24, ov=4, spots=2, spacing=2.0):
    amp = make_aperture(n, radius_px=0.45 * n, profile="tophat")
    m = n * ov
    T, pos, sint = square_lattice_target(m, spacing_fine_px=spacing * ov, n_spots=spots)
    return amp, T, pos, sint, ov


def test_design_cgh_uses_phase0():
    from foitweezers.design import design_cgh
    amp, T, _, _, ov = _setup()
    n = amp.shape[0]
    phase0 = np.full((n, n), 1.234)
    # 0 iters: optimizer should return (mod) the provided phase0 unchanged.
    out = design_cgh(T, amp, ov, method="RSS", iters=0, phase0=phase0)
    assert np.allclose(out, np.mod(phase0, 2 * np.pi), atol=1e-9)


def test_design_cgh_phase0_none_matches_seed():
    from foitweezers.design import design_cgh, initial_phase
    amp, T, _, _, ov = _setup()
    n = amp.shape[0]
    out = design_cgh(T, amp, ov, method="RSS", seed=7, iters=0)
    assert np.allclose(out, np.mod(initial_phase(n, 7), 2 * np.pi), atol=1e-9)


def test_design_cgh_torch_uses_phase0():
    pytest.importorskip("torch")
    from foitweezers.design import design_cgh
    amp, T, _, _, ov = _setup()
    n = amp.shape[0]
    phase0 = np.full((n, n), 1.234)
    out = design_cgh(T, amp, ov, method="RSS", iters=0, phase0=phase0, backend="torch")
    assert np.allclose(out, np.mod(phase0, 2 * np.pi), atol=1e-9)


def test_design_cgh_slmsuite_threads_phase0():
    pytest.importorskip("slmsuite")
    from foitweezers.design import design_cgh
    amp, T, _, _, ov = _setup(n=16, ov=2)
    n = amp.shape[0]
    phase0 = np.full((n, n), 1.234)
    # 2 iters: just confirm the slmsuite branch runs with phase0 and returns a
    # valid wrapped phase of the right shape (warm-start fidelity verified by
    # code inspection — Hologram(phase=phase0)).
    out = design_cgh(T, amp, ov, method="RSS", iters=2, phase0=phase0,
                     backend="slmsuite")
    assert out.shape == (n, n)
    assert out.min() >= 0.0 and out.max() < 2 * np.pi + 1e-9
