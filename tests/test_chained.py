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


def test_design_cgh_dual_histories_and_norm():
    from foitweezers.design import design_cgh_dual, _normalize_target_natural
    from foitweezers.losses import rss_cost_grad, foi_cost_grad
    amp, T, _, _, ov = _setup()
    n = amp.shape[0]
    phase, info = design_cgh_dual(T, amp, ov, method="RSS", seed=0, iters=8)
    # both histories present and equal length
    assert len(info["rss_history"]) == len(info["foi_history"])
    assert len(info["rss_history"]) >= 1
    # phase wrapped to [0, 2pi)
    assert phase.min() >= 0.0 and phase.max() < 2 * np.pi + 1e-9
    # active-method final cost equals RSS cost of final phase (own normalization)
    T_rss = _normalize_target_natural(T, "RSS")
    ref_rss = float(rss_cost_grad(phase, amp, T_rss, ov)[0])
    assert np.isclose(info["final_cost"], ref_rss, rtol=1e-4, atol=1e-6)
    # diagnostic FOI history uses FOI normalization (last entry matches reference)
    T_foi = _normalize_target_natural(T, "FOI")
    ref_foi = float(foi_cost_grad(phase, amp, T_foi, ov)[0])
    assert np.isclose(info["foi_history"][-1], ref_foi, rtol=1e-4, atol=1e-6)


def test_method_label_and_argparse():
    import argparse
    from optimize_mask import method_label
    from _runner import add_common_args
    assert method_label(["RSS"]) == "RSS"
    assert method_label(["RSS", "FOI"]) == "RSS-FOI"
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--method", choices=["FOI", "RSS"], nargs="+", required=True)
    ns = ap.parse_args(["--method", "RSS", "FOI"])
    assert ns.method == ["RSS", "FOI"]


def test_run_seed_chain_continuous_history():
    from optimize_mask import run_seed_chain
    amp, T, pos, sint, ov = _setup()
    cfg = dict(oversample=ov, iters=6, backend="scipy", n_spots=2)
    r = run_seed_chain(cfg, T, pos, sint, ["RSS", "FOI"], seed=0, amp=amp)
    # continuous histories equal length, span both stages
    assert len(r["rss_history"]) == len(r["foi_history"])
    assert len(r["rss_history"]) >= 2
    # exactly one stage boundary for a 2-stage chain, within the history range
    assert len(r["stage_bounds"]) == 1
    assert 0 < r["stage_bounds"][0] < len(r["rss_history"])
    assert r["stage_labels"] == ["RSS→FOI"]
    # per-stage records for metadata
    assert [s["method"] for s in r["stages"]] == ["RSS", "FOI"]
    # final_cost is the last stage's (FOI) active cost
    assert np.isclose(r["final_cost"], r["stages"][-1]["final_cost"])
    # metrics present
    for k in ("u", "e", "v"):
        assert k in r
