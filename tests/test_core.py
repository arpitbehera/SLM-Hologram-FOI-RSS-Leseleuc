"""Unit tests for the FOI/RSS forward model, gradients, and metrics."""

import os
import sys
import argparse

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from foitweezers.forward import make_aperture, reproduce_intensity, quantize_phase
from foitweezers.targets import square_lattice_target
from foitweezers.losses import foi_cost_grad, rss_cost_grad
from foitweezers.metrics import uniformity, efficiency, vp_ratio, spot_powers
from scripts._runner import (
    DEFAULT_ILLUMINATION,
    APERTURE_FRAC,
    add_common_args,
    make_illumination,
    resolve,
)
from scripts.fig3 import _ensure_axes_grid


def _setup(n=24, ov=4, spots=2, spacing=2.0, seed=0):
    amp = make_aperture(n, radius_px=0.45 * n)
    m = n * ov
    T, pos, sint = square_lattice_target(m, spacing_fine_px=spacing * ov, n_spots=spots)
    T_nat = np.fft.ifftshift(T)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, (n, n))
    return amp, T_nat, T, pos, sint, phase, ov


def test_energy_conserved():
    amp = make_aperture(40, radius_px=15)
    assert np.isclose(np.sum(amp ** 2), 1.0)
    phase = np.zeros((40, 40))
    I = reproduce_intensity(phase, amp, oversample=3)
    assert np.isclose(np.sum(I), 1.0, rtol=1e-10)


def test_gaussian_aperture_is_l2_normalized_and_truncated():
    n = 40
    radius = APERTURE_FRAC * n
    amp = make_aperture(n, radius_px=radius, profile="gaussian", gauss_radius_px=radius)
    c = (n - 1) / 2.0
    y, x = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    outside = x * x + y * y > radius ** 2

    assert np.isclose(np.sum(amp ** 2), 1.0)
    assert np.all(amp[outside] == 0.0)
    assert amp[n // 2, n // 2] > amp[n // 2, int(n // 2 + radius * 0.8)]


def test_resolve_defaults_to_gaussian_illumination():
    parser = add_common_args(argparse.ArgumentParser())
    cfg = resolve(parser.parse_args([]))

    assert DEFAULT_ILLUMINATION == "gaussian"
    assert cfg["illumination"] == "gaussian"
    assert cfg["illumination_profile"] == "gaussian"
    assert cfg["gauss_radius_px"] == pytest.approx(APERTURE_FRAC * cfg["n"])


def test_tophat_illumination_remains_selectable():
    parser = add_common_args(argparse.ArgumentParser())
    cfg = resolve(parser.parse_args(["--illumination", "tophat"]))
    amp = make_illumination(cfg)

    assert cfg["illumination"] == "tophat"
    assert cfg["illumination_profile"] == "tophat"
    assert cfg["gauss_radius_px"] is None
    assert np.isclose(np.sum(amp ** 2), 1.0)
    assert len(np.unique(amp[amp > 0])) == 1


def test_fig3_axes_grid_handles_single_spacing_column():
    axes = np.array(["foi-axis", "rss-axis"], dtype=object)

    grid = _ensure_axes_grid(axes, n_rows=2, n_cols=1)

    assert grid.shape == (2, 1)
    assert grid[0, 0] == "foi-axis"
    assert grid[1, 0] == "rss-axis"


@pytest.mark.parametrize("cost_grad", [foi_cost_grad, rss_cost_grad])
def test_gradient_matches_finite_difference(cost_grad):
    amp, T_nat, _, _, _, phase, ov = _setup()
    f0, g = cost_grad(phase, amp, T_nat, ov)
    eps = 1e-6
    rng = np.random.default_rng(1)
    ys, xs = np.where(amp > 0)
    idx = rng.choice(len(ys), 15, replace=False)
    for k in idx:
        p2 = phase.copy(); p2[ys[k], xs[k]] += eps
        p3 = phase.copy(); p3[ys[k], xs[k]] -= eps
        num = (cost_grad(p2, amp, T_nat, ov)[0] - cost_grad(p3, amp, T_nat, ov)[0]) / (2 * eps)
        assert abs(num - g[ys[k], xs[k]]) < 1e-6


def test_foi_cost_negative_and_bounded():
    amp, T_nat, _, _, _, phase, ov = _setup()
    f, _ = foi_cost_grad(phase, amp, T_nat, ov)
    # FOI = -<Itilde, Ttilde> is a negative cosine similarity in [-1, 0].
    assert -1.0 - 1e-9 <= f <= 0.0 + 1e-9


def test_quantize_levels():
    rng = np.random.default_rng(0)
    phase = rng.uniform(-10, 10, (16, 16))
    q = quantize_phase(phase, bits=8)
    levels = np.unique(np.round(q / (2 * np.pi) * 256).astype(int) % 256)
    assert q.min() >= 0 and q.max() < 2 * np.pi
    assert len(levels) <= 256


def test_metrics_on_ideal_array():
    # Place delta peaks directly and check uniformity ~ 0, efficiency ~ 1, vp ~ 0.
    m, ov = 200, 1
    T, pos, sint = square_lattice_target(m, spacing_fine_px=20, n_spots=5)
    p = spot_powers(T, pos, half_window=2)
    assert uniformity(p) < 1e-9
    assert efficiency(T, pos, half_window=2) == pytest.approx(1.0, abs=1e-9)
    assert vp_ratio(T, pos, n_spots=5) == pytest.approx(0.0, abs=1e-12)


def test_foi_beats_rss_uniformity_small():
    """FOI should produce a more uniform array than RSS (paper's central result)."""
    from foitweezers.design import design_cgh

    n, ov = 64, 5
    amp = make_aperture(n, radius_px=0.45 * n)
    m = n * ov
    T, pos, sint = square_lattice_target(m, spacing_fine_px=2.5 * ov, n_spots=5)
    foi = design_cgh(T, amp, ov, method="FOI", seed=0, iters=80, backend="scipy")
    rss = design_cgh(T, amp, ov, method="RSS", seed=0, iters=80, backend="scipy")
    I_foi = reproduce_intensity(foi, amp, ov, shift=True)
    I_rss = reproduce_intensity(rss, amp, ov, shift=True)
    u_foi = uniformity(spot_powers(I_foi, pos, sint // 2))
    u_rss = uniformity(spot_powers(I_rss, pos, sint // 2))
    assert u_foi < u_rss


def test_save_mask_uint8_dtype_and_mapping(tmp_path):
    from scripts.optimize_mask import save_mask_uint8
    phase = np.array([[0.0, np.pi], [2 * np.pi * 255 / 256, 2 * np.pi]])
    stem = str(tmp_path / "m")
    save_mask_uint8(stem, phase)
    codes = np.load(stem + ".npz")["phase_uint8"]
    assert codes.dtype == np.uint8
    assert codes[0, 0] == 0
    assert codes[0, 1] == 128
    assert codes[1, 0] == 255
    assert codes[1, 1] == 0
    assert os.path.exists(stem + ".png")


def test_aggregate_stats_single_seed_no_std():
    from scripts.optimize_mask import aggregate_stats
    agg = aggregate_stats([dict(u=0.5, e=0.1, v=0.2)])
    assert agg["uniformity_mean"] == 0.5
    assert agg["efficiency_mean"] == 0.1
    assert agg["vp_mean"] == 0.2
    assert "uniformity_std" not in agg
    assert "efficiency_std" not in agg
    assert "vp_std" not in agg


def test_aggregate_stats_multi_seed_has_std():
    from scripts.optimize_mask import aggregate_stats
    agg = aggregate_stats([dict(u=0.4, e=0.1, v=0.2), dict(u=0.6, e=0.3, v=0.4)])
    assert np.isclose(agg["uniformity_mean"], 0.5)
    assert np.isclose(agg["uniformity_std"], np.std([0.4, 0.6], ddof=1))
    assert "efficiency_std" in agg and "vp_std" in agg


def test_select_best_picks_min_final_cost():
    from scripts.optimize_mask import select_best
    runs = [dict(seed=0, final_cost=-0.5), dict(seed=1, final_cost=-0.9),
            dict(seed=2, final_cost=-0.7)]
    assert select_best(runs)["seed"] == 1


def test_optimize_mask_cli_end_to_end(tmp_path):
    import json
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..")
    subprocess.run(
        [sys.executable, "scripts/optimize_mask.py", "--method", "FOI",
         "--preset", "tiny", "--seeds", "1", "--iters", "15", "--tag", "test1"],
        cwd=repo, check=True,
    )
    stem = os.path.join(repo, "outputs", "optmask_FOI_sp1.8_test1")
    for suf in ("_phase.png", "_phase.npz", "_image.png", "_image.npz", "_meta.json"):
        assert os.path.exists(stem + suf), suf
    assert np.load(stem + "_phase.npz")["phase_uint8"].dtype == np.uint8
    meta = json.load(open(stem + "_meta.json"))["results"]
    assert meta["n_seeds"] == 1
    assert "uniformity_std" not in meta
    assert meta["best"]["seed"] == 0
    assert meta["uniformity_mean"] == meta["best"]["uniformity"]

    subprocess.run(
        [sys.executable, "scripts/optimize_mask.py", "--method", "RSS",
         "--preset", "tiny", "--seeds", "3", "--iters", "15", "--tag", "test3"],
        cwd=repo, check=True,
    )
    stem3 = os.path.join(repo, "outputs", "optmask_RSS_sp1.8_test3")
    assert os.path.exists(stem3 + "_convergence.png")
    meta3 = json.load(open(stem3 + "_meta.json"))["results"]
    assert meta3["n_seeds"] == 3
    assert "uniformity_std" in meta3 and "vp_std" in meta3
    assert "best" in meta3 and "uniformity_std" not in meta3["best"]
