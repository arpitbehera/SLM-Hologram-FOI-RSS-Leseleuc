"""Unit tests for triangular-lattice targets and triangular metrics."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from foitweezers.targets import (
    triangular_lattice_positions,
    triangular_lattice_target,
    n_triangular_sites,
)


@pytest.mark.parametrize("radius,expected", [(0, 1), (1, 7), (2, 19), (3, 37)])
def test_triangular_site_count(radius, expected):
    m = 400
    positions, s = triangular_lattice_positions(m, spacing_fine_px=20, radius=radius)
    assert n_triangular_sites(radius) == expected
    assert len(positions) == expected


def test_triangular_centered_and_deterministic():
    m = 400
    pos_a, s_a = triangular_lattice_positions(m, spacing_fine_px=20, radius=2)
    pos_b, s_b = triangular_lattice_positions(m, spacing_fine_px=20, radius=2)
    assert pos_a == pos_b  # deterministic
    assert s_a == 20
    assert (m // 2, m // 2) in pos_a  # center site present


def test_triangular_spacing_guard():
    with pytest.raises(ValueError):
        triangular_lattice_positions(400, spacing_fine_px=0.4, radius=2)


def test_triangular_target_shape_and_peaks():
    m = 400
    T, positions, s = triangular_lattice_target(m, spacing_fine_px=20, radius=2)
    assert T.shape == (m, m)
    assert float(T.sum()) == pytest.approx(len(positions))
    for r, c in positions:
        assert T[r, c] == 1.0


def test_triangular_exported_from_package():
    import foitweezers
    assert hasattr(foitweezers, "triangular_lattice_target")
    assert hasattr(foitweezers, "triangular_lattice_positions")


def test_vp_ratio_triangular_ideal_is_zero():
    from foitweezers.metrics import vp_ratio_triangular
    m = 400
    T, positions, s = triangular_lattice_target(m, spacing_fine_px=20, radius=2)
    # Ideal delta peaks: midpoints between neighbours are empty -> vp == 0.
    assert vp_ratio_triangular(T, positions, s) == pytest.approx(0.0, abs=1e-12)


def test_evaluate_triangular_path_runs():
    from foitweezers.metrics import evaluate
    m = 400
    T, positions, s = triangular_lattice_target(m, spacing_fine_px=20, radius=2)
    met = evaluate(T, positions, s, n_spots=5, lattice="triangular")
    assert met["uniformity"] < 1e-9
    assert met["efficiency"] == pytest.approx(1.0, abs=1e-9)
    assert met["vp_ratio"] == pytest.approx(0.0, abs=1e-12)


import argparse

from scripts._runner import add_common_args, resolve, build_target


def _resolve_cli(*cli):
    args = add_common_args(argparse.ArgumentParser()).parse_args(list(cli))
    return resolve(args)


def test_lattice_defaults_to_square():
    cfg = _resolve_cli("--preset", "tiny")
    assert cfg["lattice"] == "square"
    T, pos, sint = build_target(cfg, spacing_coarse=2.0)
    assert len(pos) == cfg["n_spots"] ** 2  # 5x5 = 25 by default


def test_build_target_triangular_19_sites():
    cfg = _resolve_cli("--preset", "tiny", "--lattice", "triangular", "--n-spots", "5")
    assert cfg["lattice"] == "triangular"
    T, pos, sint = build_target(cfg, spacing_coarse=2.0)
    assert len(pos) == 19  # radius = 5 // 2 = 2


def test_build_target_triangular_rejects_even_n_spots():
    cfg = _resolve_cli("--preset", "tiny", "--lattice", "triangular", "--n-spots", "4")
    with pytest.raises(ValueError, match="odd"):
        build_target(cfg, spacing_coarse=2.0)


def test_build_target_rejects_zero_n_spots():
    cfg = _resolve_cli("--preset", "tiny", "--n-spots", "0")
    with pytest.raises(ValueError):
        build_target(cfg, spacing_coarse=2.0)
