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
