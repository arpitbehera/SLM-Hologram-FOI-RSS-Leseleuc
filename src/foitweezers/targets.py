"""Target intensity patterns ``T_i`` on the (centered) reproduction grid.

Phase 1 implements the 5x5 square lattice used in Fig 1(b), Fig 3 and Table I.
Non-square lattices (Fig 5: hexagonal/kagome/triangular) are Phase 2.

Targets are defined on the centered (``fftshift``-ed) M x M farfield grid, matching
:func:`foitweezers.forward.reproduce_intensity` with ``shift=True``. Spots are
single-pixel unit-intensity peaks; ``spacing_fine_px`` is the spacing on this fine
grid, i.e. ``spacing_coarse_px * oversample``.
"""

from __future__ import annotations

import numpy as np


def square_lattice_positions(m, spacing_fine_px, n_spots=5):
    """Integer (row, col) pixel positions of an ``n_spots x n_spots`` square lattice.

    Centered on the grid. Spacing is rounded to the nearest integer pixel so that
    spots land on grid points (the paper's targets are likewise on grid points).
    """
    s = int(round(spacing_fine_px))
    if s < 1:
        raise ValueError("spacing rounds to < 1 fine pixel; increase oversample.")
    c = m // 2
    offs = (np.arange(n_spots) - (n_spots - 1) / 2.0) * s
    offs = np.round(offs).astype(int)
    rows = c + offs
    cols = c + offs
    pos = [(int(r), int(col)) for r in rows for col in cols]
    return pos, s


def square_lattice_target(m, spacing_fine_px, n_spots=5, xp=np):
    """Build the square-lattice target intensity ``T`` (M x M) and its spot list.

    Returns
    -------
    T : ndarray (m, m)
        Unit-intensity peaks at the lattice sites, zero elsewhere.
    positions : list[(int, int)]
        (row, col) of each spot, for fitting / metrics.
    spacing_int : int
        The integer fine-pixel spacing actually used.
    """
    positions, spacing_int = square_lattice_positions(m, spacing_fine_px, n_spots)
    T = xp.zeros((m, m), dtype=xp.float64)
    for (r, c) in positions:
        T[r, c] = 1.0
    return T, positions, spacing_int
