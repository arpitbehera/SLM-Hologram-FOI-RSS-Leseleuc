"""Spot-array metrics: uniformity, light-utilization efficiency, VP ratio.

Definitions follow the paper (Sec. IV, Eq. 5 and Table I):

* **Spot power** of spot k = sum of pixel intensities in a square window of
  half-width ``hw`` centered on the spot (paper integrates over
  ``[x_k - w0/2, x_k + w0/2] x [y_k - w0/2, y_k + w0/2]``).
* **Uniformity** (relative SD) = ``std(powers) / mean(powers)``.
* **Efficiency** = in-spot power / total incident power. With the unitary FFT and
  L2-normalised aperture the total farfield power is 1, so efficiency is just the
  summed in-spot fraction (the numerical definition in Table I).
* **VP ratio** = mean valley intensity / mean peak intensity, where peaks are the
  spot centers and valleys are the midpoints between nearest-neighbour spots.

Intensities here are the **centered** reproduction (``reproduce_intensity(...,
shift=True)``), matching the centered target positions.
"""

from __future__ import annotations

import numpy as np


def default_half_window(spacing_int):
    """Non-overlapping integration window half-width from the integer spacing."""
    return max(1, int(spacing_int // 2))


def spot_powers(I, positions, half_window):
    """Summed intensity in a ``(2hw+1)^2`` box around each spot position."""
    hw = int(half_window)
    powers = []
    for (r, c) in positions:
        r0, r1 = max(0, r - hw), min(I.shape[0], r + hw + 1)
        c0, c1 = max(0, c - hw), min(I.shape[1], c + hw + 1)
        powers.append(float(np.sum(I[r0:r1, c0:c1])))
    return np.asarray(powers)


def uniformity(powers):
    """Relative standard deviation SD/mean of the spot powers."""
    powers = np.asarray(powers, dtype=np.float64)
    m = np.mean(powers)
    if m == 0:
        return np.nan
    return float(np.std(powers) / m)


def efficiency(I, positions, half_window, total=None):
    """Fraction of power delivered to the trap windows.

    ``total`` defaults to ``sum(I)`` (the incident power, which equals 1 for the
    energy-conserving unitary forward model).
    """
    p = spot_powers(I, positions, half_window)
    if total is None:
        total = float(np.sum(I))
    if total == 0:
        return np.nan
    return float(np.sum(p) / total)


def _grid_from_positions(positions, n_spots):
    """Reshape the flat position list into an (n_spots, n_spots) array of (r, c)."""
    arr = np.array(positions).reshape(n_spots, n_spots, 2)
    return arr


def vp_ratio(I, positions, n_spots=5):
    """Valley-to-peak ratio of the spot array (paper Sec. IV C)."""
    grid = _grid_from_positions(positions, n_spots)
    # Peaks: intensity at each spot center.
    peaks = [float(I[r, c]) for r, c in positions]
    peak = float(np.mean(peaks))

    # Valleys: midpoints between horizontally/vertically adjacent spots.
    valleys = []
    for i in range(n_spots):
        for j in range(n_spots):
            r, c = grid[i, j]
            if j + 1 < n_spots:
                r2, c2 = grid[i, j + 1]
                valleys.append(float(I[(r + r2) // 2, (c + c2) // 2]))
            if i + 1 < n_spots:
                r2, c2 = grid[i + 1, j]
                valleys.append(float(I[(r + r2) // 2, (c + c2) // 2]))
    valley = float(np.mean(valleys))
    if peak == 0:
        return np.nan
    return valley / peak


def vp_ratio_triangular(I, positions, spacing_int):
    """Valley-to-peak ratio for a triangular lattice.

    Nearest-neighbour pairs are those at Euclidean distance < 1.5 * spacing_int
    (NN distance is ~spacing_int; the next shell is ~sqrt(3)*spacing_int). Valleys
    are the pair-midpoint intensities; peaks are the site intensities.
    """
    s = int(spacing_int)
    peaks = [float(I[r, c]) for (r, c) in positions]
    peak = float(np.mean(peaks)) if peaks else 0.0
    thr2 = (1.5 * s) ** 2
    valleys = []
    n = len(positions)
    for i in range(n):
        r1, c1 = positions[i]
        for j in range(i + 1, n):
            r2, c2 = positions[j]
            if (r1 - r2) ** 2 + (c1 - c2) ** 2 <= thr2:
                valleys.append(float(I[(r1 + r2) // 2, (c1 + c2) // 2]))
    if peak == 0 or not valleys:
        return np.nan
    return float(np.mean(valleys)) / peak


def evaluate(I, positions, spacing_int, n_spots=5, half_window=None, lattice="square"):
    """Convenience: return all Table-I / Fig-4 metrics for one reproduced image.

    ``lattice`` selects the valley-to-peak computation: "square" uses the
    axis-aligned N x N grid path; "triangular" uses nearest-neighbour midpoints.
    """
    if half_window is None:
        half_window = default_half_window(spacing_int)
    p = spot_powers(I, positions, half_window)
    if lattice == "triangular":
        vp = vp_ratio_triangular(I, positions, spacing_int)
    else:
        vp = vp_ratio(I, positions, n_spots)
    return {
        "uniformity": uniformity(p),
        "efficiency": efficiency(I, positions, half_window),
        "vp_ratio": vp,
        "half_window": half_window,
        "spot_powers": p,
    }
