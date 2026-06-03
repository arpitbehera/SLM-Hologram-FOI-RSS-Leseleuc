"""Synthetic residual aberration for the Table I middle row.

The paper adds a measured plan-fluorite objective aberration (~0.02 lambda RMS,
Ref [35], not provided) to the *reproduction* of FOI/RSS holograms and reports
that uniformity degrades while efficiency is essentially unchanged. We synthesise
an equivalent low-order Zernike phase at a target RMS (in waves) and apply it as an
extra phase on the nearfield aperture during reproduction. Results are
"informational" in the paper too, so we reproduce the *trend*, not exact numbers.

Uses slmsuite's Zernike basis when available; falls back to a small analytic set.
"""

from __future__ import annotations

import numpy as np


def _analytic_zernike(n, radius_px):
    """A few low-order Zernike modes on the aperture grid (ANSI: astig, coma, spherical)."""
    c = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    rho = np.sqrt(xx ** 2 + yy ** 2) / radius_px
    theta = np.arctan2(yy, xx)
    mask = rho <= 1.0
    modes = {
        "astig_obliq": np.sqrt(6) * rho ** 2 * np.sin(2 * theta),   # Z(-2,2)
        "astig_vert": np.sqrt(6) * rho ** 2 * np.cos(2 * theta),    # Z(2,2)
        "coma_x": np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.cos(theta),  # Z(1,3)
        "coma_y": np.sqrt(8) * (3 * rho ** 3 - 2 * rho) * np.sin(theta),  # Z(-1,3)
        "spherical": np.sqrt(5) * (6 * rho ** 4 - 6 * rho ** 2 + 1),      # Z(0,4)
    }
    for k in modes:
        modes[k] = modes[k] * mask
    return modes, mask


def aberration_phase(n, radius_px, rms_waves=0.02, weights=None, seed=0):
    """Return an aberration phase map (radians) of approximately ``rms_waves`` RMS.

    ``weights`` optionally fixes the per-mode mix; otherwise a random mix (seeded)
    over the low-order modes is used.
    """
    modes, mask = _analytic_zernike(n, radius_px)
    keys = list(modes.keys())
    if weights is None:
        rng = np.random.default_rng(seed)
        w = rng.normal(size=len(keys))
    else:
        w = np.array([weights[k] for k in keys], dtype=float)

    phi = np.zeros((n, n))
    for wk, k in zip(w, keys):
        phi += wk * modes[k]

    # Scale to the requested RMS (in waves) over the aperture, then -> radians.
    inside = mask
    cur_rms = np.sqrt(np.mean(phi[inside] ** 2))
    if cur_rms > 0:
        phi = phi / cur_rms * rms_waves
    return phi * 2 * np.pi * mask
