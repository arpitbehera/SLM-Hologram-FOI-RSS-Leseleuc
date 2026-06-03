"""Scalar Fraunhofer forward model: nearfield phase -> farfield intensity.

Implements the paper's Eq. (1) / Hamamatsu flowchart (p2):

    psi_j  = a_j * exp(i * theta_j)         (input plane, N x N)
    Psi_i  = sum_j u_ij psi_j  = FFT(psi)   (output plane, M x M, M = N*oversample)
    I_i    = |Psi_i|^2

The N x N nearfield (the optimization domain, ``a_j`` zero outside a circular
top-hat aperture) is zero-padded by ``oversample`` and Fourier transformed; this
is the Kim 2019 upsampling (paper ref [29]) that gives the reproduction a 10x
finer image-plane sampling than the coarse target grid.

A unitary (``norm="ortho"``) FFT is used throughout so that the analytic FOI
gradient in :mod:`foitweezers.losses` takes its clean inverse-FFT form and energy
is conserved between the planes.

Every function accepts an ``xp`` module (``numpy`` by default, pass ``cupy`` on a
GPU) so the same code runs on CPU and CUDA.
"""

from __future__ import annotations

import numpy as np


def make_aperture(n, radius_px, profile="tophat", gauss_radius_px=None, xp=np):
    """Build the nearfield source amplitude ``a_j`` on an ``n x n`` grid.

    Parameters
    ----------
    n : int
        Grid side length.
    radius_px : float
        Aperture radius in pixels (the illuminated/active circular region).
    profile : {"tophat", "gaussian"}
        ``"tophat"`` -> uniform amplitude inside the circle (paper's design
        assumption). ``"gaussian"`` -> truncated Gaussian inside the circle
        (Appendix A / Table II), with 1/e^2 radius ``gauss_radius_px``.
    gauss_radius_px : float OR None
        1/e^2 radius of the Gaussian in pixels (only for ``profile="gaussian"``).
    xp : module
        Array module (``numpy`` or ``cupy``).

    Returns
    -------
    amp : ndarray (n, n)
        Real, non-negative amplitude, normalized to unit L2 norm.
    """
    c = (n - 1) / 2.0
    y, x = xp.meshgrid(xp.arange(n) - c, xp.arange(n) - c, indexing="ij")
    r2 = x * x + y * y
    inside = r2 <= (radius_px ** 2)

    if profile == "tophat":
        amp = inside.astype(xp.float64)
    elif profile == "gaussian":
        if gauss_radius_px is None:
            raise ValueError("gaussian profile needs gauss_radius_px (1/e^2 radius).")
        # Truncated Gaussian: exp(-2 r^2 / w^2) inside the aperture, 0 outside.
        amp = xp.exp(-2.0 * r2 / (gauss_radius_px ** 2)) * inside
    else:
        raise ValueError(f"unknown profile {profile!r}")

    norm = xp.sqrt(xp.sum(amp * amp))
    if norm == 0:
        raise ValueError("aperture is empty (radius too small).")
    return amp / norm


def _embed(field_nn, m, xp=np):
    """Center-embed an ``n x n`` field into an ``m x m`` zero array."""
    n = field_nn.shape[0]
    if m < n:
        raise ValueError("oversampled size m must be >= n.")
    out = xp.zeros((m, m), dtype=field_nn.dtype)
    off = (m - n) // 2
    out[off:off + n, off:off + n] = field_nn
    return out


def _crop(field_mm, n, xp=np):
    """Inverse of :func:`_embed`: take the center ``n x n`` block."""
    m = field_mm.shape[0]
    off = (m - n) // 2
    return field_mm[off:off + n, off:off + n]


def farfield(phase, amp, oversample, xp=np):
    """Complex farfield ``Psi`` (M x M), in natural FFT ordering (not shifted).

    Suitable for the gradient math in :mod:`foitweezers.losses`.
    """
    n = amp.shape[0]
    m = n * oversample
    psi = amp * xp.exp(1j * phase)
    Psi = xp.fft.fft2(_embed(psi, m, xp=xp), norm="ortho")
    return Psi


def reproduce_intensity(phase, amp, oversample, shift=True, xp=np):
    """Reproduced farfield intensity ``I = |Psi|^2`` (M x M).

    Parameters
    ----------
    shift : bool
        If True, ``fftshift`` so the zero spatial frequency is centered (for
        display and for spot-array targets that are defined about the center).
    """
    Psi = farfield(phase, amp, oversample, xp=xp)
    if shift:
        Psi = xp.fft.fftshift(Psi)
    return xp.abs(Psi) ** 2


def quantize_phase(phase, bits=8, xp=np):
    """Wrap to [0, 2pi) and quantize to ``2**bits`` levels (SLM 8-bit emulation)."""
    levels = 2 ** bits
    wrapped = xp.mod(phase, 2 * np.pi)
    q = xp.round(wrapped / (2 * np.pi) * levels)
    q = xp.mod(q, levels)
    return q * (2 * np.pi / levels)
