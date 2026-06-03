"""Cost functions and gradients for optimization-based CGH design.

Two cost functions from the paper / Hamamatsu note:

* **FOI** (Fidelity Of Intensity, Eq. 3 / Hamamatsu p4):
  ``f_FOI = - sum_i Itilde_i Ttilde_i`` with Euclidean (L2) normalisation
  ``Itilde = I / ||I||_2``, ``Ttilde = T / ||T||_2``.
* **RSS** (Residual Sum of Squares, Eq. 2): ``f_RSS = sum_i (I_i - T_i)^2`` with
  total-sum normalisation. Because the unitary (ortho) FFT conserves energy and
  ``amp`` is L2-normalised, ``sum_i I_i == 1`` automatically, so ``I`` is already
  sum-normalised; ``T`` is normalised to unit sum by the caller.

The analytic gradient (paper Eq. 4 / Hamamatsu p5) is, for any cost ``f(I)``::

    df/dtheta_j = 2 * Im[ conj(psi_j) * crop_NN( IFFT( (df/dI) * Psi ) )_j ]

with FFTs taken in ``norm="ortho"`` so that ``sum_i u*_ij X_i == IFFT(X)_j``.
For FOI, ``df/dI = (-Ttilde + f * Itilde) / N_I``; for RSS, ``df/dI = 2 (I - T)``.

All arrays are in **natural FFT ordering** (not fftshifted). Build the target with
:func:`foitweezers.targets.square_lattice_target` (centered) and ``ifftshift`` it
before passing here; :mod:`foitweezers.design` handles this.

The numpy ``*_cost_grad`` functions feed :func:`scipy.optimize.minimize` (the
faithful conjugate-gradient path). The torch ``*Loss`` modules plug into
``slmsuite``'s ``Hologram.optimize(method="CG", loss=...)`` GPU path.
"""

from __future__ import annotations

import numpy as np

from .forward import _embed, _crop


def _grad_from_dfdI(dfdI, Psi, psi_nn, m, n, xp=np):
    """Generic chain-rule gradient: ``2 Im[conj(psi) crop(ifft2(dfdI * Psi))]``."""
    back = xp.fft.ifft2(dfdI * Psi, norm="ortho")
    back_nn = _crop(back, n, xp=xp)
    return 2.0 * xp.imag(xp.conj(psi_nn) * back_nn)


def _forward_natural(phase_nn, amp, oversample, xp=np):
    """Return (psi_nn, Psi_mm, I_mm) in natural FFT ordering."""
    n = amp.shape[0]
    m = n * oversample
    psi = amp * xp.exp(1j * phase_nn)
    Psi = xp.fft.fft2(_embed(psi, m, xp=xp), norm="ortho")
    I = xp.abs(Psi) ** 2
    return psi, Psi, I


def foi_cost_grad(phase_nn, amp, T_nat, oversample, xp=np):
    """FOI cost and gradient (both numpy/cupy), natural ordering.

    Parameters
    ----------
    phase_nn : ndarray (n, n)
        Nearfield phase (the optimization variable).
    amp : ndarray (n, n)
        Nearfield amplitude (L2-normalised; zero outside the aperture).
    T_nat : ndarray (m, m)
        Target intensity in natural ordering (``m = n*oversample``).
    oversample : int

    Returns
    -------
    f : float
    grad : ndarray (n, n)
    """
    n = amp.shape[0]
    m = n * oversample
    psi, Psi, I = _forward_natural(phase_nn, amp, oversample, xp=xp)

    N_I = xp.sqrt(xp.sum(I * I))
    N_T = xp.sqrt(xp.sum(T_nat * T_nat))
    Itil = I / N_I
    Ttil = T_nat / N_T

    f = -float(xp.sum(Itil * Ttil))
    # d f / d I, derived exactly: includes the N_I = ||I||_2 dependence.
    # (The transcribed paper Eq.4 / Hamamatsu pseudocode shows +f*Itil, but f<0 and
    # the rigorous derivative carries -f; verified to 4e-12 vs finite differences.)
    dfdI = (-Ttil - f * Itil) / N_I
    grad = _grad_from_dfdI(dfdI, Psi, psi, m, n, xp=xp)
    return f, grad


def rss_cost_grad(phase_nn, amp, T_nat, oversample, xp=np):
    """RSS cost and gradient, natural ordering. ``T_nat`` should be sum-normalised."""
    n = amp.shape[0]
    m = n * oversample
    psi, Psi, I = _forward_natural(phase_nn, amp, oversample, xp=xp)
    # I already sums to 1 (energy conservation); T_nat normalised by caller.
    diff = I - T_nat
    f = float(xp.sum(diff * diff))
    dfdI = 2.0 * diff
    grad = _grad_from_dfdI(dfdI, Psi, psi, m, n, xp=xp)
    return f, grad


COST_GRAD = {"FOI": foi_cost_grad, "RSS": rss_cost_grad}


# --------------------------------------------------------------------------- #
# torch modules for the slmsuite GPU path: loss(farfield_complex, target_amp)  #
# --------------------------------------------------------------------------- #
try:  # torch is optional
    import torch

    class FOILoss(torch.nn.Module):
        """FOI loss for slmsuite ``optimize(method="CG", loss=FOILoss())``.

        ``farfield`` is the complex farfield (gradient intact); ``target`` is the
        target **amplitude** (slmsuite convention), so the target intensity is
        ``target**2``.
        """

        def forward(self, farfield, target):
            I = torch.abs(farfield) ** 2
            T = target ** 2
            Itil = I / torch.linalg.vector_norm(I)
            Ttil = T / torch.linalg.vector_norm(T)
            return -torch.sum(Itil * Ttil)

    class RSSLoss(torch.nn.Module):
        """RSS loss for the slmsuite GPU path (sum-normalised intensities)."""

        def forward(self, farfield, target):
            I = torch.abs(farfield) ** 2
            T = target ** 2
            I = I / torch.sum(I)
            T = T / torch.sum(T)
            return torch.sum((I - T) ** 2)

    TORCH_LOSSES = {"FOI": FOILoss, "RSS": RSSLoss}
except Exception:  # pragma: no cover - torch absent
    torch = None
    TORCH_LOSSES = {}
