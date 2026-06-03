"""Configuration for the FOI hologram reproduction.

Two presets are provided:

* :data:`PAPER` mirrors the paper exactly (N=800 coarse grid, 10x zero-pad to an
  8000x8000 reproduction FFT, 1000 CGM iterations, 20 random seeds). This is a
  GPU-scale workload (the paper quotes 10-20 min on a GPU per hologram) and is
  intended for the user's CUDA machine.
* :data:`CPU_DEV` keeps every *ratio* identical (aperture fill, spacing/r_A,
  oversample) but shrinks the grid and iteration/seed counts so the pipeline runs
  and validates on a CPU in minutes.

All "spot spacing" values are quoted in *coarse target-plane pixels*, exactly as
in the paper (1.6/1.8/2.1/2.5 px). The fine reproduction grid is ``oversample``
times finer, so a coarse spacing ``s`` corresponds to ``s * oversample`` fine
pixels (the Kim 2019 upsampling, paper ref [29]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class OpticsConfig:
    """Physical optics parameters (paper section III)."""

    wavelength_nm: float = 820.0          # design / numerical wavelength
    na: float = 0.75                      # objective NA (air), Nikon MRH00401
    slm_pixel_um: float = 12.5            # Hamamatsu X13138-01 pixel pitch
    slm_shape_px: Tuple[int, int] = (1024, 1272)  # (rows, cols) of the device

    @property
    def rayleigh_um(self) -> float:
        """Scalar Rayleigh / Airy radius r_A = 0.61 * lambda / NA, in micrometres."""
        return 0.61 * (self.wavelength_nm * 1e-3) / self.na


@dataclass(frozen=True)
class SimConfig:
    """Numerical reproduction parameters."""

    n: int = 800                  # coarse hologram grid side (N x N), = sqrt(DOF)
    oversample: int = 10          # zero-pad factor -> reproduction FFT is M = n*oversample
    aperture_radius_frac: float = 360.0 / 800.0  # top-hat radius as fraction of n/2*... see make_aperture
    iters: int = 1000             # CGM iterations
    conv_tol: float = 1e-8        # relative cost decrease per iteration to stop
    seeds: Tuple[int, ...] = tuple(range(20))   # random seeds for statistics
    quantize_bits: int = 8        # SLM phase quantization (256 levels)
    spacings_px: Tuple[float, ...] = (1.6, 1.8, 2.1, 2.5)  # coarse target-plane px
    lattice_n: int = 5            # 5 x 5 array
    dtype: str = "float64"        # double precision FFT, as in the paper

    @property
    def m(self) -> int:
        """Side length of the (square) reproduction FFT grid."""
        return self.n * self.oversample

    @property
    def aperture_radius_px(self) -> float:
        """Top-hat aperture radius on the coarse grid, in pixels."""
        return self.aperture_radius_frac * self.n


# Paper-faithful preset (run on the GPU machine).
PAPER = SimConfig()

# CPU development/validation preset: same ratios, far cheaper.
CPU_DEV = SimConfig(
    n=160,
    oversample=10,
    iters=300,
    seeds=(0, 1, 2),
    spacings_px=(1.6, 1.8, 2.1, 2.5),
)
