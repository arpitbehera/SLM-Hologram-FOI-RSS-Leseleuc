"""Configuration for the FOI hologram reproduction.

Two presets are provided:

* :data:`PAPER` runs the SLM-native size (N=1200 coarse grid -> a 1200x1200 phase
  mask matching the 1200-row device, 10x zero-pad to a 12000x12000 reproduction
  FFT, 1000 CGM iterations, 20 random seeds). This is a GPU-scale workload (the
  paper quotes 10-20 min on a GPU per hologram at N=800; N=1200 is ~2.5x heavier)
  and is intended for the user's CUDA machine.
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

    wavelength_nm: float = 852.0          # design / numerical wavelength
    na: float = 0.7                       # objective NA (air)
    slm_pixel_um: float = 8.0             # Meadowlark E19x12-500-1200-HDM8 pixel pitch
    slm_shape_px: Tuple[int, int] = (1200, 1920)  # (rows, cols) of the device

    @property
    def rayleigh_um(self) -> float:
        """Scalar Rayleigh / Airy radius r_A = 0.61 * lambda / NA, in micrometres."""
        return 0.61 * (self.wavelength_nm * 1e-3) / self.na


@dataclass(frozen=True)
class SimConfig:
    """Numerical reproduction parameters (CGM solver + FFT grid).

    TODO(dead-config): every field below is currently a dead "user variable" --
    the script run path (scripts/_runner.py) defines its own PRESETS dict and only
    imports OpticsConfig, so editing values here has no effect on figures/tables.
    Pick one source of truth: either (a) make _runner derive its PRESETS from these
    dataclasses (route B) and delete the PRESETS literal, or (b) delete this dead
    config and keep tunables in the CLI layer. Until resolved, do NOT assume a knob
    set here is honoured by a run.
    """

    n: int = 1200                 # coarse hologram grid side (N x N) = SLM mask size
    oversample: int = 10          # zero-pad factor -> reproduction FFT is M = n*oversample
    aperture_radius_frac: float = 540.0 / 1200.0  # tophat radius / n (=0.45); radius_px = frac*n. see make_aperture
    iters: int = 1000             # CGM iterations
    conv_tol: float = 1e-8        # relative cost decrease per iteration to stop
    seeds: Tuple[int, ...] = tuple(range(20))   # random seeds for statistics
    quantize_bits: int = 8        # SLM phase quantization (256 levels)
    spacings_px: Tuple[float, ...] = (1.6, 1.8, 2.1, 2.5)  # coarse target-plane px
    # TODO: lattice_n (and SimConfig/PAPER/CPU_DEV) are currently dead in the
    # script run path -- scripts/_runner.py defines its own PRESETS dict and only
    # imports OpticsConfig. Array size is set via the --n-spots CLI flag instead.
    # Unify: make _runner PRESETS derive from SimConfig so this is the single
    # source of truth, with CLI overriding it (route B). Until then, keep n_spots
    # canonical in the CLI layer and do NOT duplicate the value here.
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


# SLM-native preset, N=1200 mask (run on the GPU machine).
PAPER = SimConfig()

# CPU development/validation preset: same ratios, far cheaper.
CPU_DEV = SimConfig(
    n=160,
    oversample=10,
    iters=300,
    seeds=(0, 1, 2),
    spacings_px=(1.6, 1.8, 2.1, 2.5),
)
