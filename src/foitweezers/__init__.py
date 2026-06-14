"""foitweezers - reproduction of the optimization-based (FOI) hologram design method.

Reference: K. Nishimura, H. Sakai, T. Ando, T. Tomita, S. de Leseleuc,
"Optimization-based hologram design for fine optical tweezer arrays and extension
of super-resolution criteria," Phys. Rev. A 113, 013119 (2026).

Phase 1 scope: Fig 1(b) CGHs, Fig 3 (FOI vs RSS 5x5 at several pitches), Table I
(spot uniformity & light-utilization efficiency).
"""

from .config import OpticsConfig, SimConfig, CPU_DEV, PAPER  # noqa: F401
from .forward import reproduce_intensity, make_aperture  # noqa: F401
from .targets import (  # noqa: F401
    square_lattice_target,
    triangular_lattice_target,
    triangular_lattice_positions,
)
from .design import design_cgh  # noqa: F401
from .metrics import spot_powers, uniformity, efficiency, vp_ratio  # noqa: F401

__all__ = [
    "OpticsConfig",
    "SimConfig",
    "CPU_DEV",
    "PAPER",
    "reproduce_intensity",
    "make_aperture",
    "square_lattice_target",
    "triangular_lattice_target",
    "triangular_lattice_positions",
    "design_cgh",
    "spot_powers",
    "uniformity",
    "efficiency",
    "vp_ratio",
]
