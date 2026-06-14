"""Shared helpers for the Phase-1 figure/table scripts.

Presets trade fidelity for runtime:

* ``tiny``  - n=96,  oversample=6,  iters=120, 1 seed   (~minutes on CPU; default)
* ``cpu``   - n=160, oversample=10, iters=300, 3 seeds  (slow on CPU; overnight)
* ``paper`` - n=1200, oversample=10, iters=1000, 20 seeds (GPU only; SLM-native 1200x1200 mask)

Runtime presets keep the same aperture fill (0.45) and report spacing in *coarse*
target-plane pixels (1.6/1.8/2.1/2.5), so the d/r_A ratios are preset-independent.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foitweezers.forward import make_aperture, reproduce_intensity  # noqa: E402
from foitweezers.targets import (  # noqa: E402
    square_lattice_target,
    triangular_lattice_target,
)
from foitweezers.design import design_cgh  # noqa: E402
from foitweezers.metrics import evaluate  # noqa: E402
from foitweezers.aberration import aberration_phase  # noqa: E402
from foitweezers.config import OpticsConfig  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

PRESETS = {
    "tiny": dict(n=96, oversample=6, iters=120, seeds=(0,)),
    "cpu": dict(n=160, oversample=10, iters=300, seeds=(0, 1, 2)),
    "paper": dict(n=1200, oversample=10, iters=1000, seeds=tuple(range(20))),
}
APERTURE_FRAC = 0.45
ILLUMINATION_PRESETS = {
    "gaussian": {"profile": "gaussian"},
    "tophat": {"profile": "tophat"},
}
DEFAULT_ILLUMINATION = "gaussian"
PRIMARY_SPACING = 1.8                        # coarse px; anchor spacing for Table I and Fig 1(b)
SPACINGS = (PRIMARY_SPACING, 2.1, 2.4, 2.7)

# --- spacing unit conversions -------------------------------------------------
# Spacings are designed in *coarse target-plane pixels*, but figures quote them in
# physical units. The Rayleigh radius in coarse px is set purely by the aperture
# fill (FFT geometry), so it is grid/preset-independent:
#     r_A = 1.22 * M / D_aperture / oversample = 1.22 / (2 * APERTURE_FRAC).
_OPT = OpticsConfig()
R_A_PX = 1.22 / (2.0 * APERTURE_FRAC)        # coarse px per Rayleigh radius
UM_PER_PX = _OPT.rayleigh_um / R_A_PX        # micrometres per coarse px
LAMBDA_REF_NM = 780.0                        # reference wavelength for "x.x λ" labels
LAMBDA_REF_UM = LAMBDA_REF_NM * 1e-3


def spacing_units(sp):
    """Convert a coarse-pixel spacing to physical units (dict of r_A / µm / λ)."""
    d_um = sp * UM_PER_PX
    return {
        "px": sp,
        "r_A": sp / R_A_PX,
        "um": d_um,
        "lambda": d_um / LAMBDA_REF_UM,
    }


def spacing_label(sp, fmt="plain"):
    """Multi-unit spacing label. ``fmt="mpl"`` uses mathtext for matplotlib."""
    u = spacing_units(sp)
    if fmt == "mpl":
        return (f"{u['r_A']:.2f}$\\,r_A$\n{u['um']:.2f} µm\n"
                f"{u['lambda']:.2f}$\\,\\lambda$")
    return f"{u['r_A']:.2f} r_A / {u['um']:.2f} um / {u['lambda']:.2f} lambda"


def add_common_args(p):
    p.add_argument("--preset", choices=list(PRESETS), default="tiny")
    p.add_argument(
        "--illumination",
        choices=list(ILLUMINATION_PRESETS),
        default=DEFAULT_ILLUMINATION,
        help="nearfield illumination amplitude preset (default: gaussian)",
    )
    p.add_argument("--backend", choices=["scipy", "torch", "slmsuite"], default="scipy")
    p.add_argument("--iters", type=int, default=None, help="override preset iterations")
    p.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    p.add_argument("--spacings", type=float, nargs="*", default=None)
    p.add_argument(
        "--lattice", choices=["square", "triangular"], default="square",
        help="target lattice geometry (default: square)",
    )
    p.add_argument(
        "--n-spots", type=int, default=5,
        help="square: array side length (N x N); "
             "triangular: must be odd, radius = n_spots // 2 "
             "(5 -> 19 sites, 7 -> 37 sites)",
    )
    return p


def resolve(args):
    cfg = dict(PRESETS[args.preset])
    if args.iters is not None:
        cfg["iters"] = args.iters
    if args.seeds is not None:
        cfg["seeds"] = tuple(range(args.seeds))
    cfg["spacings"] = tuple(args.spacings) if args.spacings else SPACINGS
    cfg["n_spots"] = args.n_spots
    cfg["lattice"] = args.lattice
    cfg["backend"] = args.backend
    cfg["aperture_radius_px"] = APERTURE_FRAC * cfg["n"]
    illum = ILLUMINATION_PRESETS[args.illumination]
    cfg["illumination"] = args.illumination
    cfg["illumination_profile"] = illum["profile"]
    cfg["gauss_radius_px"] = (
        cfg["aperture_radius_px"] if illum["profile"] == "gaussian" else None
    )
    os.makedirs(OUTDIR, exist_ok=True)
    return cfg


def make_illumination(cfg):
    """Build the resolved nearfield illumination amplitude."""
    return make_aperture(
        cfg["n"],
        radius_px=cfg["aperture_radius_px"],
        profile=cfg["illumination_profile"],
        gauss_radius_px=cfg["gauss_radius_px"],
    )


def build_target(cfg, spacing_coarse, n_spots=None):
    m = cfg["n"] * cfg["oversample"]
    if n_spots is None:
        n_spots = cfg.get("n_spots", 5)
    if n_spots < 1:
        raise ValueError(f"--n-spots must be >= 1; got {n_spots}")
    spacing_fine = spacing_coarse * cfg["oversample"]
    lattice = cfg.get("lattice", "square")
    if lattice == "triangular":
        if n_spots % 2 == 0:
            raise ValueError(
                f"triangular lattice requires an odd --n-spots "
                f"(1, 3, 5, ...); got {n_spots}"
            )
        radius = n_spots // 2
        T, pos, sint = triangular_lattice_target(
            m, spacing_fine_px=spacing_fine, radius=radius
        )
    else:
        T, pos, sint = square_lattice_target(
            m, spacing_fine_px=spacing_fine, n_spots=n_spots
        )
    return T, pos, sint


def design_and_reproduce(cfg, T, method, seed, amp=None, aberration_rms=0.0):
    """Design a CGH then reproduce it (optionally with an added reproduction aberration)."""
    if amp is None:
        amp = make_illumination(cfg)
    t0 = time.time()
    phase = design_cgh(
        T, amp, cfg["oversample"], method=method, seed=seed,
        iters=cfg["iters"], backend=cfg["backend"],
    )
    repro_phase = phase
    if aberration_rms > 0:
        ab = aberration_phase(cfg["n"], cfg["aperture_radius_px"], rms_waves=aberration_rms, seed=seed)
        repro_phase = phase + ab
    I = reproduce_intensity(repro_phase, amp, cfg["oversample"], shift=True)
    return phase, I, time.time() - t0


__all__ = [
    "add_common_args", "resolve", "build_target", "design_and_reproduce",
    "make_illumination", "make_aperture", "reproduce_intensity", "evaluate",
    "OUTDIR", "PRESETS", "ILLUMINATION_PRESETS", "DEFAULT_ILLUMINATION",
    "spacing_units", "spacing_label", "PRIMARY_SPACING", "APERTURE_FRAC",
]
