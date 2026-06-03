"""Shared helpers for the Phase-1 figure/table scripts.

Presets trade fidelity for runtime:

* ``tiny``  - n=96,  oversample=6,  iters=120, 1 seed   (~minutes on CPU; default)
* ``cpu``   - n=160, oversample=10, iters=300, 3 seeds  (slow on CPU; overnight)
* ``paper`` - n=800, oversample=10, iters=1000, 20 seeds (GPU only; the paper)

All presets keep the same aperture fill (0.45) and report spacing in *coarse*
target-plane pixels (1.6/1.8/2.1/2.5), so the d/r_A ratios are preset-independent.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foitweezers.forward import make_aperture, reproduce_intensity  # noqa: E402
from foitweezers.targets import square_lattice_target  # noqa: E402
from foitweezers.design import design_cgh  # noqa: E402
from foitweezers.metrics import evaluate  # noqa: E402
from foitweezers.aberration import aberration_phase  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

PRESETS = {
    "tiny": dict(n=96, oversample=6, iters=120, seeds=(0,)),
    "cpu": dict(n=160, oversample=10, iters=300, seeds=(0, 1, 2)),
    "paper": dict(n=800, oversample=10, iters=1000, seeds=tuple(range(20))),
}
APERTURE_FRAC = 0.45
SPACINGS = (1.6, 1.8, 2.1, 2.5)


def add_common_args(p):
    p.add_argument("--preset", choices=list(PRESETS), default="tiny")
    p.add_argument("--backend", choices=["scipy", "torch", "slmsuite"], default="scipy")
    p.add_argument("--iters", type=int, default=None, help="override preset iterations")
    p.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    p.add_argument("--spacings", type=float, nargs="*", default=None)
    return p


def resolve(args):
    cfg = dict(PRESETS[args.preset])
    if args.iters is not None:
        cfg["iters"] = args.iters
    if args.seeds is not None:
        cfg["seeds"] = tuple(range(args.seeds))
    cfg["spacings"] = tuple(args.spacings) if args.spacings else SPACINGS
    cfg["backend"] = args.backend
    cfg["aperture_radius_px"] = APERTURE_FRAC * cfg["n"]
    os.makedirs(OUTDIR, exist_ok=True)
    return cfg


def build_target(cfg, spacing_coarse, n_spots=5):
    m = cfg["n"] * cfg["oversample"]
    T, pos, sint = square_lattice_target(
        m, spacing_fine_px=spacing_coarse * cfg["oversample"], n_spots=n_spots
    )
    return T, pos, sint


def design_and_reproduce(cfg, T, method, seed, amp=None, aberration_rms=0.0):
    """Design a CGH then reproduce it (optionally with an added reproduction aberration)."""
    if amp is None:
        amp = make_aperture(cfg["n"], radius_px=cfg["aperture_radius_px"])
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
    "make_aperture", "reproduce_intensity", "evaluate", "OUTDIR", "PRESETS",
]
