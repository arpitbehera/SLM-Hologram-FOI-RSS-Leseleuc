"""Reproduce paper Fig 1(b): FOI- vs RSS-designed CGH phase patterns (5x5, 1.8 px).

Saves the two 8-bit phase holograms and a side-by-side comparison PNG to outputs/.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from _runner import (add_common_args, resolve, build_target, design_and_reproduce,
                     make_aperture, spacing_units, PRIMARY_SPACING, OUTDIR)
from foitweezers.io import save_cgh


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = resolve(args)
    amp = make_aperture(cfg["n"], radius_px=cfg["aperture_radius_px"])
    ns = cfg["n_spots"]
    T, pos, sint = build_target(cfg, spacing_coarse=PRIMARY_SPACING, n_spots=ns)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, method in zip(axes, ("FOI", "RSS")):
        phase, _, dt = design_and_reproduce(cfg, T, method, seed=cfg["seeds"][0], amp=amp)
        save_cgh(os.path.join(OUTDIR, f"fig1b_cgh_{method}"), phase, bits=8)
        ax.imshow(np.mod(phase, 2 * np.pi), cmap="twilight", vmin=0, vmax=2 * np.pi)
        ax.set_title(f"{method} CGH ({dt:.0f}s)")
        ax.axis("off")
    u = spacing_units(PRIMARY_SPACING)
    fig.suptitle("Fig 1(b): FOI vs RSS hologram phase "
                 f"({ns}x{ns}, {u['r_A']:.2f} $r_A$ / {u['um']:.2f} µm / {u['lambda']:.2f} $\\lambda$)")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig1b.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
