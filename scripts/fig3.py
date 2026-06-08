"""Reproduce paper Fig 3(a,b): numerically reproduced 5x5 images for several spot
spacings, FOI (top row) vs RSS (bottom row).

Default spacings 1.8/2.1/2.4/2.7 coarse px. FOI keeps spots separated where RSS
merges. Saves a montage and per-panel .npz to outputs/.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from _runner import (add_common_args, resolve, build_target, design_and_reproduce,
                     make_aperture, evaluate, spacing_label, OUTDIR)
from foitweezers.io import save_image, save_cgh


def _crop_center(I, pos, sint, n_spots=5, pad=2):
    rs = [p[0] for p in pos]; cs = [p[1] for p in pos]
    hw = int((n_spots + pad) * sint / 2)
    rc, cc = (min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2
    r0, r1 = max(0, rc - hw), min(I.shape[0], rc + hw)
    c0, c1 = max(0, cc - hw), min(I.shape[1], cc + hw)
    return I[r0:r1, c0:c1]


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = resolve(args)
    amp = make_aperture(cfg["n"], radius_px=cfg["aperture_radius_px"])
    spacings = cfg["spacings"]
    seed = cfg["seeds"][0]
    ns = cfg["n_spots"]

    fig, axes = plt.subplots(2, len(spacings), figsize=(3 * len(spacings), 6))
    for col, sp in enumerate(spacings):
        T, pos, sint = build_target(cfg, spacing_coarse=sp, n_spots=ns)
        for row, method in enumerate(("FOI", "RSS")):
            phase, I, dt = design_and_reproduce(cfg, T, method, seed=seed, amp=amp)
            met = evaluate(I, pos, sint, n_spots=ns)
            crop = _crop_center(I, pos, sint, n_spots=ns)
            save_image(os.path.join(OUTDIR, f"fig3_{method}_sp{sp}"), crop)
            save_cgh(os.path.join(OUTDIR, f"fig3_cgh_{method}_sp{sp}"), phase)
            ax = axes[row, col]
            ax.imshow(crop, cmap="inferno")
            ax.axis("off")
            if row == 0:
                ax.set_title(spacing_label(sp, fmt="mpl"), fontsize=8)
            ax.text(0.02, 0.04, f"{method}\n$\\sigma$={met['uniformity']:.2g}\nVP={met['vp_ratio']:.2g}",
                    transform=ax.transAxes, color="w", fontsize=7, va="bottom")
            print(f"sp={sp} {method}: sigma={met['uniformity']:.3e} eff={met['efficiency']:.3f} "
                  f"vp={met['vp_ratio']:.3e} ({dt:.0f}s)", flush=True)
    fig.suptitle(f"Fig 3: numerically reproduced {ns}x{ns} arrays (top FOI, bottom RSS)")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig3.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
