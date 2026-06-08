"""Design one optimal FOI/RSS phase mask for a single spacing.

Runs the optimization over ``--seeds`` independent random initializations, saves
ONLY the single best-performing result (lowest final optimizer cost), plus
aggregate and best-result metadata and a per-seed cost-convergence plot.

Best-seed criterion: minimum final optimizer cost (``info["final_cost"]`` from
``design_cgh``). ``--method`` is fixed per run, so all seeds minimize the same
objective and their final costs are directly comparable.

Outputs (to outputs/, stem ``optmask_{method}_sp{spacing}[_{tag}]``):
  {stem}_phase.png/.npz        best mask (uint8; code k -> phase k*2pi/256)
  {stem}_image.png/.npz        best predicted intensity image
  {stem}_convergence.png       cost vs iteration, one line per seed
  {stem}_meta.json             params + aggregate + best metadata
"""

import argparse
import os

import numpy as np

from _runner import (add_common_args, resolve, build_target, make_illumination,
                     reproduce_intensity, evaluate, spacing_units, spacing_label,
                     PRIMARY_SPACING, OUTDIR)
from foitweezers.design import design_cgh
from foitweezers.io import save_image, write_manifest


def save_mask_uint8(path_stem, phase, bits=8):
    """Save a phase mask as uint8 in both .npz and .png (256-level convention).

    Code ``k in {0..255}`` represents phase ``k * 2*pi/256`` so the codes tile
    ``[0, 2*pi)`` uniformly. Recover radians via ``codes * (2*pi/256)``.
    """
    levels = 2 ** bits
    wrapped = np.mod(phase, 2 * np.pi)
    codes = (np.round(wrapped / (2 * np.pi) * levels) % levels).astype(np.uint8)
    np.savez_compressed(path_stem + ".npz", phase_uint8=codes)
    try:
        import matplotlib.pyplot as plt
        plt.imsave(path_stem + ".png", codes, cmap="gray", vmin=0, vmax=levels - 1)
    except Exception:
        pass
    return path_stem


def aggregate_stats(runs):
    """Mean (always) and sample std (only if >1 run) of uniformity/eff/vp."""
    u = [r["u"] for r in runs]
    e = [r["e"] for r in runs]
    v = [r["v"] for r in runs]
    agg = {
        "uniformity_mean": float(np.mean(u)),
        "efficiency_mean": float(np.mean(e)),
        "vp_mean": float(np.mean(v)),
    }
    if len(runs) > 1:
        agg["uniformity_std"] = float(np.std(u, ddof=1))
        agg["efficiency_std"] = float(np.std(e, ddof=1))
        agg["vp_std"] = float(np.std(v, ddof=1))
    return agg


def select_best(runs):
    """Best seed = minimum final optimizer cost (method fixed per run)."""
    return min(runs, key=lambda r: r["final_cost"])


def run_seed(cfg, T, pos, sint, method, seed, amp):
    """Design + reproduce + evaluate one seed. Returns a run dict."""
    phase, info = design_cgh(
        T, amp, cfg["oversample"], method=method, seed=seed,
        iters=cfg["iters"], backend=cfg["backend"], return_info=True,
    )
    I = reproduce_intensity(phase, amp, cfg["oversample"], shift=True)
    met = evaluate(I, pos, sint, n_spots=cfg["n_spots"])
    return dict(
        seed=seed, phase=phase, I=I,
        final_cost=float(info["final_cost"]),
        history=list(info.get("history", []) or []),
        u=met["uniformity"], e=met["efficiency"], v=met["vp_ratio"],
    )


def plot_convergence(path, runs, best_seed, method, spacing):
    """Cost vs iteration, one line per seed (best highlighted). Skips if empty."""
    if not any(r["history"] for r in runs):
        print("convergence: no per-iteration history (non-scipy backend); plot skipped")
        return None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in runs:
        if not r["history"]:
            continue
        is_best = r["seed"] == best_seed
        ax.plot(range(1, len(r["history"]) + 1), r["history"],
                lw=2.2 if is_best else 1.0,
                label=f"seed {r['seed']}" + (" (best)" if is_best else ""),
                zorder=3 if is_best else 2)
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"{method} cost")
    ax.set_title(f"{method} convergence — {spacing_label(spacing)}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--method", choices=["FOI", "RSS"], required=True,
                    help="optimization cost function")
    ap.add_argument("--spacing", type=float, default=PRIMARY_SPACING,
                    help="single spacing in coarse target-plane px")
    ap.add_argument("--tag", type=str, default=None,
                    help="optional suffix on output filenames")
    args = ap.parse_args()
    if args.seeds is None:           # task default: 1 seed (not the preset's list)
        args.seeds = 1
    cfg = resolve(args)
    amp = make_illumination(cfg)
    ns = cfg["n_spots"]
    T, pos, sint = build_target(cfg, spacing_coarse=args.spacing, n_spots=ns)

    su = spacing_units(args.spacing)
    print(f"method={args.method} spacing={su['r_A']:.2f} r_A / {su['um']:.3f} um / "
          f"{su['lambda']:.2f} lambda  seeds={len(cfg['seeds'])}")

    runs = []
    for seed in cfg["seeds"]:
        r = run_seed(cfg, T, pos, sint, args.method, seed, amp)
        runs.append(r)
        print(f"  seed={seed}: cost={r['final_cost']:.4e} sigma={r['u']:.3e} "
              f"eff={r['e']:.3f} vp={r['v']:.3e}", flush=True)

    best = select_best(runs)
    agg = aggregate_stats(runs)

    tag = f"_{args.tag}" if args.tag else ""
    stem = os.path.join(OUTDIR, f"optmask_{args.method}_sp{args.spacing}{tag}")

    save_mask_uint8(stem + "_phase", best["phase"])
    save_image(stem + "_image", best["I"])
    plot_convergence(stem + "_convergence.png", runs, best["seed"],
                     args.method, args.spacing)

    results = {
        "method": args.method,
        "spacing_px": float(args.spacing),
        "spacing_rA": su["r_A"],
        "spacing_um": su["um"],
        "spacing_lambda": su["lambda"],
        "n_seeds": len(cfg["seeds"]),
        **agg,
        "best": {
            "seed": int(best["seed"]),
            "final_cost": best["final_cost"],
            "uniformity": best["u"],
            "efficiency": best["e"],
            "vp_ratio": best["v"],
        },
    }
    params = {
        "method": args.method, "spacing_px": float(args.spacing),
        "preset": args.preset, "illumination": args.illumination,
        "backend": args.backend, "iters": cfg["iters"],
        "n_spots": ns, "n_seeds": len(cfg["seeds"]),
        "seeds": list(cfg["seeds"]), "stem": os.path.basename(stem),
    }
    write_manifest(stem + "_meta.json", params, results)
    print("wrote", stem + "_{phase,image}.{png,npz}, _convergence.png, _meta.json")


if __name__ == "__main__":
    main()
