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
from foitweezers.design import design_cgh, design_cgh_dual
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


def method_label(methods):
    """Filename label for one or more methods: ['RSS','FOI'] -> 'RSS-FOI'."""
    return "-".join(methods)


def run_seed(cfg, T, pos, sint, method, seed, amp):
    """Design + reproduce + evaluate one seed. Returns a run dict."""
    phase, info = design_cgh(
        T, amp, cfg["oversample"], method=method, seed=seed,
        iters=cfg["iters"], backend=cfg["backend"], return_info=True,
    )
    I = reproduce_intensity(phase, amp, cfg["oversample"], shift=True)
    met = evaluate(I, pos, sint, n_spots=cfg["n_spots"], lattice=cfg.get("lattice", "square"))
    return dict(
        seed=seed, phase=phase, I=I,
        final_cost=float(info["final_cost"]),
        history=list(info.get("history", []) or []),
        u=met["uniformity"], e=met["efficiency"], v=met["vp_ratio"],
    )


def run_seed_chain(cfg, T, pos, sint, methods, seed, amp):
    """Run ``methods`` as warm-started stages from one seed.

    Each stage starts from the previous stage's converged **continuous** phase
    (float64, [0,2pi) — never the uint8 save form). On the scipy backend, both
    RSS and FOI cost are recorded per iteration over one continuous axis; other
    backends still chain but leave the histories empty (dual plot is skipped).
    Returns a run dict compatible with ``select_best``/``aggregate_stats``.
    """
    phase = None
    rss_history, foi_history = [], []
    stage_bounds, stage_labels, stages = [], [], []
    final_cost = None
    for i, method in enumerate(methods):
        if cfg["backend"] == "scipy":
            phase, info = design_cgh_dual(
                T, amp, cfg["oversample"], method=method, seed=seed,
                iters=cfg["iters"], phase0=phase,
            )
            rss_history.extend(info["rss_history"])
            foi_history.extend(info["foi_history"])
        else:
            phase, info = design_cgh(
                T, amp, cfg["oversample"], method=method, seed=seed,
                phase0=phase, iters=cfg["iters"], backend=cfg["backend"],
                return_info=True,
            )
        final_cost = float(info["final_cost"])
        stages.append(dict(method=method, final_cost=final_cost,
                           nit=int(info.get("nit", 0))))
        if i < len(methods) - 1:
            stage_bounds.append(len(rss_history))      # per-seed continuous offset
            stage_labels.append(f"{method}→{methods[i + 1]}")
    I = reproduce_intensity(phase, amp, cfg["oversample"], shift=True)
    met = evaluate(I, pos, sint, n_spots=cfg["n_spots"], lattice=cfg.get("lattice", "square"))
    return dict(
        seed=seed, phase=phase, I=I, final_cost=final_cost,
        rss_history=rss_history, foi_history=foi_history,
        stage_bounds=stage_bounds, stage_labels=stage_labels, stages=stages,
        history=[],  # keep key present; single-method path uses run_seed's history
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


def plot_dual_convergence(stem, runs, best_seed, methods, spacing):
    """Two continuous-axis plots (RSS and FOI cost vs global iteration).

    All seeds plotted, best highlighted; dashed vertical markers at the best
    seed's stage transitions. Returns the list of written paths, or [] (with a
    notice) when there is no per-iteration history (non-scipy backend).

    Caveat (by design): the x-axis is 0-based and uses each seed's *actual* CG
    iteration count (CG stops at gtol before maxiter), so per-seed stage lengths
    differ. The vertical stage markers reflect the BEST seed's transitions only;
    other seeds' transitions are not individually marked. This is exact for the
    default single-seed run.
    """
    if not any(r["rss_history"] for r in runs):
        print("convergence: no per-iteration history (non-scipy backend); dual plots skipped")
        return []
    import matplotlib.pyplot as plt
    best = next(r for r in runs if r["seed"] == best_seed)
    written = []
    for cost_key, cost_name in (("rss_history", "RSS"), ("foi_history", "FOI")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in runs:
            h = r[cost_key]
            if not h:
                continue
            is_best = r["seed"] == best_seed
            ax.plot(range(len(h)), h,
                    lw=2.2 if is_best else 1.0,
                    label=f"seed {r['seed']}" + (" (best)" if is_best else ""),
                    zorder=3 if is_best else 2)
        ytop = ax.get_ylim()[1]
        for b, lbl in zip(best["stage_bounds"], best["stage_labels"]):
            ax.axvline(b, ls="--", color="0.5", lw=1.0, zorder=1)
            ax.text(b, ytop, lbl, fontsize=7, ha="center", va="top", color="0.3")
        ax.set_xlabel("iteration (continuous across stages)")
        ax.set_ylabel(f"{cost_name} cost")
        ax.set_title(f"{cost_name} convergence — {'+'.join(methods)} — "
                     f"{spacing_label(spacing)}")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = f"{stem}_convergence_{cost_name}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--method", choices=["FOI", "RSS"], nargs="+", required=True,
                    help="optimization cost function(s); 2+ = chained stages in order")
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

    methods = args.method
    chained = len(methods) > 1

    su = spacing_units(args.spacing)
    print(f"method={'+'.join(methods)} spacing={su['r_A']:.2f} r_A / "
          f"{su['um']:.3f} um / {su['lambda']:.2f} lambda  seeds={len(cfg['seeds'])}")

    runs = []
    for seed in cfg["seeds"]:
        if chained:
            r = run_seed_chain(cfg, T, pos, sint, methods, seed, amp)
        else:
            r = run_seed(cfg, T, pos, sint, methods[0], seed, amp)
        runs.append(r)
        print(f"  seed={seed}: cost={r['final_cost']:.4e} sigma={r['u']:.3e} "
              f"eff={r['e']:.3f} vp={r['v']:.3e}", flush=True)

    best = select_best(runs)
    agg = aggregate_stats(runs)

    tag = f"_{args.tag}" if args.tag else ""
    stem = os.path.join(OUTDIR, f"optmask_{method_label(methods)}_sp{args.spacing}{tag}")

    save_mask_uint8(stem + "_phase", best["phase"])
    save_image(stem + "_image", best["I"])
    if chained:
        plot_dual_convergence(stem, runs, best["seed"], methods, args.spacing)
    else:
        plot_convergence(stem + "_convergence.png", runs, best["seed"],
                         methods[0], args.spacing)

    method_meta = methods[0] if len(methods) == 1 else list(methods)
    results = {
        "method": method_meta,
        "spacing_px": float(args.spacing),
        "spacing_rA": su["r_A"],
        "spacing_um": su["um"],
        "spacing_lambda": su["lambda"],
        "n_seeds": len(cfg["seeds"]),
        "n_sites": len(pos),
        **agg,
        "best": {
            "seed": int(best["seed"]),
            "final_cost": best["final_cost"],
            "uniformity": best["u"],
            "efficiency": best["e"],
            "vp_ratio": best["v"],
        },
    }
    if chained:
        results["stages"] = best["stages"]
        results["best"]["rss_cost"] = (best["rss_history"][-1]
                                       if best["rss_history"] else None)
        results["best"]["foi_cost"] = (best["foi_history"][-1]
                                       if best["foi_history"] else None)
    params = {
        "method": method_meta, "spacing_px": float(args.spacing),
        "preset": args.preset, "illumination": args.illumination,
        "backend": args.backend, "iters": cfg["iters"],
        "n_spots": ns, "n_seeds": len(cfg["seeds"]),
        "seeds": list(cfg["seeds"]), "stem": os.path.basename(stem),
    }
    write_manifest(stem + "_meta.json", params, results)
    if chained:
        print("wrote", stem + "_{phase,image}.{png,npz}, "
              "_convergence_{RSS,FOI}.png, _meta.json")
    else:
        print("wrote", stem + "_{phase,image}.{png,npz}, _convergence.png, _meta.json")


if __name__ == "__main__":
    main()
