"""Reproduce paper Table I: spot uniformity and light-utilization efficiency for
FOI- and RSS-designed CGHs, numerically, with and without an added reproduction
aberration. Aggregates mean +/- SE over the preset's seeds.

Paper (numerical, 1.8 px, full scale) for reference:
    FOI:  uniformity 1.21(4)e-2,  efficiency 0.120(3)
    RSS:  uniformity 5.6(10)e-2,  efficiency 0.928(4)
    FOI + aberration: uniformity 5.2(2)e-2

This CPU preset will not match the absolute numbers (smaller grid / fewer iters)
but reproduces the trend: FOI = low sigma / low eff, RSS = high sigma / high eff,
aberration worsens uniformity while leaving efficiency ~unchanged.
"""

import argparse
import os

from _runner import (add_common_args, resolve, build_target,
                     make_illumination, reproduce_intensity, evaluate,
                     spacing_units, PRIMARY_SPACING, OUTDIR)
from foitweezers.io import write_table_csv, mean_se


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--spacing", type=float, default=PRIMARY_SPACING, help="single spacing for Table I")
    ap.add_argument("--aberration-rms", type=float, default=0.02, help="added aberration RMS in waves")
    args = ap.parse_args()
    cfg = resolve(args)
    amp = make_illumination(cfg)
    ns = cfg["n_spots"]
    T, pos, sint = build_target(cfg, spacing_coarse=args.spacing, n_spots=ns)
    su = spacing_units(args.spacing)
    print(f"spacing: {su['r_A']:.2f} r_A / {su['um']:.3f} um / {su['lambda']:.2f} lambda(780nm)")

    from foitweezers.design import design_cgh
    from foitweezers.aberration import aberration_phase

    cases = (("numerical", 0.0), ("numerical+aberration", args.aberration_rms))
    acc = {(label, method): {"u": [], "e": [], "v": []} for label, _ in cases for method in ("FOI", "RSS")}

    # Design each hologram once per (method, seed); reproduce with and without aberration.
    for method in ("FOI", "RSS"):
        for seed in cfg["seeds"]:
            phase = design_cgh(T, amp, cfg["oversample"], method=method, seed=seed,
                               iters=cfg["iters"], backend=cfg["backend"])
            for label, ab in cases:
                rp = phase if ab == 0 else phase + aberration_phase(
                    cfg["n"], cfg["aperture_radius_px"], rms_waves=ab, seed=seed)
                I = reproduce_intensity(rp, amp, cfg["oversample"], shift=True)
                met = evaluate(I, pos, sint, n_spots=ns)
                d = acc[(label, method)]
                d["u"].append(met["uniformity"]); d["e"].append(met["efficiency"]); d["v"].append(met["vp_ratio"])
                print(f"[{label}] {method} seed={seed}: sigma={met['uniformity']:.3e} "
                      f"eff={met['efficiency']:.3f}", flush=True)

    rows = []
    for label, _ in cases:
        for method in ("FOI", "RSS"):
            d = acc[(label, method)]
            um, ue = mean_se(d["u"]); em, ee = mean_se(d["e"]); vm, ve = mean_se(d["v"])
            rows.append({
                "case": label, "method": method,
                "spacing_r_A": su["r_A"], "spacing_um": su["um"],
                "spacing_lambda": su["lambda"],
                "uniformity_mean": um, "uniformity_se": ue,
                "efficiency_mean": em, "efficiency_se": ee,
                "vp_mean": vm, "vp_se": ve, "n_seeds": len(cfg["seeds"]),
            })
            print(f"  => {label} {method}: sigma={um:.3e}+/-{ue:.1e}  eff={em:.3f}+/-{ee:.1e}")

    header = ["case", "method", "spacing_r_A", "spacing_um", "spacing_lambda",
              "uniformity_mean", "uniformity_se",
              "efficiency_mean", "efficiency_se", "vp_mean", "vp_se", "n_seeds"]
    out = os.path.join(OUTDIR, "table1.csv")
    write_table_csv(out, rows, header)
    print("wrote", out)


if __name__ == "__main__":
    main()
