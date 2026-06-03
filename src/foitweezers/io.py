"""Saving CGHs, reproduced images, run manifests, and aggregated tables."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from .forward import quantize_phase


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_cgh(path_stem, phase, bits=8):
    """Save a CGH as an 8-bit grayscale PNG plus the float phase in an .npz.

    PNG export uses matplotlib (already a dependency) to avoid a hard PIL/cv2 dep.
    """
    q = quantize_phase(phase, bits=bits)
    img8 = np.round(np.mod(phase, 2 * np.pi) / (2 * np.pi) * (2 ** bits - 1)).astype(np.uint8)
    np.savez_compressed(path_stem + ".npz", phase=np.asarray(phase), phase_quantized=np.asarray(q))
    try:
        import matplotlib.pyplot as plt

        plt.imsave(path_stem + ".png", img8, cmap="gray", vmin=0, vmax=2 ** bits - 1)
    except Exception:
        pass
    return path_stem


def save_image(path_stem, image, cmap="inferno"):
    """Save a reproduced intensity image (.npz + .png)."""
    np.savez_compressed(path_stem + ".npz", image=np.asarray(image))
    try:
        import matplotlib.pyplot as plt

        plt.imsave(path_stem + ".png", np.asarray(image), cmap=cmap)
    except Exception:
        pass
    return path_stem


def write_manifest(path, params, results):
    """Write a JSON manifest of parameters + results for reproducibility."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "results": results,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return path


def write_table_csv(path, rows, header):
    """Write a simple CSV (list of dict rows)."""
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def mean_se(values):
    """Mean and standard error of a 1-D sample (for Table I/II reporting)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    mean = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return (mean, se)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
