"""Reading the bake's UV atlas (`atlas.npz`): its emission map as a display PNG, and its
emitting texels.

The bake (`data_processing/glb2representations/atlas.py`) stores `emission_color` LINEAR, as
uint8 (`clip(linear, 0, 1) * 255`), and `occupancy`, the true rasterized chart mask, as
(H, W, 1) bool; every channel is dilated past the chart edges against seams, so a reader that
means the real footprint masks by `occupancy`.

`write_emission_png` makes both the annotation tool's emission view and the VLM check's
eighth review image, so the VLM sees the picture the annotators see; `emitting_texels`
serves the VLM check's empty-atlas filter and the emissive-ratio statistic
(`tools/dataset_statistics/emissive_ratio.py`).

numpy at import, Pillow inside `write_emission_png`; 3.8-clean, as the annotation tool
(Python 3.8) imports it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import numpy as np

from utils.color import linear_to_srgb

__all__ = ["emission_display", "write_emission_png", "emitting_texels"]


def emission_display(npz) -> Optional[np.ndarray]:
    """The emission map of a loaded atlas as an sRGB-encoded (H, W, 3) uint8 image, zero
    outside `occupancy`; None when the atlas has no `emission_color` or one of an unknown
    shape. A gray map is repeated to three channels, a fourth channel is dropped."""
    if "emission_color" not in npz:
        return None
    em = npz["emission_color"]
    if em.ndim == 2:
        em = np.stack([em, em, em], axis=-1)
    elif em.ndim == 3 and em.shape[2] == 1:
        em = np.repeat(em, 3, axis=2)
    elif em.ndim == 3 and em.shape[2] >= 3:
        em = em[:, :, :3]
    else:
        return None
    # The bake stores emission LINEAR (see the module docstring). Linear data written
    # into an sRGB PNG shows too dark, so encode it for display.
    lin = em.astype(np.float64) / 255.0 if em.dtype == np.uint8 else np.clip(em, 0.0, 1.0)
    em = (linear_to_srgb(lin) * 255.0).round().clip(0, 255).astype(np.uint8)
    # Drop the bake's dilated edge: keep emission only inside the true `occupancy` mask, so
    # the map shows the real footprint.
    if "occupancy" in npz:
        occ = npz["occupancy"]
        occ = occ[..., 0] if occ.ndim == 3 else occ
        em = np.where(occ.astype(bool)[..., None], em, 0).astype(np.uint8)
    return em


def write_emission_png(npz_path, out_path) -> bool:
    """Write the atlas's `emission_display` to `out_path` as a PNG; False, writing nothing,
    when the npz cannot be read or has no usable emission map.

    The write is atomic, so a concurrent reader never sees a partial file. The temp name is
    unique per CALL, not per process: the annotation server is a ThreadingHTTPServer, so two
    request threads extracting the same uuid share a pid; with a pid-keyed name they would
    interleave writes into one file and the loser's `os.replace` would raise."""
    from PIL import Image

    try:
        npz = np.load(npz_path)
    except Exception:
        return False
    em = emission_display(npz)
    if em is None:
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    Image.fromarray(em).save(tmp, format="PNG")
    os.replace(tmp, out_path)
    return True


def emitting_texels(npz_path, strict: bool = False) -> Tuple[int, int]:
    """(emitting, occupied) texels of an atlas: an occupied texel emits when any of its RGB
    channels is non-zero. An atlas without `emission_color` is (0, 0); one without
    `occupancy` counts every texel as occupied. A file that cannot be read raises.

    `strict=True` (the emissive-ratio statistic) raises KeyError instead when either key is
    missing, so such an atlas is an error row there, retried and caught by its report."""
    with np.load(npz_path) as npz:
        if strict:
            missing = sorted({"emission_color", "occupancy"} - set(npz.files))
            if missing:
                raise KeyError(f"{npz_path}: no {missing}")
        if "emission_color" not in npz:
            return 0, 0
        em = npz["emission_color"]
        occ = npz["occupancy"] if "occupancy" in npz else None
    em = em[..., :3] if em.ndim == 3 else em
    lit = em.max(axis=-1) > 0 if em.ndim == 3 else em > 0
    if occ is not None:
        occ = (occ[..., 0] if occ.ndim == 3 else occ).astype(bool)
        lit = lit & occ
        return int(lit.sum()), int(occ.sum())
    return int(lit.sum()), int(lit.size)
