"""Bilinear sampling of a texture at UV coordinates, glTF convention.

`sample_texture_bilinear` clamps UVs to [0,1]. `sample_texture_repeat` is the bake's
variant, which the evaluator's per-part reading also uses: glTF's default sampler wrap
is REPEAT, and source assets tile UVs far outside [0,1], so the bake wraps instead of
clamping.

Textures are (H, W) or (H, W, C) arrays; uint8 is scaled to [0,1], anything else is
taken as-is. glTF maps u to the column and (1 - v) to the row (V-flip).
"""

from __future__ import annotations

import numpy as np

__all__ = ["sample_texture_bilinear", "sample_texture_repeat"]


def sample_texture_bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    tex = np.asarray(texture)
    if tex.dtype == np.uint8:
        tex = tex.astype(np.float64) / 255.0
    else:
        tex = tex.astype(np.float64)
    if tex.ndim == 2:
        tex = tex[..., None]
    H, W, C = tex.shape
    uv = np.clip(np.asarray(uv, dtype=np.float64), 0.0, 1.0)
    # glTF: u -> column, (1-v) -> row (V-flip)
    x = uv[:, 0] * (W - 1)
    y = (1.0 - uv[:, 1]) * (H - 1)
    x0 = np.floor(x).astype(int); x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(y).astype(int); y1 = np.minimum(y0 + 1, H - 1)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    c00 = tex[y0, x0]; c01 = tex[y0, x1]
    c10 = tex[y1, x0]; c11 = tex[y1, x1]
    top = c00 * (1 - fx) + c01 * fx
    bot = c10 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bot * fy


def sample_texture_repeat(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """`sample_texture_bilinear` with glTF-default REPEAT wrapping (uv -> uv - floor(uv)).

    The wrap is computed in the caller's dtype, NOT in float64: the bake feeds float32
    UVs straight from nvdiffrast, and `uv - floor(uv)` rounds differently once the
    operand is widened first (a small share of the baked base-color, metallic, roughness
    and alpha bytes move by one level on tiled UVs). `sample_texture_bilinear`
    casts to float64 afterwards.
    """
    uv = np.asarray(uv)
    return sample_texture_bilinear(texture, uv - np.floor(uv))
