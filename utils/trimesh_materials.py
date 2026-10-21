"""Reading a trimesh PBR material the way the bake needs it.

Both bakes (`data_processing/glb2representations/atlas.py`, the UV atlas, and
`voxels.py`, the o_voxel pre-bake) read the same glTF material fields off a
`trimesh.visual.material.PBRMaterial`; that parse and the emission rule are defined once
here.

Duck-typed: any object with the glTF attribute names works (`emissiveTexture`,
`emissiveFactor`, `baseColorFactor`), so tests use a SimpleNamespace and this module
imports neither trimesh nor Pillow.

THE EMISSION RULE: the dataset carries the AUTHORED emission. A declared emissive texture
is the emission, whatever `emissiveFactor` says (1, 0.3, 0, or absent). Without a texture
the factor is the emission color. The `KHR_materials_emissive_strength` multiplier is not
part of the dataset's emission and is never read (trimesh does not surface it anyway).
The evaluator reads the GLB itself, every part through this rule
(`evaluation/pointsampled/load_gt.py`); the submitted paper's evaluator multiplied
texel x factor instead.
"""

from __future__ import annotations

import numpy as np

__all__ = ["factor_rgb", "factor_alpha", "factor_alpha_u8", "emission_source"]


def factor_rgb(factor, default: float) -> np.ndarray:
    """glTF factor -> linear float[3].

    trimesh stores `baseColorFactor` as uint8 (-> /255) but `emissiveFactor` as float
    linear; an integer dtype or any component > 1 means the 0..255 scale. A short
    factor is zero-padded; None is `default` in every channel.
    """
    if factor is None:
        return np.full(3, float(default), dtype=np.float64)
    raw = np.asarray(factor)
    a = raw.astype(np.float64).reshape(-1)[:3]
    if a.size < 3:
        a = np.pad(a, (0, 3 - a.size))
    if raw.dtype.kind in "iu" or (a.size and a.max() > 1.0):
        a = a / 255.0
    return a


def factor_alpha(base_color_factor) -> float:
    """`baseColorFactor[3]` as a float; 1.0 when absent or RGB-only.

    The atlas bake's parse (it multiplies this float into the texture alpha and rounds
    once at the end). uint8 or a value > 1 means the 0..255 scale. The value is NOT
    clipped: `atlas._sample_material_alpha` clips the product (and the no-texture
    factor) itself, so clipping here too would move baked alphas.
    """
    if base_color_factor is None:
        return 1.0
    raw = np.asarray(base_color_factor).reshape(-1)
    if raw.size < 4:
        return 1.0
    a = float(raw[3])
    if raw.dtype.kind in "iu" or a > 1.0:
        a /= 255.0
    return a


def factor_alpha_u8(base_color_factor) -> int:
    """`baseColorFactor[3]` as a byte, with the voxel bake's exact quantization.

    The o_voxel pre-bake writes the factor alpha into a uint8 texture plane: an integer
    factor is used as-is, a float one is TRUNCATED (`int(a * 255)`), which is how the
    published voxels were made. Kept separate from `factor_alpha` on purpose: rounding
    instead would change every voxel alpha by up to one level.
    """
    if base_color_factor is None:
        return 255
    raw = np.asarray(base_color_factor).reshape(-1)
    if raw.size < 4:
        return 255
    if raw.dtype.kind in "iu" or float(raw[3]) > 1.0:
        return int(np.clip(raw[3], 0, 255))
    return int(np.clip(float(raw[3]), 0.0, 1.0) * 255.0)


def emission_source(mat):
    """What a material emits: ('texture', image) | ('color', rgb float[3] linear) | ('none', None).

    The picture wins: a declared emissive texture is the emission whatever the factor
    says. Without one the factor is the color. The strength extension is never read.
    The image is returned as stored on the material (a PIL image for trimesh); the
    caller samples or converts it.
    """
    tex = getattr(mat, "emissiveTexture", None)
    if tex is not None:
        return "texture", tex
    rgb = factor_rgb(getattr(mat, "emissiveFactor", None), default=0.0)
    if float(rgb.max()) > 0.0:
        return "color", rgb
    return "none", None
