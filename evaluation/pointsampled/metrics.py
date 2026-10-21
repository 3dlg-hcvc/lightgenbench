"""Point-cloud emission metrics: IoU (primary), MAE and normalized PSNR (magnitude diagnostics).

IoU thresholds each representation's NATIVE values at eps; the magnitude metrics first bring
everything into one canonical linear space (`gt_to_canonical_linear` / `baked_to_canonical_linear`).
Keeping the two separate is deliberate: a change to the value space does not move the IoU.
"""
import numpy as np

# Re-exported: `srgb_to_linear` is defined once, in utils/color.py.
from utils.color import srgb_to_linear  # noqa: F401


def emission_mask(emission: np.ndarray, eps: float) -> np.ndarray:
    return np.asarray(emission).max(axis=-1) > eps


def point_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool); b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / float(union)


def point_precision(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """|pred AND gt| / |pred| — the share of predicted-emissive points that really emit.

    Empty-set convention mirrors `point_iou`'s empty union: an empty prediction scores 1.0 when
    the GT is also empty (correct abstention on a non-emissive shape) and 0.0 otherwise, so a
    model that abstains everywhere is not rewarded on shapes that do emit.
    """
    p = np.asarray(pred_mask, dtype=bool); g = np.asarray(gt_mask, dtype=bool)
    n_pred = p.sum()
    if n_pred == 0:
        return 1.0 if g.sum() == 0 else 0.0
    return float(np.logical_and(p, g).sum()) / float(n_pred)


def point_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """|pred AND gt| / |gt| — the share of truly emissive points the prediction found.

    Empty-set convention mirrors `point_precision`: an empty GT scores 1.0 when the prediction is
    also empty and 0.0 otherwise.
    """
    p = np.asarray(pred_mask, dtype=bool); g = np.asarray(gt_mask, dtype=bool)
    n_gt = g.sum()
    if n_gt == 0:
        return 1.0 if p.sum() == 0 else 0.0
    return float(np.logical_and(p, g).sum()) / float(n_gt)


def gt_to_canonical_linear(texel_srgb, factor, strength: float = 1.0) -> np.ndarray:
    """Mesh-GT emission in the canonical space every metric compares in.

    Canonical space = ``clip(srgb_to_linear(texel) * emissiveFactor, 0, 1)``: linear, clipped to
    [0,1], and deliberately WITHOUT ``KHR_materials_emissive_strength``:

    * ``factor`` multiplies AFTER the sRGB decode, because emissiveFactor is a linear quantity in
      the glTF spec.
    * ``strength`` is excluded because the bakes do not carry it (trimesh does not surface the
      extension). The bakes also store a declared emissive texture without the factor
      (``utils.trimesh_materials.emission_source``), so this canonical space and the training
      target differ on textured shapes with factor != 1; `load_gt.part_gt_linear` scores against
      that target by passing a factor of 1 for a declared texture. The ``strength`` argument is
      accepted so callers can pass what they parsed, and has no effect.

    Both bakes store the declared emission clipped to [0,1] as uint8, so this is exactly their
    space wherever the emissive factor is 1 or the material has no emissive texture.
    """
    del strength  # see docstring: the bakes are strength-free, so the canonical space is too
    texel = np.asarray(texel_srgb, dtype=np.float64)
    return np.clip(srgb_to_linear(texel) * np.asarray(factor, dtype=np.float64), 0.0, 1.0)


def baked_to_canonical_linear(emission) -> np.ndarray:
    """A baked atlas/voxel emission value into the canonical space.

    Bakes already store canonical values (linear, clipped, uint8-quantized), so this only
    re-asserts the range. Kept as a named function so every path reaches the canonical space
    through this module rather than through an ad-hoc conversion at the call site.
    """
    return np.clip(np.asarray(emission, dtype=np.float64), 0.0, 1.0)


def point_mae(pred_lin, gt_lin, mask=None) -> float:
    """Mean absolute error in canonical linear space.

    Reported over ALL sampled surface points (``mask=None``). Passing ``mask=gt_mask`` gives the
    GT-mask variant stored as ``mae_gton``: an empty GT mask leaves no
    points, so the result is nan and drops out of the finite means.
    """
    p = np.asarray(pred_lin, dtype=np.float64); g = np.asarray(gt_lin, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        p = p[m]; g = g[m]
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - g)))


PSNR_MSE_EPS = 1e-4  # cap = -10*log10(eps) = 40 dB at max_val 1; binds on empty-union shapes


def point_psnr(pred_lin, gt_lin, mask=None, max_val: float = 1.0,
               eps: float = PSNR_MSE_EPS) -> float:
    # An empty mask (GT dark AND pred dark) and an exact match both score the 40 dB
    # cap rather than nan/inf, so correct abstention stays in every aggregate and all
    # models share one denominator.
    p = np.asarray(pred_lin, dtype=np.float64); g = np.asarray(gt_lin, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        p = p[m]; g = g[m]
    mse = float(np.mean((p - g) ** 2)) if p.size else 0.0
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(max(mse, eps))
