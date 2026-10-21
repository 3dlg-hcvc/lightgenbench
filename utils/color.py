"""Color-space helpers shared by the bake, the evaluator and the atlas display image.

`evaluation/pointsampled/metrics.py` re-exports `srgb_to_linear`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["srgb_to_linear", "linear_to_srgb"]


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB-encoded [0,1] -> linear [0,1] (IEC 61966-2-1). Input is clipped to [0,1]."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear [0,1] -> sRGB-encoded [0,1] (IEC 61966-2-1), the inverse of `srgb_to_linear`.
    Input is clipped to [0,1]. The emission display image of `utils/atlas_io.py` encodes with
    it, behind the annotation tool's view and the VLM check's UV image."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
