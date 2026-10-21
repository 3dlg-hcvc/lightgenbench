"""Atlas output -> the common points: each surface point reads the prediction at the nearest occupied
atlas texel by 3D position (TEXGen-Emission; Hunyuan3D-Emission after `multiview.py`).

A prediction on the dataset's UV atlas cannot be read at a point's GLB UV: the atlas has its own
layout. The atlas (`atlas.npz`, from `data_processing/glb2representations/atlas.py`) stores each
texel's 3D `position` (uint16, canonical [-1, 1]) and `occupancy`; each surface point finds the
nearest occupied texel and reads the prediction there. Canonical positions are centered by
construction and map into the voxel frame, where the points are, by ``TIGHT_SCALE_NUM /
CANONICAL_SCALE_NUM``, the ratio the voxelizer uses.
"""
import numpy as np
from scipy.spatial import cKDTree

# The voxelizer's scale constants, kept local so this module does not import glb2representations.
_CANONICAL_SCALE_NUM = 1.9999
_TIGHT_SCALE_NUM = 0.99999
_ATLAS_TO_VOXEL = _TIGHT_SCALE_NUM / _CANONICAL_SCALE_NUM


def load_posocc(npz_path):
    """``position`` (canonical [-1, 1], decoded ``/65535*2 - 1``) and ``occupancy`` of an atlas.npz."""
    z = np.load(npz_path)
    position = z["position"].astype(np.float64) / 65535.0 * 2.0 - 1.0
    occ = z["occupancy"]
    occ = occ[..., 0] if occ.ndim == 3 else occ
    return position, occ.astype(bool)


def sample_atlas_emission(points_vf, position_map, occupancy, pred_img, cap=2.0 / 256):
    """Read the prediction at each surface point via the nearest occupied atlas texel.

    points_vf   : [N,3] surface points in the [-0.5,0.5] voxel frame (the GLB's own bounding box),
                  the same points the voxel converter reads.
    position_map: [H,W,3] canonical [-1,1] atlas positions (`load_posocc`).
    occupancy   : [H,W] bool.
    pred_img    : [h,w,3] the prediction, on the same atlas layout (it may have another resolution;
                  it is indexed proportionally).
    cap         : max distance (in the voxel frame) for a point to count as on the atlas surface;
                  beyond it the point reads 0 (no emission).
    Returns (emission [N,3] in [0,1], miss_fraction).
    """
    H, W = occupancy.shape
    rows, cols = np.where(occupancy)
    if len(rows) == 0:
        pts = np.asarray(points_vf, dtype=np.float64)
        return np.zeros((len(pts), 3)), 1.0
    P = np.asarray(position_map[rows, cols], dtype=np.float64) * _ATLAS_TO_VOXEL   # [M,3] in the voxel frame
    dist, idx = cKDTree(P).query(np.asarray(points_vf, dtype=np.float64), k=1)

    tex = np.asarray(pred_img, dtype=np.float64)
    if tex.ndim == 2:
        tex = tex[..., None].repeat(3, axis=-1)
    tex = tex[..., :3]
    h, w = tex.shape[:2]
    # atlas pixel -> prediction pixel (proportional; atlas layouts align, no UV V-flip)
    trow = np.minimum((rows[idx] * h) // H, h - 1)
    tcol = np.minimum((cols[idx] * w) // W, w - 1)
    emis = tex[trow, tcol].copy()                            # [N,3]
    miss = dist > cap
    emis[miss] = 0.0
    return emis, float(miss.mean())
