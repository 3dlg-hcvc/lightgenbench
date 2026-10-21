"""Voxel output -> the common points: each surface point, mapped into the 256^3 voxel frame, reads the
nearest occupied predicted voxel (TRELLIS.2-Emission, SegviGen-Emission)."""
import numpy as np
from scipy.spatial import cKDTree


def compute_voxel_transform(vertices: np.ndarray):
    v = np.asarray(vertices, dtype=np.float64)
    vmin = v.min(axis=0); vmax = v.max(axis=0)
    center = (vmin + vmax) / 2.0
    ext = float((vmax - vmin).max())
    scale = 0.99999 / ext if ext > 0 else 1.0
    return center, scale


def apply_voxel_transform(points, center, scale):
    return (np.asarray(points, dtype=np.float64) - center) * scale


def voxel_centers(coords: np.ndarray, grid: int = 256) -> np.ndarray:
    return -0.5 + (np.asarray(coords, dtype=np.float64) + 0.5) / grid


def sample_voxel_emission(points_vf, coords, values, grid: int = 256, cap_voxels: float = 1.5):
    centers = voxel_centers(coords, grid)
    tree = cKDTree(centers)
    dist, idx = tree.query(np.asarray(points_vf, dtype=np.float64), k=1)
    emis = np.asarray(values, dtype=np.float64)[idx].copy()
    miss = dist > (cap_voxels / grid)
    emis[miss] = 0.0
    return emis, float(miss.mean())


def occupancy_overlap(points_vf, gt_coords, grid: int = 256, cap_voxels: float = 1.5):
    """(share of points within `cap_voxels` of an occupied voxel, share of voxels within it of a point).

    The first is the evaluator's ``align_fp``: a low value means the GLB and the voxels are not in one
    frame. The second reads low under point undersampling even when the two are aligned, so nothing
    decides on it.
    """
    centers = voxel_centers(gt_coords, grid)
    cap = cap_voxels / grid
    pts = np.asarray(points_vf, dtype=np.float64)
    d_pts, _ = cKDTree(centers).query(pts, k=1)
    d_vox, _ = cKDTree(pts).query(centers, k=1)
    return float((d_pts <= cap).mean()), float((d_vox <= cap).mean())
