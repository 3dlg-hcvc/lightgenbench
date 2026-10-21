"""The evaluation's ground truth: points sampled on the GLB's surface and the reference emission
at each, the GLB read part by part the way the bake reads it.

The parts are `scene.dump()` (one per scene-graph node, world transform baked in), in dump order,
keeping `Trimesh` parts with vertices and faces. Nothing is merged: every part keeps its own
material and UVs. A part without per-vertex UVs reads UV (0, 0); a part whose visual carries no
material emits nothing. As in the bake, every emissive texture goes through `convert("RGB")` and is
read at TEXCOORD_0 with REPEAT wrap, without `KHR_texture_transform`, whatever the material's
`texCoord` or sampler says.

The points stay in the GLB's world frame; the voxel frame comes from the bounding box of the kept
parts (`converters.voxels.compute_voxel_transform`), the box the bake normalizes. One
`trimesh.sample.sample_surface` call over the concatenated parts draws faces by world-space area,
and each point's UV is interpolated from its own part's UVs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from evaluation.pointsampled.metrics import gt_to_canonical_linear
from utils.texture_sampling import sample_texture_repeat
from utils.trimesh_materials import emission_source

@dataclass
class ShapeParts:
    """The kept parts of one GLB, concatenated in dump order, in the GLB's world frame."""
    vertices: np.ndarray     # (V, 3) float64
    faces: np.ndarray        # (F, 3) int64, into `vertices`
    face_part: np.ndarray    # (F,) int64, index into `materials`
    vertex_uv: np.ndarray    # (V, 2) float64, each part's own UVs; zeros on a part without UVs
    part_has_uv: np.ndarray  # (P,) bool
    materials: list          # (P,) the part's material, or None when its visual carries none


@dataclass
class PartSample:
    """Surface points drawn from a `ShapeParts`."""
    points: np.ndarray       # (N, 3) float64
    face_index: np.ndarray   # (N,) int64, into `ShapeParts.faces`
    part_index: np.ndarray   # (N,) int64
    uv: np.ndarray           # (N, 2) float64; (0, 0) on a part without UVs


def _part_material(part):
    """The part's material, or None where the bake synthesizes an emission-free PBR material."""
    vis = part.visual
    if isinstance(vis, trimesh.visual.TextureVisuals):
        return vis.material  # may be None: the bake gives it a default PBRMaterial, which emits nothing
    return None               # ColorVisuals / no visual: the bake's constant-baseColor material, no emission


def _part_uv(part):
    """(V, 2) float64 UVs, or None when the part has no per-vertex UVs (the bake's `_face_uv` test)."""
    uv = getattr(part.visual, "uv", None)
    if uv is None or len(uv) != len(part.vertices):
        return None
    return np.asarray(uv, dtype=np.float64)


def parts_from_scene(loaded) -> ShapeParts:
    """Split a loaded trimesh Scene (or single Trimesh) into the bake's parts, without normalizing."""
    if isinstance(loaded, trimesh.Scene):
        parts = loaded.dump()  # concatenate=False: one copy per node, world transform baked in
    else:
        parts = [loaded]
    parts = [p for p in parts if isinstance(p, trimesh.Trimesh)
             and len(p.vertices) and len(p.faces)]
    if not parts:
        raise ValueError("no mesh geometry in loaded input")

    # The same call util.concatenate (hence to_geometry) makes, so the arrays match it exactly.
    vertices, faces = trimesh.util.append_faces(
        [p.vertices.copy() for p in parts], [p.faces.copy() for p in parts])
    uvs = [_part_uv(p) for p in parts]
    vertex_uv = np.concatenate(
        [np.zeros((len(p.vertices), 2)) if uv is None else uv for p, uv in zip(parts, uvs)], axis=0)
    face_part = np.concatenate(
        [np.full(len(p.faces), k, dtype=np.int64) for k, p in enumerate(parts)])
    return ShapeParts(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        face_part=face_part,
        vertex_uv=vertex_uv,
        part_has_uv=np.array([uv is not None for uv in uvs], dtype=bool),
        materials=[_part_material(p) for p in parts],
    )


def load_parts(glb_path: str) -> ShapeParts:
    return parts_from_scene(trimesh.load(glb_path, process=False))


def sample_parts(parts: ShapeParts, n_points: int, seed: int = 0) -> PartSample:
    mesh = trimesh.Trimesh(vertices=parts.vertices, faces=parts.faces, process=False)
    points, face_index = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    points = np.asarray(points, dtype=np.float64)
    face_index = np.asarray(face_index)
    # Barycentric UVs from the global per-vertex UV array: a triangle's three corners belong to
    # one part, so this interpolates that part's own UVs.
    tri = mesh.triangles[face_index]
    bary = trimesh.triangles.points_to_barycentric(tri, points)
    uv = np.einsum("nj,njk->nk", bary, parts.vertex_uv[parts.faces[face_index]])
    part_index = parts.face_part[face_index]
    # A part without UVs has zero corner UVs, so this matters only for a non-finite bary (a sliver
    # triangle). On a part WITH UVs a non-finite UV is left alone and errors the shape in the
    # sampler.
    uv[~parts.part_has_uv[part_index]] = 0.0
    return PartSample(points=points, face_index=face_index, part_index=part_index, uv=uv)


def _texel_and_factor(mat, uv: np.ndarray):
    """(sRGB texel, linear factor) per point for one part's material."""
    k = len(uv)
    kind, src = emission_source(mat)
    if kind == "texture":
        # atlas._tex_rgb: every image mode ('P', '1', 'L', 'LA', 'RGBA', ...) through RGB.
        # Converted once per part and dropped after it: `Scene.dump()` deep-copies each node's
        # material, so no two parts share an image object and a cache would never hit.
        texel = sample_texture_repeat(np.asarray(src.convert("RGB")), uv)
        return texel, np.broadcast_to(np.ones(3, dtype=np.float64), (k, 3))  # the texture alone
    ones = np.ones((k, 3), dtype=np.float64)
    if kind == "color":
        return ones, np.broadcast_to(src, (k, 3))
    return ones, np.zeros((k, 3), dtype=np.float64)


def part_gt_linear(parts: ShapeParts, sample: PartSample) -> np.ndarray:
    """Per-point GT emission in the canonical space (`metrics.gt_to_canonical_linear`), (N, 3).

    Each point is read through its own part's material (`utils.trimesh_materials.emission_source`)
    by the bake's rule, what a model trained on the released dataset saw: a declared emissive texture
    IS the emission, ``srgb_to_linear(texel)`` with the factor ignored; without a texture the factor
    is the (linear) color, parsed by `factor_rgb` as the bake parses it; neither -> 0. Neither
    ``KHR_materials_emissive_strength`` nor alpha is read. Values are clipped to [0, 1].
    """
    n = len(sample.points)
    texel = np.ones((n, 3), dtype=np.float64)
    factor = np.zeros((n, 3), dtype=np.float64)
    # Group the points by part with one sort (a shape can have thousands of parts).
    order = np.argsort(sample.part_index, kind="stable")
    ids, starts = np.unique(sample.part_index[order], return_index=True)
    ends = np.append(starts[1:], n)
    for k, a, b in zip(ids, starts, ends):
        sel = order[a:b]
        texel[sel], factor[sel] = _texel_and_factor(parts.materials[int(k)], sample.uv[sel])
    return gt_to_canonical_linear(texel, factor)
