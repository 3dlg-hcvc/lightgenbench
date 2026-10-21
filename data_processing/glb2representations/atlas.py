"""The UV atlas: xatlas re-parametrizes the canonical mesh and nvdiffrast bakes its PBR
maps into `atlas.npz`.

`bake_atlas(bundle, out_dir, sha, res=512)` takes a canonical `MeshBundle` (from
`utils.load_and_merge`) and writes the 8 keys with the key, dtype and encoding TEXGen's loader reads, so
TEXGen's `spuv/data/mesh_uv.py` loads it unchanged. xatlas keeps face order, so every texel
maps back to its source face, material and UV. Values are linear: base color is factor x
texture, emission follows `utils.trimesh_materials.emission_source` (a declared texture
alone, else the factor as a color, no strength), and alpha is glTF opacity as in
`voxels.py`. Needs a CUDA GPU (nvdiffrast).
"""
import os
import time

import numpy as np

from data_processing.glb2representations.utils import (
    AtlasResult,
    encode_signed_uint16,
    float01_to_uint8,
)
from utils.color import srgb_to_linear
from utils.texture_sampling import sample_texture_repeat
from utils.trimesh_materials import emission_source, factor_alpha, factor_rgb


# --- source-material helpers ------------------------------------------------
def _tex_rgb(texture):
    """PIL texture -> (H, W, 3) uint8 array, or None."""
    if texture is None:
        return None
    return np.asarray(texture.convert("RGB"))


def _sample_material(mat, uv: np.ndarray, has_uv: bool):
    """Per-texel (color, emission, metal, rough) for one material, all LINEAR.

    `uv` is (K, 2) original UVs; `has_uv` is the part-level flag: base color and
    metal/rough sample a texture only when the part carried source UVs, otherwise the
    material factor (never a texture at a bogus (0,0)).

    Emission follows `utils.trimesh_materials.emission_source`: a declared emissive
    texture is the emission whatever the factor says, and it is sampled even on a part
    without UVs (the bundle's UVs are zero there, so it reads texel (0,0), as a renderer
    does for a missing attribute); without a texture the factor is the color. No
    strength, no factor multiply.
    """
    k = len(uv)
    base_factor = factor_rgb(getattr(mat, "baseColorFactor", None), default=1.0)
    metal_factor = getattr(mat, "metallicFactor", None)
    metal_factor = 1.0 if metal_factor is None else float(metal_factor)
    rough_factor = getattr(mat, "roughnessFactor", None)
    rough_factor = 1.0 if rough_factor is None else float(rough_factor)

    base_tex = _tex_rgb(getattr(mat, "baseColorTexture", None)) if has_uv else None
    mr_tex = _tex_rgb(getattr(mat, "metallicRoughnessTexture", None)) if has_uv else None

    if base_tex is not None:
        # baseColor texture is sRGB -> linear, times the (linear) factor.
        color = srgb_to_linear(sample_texture_repeat(base_tex, uv)) * base_factor
    else:
        color = np.tile(base_factor, (k, 1))

    kind, src = emission_source(mat)
    if kind == "texture":
        emission = srgb_to_linear(sample_texture_repeat(_tex_rgb(src), uv))
    elif kind == "color":
        emission = np.tile(src, (k, 1))
    else:
        emission = np.zeros((k, 3), dtype=np.float64)

    if mr_tex is not None:
        mr = sample_texture_repeat(mr_tex, uv)  # linear data, no sRGB
        metal = metal_factor * mr[:, 2]
        rough = rough_factor * mr[:, 1]
    else:
        metal = np.full(k, metal_factor, dtype=np.float64)
        rough = np.full(k, rough_factor, dtype=np.float64)

    return color, emission, metal, rough


def _sample_material_alpha(mat, uv: np.ndarray, has_uv: bool) -> np.ndarray:
    """Per-texel glTF opacity in [0,1], mirroring o_voxel's convention exactly.

    glTF semantics (and o_voxel's C++): OPAQUE ignores alpha entirely -> 1.0;
    BLEND = baseColorFactor[3] x baseColorTexture A; MASK additionally
    binarizes at alphaCutoff. Alpha is linear coverage data — no sRGB.
    """
    k = len(uv)
    alpha_mode = getattr(mat, "alphaMode", None)
    # Exact o_voxel wrapper parity: only literal "BLEND"/"MASK" carry alpha;
    # OPAQUE, None, and any unrecognized value all mean fully opaque.
    if alpha_mode not in ("BLEND", "MASK"):
        return np.ones(k, dtype=np.float64)
    a = factor_alpha(getattr(mat, "baseColorFactor", None))
    base_tex = getattr(mat, "baseColorTexture", None) if has_uv else None
    if base_tex is not None:
        # Always go through RGBA: palette ('P') images carry transparency that
        # getbands() does not expose as an 'A' band. Plain RGB converts to a
        # constant-255 A channel, which reduces to the factor — harmless.
        ta = np.asarray(base_tex.convert("RGBA"))[..., 3:4]
        # `sample_texture_repeat` already maps uint8 -> [0,1]; do NOT divide again.
        av = sample_texture_repeat(np.repeat(ta, 3, axis=2), uv)[:, 0]
        out = np.clip(a * av, 0.0, 1.0)
    else:
        out = np.full(k, np.clip(a, 0.0, 1.0), dtype=np.float64)
    if alpha_mode == "MASK":
        cutoff = getattr(mat, "alphaCutoff", None)
        cutoff = 0.5 if cutoff is None else float(cutoff)
        out = np.where(out < cutoff, 0.0, 1.0)
    return out


# --- edge dilation ----------------------------------------------------------
def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift `a` by (dy, dx) with zero fill (no wrap-around)."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    out[ys:ye, xs:xe] = a[sy0:sy1, sx0:sx1]
    return out


_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _dilate(img: np.ndarray, valid: np.ndarray, iters: int) -> np.ndarray:
    """Iterative nearest-valid-neighbor fill of the unoccupied border ring, to
    push chart colors a few px past their boundary (anti-seam). Does NOT touch
    the occupancy mask -- that stays the true rasterized coverage."""
    img = img.astype(np.float64).copy()
    valid = valid.copy()
    for _ in range(iters):
        acc = np.zeros_like(img)
        cnt = np.zeros(valid.shape, dtype=np.float64)
        for dy, dx in _NEIGHBORS:
            sv = _shift(valid.astype(np.float64), dy, dx)
            acc += _shift(img, dy, dx) * sv[..., None]
            cnt += sv
        new = (~valid) & (cnt > 0)
        safe = cnt.copy()
        safe[safe == 0] = 1.0
        filled = acc / safe[..., None]
        img[new] = filled[new]
        valid = valid | new
    return img


# --- main -------------------------------------------------------------------
def bake_atlas(bundle, out_dir: str, sha: str, res: int = 512, dilate_iters: int = 4) -> AtlasResult:
    import torch
    import nvdiffrast.torch as dr
    import xatlas

    t = {}
    os.makedirs(out_dir, exist_ok=True)

    verts = np.ascontiguousarray(bundle.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(bundle.faces, dtype=np.int64)

    # 1. xatlas re-parametrize the whole merged mesh.
    t0 = time.time()
    vmapping, indices, uvs = xatlas.parametrize(
        verts.astype(np.float32), faces.astype(np.uint32)
    )
    t["xatlas"] = time.time() - t0
    indices = np.asarray(indices, dtype=np.int64)
    vmapping = np.asarray(vmapping, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32)
    # xatlas preserves face order: output face i <-> original face i.
    assert indices.shape[0] == faces.shape[0], (
        f"xatlas changed face count {indices.shape[0]} != {faces.shape[0]}"
    )

    # 2. Rasterize the atlas UV as clip space (2u-1, 2v-1, 0, 1).
    t0 = time.time()
    device = "cuda"
    uv_clip = torch.from_numpy(uvs).to(device) * 2.0 - 1.0
    ones = torch.ones_like(uv_clip[:, :1])
    zeros = torch.zeros_like(uv_clip[:, :1])
    pos_clip = torch.cat([uv_clip, zeros, ones], dim=-1)[None]      # (1, V, 4)
    tri = torch.from_numpy(indices.astype(np.int32)).to(device).contiguous()
    glctx = dr.RasterizeCudaContext(device=device)
    rast, _ = dr.rasterize(glctx, pos_clip.contiguous(), tri, resolution=[res, res])

    # 3a. Interpolate canonical position and original UV via nvdiffrast.
    vpos = torch.from_numpy(verts[vmapping].astype(np.float32)).to(device)[None]
    pos_map, _ = dr.interpolate(vpos, rast, tri)                   # (1, H, W, 3)

    # per-original-vertex original UV (source UVs are per-vertex within a part;
    # zeros for no-UV parts, guarded by face_has_uv downstream).
    orig_uv_pv = np.zeros((verts.shape[0], 2), dtype=np.float32)
    fu = np.asarray(bundle.face_uv, dtype=np.float32)
    for corner in range(3):
        orig_uv_pv[faces[:, corner]] = fu[:, corner, :]
    ouv = torch.from_numpy(orig_uv_pv[vmapping]).to(device)[None]
    ouv_map, _ = dr.interpolate(ouv, rast, tri)                    # (1, H, W, 2)
    t["rasterize"] = time.time() - t0

    rast_np = rast[0].detach().cpu().numpy()
    tri_id = rast_np[..., 3].astype(np.int64)                      # 0 = empty
    occ = tri_id > 0
    pos_all = pos_map[0].detach().cpu().numpy()
    ouv_all = ouv_map[0].detach().cpu().numpy()

    occupied_texels = int(occ.sum())

    # 3b. Per-face gathers (normal, material id, has-uv) at occupied texels.
    face_hit = tri_id[occ] - 1
    tri3 = verts[faces]                                            # (M, 3, 3)
    fn = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.where(ln > 1e-12, fn / np.clip(ln, 1e-12, None), np.array([0.0, 0.0, 1.0]))
    normal_hit = fn[face_hit]
    mat_hit = np.asarray(bundle.face_mat, dtype=np.int64)[face_hit]
    has_uv_hit = np.asarray(bundle.face_has_uv, dtype=bool)[face_hit]
    uv_hit = ouv_all[occ]
    pos_hit = pos_all[occ]

    # 4. Sample source PBR per material (grouped for texture-load efficiency).
    t0 = time.time()
    k = occupied_texels
    color = np.zeros((k, 3), dtype=np.float64)
    emission = np.zeros((k, 3), dtype=np.float64)
    metal = np.zeros(k, dtype=np.float64)
    rough = np.zeros(k, dtype=np.float64)
    alpha = np.ones(k, dtype=np.float64)
    for mid in np.unique(mat_hit):
        sel = mat_hit == mid
        mat = bundle.geometries[int(mid)].visual.material
        part_has_uv = bool(has_uv_hit[sel].any())
        c, e, m, r = _sample_material(mat, uv_hit[sel], part_has_uv)
        color[sel] = c
        emission[sel] = e
        metal[sel] = m
        rough[sel] = r
        alpha[sel] = _sample_material_alpha(mat, uv_hit[sel], part_has_uv)
    t["sample_pbr"] = time.time() - t0

    # 5. Assemble dense maps; occupancy stays the rasterized coverage.
    h = w = res
    position = np.zeros((h, w, 3), dtype=np.float64)
    objnormal = np.zeros((h, w, 3), dtype=np.float64)
    color_map = np.zeros((h, w, 3), dtype=np.float64)
    emission_map = np.zeros((h, w, 3), dtype=np.float64)
    metal_map = np.zeros((h, w, 1), dtype=np.float64)
    rough_map = np.zeros((h, w, 1), dtype=np.float64)
    alpha_map = np.zeros((h, w, 1), dtype=np.float64)
    position[occ] = pos_hit
    objnormal[occ] = normal_hit
    color_map[occ] = color
    emission_map[occ] = emission
    metal_map[occ] = metal[:, None]
    rough_map[occ] = rough[:, None]
    alpha_map[occ] = alpha[:, None]

    # 6. Dilate all channels together past chart edges (occupancy stays true).
    t0 = time.time()
    if dilate_iters > 0 and occ.any():
        stack = np.concatenate(
            [position, objnormal, color_map, emission_map, metal_map, rough_map,
             alpha_map], axis=2
        )
        stack = _dilate(stack, occ, dilate_iters)
        position = stack[..., 0:3]
        objnormal = stack[..., 3:6]
        color_map = stack[..., 6:9]
        emission_map = stack[..., 9:12]
        metal_map = stack[..., 12:13]
        rough_map = stack[..., 13:14]
        alpha_map = stack[..., 14:15]
    t["dilate"] = time.time() - t0

    # re-normalize normals after dilation averaging (keep unit for the encoder).
    nln = np.linalg.norm(objnormal, axis=2, keepdims=True)
    objnormal = np.where(nln > 1e-8, objnormal / np.clip(nln, 1e-8, None), objnormal)

    # 7. Encode (TEXGen's atlas format) + save the 8 keys.
    out = {
        "occupancy": occ[..., None].astype(bool),
        "position": encode_signed_uint16(np.clip(position, -1.0, 1.0)),
        "objnormal": encode_signed_uint16(np.clip(objnormal, -1.0, 1.0)),
        "color": float01_to_uint8(np.clip(color_map, 0.0, 1.0)),
        "metal": float01_to_uint8(np.clip(metal_map, 0.0, 1.0)),
        "rough": float01_to_uint8(np.clip(rough_map, 0.0, 1.0)),
        "emission_color": float01_to_uint8(np.clip(emission_map, 0.0, 1.0)),
        "alpha": float01_to_uint8(np.clip(alpha_map, 0.0, 1.0)),
    }
    atlas_path = os.path.join(out_dir, "atlas.npz")
    np.savez(atlas_path, **out)

    return AtlasResult(
        sha=sha, atlas_path=atlas_path, occupied_texels=occupied_texels, timings=t
    )
