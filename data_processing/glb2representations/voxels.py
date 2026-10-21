"""The o-voxels: voxelize a canonical `MeshBundle` with o_voxel into emission and PBR `.vxz`
files, plus a `coords.npz` for the alignment gate.

The voxel frame is `[-0.5, 0.5]`: `_tight_scene` re-normalizes *copies* of the geometries
into it, so the bundle the atlas and the validator read stays in `[-1, 1]`. o_voxel's
trimesh path samples textures as raw bytes with no sRGB decode and needs square
power-of-two textures, so the copies get linear `factor x texture` baked into uint8
textures with white factors (matching the atlas bake), resized to square power of two.
"""
import os

import numpy as np
import torch
import trimesh
import o_voxel
from PIL import Image

from data_processing.glb2representations.utils import (
    GRID,
    TIGHT_SCALE_NUM,
    VOXEL_HALF,
    MeshBundle,
    VoxelResult,
)
from utils.color import srgb_to_linear
from utils.trimesh_materials import emission_source, factor_alpha_u8, factor_rgb

_TEXTURE_ATTRS = (
    "baseColorTexture",
    "emissiveTexture",
    "metallicRoughnessTexture",
    "normalTexture",
    "occlusionTexture",
)

# o_voxel's LINEAR mip sampling of EMISSIVE textures is buggy: for textures >=512^2
# (and smaller ones on fine-triangle geometry) the coarse mip levels read garbage, and
# the emissive default is white {1,1,1}, so a black/sparse emissive texture voxelizes to
# near-white -- a catastrophic emission false positive (base_color/metal/rough are NOT
# affected; only emissive). Forcing the finest mip (a very negative offset makes
# o_voxel's `mip_level <= 0` branch sample the base texture) sidesteps the bug entirely.
# Point-sampling the full-res texture at each voxel center is also the correct behavior
# for building 256^3 GT voxels -- it matches the nvdiffrast atlas bake (which point-
# samples too) and, empirically, leaves color/metal/rough MAE unchanged-or-better.
_FINEST_MIP_OFFSET = -1e6


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _nearest_power_of_two(n: int) -> int:
    """Same tie-break as o_voxel.convert.volumetic_attr.nearest_power_of_two."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if _is_power_of_two(n):
        return n
    lower = 2 ** (n.bit_length() - 1)
    upper = 2 ** n.bit_length()
    return lower if n - lower < upper - n else upper


def _resize_texture_square_pot(image: Image.Image) -> Image.Image:
    """Stretch-resize a PIL image to square power-of-two (UV-normalized safe).

    Images with an alpha band are resized per-plane: PIL resamples RGBA/LA in
    premultiplied-alpha space, which permanently blacks out the color wherever
    alpha == 0 (o_voxel samples the color and alpha planes independently, so
    the color under transparent texels is real data we must preserve).
    """
    w, h = image.size
    if w == h and _is_power_of_two(w):
        return image
    size = _nearest_power_of_two(max(w, h))
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        rgb = rgba.convert("RGB").resize((size, size), Image.LANCZOS)
        out = rgb.convert("RGBA")
        out.putalpha(rgba.getchannel("A").resize((size, size), Image.LANCZOS))
        return out
    return image.resize((size, size), Image.LANCZOS)


def _ensure_square_pot_textures(geoms) -> None:
    """In-place: resize each material texture on `geoms` to square PoT.

    Operates on geometry *copies* only. Shared PIL images across geoms are
    resized once (keyed by `id(image)`).
    """
    cache = {}
    for g in geoms:
        mat = getattr(getattr(g, "visual", None), "material", None)
        if mat is None:
            continue
        for attr in _TEXTURE_ATTRS:
            img = getattr(mat, attr, None)
            if img is None or not isinstance(img, Image.Image):
                continue
            key = id(img)
            if key not in cache:
                cache[key] = _resize_texture_square_pot(img)
            setattr(mat, attr, cache[key])


def _ensure_uv(geom) -> None:
    """Guarantee per-vertex UVs so a 1x1 constant texture samples correctly."""
    uv = getattr(geom.visual, "uv", None)
    if uv is None or len(uv) != len(geom.vertices):
        geom.visual.uv = np.zeros((len(geom.vertices), 2), dtype=np.float32)


def _linearize_materials(geoms) -> None:
    """Rewrite PBR on geom *copies* so o_voxel stores linear uint8.

    Base color: the same factor×texture contract as `atlas._sample_material`, alpha
    included. Emission: `utils.trimesh_materials.emission_source`, the same rule as the
    atlas (texture alone, else the factor color, no strength). Textures become
    `clip(linear_rgb,0,1)*255` with white factors; factor-only materials get a
    1×1 constant texture. Metal/rough are already linear — left unchanged.
    """
    for g in geoms:
        mat = getattr(getattr(g, "visual", None), "material", None)
        if mat is None:
            continue
        has_uv = (
            getattr(g.visual, "uv", None) is not None
            and len(g.visual.uv) == len(g.vertices)
        )

        # --- base color (sRGB texture → linear; factor already linear) ---
        base_f = factor_rgb(getattr(mat, "baseColorFactor", None), default=1.0)
        alpha = factor_alpha_u8(getattr(mat, "baseColorFactor", None))
        base_tex = getattr(mat, "baseColorTexture", None) if has_uv else None
        if base_tex is not None:
            arr = np.asarray(base_tex.convert("RGBA"))
            rgb = srgb_to_linear(arr[..., :3].astype(np.float64) / 255.0) * base_f
            out = np.empty_like(arr)
            out[..., :3] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
            # The factor's ALPHA has to be folded in here for the same reason its RGB does:
            # `baseColorFactor` is reset to opaque white below, so anything not baked into
            # the texture is lost. glTF defines base color as factor x texture on all four
            # channels, the same product atlas.py's `_sample_material_alpha` computes.
            out[..., 3] = np.clip(
                np.round(arr[..., 3].astype(np.float64) * (alpha / 255.0)), 0, 255
            ).astype(np.uint8)
            mat.baseColorTexture = Image.fromarray(out, mode="RGBA")
        else:
            rgb = np.clip(np.round(base_f * 255.0), 0, 255).astype(np.uint8)
            px = np.zeros((1, 1, 4), dtype=np.uint8)
            px[0, 0, :3] = rgb
            px[0, 0, 3] = alpha
            mat.baseColorTexture = Image.fromarray(px, mode="RGBA")
            _ensure_uv(g)
        mat.baseColorFactor = np.asarray([255, 255, 255, 255], dtype=np.uint8)

        # --- emission: the rule in utils.trimesh_materials.emission_source ---
        # A declared texture is the emission whatever the factor says, and it is kept on
        # a part without source UVs too (o_voxel then samples texel (0,0), as the atlas
        # does); without a texture the factor is the color. No strength, no multiply.
        kind, src = emission_source(mat)
        if kind == "texture":
            arr = np.asarray(src.convert("RGB"))
            rgb = srgb_to_linear(arr.astype(np.float64) / 255.0)
            out = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
            mat.emissiveTexture = Image.fromarray(out, mode="RGB")
            mat.emissiveFactor = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
            _ensure_uv(g)
        elif kind == "color":
            rgb = np.clip(np.round(src * 255.0), 0, 255).astype(np.uint8)
            mat.emissiveTexture = Image.fromarray(rgb.reshape(1, 1, 3), mode="RGB")
            mat.emissiveFactor = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
            _ensure_uv(g)
        else:
            mat.emissiveFactor = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
            mat.emissiveTexture = None


def _tight_scene(bundle: MeshBundle) -> trimesh.Scene:
    """Copy geometries and tight-normalize vertices into `[-0.5, 0.5]`."""
    copies = [g.copy() for g in bundle.geometries]
    copies = [c for c in copies if len(c.vertices) and len(c.faces)]
    if not copies:
        raise ValueError("no mesh geometry in bundle")

    # Linearize BEFORE PoT resize so LANCZOS runs on already-linear bytes.
    _linearize_materials(copies)

    # o_voxel's trimesh path asserts square power-of-two textures; resize the
    # copies so the canonical atlas bundle is untouched.
    _ensure_square_pot_textures(copies)

    allv = np.concatenate(
        [np.asarray(c.vertices, dtype=np.float64) for c in copies], axis=0
    )
    center = (allv.min(0) + allv.max(0)) / 2.0
    ext = float((allv.max(0) - allv.min(0)).max())
    scale = TIGHT_SCALE_NUM / ext if ext > 0 else 1.0
    for c in copies:
        c.vertices = (np.asarray(c.vertices, dtype=np.float64) - center) * scale
        assert np.all(c.vertices >= -VOXEL_HALF - 1e-4) and np.all(
            c.vertices <= VOXEL_HALF + 1e-4
        ), "vertices out of voxel frame after tight normalize"

    return trimesh.Scene(copies)


def voxelize_bundle(
    bundle: MeshBundle,
    out_dir: str,
    sha: str,
    resolution: int = GRID,
) -> VoxelResult:
    """Voxelize `bundle` into emission/pbr `.vxz` + `coords.npz` under `out_dir`."""
    scene = _tight_scene(bundle)
    coord, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        scene,
        grid_size=resolution,
        aabb=[[-VOXEL_HALF, -VOXEL_HALF, -VOXEL_HALF],
              [VOXEL_HALF, VOXEL_HALF, VOXEL_HALF]],
        mip_level_offset=_FINEST_MIP_OFFSET,
        verbose=False,
        timing=False,
    )
    attr.pop("normal", None)

    emis_dir = os.path.join(out_dir, f"emission_voxels_{resolution}")
    pbr_dir = os.path.join(out_dir, f"pbr_voxels_{resolution}")
    os.makedirs(emis_dir, exist_ok=True)
    os.makedirs(pbr_dir, exist_ok=True)
    emis_vxz = os.path.join(emis_dir, f"{sha}.vxz")
    pbr_vxz = os.path.join(pbr_dir, f"{sha}.vxz")

    # Zero-emission fallback: a mesh with no emissive attr still writes a valid,
    # all-zero emission .vxz co-located on `coord`.
    if "emissive" in attr:
        emissive = attr["emissive"]
    else:
        emissive = torch.zeros((coord.shape[0], 3), dtype=torch.uint8)
    o_voxel.io.write_vxz(emis_vxz, coord, {"emissive": emissive})
    o_voxel.io.write_vxz(pbr_vxz, coord, {
        "base_color": attr["base_color"],
        "metallic": attr["metallic"],
        "roughness": attr["roughness"],
        "alpha": attr["alpha"],
    })

    coords_np = coord.cpu().numpy().astype(np.int32)
    coords_path = os.path.join(out_dir, f"{sha}.coords.npz")
    np.savez(coords_path, coords=coords_np)
    return VoxelResult(
        sha=sha,
        num_voxels=int(coords_np.shape[0]),
        coords_path=coords_path,
        emission_vxz=emis_vxz,
        pbr_vxz=pbr_vxz,
    )
