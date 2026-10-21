"""Multiview renderings: the six fixed Hunyuan3D-2.1 Paint cameras, rasterized with
nvdiffrast and shaded from the shape's `atlas.npz`.

`render_geometry` rasterizes a `MeshBundle` (mask, position, world-space normal, depth)
with upstream's orthographic projection (`_proj`) in upstream's Z-up frame
(`canonical_to_hunyuan_frame`). `build_texel_cloud` and `shade_views` give every pixel the
albedo, metal, rough, alpha and emission of its nearest occupied atlas texel in that same
frame. `render_from_bundle` writes the `render_tex/` and `render_cond/` layout upstream
trains on; upstream's `_pos` and `_normal` channel permutations are applied at save time
only, and `render_geometry`'s raw row order (row 0 = bottom) gets one `np.flipud` right
before saving. torch and nvdiffrast are imported inside `render_geometry`, so `CAMERAS`
and `VIEW_MAPS` import without them.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from data_processing.glb2representations.utils import decode_signed_uint16

RES = 512
BG_GRAY = 127            # background fill outside `mask` on every saved map:
                          # upstream's `bg_gray` training default
COVERAGE_TOL = 2.0 / RES  # nearest-texel distance gate (Hunyuan-frame units)
PNG_COMPRESS_LEVEL = 3    # PIL default is 6; zlib encode is most of the per-shape
                          # time, and level 3 encodes about 1.6x faster for about
                          # 1.16x the size (PNG is lossless at any level)

#: The six maps saved per view (36 PNGs per shape).
VIEW_MAPS = ("albedo", "mr", "normal", "pos", "emission", "alpha")


# --- the cameras -------------------------------------------------------------
#
# The six Hunyuan3D-2.1 Paint training-example cameras, copied verbatim from
# `Hunyuan3D-Emission/hy3dpaint/train_examples/001/render_tex/transforms.json`
# (frames 000-005: four around the object plus top and bottom, all at
# `cam_dis=1.5`), so the views match upstream's training distribution. Upstream
# renders them orthographically, so `camera_angle_x` is kept verbatim but plays no
# part in the projection (`_proj`). `regenerate()` re-reads that file (or
# `$HUNYUAN_TRANSFORMS_JSON` when the submodule is absent) and asserts the frames
# still match CAMERAS bit for bit.

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_TRANSFORMS_JSON = (
    REPO_ROOT / "Hunyuan3D-Emission" / "hy3dpaint" / "train_examples"
    / "001" / "render_tex" / "transforms.json"
)
TRANSFORMS_JSON_ENV_VAR = "HUNYUAN_TRANSFORMS_JSON"


def _c2w(rows):
    return np.array(rows, dtype=np.float64)


# Verbatim from upstream transforms.json (aabb [[-0.5]*3, [0.5]*3], their own
# per-asset scale/offset are NOT reused -- see write_transforms_json below).
CAMERAS = [
    dict(
        file_stem="000",
        azimuth=-1.5707963267948966,
        elevation=0,
        cam_dis=1.5,
        camera_angle_x=0.9232195630055808,
        c2w=_c2w([
            [1.0, 0.0, 6.123233601181349e-17, 9.184850732644269e-17],
            [6.123233601181349e-17, 3.422854177870249e-08, -0.9999999403953552, -1.5],
            [0.0, 0.9999999403953552, 3.422854177870249e-08, 0.0],
            [0, 0, 0, 1],
        ]),
    ),
    dict(
        file_stem="001",
        azimuth=0,
        elevation=0,
        cam_dis=1.5,
        camera_angle_x=0.9232195630055808,
        c2w=_c2w([
            [-2.220446049250313e-16, 0.0, 1.0, 1.5],
            [1.0, -2.220446049250313e-16, 0.0, 0.0],
            [0.0, 1.0, -2.220446049250313e-16, 0.0],
            [0, 0, 0, 1],
        ]),
    ),
    dict(
        file_stem="002",
        azimuth=1.5707963267948966,
        elevation=0,
        cam_dis=1.5,
        camera_angle_x=0.9232195630055808,
        c2w=_c2w([
            [-0.9999999403953552, 0.0, 6.123233601181349e-17, 9.184850732644269e-17],
            [6.123233601181349e-17, 3.422854177870249e-08, 0.9999999403953552, 1.5],
            [0.0, 0.9999999403953552, 3.422854177870249e-08, 0.0],
            [0, 0, 0, 1],
        ]),
    ),
    dict(
        file_stem="003",
        azimuth=3.141592653589793,
        elevation=0,
        cam_dis=1.5,
        camera_angle_x=0.9232195630055808,
        c2w=_c2w([
            [-2.220446049250313e-16, 0.0, -1.0, -1.5],
            [-1.0, -2.220446049250313e-16, 0.0, 1.8369701465288538e-16],
            [0.0, 1.0, -2.220446049250313e-16, 0.0],
            [0, 0, 0, 1],
        ]),
    ),
    dict(
        file_stem="004",
        azimuth=-1.5707963267948966,
        elevation=1.5707963267948966,
        cam_dis=1.5,
        camera_angle_x=0.9232195630055808,
        c2w=_c2w([
            [1.0, -6.123233601181349e-17, 2.812049544789739e-33, 5.624099089579478e-33],
            [6.123233601181349e-17, 1.0, -3.0616171314629196e-17, -9.184850732644269e-17],
            [-9.373498482632464e-34, 3.0616171314629196e-17, 1.0, 1.5],
            [0, 0, 0, 1],
        ]),
    ),
    dict(
        file_stem="005",
        azimuth=-1.5707963267948966,
        elevation=-1.5707963267948966,
        cam_dis=1.5,
        camera_angle_x=0.9232195630055808,
        c2w=_c2w([
            [1.0, 6.123233601181349e-17, 2.812049544789739e-33, 5.624099089579478e-33],
            [6.123233601181349e-17, -1.0, -3.0616171314629196e-17, -9.184850732644269e-17],
            [9.373498482632464e-34, 3.0616171314629196e-17, -1.0, -1.5],
            [0, 0, 0, 1],
        ]),
    ),
]


def _resolve_transforms_json_path(transforms_json_path=None):
    """Discovery logic for `regenerate()`: an explicit path wins, else prefer the
    submodule path, else the `HUNYUAN_TRANSFORMS_JSON` env var. Raises
    FileNotFoundError if none is available."""
    if transforms_json_path is not None:
        return transforms_json_path
    if SUBMODULE_TRANSFORMS_JSON.exists():
        return SUBMODULE_TRANSFORMS_JSON
    if os.environ.get(TRANSFORMS_JSON_ENV_VAR):
        return os.environ[TRANSFORMS_JSON_ENV_VAR]
    raise FileNotFoundError(
        f"no transforms.json found: neither {SUBMODULE_TRANSFORMS_JSON} "
        f"nor ${TRANSFORMS_JSON_ENV_VAR} is set"
    )


def regenerate(transforms_json_path=None):
    """Re-read a transforms.json and assert its frames 000-005 match CAMERAS.

    `transforms_json_path` overrides discovery; otherwise prefers the
    submodule path, then the `HUNYUAN_TRANSFORMS_JSON` env var (see
    `_resolve_transforms_json_path`). Raises FileNotFoundError if neither is
    available, AssertionError if the file's frames don't match CAMERAS
    bit-for-bit. Returns True on success.
    """
    path = _resolve_transforms_json_path(transforms_json_path)
    with open(path) as f:
        data = json.load(f)
    frames = data["frames"]
    assert len(frames) == len(CAMERAS), f"expected {len(CAMERAS)} frames, got {len(frames)}"
    for cam, frame in zip(CAMERAS, frames):
        stem = os.path.splitext(frame["file_path"])[0]
        assert stem == cam["file_stem"], (stem, cam["file_stem"])
        assert frame["azimuth"] == cam["azimuth"], (frame["azimuth"], cam["azimuth"])
        assert frame["elevation"] == cam["elevation"], (frame["elevation"], cam["elevation"])
        assert frame["cam_dis"] == cam["cam_dis"], (frame["cam_dis"], cam["cam_dis"])
        assert frame["camera_angle_x"] == cam["camera_angle_x"], (
            frame["camera_angle_x"], cam["camera_angle_x"]
        )
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        assert np.array_equal(c2w, cam["c2w"]), (cam["file_stem"], c2w, cam["c2w"])
    return True


def write_transforms_json(out_dir, scale=0.5, offset=(0.0, 0.0, 0.0)):
    """Write OUR fixture's transforms.json: upstream's six camera frames
    verbatim (azimuth/elevation/cam_dis/camera_angle_x/transform_matrix), but
    OUR OWN scale/offset. Our canonical glTF Y-up mesh frame is [-1,1];
    `canonical_to_hunyuan_frame` first applies the Z-up axis
    swap (x,y,z) -> (x,-z,y) -- a rotation, which this json format has no
    field for (only a plain isotropic scale + offset) -- and then a uniform
    *0.5 into upstream's aabb [-0.5,0.5]. So scale=0.5, offset=[0,0,0] here
    records only that magnitude step; the rotation lives in our own pipeline
    code, not in this json. Unlike upstream's own per-asset scale/offset,
    which centers/fits THEIR example mesh into the same aabb with no
    rotation of their own.
    """
    data = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": list(offset),
        "frames": [
            {
                "file_path": f"{cam['file_stem']}.png",
                "camera_angle_x": cam["camera_angle_x"],
                "proj_type": 1,
                "azimuth": cam["azimuth"],
                "elevation": cam["elevation"],
                "cam_dis": cam["cam_dis"],
                "transform_matrix": cam["c2w"].tolist(),
            }
            for cam in CAMERAS
        ],
    }
    out_path = Path(out_dir) / "transforms.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    return str(out_path)


def canonical_to_hunyuan_frame(p):
    """glTF Y-up canonical [-1,1] -> upstream Z-up aabb [-0.5,0.5].
    (x,y,z)_gltf -> (x,-z,y)_zup, then *0.5. Used by BOTH the mesh path and the
    atlas texel-cloud path -- the frames must never diverge."""
    q = np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)
    return q * 0.5


# --- upstream's saved-map byte conventions -----------------------------------
#
# Both derivations below come from upstream's source and agree with upstream's
# shipped `train_examples/001/render_tex/*` and with a render made by upstream's own
# `hy3dshape/tools/render/render.py`. The shared final step in both upstream paths is
# `cv2.imwrite(path, arr)`, which assumes `arr` is BGR: it writes file R <- arr[...,2],
# G <- arr[...,1], B <- arr[...,0]. That single reversal is why neither map is the
# identity.
#
# One deliberate byte-level divergence: upstream quantizes with a truncating
# `astype('uint8')`, we round (`_to_png_u8`). So a neutral component is 128 here and
# 127 upstream -- half an LSB -- and a zero-valued surface pixel stays distinct from
# the BG_GRAY=127 background.

def hunyuan_pos_encode(pos_hunyuan):
    """Hunyuan Z-up position in [-0.5,0.5] -> upstream's saved `_pos` map,
    float RGB in [0,1] (row/col order untouched; caller still flips + scales).

    Source: `hy3dshape/tools/render/render.py::ConvertDepthMap`, which relabels the
    world position Z-up -> Y-up as `(x, z, -y)`, maps it by `(0.5 + p) * 255` and
    writes it with `cv2.imwrite`. With imwrite's BGR reversal that gives

        file R = 0.5 - y,   file G = 0.5 + z,   file B = 0.5 + x

    with (x,y,z) the Blender/Hunyuan Z-up position. Reprojecting upstream's own
    renders confirms it to the 8-bit quantization floor.
    """
    p = np.asarray(pos_hunyuan)
    return np.stack([-p[..., 1], p[..., 2], p[..., 0]], axis=-1) + 0.5


def hunyuan_pos_decode(rgb01):
    """Inverse of `hunyuan_pos_encode`, for anything reading the fixture
    back: saved `_pos` RGB in [0,1] -> Hunyuan Z-up position in [-0.5,0.5]."""
    c = np.asarray(rgb01) - 0.5
    return np.stack([c[..., 2], -c[..., 0], c[..., 1]], axis=-1)


def hunyuan_normal_encode(n_hunyuan):
    """Hunyuan Z-up unit WORLD-space normal in [-1,1] -> upstream's saved
    `_normal` map, float RGB in [0,1].

    1. WORLD-space, not camera-space. Blender's Cycles `Normal` render pass is
       world-space and `render.py` writes it with no vector transform; upstream's
       own renders are front-facing in every view under that reading, and its
       inference-side `render_normal_multiview(..., use_abs_coor=True)` asks for
       world coordinates too.

    2. Channel order is REVERSED: file R = z, G = y, B = x. Of the two
       `ConvertNormalMap` versions in `render.py`, the one that matches the shipped
       maps reads the EXR's X, Y, Z channels by name and writes them with
       `cv2.imwrite`, whose BGR reversal lands x in Blue and z in Red.

        file R = 0.5 + z/2,  file G = 0.5 + y/2,  file B = 0.5 + x/2

    Orientation is `render_geometry`'s job, not this function's: it hands over
    a normal already flipped to face the view's camera (Cycles' behavior), and
    this function only relabels channels and rescales.

    Note this is NOT the same permutation as `hunyuan_pos_encode`'s (upstream's
    two converters disagree with each other: pos also relabels Z-up -> Y-up
    first, normal does not). Reproduced faithfully rather than "fixed", since
    matching upstream's training-data bytes is the point.
    """
    n = np.asarray(n_hunyuan)
    return np.stack([n[..., 2], n[..., 1], n[..., 0]], axis=-1) * 0.5 + 0.5


def hunyuan_normal_decode(rgb01):
    """Inverse of `hunyuan_normal_encode`: saved `_normal` RGB in [0,1] ->
    Hunyuan Z-up world-space normal in [-1,1]."""
    c = np.asarray(rgb01) * 2.0 - 1.0
    return np.stack([c[..., 2], c[..., 1], c[..., 0]], axis=-1)


def _mesh_to_hunyuan_frame(bundle):
    """Canonical glTF Y-up [-1,1] mesh vertices -> upstream's Z-up aabb
    [-0.5,0.5] via `canonical_to_hunyuan_frame` (shared with the atlas texel-cloud
    path so the two frames never diverge; matches `write_transforms_json`'s
    scale=0.5, offset=[0,0,0])."""
    return canonical_to_hunyuan_frame(np.asarray(bundle.vertices, dtype=np.float32))


def _proj(camera_angle_x=None, near=0.01, far=10.0, ortho_scale=1.2):
    """Orthographic projection, as upstream renders these six views: Blender's
    ORTHO camera with `ortho_scale=1.2` when `proj_type==1` (what
    `write_transforms_json` writes), and `MeshRender.py`'s `camera_type="orth"`.
    `camera_angle_x` is accepted but unused.

    Standard GL orthographic projection for l/r/b/t = -0.6/0.6/-0.6/0.6:
        diag(2/ortho_scale, 2/ortho_scale, -2/(far-near), 1)
    plus the usual z translation (far+near)/(near-far); clip.w is always 1.
    """
    s = 2.0 / ortho_scale
    return np.array([
        [s, 0, 0, 0],
        [0, s, 0, 0],
        [0, 0, -2.0 / (far - near), (far + near) / (near - far)],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def render_geometry(bundle, glctx, cameras):
    """Rasterize `bundle` (an object with `.vertices` (N,3) and `.faces` (M,3),
    e.g. a `data_processing.glb2representations.utils.MeshBundle`) from each
    camera in `cameras` (see `CAMERAS`).

    Returns a list (one dict per camera, in `cameras` order), each with:
      mask    (RES,RES)   float32 in {0,1}
      normal  (RES,RES,3) float32 in [-1,1]: WORLD-space unit vertex normal, in
                          the same Hunyuan Z-up frame as `pos_raw`, flipped
                          where needed to point TOWARD this view's camera (see
                          the flip below -- that is Cycles' semantics, so the
                          field is world-space but view-DEPENDENT). RAW
                          direction -- NOT image-encoded: the *0.5+0.5 and the
                          channel permutation upstream saves it in are applied
                          only at save time, by `hunyuan_normal_encode`.
      depth   (RES,RES)   float32 raw NDC z under OUR orthographic projection
                          (near=0.01, far=10.0 -- our own choice, NOT
                          upstream-comparable; since ortho clip.w == 1, this is
                          just z, not z/w; 0 outside mask)
      pos_raw (RES,RES,3) float32 object-space position in [-0.5,0.5], Hunyuan
                          Z-up frame. Also raw -- `hunyuan_pos_encode` turns it
                          into the saved `_pos` map at save time.
    All in nvdiffrast's raw (unflipped) row order -- see the module docstring.

    There is deliberately no pre-encoded `pos`/`normal` image field here: a
    naive `+0.5` / `*0.5+0.5` array is NOT what upstream saves (both maps
    permute channels), so offering one would be a trap for callers.
    """
    import torch
    import nvdiffrast.torch as dr

    v = torch.from_numpy(_mesh_to_hunyuan_frame(bundle)).cuda()
    f = torch.from_numpy(np.asarray(bundle.faces, dtype=np.int32)).cuda()
    v_h = torch.cat([v, torch.ones_like(v[:, :1])], dim=1)

    # Per-vertex normals from face normals (area-weighted via the cross-product
    # magnitude, since fn is not pre-normalized), in the same object frame as v.
    fn = torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = torch.zeros_like(v).index_add_(
        0, f.reshape(-1).long(), fn.repeat_interleave(3, dim=0)
    )
    vn = torch.nn.functional.normalize(vn, dim=1)

    outs = []
    for cam in cameras:
        w2c = torch.from_numpy(np.linalg.inv(cam["c2w"]).astype(np.float32)).cuda()
        mvp = torch.from_numpy(_proj(cam["camera_angle_x"])).cuda() @ w2c
        clip = (v_h @ mvp.T)[None]
        rast, _ = dr.rasterize(glctx, clip, f, resolution=[RES, RES])
        mask = (rast[..., 3:] > 0).float()
        pos_img, _ = dr.interpolate(v[None], rast, f)
        nrm_img, _ = dr.interpolate(vn[None], rast, f)
        # WORLD-space normals -- no w2c rotation. Upstream's normal maps are
        # world-space (see `hunyuan_normal_encode`).
        nrm_world = torch.nn.functional.normalize(nrm_img[0], dim=-1)
        # ...but oriented TOWARD the camera, matching Cycles. Cycles flips the
        # shading normal on a back-facing hit, so upstream's normal maps are
        # front-facing; nvdiffrast does no such flip, and on assets with
        # single-sided or open surfaces (very common in TexVerse) we would
        # otherwise emit normals pointing away from the camera. `c2w`'s +Z
        # column IS the world-space direction from the object toward the
        # camera (the camera looks down -Z_cam), and it is constant per view
        # because these six cameras are orthographic.
        dir_to_cam = torch.from_numpy(
            np.ascontiguousarray(cam["c2w"][:3, 2], dtype=np.float32)
        ).cuda()
        nrm_world = torch.where(
            (nrm_world @ dir_to_cam)[..., None] < 0, -nrm_world, nrm_world
        )
        outs.append(dict(
            mask=mask[0, ..., 0].cpu().numpy(),
            normal=nrm_world.cpu().numpy(),
            depth=rast[0, ..., 2].cpu().numpy(),
            pos_raw=pos_img[0].cpu().numpy(),
        ))
    return outs


# --- attribute pass (albedo/mr/emission from the atlas texel cloud) ----------

def build_texel_cloud(atlas_npz):
    """Load an atlas -- a path to an `atlas.npz`, or the loaded arrays themselves, both
    carrying the 8-key contract of
    `data_processing.glb2representations.atlas.bake_atlas` -- into a KD-tree of
    occupied-texel positions plus per-texel albedo/emission/metal/rough/alpha.
    The positions go through `canonical_to_hunyuan_frame`, the same Z-up
    [-0.5,0.5] frame as `render_geometry`'s `pos_raw`; a plain `* 0.5` would miss
    the axis swap and almost every nearest-neighbor lookup.

    Returns (tree, attrs) where attrs holds float32 arrays in [0,1], LINEAR
    (the atlas is baked linear, so these are written to PNG verbatim, no gamma).

    `alpha` (per-texel glTF opacity) is REQUIRED, with no fallback: the multiview
    baseline gets the same opacity condition as TEXGen and TRELLIS.2, and silently
    substituting ones would give data that trains fine but is not comparable.
    TEXGen's `spuv/data/mesh_uv.py` follows the same rule.
    """
    d = np.load(atlas_npz) if isinstance(atlas_npz, (str, os.PathLike)) else atlas_npz
    keys = getattr(d, "files", None) or list(d)
    if "alpha" not in keys:
        raise KeyError(
            f"atlas has no 'alpha' key (an older 7-key bake), cannot render the opacity "
            f"condition: {atlas_npz}. Re-run the bake (`glb2representations.run`) for this shape."
        )
    occ = d["occupancy"][..., 0].astype(bool)
    if not occ.any():
        # cKDTree on 0 points is legal but every query returns idx == n, which
        # then blows up as an opaque IndexError deep inside `shade_views`.
        # Fail here instead, naming the shape.
        raise ValueError(
            f"atlas has zero occupied texels, cannot shade any view: {atlas_npz}"
        )
    canonical_pos = decode_signed_uint16(d["position"])[occ]  # gltf Y-up [-1,1]
    pos = canonical_to_hunyuan_frame(canonical_pos)            # Z-up [-0.5,0.5]
    attrs = dict(
        albedo=d["color"][occ].astype(np.float32) / 255.0,
        emission=d["emission_color"][occ].astype(np.float32) / 255.0,
        metal=d["metal"][occ, 0].astype(np.float32) / 255.0,
        rough=d["rough"][occ, 0].astype(np.float32) / 255.0,
        alpha=d["alpha"][occ, 0].astype(np.float32) / 255.0,
    )
    return cKDTree(pos), attrs


def shade_views(geo_outs, tree, attrs):
    """Nearest-neighbor-shade each `render_geometry` view from the atlas texel
    cloud (`build_texel_cloud`). Returns a list (one dict per view, matching
    `geo_outs` order) with:
      albedo  (RES,RES,3) float32 in [0,1], linear, 0 outside mask
      emission(RES,RES,3) float32 in [0,1], linear, 0 outside mask
      alpha   (RES,RES,3) float32 in [0,1], the per-texel glTF opacity
              REPLICATED across all three channels. Stored as RGB rather than
              a 1-channel gray PNG purely for fixture uniformity: every other
              saved map is RGB, the loader's `.convert("RGB")` would expand a
              gray one anyway, and the completeness checks count PNGs per
              shape. Replicated channels cost almost nothing after zlib.
              NOTE this is *material* opacity (glass, decals, BLEND/MASK
              materials) -- NOT the object silhouette, which the renders
              already carry via the mask/BG_GRAY fill.
      mr      (RES,RES,3) float32: glTF ORM layout R=occlusion, G=roughness,
              B=metalness, as in upstream's `train_examples` `_mr.png` maps.
              Upstream's R is close to a constant 255 and our atlas has no
              occlusion channel, so R is a hard-coded 1.0.
      coverage_bad_frac: float, a coverage sanity number -- fraction of masked
              pixels whose nearest-texel distance (in Hunyuan-frame units)
              exceeds `COVERAGE_TOL` (2/512). >5% signals a likely frame
              mismatch between the atlas cloud and the rasterized mesh.
    """
    shaded = []
    for o in geo_outs:
        m = o["mask"] > 0.5
        if m.any():
            dist, idx = tree.query(o["pos_raw"][m], k=1, workers=-1)
        else:
            dist, idx = np.zeros(0), np.zeros(0, dtype=np.int64)

        def lookup(vals):
            out = np.zeros((RES, RES, vals.shape[1]), np.float32)
            out[m] = vals[idx]
            return out

        mr = np.zeros((RES, RES, 3), np.float32)
        mr[..., 0][m] = 1.0  # hard-coded occlusion=1 (see docstring above)
        mr[..., 1] = lookup(attrs["rough"][:, None])[..., 0]
        mr[..., 2] = lookup(attrs["metal"][:, None])[..., 0]
        shaded.append(dict(
            albedo=lookup(attrs["albedo"]),
            emission=lookup(attrs["emission"]),
            mr=mr,
            alpha=np.repeat(lookup(attrs["alpha"][:, None]), 3, axis=-1),
            coverage_bad_frac=float((dist > COVERAGE_TOL).mean()) if dist.size else 0.0,
        ))
    return shaded


def _to_png_u8(img01, mask):
    """float [0,1] (RES,RES,C) -> uint8, background (outside `mask`) forced to
    `BG_GRAY` on every channel -- so a channel that happens to zero-fill
    outside the mask (albedo/mr/emission all start as `np.zeros`) never
    leaks through as black; matches upstream's `bg_gray` training default on
    every saved map."""
    out = np.full(img01.shape, BG_GRAY, dtype=np.uint8)
    out[mask] = np.clip(img01[mask] * 255.0, 0, 255).round().astype(np.uint8)
    return out


def render_from_bundle(bundle, atlas, sha, out_dir, glctx, cameras):
    """Render+shade all `cameras` views for one already-loaded `bundle` and write the
    fixture layout DIRECTLY under `out_dir`: `render_tex/00N_{albedo,mr,normal,pos,
    emission,alpha}.png` + `render_tex/transforms.json` + `render_cond/000.png`.

    That is 36 PNGs per shape (6 maps x 6 views).

    `atlas` is a path to an `atlas.npz` or the loaded arrays (`build_texel_cloud`
    takes either). The bake worker (`run.process_one`) calls it with the geometry it
    already holds.

    Promotion is the CALLER's: `out_dir` is written in place, so the worker renders
    into the driver's staging directory.

    Returns a stats dict (per-view mask/coverage/emission numbers) for checks,
    not itself part of the output.
    """
    geo_outs = render_geometry(bundle, glctx, cameras)
    tree, attrs = build_texel_cloud(atlas)
    shaded = shade_views(geo_outs, tree, attrs)

    render_tex_dir = os.path.join(out_dir, "render_tex")
    render_cond_dir = os.path.join(out_dir, "render_cond")
    os.makedirs(render_tex_dir, exist_ok=True)
    os.makedirs(render_cond_dir, exist_ok=True)

    stats = {"sha": sha, "views": []}
    for cam, o, s in zip(cameras, geo_outs, shaded):
        stem = cam["file_stem"]
        m = o["mask"] > 0.5
        # pos/normal go through upstream's saved-map channel conventions HERE,
        # at save time only -- everything upstream of this line (pos_raw, the
        # KD-tree, the mask) stays in the plain Hunyuan Z-up frame.
        maps = dict(
            albedo=s["albedo"], mr=s["mr"], alpha=s["alpha"],
            normal=hunyuan_normal_encode(o["normal"]),
            pos=hunyuan_pos_encode(o["pos_raw"]),
            emission=s["emission"],
        )
        for name, img in maps.items():
            arr = np.flipud(_to_png_u8(img, m))  # ONE flip, right before save
            Image.fromarray(arr).save(
                os.path.join(render_tex_dir, f"{stem}_{name}.png"),
                compress_level=PNG_COMPRESS_LEVEL,
            )
        stats["views"].append(dict(
            stem=stem,
            mask_px=int(m.sum()),
            coverage_bad_frac=s["coverage_bad_frac"],
            emission_nonzero=bool(m.any() and (s["emission"][m] > 0).any()),
            # Fraction of the object's visible pixels that are not fully opaque.
            # 0.0 for the great majority of shapes (only literal BLEND/MASK
            # materials carry alpha), so surface it per view to catch a map
            # that is all ones.
            alpha_transparent_frac=float((s["alpha"][m][..., 0] < 0.99).mean()) if m.any() else 0.0,
        ))

    write_transforms_json(render_tex_dir)

    # render_cond/000.png = the _albedo image of the azimuth-0 frame -- found
    # by azimuth, not assumed to be a fixed index (that frame is
    # CAMERAS[1]/file_stem "001").
    front_stem = next(cam["file_stem"] for cam in cameras if cam["azimuth"] == 0)
    shutil.copyfile(
        os.path.join(render_tex_dir, f"{front_stem}_albedo.png"),
        os.path.join(render_cond_dir, "000.png"),
    )

    return stats

