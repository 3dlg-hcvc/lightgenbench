"""Multiview output -> the common point-sampled representation (Hunyuan3D-Emission).

The six emission views of one shape are back-projected onto its 512^2 UV atlas; the point-sampled
evaluator then reads that atlas like any atlas output (`atlas.py`, `--pred_format atlas`).

Per shape: take the atlas's occupied texels (position and object normal, from `atlas.npz`), map them
into the renderer's frame, and project them into each orthographic view. A view contributes a texel
when the texel lands inside the frame and the view mask, passes a depth test against the mesh
re-rasterized with `multiview.render_geometry` (the function that produced the views), and passes
the facing gate (`--facing_gate`, see FACING_GATES). The contributions are weighted by
|dot(texel normal, direction to camera)|. Every camera, projection and frame convention is imported
from `data_processing.glb2representations.multiview`, none re-derived. Predicted views and GT views
go through the same bridge, so the GT views give the per-shape ceiling:

  mode=pred     views at <views_root>/<sha>/seed<NN>  (--seed NN; the predictor's layout)
  mode=ceiling  views at <views_root>/<sha>           (the rendered GT views; render_tex/ inside)

Writes <out_dir>/<sha>.png (linear, uint8) + <sha>.npz, the layout the evaluator reads with
--pred_format atlas. Env: `lightgen`, with nvdiffrast installed (torch + CUDA for the depth re-raster).

    conda run -n lightgen python -m evaluation.converters.multiview \
        --mode pred --views_root <pred_root> --out_dir <dir> --seed 0 --blacken_bg \
        --dataset_root <released dataset> [--split test | --shas_file <uuids>] \
        --metadata data/texverse_index.parquet --glb_root data/TexVerse-1K
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_render_pipeline():
    """Import the multiview renderer's geometry/convention module (cameras included).

    Returns `multiview` from `data_processing.glb2representations`, with the repo root on
    `sys.path` so the import also works when this file is run as a script by
    path.

    Deliberately a function, not a top-level import: the renderer's GPU path pulls in
    torch + nvdiffrast + a CUDA context, and the pure-numpy helpers in this module
    (`project_texels`, `gather_from_views`, `bridge`) are unit-testable without any
    of it.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from data_processing.glb2representations import multiview  # noqa: E402

    return multiview


UV_RES = 512      # atlas resolution (the bake's)
VIEW_RES = 512    # a mirror of multiview.RES; `bridge` asserts they agree, so
                  # this constant can be used by GPU-free callers/tests without
                  # importing the renderer (and cannot silently drift from it)
# Base depth-test slack, in Hunyuan world units (the object spans 1.0 there), and
# SLOPE-SCALED at use (see `gather_from_views`). A texel is the atlas bake's own
# sample of the surface; the depth buffer holds the same surface interpolated at a
# pixel CENTER. The two therefore differ by (sub-pixel lateral offset) x (surface
# slope along the view axis) -- up to half an ortho pixel, 1.2/512/2 = 0.0012 world
# units, times 1/|cos|. 0.002 is that half-pixel with a little margin; the 1/|cos|
# is applied per texel rather than folded in as a constant, which is the standard
# slope-scaled depth bias and is what keeps a face-on thin plate's two sides from
# leaking into each other while still admitting grazing surfaces.
# On GT-ceiling shapes, slope-scaled 0.002 beats a flat 0.005 on ceiling IoU and on
# visible-texel IoU. A much looser flat eps (0.02) buys more ceiling coverage but a
# much larger value error -- it stops testing visibility and starts merging thin
# geometry.
DEFAULT_DEPTH_EPS = 0.002
# Hard ceiling on the slope-scaled tolerance, in the same units. Without it the
# scaling tops out at DEFAULT_DEPTH_EPS / DEFAULT_COS_MIN = 0.02 -- which is
# exactly the flat tolerance that merges the two faces of thin geometry. At
# grazing angles the slope term is meant to absorb a half-pixel of
# lateral offset, not to license seeing through a 2%-of-object-extent slab, so it
# is capped rather than left to the cos_min floor.
MAX_DEPTH_EPS = 0.01
# Cosine floor, applied to the facing gate and to the slope scaling above. A view
# whose surface normal is nearly perpendicular to the view direction sees the
# texel as a sub-pixel sliver, so its color is dominated by whatever neighboring
# surface shares the pixel. Such views are dropped rather than merely
# down-weighted. cos < 0.1 is ~84 degrees off-axis. It also floors the gather
# WEIGHT, so a texel that passes the gate always carries strictly positive weight
# -- otherwise a degenerate atlas normal could mark a texel `seen` while
# contributing nothing, which reads downstream as a confident zero.
DEFAULT_COS_MIN = 0.1

# Which cosine decides whether a view may contribute a texel. `abs_atlas` (the
# texel's own |cos|) is the default because it measured best on GT-ceiling shapes:
# higher ceiling IoU, visible-texel IoU and lower value error than `raster`.
#
# `raster` gates on `render_geometry`'s camera-flipped normal at the texel's
# pixel. That normal IS sign-reliable, which the atlas objnormal is not -- but
# that is also precisely why it cannot serve as a wrong-side defense: it faces the
# camera at EVERY rasterized pixel by construction, so it admits every back-face
# texel whose depth happens to pass. It is kept selectable because it is the
# strictest available statement about the surface the depth buffer actually holds.
# The real defense against seeing through thin geometry is MAX_DEPTH_EPS above.
FACING_GATES = ("abs_atlas", "raster", "both")
DEFAULT_FACING_GATE = "abs_atlas"


# ---------------------------------------------------------------- atlas side

def load_atlas_texels(atlas_npz):
    """Occupied-texel positions/normals/GT emission from an `atlas.npz`.

    Returns a dict with
      occ       (512,512) bool           occupancy mask
      rows/cols (K,) int                 texel coordinates of the occupied texels
      pos_h     (K,3) float64            position in the Hunyuan Z-up [-0.5,0.5] frame
      nrm_h     (K,3) float64            unit object normal in the same frame
      emission  (K,3) float64 in [0,1]   the baked GT emission (linear)
    """
    multiview = _import_render_pipeline()
    from data_processing.glb2representations.utils import decode_signed_uint16

    d = np.load(atlas_npz)
    occ = d["occupancy"][..., 0].astype(bool)
    if not occ.any():
        # Every downstream array would still be well-formed, and the bridge would
        # emit a silent all-zero atlas -- indistinguishable from "the model
        # predicted nothing". Fail here, naming the shape, the way
        # `multiview.build_texel_cloud` does for the same reason.
        raise ValueError(f"atlas has zero occupied texels, nothing to bridge onto: {atlas_npz}")
    rows, cols = np.nonzero(occ)
    pos_canonical = decode_signed_uint16(d["position"])[occ]      # glTF Y-up [-1,1]
    nrm_canonical = decode_signed_uint16(d["objnormal"])[occ]     # glTF Y-up unit-ish

    pos_h = multiview.canonical_to_hunyuan_frame(pos_canonical)
    # Same axis swap for the normal -- a rotation, so it applies to directions
    # unchanged; the *0.5 inside canonical_to_hunyuan_frame is an isotropic scale
    # and is normalized away here rather than skipped, so the two paths cannot
    # drift apart if that function ever changes.
    nrm_h = multiview.canonical_to_hunyuan_frame(nrm_canonical)
    norm = np.linalg.norm(nrm_h, axis=-1, keepdims=True)
    nrm_h = np.divide(nrm_h, norm, out=np.zeros_like(nrm_h), where=norm > 1e-8)

    return dict(
        occ=occ, rows=rows, cols=cols,
        pos_h=pos_h.astype(np.float64), nrm_h=nrm_h.astype(np.float64),
        emission=d["emission_color"][occ].astype(np.float64) / 255.0,
    )


# ---------------------------------------------------------------- view side

VIEW_NAME_PATTERNS = ("{stem}_emission.png", "{stem}.png", "{stem}_pred.png")


def resolve_views_dir(views_dir):
    """Accept either a fixture shape dir (which holds `render_tex/`) or a flat
    directory of six emission PNGs, so the SAME command bridges GT fixtures and
    prediction dirs."""
    views_dir = Path(views_dir)
    render_tex = views_dir / "render_tex"
    return render_tex if render_tex.is_dir() else views_dir


def find_view_png(views_dir, stem):
    for pattern in VIEW_NAME_PATTERNS:
        candidate = Path(views_dir) / pattern.format(stem=stem)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no emission PNG for view {stem} under {views_dir} "
        f"(tried {[p.format(stem=stem) for p in VIEW_NAME_PATTERNS]})"
    )


def _load_view_raw(path, res=None):
    """Saved emission PNG -> float [0,1] in nvdiffrast's RAW row order.

    ONE `np.flipud`, undoing the single flip `multiview.render_from_bundle` applies
    right before saving. Everything else in this module -- the depth buffer, the
    projected pixel coordinates -- is in raw row order, so this is the only place
    row order is touched.

    `res` is checked when given, because a size mismatch is the one failure here
    that can be SILENT: pixel coordinates are computed from the render resolution,
    so a view saved LARGER than expected (a prediction run at a different
    `view_size`, say) would be sampled only in its top-left corner and produce a
    plausible-looking, wrong atlas. A smaller one would merely IndexError.
    """
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    if res is not None and arr.shape[:2] != (res, res):
        raise ValueError(
            f"{path} is {arr.shape[1]}x{arr.shape[0]}, expected {res}x{res} -- the "
            f"bridge projects into a {res}^2 raster. Re-render (or re-predict) at "
            f"that resolution rather than resampling here."
        )
    return np.flipud(arr).copy()


# ------------------------------------------------------------- projection

def project_texels(pos_h, cam, proj, res=VIEW_RES):
    """Project Hunyuan-frame points into one orthographic view.

    Returns `(px, py, ndc_z)`: continuous raster coordinates (px = column, py =
    row, both in raw nvdiffrast row order where row index grows with NDC y) and
    the NDC depth used for the depth test. Pure numpy: `proj` is passed in (from `multiview._proj`) rather
    than imported here, so this function is testable without a GPU.
    """
    w2c = np.linalg.inv(cam["c2w"])
    mvp = proj.astype(np.float64) @ w2c
    h = np.concatenate([pos_h, np.ones((pos_h.shape[0], 1))], axis=1)
    clip = h @ mvp.T
    ndc = clip[:, :3] / clip[:, 3:4]   # ortho: clip.w == 1, kept general anyway
    px = (ndc[:, 0] * 0.5 + 0.5) * res
    py = (ndc[:, 1] * 0.5 + 0.5) * res
    return px, py, ndc[:, 2]


def _ndc_depth_eps(eps_world, near=0.01, far=10.0):
    """Depth epsilon in Hunyuan world units -> the NDC z units `render_geometry`'s
    depth buffer is in. Our orthographic projection maps a world-z span of
    (far-near) onto an NDC span of 2, so the factor is 2/(far-near). Defaults must
    match `multiview._proj`'s."""
    return eps_world * 2.0 / (far - near)


def gather_from_views(texels, views, cos_min=DEFAULT_COS_MIN,
                      depth_eps=DEFAULT_DEPTH_EPS, max_depth_eps=MAX_DEPTH_EPS,
                      facing_gate=DEFAULT_FACING_GATE, erode_px=0):
    """Cosine-weighted gather of per-view emission onto the texels.

    `views` is a list of dicts, one per camera in `multiview.CAMERAS` order:
        px, py   (K,) float   raster coordinates from `project_texels`
        ndc_z    (K,) float   texel depth
        depth    (R,R) float  the view's depth buffer (raw row order)
        mask     (R,R) float  the view's coverage mask (raw row order)
        normal   (R,R,3)      `render_geometry`'s world-space normal, ALREADY
                              flipped to face this camera (raw row order).
                              Optional -- only the "raster"/"both" facing gates
                              read it.
        emission (R,R,3)      the view's emission image (raw row order, [0,1])
        dir_to_cam (3,)       world-space direction from the object to the camera
    `depth_eps` is in Hunyuan world units, slope-scaled and capped at
    `max_depth_eps`, and converted to NDC here. `facing_gate` selects which cosine
    decides membership -- see FACING_GATES for the measured comparison.
    Returns `(emission (K,3), n_views (K,) int, weight_sum (K,))`.

    Gate and WEIGHT are separate concerns and are kept separate. The weight is
    always `|cos_atlas|`, the texel's OWN normal, floored at `cos_min`: it answers
    "how much of this view's resolution does this texel get", which is a property
    of the texel, not of whatever else shares its pixel. The floor guarantees a
    gated-in texel carries positive weight, so `n_views > 0` can never coexist
    with `weight_sum == 0` (that combination would surface downstream as a
    confident zero rather than as missing data).

    Pure numpy -- the GPU work (producing `depth`/`mask`/`normal`) happens in
    `bridge`.
    """
    if facing_gate not in FACING_GATES:
        raise ValueError(f"facing_gate must be one of {FACING_GATES}, got {facing_gate!r}")
    K = texels["pos_h"].shape[0]
    acc = np.zeros((K, 3))
    wsum = np.zeros(K)
    nviews = np.zeros(K, dtype=np.int32)

    for v in views:
        col = np.floor(v["px"]).astype(np.int64)
        row = np.floor(v["py"]).astype(np.int64)
        res = v["depth"].shape[0]
        inside = (col >= 0) & (col < res) & (row >= 0) & (row < res)
        # Clamp before indexing so the gather is legal for every texel; `inside`
        # is what actually decides membership.
        c = np.clip(col, 0, res - 1)
        r = np.clip(row, 0, res - 1)

        vmask = v["mask"]
        if erode_px > 0:
            from scipy.ndimage import binary_erosion
            vmask = binary_erosion(vmask > 0.5, iterations=int(erode_px)).astype(vmask.dtype)
        covered = vmask[r, c] > 0.5

        # Two cosines, one to gate and one to weight. Which one GATES: see FACING_GATES.
        cos_atlas = np.abs(texels["nrm_h"] @ v["dir_to_cam"])   # the texel's own
        if facing_gate == "abs_atlas":
            gate_cos = cos_atlas
        else:
            if v.get("normal") is None:
                raise ValueError(
                    f"facing_gate={facing_gate!r} needs the rasterized normal, but this "
                    f"view carries none. `bridge` always supplies it; a hand-built view "
                    f"dict must too (or use facing_gate='abs_atlas')."
                )
            # `render_geometry` already flipped this toward the camera, so it is
            # sign-reliable -- but that is also why it cannot detect a back-face
            # texel: at ANY pixel it faces the camera by construction.
            cos_raster = v["normal"][r, c] @ v["dir_to_cam"]
            gate_cos = cos_raster if facing_gate == "raster" else np.minimum(cos_raster, cos_atlas)
        facing = gate_cos > cos_min

        # Depth test: our orthographic NDC z grows with distance from the camera,
        # so the visible surface is the SMALLEST z. A texel is visible when it is
        # no further than the rasterized surface, within eps. The tolerance is
        # slope-scaled (the gate cosine's own slope), floored by cos_min and capped
        # by max_depth_eps -- see DEFAULT_DEPTH_EPS and MAX_DEPTH_EPS.
        slope_eps = np.minimum(depth_eps / np.maximum(np.abs(gate_cos), cos_min),
                               max_depth_eps)
        in_front = v["ndc_z"] <= v["depth"][r, c] + _ndc_depth_eps(slope_eps)

        vis = inside & covered & in_front & facing
        if not vis.any():
            continue
        w = np.where(vis, np.maximum(cos_atlas, cos_min), 0.0)
        acc += w[:, None] * v["emission"][r, c]
        wsum += w
        nviews += vis.astype(np.int32)

    out = np.zeros((K, 3))
    hit = wsum > 0
    out[hit] = acc[hit] / wsum[hit, None]
    return out, nviews, wsum


# ------------------------------------------------------------------ driver

def _uncomposite_bg(emis, mask, blacken_bg, bg_srgb):
    """Remove the gray view background Hunyuan renders on. Its emission views put the
    object (emission over BLACK) on a uniform gray bg; anti-aliased silhouette pixels are
    then `alpha*E + (1-alpha)*gray`, which the hard geometry mask cannot reject and which
    the view->UV bridge otherwise paints on as spurious rim emission. The rasterized mask
    is hard (0/1), but the emission image's edge is soft, so we recover a soft coverage by
    blurring the hard mask ~1px and subtract the background it implies:
        E_true = clip(observed - (1 - coverage) * gray, 0)
    At a silhouette pixel (coverage~0.5, object E=0) this lands exactly on 0; interior
    (coverage=1) is untouched."""
    if not blacken_bg:
        return emis
    import numpy as _np
    from scipy.ndimage import gaussian_filter
    m = _np.asarray(mask, _np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    cov = gaussian_filter(m, sigma=0.8)                       # soft coverage matching the emission AA
    bg = _np.float32(bg_srgb)  # emission views are raw bytes in [0,1]; subtract the gray directly
    return _np.clip(emis - (1.0 - cov[..., None]) * bg, 0.0, None)


def bridge(views_dir, atlas_npz, glb_path, glctx=None,
           depth_eps=DEFAULT_DEPTH_EPS, cos_min=DEFAULT_COS_MIN,
           max_depth_eps=MAX_DEPTH_EPS, facing_gate=DEFAULT_FACING_GATE, erode_px=0, blacken_bg=False, bg_srgb=None):
    """Bridge one shape's six emission views back onto its UV atlas.

    Returns a dict:
      uv        (512,512,3) float64 in [0,1]  bridged emission, 0 where unseen
      gt_uv     (512,512,3) float64 in [0,1]  the atlas's own emission (the target)
      occ       (512,512)   bool              atlas occupancy
      seen      (512,512)   bool              texels visible in >=1 view
      n_views   (512,512)   int32             how many views saw each texel
      coverage  float                         seen.sum() / occ.sum()

    `glb_path` is needed because visibility is decided by RE-RASTERIZING the mesh
    with the same `multiview.render_geometry` that produced the views -- a
    depth buffer read out of the saved 8-bit `_pos` maps would be quantized to
    ~1/256 of the object extent, which is coarser than the depth test's own
    tolerance.
    """
    multiview = _import_render_pipeline()
    from data_processing.glb2representations.utils import load_and_merge
    if bg_srgb is None:
        # the renderer's own fill (BG_GRAY = 127, Hunyuan's training background)
        bg_srgb = multiview.BG_GRAY / 255.0

    assert multiview.RES == VIEW_RES, (
        f"render resolution changed under this bridge: multiview.RES="
        f"{multiview.RES} vs VIEW_RES={VIEW_RES}"
    )
    if glctx is None:
        import nvdiffrast.torch as dr
        glctx = dr.RasterizeCudaContext()

    cameras = multiview.CAMERAS
    texels = load_atlas_texels(atlas_npz)
    bundle = load_and_merge(glb_path)
    geo = multiview.render_geometry(bundle, glctx, cameras)

    proj = multiview._proj()
    src_dir = resolve_views_dir(views_dir)

    views = []
    for cam, g in zip(cameras, geo):
        px, py, ndc_z = project_texels(texels["pos_h"], cam, proj, res=multiview.RES)
        views.append(dict(
            px=px, py=py, ndc_z=ndc_z,
            depth=g["depth"], mask=g["mask"], normal=g["normal"],
            emission=_uncomposite_bg(
                        _load_view_raw(find_view_png(src_dir, cam["file_stem"]), res=multiview.RES),
                        g["mask"], blacken_bg, bg_srgb),
            # c2w's +Z column IS the world-space direction object -> camera for
            # these orthographic views (same fact render_geometry uses to orient
            # its normals); constant per view because the projection is ortho.
            dir_to_cam=np.asarray(cam["c2w"][:3, 2], dtype=np.float64),
        ))

    values, n_views, _ = gather_from_views(texels, views, cos_min=cos_min,
                                           depth_eps=depth_eps,
                                           max_depth_eps=max_depth_eps,
                                           facing_gate=facing_gate, erode_px=erode_px)

    uv = np.zeros((UV_RES, UV_RES, 3))
    nv = np.zeros((UV_RES, UV_RES), dtype=np.int32)
    gt_uv = np.zeros((UV_RES, UV_RES, 3))
    uv[texels["rows"], texels["cols"]] = values
    nv[texels["rows"], texels["cols"]] = n_views
    gt_uv[texels["rows"], texels["cols"]] = texels["emission"]

    occ = texels["occ"]
    seen = nv > 0
    return dict(
        uv=uv, gt_uv=gt_uv, occ=occ, seen=seen, n_views=nv,
        coverage=float(seen.sum()) / float(max(occ.sum(), 1)),
    )


def write_outputs(result, out_prefix):
    """Write `<out_prefix>.png` (uint8, linear, background 0) and `<out_prefix>.npz`.

    PNG is uint8 for the same reason the atlas is: it is what
    the evaluator's atlas reader and the UV viewers read. The npz keeps the float map,
    the per-texel view count and the masks, so nothing downstream has to
    re-derive coverage from a quantized image.
    """
    out_prefix = str(out_prefix)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    png = np.clip(result["uv"] * 255.0, 0, 255).round().astype(np.uint8)
    Image.fromarray(png).save(out_prefix + ".png")
    np.savez_compressed(
        out_prefix + ".npz",
        emission=result["uv"].astype(np.float32),
        n_views=result["n_views"],
        occupancy=result["occ"],
        seen=result["seen"],
    )
    return out_prefix + ".png", out_prefix + ".npz"


def main():
    ap = argparse.ArgumentParser(description="Back-project multiview emission onto UV atlases, in one process.")
    ap.add_argument("--mode", choices=["pred", "ceiling"], required=True)
    ap.add_argument("--views_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_root", required=True,
                    help="the released dataset: splits.json and <uuid>/atlas.npz")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--shas_file", default=None, help="bridge these uuids (one per line) instead of the split")
    ap.add_argument("--metadata", required=True, help="the GLB index (data/texverse_index.parquet): glb_1k_path per uuid")
    ap.add_argument("--glb_root", required=True, help="TexVerse-1K root")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--blacken_bg", action="store_true",
                    help="un-composite Hunyuan's grey view background before projecting (removes the "
                         "spurious grey-bg silhouette bleed; keeps the object's own emission)")
    ap.add_argument("--erode_px", type=int, default=0,
                    help="erode each view's coverage mask by N px before gather (extra silhouette trim)")
    ap.add_argument("--seed", type=int, default=0,
                    help="which seed subdir under <views_root>/<sha>/ to bridge. pred mode "
                         "only -- GT views have no seed subdir.")
    a = ap.parse_args()

    if a.mode == "ceiling" and a.seed != 0:
        ap.error("--seed is meaningless in ceiling mode: GT views have no seed subdir")

    import nvdiffrast.torch as dr
    import pandas as pd

    from utils.glb_io import resolve_glb

    if a.shas_file:
        shas = [ln.strip() for ln in open(a.shas_file) if ln.strip()]
    else:
        with open(os.path.join(a.dataset_root, "splits.json")) as f:
            shas = json.load(f)[a.split]
    if a.limit:
        shas = shas[:a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    # A per-seed out_dir is MANDATORY. The cache short-circuit below keys only on
    # <out_dir>/<sha>.png|.npz, so a reused directory would silently report every later
    # seed as `cached` and score every seed identically. Stamp the seed and refuse a mismatch.
    stamp = os.path.join(a.out_dir, ".bridge_seed")
    want = f"{a.mode}:{a.seed}"
    if os.path.exists(stamp):
        have = open(stamp).read().strip()
        if have != want:
            raise SystemExit(
                f"ABORT: {a.out_dir} already holds bridged output for '{have}', not '{want}'. "
                f"Give each seed its own --out_dir.")
    else:
        with open(stamp, "w") as f:
            f.write(want)

    meta = pd.read_parquet(a.metadata, columns=["glb_1k_path"])   # once, not per shape
    glctx = dr.RasterizeCudaContext()                             # once, not per shape
    print(f"[bridge] mode={a.mode} seed={a.seed} shapes={len(shas)} out={a.out_dir}", flush=True)

    n_ok = n_skip = n_fail = n_cached = 0
    t0 = time.time()
    for i, sha in enumerate(shas, 1):
        out_prefix = os.path.join(a.out_dir, sha)
        if os.path.exists(out_prefix + ".png") and os.path.exists(out_prefix + ".npz"):
            n_cached += 1
            continue
        views = (os.path.join(a.views_root, sha, f"seed{a.seed:02d}") if a.mode == "pred"
                 else os.path.join(a.views_root, sha))
        atlas = os.path.join(a.dataset_root, sha, "atlas.npz")
        if not os.path.isdir(views) or not os.path.exists(atlas):
            print(f"  SKIP {sha} (views={os.path.isdir(views)} atlas={os.path.exists(atlas)})",
                  flush=True)
            n_skip += 1
            continue
        try:
            glb = resolve_glb(sha, a.glb_root, meta)
            result = bridge(views, atlas, glb, glctx=glctx,
                            blacken_bg=a.blacken_bg, erode_px=a.erode_px)
            write_outputs(result, out_prefix)
            n_ok += 1
        except Exception as e:                                     # keep going, record the failure
            print(f"  FAIL {sha}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1
        if i % 50 == 0:
            rate = (time.time() - t0) / max(n_ok, 1)
            print(f"  [{i}/{len(shas)}] ok={n_ok} fail={n_fail} skip={n_skip} "
                  f"({rate:.2f}s/shape)", flush=True)

    n_png = len([f for f in os.listdir(a.out_dir) if f.endswith(".png")])
    print(f"BRIDGE_DONE mode={a.mode} ok={n_ok} cached={n_cached} skip={n_skip} fail={n_fail} "
          f"png_total={n_png} elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    # `main` imports repo modules (`utils.glb_io`), so the repo root goes on
    # sys.path first; the renderer itself is imported lazily inside `bridge`, after the
    # arguments are parsed.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
