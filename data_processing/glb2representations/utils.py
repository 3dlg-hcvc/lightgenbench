"""What the three representations share: the frames and encodings they are written in,
the one load and normalize of the GLB they all start from, and the alignment gate.

One trimesh load -> merge multi-part -> normalize once gives the canonical
`[-1, 1]` geometry from which the UV atlas (`atlas.py`, xatlas + nvdiffrast), the
o-voxels (`voxels.py`) and the multiview renderings (`multiview.py`) all derive, so
they align by construction. No Blender.

The atlas is written with the key/dtype/value encoding the TEXGen
`spuv/data/mesh_uv.py` loader reads, so it consumes it unchanged: position uint16 encodes
canonical [-1,1] via (p*0.5+0.5)*65535; objnormal likewise; color/metal/rough/
emission_color are uint8 in [0,1]*255.
"""
from dataclasses import dataclass, field

import numpy as np
import trimesh

from evaluation.converters.voxels import (
    compute_voxel_transform,
    apply_voxel_transform,
    occupancy_overlap,
)

# --- Canonical + voxel + atlas frame constants -----------------------------
CANONICAL_SCALE_NUM = 1.9999   # canonical normalize: scale = NUM / max_extent -> [-0.9999, 0.9999]
TIGHT_SCALE_NUM = 0.99999      # o-voxel tight normalize: scale = NUM / max_extent -> [-0.5, 0.5]
GRID = 256                     # o-voxel grid resolution
VOXEL_HALF = 0.5               # voxel frame is [-VOXEL_HALF, VOXEL_HALF]
ATLAS_RES = 512                # UV atlas resolution (HxW)
CAP_VOXELS = 1.5               # nearest-voxel lookup cap (voxels)
FP_GATE = 0.99                 # alignment pass threshold


# --- the atlas encode/decode (position/objnormal uint16; others uint8) --
def float01_to_uint16(x: np.ndarray) -> np.ndarray:
    """[0,1] float -> uint16, the atlas encoding."""
    return np.clip(np.asarray(x) * 65535.0, 0, 65535).round().astype(np.uint16)


def float01_to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x) * 255.0, 0, 255).round().astype(np.uint8)


def encode_signed_uint16(x: np.ndarray) -> np.ndarray:
    """[-1,1] float -> uint16 via (x*0.5+0.5). Used for `position` and `objnormal`
    (TEXGen decodes position as *2, objnormal as [-1,1])."""
    return float01_to_uint16(np.asarray(x) * 0.5 + 0.5)


def decode_signed_uint16(x: np.ndarray) -> np.ndarray:
    """uint16 -> [-1,1] float (inverse of encode_signed_uint16)."""
    return np.asarray(x).astype(np.float32) / 65535.0 * 2.0 - 1.0


@dataclass
class MeshBundle:
    """One canonical-frame mesh, materials preserved for both outputs.

    vertices/faces are the merged whole-mesh geometry in canonical `[-1,1]`.
    face_uv holds the ORIGINAL per-corner UVs (for sampling source textures during
    the atlas bake); face_has_uv is True only when that part actually had source
    UVs (do NOT treat all-zero face_uv as "has UV" — zeros are the no-UV sentinel).
    face_mat indexes `geometries` (each a trimesh.Trimesh whose `.visual.material`
    is the source PBR material). center/scale record the single canonical
    normalization applied.
    """
    vertices: np.ndarray            # (N, 3) float64, canonical [-1, 1]
    faces: np.ndarray               # (M, 3) int64
    face_uv: np.ndarray             # (M, 3, 2) float32 original UVs (0 if none)
    face_has_uv: np.ndarray         # (M,) bool — True iff part had source UVs
    face_mat: np.ndarray            # (M,) int64 -> index into geometries
    geometries: list                # list[trimesh.Trimesh], canonical frame
    center: np.ndarray              # (3,) canonical normalize center
    scale: float                    # canonical normalize scale

    def scene(self):
        return trimesh.Scene(list(self.geometries))


@dataclass
class AtlasResult:
    sha: str
    atlas_path: str = ""
    occupied_texels: int = 0
    timings: dict = field(default_factory=dict)


@dataclass
class VoxelResult:
    sha: str
    num_voxels: int
    coords_path: str
    emission_vxz: str
    pbr_vxz: str


# --- one trimesh load + multi-part merge + single canonical normalize --------
#
# Loads a .glb, bakes the scene-graph transforms, merges every part into one mesh
# while PRESERVING per-face material provenance (an index into the kept
# `geometries` list) and each part's original per-corner UVs, then normalizes the
# whole thing ONCE into the canonical `[-1, 1]` frame. Everything downstream (UV
# atlas + o-voxel) derives from this single geometry, so they align by construction.
#
# We deliberately do NOT use `Scene.to_geometry()` -- it collapses per-part
# materials into a single baked visual. Instead we keep the per-part geometries
# (each carrying its own `.visual.material`) as the material table.

def _as_texture_visuals(geom):
    """Ensure geom.visual is a TextureVisuals with a PBRMaterial (o_voxel asserts
    this, and the atlas bake reads factors/textures off `.material`). Convert a
    ColorVisuals (vertex/face color) into a constant-baseColor PBR material."""
    vis = geom.visual
    if isinstance(vis, trimesh.visual.TextureVisuals):
        if vis.material is None:
            vis.material = trimesh.visual.material.PBRMaterial()
        return geom
    # ColorVisuals (or bare) -> synthesize a PBRMaterial from the mean color.
    base = np.array([200, 200, 200, 255], dtype=np.uint8)
    try:
        cols = np.asarray(vis.main_color, dtype=np.uint8).reshape(-1)
        if cols.size >= 3:
            base[: min(4, cols.size)] = cols[: min(4, cols.size)]
    except Exception:
        pass
    mat = trimesh.visual.material.PBRMaterial(baseColorFactor=base)
    geom.visual = trimesh.visual.TextureVisuals(uv=None, material=mat)
    return geom


def _face_uv(geom):
    """Return (face_uv, has_uv) for one part.

    face_uv is (F,3,2) original UVs per face-corner (zeros if none).
    has_uv is a scalar bool: True only when the part actually carried per-vertex
    source UVs. Atlas baking must use this flag — not "are the UVs nonzero" —
    so constant-factor / no-texture parts fall back to material factors instead
    of sampling a texture at arbitrary (0,0) coordinates.
    """
    n = len(geom.faces)
    uv = getattr(geom.visual, "uv", None)
    if uv is None or len(uv) != len(geom.vertices):
        return np.zeros((n, 3, 2), dtype=np.float32), False
    return np.asarray(uv, dtype=np.float32)[geom.faces], True


def load_and_merge(glb_path: str) -> MeshBundle:
    """Load a .glb and merge+normalize it into one canonical MeshBundle."""
    return merge_and_normalize(trimesh.load(glb_path, process=False))


def merge_and_normalize(loaded) -> MeshBundle:
    """Merge a loaded trimesh Scene/Trimesh into one canonical MeshBundle.

    Split out from `load_and_merge` so the merge+normalize logic can be tested
    on a synthetic in-memory scene without a .glb on disk.
    """
    if isinstance(loaded, trimesh.Scene):
        # concatenate=False bakes each node's world transform into a copy of its
        # geometry but keeps the parts (and their individual materials) separate.
        parts = loaded.dump(concatenate=False)
    else:
        parts = [loaded]
    parts = [p for p in parts if isinstance(p, trimesh.Trimesh)
             and len(p.vertices) and len(p.faces)]
    if not parts:
        raise ValueError("no mesh geometry in loaded input")
    parts = [_as_texture_visuals(p) for p in parts]

    # --- single canonical normalize over the WHOLE merged mesh ---
    allv = np.concatenate([np.asarray(p.vertices, dtype=np.float64) for p in parts], axis=0)
    center = (allv.min(0) + allv.max(0)) / 2.0
    ext = float((allv.max(0) - allv.min(0)).max())
    scale = CANONICAL_SCALE_NUM / ext if ext > 0 else 1.0
    for p in parts:
        p.vertices = (np.asarray(p.vertices, dtype=np.float64) - center) * scale

    # --- concatenate geometry, tracking per-face material provenance ---
    verts, faces, face_uv, face_has_uv, face_mat = [], [], [], [], []
    voff = 0
    for mid, p in enumerate(parts):
        v = np.asarray(p.vertices, dtype=np.float64)
        f = np.asarray(p.faces, dtype=np.int64) + voff
        uv, has_uv = _face_uv(p)
        verts.append(v)
        faces.append(f)
        face_uv.append(uv)
        face_has_uv.append(np.full(len(p.faces), has_uv, dtype=bool))
        face_mat.append(np.full(len(p.faces), mid, dtype=np.int64))
        voff += len(v)

    return MeshBundle(
        vertices=np.concatenate(verts, axis=0),
        faces=np.concatenate(faces, axis=0),
        face_uv=np.concatenate(face_uv, axis=0),
        face_has_uv=np.concatenate(face_has_uv, axis=0),
        face_mat=np.concatenate(face_mat, axis=0),
        geometries=parts,
        center=np.asarray(center, dtype=np.float64),
        scale=float(scale),
    )


# --- alignment validation -----------------------------------------------------
#
# Reuses the evaluator's voxel helpers. Consumes coords.npz (`voxels.py`) and
# atlas.npz position/occupancy (`atlas.py`). Never decodes a .vxz (that needs
# o_voxel / GPU).
#
# PRIMARY gate = fp_voxel (canonical mesh surface <-> voxel coords). fp_atlas is
# a diagnostic: occupied atlas positions pushed through the SAME mesh-frame
# center/scale and overlapped against the same voxel coords.

@dataclass
class ValidationResult:
    sha: str
    fp_voxel: float
    fp_atlas: float
    passed: bool


def _squeeze_occ(occ: np.ndarray) -> np.ndarray:
    """Accept occupancy as [H,W] or [H,W,1] (both occur)."""
    occ = np.asarray(occ).astype(bool)
    if occ.ndim == 3 and occ.shape[-1] == 1:
        occ = occ[..., 0]
    return occ


def atlas_occupied_positions(atlas_npz_path: str) -> np.ndarray:
    """Decode occupied atlas `position` texels to canonical [-1,1] points [N,3]."""
    s = np.load(atlas_npz_path)
    pos = s["position"]
    occ = _squeeze_occ(s["occupancy"])
    if pos.ndim != 3 or pos.shape[-1] != 3 or occ.shape != pos.shape[:2]:
        raise ValueError(
            f"atlas shape mismatch: position={pos.shape}, "
            f"occupancy={s['occupancy'].shape}"
        )
    # uint16 atlas encode; a float already in [-1,1] is accepted as is.
    if np.issubdtype(pos.dtype, np.integer):
        p = decode_signed_uint16(pos).astype(np.float64)
    else:
        p = pos.astype(np.float64)
    return p[occ]


def atlas_to_coords(
    atlas_npz_path: str,
    center=None,
    scale=None,
) -> np.ndarray:
    """Atlas position map -> unique integer voxel coords [M,3] in [0, GRID).

    Pass the mesh-frame (center, scale) from compute_voxel_transform so atlas points
    share the voxelize frame; omitting them falls back to the atlas points' own tight
    bbox.
    """
    p = atlas_occupied_positions(atlas_npz_path)
    if center is None or scale is None:
        center = (p.min(0) + p.max(0)) / 2.0
        ext = float((p.max(0) - p.min(0)).max())
        scale = 0.99999 / ext if ext > 0 else 1.0
    pvf = (p - center) * scale
    coords = np.floor((pvf + 0.5) * GRID).astype(int)
    coords = np.clip(coords, 0, GRID - 1)
    return np.unique(coords, axis=0)


def _sample_mesh_pvf(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int = 50000,
    seed: int = 0,
):
    """Surface-sample a mesh and map into the voxel frame.

    Returns (pvf, center, scale). UV-optional — alignment only needs 3D points.
    """
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    points, _face_index = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    points = np.asarray(points, dtype=np.float64)
    center, scale = compute_voxel_transform(mesh.vertices)
    pvf = apply_voxel_transform(points, center, scale)
    return pvf, center, scale


def alignment_fp(
    vertices: np.ndarray,
    faces: np.ndarray,
    coords: np.ndarray,
    n_points: int = 50000,
    seed: int = 0,
) -> float:
    """fp = fraction of mesh surface samples within CAP_VOXELS of a voxel center."""
    pvf, _center, _scale = _sample_mesh_pvf(vertices, faces, n_points=n_points, seed=seed)
    fp, _fv = occupancy_overlap(
        pvf, np.asarray(coords, dtype=int), grid=GRID, cap_voxels=CAP_VOXELS
    )
    return float(fp)


def validate_shape(
    out_dir: str,
    sha: str,
    coords_npz: str,
    atlas_npz: str,
    bundle: MeshBundle,
    n_points: int = 50000,
    seed: int = 0,
) -> ValidationResult:
    """Compute fp_voxel (gate) and fp_atlas (diagnostic) for one shape.

    `out_dir` is unused; paths are taken from the explicit npz arguments. `bundle`
    supplies the canonical mesh (this stage does not write a canonical.glb).
    """
    del out_dir  # reserved for driver layout; paths are explicit
    voxel_coords = np.load(coords_npz)["coords"]
    pvf, center, scale = _sample_mesh_pvf(
        bundle.vertices, bundle.faces, n_points=n_points, seed=seed
    )
    fp_voxel, _fv = occupancy_overlap(
        pvf, np.asarray(voxel_coords, dtype=int), grid=GRID, cap_voxels=CAP_VOXELS
    )

    # Diagnostic: atlas occupied positions in the SAME mesh frame vs voxel coords.
    atlas_pts = atlas_occupied_positions(atlas_npz)
    atlas_pvf = apply_voxel_transform(atlas_pts, center, scale)
    fp_atlas, _fv_a = occupancy_overlap(
        atlas_pvf, np.asarray(voxel_coords, dtype=int), grid=GRID, cap_voxels=CAP_VOXELS
    )

    passed = float(fp_voxel) >= FP_GATE
    return ValidationResult(
        sha=sha,
        fp_voxel=float(fp_voxel),
        fp_atlas=float(fp_atlas),
        passed=passed,
    )
