"""Where each review image is: uuid + kind -> the file on disk.

The review images are the eight images a shape is judged on (paper §4, VLM filtering): the
three card stills, the dark-box pair, the multiview and emit-only orbits, and the bake's atlas.
The annotation tool's card and its "more" overlay (`data_processing/tools/human_annotation/`)
and the VLM check's bundle (`data_processing/vlm_filtering/vlm_check.py`) read exactly these
files; the render lanes of `data_processing/render/` (`single_view.py`, `orbit.py`) write new
ones where `new_still`, `new_orbit_view` and `new_webp` say. Every consumer asks here rather
than spelling a root itself, so the two-root lookup is in one place and a shape that already
has an image is never re-rendered.

Standard library and `utils.paths` only, and 3.8-clean: the annotation tool (Python 3.8)
imports this module.

Every image kind has TWO roots. The legacy root is where earlier renders already sit, and
is READ ONLY: this stage never writes there. The new root is where every render this stage
produces lands. The lookups below check legacy first.

The roots are read off this module inside each call, never snapshotted at import: a
one-off run can repoint them on this module (`utils.review_images`).
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.paths import DATA_ROOT, REPRESENTATIONS_ROOT

#: The bake's tree, one directory per shape: the eighth image, the atlas, is read
#: from it rather than rendered.
ATLAS_ROOT = REPRESENTATIONS_ROOT

# --- legacy roots (read only) ------------------------------------------------------
_LEGACY_ROOT = DATA_ROOT / "review_images_legacy"
#: The three card stills, sharded `<shard>/<uuid>/render.png`.
LEGACY_STILL_ROOTS = {
    "glb": _LEGACY_ROOT / "glb_renders",
    "emit": _LEGACY_ROOT / "lightgen/renders_dc80k/emit_only",
    "brighter": _LEGACY_ROOT / "lightgen/renders_dc80k/emissive_brighter",
}
#: Orbit stage 1, `<preset>/<view>/<shard>/<uuid>/render.png`.
LEGACY_ORBIT_ROOT = _LEGACY_ROOT / "lightgen/renders_dc80k_multiview"
#: Orbit stage 2, `<preset>/<uuid>.webp`, flat by uuid.
LEGACY_WEBP_ROOT = _LEGACY_ROOT / "lightgen/multiview_webp"

# --- new roots (the render lanes write here) ----------------------------------------
REVIEW_ROOT = DATA_ROOT / "review_images"
NEW_STILL_ROOTS = {
    "glb": REVIEW_ROOT / "glb",
    "emit": REVIEW_ROOT / "emit_only",
    "brighter": REVIEW_ROOT / "emissive_brighter",
}
NEW_ORBIT_ROOT = REVIEW_ROOT / "multiview"
NEW_WEBP_ROOT = REVIEW_ROOT / "webp"
#: The dark-box pair, flat by uuid.
BOX_ROOT = DATA_ROOT / "box_renders"

# --- a one-off run (`data_processing/vlm_filtering/run.sh --vlm`) -----------------------------
#: `LIGHTGEN_REVIEW_ROOT=<dir>` makes `<dir>` the only root: every kind is read from and
#: written to it, in `REVIEW_ROOT`'s layout (the dark-box pair in `<dir>/box_renders`), and
#: neither default root is read. `LIGHTGEN_ATLAS_ROOT=<tree>` names the bake tree the
#: atlas is read from. Unset, the default, nothing above changes.
if os.environ.get("LIGHTGEN_REVIEW_ROOT"):
    REVIEW_ROOT = Path(os.environ["LIGHTGEN_REVIEW_ROOT"])
    NEW_STILL_ROOTS = {
        "glb": REVIEW_ROOT / "glb",
        "emit": REVIEW_ROOT / "emit_only",
        "brighter": REVIEW_ROOT / "emissive_brighter",
    }
    NEW_ORBIT_ROOT = REVIEW_ROOT / "multiview"
    NEW_WEBP_ROOT = REVIEW_ROOT / "webp"
    BOX_ROOT = REVIEW_ROOT / "box_renders"
    LEGACY_STILL_ROOTS = dict(NEW_STILL_ROOTS)
    LEGACY_ORBIT_ROOT, LEGACY_WEBP_ROOT = NEW_ORBIT_ROOT, NEW_WEBP_ROOT
if os.environ.get("LIGHTGEN_ATLAS_ROOT"):
    ATLAS_ROOT = Path(os.environ["LIGHTGEN_ATLAS_ROOT"])

# --- the image kinds a lookup takes ----------------------------------------------------
STILL_KINDS = ("glb", "emit", "brighter")
ORBIT_PRESETS = ("standard", "emit_only")


# --- the lookups: legacy root first, the new root second --------------------------------
def _first_existing(*candidates):
    for c in candidates:
        c = Path(c)
        try:
            # Size, not existence: a failed write on an NFS share can leave a 0-byte file.
            # Counting that as a render is the worst outcome available -- the coverage
            # scan calls the shape done so it is never re-rendered, and because legacy is
            # checked first the empty file also shadows any good new render. Nothing
            # resolved here (PNG, webp, npz) is ever legitimately empty.
            if c.stat().st_size > 0:
                return c
        except OSError:
            continue
    return None


# An explicit `roots`/`root` replaces the configured roots outright, so a one-off run
# never reaches them.
# `is not None`, never truthiness: an empty `roots` means "look nowhere", and falling back
# to the default legacy root there would read a tree the caller deliberately excluded.
def _still_roots(kind, roots=None):
    if roots is not None:
        return tuple(Path(r) for r in roots)
    return (LEGACY_STILL_ROOTS[kind], NEW_STILL_ROOTS[kind])


def still(kind, shard, uuid, roots=None):
    """The card still of `kind` for `uuid`, or None. Layout `<shard>/<uuid>/render.png`."""
    return _first_existing(*[r / shard / uuid / "render.png"
                             for r in _still_roots(kind, roots)])


def orbit_view(preset, view, shard, uuid, roots=None):
    """One of the six orbit views, or None. `<preset>/<view>/<shard>/<uuid>/render.png`."""
    rs = tuple(Path(r) for r in
               (roots if roots is not None else (LEGACY_ORBIT_ROOT, NEW_ORBIT_ROOT)))
    return _first_existing(*[r / preset / view / shard / uuid / "render.png" for r in rs])


def webp(preset, uuid, roots=None):
    """The packed orbit webp, or None. `<preset>/<uuid>.webp`, flat by uuid."""
    rs = tuple(Path(r) for r in
               (roots if roots is not None else (LEGACY_WEBP_ROOT, NEW_WEBP_ROOT)))
    return _first_existing(*[r / preset / f"{uuid}.webp" for r in rs])


def box(uuid, root=None):
    """(emission ON, emission OFF), each None when absent. One root. Half a pair is a
    real state on disk (a crashed render), so the two sides resolve independently."""
    root = Path(root if root is not None else BOX_ROOT)
    return (_first_existing(root / f"{uuid}.png"),
            _first_existing(root / f"{uuid}_off.png"))


def atlas(uuid, root=None):
    """The bake npz the emission UV map is extracted from, or None."""
    return _first_existing(
        Path(root if root is not None else ATLAS_ROOT) / uuid / "atlas.npz")


# --- where a NEW render goes (no existence check) -------------------------------------
# The one place every new output path is built.
def new_still(kind, shard, uuid):
    return NEW_STILL_ROOTS[kind] / shard / uuid / "render.png"


def new_orbit_view(preset, view, shard, uuid):
    return NEW_ORBIT_ROOT / preset / view / shard / uuid / "render.png"


def new_webp(preset, uuid):
    return NEW_WEBP_ROOT / preset / f"{uuid}.webp"
