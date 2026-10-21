"""The tool's configuration, where an item's images live, and the emission view's cache.

Paths are env-overridable (`ANN74K_*`).

Image lookups: `thumb_path` and `atlas_npz_path` are pure spellings of a root. The four render
builders are not: they ask `utils.review_images`, which checks the read-only legacy root first
and the new root second. Two consequences for callers:

* they touch the disk, and
* they return None when neither root holds the image — a normal state, not an error
  (the card renders a missing image as "not rendered"), so no caller may assume a Path.

The emission view: a displayable emission PNG from the bake's atlas (key `emission_color`),
cached per uuid. The PNG itself is `utils.atlas_io.write_emission_png` (the VLM check writes
its eighth image with the same function): sRGB-encoded, masked by `occupancy`, written
atomically.
"""
import os
from pathlib import Path

from utils import review_images
from utils.atlas_io import write_emission_png
from utils.metadata_io import shard_of

# The set the tool serves and the metadata it addresses shapes through. Imported rather
# than re-spelled so this stage and the review renders cannot drift onto two different
# lists: INPUT_LIST is the heuristic stage's pass list (the manifest's row source) and
# METADATA_PARQUET, the GLB index, has `ditem_dir` for every downloaded GLB. Both come
# from `utils/paths.py`.
from utils.paths import (
    ALBEDO_COPY_PASS_LIST as INPUT_LIST,
    REPO_ROOT,
    TEXVERSE_METADATA_PARQUET as METADATA_PARQUET,
    REPRESENTATIONS_ROOT,
)

# ---- configuration ---------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent

THUMB_DIR = Path(os.environ.get(
    "ANN74K_THUMB_DIR", REPO_ROOT / "processed_data" / "emissive_thumbnails"))
CATEGORIES_FINAL = Path(os.environ.get(
    "ANN74K_CATEGORIES_FINAL",
    REPO_ROOT / "data_processing" / "tools" / "categories" / "out_full" / "categories_final.jsonl"))
#: Tier 2 of the category chain: what `tools/categories/backfill.py` wrote for
#: the shapes new to the filtered set. Absent until that runs, which is normal.
CATEGORIES_BACKFILL = Path(os.environ.get(
    "ANN74K_CATEGORIES_BACKFILL",
    CATEGORIES_FINAL.parent / "categories_backfill.jsonl"))

DATA_DIR = Path(os.environ.get("ANN74K_DATA_DIR", PKG_DIR / "_data"))
DB_PATH = DATA_DIR / "annotations.db"
# user-study submissions (the study page POSTs here); one json per submit + a
# submissions.jsonl index.
STUDY_RESULTS_DIR = Path(os.environ.get("ANN74K_STUDY_DIR", DATA_DIR / "study_results"))
MANIFEST_PATH = DATA_DIR / "manifest.parquet"
CATEGORIES_PATH = DATA_DIR / "categories.json"
# Entries are keyed by uuid alone, so a bake change needs a new DIRECTORY, not a new
# filename — an existing entry is served without re-extraction. Rollback is pointing this
# back at an older directory. Override to shared disk to persist/prewarm; a deploy that
# sets it serves that directory, not this default.
EMISSION_CACHE_DIR = Path(os.environ.get(
    "ANN74K_EMISSION_CACHE_DIR", DATA_DIR / "emission_cache_73432"))

FRONTEND_DIR = PKG_DIR / "frontend"
# annotator guide, served raw at /guide and rendered client-side; its `static/*.png`
# images resolve in the repo through the `static -> frontend` symlink
GUIDE_PATH = PKG_DIR / "guide.md"

# Emission is read from the bake (`glb2representations/`, UV atlases, `atlas.npz`), which
# stores the source emission texture alone (no factor, no strength). A shape with no atlas
# shows "not rendered"; there is no fallback to another bake.
ATLAS_74K_ROOT = Path(os.environ.get("ANN74K_ATLAS_ROOT", str(REPRESENTATIONS_ROOT)))

# Animated 6-view orbit webp (one file per uuid), shown in the "more" overlay. Every render
# root, this one and the dark-box pair's included, is read through `utils.review_images`
# (the lookups below).
MULTIVIEW_PRESETS = {"standard": "standard", "emit": "emit_only"}  # url key -> on-disk dir

# Reason tags shown after an object is marked "Not emissive" (stored in the DB `flags` column).
FLAGS = ["not_emissive_object", "wrong_emissive_maps", "zero_emission_map", "multiple_object"]

# Style tags shown after an object is marked "No issue" (stored in the DB `style` column).
STYLES = ["artistic", "real"]

PORT = int(os.environ.get("ANN74K_PORT", "8074"))

# Shared secret for the read-only admin dashboard (/admin + /api/admin/*). The service may
# be publicly reachable through a reverse proxy, so the admin surface is token-gated.
# Unset (the default) disables it entirely and every admin route 404s, so a deploy that
# forgets to set the token is closed rather than open. Never commit a value here — export
# ANN74K_ADMIN_TOKEN from a file outside the repo.
ADMIN_TOKEN = os.environ.get("ANN74K_ADMIN_TOKEN") or None


# ---- where an item's images live ------------------------------------------------------

def thumb_path(uuid: str) -> Path:
    return THUMB_DIR / f"{uuid}.png"


def render_path(kind: str, ditem_dir: str, uuid: str):
    """The card still of `kind`, from whichever root holds it, or None."""
    return review_images.still(kind, shard_of(ditem_dir), uuid)


def multiview_webp_path(preset: str, uuid: str):
    """The animated 6-view orbit webp, or None. `preset` is the url key."""
    return review_images.webp(MULTIVIEW_PRESETS[preset], uuid)


def box_render_path(uuid: str):
    """Dark-box still (emission ON) — flat by uuid, unlike the sharded still kinds."""
    return review_images.box(uuid)[0]


def box_render_off_path(uuid: str):
    """The emission-strength-0 companion of the dark-box still."""
    return review_images.box(uuid)[1]


def atlas_npz_path(uuid: str) -> Path:
    """The bake's UV atlas (`glb2representations/`) — flat by uuid, one directory per shape."""
    return ATLAS_74K_ROOT / uuid / "atlas.npz"


# ---- the emission view ------------------------------------------------------------------

def resolve_emission_source(uuid: str):
    """The bake npz a shape's emission is read from, or None.

    There is no fallback to another bake (its emission convention would differ in front of
    the same annotator): a shape without an atlas shows "not rendered", which the card
    already handles.
    """
    atlas = atlas_npz_path(uuid)
    return atlas if atlas.exists() else None


def cache_png_path(uuid: str):
    """Cache entry for a shape.

    Isolation from other caches is by DIRECTORY (`EMISSION_CACHE_DIR`), not by
    filename: the key is the uuid alone and carries no trace of which bake produced
    the image, so a bake change that kept the directory would serve every already
    extracted shape from the old bake forever. Keeping old dirs untouched makes
    rollback a config flip.
    """
    return EMISSION_CACHE_DIR / f"{uuid}.png"


def get_emission_cached(uuid: str):
    """Return cached emission PNG path, extracting on first call. None if unavailable."""
    npz_path = resolve_emission_source(uuid)
    if npz_path is None:
        return None
    out_path = cache_png_path(uuid)
    if out_path.exists():
        return out_path
    if write_emission_png(npz_path, out_path):
        return out_path
    return None
