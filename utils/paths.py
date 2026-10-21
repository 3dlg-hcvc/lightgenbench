"""Repo-wide dataset paths, the single place these are spelled out.

A tool folder's own `config.py` imports from here and adds only what is specific to
that tool. Every default is inside the repo: data lives under `data/` (a download, or a
symlink to where it is); pass explicit roots rather than editing this file.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "DATA_ROOT",
    "TEXVERSE_ROOT",
    "TEXVERSE_METADATA_PARQUET",
    "TEXVERSE_THUMBNAIL_ROOT",
    "TEXVERSE_1K_ROOTS",
    "REPRESENTATIONS_ROOT",
    "RELEASE_ROOT",
    "LABEL_DBS_ROOT",
    "ALBEDO_COPY_PASS_LIST",
    "OPENAI_KEY_PATH",
    "CATEGORIES_OUT_FULL",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Where the data lives.
DATA_ROOT = REPO_ROOT / "data"

# --- TexVerse source assets -------------------------------------------------------
#: The TexVerse dataset root (metadata + thumbnails).
TEXVERSE_ROOT = DATA_ROOT / "TexVerse"
#: The GLB index, one row per TexVerse id (index `uuid`): `glb_1k_path`, `ditem_dir`,
#: `if_downloaded_glb`, `thumbnail_path`. Generated from the TexVerse and TexVerse-1K downloads
#: by `data_processing/metadata_filtering/glb_index.py`, which that stage's `run.sh` runs first.
TEXVERSE_METADATA_PARQUET = DATA_ROOT / "texverse_index.parquet"
#: Where `thumbnail_path` in that parquet is relative to.
TEXVERSE_THUMBNAIL_ROOT = TEXVERSE_ROOT / "thumbnails" / "thumbnails_batch"
#: The roots holding the 1K-decimated GLBs (TexVerse-1K), tried in this order.
TEXVERSE_1K_ROOTS = (DATA_ROOT / "TexVerse-1K",)

# --- the stages' trees under data/ -----------------------------------------------------
#: The representations stage's output over the albedo-copy check's pass list (texture-alone
#: emission). One directory per shape holding all three representations:
#: <uuid>/{atlas.npz, <uuid>.coords.npz, emission_voxels_256/, pbr_voxels_256/, render_tex/}.
REPRESENTATIONS_ROOT = DATA_ROOT / "representations"
#: The released dataset (Hugging Face `3dlg-hcvc/LightgenBench`), unpacked: `splits.json`
#: and one `<uuid>/` per shape.
RELEASE_ROOT = DATA_ROOT / "lightgenbench"
#: Copies of the annotation tool's SQLite DBs (`annotations_<name>.db`), read by the VLM
#: check's label freeze and the copy-check precision count.
LABEL_DBS_ROOT = DATA_ROOT / "annotation_dbs"

# --- lists and files several stages read ------------------------------------------------
#: The heuristic stage's albedo-copy pass list, one uuid per line: the bake's
#: input, the review images' and the annotation tool's set (`glb2representations/run.py`,
#: `render/utils.py` and `tools/human_annotation/utils.py` name it `INPUT_LIST`), and the
#: default `--list` of `tools/categories/composites.py`.
ALBEDO_COPY_PASS_LIST = REPO_ROOT / "data_processing" / "heuristic_filtering" / "out" / "albedo_copy_pass.txt"
#: The OpenAI key file (gitignored; else the `OPENAI_API_KEY` environment variable), read by
#: the VLM check and the category pipeline. Never written anywhere.
OPENAI_KEY_PATH = REPO_ROOT / "api_key" / "key.config"
#: The category pipeline's records (gitignored), written by `tools/categories/` and read
#: by it and by `tools/dataset_statistics/category_counts.py`.
CATEGORIES_OUT_FULL = REPO_ROOT / "data_processing" / "tools" / "categories" / "out_full"
