"""Locating a TexVerse source GLB on disk.

One shape id (our SHA, which is also the Sketchfab model uid) plus the metadata
parquet gives one file path.

    from utils.glb_io import resolve_glb, find_glb

``resolve_glb`` is the strict form used inside the pipelines: you pass the root, and
a missing shape raises. ``find_glb`` is the convenience form for analysis and
one-offs: it tries the known roots and returns None when the file is not there.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from utils.paths import TEXVERSE_1K_ROOTS  # re-exported; defined in utils/paths.py

__all__ = [
    "TEXVERSE_1K_ROOTS",
    "GLB_PATH_COLUMN",
    "resolve_glb",
    "find_glb",
]

#: Column of the TexVerse metadata parquet holding the GLB path relative to a root.
GLB_PATH_COLUMN = "glb_1k_path"


def resolve_glb(sha: str, glb_root: str, metadata) -> str:
    """``glb_root`` joined with the shape's ``glb_1k_path``. Raises if absent.

    The shape id is the parquet's INDEX, not a column. Existence is the caller's
    business: this function only joins the path.

    Args:
        sha: shape id / ditem_id.
        glb_root: root to join against, e.g. one of ``TEXVERSE_1K_ROOTS``.
        metadata: DataFrame indexed by shape id with a ``glb_1k_path`` column.
    """
    row = metadata.loc[sha]
    return os.path.join(glb_root, row[GLB_PATH_COLUMN])


def find_glb(sha: str, metadata, roots: Sequence["os.PathLike | str"] = TEXVERSE_1K_ROOTS) -> Optional[str]:
    """First existing GLB for ``sha`` across ``roots``, or None.

    The forgiving counterpart to ``resolve_glb``, for scans and analysis that must
    survive a shape missing from the parquet or absent from a root. Returns None
    rather than raising in both cases, so a caller can count what it skipped.
    """
    try:
        rel = metadata.loc[sha][GLB_PATH_COLUMN]
    except KeyError:
        return None
    if rel is None or not isinstance(rel, str) or not rel:
        return None
    for root in roots:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            return path
    return None
