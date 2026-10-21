"""The GLB index: where each TexVerse shape's GLB and thumbnail are, from the download alone.

    conda run --no-capture-output -n lightgen python -m \\
      data_processing.metadata_filtering.glb_index [--out <parquet>]

Every stage finds its GLBs through this file (`utils.paths.TEXVERSE_METADATA_PARQUET`).
`run.sh` here runs this at every run, so the index always follows the download on disk; the
file is rewritten only when its bytes would change. It reads three things TexVerse ships: the
id list `TexVerse_id_list.txt` (the rows), the TexVerse-1K tree
`glbs/glbs_1k/<folder>/<uuid>_1024.glb` (listing it takes a few minutes over NFS) and the
thumbnails' `batch_image_list.txt` (optional: without it `thumbnail_path` stays empty). One
row per id, index `uuid`:

  glb_1k_path        glbs/glbs_1k/<folder>/<uuid>_1024.glb, relative to the TexVerse-1K root;
                     a uuid in two folders takes the lowest-named one. Null without a GLB.
  ditem_dir          <folder>/<uuid>; the folder is the bucket of a sharded render tree.
                     Null without a GLB.
  if_downloaded_glb  the GLB is in the tree: the set the metadata scan reads.
  thumbnail_path     batch_NNNNN/<uuid>.png, relative to the thumbnail root. Null without one.

The rows with a GLB come first, in `ditem_dir` order, then the rest by uuid, so the scan's
input list has one fixed order. A GLB whose uuid is not in the id list gets a row too.

The released data was built from the two Hugging Face datasets at the revisions below. An
index built from them has `RELEASED_DIGEST` (`glb_digest`: the rows and their GLB paths);
any other download, such as a later TexVerse-1K revision or a partial one, gets a warning,
because the stages cannot give the released data back from it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.metadata_io import read_thumbnail_list  # noqa: E402
from utils.paths import (  # noqa: E402
    TEXVERSE_1K_ROOTS,
    TEXVERSE_METADATA_PARQUET,
    TEXVERSE_ROOT,
    TEXVERSE_THUMBNAIL_ROOT,
)

#: TexVerse's list of every model id, one per line.
ID_LIST = TEXVERSE_ROOT / "TexVerse_id_list.txt"
#: The thumbnail archive's own list of its files (`batch_NNNNN/<uuid>.png`).
THUMBNAIL_LIST = TEXVERSE_THUMBNAIL_ROOT / "batch_image_list.txt"
#: Where the GLB folders are, under the TexVerse-1K root; `glb_1k_path` starts with it.
GLB_DIR = "glbs/glbs_1k"

#: The Hugging Face datasets the released data was built from, at the revisions it read:
#: TexVerse for the id list and the thumbnails, TexVerse-1K for the GLBs.
TEXVERSE_REPO = "YiboZhang2001/TexVerse"
TEXVERSE_REVISION = "a4fe9f473d6628e06c7187604a01c0d8b13de5e2"
TEXVERSE_1K_REPO = "YiboZhang2001/TexVerse-1K"
TEXVERSE_1K_REVISION = "8b4bf4a6eb0c3b3ce5244a863621d79b4d3f344d"
#: `glb_digest` of an index built from those revisions, and what that download holds.
RELEASED_DIGEST = "d4d5f6c30d73ac244f90723401a8f947fb316008d8042bca311c46ba89bf914f"
RELEASED_COUNTS = {"ids": 858669, "GLBs": 856590, "folders": 88, "second copies": 1408}

_UUID = re.compile(r"^[0-9a-f]{32}$")
_GLB = re.compile(r"^([0-9a-f]{32})_1024\.glb$")


def read_ids(path) -> list:
    """The id list's uuids in file order. Raises on a malformed or repeated line."""
    ids, seen = [], set()
    for line in Path(path).read_text().splitlines():
        u = line.strip()
        if not u:
            continue
        if not _UUID.match(u):
            raise ValueError(f"{path}: not a uuid: {u!r}")
        if u in seen:
            raise ValueError(f"{path}: {u} listed twice")
        seen.add(u)
        ids.append(u)
    return ids


def find_glbs(glb_root, workers: int):
    """uuid -> the lowest-named folder holding `<uuid>_1024.glb`, with the tree's counts:
    folders, second copies skipped, and other entries (not a uuid's GLB) skipped."""
    base = Path(glb_root) / GLB_DIR
    with os.scandir(base) as it:
        folders = sorted(e.name for e in it if e.is_dir())
    if not folders:
        raise FileNotFoundError(f"no GLB folders under {base}")
    with ThreadPoolExecutor(workers) as ex:
        listed = list(ex.map(lambda f: sorted(os.listdir(base / f)), folders))
    first, copies, other = {}, 0, 0
    for folder, names in zip(folders, listed):
        for name in names:
            m = _GLB.match(name)
            if m is None:
                other += 1
            elif m.group(1) in first:
                copies += 1
            else:
                first[m.group(1)] = folder
    return first, {"folders": len(folders), "second copies": copies, "other entries": other}


def build(id_list, glb_root, thumbnail_list, workers: int):
    """The index, and the download's counts for the check against the released data's."""
    ids = read_ids(id_list)
    glbs, tree = find_glbs(glb_root, workers)
    if Path(thumbnail_list).is_file():
        thumbs = read_thumbnail_list(thumbnail_list)
    else:
        thumbs = {}
        print(f"no thumbnail list at {thumbnail_list}: thumbnail_path stays empty "
              f"(only `check_emission scan --thumbnails_out` reads it)", flush=True)
    ditem = {u: f"{folder}/{u}" for u, folder in glbs.items()}
    order = sorted(ditem, key=ditem.get) + sorted(set(ids) - ditem.keys())
    df = pd.DataFrame(
        {
            "glb_1k_path": pd.array([f"{GLB_DIR}/{ditem[u]}_1024.glb" if u in ditem else None
                                     for u in order], dtype="string"),
            "ditem_dir": pd.array([ditem.get(u) for u in order], dtype="string"),
            "if_downloaded_glb": [u in ditem for u in order],
            "thumbnail_path": pd.array([thumbs.get(u) for u in order], dtype="string"),
        },
        index=pd.Index(order, name="uuid"),
    )
    print(f"{len(ids)} ids in {id_list}; {len(glbs)} GLBs in {tree['folders']} folders under "
          f"{Path(glb_root) / GLB_DIR} ({tree['second copies']} second copies and "
          f"{tree['other entries']} other entries skipped, {len(ditem.keys() - set(ids))} not "
          f"in the id list); {int(df['thumbnail_path'].notna().sum())} rows with a thumbnail",
          flush=True)
    return df, {"ids": len(ids), "GLBs": len(glbs), **tree}


def glb_digest(df: pd.DataFrame) -> str:
    """sha256 of `<uuid>\\t<glb_1k_path>\\n` per row in index order (an empty path without a
    GLB): the rows, their order and the GLB each stage reads, not the thumbnails."""
    h = hashlib.sha256()
    for u, p in zip(df.index, df["glb_1k_path"]):
        h.update(f"{u}\t{'' if pd.isna(p) else p}\n".encode())
    return h.hexdigest()


def check_released(df: pd.DataFrame, counts: dict) -> bool:
    """Say whether this index is the one the released data was built from; warn when not."""
    if glb_digest(df) == RELEASED_DIGEST:
        print(f"the GLB index is the released data's ({TEXVERSE_1K_REPO} at "
              f"{TEXVERSE_1K_REVISION[:8]}, {TEXVERSE_REPO} at {TEXVERSE_REVISION[:8]})", flush=True)
        return True
    keys = list(RELEASED_COUNTS)
    print("\n".join([
        "WARNING: this TexVerse download is not the one the released data was built from, so the",
        "         stages cannot give the released data back from it.",
        "  this download:     " + ", ".join(f"{counts[k]} {k}" for k in keys),
        "  the released data: " + ", ".join(f"{RELEASED_COUNTS[k]} {k}" for k in keys),
        f"  Download {TEXVERSE_1K_REPO} at revision {TEXVERSE_1K_REVISION}",
        f"  and {TEXVERSE_REPO} at revision {TEXVERSE_REVISION}",
        "  (the repository README).",
    ]), file=sys.stderr, flush=True)
    return False


def write_if_changed(df: pd.DataFrame, out: Path) -> bool:
    """Write `out` unless it already holds these bytes, so a rerun leaves the file (and the
    readers that have it open) alone. Written beside `out` and renamed, so a killed build
    never leaves a file that looks done."""
    buf = io.BytesIO()
    df.to_parquet(buf)
    data = buf.getvalue()
    if out.is_file() and out.stat().st_size == len(data) and out.read_bytes() == data:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    umask = os.umask(0)
    os.umask(umask)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o666 & ~umask)    # mkstemp makes it private; the index is shared
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(TEXVERSE_METADATA_PARQUET))
    ap.add_argument("--id_list", default=str(ID_LIST))
    ap.add_argument("--glb_root", default=str(TEXVERSE_1K_ROOTS[0]), help="the TexVerse-1K root")
    ap.add_argument("--thumbnail_list", default=str(THUMBNAIL_LIST),
                    help="optional: without it thumbnail_path stays empty")
    ap.add_argument("--workers", type=int, default=16, help="folders listed at once")
    a = ap.parse_args(argv)
    out = Path(a.out)
    df, counts = build(a.id_list, a.glb_root, a.thumbnail_list, a.workers)
    wrote = write_if_changed(df, out)
    print(f"{len(df)} rows, {int(df['if_downloaded_glb'].sum())} with a GLB -> {out}"
          f"{'' if wrote else ' (unchanged)'}", flush=True)
    check_released(df, counts)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # pyarrow's threads can abort the interpreter's teardown under load
