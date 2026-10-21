"""The TexVerse thumbnail index: which shapes have a thumbnail, and where it is.

  python -m data_processing.data_split.thumbnail_index

Reads `batch_image_list.txt` beside the thumbnail batches (one `batch_NNNNN/<uuid>.png` per
line; the files hold JPEG data under the `.png` name) and writes `out/thumbnail_pass.txt`,
the uuids that have one (one of the split's `--require` lists).
Standard library only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.metadata_io import read_thumbnail_list
from utils.paths import REPO_ROOT, TEXVERSE_THUMBNAIL_ROOT

INDEX = TEXVERSE_THUMBNAIL_ROOT / "batch_image_list.txt"
PASS_LIST = REPO_ROOT / "data_processing/data_split/out/thumbnail_pass.txt"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", default=str(INDEX))
    ap.add_argument("--out", default=str(PASS_LIST))
    a = ap.parse_args(argv)
    uuids = sorted(read_thumbnail_list(a.list))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(u + "\n" for u in uuids))
    print(f"{len(uuids)} thumbnails -> {a.out}")


if __name__ == "__main__":
    main()
