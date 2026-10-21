"""Every stage's uuid lists, file hashes, the shard bucket of a sharded render tree and
TexVerse's thumbnail list.

    from utils.metadata_io import read_uuids, write_uuids, sha256_of_file, shard_of
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, List, Optional

__all__ = [
    "read_uuids",
    "write_uuids",
    "sha256_of_file",
    "shard_of",
    "read_thumbnail_list",
]


# --- uuid lists, file hashes and the shard bucket ------------------------------------
# One reader and one writer for every stage's uuid list.
def read_uuids(list_path) -> List[str]:
    """One uuid per line, whitespace stripped; blank lines and `#` comment lines skipped
    (the VLM stage's pass list carries a two-line header, `write_uuids(..., header=)`)."""
    lines = Path(list_path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def write_uuids(path, uuids: Iterable[str], header: Optional[Iterable[str]] = None) -> None:
    """One uuid per line, each line ending in a newline, so an empty list is an empty file.
    `header` lines come first, each as `# <line>`, which `read_uuids` skips. No directory is
    created."""
    text = "".join(f"# {ln}\n" for ln in (header or ())) + "".join(f"{u}\n" for u in uuids)
    Path(path).write_text(text, encoding="utf-8")


def sha256_of_file(path) -> str:
    """The hex sha256 of a file's bytes (a record's provenance; the split's printed digest)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shard_of(ditem_dir: str) -> str:
    """The bucket directory a sharded render tree keys on: `000-000` of `000-000/<uuid>`.
    A bare uuid comes back unchanged; there is no error for it."""
    return str(ditem_dir).split("/")[0]


# --- TexVerse's thumbnail list -------------------------------------------------------
_UUID = re.compile(r"^[0-9a-f]{32}$")


def read_thumbnail_list(list_path) -> dict:
    """uuid -> path relative to the thumbnail root, from `batch_image_list.txt` (one
    `batch_NNNNN/<uuid>.png` per line). Raises on a malformed or repeated name."""
    out = {}
    for line in Path(list_path).read_text().splitlines():
        rel = line.strip()
        if not rel:
            continue
        uuid = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if not _UUID.match(uuid):
            raise ValueError(f"not a uuid thumbnail name: {rel}")
        if uuid in out:
            raise ValueError(f"{uuid} listed twice: {out[uuid]} and {rel}")
        out[uuid] = rel
    return out
