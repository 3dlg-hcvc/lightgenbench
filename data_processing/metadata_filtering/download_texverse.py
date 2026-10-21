"""Download TexVerse's id list and metadata, and the TexVerse-1K GLBs of a released split.

    conda run --no-capture-output -n lightgen python -m \\
      data_processing.metadata_filtering.download_texverse --split {train,val,test,all}

Into `data/TexVerse`: `TexVerse_id_list.txt` and `metadata.json`, which lists each model's GLB
paths. Into `data/TexVerse-1K`: the GLB of every shape in the split (read from the released
dataset's `splits.json`), at `glbs/glbs_1k/<folder>/<uuid>_1024.glb`, the lowest-named folder
when there are two, the one `glb_index.py` picks. Each file is one request, so nothing lists
the whole repository. A file already on disk is skipped, so running the command again resumes.
It exits non-zero listing every uuid without a GLB path in the metadata or whose download
failed. Last, it writes the GLB index (`data/texverse_index.parquet`, `glb_index.build` over
what is on disk), through which the evaluation finds each GLB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data_processing.metadata_filtering.glb_index import (  # noqa: E402
    GLB_DIR,
    THUMBNAIL_LIST,
    TEXVERSE_1K_REPO,
    TEXVERSE_1K_REVISION,
    TEXVERSE_REPO,
    TEXVERSE_REVISION,
    build,
    write_if_changed,
)
from utils.paths import RELEASE_ROOT, TEXVERSE_1K_ROOTS, TEXVERSE_METADATA_PARQUET, TEXVERSE_ROOT  # noqa: E402

#: The release revision's `metadata.json` is no longer served; this later one gives the same
#: GLB paths for the released shapes.
METADATA_REVISION = "8166bb11d24c03fb2c34d0ee099d754ce492bb54"
#: The TexVerse files to fetch, with the revision each is read at.
TEXVERSE_FILES = {"TexVerse_id_list.txt": TEXVERSE_REVISION, "metadata.json": METADATA_REVISION}
SPLITS = ("train", "val", "test")
#: Print a progress line after this many GLBs.
PROGRESS_EVERY = 100


def download(repo: str, name: str, revision: str, local_dir: Path) -> None:
    """One file of a dataset repository into `local_dir`. `force_download` skips
    huggingface_hub's copy from its own cache straight into place, so a run stopped midway
    never leaves a partial file that the next run would take as done."""
    hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=local_dir,
                    force_download=True)


def fetch_texverse(texverse_dir: Path) -> None:
    """The id list and the metadata into `texverse_dir`, each skipped when already there."""
    for name, revision in TEXVERSE_FILES.items():
        if (texverse_dir / name).is_file():
            print(f"{texverse_dir / name} is there, skipped", flush=True)
            continue
        print(f"downloading {TEXVERSE_REPO}/{name} at {revision[:8]}", flush=True)
        download(TEXVERSE_REPO, name, revision, texverse_dir)


def read_split(splits_json, split: str) -> list:
    """The split's uuids in file order; `all` is train, val and test."""
    splits = json.loads(Path(splits_json).read_text())
    names = SPLITS if split == "all" else (split,)
    return list(dict.fromkeys(u for s in names for u in splits[s]))


def glb_paths(metadata_json, uuids) -> dict:
    """uuid -> its GLB under the TexVerse-1K root: the smallest `glb_paths` entry in GLB_DIR.
    A uuid with no such entry is left out."""
    meta = json.loads(Path(metadata_json).read_text())
    out = {}
    for u in uuids:
        paths = [p for p in (meta.get(u) or {}).get("glb_paths") or [] if p.startswith(GLB_DIR + "/")]
        if paths:
            out[u] = min(paths)
    return out


def fetch_glbs(paths: dict, glb_dir: Path, workers: int) -> dict:
    """Download each GLB not yet in `glb_dir`, one file per call. Returns uuid -> error."""
    todo = {u: p for u, p in paths.items() if not (glb_dir / p).is_file()}
    print(f"{len(paths) - len(todo)} GLBs are there, {len(todo)} to download", flush=True)
    failed = {}
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(download, TEXVERSE_1K_REPO, p, TEXVERSE_1K_REVISION, glb_dir): u
                   for u, p in todo.items()}
        for n, fut in enumerate(as_completed(futures), 1):
            if fut.exception() is not None:
                failed[futures[fut]] = repr(fut.exception())
            if n % PROGRESS_EVERY == 0:
                print(f"{n}/{len(todo)} done, {len(failed)} failed", flush=True)
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=SPLITS + ("all",))
    ap.add_argument("--splits", default=str(RELEASE_ROOT / "splits.json"),
                    help="the released dataset's split file")
    ap.add_argument("--texverse_dir", default=str(TEXVERSE_ROOT),
                    help="where the id list and metadata.json go")
    ap.add_argument("--glb_dir", default=str(TEXVERSE_1K_ROOTS[0]), help="the TexVerse-1K root")
    ap.add_argument("--workers", type=int, default=8, help="GLBs downloaded at once")
    ap.add_argument("--index_out", default=str(TEXVERSE_METADATA_PARQUET), help="the GLB index it writes")
    a = ap.parse_args(argv)
    texverse_dir, glb_dir = Path(a.texverse_dir), Path(a.glb_dir)

    fetch_texverse(texverse_dir)
    uuids = read_split(a.splits, a.split)
    paths = glb_paths(texverse_dir / "metadata.json", uuids)
    missing = [u for u in uuids if u not in paths]
    print(f"{len(uuids)} shapes in the {a.split} split, {len(paths)} with a GLB path", flush=True)
    failed = fetch_glbs(paths, glb_dir, a.workers)
    print(f"{len(paths) - len(failed)} of {len(paths)} GLBs under {glb_dir / GLB_DIR}", flush=True)
    thumbs = texverse_dir / THUMBNAIL_LIST.relative_to(TEXVERSE_ROOT)  # optional, as in glb_index
    df, _ = build(texverse_dir / "TexVerse_id_list.txt", glb_dir, thumbs, a.workers)
    wrote = write_if_changed(df, Path(a.index_out))
    print(f"GLB index: {int(df['if_downloaded_glb'].sum())} GLBs -> {a.index_out}"
          f"{'' if wrote else ' (unchanged)'}", flush=True)

    for u in missing:
        print(f"no GLB path in the metadata: {u}", file=sys.stderr)
    for u, err in failed.items():
        print(f"download failed: {u} {paths[u]}: {err}", file=sys.stderr)
    if missing or failed:
        sys.exit(f"{len(missing)} without a GLB path, {len(failed)} failed")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # pyarrow's threads can abort the interpreter's teardown under load
