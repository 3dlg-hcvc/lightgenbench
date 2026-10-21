"""One 2x2 composite per shape, the picture every VLM step of the category run sees.

Top left the TexVerse thumbnail, then the `standard` orbit views along +x, -y and +z, each
center-cropped to a square and resized to 512 px. The shapes are the uuids of `--list`, by
default the heuristic stage's albedo-copy pass list (the bake's input), each with its bucket from
the TexVerse metadata; the orbit views are read from the legacy orbit root.
Resume-safe: a uuid that already has a composite on disk is skipped.

Output: `out_full/composites_2x2/<uuid>.png`.

Usage:
    python -m data_processing.tools.categories.composites --workers 64
"""
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from data_processing.tools.categories.utils import (
    COMPOSITES_FULL, METADATA_PARQUET, MULTIVIEW_AXES, MULTIVIEW_ROOT, PANE_SIZE, THUMBNAILS_ROOT,
    resolve_buckets,
)
from utils.metadata_io import read_uuids
from utils.paths import ALBEDO_COPY_PASS_LIST


def _center_square_crop(img: Image.Image) -> Image.Image:
    """Center-crop to a square using the shorter side. No-op when already square."""
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _load_panes(
    uuid: str,
    thumbnail_root: Path,
    thumbnail_batch: str,
    multiview_root: Path,
    multiview_bucket: str,
    pane_size: int,
) -> Optional[List[Image.Image]]:
    paths = [thumbnail_root / thumbnail_batch / f"{uuid}.png"]
    for axis in MULTIVIEW_AXES:
        paths.append(multiview_root / "standard" / axis / multiview_bucket / uuid / "render.png")
    if not all(p.exists() for p in paths):
        return None
    return [
        _center_square_crop(Image.open(p).convert("RGB"))
        .resize((pane_size, pane_size), Image.LANCZOS)
        for p in paths
    ]


def _paste_2x2(panes: List[Image.Image], pane_size: int) -> Image.Image:
    canvas = Image.new("RGB", (2 * pane_size, 2 * pane_size), (255, 255, 255))
    canvas.paste(panes[0], (0, 0))
    canvas.paste(panes[1], (pane_size, 0))
    canvas.paste(panes[2], (0, pane_size))
    canvas.paste(panes[3], (pane_size, pane_size))
    return canvas


def build_composite_2x2(*, uuid: str, thumbnail_root: Path, thumbnail_batch: str,
                         multiview_root: Path, multiview_bucket: str, out_path: Path,
                         pane_size: int) -> bool:
    """One composite from explicit roots (the backfill's call); False when a pane is missing."""
    panes = _load_panes(uuid, thumbnail_root, thumbnail_batch, multiview_root, multiview_bucket, pane_size)
    if panes is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _paste_2x2(panes, pane_size).save(out_path)
    return True


def build_thumbnail_index(thumbnail_root: Path) -> Dict[str, str]:
    """Walk batch_<NNNNN>/<uuid>.png to map uuid -> batch dirname."""
    index: Dict[str, str] = {}
    for batch_dir in sorted(thumbnail_root.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        for png in batch_dir.glob("*.png"):
            index[png.stem] = batch_dir.name
    return index


# Worker globals — populated by _init_worker so we don't pass huge dicts as args.
_WORKER_THUMB_INDEX: Dict[str, str] | None = None


def _init_worker(thumb_index: Dict[str, str]) -> None:
    global _WORKER_THUMB_INDEX
    _WORKER_THUMB_INDEX = thumb_index


def _build_one(args: Tuple[str, str, int, Path]) -> Tuple[str, bool, bool]:
    """Build the 2×2 composite for one uuid. Returns (uuid, was_skipped, ok)."""
    uuid, bucket, pane_size, out_dir = args
    out_path = out_dir / f"{uuid}.png"
    if out_path.exists():
        return uuid, True, True
    batch = _WORKER_THUMB_INDEX.get(uuid) if _WORKER_THUMB_INDEX else None
    if batch is None:
        return uuid, False, False
    panes = _load_panes(
        uuid=uuid,
        thumbnail_root=THUMBNAILS_ROOT,
        thumbnail_batch=batch,
        multiview_root=MULTIVIEW_ROOT,
        multiview_bucket=bucket,
        pane_size=pane_size,
    )
    if panes is None:
        return uuid, False, False
    _paste_2x2(panes, pane_size).save(out_path)
    return uuid, False, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--list", default=str(ALBEDO_COPY_PASS_LIST),
                        help="uuids, one per line (default: the heuristic stage's albedo-copy pass list)")
    parser.add_argument("--pane-size", type=int, default=PANE_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke testing: only process the first N ids.")
    args = parser.parse_args()

    out_dir = COMPOSITES_FULL
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] loading the shape list {args.list} ...")
    uuids = read_uuids(args.list)
    if args.limit:
        uuids = uuids[: args.limit]
    print(f"   {len(uuids)} uuids")

    print(f"[2/3] resolving uuid -> bucket via the TexVerse metadata ...")
    resolved = resolve_buckets(uuids, METADATA_PARQUET)
    print(f"   {len(resolved)} resolved")

    print(f"[2.5/3] building thumbnail index (one-time scan) ...")
    thumb_index = build_thumbnail_index(THUMBNAILS_ROOT)
    print(f"   thumbnail_index: {len(thumb_index)} uuids")

    work: List[Tuple[str, str, int, Path]] = [
        (r.uuid, r.bucket, args.pane_size, out_dir) for r in resolved
    ]

    print(f"[3/3] building composites with {args.workers} workers ...")
    built = 0
    skipped = 0
    failed = 0
    n = len(work)
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(thumb_index,)) as pool:
        for i, (uuid, was_skipped, ok) in enumerate(
            pool.imap_unordered(_build_one, work, chunksize=8), start=1
        ):
            if was_skipped:
                skipped += 1
            elif ok:
                built += 1
            else:
                failed += 1
            if i % 1000 == 0:
                print(f"   [{i}/{n}] built={built} resumed-skip={skipped} failed={failed}")
    print(f"\nDone. built={built}  resumed-skip={skipped}  failed={failed}  total={n}")


if __name__ == "__main__":
    main()
