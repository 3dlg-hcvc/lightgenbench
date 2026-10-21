"""Images 1 to 3, the single-view images: the lit object, emit-only, and emission-boosted.

One render each, all three from the camera baked into the preset blend:
`render_utils.render_scene` only moves the camera when `camera_position` is passed, so
leaving it out is what reproduces the shipped stills. 8 samples + denoise is the shipped
setting for all three kinds.

    python -u -m data_processing.render.single_view --shard <shard file> --rank 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from utils import review_images

from . import utils as render_utils
from .utils import ShapeGuard


def blend_path(key: str) -> Path:
    """The preset blend for a still kind or an orbit preset, from `assets/`."""
    return render_utils.BLEND_ASSETS_DIR / render_utils.PRESET_BLEND[key]


def render_args(kind: str) -> dict:
    """The kwargs handed to `render_utils.render_glb`.

    No `camera_position`: see the module docstring. `shadow_catcher=False` matches the
    shipped stills - with it on the floor object catches a shadow the originals do not
    have."""
    return {
        "preset_blend": str(blend_path(kind)),
        "shadow_catcher": False,
        "resolution": render_utils.REVIEW_RESOLUTION,
        "samples": render_utils.REVIEW_SAMPLES,
        "denoise": render_utils.REVIEW_DENOISE,
    }


def glb_path_for(uuid: str, metadata_path=None, glb_root=None) -> Path:
    import pandas as pd

    meta = pd.read_parquet(Path(metadata_path or render_utils.DEFAULT_METADATA),
                           columns=["glb_1k_path"])
    root = Path(glb_root or render_utils.DEFAULT_GLB_ROOTS[0])
    return root / meta.loc[uuid, "glb_1k_path"]


def render_one(kind: str, glb_path, out_path) -> bool:
    """Render one still.

    A *render* failure returns False rather than raising: one bad GLB must not take the
    shard down. A *write* failure raises; when the store refused it for space or quota
    (`render_utils.is_disk_full`), the shape guard lets it stop the shard rather than count
    every shape left as failed, and any other write error fails the shape. It leaves no temp
    file behind either way."""
    # Outside the try on purpose: a host without bpy or PIL fails the whole shape (the
    # guard names it in the failures file) rather than each still one by one.
    import bpy  # noqa: F401
    from PIL import Image  # noqa: F401

    out_path = Path(out_path)
    try:
        img = render_utils.render_glb(str(glb_path), **render_args(kind))
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        if render_utils.is_disk_full(e):        # a full store stops the shard (the guard)
            raise
        print(f"[stills] {kind} {glb_path}: {type(e).__name__}: {e}", flush=True)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".tmp{os.getpid()}.png")
    try:
        render_utils.save_img(str(tmp), img)
        os.replace(tmp, out_path)               # atomic: a reader never sees a partial PNG
    finally:
        # A failed write (a full share) raises, as it should, but it must not leave a
        # half-written temp file behind on the store.
        tmp.unlink(missing_ok=True)
    return True


def _parse_shard_line(line: str):
    """`<uuid>\t<shard>\t<glb path>` - the shard file carries resolved paths so the
    render worker needs bpy only, not pandas."""
    uuid, shard, glb = line.rstrip("\n").split("\t")
    return uuid, shard, glb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True, help="uuid<TAB>shard<TAB>glb, one per line")
    ap.add_argument("--kinds", default=",".join(review_images.STILL_KINDS))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--timeout_s", type=int, default=600,
                    help="per-shape wall-clock cap; a shape over it is named and skipped")
    ap.add_argument("--guard_dir", default=None,
                    help="markers and failures file; default <run dir>/guard/<lane>")
    a = ap.parse_args()

    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    for k in kinds:
        if k not in review_images.STILL_KINDS:
            ap.error(f"unknown kind {k!r}; expected one of {review_images.STILL_KINDS}")

    rows = [_parse_shard_line(ln) for ln in open(a.shard) if ln.strip()]
    # The guard dir sits beside the shard files (`<run dir>/shards/` on the node), so a
    # rank resubmitted onto another node still sees its own markers and failures.
    guard = ShapeGuard(a.guard_dir or Path(a.shard).resolve().parent.parent / "guard" / "stills",
                       a.rank, a.timeout_s)
    skip = guard.adopt()
    done = failed = skipped = shapes_failed = 0
    t0 = time.time()
    for i, (uuid, shard, glb) in enumerate(rows):
        if uuid in skip:                            # named in the failures file: once
            shapes_failed += 1
            continue
        with guard.attempt(uuid) as att:
            for kind in kinds:
                if review_images.still(kind, shard, uuid) is not None:
                    skipped += 1                     # resume: already on disk in either root
                    continue
                if render_one(kind, glb, review_images.new_still(kind, shard, uuid)):
                    done += 1
                else:
                    failed += 1
        if att.failed:
            shapes_failed += 1
            print(f"[stills] {uuid}: {att.error} ({att.seconds:.0f}s)", flush=True)
        if (i + 1) % 100 == 0:
            print(f"[stills rank{a.rank}] {i + 1}/{len(rows)} "
                  f"done={done} skipped={skipped} failed={failed} shapes_failed={shapes_failed} "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"[stills rank{a.rank}] ALL_DONE done={done} skipped={skipped} failed={failed} shapes_failed={shapes_failed}",
          flush=True)
    sys.stdout.flush()
    os._exit(0)                                  # bpy segfaults on teardown


if __name__ == "__main__":
    main()
