"""Images 6 and 7: the six orbit views of a preset (`standard`, lit; `emit_only`), and
`webp`, which packs them into one animated webp per shape.

Unlike the card stills, these override the blend's camera: six axis positions looking at the
origin, in the frame order the packer writes. The work list is a shard file cut by
`dark_box shards --with_shard`.

    python -u -m data_processing.render.orbit --shard <shard file> --preset emit_only --rank 0

`webp` packs one preset per invocation and burns the view label into every frame, as in every
shipped webp: the annotation tool plays the frames and the VLM check reads them uncropped, so
a webp without labels would read as a different kind of image. A set with any view missing is
skipped, never packed short: a five-frame webp reads as a complete orbit and hides a missing
render. A packed webp is animated, not a still:
    im = Image.open(path)
    for i in range(im.n_frames):   # 0..5 = Front, Right, Back, Left, Top, Bottom
        im.seek(i); frame = im.convert("RGB")

    python -u -m data_processing.render.orbit webp --list <uuids> --preset emit_only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from utils import review_images

from . import utils as render_utils
# From `single_view`: the same asset lookup and the same shard line format
# (`<uuid>\t<shard>\t<glb>`), which `dark_box shards --with_shard` writes for both lanes.
from .single_view import _parse_shard_line, blend_path
from .utils import ShapeGuard


def render_args(preset: str, view: str) -> dict:
    """The kwargs handed to `render_utils.render_glb`.

    `camera_position` and `camera_up` are the point of this module: the blend's own
    three-quarter camera is what the card stills keep, and passing these is what turns
    the six views into six views rather than six copies of that still."""
    pos, up = render_utils.VIEW_CAMERA[view]
    return {
        "preset_blend": str(blend_path(preset)),
        "shadow_catcher": False,
        "resolution": render_utils.REVIEW_RESOLUTION,
        "samples": render_utils.REVIEW_SAMPLES,
        "denoise": render_utils.REVIEW_DENOISE,
        "camera_position": pos,
        "camera_up": up,
    }


def render_view(preset: str, view: str, glb_path, out_path) -> bool:
    """Render one view.

    Failure semantics are the stills': a *render* failure returns False, because one bad
    GLB must not take the shard down, while a *write* failure raises; a full store or a used-up
    quota (`render_utils.is_disk_full`) then stops the shard, any other write error fails the
    shape. It leaves no temp file behind either way."""
    # Outside the try on purpose: a host without bpy or PIL fails the whole shape (the
    # guard names it in the failures file) rather than each view one by one.
    import bpy  # noqa: F401
    from PIL import Image  # noqa: F401

    out_path = Path(out_path)
    try:
        img = render_utils.render_glb(str(glb_path), **render_args(preset, view))
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        if render_utils.is_disk_full(e):        # a full store stops the shard (the guard)
            raise
        print(f"[orbit] {preset}/{view} {glb_path}: {type(e).__name__}: {e}", flush=True)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".tmp{os.getpid()}.png")
    try:
        render_utils.save_img(str(tmp), img)
        os.replace(tmp, out_path)               # atomic: a reader never sees a partial PNG
    finally:
        tmp.unlink(missing_ok=True)
    return True


def render_main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m data_processing.render.orbit",
                                 description=__doc__)
    ap.add_argument("--shard", required=True, help="uuid<TAB>shard<TAB>glb, one per line")
    ap.add_argument("--preset", required=True, choices=list(review_images.ORBIT_PRESETS))
    ap.add_argument("--views", default=",".join(render_utils.VIEWS),
                    help="comma separated; the three view names that start with a dash "
                         "need the --views=-y_to_+y form")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--timeout_s", type=int, default=600,
                    help="per-shape wall-clock cap; a shape over it is named and skipped")
    ap.add_argument("--guard_dir", default=None,
                    help="markers and failures file; default <run dir>/guard/<lane>")
    a = ap.parse_args(argv)

    views = [v.strip() for v in a.views.split(",") if v.strip()]
    for v in views:
        if v not in render_utils.VIEWS:
            ap.error(f"unknown view {v!r}; expected one of {render_utils.VIEWS}")

    with open(a.shard) as f:
        rows = [_parse_shard_line(ln) for ln in f if ln.strip()]

    # The guard dir sits beside the shard files (`<run dir>/shards/` on the node), so a
    # rank resubmitted onto another node still sees its own markers and failures.
    guard = ShapeGuard(a.guard_dir or Path(a.shard).resolve().parent.parent / "guard" / a.preset,
                       a.rank, a.timeout_s)
    skip = guard.adopt()
    done = failed = skipped = shapes_failed = 0
    t0 = time.time()
    for i, (uuid, shard, glb) in enumerate(rows):
        if uuid in skip:                            # named in the failures file: once
            shapes_failed += 1
            continue
        with guard.attempt(uuid) as att:
            for view in views:
                if review_images.orbit_view(a.preset, view, shard, uuid) is not None:
                    skipped += 1                     # resume: already on disk in either root
                    continue
                if render_view(a.preset, view, glb,
                               review_images.new_orbit_view(a.preset, view, shard, uuid)):
                    done += 1
                else:
                    failed += 1
        if att.failed:
            shapes_failed += 1
            print(f"[orbit] {uuid}: {att.error} ({att.seconds:.0f}s)", flush=True)
        if (i + 1) % 50 == 0:
            print(f"[orbit rank{a.rank} {a.preset}] {i + 1}/{len(rows)} "
                  f"done={done} skipped={skipped} failed={failed} shapes_failed={shapes_failed} "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"[orbit rank{a.rank} {a.preset}] ALL_DONE "
          f"done={done} skipped={skipped} failed={failed} shapes_failed={shapes_failed}", flush=True)
    sys.stdout.flush()
    os._exit(0)                                  # bpy segfaults on teardown


# ================================================================ the packing step (`webp`)
# PIL is imported inside these functions, so the render lane imports this file without it.
#: 1000 ms per frame, a 6 s loop - the shipped set's timing.
FRAME_MS = 1000
#: Lossy at 90, the shipped set's setting. Lossless would not match it.
QUALITY = 90
# The label's typeface, size and placement live in `utils`, beside `VIEW_LABELS`.

_font = None


def _label_font():
    """The label typeface. A missing font raises rather than falling back to Pillow's
    bitmap default: the run is verified by counting files, so a substitute face would give
    new shapes a banner unlike the shipped webps and nothing would report it."""
    from PIL import ImageFont

    global _font
    if _font is None:
        try:
            _font = ImageFont.truetype(render_utils.LABEL_FONT_PATH, render_utils.LABEL_FONT_SIZE)
        except OSError as e:
            raise RuntimeError(
                f"the view label font is missing: {render_utils.LABEL_FONT_PATH}. Install it "
                f"(fonts-dejavu-core) or point utils.LABEL_FONT_PATH at a copy. There is "
                f"deliberately no fallback - one dataset must not carry two kinds of "
                f"review image.") from e
    return _font


def label_frame(img: Image.Image, text: str) -> Image.Image:
    """The view name in white on a translucent black box, centered at the top.

    The box is sized from the font's own measured text width, so a machine whose DejaVu
    or Pillow build measures a word differently places its edges a pixel or two apart
    from the shipped frames. That is banner-only drift; the frame body is unaffected."""
    from PIL import ImageDraw

    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), text, font=_label_font())
    width, height = box[2] - box[0], box[3] - box[1]
    x, y, pad = (img.width - width) // 2, render_utils.LABEL_TOP, render_utils.LABEL_PAD
    draw.rectangle([x - pad, y - pad, x + width + pad, y + height + pad],
                   fill=render_utils.LABEL_BOX_FILL)
    draw.text((x, y), text, fill=render_utils.LABEL_TEXT_FILL, font=_label_font())
    return img


def pack(view_paths, out_path):
    """Six PNGs in, one six-frame webp out. None (and no file) if the set is short or a
    frame cannot be read.

    The paths arrive in `VIEWS` order and keep their positions - a missing view is
    a None in place, never a gap - so frame i is always labeled `VIEW_LABELS[i]`.

    A write failure raises, the way the renderers' does: a full share should stop the run
    rather than count every remaining shape as short. Either way no temp file is left."""
    from PIL import Image

    view_paths = list(view_paths)
    if len(view_paths) != len(render_utils.VIEWS) or any(p is None for p in view_paths):
        return None
    frames = []
    try:
        for path, label in zip(view_paths, render_utils.VIEW_LABELS):
            with Image.open(path) as src:        # closed at once: a run opens a great many files
                frames.append(label_frame(src, label))
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        print(f"[webp] {out_path}: {type(e).__name__}: {e}", flush=True)
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".tmp{os.getpid()}.webp")
    try:
        frames[0].save(tmp, format="WEBP", save_all=True, append_images=frames[1:],
                       duration=FRAME_MS, loop=0, lossless=False, quality=QUALITY)
        os.replace(tmp, out_path)               # atomic: a reader never sees half an orbit
    finally:
        tmp.unlink(missing_ok=True)
    return out_path


def pack_one(preset: str, shard: str, uuid: str):
    """Pack `uuid`'s `preset` orbit from whichever root each view lives in."""
    views = [review_images.orbit_view(preset, v, shard, uuid) for v in render_utils.VIEWS]
    return pack(views, review_images.new_webp(preset, uuid))


def webp_main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m data_processing.render.orbit webp",
                                 description="Pack the six orbit views of every uuid in a list "
                                             "into one animated webp, one preset per invocation.")
    ap.add_argument("--list", required=True, help="uuid per line")
    ap.add_argument("--preset", required=True, choices=list(review_images.ORBIT_PRESETS))
    ap.add_argument("--metadata", default=str(render_utils.DEFAULT_METADATA))
    a = ap.parse_args(argv)

    # Before anything else: a node without the label font must stop here, not pack a
    # whole list with a substitute face or walk it counting every shape as short.
    _label_font()

    # pandas only here: the orbit render ranks import this module and need no parquet reader.
    import pandas as pd

    from utils.metadata_io import read_uuids, shard_of

    uuids = read_uuids(a.list)
    ditem = pd.read_parquet(a.metadata, columns=["ditem_dir"])["ditem_dir"].to_dict()
    packed = skipped = short = 0
    for i, uuid in enumerate(uuids):
        if review_images.webp(a.preset, uuid) is not None:
            skipped += 1                         # resume: already packed in either root
            continue
        dd = ditem.get(uuid)
        if not isinstance(dd, str):
            short += 1
            continue
        if pack_one(a.preset, shard_of(dd), uuid) is None:
            short += 1
        else:
            packed += 1
        if (i + 1) % 500 == 0:
            print(f"[webp {a.preset}] {i + 1}/{len(uuids)} "
                  f"packed={packed} skipped={skipped} short={short}", flush=True)
    print(f"[webp {a.preset}] ALL_DONE packed={packed} skipped={skipped} short={short}",
          flush=True)


def main(argv=None):
    """`webp` is a subcommand; any other argument list renders views."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["webp"]:
        return webp_main(argv[1:])
    return render_main(argv)


if __name__ == "__main__":
    main()
