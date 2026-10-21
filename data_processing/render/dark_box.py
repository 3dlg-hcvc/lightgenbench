"""Images 4 and 5: the dark-box pair, one PNG each per shape, and the shard cut every render
lane works from.

The shape sits in the dark Cornell box lit only by its own emission (`set_true_strength` at
`EMIT_STRENGTH`), then is rendered a second time with that strength at 0, so the difference
between the pair is the emission. The worker runs over a shard file from `dark_box shards`,
skips shapes whose PNGs exist, caps each shape with a SIGALRM timer, and holds a
`<uuid>.inprogress.<rank>` marker during each render so a process death marks the shape
failed on the next launch. It always leaves via `os._exit(0)` after printing ALL_DONE (bpy
can segfault on teardown), so verify by counting files, not exit codes.

    python -u -m data_processing.render.dark_box --shard <shard file> --out <dir> --rank 0

`shards` stripes the shard files for the render lanes: which shape each worker takes, and in
what order. `--list` takes a uuid list, in file order, with nothing dropped. `--export` takes
the annotation tool's /api/export JSON: unverified shapes first, then human-verified, with
auto-verified shapes dropped; striping keeps that order within every worker. GLB paths are
resolved here from the TexVerse metadata parquet, so the render worker needs bpy only. The
default parquet is read through the repo's `data/TexVerse` symlink; without it, pass
`--parquet` and `--glb_root`.

    python -m data_processing.render.dark_box shards --list <uuids> --world_size 16
"""
import argparse
import json
import os
import signal
import sys
import time
import urllib.request


from . import utils as render_utils


# ---------------------------------------------------------------- pure helpers
def parse_shard_line(line):
    uuid, glb = line.rstrip("\n").split("\t")
    return uuid, glb


def out_path(out_root, uuid):
    return os.path.join(out_root, f"{uuid}.png")


def off_path(out_root, uuid):
    """The emission-strength-0 companion (same box, same fill — the delta
    between the pair is exactly what the emission contributes)."""
    return os.path.join(out_root, f"{uuid}_off.png")


def marker_path(guard_dir, uuid, rank):
    return os.path.join(guard_dir, f"{uuid}.inprogress.{rank}")


def adopt_crash_markers(guard_dir, out_root, rank):
    """Uuids whose previous attempt died mid-render. Consumes the markers and
    deletes any partial pair files — a crash can truncate the PNG being
    written, and exists()-based resume would otherwise serve it forever.

    Markers and the failures file live in `guard_dir`, a per-run directory, not in the
    shared output root, so a new run never inherits an older run's failures."""
    suffix = f".inprogress.{rank}"
    crashed = []
    for name in sorted(os.listdir(guard_dir)):
        if name.endswith(suffix):
            uuid = name[: -len(suffix)]
            crashed.append(uuid)
            os.remove(os.path.join(guard_dir, name))
            for p in (out_path(out_root, uuid), off_path(out_root, uuid)):
                if os.path.exists(p):
                    os.remove(p)
    return crashed


def pending_items(pairs, out_root, skip, with_off=True):
    """A shape is done only when every expected output exists."""
    def done(u):
        # Non-empty, not merely present: a refused write can leave a 0-byte PNG.
        def ok(p):
            return os.path.exists(p) and os.path.getsize(p) > 0
        if not ok(out_path(out_root, u)):
            return False
        return not with_off or ok(off_path(out_root, u))
    return [(u, g) for u, g in pairs if u not in skip and not done(u)]


# ------------------------------------------------------------------ bpy stages
class ShapeTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise ShapeTimeout()


def render_one(uuid, glb_path, a, rem, ramp):
    """Fresh scene -> GLB -> auto camera -> dark box -> one still."""
    import bpy

    # This stage's own renderer -- the same helpers single_view.py and orbit.py use -- and its
    # tracked preset blend (render_utils.BLEND_ASSETS_DIR).
    render_utils.load_blend(
        str(render_utils.BLEND_ASSETS_DIR / render_utils.PRESET_BLEND["standard"]))
    render_utils.clear_collection("workbench")
    try:
        obj = render_utils.load_glb(glb_path, import_shading=None)
    except TypeError:
        # bpy 4.0 glTF importer lacks the newer armature kwargs
        obj = render_utils.load_glb(glb_path, import_shading=None,
                                    apply_armature=False)
    rem.drop_to_floor(obj)
    rem.place_camera(obj, a.azimuth, a.elevation)
    facts = rem.analyze(obj)
    floor = bpy.data.objects.get("Floor")
    if floor is not None:
        floor.location.z = -0.004
    ramp.apply_device(a.device)
    rem.set_true_strength(facts, a.strength if a.strength else rem.EMIT_STRENGTH)
    rem.emission_only_box(obj, a.azimuth, wall=a.wall, scale=a.box_scale,
                          height=a.box_height, depth=a.box_depth)
    if a.fill > 0:
        # Faint world fill entering through the box's open face, so a shape
        # whose emission is tiny is still visible as a silhouette. Emission
        # stays the dominant light.
        bg_node = next(n for n in bpy.context.scene.world.node_tree.nodes
                       if n.type == "BACKGROUND")
        bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg_node.inputs[1].default_value = a.fill
    else:
        rem.assert_emission_is_only_light()
    bpy.context.scene.cycles.max_bounces = a.max_bounces
    bpy.context.scene.cycles.diffuse_bounces = a.diffuse_bounces
    rem.add_bloom(size=7, threshold=1.0, mix=-0.45)
    _render_atomic(rem, out_path(a.out, uuid), a)
    if a.off:
        # same scene, emission killed: the pair differs ONLY in emission
        rem.set_true_strength(facts, 0.0)
        _render_atomic(rem, off_path(a.out, uuid), a)


def _render_atomic(rem, path, a):
    """Blender writes the PNG itself, so it goes to a temp name and is renamed into place
    only once whole: a refused write cannot leave a partial PNG that resume takes as done."""
    tmp = f"{path[:-len('.png')]}.tmp{os.getpid()}.png"
    try:
        rem.render(tmp, (a.res, a.res), a.samples, False, a.view_transform, a.exposure)
        render_utils.check_written(tmp)
        if os.path.getsize(tmp) == 0:
            raise RuntimeError(f"{tmp}: Blender wrote an empty PNG")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def render_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m data_processing.render.dark_box",
        description="The dark-box pair over one shard file. "
                    "`dark_box shards -h`: the shard cut.")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="stop after N (look gate)")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--azimuth", type=float, default=38.0)
    ap.add_argument("--elevation", type=float, default=17.0)
    ap.add_argument("--wall", type=float, default=0.80)
    # defaults: the look settled for the review image
    ap.add_argument("--strength", type=float, default=0.0,
                    help="emission strength; 0 = render_emissive's EMIT_STRENGTH (4.0)")
    # Filmic, not AgX: AgX crushes the dark box's shadows and desaturates emissive
    # cores to white. Exposure stays at 0: the ramp videos lift +1.5, but on a
    # review still that reads as not dark enough.
    ap.add_argument("--view_transform", default="Filmic")
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--fill", type=float, default=0.05,
                    help="faint world-fill strength (0 = pure emission-only box)")
    ap.add_argument("--box_scale", type=float, default=1.2)
    ap.add_argument("--box_height", type=float, default=1.3)
    ap.add_argument("--box_depth", type=float, default=1.2)
    ap.add_argument("--off", type=int, default=1,
                    help="also render the emission-strength-0 companion")
    ap.add_argument("--max_bounces", type=int, default=32)
    ap.add_argument("--diffuse_bounces", type=int, default=16)
    ap.add_argument("--timeout_s", type=int, default=120)
    ap.add_argument("--guard_dir", default=None,
                    help="markers and failures file; default <run dir>/guard/box")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    a = ap.parse_args(argv)

    from . import render_emissive as rem      # imports bpy
    from . import render_ramp_videos as ramp

    os.makedirs(a.out, exist_ok=True)
    guard_dir = a.guard_dir or os.path.join(os.path.dirname(os.path.abspath(a.shard)),
                                            os.pardir, "guard", "box")
    os.makedirs(guard_dir, exist_ok=True)
    fail_path = os.path.join(guard_dir, f"failures_rank{a.rank}.txt")
    crashed = adopt_crash_markers(guard_dir, a.out, a.rank)
    if crashed:
        for u in crashed:
            render_utils.append_line(fail_path, f"{u}\tcrashed_process\n")
    skip = set(crashed)
    if os.path.exists(fail_path):
        skip |= {ln.split("\t")[0] for ln in open(fail_path) if ln.strip()}

    pairs = [(u, g)
             for u, g in (parse_shard_line(ln) for ln in open(a.shard) if ln.strip())]
    todo = pending_items(pairs, a.out, skip, with_off=bool(a.off))
    if a.limit:
        todo = todo[: a.limit]
    print(f"rank {a.rank}: {len(todo)} to render of {len(pairs)} in shard",
          flush=True)

    signal.signal(signal.SIGALRM, _alarm)
    done = 0
    for uuid, glb in todo:
        m = marker_path(guard_dir, uuid, a.rank)
        open(m, "w").close()
        t0 = time.time()
        try:
            signal.alarm(a.timeout_s)
            render_one(uuid, glb, a, rem, ramp)
            done += 1
            print(f"OK {uuid} {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            if render_utils.is_disk_full(e):
                # Stop the shard (render_utils.is_disk_full): drop this shape's partial pair
                # and record nothing, so the rerun after freeing space renders it again.
                for p in (out_path(a.out, uuid), off_path(a.out, uuid)):
                    if os.path.exists(p):
                        os.remove(p)
                print(f"STOP {uuid}: the store refused a write ({e}); rerun after freeing space",
                      flush=True)
                raise
            render_utils.append_line(fail_path, f"{uuid}\t{type(e).__name__}: {e}\n")
            print(f"FAIL {uuid} {type(e).__name__}: {e}", flush=True)
        finally:
            signal.alarm(0)
            os.remove(m)
    print(f"ALL_DONE rank={a.rank} done={done}/{len(todo)}", flush=True)
    # bpy teardown segfaults after everything is on disk (render_emissive trap);
    # exit 0 also tells a relaunch loop this pass completed
    os._exit(0)


# ================================================================ the shard cut (`shards`)
DEFAULT_GLB_ROOT = str(render_utils.DEFAULT_GLB_ROOTS[0])
DEFAULT_PARQUET = str(render_utils.DEFAULT_METADATA)
DEFAULT_OUT_DIR = str(render_utils.REVIEW_OUT_DIR / "shards")


def priority_order(items):
    """Unverified first, then human-verified; auto-verified dropped.

    Mirrors store.py's class SQL: auto is exactly verified=1 AND annotator='auto'
    (an empty or NULL annotator with verified=1 is a *human* row). Stable within
    each class.
    """
    remaining, human = [], []
    for it in items:
        if it.get("verified") == 1:
            if it.get("annotator") == "auto":
                continue
            human.append(it["uuid"])
        else:
            remaining.append(it["uuid"])
    return remaining + human


def order_from_list(list_path):
    """Every uuid in the file, in file order, with nothing dropped.

    The DB-export mode drops auto-verified shapes on purpose (pitch black in the box).
    This mode must not: its input is the heuristic stage's pass list, which has already
    removed emission that cannot show.
    """
    with open(list_path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def stripe(shas, world_size, weights=None):
    """Weighted round-robin: rank r gets weights[r] slots per cycle, so faster
    workers (e.g. a 4090 next to a shared A40) get proportionally more
    shapes while the global priority order still interleaves into every shard."""
    if weights is None:
        weights = [1] * world_size
    assert len(weights) == world_size and all(w >= 1 for w in weights)
    schedule = [r for r in range(world_size) for _ in range(weights[r])]
    parts = [[] for _ in range(world_size)]
    for i, s in enumerate(shas):
        parts[schedule[i % len(schedule)]].append(s)
    return parts


def shard_lines(order, glb_by_sha, shard_by_sha=None, with_shard=False):
    """Shard file lines. `with_shard` writes `uuid<TAB>shard<TAB>glb` for the stills and
    orbit workers; without it, `uuid<TAB>glb`, which is what the box worker parses."""
    out = []
    for s in order:
        if with_shard:
            out.append(f"{s}\t{shard_by_sha[s]}\t{glb_by_sha[s]}")
        else:
            out.append(f"{s}\t{glb_by_sha[s]}")
    return out


def write_shards(order, glb_by_sha, world_size, out_dir, weights=None,
                 shard_by_sha=None, with_shard=False, prefix="box"):
    os.makedirs(out_dir, exist_ok=True)
    for r, part in enumerate(stripe(order, world_size, weights)):
        lines = shard_lines(part, glb_by_sha, shard_by_sha, with_shard)
        with open(os.path.join(out_dir, f"{prefix}_rank{r:02d}.txt"), "w") as f:
            f.write("".join(f"{ln}\n" for ln in lines))


def resolve_glbs(parquet_path, shas, glb_root):
    """`(glb path by sha, shard by sha)`. The shard comes from `utils.metadata_io`, not from
    a second copy of the rule here; a sha with no row, no `glb_1k_path` or no `ditem_dir`
    has nothing to render and appears in neither map.

    Both fields are checked for a real string rather than for truth. A missing parquet
    field arrives as float NaN, which is truthy: a NaN `glb_1k_path` would reach
    `os.path.join` and take the whole shard cut down, and a NaN `ditem_dir` would shard
    as the string `"nan"` — silently, with the stills landing under `<root>/nan/<uuid>/`
    where the coverage scan (which keys that shape on an empty shard) never looks.
    """
    import pandas as pd

    from utils.metadata_io import shard_of
    df = pd.read_parquet(parquet_path, columns=["glb_1k_path", "ditem_dir"])
    glbs, shards = {}, {}
    for s in shas:
        if s not in df.index:
            continue
        glb, ditem = df.loc[s, "glb_1k_path"], df.loc[s, "ditem_dir"]
        if not isinstance(glb, str) or not glb:
            continue
        if not isinstance(ditem, str) or not ditem:
            continue
        glbs[s] = os.path.join(glb_root, glb)
        shards[s] = shard_of(ditem)
    return glbs, shards


def shards_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m data_processing.render.dark_box shards",
        description="Stripe shard files for the render lanes: which shape each worker "
                    "takes, and in what order.")
    ap.add_argument("--export", help="path or URL of /api/export JSON (priority order)")
    ap.add_argument("--list", help="uuid list, one per line (nothing dropped)")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--glb_root", default=DEFAULT_GLB_ROOT)
    ap.add_argument("--world_size", type=int, required=True)
    ap.add_argument("--weights", default=None,
                    help="comma-separated per-rank weights (len == world_size)")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default="box", help="shard file name prefix")
    ap.add_argument("--with_shard", action="store_true",
                    help="write uuid<TAB>shard<TAB>glb (stills/orbit) instead of "
                         "uuid<TAB>glb (box)")
    a = ap.parse_args(argv)
    if bool(a.export) == bool(a.list):
        ap.error("pass exactly one of --export or --list")
    if a.world_size < 1:
        ap.error(f"--world_size must be >= 1, got {a.world_size}")
    weights = None
    if a.weights:
        weights = [int(w) for w in a.weights.split(",")]
        if len(weights) != a.world_size or any(w < 1 for w in weights):
            ap.error("--weights must list a positive weight per rank")

    if a.export:
        if a.export.startswith(("http://", "https://")):
            data = json.load(urllib.request.urlopen(a.export))
        else:
            data = json.load(open(a.export))
        order = priority_order(data["items"])
    else:
        order = order_from_list(a.list)
    glbs, shards = resolve_glbs(a.parquet, order, a.glb_root)
    missing = [s for s in order if s not in glbs]
    order = [s for s in order if s in glbs]
    write_shards(order, glbs, a.world_size, a.out_dir, weights,
                 shard_by_sha=shards, with_shard=a.with_shard, prefix=a.prefix)
    print(f"{len(order)} shas -> {a.world_size} shards in {a.out_dir}; "
          f"{len(missing)} dropped (no metadata row, no glb_1k_path or no ditem_dir)")


def main(argv=None):
    """`shards` is a subcommand; any other argument list renders the pair."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["shards"]:
        return shards_main(argv[1:])
    return render_main(argv)


if __name__ == "__main__":
    main()
