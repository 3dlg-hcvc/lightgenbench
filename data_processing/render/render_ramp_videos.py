#!/usr/bin/env python3
"""Emission-strength ramp videos for the user study.

One mp4 per (shape, condition), camera fixed at the shape's solved view, emission
strength ramping 0 -> smax across frames. Conditions of a shape differ ONLY in what
feeds the emission:

  gt               the asset's own emission (set_true_strength)
  texgen           bridged predicted RGB   (<pred_root>/texgen)
  albedo2emission  bridged predicted RGB   (<pred_root>/albedo2emission)
  pbr2emission     bridged predicted RGB   (<pred_root>/pbr2emission)
  mvpaint          bridged predicted RGB   (<pred_root>/mvpaint), the
                   pbr->emission (multiview) Hunyuan3D-Paint baseline
  segvigen         a GLB-carried condition (--glb_conditions): the scene loads
                   <dir>/<sid>_mod.glb whose emission (mask x albedo at unit
                   strength) is baked in, and the ramp drives that emission
                   exactly like gt
  <name in
   --mask_conditions>
                   a bridged predicted MASK (<pred_root>/<name>, one
                   <sid>__mat<N>__emis.png per material), fed through
                   render_emissive's --pred_masks path (albedo x mask) rather
                   than --pred_rgb, for a method whose raw predicted RGB has
                   no usable chroma

Predictions are read already bridged onto each asset's own UVs, <pred_root>/<cond>/;
the bridge and the solved cameras are not part of this repo.

Two treatments (--treatment):
  box (default)  the object inside the Cornell dark room, its emission the
                 ONLY light — the ramp shows the room lighting up
                 (Filmic +1.5, 32/16 bounces, wall 0.80)
  keylit         dark background + key light 8.0, AgX — shows the object's
                 shell; emission reads as glow, not as illumination

Reuses render_emissive.py machinery (import, not copy); renders frames as
PNG and encodes with the system ffmpeg. Verify by counting files, not exit
codes: bpy's teardown segfault trap applies here exactly as it does there.

Run (bpy >= 4.1 for the Khronos PBR Neutral lit still):
  python -m data_processing.render.render_ramp_videos --sids <sid,...> \
      --cameras <dir of <sid>.json> --pred_root <bridged predictions> --out <dir>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

_here = os.path.dirname(os.path.abspath(__file__))
# Run as a script, sys.path[0] is this folder, where `utils` is render/utils.py: put the
# repo root first before importing the repo's utils/.
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "..")))

from utils.paths import TEXVERSE_1K_ROOTS  # noqa: E402

DEFAULT_SIDS = [
    "2b7ace1f2de04e98a5c8874866dc473b",
    "75da9a74403946dda954f08a067e8ad5",
    "e1fbc7943362485489dba5a951ebc4b1",
]
PRED_CONDITIONS = ["texgen", "albedo2emission", "pbr2emission", "mvpaint",
                   "segvigen"]
DEFAULT_GLB_ROOT = str(TEXVERSE_1K_ROOTS[0] / "glbs" / "glbs_1k")


# ---------------------------------------------------------------- pure helpers
def ramp_schedule(frames, smax):
    """Linear emission-strength schedule 0 -> smax inclusive."""
    assert frames >= 2
    return [smax * i / (frames - 1) for i in range(frames)]


def poster_index(schedule, target):
    """Index of the frame whose strength is closest to `target`."""
    return min(range(len(schedule)), key=lambda i: abs(schedule[i] - target))


def out_paths(out_dir, sid, cond):
    return {
        "mp4": os.path.join(out_dir, f"{sid}__{cond}.mp4"),
        "poster": os.path.join(out_dir, f"{sid}__{cond}_poster.png"),
        "json": os.path.join(out_dir, f"{sid}__{cond}.json"),
        "frames_dir": os.path.join(out_dir, "frames", f"{sid}__{cond}"),
    }


def find_glb(glb_root, sid):
    """TexVerse-1K layout: <root>/<shard>/<sid>_1024.glb, shard unknown."""
    import glob as _glob
    hits = _glob.glob(os.path.join(glb_root, "*", f"{sid}_1024.glb"))
    assert len(hits) == 1, f"{sid}: expected exactly one GLB, got {hits}"
    return hits[0]


def encode_cmd(frames_dir, out_path, fps):
    """The ffmpeg argv. `-f mp4` is NOT optional here: the real target is a
    `.mp4.part`, and ffmpeg picks its muxer from the extension -- without the
    explicit format it exits 234 ("unable to find a suitable output format")."""
    return ["ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "f%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-movflags", "+faststart", "-f", "mp4", out_path]


def encode_video(frames_dir, mp4_path, fps):
    """PNG frame sequence -> H.264 yuv420p mp4 (browser-safe).

    Encodes to a .part and renames only on success. The renderer SKIPS any
    shape whose .mp4 already exists, so a truncated file left behind by a
    killed or failed ffmpeg would be treated as a finished video for good --
    and a re-run would never repair it.
    """
    part = mp4_path + ".part"
    cmd = encode_cmd(frames_dir, part, fps)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(part, mp4_path)
    finally:
        if os.path.exists(part):
            os.remove(part)


# ------------------------------------------------------------------ bpy stages
_GPU_BACKEND = None


def enable_gpu_prefs():
    """Configure Cycles GPU compute once per process (OptiX, else CUDA).

    Returns the backend name or None if no GPU device is exposed. Scene-level
    `cycles.device = 'GPU'` still has to be set per scene rebuild — the
    preset load resets scene state, but these preferences persist.
    """
    global _GPU_BACKEND
    if _GPU_BACKEND is not None:
        return _GPU_BACKEND
    import bpy
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        if hasattr(prefs, "refresh_devices"):
            prefs.refresh_devices()
        else:
            prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type != "CPU"]
        if not gpus:
            continue
        for d in prefs.devices:
            d.use = d.type != "CPU"
        _GPU_BACKEND = backend
        print(f"GPU: {backend} on {[d.name for d in gpus]}", flush=True)
        return backend
    print("GPU: no OPTIX/CUDA device exposed, staying on CPU", flush=True)
    return None


def apply_device(device):
    """Point the CURRENT scene at the requested compute device."""
    import bpy
    if device == "gpu" and enable_gpu_prefs():
        bpy.context.scene.cycles.device = "GPU"
    else:
        bpy.context.scene.cycles.device = "CPU"


def build_scene(sid, glb_path, cam_path, rem):
    """Fresh scene + GLB + solved camera. Returns (obj, facts, areas, campos)."""
    import bpy
    from data_processing.render import utils as render_utils

    render_utils.load_blend(rem.PRESET_GLB)
    render_utils.clear_collection("workbench")
    try:
        obj = render_utils.load_glb(glb_path, import_shading=None)
    except TypeError:
        # bpy 4.0 glTF importer lacks the newer armature kwargs load_glb passes
        obj = render_utils.load_glb(glb_path, import_shading=None,
                                    apply_armature=False)
    rem.drop_to_floor(obj)
    cam = json.load(open(cam_path))
    pos = rem.apply_camera(cam["position"], cam["target"])
    facts = rem.analyze(obj)
    areas = rem.material_areas(obj)
    floor = bpy.data.objects.get("Floor")
    if floor is not None:
        floor.location.z = -0.004
    return obj, facts, areas, pos


def set_emission_for_condition(sid, obj, facts, areas, cond, pred_root, rem,
                               glb_carried=False, mask_conditions=frozenset()):
    """Wire the condition's emission. Returns the list of BSDF nodes to ramp."""
    if cond == "gt" or glb_carried:
        # the loaded GLB's own emission IS the signal (the asset's for gt, the
        # baked prediction for a --glb_conditions entry)
        rem.set_true_strength(facts, rem.EMIT_STRENGTH)
    elif cond in mask_conditions:
        # a mask-predicting method (e.g. SegviGen): albedo x mask, not raw RGB
        # -- its predicted color carries no usable chroma
        args = SimpleNamespace(pred_rgb=None,
                               pred_masks=os.path.join(pred_root, cond),
                               pred_emission=None,
                               emit_strength=rem.EMIT_STRENGTH)
        rem.rebuild_from_args(sid, obj, facts, areas, args)
    else:
        args = SimpleNamespace(pred_rgb=os.path.join(pred_root, cond),
                               pred_masks=None, pred_emission=None,
                               emit_strength=rem.EMIT_STRENGTH)
        rem.rebuild_from_args(sid, obj, facts, areas, args)
    ramped = []
    for slot in obj.material_slots:
        bsdf = rem.principled(slot.material)
        if bsdf is not None and \
                bsdf.inputs["Emission Strength"].default_value > 0:
            ramped.append(bsdf)
    return ramped


def render_lit_still(sid, glb_path, cam_path, out_dir, res, samples, rem,
                     device="cpu"):
    """The studio-preset context still, once per shape."""
    import bpy
    obj, facts, areas, pos = build_scene(sid, glb_path, cam_path, rem)
    apply_device(device)
    rem.clear_compositor()
    floor = bpy.data.objects.get("Floor")
    if floor is not None:
        floor.cycles.is_shadow_catcher = True
    rem.render(os.path.join(out_dir, f"{sid}__lit.png"), (res, res),
               samples, True, "Khronos PBR Neutral")


def set_ambient(strength, colour=(1.0, 1.0, 1.0)):
    """Raise the Cornell box's world background off zero by a small amount.

    A pure dark room shows the emission and nothing else -- you cannot tell WHERE on the
    object the glow sits, only that it glows. A weak neutral world puts the silhouette back
    without lighting the scene: at 0.02-0.05 the walls stay near black and the object reads
    as a shape with a glowing region on it.

    The cost is that it lifts the floor of the whole image, so the gap between a faint
    emission and no emission narrows -- which is the very thing the ranking asks people to
    judge. Keep it as low as makes the silhouette legible, and pin one value for the whole
    set: a stimulus graded two ways is not comparable across shapes.
    """
    import bpy
    nt = bpy.context.scene.world.node_tree
    for n in nt.nodes:
        if n.type == "BACKGROUND":
            n.inputs[0].default_value = (colour[0], colour[1], colour[2], 1.0)
            n.inputs[1].default_value = float(strength)


def assert_only_ambient(strength):
    """The emission-only claim is relaxed deliberately -- so check it landed exactly.

    Not a skipped assertion: a stray lamp or a world left at some other strength is still a
    bug. This says "the only non-emission light is the ambient I asked for, at the value I
    asked for", which is a claim the sidecar can carry honestly.
    """
    import bpy
    lights = [o for o in bpy.data.objects if o.type == "LIGHT"]
    assert not lights, f"lights still in the scene: {[o.name for o in lights]}"
    bgs = [n for n in bpy.context.scene.world.node_tree.nodes
           if n.type == "BACKGROUND"]
    assert bgs, "no world background node to carry the ambient"
    for n in bgs:
        got = float(n.inputs[1].default_value)
        # Tolerance has to clear a float32 round-trip: Blender stores the socket as float32,
        # so 0.1 reads back as 0.10000000149011612.
        assert abs(got - strength) < 1e-6, f"world strength {got}, expected {strength}"


def render_condition(sid, cond, glb_path, cam_path, a, rem):
    p = out_paths(a.out, sid, cond)
    if os.path.exists(p["mp4"]) and not a.overwrite:
        print(f"SKIP {sid} {cond}", flush=True)
        return
    import bpy
    obj, facts, areas, pos = build_scene(sid, glb_path, cam_path, rem)
    apply_device(a.device)
    ramped = set_emission_for_condition(sid, obj, facts, areas, cond,
                                        a.pred_root, rem,
                                        glb_carried=cond in a.glb_map,
                                        mask_conditions=a.mask_set)
    if a.treatment == "box":
        rem.emission_only_box(obj, a.azimuth, wall=a.wall, scale=a.box_scale,
                              height=a.box_height, depth=a.box_depth)
        if a.ambient > 0:
            set_ambient(a.ambient)
            assert_only_ambient(a.ambient)
        else:
            rem.assert_emission_is_only_light()
        bpy.context.scene.cycles.max_bounces = a.max_bounces
        bpy.context.scene.cycles.diffuse_bounces = a.diffuse_bounces
    else:
        rem.dark_room(bg=a.bg, key=a.key)
        floor = bpy.data.objects.get("Floor")
        if floor is not None:
            floor.location.z = -0.004
    rem.add_bloom(size=a.bloom_size, threshold=a.bloom_threshold,
                  mix=a.bloom_mix)
    schedule = ramp_schedule(a.frames, a.smax)
    os.makedirs(p["frames_dir"], exist_ok=True)
    for i, s in enumerate(schedule):
        for bsdf in ramped:
            bsdf.inputs["Emission Strength"].default_value = s
        rem.render(os.path.join(p["frames_dir"], f"f{i:03d}.png"),
                   (a.res, a.res), a.samples, False, a.view_transform,
                   a.exposure)
    encode_video(p["frames_dir"], p["mp4"], a.fps)
    shutil.copyfile(
        os.path.join(p["frames_dir"],
                     f"f{poster_index(schedule, rem.EMIT_STRENGTH):03d}.png"),
        p["poster"])
    sidecar = {"sid": sid, "condition": cond, "n_ramped_materials": len(ramped),
               "glb": glb_path, "treatment": a.treatment,
               "frames": a.frames, "smax": a.smax, "fps": a.fps,
               "res": a.res, "samples": a.samples, "bg": a.bg, "key": a.key,
               "view_transform": a.view_transform, "exposure": a.exposure,
               "ambient": a.ambient,
               "mask_condition": cond in a.mask_set,
               "wall": a.wall, "max_bounces": a.max_bounces,
               "diffuse_bounces": a.diffuse_bounces,
               "bloom": {"size": a.bloom_size, "threshold": a.bloom_threshold,
                         "mix": a.bloom_mix},
               "camera": json.load(open(cam_path))}
    with open(p["json"], "w") as f:
        json.dump(sidecar, f, indent=1)
    if not a.keep_frames:
        shutil.rmtree(p["frames_dir"])
    print(f"OK {sid} {cond} ramped={len(ramped)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", default=",".join(DEFAULT_SIDS))
    ap.add_argument("--conditions", default="gt," + ",".join(PRED_CONDITIONS))
    ap.add_argument("--glb_root", default=DEFAULT_GLB_ROOT)
    ap.add_argument("--pred_root", default=None,
                    help="bridged predictions, <pred_root>/<cond>/ per condition; "
                         "needed for every condition but gt")
    ap.add_argument("--cameras", required=True,
                    help="the solved cameras: <sid>.json per shape, with position and target")
    ap.add_argument("--out", required=True)
    ap.add_argument("--treatment", default="box", choices=["box", "keylit"])
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--smax", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--res", type=int, default=None,
                    help="default: 512 (box) / 640 (keylit)")
    ap.add_argument("--samples", type=int, default=None,
                    help="default: 128 (box; emission-only needs more) / 96")
    ap.add_argument("--samples_lit", type=int, default=96)
    ap.add_argument("--bg", type=float, default=0.004)
    ap.add_argument("--key", type=float, default=8.0)
    ap.add_argument("--view_transform", default=None,
                    help="default: Filmic (box) / AgX (keylit)")
    ap.add_argument("--exposure", type=float, default=None,
                    help="default: +1.5 (box) / 0.0 (keylit)")
    ap.add_argument("--azimuth", type=float, default=38.0,
                    help="box rotation; must match the solved cameras' azimuth")
    ap.add_argument("--ambient", type=float, default=0.0,
                    help="box only: weak neutral world light so the silhouette is "
                         "visible where the emission is tiny (try 0.02-0.05). 0 keeps "
                         "the strict emission-is-the-only-light room")
    ap.add_argument("--wall", type=float, default=0.80)
    ap.add_argument("--box_scale", type=float, default=2.0)
    ap.add_argument("--box_height", type=float, default=1.7)
    ap.add_argument("--box_depth", type=float, default=1.8)
    ap.add_argument("--max_bounces", type=int, default=32)
    ap.add_argument("--diffuse_bounces", type=int, default=16)
    ap.add_argument("--bloom_size", type=int, default=7)
    ap.add_argument("--bloom_threshold", type=float, default=1.0)
    ap.add_argument("--bloom_mix", type=float, default=-0.45)
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--keep_frames", type=int, default=0)
    ap.add_argument("--skip_lit", type=int, default=0)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"],
                    help="gpu = Cycles OptiX/CUDA on the local card, with "
                         "silent CPU fallback when no device is exposed")
    ap.add_argument("--glb_conditions", default="",
                    help="comma-separated name=dir pairs; such a condition "
                         "loads <dir>/<sid>_mod.glb (emission baked at unit "
                         "strength) instead of the source GLB and ramps its "
                         "own emission")
    ap.add_argument("--mask_conditions", default="",
                    help="comma-separated condition names that go through "
                         "render_emissive's --pred_masks path (albedo x "
                         "predicted mask) instead of --pred_rgb -- for a "
                         "method (e.g. SegviGen) whose predicted RGB carries "
                         "no usable chroma. Not the same as --glb_conditions: "
                         "these still read <pred_root>/<cond>, just as a "
                         "mask directory instead of an RGB one")
    a = ap.parse_args()
    a.glb_map = dict(kv.split("=", 1)
                     for kv in a.glb_conditions.split(",") if kv)
    a.mask_set = set(c for c in a.mask_conditions.split(",") if c)
    if a.pred_root is None and any(c != "gt" and c not in a.glb_map for c in a.conditions.split(",") if c):
        ap.error("--pred_root is required for a prediction condition")

    # per-treatment defaults; the sidecar records what was actually used
    box = a.treatment == "box"
    if a.res is None:
        a.res = 512 if box else 640
    if a.samples is None:
        a.samples = 128 if box else 96
    if a.view_transform is None:
        a.view_transform = "Filmic" if box else "AgX"
    if a.exposure is None:
        a.exposure = 1.5 if box else 0.0

    from data_processing.render import render_emissive as rem

    os.makedirs(a.out, exist_ok=True)
    failed = []
    for sid in a.sids.split(","):
        glb_path = find_glb(a.glb_root, sid)
        cam_path = os.path.join(a.cameras, f"{sid}.json")
        assert os.path.exists(cam_path), f"no solved camera for {sid}"
        lit = os.path.join(a.out, f"{sid}__lit.png")
        try:
            if not a.skip_lit and (a.overwrite or not os.path.exists(lit)):
                render_lit_still(sid, glb_path, cam_path, a.out, a.res,
                                 a.samples_lit, rem, device=a.device)
            for cond in a.conditions.split(","):
                if cond in a.glb_map:
                    cond_glb = os.path.join(a.glb_map[cond],
                                            f"{sid}_mod.glb")
                    assert os.path.exists(cond_glb), cond_glb
                else:
                    cond_glb = glb_path
                render_condition(sid, cond, cond_glb, cam_path, a, rem)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"FAIL {sid}", flush=True)
            failed.append(sid)
    print("ALL_DONE", flush=True)
    # same teardown-segfault workaround as render_emissive.py: everything is
    # on disk by here; carry real failures out in the exit status only
    if failed:
        print(f"FAILED_SHAPES {len(failed)}: {' '.join(failed)}", flush=True)
        os._exit(1)
    os._exit(0)


if __name__ == "__main__":
    main()
