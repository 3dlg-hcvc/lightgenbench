#!/usr/bin/env python3
"""Render TexVerse GLBs in a dark room with their emission rebuilt as mask x albedo, or
with a model's prediction (--pred_masks, --pred_rgb, --pred_emission) in its place.

For every material of a shape:
  mask   = the shape's OWN emissive channel, binarized (any nonzero emission)
  albedo = whatever feeds the Principled BSDF's Base Color
  new emission color = mask * albedo, at one fixed emission strength for all shapes

The mask is ground truth, read off the source asset, not a model prediction.

Outputs per sid, under --out:
  <sid>_glow.png   dark room, emission = mask x albedo, bloom via the Glare node
  <sid>_lit.png    the same camera under the studio preset (verification only)
  <sid>_true.png   dark room, the asset's ORIGINAL emission (the comparison)
  <sid>_mod.glb    the modified asset, for the web viewer
  <sid>_stats.json per-material mask coverage and the mask x albedo vs true error

CARE: Blender's Principled BSDF defaults Emission Color to white with Emission
Strength 0, so "is this material emissive" must test the STRENGTH as well as the
color. Reading the two independently, and reading both before anything is
modified, is what keeps a non-emissive body from being turned into a lamp.

Runs headless with the bpy pip package:
    python -m data_processing.render.render_emissive --manifest <rows.json> \
        --glb_dir <dir of <sid>.glb> --out <dir> [--mode method|random|allemissive|box]
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np

import bpy  # noqa: E402

# The scene and the GLB loader are `utils.py` beside this file, with the preset blends in
# `assets/`. Run as a script, the repo root is not on sys.path, so it is added here, at the
# END: nothing a caller imports is shadowed.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.append(_REPO)
from data_processing.render import utils as render_utils  # noqa: E402

#: The lit studio preset every render here starts from (the review lanes' `standard`).
PRESET_GLB = str(render_utils.BLEND_ASSETS_DIR / render_utils.PRESET_BLEND["standard"])

# Any emission at all counts, matching the dataset's >1/255 rule. Blender image
# pixels are linear float; 1/255 in sRGB is about 5.7e-6 linear, so this floor
# sits just above it and still admits the faintest authored emission.
LIN_EPS = 1e-5

# One strength for every shape on the page, so no shape is flattered by a
# per-shape exposure choice, and so the true / mask x albedo pair differ only in
# color.
#
# THIS IS A LOOK CHOICE, NOT A MEASUREMENT. The bake stores emission as uint8
# and drops the KHR_materials_emissive_strength extension, which some source GLBs
# carry, so nothing in the data says how brightly a surface emits.
# Overridable with --emit_strength.
EMIT_STRENGTH = 4.0


# ------------------------------------------------------------------ helpers
def principled(mat):
    if not mat or not mat.use_nodes:
        return None
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    return None


def upstream_image(sock):
    """First TEX_IMAGE node upstream of `sock`, breadth first, or None."""
    if not sock.is_linked:
        return None
    queue, seen = [sock.links[0].from_node], set()
    while queue:
        n = queue.pop(0)
        if n.name in seen:
            continue
        seen.add(n.name)
        if n.type == "TEX_IMAGE" and n.image is not None:
            return n
        for i in n.inputs:
            for l in i.links:
                queue.append(l.from_node)
    return None


def img_array(image):
    """(H, W, 4) float32, top row first. Blender stores pixels bottom-up."""
    w, h = image.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)[::-1]


def new_image(name, arr):
    h, w = arr.shape[:2]
    im = bpy.data.images.new(name, width=w, height=h, alpha=True, float_buffer=True)
    im.colorspace_settings.name = "Non-Color"   # the array is already linear
    im.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).ravel().astype(np.float32))
    im.pack()
    return im


def resize_to(arr, hw):
    """Nearest-neighbor resample to (H, W); no scipy, no PIL dependency."""
    H, W = hw
    h, w = arr.shape[:2]
    if (h, w) == (H, W):
        return arr
    yi = (np.arange(H) * (h / H)).astype(np.int64).clip(0, h - 1)
    xi = (np.arange(W) * (w / W)).astype(np.int64).clip(0, w - 1)
    return arr[yi][:, xi]


def socket_rgb(sock):
    return np.array(sock.default_value[:3], dtype=np.float32)


def material_areas(obj):
    """World-space surface area per material slot index."""
    areas = np.zeros(max(len(obj.material_slots), 1), dtype=np.float64)
    mw = obj.matrix_world
    me = obj.data
    for poly in me.polygons:
        idx = min(poly.material_index, len(areas) - 1)
        areas[idx] += poly.area
    _ = mw
    return areas


# --------------------------------------------------------------- inspection
def analyze(obj):
    """Read every material's emission BEFORE anything is modified.

    Returns [{slot, mat, bsdf, e_node, e_const, strength, emits}].
    """
    facts = []
    for i, slot in enumerate(obj.material_slots):
        mat = slot.material
        bsdf = principled(mat)
        if bsdf is None:
            facts.append(dict(slot=i, mat=mat, bsdf=None, emits=False))
            continue
        e_sock = bsdf.inputs["Emission Color"]
        s_sock = bsdf.inputs["Emission Strength"]
        e_node = upstream_image(e_sock)
        e_const = socket_rgb(e_sock)
        strength = 1.0 if s_sock.is_linked else float(s_sock.default_value)
        # a texture is only emissive if it actually carries nonzero pixels
        tex_max = float(img_array(e_node.image)[..., :3].max()) if e_node else 0.0
        emits = strength > 0 and (
            (e_node is not None and tex_max > LIN_EPS and float(e_const.max()) > LIN_EPS)
            or (e_node is None and float(e_const.max()) > LIN_EPS))
        facts.append(dict(slot=i, mat=mat, bsdf=bsdf, e_node=e_node,
                          e_const=e_const, strength=strength, emits=emits,
                          tex_max=tex_max))
    return facts


def set_true_strength(facts, value):
    """Give every genuinely emissive material one common strength, so the true
    render and the mask x albedo render differ in color and nothing else."""
    for f in facts:
        if f.get("emits"):
            f["bsdf"].inputs["Emission Strength"].default_value = value


def measure_lit_fraction(facts, areas):
    """Share of surface area the ground-truth mask turns on, WITHOUT modifying
    anything. The baselines need this number before the materials are rewritten,
    so a random mask can be drawn at the shape's own emissive density rather than
    at some arbitrary one."""
    total = lit = 0.0
    for f in facts:
        area = float(areas[f["slot"]]) if f["slot"] < len(areas) else 0.0
        total += area
        if not f.get("emits"):
            continue
        e_node, e_const = f.get("e_node"), f.get("e_const")
        if e_node is None:
            lit += area                      # uniform emitter: all of it
        else:
            e_arr = img_array(e_node.image)[..., :3] * e_const
            lit += area * float((e_arr.max(axis=-1) > LIN_EPS).mean())
    return lit / total if total else 0.0


def blocky_mask(shape, density, blocks, rng):
    """A random binary mask at `density`, coherent over `blocks` cells per side.

    Per-texel noise would render as an even dusting over the whole object, which
    reads as a dim uniform glow rather than as a wrong REGION. Block noise puts
    the guess in patches, so the failure the baseline is meant to show, emission
    in the wrong place, is the thing the eye actually sees.
    """
    H, W = shape
    field = rng.random((blocks, blocks))
    thresh = np.quantile(field, 1.0 - density) if 0 < density < 1 else (
        -1.0 if density >= 1 else 2.0)
    small = (field > thresh).astype(np.float32)
    yi = (np.arange(H) * (blocks / H)).astype(np.int64).clip(0, blocks - 1)
    xi = (np.arange(W) * (blocks / W)).astype(np.int64).clip(0, blocks - 1)
    return small[yi][:, xi]


def rebuild_emission_baseline(obj, facts, areas, mode, rng, blocks=28,
                              strength=EMIT_STRENGTH,
                              tex_size=512):
    """The two dummy baselines, both applied to EVERY material of the shape.

    A baseline that only touched the materials that happen to carry an emissive
    texture would already know the answer it is supposed to be guessing, so both
    of these are free to put emission anywhere on the object.

      random        a random mask at the shape's own emissive density, times albedo
      allemissive   every surface texel emissive, so emission is just the albedo
    """
    density = measure_lit_fraction(facts, areas) if mode == "random" else 1.0
    for f in facts:
        bsdf = f.get("bsdf")
        if bsdf is None:
            continue
        mat = f["mat"]
        nt = mat.node_tree
        e_sock = bsdf.inputs["Emission Color"]
        b_sock = bsdf.inputs["Base Color"]
        b_node = upstream_image(b_sock)
        b_const = socket_rgb(b_sock)
        for l in list(e_sock.links):
            nt.links.remove(l)

        if mode == "allemissive":
            if b_node is not None:
                nt.links.new(b_node.outputs["Color"], e_sock)
                e_sock.default_value = (1, 1, 1, 1)
            else:
                e_sock.default_value = (*b_const, 1.0)
            bsdf.inputs["Emission Strength"].default_value = strength
            continue

        if b_node is not None:
            alb = img_array(b_node.image)[..., :3] * b_const
        else:
            alb = np.broadcast_to(b_const, (tex_size, tex_size, 3))
        mask = blocky_mask(alb.shape[:2], density, blocks, rng)
        out = np.empty((*mask.shape, 4), dtype=np.float32)
        out[..., :3] = alb * mask[..., None]
        out[..., 3] = 1.0
        im = new_image(f"rnd_{mat.name}", out)
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = im
        if b_node is not None and b_node.inputs["Vector"].is_linked:
            nt.links.new(b_node.inputs["Vector"].links[0].from_socket,
                         node.inputs["Vector"])
        nt.links.new(node.outputs["Color"], e_sock)
        e_sock.default_value = (1, 1, 1, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
    return density


# ------------------------------------------------- predicted-mask rewrite
def load_pred(pred_dir, sid, n_slots):
    """Read a model's predicted mask for one shape.

    Per material slot, either a texel mask (`<sid>__mat<N>__emis.png`, any
    resolution, nonzero means emissive) or, for a slot whose primitives carry no
    UVs and therefore cannot hold a texture, a scalar in
    `<sid>__stats.json` under "uniform" keyed by slot index.

    A slot with neither is a slot the model predicted nothing on.
    """
    import glob as _glob
    masks, uniform = {}, {}
    stats_path = os.path.join(pred_dir, f"{sid}__stats.json")
    if os.path.exists(stats_path):
        st = json.load(open(stats_path))
        for k, v in (st.get("uniform") or {}).items():
            uniform[int(k)] = float(v)
    for p in _glob.glob(os.path.join(pred_dir, f"{sid}__mat*__emis.png")):
        n = int(os.path.basename(p).split("__mat")[1].split("__")[0])
        import matplotlib.image as mpimg
        a = mpimg.imread(p)
        masks[n] = (a[..., :3].max(axis=-1) > 0.5).astype(np.float32)
    unknown = [n for n in list(masks) + list(uniform) if n >= n_slots]
    assert not unknown, (f"{sid}: prediction names material slots {unknown} but "
                         f"the asset has {n_slots}; slot indices must be the "
                         f"asset's own material order")
    # A range check is NOT enough: files keyed by glTF material index instead
    # of slot index can pass it and put the masks on the WRONG materials. The
    # names are the only thing that pins the mapping, so a prediction that
    # cannot be checked against them is refused.
    names = (json.load(open(stats_path)).get("materials")
             if os.path.exists(stats_path) else None)
    assert names, (
        f"{sid}: {os.path.basename(stats_path)} carries no 'materials' list, so "
        f"the slot keying cannot be verified. Emit one name per slot, in the "
        f"asset's material order.")
    if isinstance(names[0], dict):
        names = [m.get("material") for m in sorted(names, key=lambda m: m["slot"])]
    return masks, uniform, names


def _read_emisrgb(path):
    """Linear float32 (H,W,3) from a bridged `__emisrgb.png`, 8- or 16-bit.

    matplotlib/PIL silently collapse a 16-bit RGB PNG to 8 bits, which would
    re-impose the 1/510 floor the 16-bit file exists to remove, so the 16-bit
    case goes through cv2 (which returns BGR).
    """
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        import matplotlib.image as mpimg
        a = mpimg.imread(path)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=-1)
        return a[..., :3].astype(np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    a = a[..., :3][..., ::-1]  # BGR(A) -> RGB
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    return np.ascontiguousarray(a).astype(np.float32) / scale


def load_pred_rgb(pred_dir, sid, n_slots):
    """Read a model's predicted emission RGB for one shape.

    Same directory discipline as load_pred, but the per-slot texture carries
    the predicted emission COLOR (linear, bridged onto the asset's UVs), not
    a binary mask: `<sid>__mat<N>__emisrgb.png`
    per material slot, and for a UV-less slot an [r, g, b] triple in
    `<sid>__stats.json` under "uniform" keyed by slot index. A slot with
    neither is a slot the model predicted nothing on.
    """
    import glob as _glob
    rgbs, uniform = {}, {}
    stats_path = os.path.join(pred_dir, f"{sid}__stats.json")
    if os.path.exists(stats_path):
        st = json.load(open(stats_path))
        assert st.get("slot_match_ok", True), (
            f"{sid}: the bridge could not verify this shape's material-slot "
            f"mapping (slot_match_ok false); rendering it would guess")
        for k, v in (st.get("uniform") or {}).items():
            uniform[int(k)] = np.asarray(v, dtype=np.float32)[:3]
    for p in _glob.glob(os.path.join(pred_dir, f"{sid}__mat*__emisrgb.png")):
        n = int(os.path.basename(p).split("__mat")[1].split("__")[0])
        rgbs[n] = _read_emisrgb(p)
    unknown = [n for n in list(rgbs) + list(uniform) if n >= n_slots]
    assert not unknown, (f"{sid}: prediction names material slots {unknown} but "
                         f"the asset has {n_slots}; slot indices must be the "
                         f"asset's own material order")
    # the slot-name verification is identical to load_pred's, and for the same
    # reason: a range check cannot catch glTF-material-index keying
    names = (json.load(open(stats_path)).get("materials")
             if os.path.exists(stats_path) else None)
    assert names, (
        f"{sid}: {os.path.basename(stats_path)} carries no 'materials' list, so "
        f"the slot keying cannot be verified. Emit one name per slot, in the "
        f"asset's material order.")
    if isinstance(names[0], dict):
        names = [m.get("material") for m in sorted(names, key=lambda m: m["slot"])]
    return rgbs, uniform, names


def rebuild_emission_predicted_rgb(obj, facts, areas, rgbs, uniform,
                                    strength=EMIT_STRENGTH):
    """Emission = the PREDICTED RGB itself, on the source asset.

    The counterpart of rebuild_emission_predicted for models that predict
    emission color directly (TEXGen, the TRELLIS.2 emission DiTs, the
    multiview baseline): no albedo multiply — that multiply is the mask
    method's own assumption, and pushing a predicted RGB through it would
    render something neither method predicted. Selection semantics are
    unchanged: every material is considered, and a material the model did not
    select is switched OFF even if the asset emits there.

    The predicted texture is wired through the BASE COLOR's UV chain (not the
    emissive chain): the bake that defined every model's prediction space
    samples all source textures with TEXCOORD_0, so the bridged texture lives
    in that parameterization.
    """
    stats = []
    for f in facts:
        mat, bsdf = f["mat"], f.get("bsdf")
        slot = f["slot"]
        area = float(areas[slot]) if slot < len(areas) else 0.0
        st = {"material": mat.name if mat else None, "area": area,
              "gt_emits": bool(f.get("emits")), "mask_frac": 0.0,
              "source": None}
        if bsdf is None:
            stats.append(st)
            continue

        nt = mat.node_tree
        e_sock = bsdf.inputs["Emission Color"]
        b_sock = bsdf.inputs["Base Color"]
        for l in list(e_sock.links):
            nt.links.remove(l)

        rgb = rgbs.get(slot)
        const = uniform.get(slot)
        if rgb is None and const is None:
            e_sock.default_value = (0, 0, 0, 1)
            bsdf.inputs["Emission Strength"].default_value = 0.0
            stats.append(st)
            continue

        if rgb is None:
            st["source"] = "uniform"
            on = float(const.max()) > LIN_EPS
            st["mask_frac"] = 1.0 if on else 0.0
            st["uniform_rgb"] = [float(x) for x in const]
            if not on:
                e_sock.default_value = (0, 0, 0, 1)
                bsdf.inputs["Emission Strength"].default_value = 0.0
                stats.append(st)
                continue
            e_sock.default_value = (*[float(x) for x in const], 1.0)
            bsdf.inputs["Emission Strength"].default_value = strength
            stats.append(st)
            continue

        st["source"] = "texture"
        st["mask_frac"] = float((rgb.max(axis=-1) > LIN_EPS).mean())
        if st["mask_frac"] <= 0:
            e_sock.default_value = (0, 0, 0, 1)
            bsdf.inputs["Emission Strength"].default_value = 0.0
            stats.append(st)
            continue
        out = np.empty((*rgb.shape[:2], 4), dtype=np.float32)
        out[..., :3] = rgb
        out[..., 3] = 1.0
        im = new_image(f"predrgb_{mat.name}", out)
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = im
        b_node = upstream_image(b_sock)
        donor = b_node or f.get("e_node")
        if donor is not None and donor.inputs["Vector"].is_linked:
            nt.links.new(donor.inputs["Vector"].links[0].from_socket,
                         node.inputs["Vector"])
        nt.links.new(node.outputs["Color"], e_sock)
        e_sock.default_value = (1, 1, 1, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
        stats.append(st)
    return stats


def load_emission(pred_dir, sid, n_slots):
    """Read a method's predicted emission COLOR for one shape.

    Same file convention and the same slot-name contract as load_pred, and
    deliberately so: one naming scheme, one verification, two payloads. The
    difference is what the pixels mean. Here they are the emitted color itself,
    kept as RGB, not a mask to multiply the asset's albedo by.

    A uniform entry is a per-slot RGB triple rather than a scalar, since a
    material with no UVs still has to say what color it emits.
    """
    import glob as _glob
    import matplotlib.image as mpimg
    emis, uniform = {}, {}
    stats_path = os.path.join(pred_dir, f"{sid}__stats.json")
    st = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    for k, v in (st.get("uniform") or {}).items():
        v = [float(v)] * 3 if np.isscalar(v) else [float(x) for x in v][:3]
        uniform[int(k)] = np.array(v, dtype=np.float32)
    for p in _glob.glob(os.path.join(pred_dir, f"{sid}__mat*__emis.png")):
        n = int(os.path.basename(p).split("__mat")[1].split("__")[0])
        a = mpimg.imread(p)
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        # sRGB in, linear out: an 8-bit PNG of emission is almost always written
        # through an sRGB transfer curve, and feeding it to Blender as linear
        # would darken the midtones by roughly a factor of two. If your files are
        # already linear, say so with --emission_linear.
        emis[n] = a[..., :3].astype(np.float32)
    unknown = [n for n in list(emis) + list(uniform) if n >= n_slots]
    assert not unknown, (f"{sid}: prediction names material slots {unknown} but "
                         f"the asset has {n_slots}; slot indices must be the "
                         f"asset's own material order")
    names = st.get("materials")
    assert names, (
        f"{sid}: {os.path.basename(stats_path)} carries no 'materials' list, so "
        f"the slot keying cannot be verified. Emit one name per slot, in the "
        f"asset's material order.")
    if isinstance(names[0], dict):
        names = [m.get("material") for m in sorted(names, key=lambda m: m["slot"])]
    return emis, uniform, names


def srgb_to_linear(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def rebuild_emission_direct(obj, facts, areas, emis, uniform, strength=1.0,
                            assume_linear=False):
    """Emission = the method's OWN predicted color, written in unchanged.

    This is the path for a method that predicts emission RGB. It shares the
    scene, camera, tone and bloom with every other mode and differs in one
    thing: there is no multiply by the asset's albedo. That multiply is our
    method's central assumption, so applying it to someone else's prediction
    would render something neither method produced.

    Strength defaults to 1.0 here, NOT to our 4.0, because a predicted color
    may already encode radiance and multiplying it is a second opinion about
    brightness. Pass --emit_strength to match our panels' look, knowing that is
    what you are doing.
    """
    stats = []
    for f in facts:
        mat, bsdf, slot = f["mat"], f.get("bsdf"), f["slot"]
        area = float(areas[slot]) if slot < len(areas) else 0.0
        st = {"material": mat.name if mat else None, "area": area,
              "gt_emits": bool(f.get("emits")), "lit_frac": 0.0, "source": None}
        if bsdf is None:
            stats.append(st)
            continue
        nt = mat.node_tree
        e_sock = bsdf.inputs["Emission Color"]
        for l in list(e_sock.links):
            nt.links.remove(l)

        tex, const = emis.get(slot), uniform.get(slot)
        if tex is None and const is None:
            # not predicted: dark, whatever the asset itself does, so a miss
            # reads as a miss rather than as the asset showing through
            e_sock.default_value = (0, 0, 0, 1)
            bsdf.inputs["Emission Strength"].default_value = 0.0
            stats.append(st)
            continue

        if tex is None:
            rgb = const if assume_linear else srgb_to_linear(const)
            st["source"] = "uniform"
            st["lit_frac"] = 1.0 if float(np.max(rgb)) > LIN_EPS else 0.0
            e_sock.default_value = (*[float(x) for x in rgb], 1.0)
            bsdf.inputs["Emission Strength"].default_value = (
                strength if st["lit_frac"] else 0.0)
            stats.append(st)
            continue

        rgb = tex if assume_linear else srgb_to_linear(tex)
        st["source"] = "texture"
        st["lit_frac"] = float((rgb.max(axis=-1) > LIN_EPS).mean())
        if st["lit_frac"] <= 0:
            e_sock.default_value = (0, 0, 0, 1)
            bsdf.inputs["Emission Strength"].default_value = 0.0
            stats.append(st)
            continue
        out = np.empty((*rgb.shape[:2], 4), dtype=np.float32)
        out[..., :3] = rgb
        out[..., 3] = 1.0
        im = new_image(f"emis_{mat.name}", out)
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = im
        # follow the asset's own UV wiring, so KHR_texture_transform and
        # non-default UV sets carry over exactly as they do on the mask path
        donor = f.get("e_node") or upstream_image(bsdf.inputs["Base Color"])
        if donor is not None and donor.inputs["Vector"].is_linked:
            nt.links.new(donor.inputs["Vector"].links[0].from_socket,
                         node.inputs["Vector"])
        nt.links.new(node.outputs["Color"], e_sock)
        e_sock.default_value = (1, 1, 1, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
        stats.append(st)
    return stats


def assert_slot_names(sid, obj, names):
    """Refuse a prediction whose material order is not the asset's."""
    mine = [s.material.name if s.material else None for s in obj.material_slots]
    assert len(names) == len(mine), (
        f"{sid}: prediction lists {len(names)} materials, asset has {len(mine)}")
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(names, mine)) if a != b]
    assert not bad, (
        f"{sid}: material order differs from the asset's at slots "
        f"{[i for i, _, _ in bad]} (prediction, asset): "
        f"{[(a, b) for _, a, b in bad]}")


def rebuild_emission_predicted(obj, facts, areas, masks, uniform,
                               strength=EMIT_STRENGTH):
    """Emission = PREDICTED mask x albedo, on the source asset.

    Kept separate from rebuild_emission() so changes to the predicted path never
    touch the ground-truth one. Two differences from that path, both
    required for a predicted column to be honest:

    - EVERY material is considered, not only the ones emissive in the asset. A
      model may predict emission anywhere, and restricting it to the materials
      that happen to be emissive in the ground truth would hand it the answer.
    - A material the model did NOT select is switched OFF even if the asset
      emits there. Otherwise the asset's own emission leaks into the column and
      the panel shows ground truth wearing a prediction's label.
    """
    stats = []
    for f in facts:
        mat, bsdf = f["mat"], f.get("bsdf")
        slot = f["slot"]
        area = float(areas[slot]) if slot < len(areas) else 0.0
        st = {"material": mat.name if mat else None, "area": area,
              "gt_emits": bool(f.get("emits")), "mask_frac": 0.0,
              "source": None}
        if bsdf is None:
            stats.append(st)
            continue

        nt = mat.node_tree
        e_sock = bsdf.inputs["Emission Color"]
        b_sock = bsdf.inputs["Base Color"]
        for l in list(e_sock.links):
            nt.links.remove(l)

        mask = masks.get(slot)
        scalar = uniform.get(slot)
        if mask is None and scalar is None:
            # not selected: switch this material off, whatever the asset does
            e_sock.default_value = (0, 0, 0, 1)
            bsdf.inputs["Emission Strength"].default_value = 0.0
            stats.append(st)
            continue

        b_node = upstream_image(b_sock)
        b_const = socket_rgb(b_sock)

        if mask is None:
            # a UV-less material can only answer "does it fire", so it fires or
            # it does not; the ground-truth column renders this same material as
            # a uniform emitter, so both columns use one representation here
            st["source"] = "uniform"
            on = scalar >= 0.5
            st["mask_frac"] = 1.0 if on else 0.0
            st["uniform_value"] = scalar
            if not on:
                e_sock.default_value = (0, 0, 0, 1)
                bsdf.inputs["Emission Strength"].default_value = 0.0
                stats.append(st)
                continue
            if b_node is not None:
                nt.links.new(b_node.outputs["Color"], e_sock)
                e_sock.default_value = (1, 1, 1, 1)
            else:
                e_sock.default_value = (*b_const, 1.0)
            bsdf.inputs["Emission Strength"].default_value = strength
            stats.append(st)
            continue

        st["source"] = "texture"
        st["mask_frac"] = float(mask.mean())
        if st["mask_frac"] <= 0:
            e_sock.default_value = (0, 0, 0, 1)
            bsdf.inputs["Emission Strength"].default_value = 0.0
            stats.append(st)
            continue
        if b_node is not None:
            alb = resize_to(img_array(b_node.image), mask.shape)[..., :3] * b_const
        else:
            alb = np.broadcast_to(b_const, (*mask.shape, 3))
        out = np.empty((*mask.shape, 4), dtype=np.float32)
        out[..., :3] = alb * mask[..., None]
        out[..., 3] = 1.0
        im = new_image(f"pred_{mat.name}", out)
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = im
        # follow the asset's own UV wiring so KHR_texture_transform and
        # non-default UV sets carry over: the emissive chain if the material has
        # one, else the base color's, else leave it to the active UV map
        donor = f.get("e_node") or b_node
        if donor is not None and donor.inputs["Vector"].is_linked:
            nt.links.new(donor.inputs["Vector"].links[0].from_socket,
                         node.inputs["Vector"])
        nt.links.new(node.outputs["Color"], e_sock)
        e_sock.default_value = (1, 1, 1, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
        stats.append(st)
    return stats


def rebuild_from_args(sid, obj, facts, areas, args):
    """One dispatcher for what feeds the emission, shared by the key-lit and
    box paths so the two treatments can never disagree about the prediction."""
    if args.pred_emission:
        # an outside method's own emission RGB: written in unchanged, no albedo
        # multiply (that multiply is our method, not a neutral rendering step)
        emis, uniform, names = load_emission(args.pred_emission, sid,
                                             len(obj.material_slots))
        assert_slot_names(sid, obj, names)
        return rebuild_emission_direct(
            obj, facts, areas, emis, uniform,
            strength=(args.emit_strength if args.emission_strength_ours else 1.0),
            assume_linear=bool(args.emission_linear))
    if args.pred_rgb:
        rgbs, uniform, names = load_pred_rgb(args.pred_rgb, sid,
                                             len(obj.material_slots))
        assert_slot_names(sid, obj, names)
        return rebuild_emission_predicted_rgb(obj, facts, areas, rgbs, uniform,
                                              args.emit_strength)
    if args.pred_masks:
        masks, uniform, names = load_pred(args.pred_masks, sid,
                                          len(obj.material_slots))
        assert_slot_names(sid, obj, names)
        return rebuild_emission_predicted(obj, facts, areas, masks, uniform,
                                          args.emit_strength)
    return rebuild_emission(obj, facts, areas, args.emit_strength)


# --------------------------------------------------------- emission rewrite
def rebuild_emission(obj, facts, areas, strength=EMIT_STRENGTH):
    """Replace every material's emission with mask x albedo."""
    stats = []
    for f in facts:
        mat = f["mat"]
        area = float(areas[f["slot"]]) if f["slot"] < len(areas) else 0.0
        st = {"material": mat.name if mat else None, "area": area,
              "emits": bool(f.get("emits")), "mask_frac": 0.0}
        if not f.get("emits"):
            stats.append(st)
            continue

        bsdf, nt = f["bsdf"], mat.node_tree
        e_sock = bsdf.inputs["Emission Color"]
        b_sock = bsdf.inputs["Base Color"]
        e_node, e_const = f["e_node"], f["e_const"]
        # the asset's own authored emissive strength — for the error metrics
        # below ONLY. The render strength is this function's `strength`
        # parameter; do not shadow it here.
        source_strength = f["strength"]
        b_node = upstream_image(b_sock)
        b_const = socket_rgb(b_sock)
        st["emissive_texture"] = e_node is not None
        st["albedo_texture"] = b_node is not None

        if e_node is None:
            # a uniform emitter: every texel of this material emits, so the mask
            # is 1 and the emission simply becomes the albedo
            st["mask_frac"] = 1.0
            for l in list(e_sock.links):
                nt.links.remove(l)
            if b_node is not None:
                nt.links.new(b_node.outputs["Color"], e_sock)
                e_sock.default_value = (1, 1, 1, 1)
            else:
                e_sock.default_value = (*b_const, 1.0)
                true = e_const * source_strength
                st["true_mean"] = float(true.mean())
                st["ours_mean"] = float(b_const.mean())
                st["rel_err"] = float(np.abs(b_const - true).sum()
                                      / max(true.sum(), 1e-6))
            bsdf.inputs["Emission Strength"].default_value = strength
            stats.append(st)
            continue

        e_arr = img_array(e_node.image)[..., :3] * e_const
        mask = (e_arr.max(axis=-1) > LIN_EPS).astype(np.float32)
        st["mask_frac"] = float(mask.mean())
        if st["mask_frac"] <= 0:
            stats.append(st)
            continue
        if b_node is not None:
            alb = resize_to(img_array(b_node.image), mask.shape)[..., :3] * b_const
        else:
            alb = np.broadcast_to(b_const, (*mask.shape, 3))
        out = np.empty((*mask.shape, 4), dtype=np.float32)
        out[..., :3] = alb * mask[..., None]
        out[..., 3] = 1.0

        im = new_image(f"emis_{mat.name}", out)
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = im
        node.location = (e_node.location.x, e_node.location.y - 320)
        # reuse the emissive texture's own UV wiring, so KHR_texture_transform
        # and non-default UV sets carry over unchanged
        if e_node.inputs["Vector"].is_linked:
            nt.links.new(e_node.inputs["Vector"].links[0].from_socket,
                         node.inputs["Vector"])
        for l in list(e_sock.links):
            nt.links.remove(l)
        nt.links.new(node.outputs["Color"], e_sock)
        e_sock.default_value = (1, 1, 1, 1)
        bsdf.inputs["Emission Strength"].default_value = strength

        # how close is mask x albedo to the emission the asset actually carries?
        true = e_arr * source_strength
        sel = mask > 0
        st["true_mean"] = float(true[sel].mean())
        st["ours_mean"] = float(out[..., :3][sel].mean())
        st["rel_err"] = float(np.abs(out[..., :3][sel] - true[sel]).sum()
                              / max(true[sel].sum(), 1e-6))
        # color agreement alone, with overall level divided out: this is the
        # question the method actually turns on
        t = true[sel].reshape(-1, 3); o = out[..., :3][sel].reshape(-1, 3)
        tn = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-8)
        on = o / np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-8)
        st["hue_cos"] = float((tn * on).sum(axis=1).mean())
        stats.append(st)
    return stats


# ------------------------------------------------------------------- scene
def dark_room(bg=0.004, key=8.0):
    """Turn the studio preset into a dark room: a near-black world (the preset's
    own environment is replaced outright, since a linked environment texture
    ignores the Background node's color), one dim key light so the silhouette
    stays readable, and a matte floor that catches what the object spills."""
    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg_node = nt.nodes.new("ShaderNodeBackground")
    bg_node.inputs[0].default_value = (bg, bg, bg, 1.0)
    bg_node.inputs[1].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg_node.outputs["Background"], out.inputs["Surface"])

    lights = [o for o in bpy.data.objects if o.type == "LIGHT"]
    for i, o in enumerate(lights):
        o.data.energy = key if i == 0 else key * 0.2
        if hasattr(o.data, "shadow_soft_size"):
            o.data.shadow_soft_size = max(o.data.shadow_soft_size, 0.6)

    floor = bpy.data.objects.get("Floor")
    if floor is not None:
        floor.cycles.is_shadow_catcher = False
        floor.hide_render = False
        mat = bpy.data.materials.new("DarkFloor")
        mat.use_nodes = True
        bsdf = principled(mat)
        bsdf.inputs["Base Color"].default_value = (0.10, 0.10, 0.108, 1)
        bsdf.inputs["Roughness"].default_value = 0.5
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0
        floor.data.materials.clear()
        floor.data.materials.append(mat)
    return floor


def emission_only_box(obj, azimuth, wall=0.70, scale=2.0, height=1.7,
                      depth=1.8):
    """Put the object in a Cornell box where its own emission is the ONLY light.

    Every other source is removed rather than dimmed: the preset's key lights are
    deleted outright and the world is set to zero strength. In the key-lit
    renders the object's shell is lit by a lamp, so those panels show the
    emission but cannot show what the emission DOES; here the only photons in
    the scene come off the object, so the pool on the floor and the wash up the
    walls are the object's own light and nothing else.

    The box is rotated to the camera's azimuth so its open face squares up with
    the viewer, giving the classic converging side walls instead of the camera
    staring into a corner. Walls are neutral and matte: a colored wall would
    tint the bounce, and the bounced color is exactly what is under test.
    """
    for o in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o, do_unlink=True)
    floor = bpy.data.objects.get("Floor")
    if floor is not None:
        bpy.data.objects.remove(floor, do_unlink=True)

    world = bpy.context.scene.world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value = (0, 0, 0, 1)
    bg.inputs[1].default_value = 0.0
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    lo, hi = world_bbox(obj)
    centre = (lo + hi) / 2.0
    span = hi - lo
    r = float(np.hypot(span[0], span[1])) / 2.0          # horizontal radius
    half = max(r * scale, span[2] * 0.55)                # box half width
    top = max(span[2] * height, half * 1.5)
    back = max(r * depth, half * 0.9)
    front = -half * 2.6                                  # the open side

    # local frame: x across, y from the opening (negative) to the back wall
    # (positive), z up; the object sits on z = 0
    quads = [
        [(-half, front, 0), (half, front, 0), (half, back, 0), (-half, back, 0)],
        [(-half, front, top), (-half, back, top), (half, back, top),
         (half, front, top)],
        [(-half, back, 0), (half, back, 0), (half, back, top), (-half, back, top)],
        [(-half, front, 0), (-half, back, 0), (-half, back, top),
         (-half, front, top)],
        [(half, front, 0), (half, front, top), (half, back, top), (half, back, 0)],
    ]
    verts, faces = [], []
    for q in quads:
        faces.append([len(verts) + i for i in range(4)])
        verts.extend(q)
    verts = np.array(verts, dtype=np.float64)
    theta = np.radians(azimuth)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    verts = verts @ rot.T
    verts[:, 0] += centre[0]
    verts[:, 1] += centre[1]

    me = bpy.data.meshes.new("cornell")
    me.from_pydata(verts.tolist(), [], faces)
    me.update()
    box = bpy.data.objects.new("cornell", me)
    bpy.context.collection.objects.link(box)

    mat = bpy.data.materials.new("CornellWall")
    mat.use_nodes = True
    bsdf = principled(mat)
    bsdf.inputs["Base Color"].default_value = (wall, wall, wall * 0.99, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.15
    me.materials.append(mat)
    return box


def assert_emission_is_only_light():
    """Fail loudly rather than ship a panel whose claim is false."""
    lights = [o for o in bpy.data.objects if o.type == "LIGHT"]
    assert not lights, f"lights still in the scene: {[o.name for o in lights]}"
    bgs = [n for n in bpy.context.scene.world.node_tree.nodes
           if n.type == "BACKGROUND"]
    for n in bgs:
        assert float(n.inputs[1].default_value) == 0.0, "world still emits"
        assert max(n.inputs[0].default_value[:3]) == 0.0, "world is not black"


def bloom_or_clear(args):
    """Honor --bloom on the key-lit path.

    The compositor persists across renders in one scene, so turning bloom off
    has to CLEAR the node, not merely skip adding it.
    """
    if args.bloom:
        add_bloom(size=args.bloom_size, threshold=args.bloom_threshold,
                  mix=args.bloom_mix)
    else:
        clear_compositor()


def add_bloom(size=8, threshold=1.0, mix=-0.30):
    """Fog-glow in the compositor. Cycles has no render-time bloom."""
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    glare = nt.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.quality = "HIGH"
    glare.size = size
    glare.threshold = threshold
    glare.mix = mix
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], glare.inputs["Image"])
    nt.links.new(glare.outputs["Image"], comp.inputs["Image"])
    return glare


def clear_compositor():
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], comp.inputs["Image"])


def world_bbox(obj):
    mw = obj.matrix_world
    pts = np.array([list(mw @ v.co) for v in obj.data.vertices], dtype=np.float64)
    return pts.min(axis=0), pts.max(axis=0)


def shrink_textures(obj, max_px):
    """Cap every texture the object uses at `max_px` on its long side.

    The preview GLBs are served over the web and a few source assets carry 4K
    maps, which is two orders of magnitude more texture than a thumbnail-sized
    viewer resolves. Blender's own resampler is used because the glTF exporter's
    resize path needs an image encoder that is not installed here.
    """
    seen, before, after = set(), 0, 0
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            im = node.image
            if im.name in seen:
                continue
            seen.add(im.name)
            w, h = im.size
            before += w * h
            if max(w, h) > max_px:
                s = max_px / max(w, h)
                im.scale(max(1, int(w * s)), max(1, int(h * s)))
            after += im.size[0] * im.size[1]
    return len(seen), before, after


def decimate(obj, max_faces):
    """Collapse the mesh toward `max_faces` triangles, UVs preserved."""
    n = len(obj.data.polygons)
    if n <= max_faces:
        return n, n
    mod = obj.modifiers.new("preview_decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = max_faces / n
    mod.use_collapse_triangulate = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return n, len(obj.data.polygons)


def drop_to_floor(obj):
    """Sit the object on z = 0, so the floor reads as ground, not as a backdrop."""
    lo, hi = world_bbox(obj)
    obj.location.z -= lo[2]
    bpy.context.view_layer.update()
    return world_bbox(obj)


def apply_camera(position, target, lens=52.0):
    """Use a camera someone else solved, instead of solving from this mesh.

    A prediction is a REMESHED version of the shape, so its bounding box is not
    the source asset's. Solving the camera from it would frame each panel to its
    own bounds and silently shift the viewpoint between columns of the same row,
    which is precisely the comparison the figure exists to make. Pass the source
    shape's camera here and every column stays on one view.
    """
    cam = bpy.context.scene.camera
    for c in list(cam.constraints):
        cam.constraints.remove(c)
    cam.data.lens = lens
    cam.data.sensor_fit = "AUTO"
    render_utils.set_camera_orientation(tuple(position), target=tuple(target),
                                        up=(0, 0, 1))
    return np.array(position, dtype=float)


def place_camera(obj, azimuth=38.0, elevation=17.0, lens=52.0, margin=1.06):
    """A product-shot three-quarter view from slightly above eye level, with the
    distance SOLVED so the bounding box's eight corners just fit the frame.

    Fitting on the bounding sphere instead leaves a flat or elongated object
    swimming in empty frame: the sphere's radius is the box DIAGONAL, which for
    a wide low object is far larger than anything the camera actually sees. Each
    corner is decomposed into its along-view and perpendicular parts, and the
    distance is the smallest one that keeps every corner inside the half-angle.

    Elevation is positive by construction: the camera never looks up at the
    object from below.
    """
    lo, hi = world_bbox(obj)
    centre = (lo + hi) / 2.0

    cam = bpy.context.scene.camera
    # The preset camera carries a TRACK_TO constraint aimed at the origin, which
    # is evaluated AFTER matrix_world is assigned: without clearing it the camera
    # silently keeps looking at (0,0,0) whatever target is passed, and every
    # object taller than it is wide gets its top cut off.
    for c in list(cam.constraints):
        cam.constraints.remove(c)
    cam.data.lens = lens
    cam.data.sensor_fit = "AUTO"
    half_fov = np.arctan(cam.data.sensor_width / (2.0 * lens))
    tan_h = np.tan(half_fov) / margin

    az, el = np.radians(azimuth), np.radians(elevation)
    direction = np.array([np.cos(el) * np.sin(az),
                          -np.cos(el) * np.cos(az),
                          np.sin(el)])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    rel = corners - centre
    along = rel @ direction                      # toward the camera is positive
    perp = np.linalg.norm(rel - np.outer(along, direction), axis=1)
    dist = float(np.max(perp / tan_h + along))

    pos = centre + direction * dist
    render_utils.set_camera_orientation(tuple(pos), target=tuple(centre), up=(0, 0, 1))
    return pos, centre, dist


def render(out_path, resolution, samples, transparent, view_transform,
           exposure=0.0, file_format=None):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.film_transparent = transparent
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    fmt = file_format or "PNG"
    scene.render.image_settings.file_format = fmt
    scene.render.image_settings.color_mode = "RGBA"
    # EXR carries the SCENE-referred linear result, so the view transform is not
    # baked in and the sweep can re-grade it; PNG is display-referred 8 bit.
    if fmt == "OPEN_EXR":
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "ZIP"
    else:
        scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = view_transform
    scene.view_settings.look = "None"
    scene.view_settings.exposure = exposure
    scene.render.use_compositing = True
    scene.view_layers["ViewLayer"].use_freestyle = False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


# -------------------------------------------------------------------- main
def one(sid, glb, out, args):
    res = (args.res, args.res)
    render_utils.load_blend(PRESET_GLB)
    render_utils.clear_collection("workbench")
    try:
        obj = render_utils.load_glb(glb, import_shading=None)
    except TypeError:
        # bpy 4.0's glTF importer lacks the newer armature kwargs load_glb
        # passes (disable_bone_shape / guess_original_bind_pose); dropping
        # them only loses those two importer options, not armature import
        obj = render_utils.load_glb(glb, import_shading=None, apply_armature=False)
    lo, hi = drop_to_floor(obj)
    if args.camera_json:
        cam = json.load(open(args.camera_json))
        pos = apply_camera(cam["position"], cam["target"], args.lens)
        centre, dist = np.array(cam["target"], dtype=float), None
    else:
        pos, centre, dist = place_camera(obj, args.azimuth, args.elevation,
                                         args.lens, args.margin)
    facts = analyze(obj)
    areas = material_areas(obj)
    n_emit = sum(1 for f in facts if f.get("emits"))

    floor = bpy.data.objects.get("Floor")
    if floor is not None:
        floor.location.z = -0.004

    # The box run: same camera, same mask x albedo emission, but the object is
    # the only light in the scene and it sits in a room that can show what that
    # light does.
    if args.mode == "box":
        stats = rebuild_from_args(sid, obj, facts, areas, args)
        emission_only_box(obj, args.azimuth, wall=args.wall,
                          scale=args.box_scale, height=args.box_height,
                          depth=args.box_depth)
        assert_emission_is_only_light()
        # In a box lit only by the object, the ambient fill IS multi-bounce
        # diffuse light, so Cycles' default 4 diffuse bounces is not a quality
        # setting, it is a truncation of the physics: at wall albedo 0.8 the
        # fourth bounce still carries 41 percent of the energy and bounces 5
        # onward are what light the object's shadowed side. A real room has no
        # such cap.
        bpy.context.scene.cycles.max_bounces = args.max_bounces
        bpy.context.scene.cycles.diffuse_bounces = args.diffuse_bounces
        if args.bloom:
            add_bloom(size=args.bloom_size, threshold=args.bloom_threshold,
                      mix=args.bloom_mix)
        else:
            # bloom is post-process, so a glare-free render can be saved once as
            # linear EXR and re-graded through the Glare node afterwards; that
            # turns a parameter sweep from one render per cell into one per shape
            clear_compositor()
        ext = "exr" if args.out_format == "OPEN_EXR" else "png"
        render(os.path.join(out, f"{sid}_box{args.tag}.{ext}"), res, args.samples,
               False, args.view_transform, args.exposure,
               file_format=args.out_format)
        lit = sum(s["area"] * s.get("mask_frac", s.get("lit_frac", 0.0))
                  for s in stats)
        total = sum(s["area"] for s in stats) or 1.0
        summary = {"sid": sid, "mode": "box", "area_lit_frac": lit / total,
                   "pred_masks": args.pred_masks, "pred_rgb": args.pred_rgb,
                   "pred_emission": args.pred_emission,
                   "camera_target": list(map(float, centre)),
                   "samples": args.samples, "max_bounces": args.max_bounces,
                   "diffuse_bounces": args.diffuse_bounces, "wall": args.wall,
                   "emit_strength": args.emit_strength,
                   "view_transform": args.view_transform,
                   "exposure": args.exposure, "camera": list(map(float, pos))}
        with open(os.path.join(out, f"{sid}_box{args.tag}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        return summary

    # A baseline run reuses everything above (same camera, same dark room, same
    # emission strength) and differs only in what feeds the emission.
    if args.mode != "method":
        dark_room(bg=args.bg, key=args.key)
        if floor is not None:
            floor.location.z = -0.004
        bloom_or_clear(args)
        rng = np.random.default_rng(args.seed)
        density = rebuild_emission_baseline(obj, facts, areas, args.mode, rng,
                                            strength=args.emit_strength)
        render(os.path.join(out, f"{sid}_{args.mode}.png"), res, args.samples,
               False, args.view_transform, args.exposure)
        summary = {"sid": sid, "mode": args.mode, "density": density,
                   "camera": list(map(float, pos)),
                   # the baseline panels sit in the same figure as the method
                   # ones and have to record the same treatment
                   "treatment": {"view_transform": args.view_transform,
                                 "exposure": args.exposure,
                                 "key": args.key,
                                 "bg": args.bg,
                                 "samples": args.samples,
                                 "bloom_size": args.bloom_size,
                                 "bloom_threshold": args.bloom_threshold,
                                 "bloom_mix": args.bloom_mix}}
        with open(os.path.join(out, f"{sid}_{args.mode}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        return summary

    if not args.glb_only:
        # 1. the studio preset, unchanged: what the object is, for verification
        if not args.skip_reference:
            clear_compositor()
            if floor is not None:
                floor.cycles.is_shadow_catcher = True
            render(os.path.join(out, f"{sid}_lit.png"), res, args.samples_lit,
                   True, "Khronos PBR Neutral")

        # 2. dark room, the asset's own emission. The dark room + bloom are
        # scene state the glow render needs too, so --skip_reference skips
        # only the two reference stills, never this setup.
        dark_room(bg=args.bg, key=args.key)
        if floor is not None:
            floor.location.z = -0.004
        bloom_or_clear(args)
        if not args.skip_reference:
            set_true_strength(facts, args.emit_strength)
            render(os.path.join(out, f"{sid}_true.png"), res, args.samples,
                   False, args.view_transform, args.exposure)

    # 3. dark room, emission = the prediction (or the asset's own mask x albedo)
    stats = rebuild_from_args(sid, obj, facts, areas, args)
    if not args.glb_only:
        render(os.path.join(out, f"{sid}_glow.png"), res, args.samples,
               False, args.view_transform, args.exposure)

    # 4. the modified asset, for the web viewer. A texture Blender's WEBP
    # encoder refuses ("Could not write image: Success") must not cost the shape
    # its renders, so the export falls back and then gives up quietly.
    glb_written = False
    tex_info = faces_info = None
    if args.export_glb:
        if args.max_tex:
            tex_info = shrink_textures(obj, args.max_tex)
        if args.max_faces:
            faces_info = decimate(obj, args.max_faces)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        for fmt in ("WEBP", "JPEG", "AUTO"):
            try:
                bpy.ops.export_scene.gltf(
                    filepath=os.path.join(out, f"{sid}_mod.glb"),
                    export_format="GLB", use_selection=True,
                    export_image_format=fmt, export_image_quality=80,
                    export_yup=True)
                glb_written = True
                break
            except Exception as exc:
                print(f"  glb export {fmt} failed: {exc}", flush=True)

    total = float(sum(s["area"] for s in stats)) or 1.0
    lit_area = float(sum(s["area"] * s.get("mask_frac", s.get("lit_frac", 0.0))
                         for s in stats))
    summary = {
        "sid": sid,
        "n_materials": len(stats),
        "n_emissive_materials": n_emit,
        "glb_written": glb_written,
        "treatment": {"view_transform": args.view_transform,
                      "exposure": args.exposure, "key": args.key, "bg": args.bg,
                      "samples": args.samples,
                      "bloom_size": args.bloom_size,
                      "bloom_threshold": args.bloom_threshold,
                      "bloom_mix": args.bloom_mix,
                      "emit_strength": args.emit_strength,
                      "emission_linear": args.emission_linear,
                      "emission_strength_ours": args.emission_strength_ours},
        "pred_masks": args.pred_masks,
        "pred_rgb": args.pred_rgb,
        "pred_emission": args.pred_emission,
        "textures": tex_info,
        "faces": faces_info,
        "bbox": [list(lo), list(hi)],
        "camera": list(map(float, pos)),
        "camera_distance": dist,
        "camera_target": list(map(float, centre)),
        # share of the object's surface the ground-truth mask turns on. Near 1
        # means a fullbright asset, where the whole object is its own light and
        # nothing is illustrated.
        "area_lit_frac": lit_area / total,
        "materials": stats,
    }
    name = f"{sid}_glbstats.json" if args.glb_only else f"{sid}_stats.json"
    with open(os.path.join(out, name), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--glb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated sids")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--samples_lit", type=int, default=96)
    ap.add_argument("--azimuth", type=float, default=38.0)
    ap.add_argument("--elevation", type=float, default=17.0)
    ap.add_argument("--lens", type=float, default=52.0)
    ap.add_argument("--margin", type=float, default=1.06)
    ap.add_argument("--bg", type=float, default=0.004)
    ap.add_argument("--key", type=float, default=8.0)
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--view_transform", default="AgX")
    ap.add_argument("--bloom_size", type=int, default=7,
                    help="fog-glow radius. 7 is the largest radius that leaves\n                         emissive structure crisp while still showing spill on\n                         the dimmest shape")
    ap.add_argument("--bloom_threshold", type=float, default=1.0)
    ap.add_argument("--bloom_mix", type=float, default=-0.45,
                    help="-1 is the original image, 0 is a 50/50 blend; -0.15\n                         was close to half glare")
    ap.add_argument("--export_glb", type=int, default=1)
    ap.add_argument("--glb_only", type=int, default=0)
    ap.add_argument("--max_tex", type=int, default=0,
                    help="cap preview textures at N px on the long side")
    ap.add_argument("--max_faces", type=int, default=0,
                    help="decimate the preview mesh toward N triangles")
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--mode", default="method",
                    choices=["method", "random", "allemissive", "box"],
                    help="method = mask x albedo; the other two are the "
                         "dummy baselines for the comparison figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bloom", type=int, default=1,
                    help="0 renders with the Glare node bypassed")
    ap.add_argument("--out_format", default="PNG",
                    choices=["PNG", "OPEN_EXR"])
    ap.add_argument("--wall", type=float, default=0.80,
                    help="Cornell wall albedo; pure white blows out and hides "
                         "the very gradient the panel exists to show. 0.80 over "
                         "0.70 compounds across bounces: 0.8^4 = 0.41 against "
                         "0.7^4 = 0.24")
    ap.add_argument("--box_scale", type=float, default=2.0)
    ap.add_argument("--box_height", type=float, default=1.7)
    ap.add_argument("--box_depth", type=float, default=1.8)
    ap.add_argument("--max_bounces", type=int, default=32)
    ap.add_argument("--diffuse_bounces", type=int, default=16)
    ap.add_argument("--pred_masks", default=None,
                    help="directory of predicted per-material masks; "
                         "replaces the asset's own mask, and switches OFF "
                         "any material the model did not select")
    ap.add_argument("--pred_rgb", default=None,
                    help="directory of predicted per-material emission RGB "
                         "(bridge_pred_to_source_uv.py output); the predicted "
                         "colour is the emission, no albedo multiply. Same "
                         "off-switch semantics as --pred_masks")
    ap.add_argument("--pred_emission", default=None,
                    help="directory of predicted emission COLOUR textures, same "
                         "file convention and slot-name contract as "
                         "--pred_masks. Use this when your method predicts "
                         "emission RGB: the values are written in unchanged, "
                         "with NO multiply by the asset's albedo")
    ap.add_argument("--emission_linear", type=int, default=0,
                    help="1 if --pred_emission files are already linear. The "
                         "default treats 8-bit PNGs as sRGB, which is what they "
                         "almost always are; getting this wrong shifts the "
                         "midtones by about a factor of two and looks like a "
                         "brightness disagreement between methods")
    ap.add_argument("--emission_strength_ours", type=int, default=0,
                    help="1 multiplies --pred_emission values by "
                         "--emit_strength, matching our panels' look. The "
                         "default of 0 leaves predicted radiance alone, because "
                         "scaling someone else's prediction is a second opinion "
                         "about brightness rather than a rendering choice")
    ap.add_argument("--emit_strength", type=float, default=EMIT_STRENGTH,
                    help=f"emission strength for the mask paths (default "
                         f"{EMIT_STRENGTH}). A LOOK CHOICE, not recovered from "
                         f"the data: the bake stores emission as uint8 and drops "
                         f"KHR_materials_emissive_strength")
    ap.add_argument("--camera_json", default=None,
                    help="{\"position\":[x,y,z],\"target\":[x,y,z]}; use a\n                         camera solved from ANOTHER mesh, so a remeshed\n                         prediction is framed like the shape it predicts")
    ap.add_argument("--tag", default="",
                    help="suffix on the box output name, for sweeps")
    ap.add_argument("--skip_reference", type=int, default=0,
                    help="method mode: skip the lit/true reference stills "
                         "(they are prediction-independent; render them once "
                         "in the GT pass, not once per model)")
    args = ap.parse_args()
    n_pred = sum(1 for p in (args.pred_masks, args.pred_rgb,
                             args.pred_emission) if p)
    assert n_pred <= 1, (
        "--pred_masks, --pred_rgb and --pred_emission are three different "
        "claims about what the model predicts; pass exactly one")

    rows = json.load(open(args.manifest))
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["sid"] in keep]
    rows = rows[args.shard::args.nshards]
    os.makedirs(args.out, exist_ok=True)
    failed = []
    for r in rows:
        sid = r["sid"]
        done = os.path.join(args.out, f"{sid}_stats.json" if args.mode == "method"
                            else f"{sid}_{args.mode}{args.tag}.json")
        if os.path.exists(done) and not args.overwrite:
            print(f"SKIP {sid}", flush=True)
            continue
        glb = os.path.join(args.glb_dir, f"{sid}.glb")
        print(f"=== {sid} {r.get('cat','')} ef={r.get('ef',0):.3f}", flush=True)
        try:
            s = one(sid, glb, args.out, args)
            if args.mode == "box":
                print(f"OK {sid} box lit={s['area_lit_frac']:.4f} "
                      f"samples={s['samples']}", flush=True)
            elif args.mode != "method":
                print(f"OK {sid} {args.mode} density={s['density']:.4f}", flush=True)
            else:
                print(f"OK {sid} mats={s['n_materials']} emissive={s['n_emissive_materials']} "
                      f"area_lit={s['area_lit_frac']:.3f}", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"FAIL {sid}", flush=True)
            failed.append(sid)
    print("ALL_DONE", flush=True)
    # bpy segfaults on interpreter teardown after a run of imports; every output
    # is already on disk by here, so leave before it can turn a good run into a
    # nonzero exit. Failures are carried out in the exit status, so the
    # workaround only suppresses the teardown crash and never a real one.
    if failed:
        print(f"FAILED_SHAPES {len(failed)}: {' '.join(failed)}", flush=True)
        os._exit(1)
    os._exit(0)


if __name__ == "__main__":
    main()
