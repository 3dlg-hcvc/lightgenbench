"""Non-zero emission check: does the declared emission show when the object is drawn?

First check of the heuristic stage, on the GLB alone (nothing is baked yet). A shape
passes when at least one material is emitting (a declared emissive texture that decodes
to something other than all-black, or, with no texture, an emissiveFactor above zero),
wired (reachable from the file's scene) and visible (not fully transparent). Reject
reasons, one per shape, in this order: `no_emission`, `no_scene`, `not_wired`,
`transparent`.

    conda run --no-capture-output -n lightgen python -m \\
      data_processing.heuristic_filtering.nonzero_emission_check --workers 16

The CSV is the resume marker: a rerun skips the uuids already in it (see
`utils/check_runner.py`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.check_runner import run_check  # noqa: E402
from utils.glb_io import find_glb  # noqa: E402
from utils.gltf_emission import emissive_texture_max, material_emission_from_json, read_gltf_json  # noqa: E402
from utils.gltf_materials import base_color_alpha_max, drawn_materials, materials_from_json  # noqa: E402
from utils.paths import REPO_ROOT  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

# --- input --------------------------------------------------------------------------
#: The metadata stage's pass list: every shape with a used material that declares an
#: emissive texture, or a factor > 0 without one, and whose materials write albedo,
#: metallic and roughness (declaration level, no decode; the shapes that only declare
#: emission are `emission_only_pass.txt` beside it). The filter's input.
INPUT_LIST = REPO_ROOT / "data_processing/metadata_filtering/out/emissive_pass.txt"

# --- outputs ------------------------------------------------------------------------
DEFAULT_OUT = OUT_DIR / "nonzero_emission_check.csv"
#: The check's result: uuids that pass the non-zero emission check, one per line.
DEFAULT_PASS_LIST = OUT_DIR / "nonzero_emission_pass.txt"

# --- schema -------------------------------------------------------------------------
#: One row per shape. "Emitting" = passes the content test (loose rule); "wired" =
#: referenced by a primitive of a mesh that a node reachable from the scene draws;
#: "visible" = wired and not fully transparent through alpha.
FIELDS = [
    "uuid",
    "status",                   # ok | no_file | err:<name>
    "has_scene",                # 1 if the file declares a non-empty `scenes`
    "n_used",                   # materials some mesh primitive references
    "n_drawn",                  # materials reachable from the scene
    "n_emitting",               # used materials passing the content test
    "n_wired_emitting",         # emitting AND drawn
    "n_visible_emitting",       # wired AND not fully transparent  -> the verdict
    "n_wired_emitting_tri",     # wired emitting on a triangle primitive (mode 4/5/6); record only
    "n_wired_emit_tex",         # wired emitting through a texture; record only
    "n_wired_emit_color",       # wired emitting through the factor alone (no texture); record only
    "n_wired_emit_unlit",       # wired emitting materials that are KHR_materials_unlit; record only
    "n_tex_undecodable",        # declared textures (emissive or base-color alpha) that could not be decoded; count as passing
    "reject",                   # blank | no_emission (black texture) | no_scene | not_wired | transparent
]
EMPTY_ON_FAILURE = FIELDS[2:]
REJECT_REASONS = ("no_emission", "no_scene", "not_wired", "transparent")

#: glTF primitive modes that produce faces: TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN.
#: 0-3 are POINTS / LINES variants. Recorded only.
TRIANGLE_MODES = {4, 5, 6}
DEFAULT_ALPHA_CUTOFF = 0.5


def check_one(sha: str, metadata: pd.DataFrame, roots) -> dict:
    """One shape's row. Never raises; read failures land in `status`."""
    path = find_glb(sha, metadata, roots)
    if path is None:
        return {"uuid": sha, "status": "no_file", **{k: "" for k in EMPTY_ON_FAILURE}}
    try:
        js = read_gltf_json(path)
    except OSError as exc:
        return {"uuid": sha, "status": "err:" + type(exc).__name__, **{k: "" for k in EMPTY_ON_FAILURE}}
    if js is None:
        return {"uuid": sha, "status": "err:BadGLB", **{k: "" for k in EMPTY_ON_FAILURE}}

    used = [m for m in material_emission_from_json(js) if m.is_used]
    has_scene, drawn = drawn_materials(js)
    chan = {c.index: c for c in materials_from_json(js).materials}
    undecodable = 0

    # 1. Content, per material (loose reading: a texture counts on its own). Decoding
    # is the expensive step; only shapes that declare a texture pay for it. An
    # undecodable texture (-1) keeps the declaration-level pass.
    textured = [m for m in used if m.has_emissive_texture]
    tmax: Dict[int, int] = {}
    if textured:
        try:
            tmax = emissive_texture_max(path)
        except OSError:
            tmax = {}
    undecodable += sum(1 for m in textured if tmax.get(m.index, -1) == -1)

    def emitting(m) -> bool:
        if m.has_emissive_texture:
            return tmax.get(m.index, -1) != 0
        return any(c > 0 for c in m.emissive_factor)

    emit = [m for m in used if emitting(m)]

    # 2. Wired: reachable from the scene.
    wired = [m for m in emit if m.index in drawn]

    # 3. Visible: not fully transparent. The base-color texture's alpha is decoded
    # only when a wired emitter is BLEND/MASK and has one.
    need_alpha = [m for m in wired if chan[m.index].alpha_mode in ("BLEND", "MASK") and chan[m.index].has_base_color_texture]
    amax: Dict[int, int] = {}
    if need_alpha:
        try:
            amax = base_color_alpha_max(path)
        except OSError:
            amax = {}
        undecodable += sum(1 for m in need_alpha if amax.get(m.index, -1) == -1)

    def visible(m) -> bool:
        c = chan[m.index]
        if c.alpha_mode not in ("BLEND", "MASK"):
            return True
        alpha = c.base_color_factor[3]
        if c.has_base_color_texture:
            t = amax.get(m.index, -1)
            if t >= 0:                         # undecodable: treat the texture as opaque,
                alpha *= t / 255.0             # the factor and cutoff still apply
        if c.alpha_mode == "BLEND":
            return alpha > 0
        cutoff = (js["materials"][m.index].get("alphaCutoff", DEFAULT_ALPHA_CUTOFF))
        try:
            cutoff = float(cutoff)
        except (TypeError, ValueError):
            cutoff = DEFAULT_ALPHA_CUTOFF
        return alpha >= cutoff

    vis = [m for m in wired if visible(m)]

    if not emit:
        reject = "no_emission"
    elif not has_scene:
        reject = "no_scene"
    elif not wired:
        reject = "not_wired"
    elif not vis:
        reject = "transparent"
    else:
        reject = ""
    return {
        "uuid": sha,
        "status": "ok",
        "has_scene": int(has_scene),
        "n_used": len(used),
        "n_drawn": len(drawn),
        "n_emitting": len(emit),
        "n_wired_emitting": len(wired),
        "n_visible_emitting": len(vis),
        "n_wired_emitting_tri": sum(1 for m in wired if drawn[m.index] & TRIANGLE_MODES),
        "n_wired_emit_tex": sum(1 for m in wired if m.has_emissive_texture),
        "n_wired_emit_color": sum(1 for m in wired if not m.has_emissive_texture),
        "n_wired_emit_unlit": sum(1 for m in wired if chan[m.index].is_unlit),
        "n_tex_undecodable": undecodable,
        "reject": reject,
    }


def main():
    run_check(check_one, FIELDS, REJECT_REASONS, INPUT_LIST, DEFAULT_OUT, DEFAULT_PASS_LIST, __doc__)


if __name__ == "__main__":
    main()
