"""Albedo-copy check: is the entire base color copied into the entire emission?

Second check of the heuristic stage, on the GLB alone. The Sketchfab exporter often writes
a material's base color into its emissive slot, so the whole object would glow in its own
albedo. A drawn material is "copied" when its emissive slot holds the base-color picture
(the same texture entry, the same image, or decoded pixels that agree at 256x256 to a mean
difference below 1/255) or, with no base-color texture, when `emissiveFactor` RGB equals
`baseColorFactor` RGB and is not black. A shape is rejected (`albedo_copy`) when every
drawn material is copied. `--specgloss` (opt-in, off for the released dataset) also counts
the spec/gloss `diffuseTexture` / `diffuseFactor` as the base color.

    conda run --no-capture-output -n lightgen python -m \\
      data_processing.heuristic_filtering.albedo_copy_check     # over out/nonzero_emission_pass.txt
    ... --specgloss                                             # the opt-in rule above

The CSV is the resume marker (see `utils/check_runner.py`).
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.check_runner import run_check  # noqa: E402
from utils.glb_io import find_glb  # noqa: E402
from utils.gltf_emission import _image_bytes, read_glb, read_gltf_json  # noqa: E402
from utils.gltf_materials import drawn_materials  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

# --- input and outputs ----------------------------------------------------------------
#: Input: the non-zero emission check's pass list. Output: this check's rows and pass list.
INPUT_LIST = OUT_DIR / "nonzero_emission_pass.txt"
DEFAULT_OUT = OUT_DIR / "albedo_copy_check.csv"
DEFAULT_PASS_LIST = OUT_DIR / "albedo_copy_pass.txt"
#: `--specgloss` (opt-in, not the released rule) writes apart so it never resumes from the default rule's rows.
SPECGLOSS_OUT = OUT_DIR / "albedo_copy_check_specgloss.csv"
SPECGLOSS_PASS_LIST = OUT_DIR / "albedo_copy_pass_specgloss.txt"

# --- schema -----------------------------------------------------------------------------
#: One row per shape. "Drawn" = used by a primitive of a mesh the scene draws (all of
#: them, emitting or not). A textured material is copied when its emissive slot holds
#: the base-color picture (same textures[] entry, same images[] source, or decoded
#: pixels that agree); a constant-color one when emissiveFactor equals baseColorFactor
#: (not black). Reject when every drawn material is copied.
FIELDS = [
    "uuid",
    "status",               # ok | no_file | err:<name>
    "n_drawn",              # materials reachable from the scene
    "n_base_texture",       # ... of which have a base-color texture (record only)
    "n_same_slot",          # textured materials whose emissive slot holds the base-color picture
    "n_same_color",         # constant-color materials whose emissiveFactor equals baseColorFactor
    "reject",               # blank | albedo_copy
]
EMPTY_ON_FAILURE = FIELDS[2:]
REJECT_REASONS = ("albedo_copy",)


def _rgb(v, default):
    try:
        return tuple(float(c) for c in v[:3])
    except (TypeError, ValueError):
        return default


SPECGLOSS = "KHR_materials_pbrSpecularGlossiness"


def _has_specgloss(mat: dict) -> bool:
    """The extension object may be `{}`; that still selects the workflow, with every
    field at its default (diffuseFactor white)."""
    return SPECGLOSS in (mat.get("extensions") or {})


def _specgloss(mat: dict) -> dict:
    return (mat.get("extensions") or {}).get(SPECGLOSS) or {}


def _base_texture_refs(mat: dict, specgloss: bool = False) -> list:
    """The slots that hold a material's base-color picture: the metallic-roughness
    `baseColorTexture`, and with `specgloss` also the spec/gloss `diffuseTexture`."""
    refs = [(mat.get("pbrMetallicRoughness") or {}).get("baseColorTexture")]
    if specgloss:
        refs.append(_specgloss(mat).get("diffuseTexture"))
    return [r for r in refs if r]


def has_base_texture(mat: dict, specgloss: bool = False) -> bool:
    return bool(_base_texture_refs(mat, specgloss))


#: Mean absolute RGB difference (0..255) under which two decoded images are the same picture.
SAME_PICTURE_TOL = 1.0
_CMP_SIZE = 256


class _Pictures:
    """Decoded images of one file, compared lazily and cached per image-index pair."""

    def __init__(self, js, path):
        self.js, self.path, self.binary, self.arrays, self.pairs = js, path, None, {}, {}

    def _array(self, image_index):
        if image_index not in self.arrays:
            import io

            import numpy as np
            from PIL import Image

            if self.binary is None:
                parsed = read_glb(self.path)
                self.binary = parsed[1] if parsed else b""
            raw = _image_bytes(self.js, self.binary, image_index, self.path)
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    self.arrays[image_index] = np.asarray(im.convert("RGB").resize((_CMP_SIZE, _CMP_SIZE))).astype(np.int16)
            except Exception:
                self.arrays[image_index] = None
        return self.arrays[image_index]

    def same(self, a, b) -> bool:
        key = (min(a, b), max(a, b))
        if key not in self.pairs:
            import numpy as np

            x, y = self._array(a), self._array(b)
            self.pairs[key] = x is not None and y is not None and float(np.abs(x - y).mean()) < SAME_PICTURE_TOL
        return self.pairs[key]


def _source(js: dict, ref: dict):
    try:
        return int(js["textures"][int(ref["index"])]["source"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def same_slot(mat: dict, js: dict = None, pictures: "_Pictures" = None, specgloss: bool = False) -> bool:
    """Textured case: the emissive slot holds the base-color picture.

    Same `textures[]` entry -> True without reading anything. Different entries -> the
    same `images[]` source, or (with `js` and `pictures` given) decoded pixels that agree.
    """
    emis = mat.get("emissiveTexture")
    if not emis:
        return False
    return any(_same_picture(base, emis, js, pictures) for base in _base_texture_refs(mat, specgloss))


def _same_picture(base: dict, emis: dict, js, pictures) -> bool:
    try:
        if int(base["index"]) == int(emis["index"]):
            return True
    except (KeyError, TypeError, ValueError):
        return False
    if js is None:
        return False
    sb, se = _source(js, base), _source(js, emis)
    if sb is None or se is None:
        return False
    if sb == se:
        return True
    return pictures is not None and pictures.same(sb, se)


def same_color(mat: dict, specgloss: bool = False) -> bool:
    """Constant case: no base-color texture, emissiveFactor equals the base constant
    (`baseColorFactor`, and with `specgloss` also the spec/gloss `diffuseFactor`; white
    when absent), not black."""
    if has_base_texture(mat, specgloss) or "emissiveTexture" in mat:
        return False
    bases = [_rgb((mat.get("pbrMetallicRoughness") or {}).get("baseColorFactor"), (1.0, 1.0, 1.0))]
    if specgloss and _has_specgloss(mat):
        bases.append(_rgb(_specgloss(mat).get("diffuseFactor"), (1.0, 1.0, 1.0)))
    emis = _rgb(mat.get("emissiveFactor"), (0.0, 0.0, 0.0))
    return any(c > 0 for c in emis) and any(all(abs(a - b) < 1e-6 for a, b in zip(base, emis)) for base in bases)


def declares_emission(mat: dict) -> bool:
    if "emissiveTexture" in mat:
        return True
    return any(c > 0 for c in _rgb(mat.get("emissiveFactor"), (0.0, 0.0, 0.0)))


def copied(mat: dict, specgloss: bool = False) -> bool:
    return same_slot(mat, specgloss=specgloss) or same_color(mat, specgloss)


def check_one(sha: str, metadata: pd.DataFrame, roots, specgloss: bool = False) -> dict:
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
    _, drawn = drawn_materials(js)
    mats = js.get("materials") or []
    drawn = [i for i in sorted(drawn) if 0 <= i < len(mats)]
    # Gate: a drawn material with no emission is not copied, so the shape passes and
    # nothing needs decoding. Only an all-emissive object can be a whole copy.
    all_emissive = bool(drawn) and all(declares_emission(mats[i]) for i in drawn)
    pictures = _Pictures(js, path) if all_emissive else None
    n_slot = sum(1 for i in drawn if same_slot(mats[i], js, pictures, specgloss))
    n_color = sum(1 for i in drawn if same_color(mats[i], specgloss))
    reject = "albedo_copy" if all_emissive and n_slot + n_color == len(drawn) else ""
    return {"uuid": sha, "status": "ok", "n_drawn": len(drawn), "n_base_texture": sum(1 for i in drawn if has_base_texture(mats[i], specgloss)),
            "n_same_slot": n_slot, "n_same_color": n_color, "reject": reject}


def main():
    # `--specgloss` is this check's own flag; the shared runner parses the rest. Its outputs
    # default apart, so it does not resume from, or overwrite, the default rule's rows unless
    # `--out` / `--pass_list` name them.
    flag = argparse.ArgumentParser(add_help=False)
    flag.add_argument("--specgloss", action="store_true")
    opts, rest = flag.parse_known_args()
    sys.argv[1:] = rest
    if opts.specgloss:
        run_check(functools.partial(check_one, specgloss=True), FIELDS, REJECT_REASONS, INPUT_LIST,
                  SPECGLOSS_OUT, SPECGLOSS_PASS_LIST, __doc__)
    else:
        run_check(check_one, FIELDS, REJECT_REASONS, INPUT_LIST, DEFAULT_OUT, DEFAULT_PASS_LIST, __doc__)


if __name__ == "__main__":
    main()
