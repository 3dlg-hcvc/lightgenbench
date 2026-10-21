"""Reader for a glTF/GLB asset's declared emission.

Reads only the GLB header + JSON chunk with the standard library, so it imports
in every conda env in this repo and needs no trimesh, bpy, or o_voxel.

    from utils.gltf_emission import max_emissive_strength, read_material_emission

Two independent things in a glTF material declare emission:

  * ``emissiveFactor`` (+ ``emissiveTexture``): the emissive color, core glTF.
  * ``KHR_materials_emissive_strength``: an optional multiplier on that color.
    Absent means 1.0.

This module reports both. It does NOT decide whether the multiplier should be
applied, because the consumers in this repo deliberately disagree:

  * ``data_processing/glb2representations`` (both bakes, through
    ``utils.trimesh_materials.emission_source``) stores a declared emissive texture
    ALONE, no factor, no strength; without a texture the factor is the color.
    ``evaluation/pointsampled/`` scores the release by the same rule (``load_gt.py``).
  * The submitted paper's evaluator applied the FACTOR after the sRGB decode and
    DROPPED the strength, so every number in the submitted paper is factor-only.

So the submitted paper's metrics and the bake disagree on textured shapes whose factor
is not 1. **Pointing an existing caller at this module changes published numbers.** Use
it for new code and for analysis; changing a caller is a deliberate, separately reviewed
decision.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = [
    "EXTENSION",
    "MaterialEmission",
    "read_gltf_json",
    "read_glb",
    "used_material_indices",
    "material_emission_from_json",
    "read_material_emission",
    "max_emissive_strength",
    "declares_emission",
    "emissive_texture_max",
]

EXTENSION = "KHR_materials_emissive_strength"

#: glTF's default when the extension is absent.
DEFAULT_STRENGTH = 1.0


@dataclass(frozen=True)
class MaterialEmission:
    """One material's declared emission."""

    index: int
    emissive_factor: tuple  #: (r, g, b), glTF default (0, 0, 0)
    has_emissive_texture: bool
    strength: float  #: the extension's value, or 1.0 when absent
    declares_strength: bool  #: whether the extension was actually present
    is_used: bool  #: referenced by at least one mesh primitive (see below)

    @property
    def is_emissive(self) -> bool:
        """True when the material DECLARES emission: an emissive texture, or a
        non-zero ``emissiveFactor``. A declaration-level statistic, not a verdict.

        This is the loose reading (texture OR factor), and it is the metadata filter's
        pass rule (``data_processing/metadata_filtering/check_emission.py``): a declared
        texture counts on its own, a factor only without one. Whether the declared
        texture is all-black is tested downstream, in
        ``data_processing/heuristic_filtering/nonzero_emission_check.py``.

        A material may also carry the strength extension while declaring no
        emission; the multiplier is meaningless there.
        """
        return self.has_emissive_texture or any(c > 0 for c in self.emissive_factor)


def read_gltf_json(glb_path: str) -> Optional[dict]:
    """Parse the JSON chunk of a binary glTF. None if the file is not a valid GLB.

    Per the glTF 2.0 spec the first chunk of a GLB is always the JSON chunk, so
    only the header and that chunk are read — the binary payload is never touched,
    which is what makes a full scan cheap.
    """
    with open(glb_path, "rb") as f:
        head = f.read(20)
        if len(head) < 20 or head[:4] != b"glTF":
            return None
        chunk_len = struct.unpack_from("<I", head, 12)[0]
        if head[16:20] != b"JSON":
            return None
        raw = f.read(chunk_len)
    if len(raw) < chunk_len:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def read_glb(glb_path: str) -> Optional[tuple]:
    """(json, binary_chunk_or_None) of a GLB — the whole file, unlike read_gltf_json.

    Only needed when texture bytes must be decoded; the JSON-only reader is the one
    to use for declarations. None if the file is not a valid GLB.
    """
    with open(glb_path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b"glTF":
        return None
    off, js, binary = 12, None, None
    while off + 8 <= len(data):
        length = struct.unpack_from("<I", data, off)[0]
        ctype = data[off + 4:off + 8]
        body = data[off + 8:off + 8 + length]
        if ctype == b"JSON" and js is None:
            try:
                js = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
        elif ctype == b"BIN\x00" and binary is None:
            binary = body
        off += 8 + length
    return (js, binary) if js is not None else None


def _image_bytes(js: dict, binary: Optional[bytes], image_index: int, glb_path: str) -> Optional[bytes]:
    """Raw encoded bytes (PNG/JPEG) of ``images[image_index]``, embedded or external."""
    try:
        img = js["images"][image_index]
    except (KeyError, IndexError, TypeError):
        return None
    if "bufferView" in img:
        if binary is None:
            return None
        try:
            bv = js["bufferViews"][img["bufferView"]]
        except (KeyError, IndexError, TypeError):
            return None
        start = int(bv.get("byteOffset", 0))
        return binary[start:start + int(bv["byteLength"])]
    uri = img.get("uri")
    if not uri:
        return None
    if uri.startswith("data:"):
        import base64
        try:
            return base64.b64decode(uri.split(",", 1)[1])
        except (IndexError, ValueError):
            return None
    import os
    path = os.path.join(os.path.dirname(glb_path), uri)
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def emissive_texture_max(glb_path: str, only_used: bool = True) -> Dict[int, int]:
    """Max RGB byte (0..255) of each declared emissive texture, keyed by material index.

    The content half of the emission check. A material can DECLARE an emissive
    texture that is entirely black, and such a texture emits nothing whatever the
    factor and strength say. This decodes the image out of the GLB and reports its
    brightest RGB byte, so the caller can require it to be above zero (or above any
    threshold it prefers — the raw max is returned, not a verdict). Alpha is ignored.

    Only materials with a declared emissive texture appear in the result; a material
    whose texture cannot be located or decoded maps to -1. With ``only_used`` (the
    default) materials no mesh primitive references are skipped.

    Needs Pillow and numpy, imported lazily so the module still loads without them.
    This reads and decodes the whole file, so it is 10–100× the cost of the JSON
    readers: call it on the shapes that pass the declaration check, not the full set.
    """
    import io

    import numpy as np
    from PIL import Image

    parsed = read_glb(glb_path)
    if parsed is None:
        return {}
    js, binary = parsed
    used = used_material_indices(js)
    out: Dict[int, int] = {}
    cache: Dict[int, int] = {}  # image index -> max, several materials may share one image
    for i, mat in enumerate(js.get("materials") or []):
        tex_ref = mat.get("emissiveTexture")
        if not tex_ref or (only_used and i not in used):
            continue
        try:
            image_index = int(js["textures"][int(tex_ref["index"])]["source"])
        except (KeyError, IndexError, TypeError, ValueError):
            out[i] = -1
            continue
        if image_index not in cache:
            raw = _image_bytes(js, binary, image_index, glb_path)
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    arr = np.asarray(im.convert("RGB"))
                cache[image_index] = int(arr.max()) if arr.size else 0
            except Exception:  # PIL raises a zoo of types for a corrupt image
                cache[image_index] = -1
        out[i] = cache[image_index]
    return out


def used_material_indices(js: dict) -> set:
    """Indices of materials that some ``meshes[].primitives[].material`` references.

    glTF allows a listed material that no geometry references; such a material
    contributes nothing to what is drawn. Source GLBs do carry such materials, so the
    gate is real.
    """
    return {
        prim.get("material")
        for mesh in (js.get("meshes") or [])
        for prim in (mesh.get("primitives") or [])
        if "material" in prim
    }


def material_emission_from_json(js: dict) -> List[MaterialEmission]:
    """Per-material declared emission from an already-parsed glTF JSON (no file I/O).

    Use this when the JSON is already in hand (e.g. alongside
    ``utils.gltf_materials.materials_from_json``) to avoid parsing the file twice.
    """
    used = used_material_indices(js)
    out = []
    for i, mat in enumerate(js.get("materials") or []):
        factor = mat.get("emissiveFactor") or [0.0, 0.0, 0.0]
        try:
            factor = tuple(float(c) for c in factor[:3])
        except (TypeError, ValueError):
            factor = (0.0, 0.0, 0.0)
        ext = (mat.get("extensions") or {}).get(EXTENSION) or {}
        declared = "emissiveStrength" in ext
        try:
            strength = float(ext["emissiveStrength"]) if declared else DEFAULT_STRENGTH
        except (TypeError, ValueError):
            strength, declared = DEFAULT_STRENGTH, False
        out.append(
            MaterialEmission(
                index=i,
                emissive_factor=factor,
                has_emissive_texture="emissiveTexture" in mat,
                strength=strength,
                declares_strength=declared,
                is_used=i in used,
            )
        )
    return out


def read_material_emission(glb_path: str) -> List[MaterialEmission]:
    """Per-material declared emission, in glTF material order. [] if unreadable.

    ``is_used`` says whether any mesh primitive points at the material; a filter
    should test ``m.is_used and m.is_emissive`` (see ``used_material_indices``).
    """
    js = read_gltf_json(glb_path)
    return [] if js is None else material_emission_from_json(js)


def max_emissive_strength(glb_path: str) -> float:
    """Largest declared emissive strength across all materials, else 1.0.

    Byte-for-byte equivalent to the submitted paper's evaluator's ``parse_emissive_strength``,
    including its conventions: the max is taken over every material that carries the
    extension, emissive or not, used or not, and an unreadable or extension-free file
    yields 1.0 rather than an error. It
    deliberately does not apply the ``is_used`` gate, so it keeps matching the
    evaluator that produced the published numbers.
    """
    strengths = [m.strength for m in read_material_emission(glb_path) if m.declares_strength]
    return max(strengths) if strengths else DEFAULT_STRENGTH


def declares_emission(glb_path: str) -> bool:
    """True when any material that some mesh actually uses DECLARES emission
    (texture or factor > 0): the cheap, JSON-only, declaration-level test.

    This is the metadata filter's verdict
    (``data_processing/metadata_filtering/check_emission.py``). Whether the emission
    can actually show (black texture, unreachable mesh, transparent surface) is the
    heuristic stage's question.
    """
    return any(m.is_used and m.is_emissive for m in read_material_emission(glb_path))
