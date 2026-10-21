"""Reader for a glTF/GLB asset's declared material channels.

What each material declares (base color, metallic/roughness, normal, occlusion,
emissive, alpha mode, the unlit and specular-glossiness extensions), read from the
GLB's JSON chunk with the standard library. Companion to ``gltf_emission`` (which
answers "does it emit") and the reader behind the metadata-filtering scan's channel
columns.

    from utils.gltf_materials import read_glb_materials, summarize_channels

Two rollups:
  * ``summarize_channels(g, only_used=True)``: the filter's view, materials that
    some mesh primitive actually references (what is drawn).
  * ``summarize_channels(g, only_used=False)``: the authoring view, any material
    in the file, referenced or not.

There is no single "is PBR" switch in glTF 2.0 (every material gets a
``pbrMetallicRoughness`` block by default), so PBR-ness is read off which maps and
extensions are present. ``KHR_materials_unlit`` is the only explicit "not PBR"
declaration; ``KHR_materials_pbrSpecularGlossiness`` is the legacy spec/gloss workflow.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from utils.gltf_emission import read_gltf_json, used_material_indices

__all__ = [
    "MaterialChannels",
    "GlbMaterials",
    "materials_from_json",
    "read_glb_materials",
    "summarize_channels",
    "base_color_texture_index",
    "drawn_materials",
    "CHANNEL_COUNT_FIELDS",
    "base_color_alpha_max",
    "base_color_alpha_min",
]

UNLIT = "KHR_materials_unlit"
SPEC_GLOSS = "KHR_materials_pbrSpecularGlossiness"


@dataclass(frozen=True)
class MaterialChannels:
    """One material's declared channels. Defaults are glTF's own defaults."""

    index: int
    name: str
    is_used: bool  #: referenced by at least one mesh primitive
    base_color_factor: tuple  #: (r, g, b, a), default (1, 1, 1, 1)
    has_base_color_texture: bool
    metallic_factor: float  #: default 1.0
    roughness_factor: float  #: default 1.0
    has_metallic_roughness_texture: bool
    has_normal_texture: bool
    has_occlusion_texture: bool
    has_emissive_texture: bool
    emissive_factor: tuple  #: (r, g, b), default (0, 0, 0)
    alpha_mode: str  #: OPAQUE (default) | MASK | BLEND
    double_sided: bool
    extensions: tuple  #: extension names on this material
    #: "Defined" = the file WRITES the channel, as a texture or a number. A material
    #: that writes nothing renders with glTF's defaults (white albedo, metallic 1.0,
    #: roughness 1.0), which is bare metal and rarely what the artist meant; these
    #: flags tell that case apart from an explicit value.
    #:
    #: A KHR_materials_pbrSpecularGlossiness material writes the same information in
    #: the older form (diffuse, specular, glossiness), and every loader here (trimesh,
    #: hence the bake) converts it: diffuse -> base color, glossiness -> roughness,
    #: specular -> metallic. So for those materials the flags read the spec/gloss keys.
    declares_base_color: bool = False  #: baseColorTexture/baseColorFactor, or diffuseTexture/diffuseFactor
    declares_metallic: bool = False  #: metallicRoughnessTexture/metallicFactor, or specularGlossinessTexture/specularFactor
    declares_roughness: bool = False  #: metallicRoughnessTexture/roughnessFactor, or specularGlossinessTexture/glossinessFactor

    @property
    def declares_pbr(self) -> bool:
        """Albedo, metallic and roughness all written (texture or number)."""
        return self.declares_base_color and self.declares_metallic and self.declares_roughness

    @property
    def is_unlit(self) -> bool:
        return UNLIT in self.extensions

    @property
    def is_spec_gloss(self) -> bool:
        return SPEC_GLOSS in self.extensions

    @property
    def nondefault_metal(self) -> bool:
        """metallicFactor declared and not 1.0."""
        return self.metallic_factor != 1.0

    @property
    def nondefault_rough(self) -> bool:
        return self.roughness_factor != 1.0


@dataclass(frozen=True)
class GlbMaterials:
    materials: List[MaterialChannels]
    extensions_used: tuple  #: file-level `extensionsUsed`, sorted
    n_images: int


def _f(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _tuple(v, default):
    try:
        return tuple(float(c) for c in v[: len(default)])
    except (TypeError, ValueError):
        return default


def materials_from_json(js: dict) -> GlbMaterials:
    """Build the material table from an already-parsed glTF JSON (no file I/O)."""
    used = used_material_indices(js)
    out = []
    for i, mat in enumerate(js.get("materials") or []):
        pbr = mat.get("pbrMetallicRoughness") or {}
        sg = (mat.get("extensions") or {}).get(SPEC_GLOSS) or {}
        out.append(
            MaterialChannels(
                index=i,
                name=str(mat.get("name", "")),
                is_used=i in used,
                base_color_factor=_tuple(pbr.get("baseColorFactor"), (1.0, 1.0, 1.0, 1.0)),
                has_base_color_texture="baseColorTexture" in pbr or "diffuseTexture" in sg,
                metallic_factor=_f(pbr.get("metallicFactor"), 1.0),
                roughness_factor=_f(pbr.get("roughnessFactor"), 1.0),
                has_metallic_roughness_texture="metallicRoughnessTexture" in pbr,
                has_normal_texture="normalTexture" in mat,
                has_occlusion_texture="occlusionTexture" in mat,
                has_emissive_texture="emissiveTexture" in mat,
                emissive_factor=_tuple(mat.get("emissiveFactor"), (0.0, 0.0, 0.0)),
                alpha_mode=str(mat.get("alphaMode", "OPAQUE")),
                double_sided=bool(mat.get("doubleSided", False)),
                extensions=tuple(sorted((mat.get("extensions") or {}).keys())),
                declares_base_color=("baseColorTexture" in pbr or "baseColorFactor" in pbr
                                     or "diffuseTexture" in sg or "diffuseFactor" in sg),
                declares_metallic=("metallicRoughnessTexture" in pbr or "metallicFactor" in pbr
                                   or "specularGlossinessTexture" in sg or "specularFactor" in sg),
                declares_roughness=("metallicRoughnessTexture" in pbr or "roughnessFactor" in pbr
                                    or "specularGlossinessTexture" in sg or "glossinessFactor" in sg),
            )
        )
    return GlbMaterials(
        materials=out,
        extensions_used=tuple(sorted(set(js.get("extensionsUsed") or []))),
        n_images=len(js.get("images") or []),
    )


def base_color_texture_index(material_json: dict) -> Optional[int]:
    """The material's base-color texture index, whichever workflow wrote it:
    ``pbrMetallicRoughness.baseColorTexture`` or, for spec/gloss, the extension's
    ``diffuseTexture``. None when the material has no color texture.

    Use this, not ``baseColorTexture`` alone, when comparing an emissive texture against
    the albedo: the Sketchfab exporter copies the diffuse map into the emissive slot in
    both workflows.
    """
    pbr = material_json.get("pbrMetallicRoughness") or {}
    sg = (material_json.get("extensions") or {}).get(SPEC_GLOSS) or {}
    tex = pbr.get("baseColorTexture") or sg.get("diffuseTexture") or {}
    idx = tex.get("index")
    return int(idx) if idx is not None else None


def read_glb_materials(glb_path: str) -> Optional[GlbMaterials]:
    """Material table of a GLB, from its JSON chunk only. None if not a valid GLB."""
    js = read_gltf_json(glb_path)
    return None if js is None else materials_from_json(js)


def drawn_materials(js: dict) -> Tuple[bool, Dict[int, Set[int]]]:
    """(has_scene, {material index: primitive modes it is drawn with}).

    Walks the nodes reachable from `scenes[scene]` (default scene 0, as trimesh and
    every viewer do), through `children`, and collects the materials of every
    primitive of every mesh those nodes instantiate. Cycles and bad indices are
    ignored. A file with no `scenes` draws nothing: (False, {}).

    The three heuristic checks (non-zero emission, albedo copy, duplicates) use it.
    """
    scenes = js.get("scenes") or []
    if not scenes:
        return False, {}
    idx = js.get("scene", 0)
    if not isinstance(idx, int) or not 0 <= idx < len(scenes):
        idx = 0
    nodes = js.get("nodes") or []
    meshes = js.get("meshes") or []
    out: Dict[int, Set[int]] = {}
    seen: Set[int] = set()
    queue = deque(scenes[idx].get("nodes") or [])
    while queue:
        n = queue.popleft()
        if not isinstance(n, int) or n in seen or not 0 <= n < len(nodes):
            continue
        seen.add(n)
        node = nodes[n] or {}
        queue.extend(node.get("children") or [])
        m = node.get("mesh")
        if isinstance(m, int) and 0 <= m < len(meshes):
            for prim in (meshes[m] or {}).get("primitives") or []:
                if "material" in prim:
                    out.setdefault(prim["material"], set()).add(int(prim.get("mode", 4)))
    return True, out


#: Per-shape rollup columns, in output order. Counts of MATERIALS having each channel.
CHANNEL_COUNT_FIELDS = [
    "n_basecolor_tex",
    "n_mr_tex",
    "n_normal_tex",
    "n_occlusion_tex",
    "n_unlit",
    "n_specgloss",
    "n_nondefault_metal",
    "n_nondefault_rough",
    "n_alpha_blend",
    "n_alpha_mask",
    "n_albedo_defined",      # base-color texture or baseColorFactor written
    "n_metallic_defined",    # metallic-roughness texture or metallicFactor written
    "n_roughness_defined",   # metallic-roughness texture or roughnessFactor written
    "n_pbr_defined",         # all three written on the same material
    "n_images",
]


def summarize_channels(g: GlbMaterials, only_used: bool = True) -> Dict[str, int]:
    """Count, per channel, how many materials declare it (``CHANNEL_COUNT_FIELDS``).

    ``only_used`` restricts to materials some mesh primitive references, the right
    view for "what will be drawn". ``only_used=False`` is the authoring view (any
    material in the file).
    """
    mats = [m for m in g.materials if m.is_used] if only_used else list(g.materials)
    return {
        "n_basecolor_tex": sum(m.has_base_color_texture for m in mats),
        "n_mr_tex": sum(m.has_metallic_roughness_texture for m in mats),
        "n_normal_tex": sum(m.has_normal_texture for m in mats),
        "n_occlusion_tex": sum(m.has_occlusion_texture for m in mats),
        "n_unlit": sum(m.is_unlit for m in mats),
        "n_specgloss": sum(m.is_spec_gloss for m in mats),
        "n_nondefault_metal": sum(m.nondefault_metal for m in mats),
        "n_nondefault_rough": sum(m.nondefault_rough for m in mats),
        "n_alpha_blend": sum(m.alpha_mode == "BLEND" for m in mats),
        "n_alpha_mask": sum(m.alpha_mode == "MASK" for m in mats),
        "n_albedo_defined": sum(m.declares_base_color for m in mats),
        "n_metallic_defined": sum(m.declares_metallic for m in mats),
        "n_roughness_defined": sum(m.declares_roughness for m in mats),
        "n_pbr_defined": sum(m.declares_pbr for m in mats),
        "n_images": g.n_images,
    }


def _base_color_alpha(glb_path: str, only_used: bool, want_min: bool,
                      include_spec_gloss: bool) -> Dict[int, int]:
    """Shared body of ``base_color_alpha_max`` and ``base_color_alpha_min``.

    One decode per image (cached), then the max or the min over its alpha band. An
    image with no alpha channel maps to 255 (opaque); one that cannot be located or
    decoded maps to -1. ``include_spec_gloss`` also reads the legacy spec/gloss
    ``diffuseTexture`` (see ``base_color_texture_index``).
    """
    import io

    import numpy as np
    from PIL import Image

    from utils.gltf_emission import _image_bytes, read_glb

    parsed = read_glb(glb_path)
    if parsed is None:
        return {}
    js, binary = parsed
    used = used_material_indices(js)
    out: Dict[int, int] = {}
    cache: Dict[int, int] = {}
    for i, mat in enumerate(js.get("materials") or []):
        pbr = mat.get("pbrMetallicRoughness") or {}
        tex_ref = pbr.get("baseColorTexture")
        if tex_ref is None and include_spec_gloss:
            tex_ref = ((mat.get("extensions") or {}).get(SPEC_GLOSS) or {}).get("diffuseTexture")
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
                    # tRNS transparency can sit on RGB / L / P images too; convert applies it.
                    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
                        arr = np.asarray(im.convert("RGBA"))[..., 3]
                        if not arr.size:
                            cache[image_index] = 0
                        else:
                            cache[image_index] = int(arr.min()) if want_min else int(arr.max())
                    else:
                        im.load()             # PIL opens lazily; a corrupt body must still raise
                        cache[image_index] = 255
            except Exception:  # PIL raises a zoo of types for a corrupt image
                cache[image_index] = -1
        out[i] = cache[image_index]
    return out


def base_color_alpha_max(glb_path: str, only_used: bool = True) -> Dict[int, int]:
    """Max alpha byte (0..255) of each declared base-color texture, keyed by material index.

    The visibility half of an emission check: an emissive material whose surface is
    fully transparent (``alphaMode`` BLEND with alpha 0, or MASK with every texel
    below the cutoff) is never seen. The texture's alpha is one factor of that;
    the caller multiplies in ``baseColorFactor[3]`` and applies the mode.

    Only materials with a ``pbrMetallicRoughness.baseColorTexture`` appear in the
    result; a spec/gloss ``diffuseTexture`` does NOT, unlike
    ``base_color_texture_index``. That is deliberate: the heuristic filtering stage's
    published counts come from this function's verdicts, so widening it would move
    them. ``base_color_alpha_min`` reads both workflows.

    An image with no alpha channel maps to 255 (opaque); one that cannot be located
    or decoded maps to -1. With ``only_used`` (the default) materials no primitive
    references are skipped. Same cost profile as
    ``gltf_emission.emissive_texture_max``: decodes the file, so call it on the
    shapes that need it, not the full set.
    """
    return _base_color_alpha(glb_path, only_used, want_min=False, include_spec_gloss=False)


def base_color_alpha_min(glb_path: str, only_used: bool = True) -> Dict[int, int]:
    """Min alpha byte (0..255) of each declared base-color texture, keyed by material index.

    The counterpart of ``base_color_alpha_max``: "is any texel transparent", the
    question the bake asks. A base-color texture with a texel below 255 is resampled
    in premultiplied-alpha space by PIL whenever the voxelizer has to stretch it to
    square power-of-two, which blacks out the color under the transparent texels
    (``glb2representations/voxels.py``); the scan pairs this with the image size.

    Both base-color workflows count here, the metallic/roughness ``baseColorTexture``
    and the legacy spec/gloss ``diffuseTexture``, because trimesh converts the latter
    and the bake therefore resizes it too. Conventions otherwise as in
    ``base_color_alpha_max``: no alpha channel is 255, undecodable is -1, and the
    file is decoded, so call it on the shapes that need it.
    """
    return _base_color_alpha(glb_path, only_used, want_min=True, include_spec_gloss=True)
