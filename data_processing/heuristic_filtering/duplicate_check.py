"""Duplicate check: drop the copies of one object, keeping one shape per group.

Third check of the heuristic stage, in three subcommands, run in this order:

  signatures  fingerprints every GLB by what it contains, not its bytes (decoded vertex
              positions rounded and sorted, image hashes, material declarations); no verdict.
  keys        reads again only the shapes that share a geometry hash with another: the image
              hashes of their drawn albedo and emission, their emission colors, and a strict
              key of the drawn object (placement, positions, indices, primitive mode, and the
              materials with their images).
  list        the copy groups from those two tables (`copy_groups` below); each group
              keeps its smallest uuid, and the pass list is the input list, in its order,
              minus the rest. Reads no GLB.

    conda run --no-capture-output -n lightgen python -m \\
      data_processing.heuristic_filtering.duplicate_check signatures --workers 16
    conda run --no-capture-output -n lightgen python -m \\
      data_processing.heuristic_filtering.duplicate_check keys --workers 16
    conda run --no-capture-output -n lightgen python -m \\
      data_processing.heuristic_filtering.duplicate_check list

The `signatures` CSV is the resume marker (see `utils/check_runner.py`).
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.check_runner import DEFAULT_GLB_ROOTS, DEFAULT_METADATA, run_check  # noqa: E402
from utils.glb_io import find_glb  # noqa: E402
from utils.gltf_emission import _image_bytes, read_glb  # noqa: E402
from utils.gltf_materials import drawn_materials  # noqa: E402
from utils.metadata_io import read_uuids, write_uuids  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

# --- inputs and outputs ---------------------------------------------------------------
#: Input: the albedo-copy check's pass list.
INPUT_LIST = OUT_DIR / "albedo_copy_pass.txt"
#: The signature pass's rows: what `signatures` writes and `keys` and `list` read.
SIGNATURES = OUT_DIR / "signatures.csv"
#: The candidates' albedo, emission and strict keys: what `keys` writes and `list` reads.
KEYS = OUT_DIR / "duplicate_keys.csv"
#: What `list` writes: the pass list and the copy groups.
PASS_LIST = OUT_DIR / "duplicate_pass.txt"
GROUPS = OUT_DIR / "duplicate_groups.csv"

# --- the signature pass ---------------------------------------------------------------
#: Vertex positions are rounded to this many decimals before hashing, so copies whose
#: positions differ only in float noise hash alike.
SIG_ROUND_DECIMALS = 5
#: Hex characters kept per hash. 16 = 64 bits: a chance collision among n shapes has
#: probability about n^2 / 2^65, and a collision would show up as a family whose members
#: differ elsewhere.
SIG_HASH_CHARS = 16
#: One row per shape; the pass only fingerprints, `list` removes the copies.
SIG_FIELDS = [
    "uuid",
    "status",            # ok | no_file | err:<name>
    "file_size",
    "n_meshes",
    "n_prims",           # primitives declared
    "n_prims_read",      # primitives whose POSITION was readable (Draco/external = not)
    "n_vertices",        # positions hashed
    "geom_hash",         # rounded, lexsorted positions
    "geom_hash_norm",    # same after centering and scaling to unit max extent
    "bbox_extent",       # "x,y,z" of the raw positions; diagnostic only
    "n_materials",
    "n_images",
    "tex_hash",          # hash of the sorted per-image byte hashes
    "image_hashes",      # the sorted per-image hashes, ";"-joined (capped, see below)
    "material_sig",      # hash of the sorted material declarations, names dropped
]
#: Filled with "" when the file cannot be read.
SIG_EMPTY_ON_FAILURE = [f for f in SIG_FIELDS if f not in ("uuid", "status")]
#: `image_hashes` is capped so one pathological shape cannot blow up the CSV; the
#: uncapped count stays in `n_images` and `tex_hash` always covers every image.
SIG_MAX_IMAGE_HASHES = 64


#: glTF componentType -> numpy dtype. Little-endian per the spec.
_DTYPE = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:SIG_HASH_CHARS]


def read_accessor(js: dict, binary: bytes, index: int):
    """The accessor's values as an (count, ncomp) array, or None if unreadable.

    Honors `byteStride`: interleaved bufferViews are common in these files. Returns None
    rather than raising for the layouts this pass does not resolve — a Draco-compressed
    primitive (no bufferView) or an accessor into a buffer that lives outside the GLB.
    """
    def at(seq, i):
        """seq[i] only for an in-range non-negative int — never Python's wrap-around."""
        if not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(seq):
            raise IndexError(i)
        return seq[i]

    try:
        acc = at(js["accessors"], index)
        if "bufferView" not in acc:          # Draco, or a sparse-only accessor
            return None
        if "sparse" in acc:
            return None                      # an override this pass does not apply
        if acc.get("normalized"):
            return None                      # quantized ints; raw values are not floats
        bv = at(js["bufferViews"], acc["bufferView"])
        if at(js["buffers"], bv.get("buffer", 0)).get("uri"):
            return None                      # payload is not in the BIN chunk
        dt = np.dtype(_DTYPE[acc["componentType"]])
        ncomp = _NCOMP[acc["type"]]
        n = int(acc["count"])
        view_start = int(bv.get("byteOffset", 0))
        view_len = int(bv["byteLength"])
        start = view_start + int(acc.get("byteOffset", 0))
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if n <= 0 or start < view_start:
        return None
    itemsize = ncomp * dt.itemsize
    stride = bv.get("byteStride") or itemsize
    # The last element ENDS at (n-1)*stride + itemsize; requiring n*stride would reject
    # a valid accessor whose final element finishes exactly at the view's end.
    need = (n - 1) * stride + itemsize
    if start + need > view_start + view_len or start + need > len(binary):
        return None
    if stride != itemsize:
        raw = np.frombuffer(binary, np.uint8, count=need, offset=start)
        padded = np.zeros(n * stride, np.uint8)     # the last row may be short
        padded[:need] = raw
        return padded.reshape(n, stride)[:, :itemsize].copy().view(dt).reshape(n, ncomp)
    return np.frombuffer(binary, dt, count=n * ncomp, offset=start).reshape(n, ncomp).copy()


def positions(js: dict, binary) -> tuple:
    """(list of (N,3) position arrays, primitives declared, primitives read)."""
    out, declared, read = [], 0, 0
    if binary is None:
        return out, declared, read
    for mesh in js.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            declared += 1
            idx = (prim.get("attributes") or {}).get("POSITION")
            if idx is None:
                continue
            v = read_accessor(js, binary, idx)
            if v is None or v.shape[1] != 3:
                continue
            out.append(np.asarray(v, np.float64))
            read += 1
    return out, declared, read


def _geometry_hashes(parts) -> tuple:
    """(geom_hash, geom_hash_norm, n_vertices, bbox_extent) for the position arrays.

    Adding 0.0 after rounding turns -0.0 into 0.0, so two files that differ only in the
    sign of a zero coordinate hash alike.
    """
    if not parts:
        return "", "", 0, ""
    v = np.concatenate(parts)

    def sorted_hash(a):
        a = np.round(a, SIG_ROUND_DECIMALS) + 0.0
        return _hash(np.ascontiguousarray(a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]).tobytes())

    lo, hi = v.min(0), v.max(0)
    extent = hi - lo
    scale = float(extent.max())
    norm = (v - (lo + hi) / 2.0) / (scale if scale > 0 else 1.0)
    return (sorted_hash(v), sorted_hash(norm), len(v),
            ",".join(f"{x:.6f}" for x in extent))


def _material_sig(js: dict) -> str:
    """Hash of the material declarations, sorted, with the authored names dropped.

    A re-texture variant usually keeps the material layout and changes only which image
    each slot points at, so this separates "same materials" from "same pictures".
    """
    mats = js.get("materials") or []
    canon = []
    for m in mats:
        m = {k: v for k, v in (m or {}).items() if k not in ("name", "extras")}
        canon.append(json.dumps(m, sort_keys=True, separators=(",", ":")))
    return _hash("\n".join(sorted(canon)).encode()) if canon else ""


def signatures(js: dict, binary) -> dict:
    """Every fingerprint of one parsed GLB. Never raises on odd content."""
    parts, declared, read = positions(js, binary)
    # A partial read must not supply an equality key: hashing only the primitives that
    # decoded makes the shape indistinguishable from one that genuinely has just those.
    complete = declared > 0 and read == declared
    geom, geom_norm, nverts, extent = _geometry_hashes(parts if complete else [])
    if not complete:
        nverts = sum(len(p) for p in parts)
    img_hashes = []
    for i in range(len(js.get("images") or [])):
        raw = _image_bytes(js, binary, i, "")
        img_hashes.append(_hash(raw) if raw else "none")
    img_hashes.sort()
    return {
        "n_meshes": len(js.get("meshes") or []),
        "n_prims": declared,
        "n_prims_read": read,
        "n_vertices": nverts,
        "geom_hash": geom,
        "geom_hash_norm": geom_norm,
        "bbox_extent": extent,
        "n_materials": len(js.get("materials") or []),
        "n_images": len(img_hashes),
        "tex_hash": _hash(";".join(img_hashes).encode()) if img_hashes else "",
        "image_hashes": ";".join(img_hashes[:SIG_MAX_IMAGE_HASHES]),
        "material_sig": _material_sig(js),
    }


def check_one(sha: str, metadata, roots) -> dict:
    """One CSV row. Read failures land in `status`; this never raises."""
    empty = {k: "" for k in SIG_EMPTY_ON_FAILURE}
    path = find_glb(sha, metadata, roots)
    if not path:
        return {"uuid": sha, "status": "no_file", **empty}
    try:
        size = os.path.getsize(path)
        parsed = read_glb(path)
    except OSError as exc:
        return {"uuid": sha, "status": "err:" + type(exc).__name__, **empty}
    if parsed is None or parsed[0] is None:
        return {"uuid": sha, "status": "err:BadGLB", **empty}
    js, binary = parsed
    try:
        sig = signatures(js, binary)
    except Exception as exc:                 # a malformed accessor must not kill the run
        return {"uuid": sha, "status": "err:" + type(exc).__name__, **empty}
    return {"uuid": sha, "status": "ok", "file_size": size, **sig}


# --- the keys of the candidates -----------------------------------------------------------
#: Columns of the `keys` CSV. `albedo` and `emissive`: the ";"-joined sorted image hashes in
#: those slots of the drawn materials; `emissive_factors`: what `factor_colors` returns;
#: `strict_key`: what `strict_key` returns.
KEY_FIELDS = ["uuid", "status", "albedo", "emissive", "emissive_factors", "strict_key"]


def slot_hashes(js, binary) -> dict:
    """{"albedo" | "emissive": frozenset of image hashes} over the materials the scene draws.

    An image that cannot be read hashes to "unreadable_<index>".
    """
    _, drawn = drawn_materials(js)
    mats = js.get("materials") or []
    textures = js.get("textures") or []
    cache = {}

    def img_hash(info):
        tex_i = (info or {}).get("index")
        if not isinstance(tex_i, int) or not (0 <= tex_i < len(textures)):
            return None
        src = textures[tex_i].get("source")
        if not isinstance(src, int):
            return None
        if src not in cache:
            raw = _image_bytes(js, binary, src, "")
            cache[src] = _hash(raw) if raw else f"unreadable_{src}"
        return cache[src]

    out = {"albedo": set(), "emissive": set()}
    for i in sorted(drawn):
        if not (0 <= i < len(mats)):
            continue
        mat = mats[i] or {}
        for slot, info in (("albedo", (mat.get("pbrMetallicRoughness") or {}).get("baseColorTexture")),
                           ("emissive", mat.get("emissiveTexture"))):
            h = img_hash(info)
            if h:
                out[slot].add(h)
    return {s: frozenset(v) for s, v in out.items()}


def factor_colors(js) -> str:
    """Sorted non-zero emissive factors of drawn materials without an emissive texture."""
    _, drawn = drawn_materials(js)
    mats = js.get("materials") or []
    out = []
    for i in sorted(drawn):
        if not (0 <= i < len(mats)):
            continue
        m = mats[i] or {}
        if "emissiveTexture" in m:
            continue
        f = [round(float(x), 4) for x in (m.get("emissiveFactor") or [0, 0, 0])[:3]]
        if any(c > 0 for c in f):
            out.append(tuple(f))
    return json.dumps(sorted(out))


def _node_matrix(node) -> np.ndarray:
    """The node's local transform, from `matrix` or from TRS."""
    if "matrix" in node:                       # glTF stores it column-major
        return np.array(node["matrix"], np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in node:
        m = np.diag(list(node["scale"]) + [1.0]) @ m
    if "rotation" in node:                     # quaternion (x, y, z, w)
        x, y, z, w = node["rotation"]
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)
        rm = np.eye(4)
        rm[:3, :3] = r
        m = rm @ m
    if "translation" in node:
        t = np.eye(4)
        t[:3, 3] = node["translation"]
        m = t @ m
    return m


def _resolved_material(js, mat_index, img_hashes) -> str:
    """The material declaration, names dropped, with every texture index replaced by its
    image's hash: renumbering `textures[]` is then invisible, swapping two images is not."""
    mats = js.get("materials") or []
    if mat_index is None or not (0 <= mat_index < len(mats)):
        return "none"
    m = json.loads(json.dumps({k: v for k, v in (mats[mat_index] or {}).items()
                               if k not in ("name", "extras")}))

    def walk(o):
        if isinstance(o, dict):
            if "index" in o and isinstance(o["index"], int):
                tex = (js.get("textures") or [])
                src = tex[o["index"]].get("source") if 0 <= o["index"] < len(tex) else None
                o["index"] = img_hashes[src] if src is not None and 0 <= src < len(img_hashes) else "?"
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(m)
    return json.dumps(m, sort_keys=True, separators=(",", ":"))


def strict_key(js, binary) -> str:
    """One hash of every drawn primitive, independent of their order: its world matrix
    (rounded to 1e-6), its positions and indices at full precision, its primitive mode and
    its resolved material. What `signatures` leaves out (topology, placement, which material
    and image sit on which piece) enters here. An image that cannot be read enters as
    "unreadable_<index>", as in `slot_hashes`. "" (never matches) when nothing is drawn.
    Raises on what it cannot hash faithfully: a bad `scene`, node or mesh index, a node
    reached twice, or a declared POSITION or indices accessor that `read_accessor` cannot
    read; `keys_row` then writes "". Skipping or stubbing those could make two different
    files match."""
    def at(seq, i, what):
        if not isinstance(i, int) or isinstance(i, bool) or not (0 <= i < len(seq)):
            raise ValueError(f"bad {what} index {i!r}")
        return seq[i]

    img_hashes = []
    for i in range(len(js.get("images") or [])):
        raw = _image_bytes(js, binary, i, "")
        img_hashes.append(_hash(raw) if raw else f"unreadable_{i}")
    nodes = js.get("nodes") or []
    meshes = js.get("meshes") or []
    scenes = js.get("scenes") or []
    roots = at(scenes, js.get("scene", 0), "scene").get("nodes", []) if scenes else []
    descs, stack = [], [(r, np.eye(4)) for r in roots]
    seen = set()
    while stack:
        idx, parent = stack.pop()
        node = at(nodes, idx, "node") or {}
        if idx in seen:
            raise ValueError(f"node {idx} reached twice")
        seen.add(idx)
        world = parent @ _node_matrix(node)
        for child in node.get("children", []):
            stack.append((child, world))
        if "mesh" not in node:
            continue
        wm = _hash(np.round(world, 6).astype(np.float64).tobytes())
        for prim in (at(meshes, node["mesh"], "mesh").get("primitives") or []):
            parts = [wm]
            for name, owner, missing in (("POSITION", prim.get("attributes", {}), "nopos"),
                                         ("indices", prim, "noidx")):
                if name not in owner:          # absent; a declared null is a bad index below
                    parts.append(missing)
                    continue
                acc = owner[name]
                arr = read_accessor(js, binary, acc)
                if arr is None:
                    raise ValueError(f"unreadable {name} accessor {acc!r}")
                parts.append(_hash(np.ascontiguousarray(arr).tobytes()))
            parts += [str(prim.get("mode", 4)),
                      _hash(_resolved_material(js, prim.get("material"), img_hashes).encode())]
            descs.append("|".join(parts))
    return _hash("\n".join(sorted(descs)).encode()) if descs else ""


def keys_row(u: str, meta, roots) -> list:
    """One `keys` row (KEY_FIELDS). Read failures land in `status`; this never raises."""
    empty = ["", "", "", ""]
    try:
        p = find_glb(u, meta, roots)
        if not p:
            return [u, "no_file"] + empty
        parsed = read_glb(p)
        if parsed is None or parsed[0] is None:
            return [u, "bad_glb"] + empty
        js, binary = parsed
        s = slot_hashes(js, binary)
        # an unreadable image hashes to "unreadable_<index>", a value two files can share
        st = "unreadable_image" if any(h.startswith("unreadable_") for h in
                                       s["albedo"] | s["emissive"]) else "ok"
        try:
            sk = strict_key(js, binary)
        except Exception:                    # a bad node tree costs only the identical step
            sk = ""
        return [u, st, ";".join(sorted(s["albedo"])), ";".join(sorted(s["emissive"])),
                factor_colors(js), sk]
    except Exception as exc:
        return [u, "err:" + type(exc).__name__] + empty


# --- the rule ------------------------------------------------------------------------------
GEOM_KEYS = ("geom_hash", "geom_hash_norm")
#: The steps of the rule that make groups, in the order the groups CSV lists them.
STEPS = ("identical", "same_mesh", "rescaled")


def load_signatures(list_path=None, sig_path=None) -> pd.DataFrame:
    """The `ok` signature rows of the input list's shapes (defaults: `INPUT_LIST`, `SIGNATURES`)."""
    keep = set(read_uuids(list_path or INPUT_LIST))
    sig = pd.read_csv(sig_path or SIGNATURES, dtype=str, keep_default_na=False)
    sig = sig[sig.uuid.isin(keep)]
    if sig.uuid.duplicated().any():          # a shape twice would be its own copy
        sys.exit(f"{sig.uuid.duplicated().sum()} uuids of the list have more than one row in "
                 f"{sig_path or SIGNATURES}")
    return sig[sig.status == "ok"]


def families(rows, cols) -> list:
    """The uuids that share every one of `cols` with another row, in groups of 2+.
    An empty value never matches."""
    by = collections.defaultdict(list)
    for r in rows:
        v = tuple(r[c] for c in cols)
        if all(v):
            by[v].append(r["uuid"])
    return [m for m in by.values() if len(m) > 1]


def candidates(rows) -> list:
    """The shapes `keys` reads: those that share either geometry hash with another."""
    return sorted({u for gk in GEOM_KEYS for m in families(rows, (gk,)) for u in m})


def copy_groups(rows, keys: dict) -> list:
    """[(step, sorted members)], every group of 2+ copies the rule finds.

    `rows` are the signature rows (dicts), `keys` maps uuid -> `keys` row of every candidate.
    A candidate whose `keys` status is not `ok` is kept and never joins a group.

    1. identical: the shapes sharing (geom_hash, tex_hash, material_sig) are the identical-
       content families; their members, all families together, are grouped by strict key.
    2. same_mesh: each geom_hash family minus every member of an identical-content family
       (split by the strict key or not) is a group when all its members have one albedo.
    3. rescaled: each geom_hash_norm family whose members do not all have one geom_hash,
       taken whole, is a group when they all have one tex_hash or one albedo.
    The same_mesh and rescaled groups are then split by emission (`emissive`,
    `emissive_factors`); the identical groups already agree on it.
    """
    sig = {r["uuid"]: r for r in rows}
    ok = lambda u: keys[u]["status"] == "ok"
    albedo = lambda u: keys[u]["albedo"]
    groups = []
    identical = families(rows, ("geom_hash", "tex_hash", "material_sig"))
    by_strict = collections.defaultdict(list)
    for m in identical:
        for u in m:
            if ok(u) and keys[u]["strict_key"]:
                by_strict[keys[u]["strict_key"]].append(u)
    groups += [("identical", m) for m in by_strict.values() if len(m) > 1]

    in_identical = {u for m in identical for u in m}
    by_rule = []
    for m in families(rows, ("geom_hash",)):
        rest = [u for u in m if u not in in_identical and ok(u)]
        if len(rest) > 1 and len({albedo(u) for u in rest}) == 1:
            by_rule.append(("same_mesh", rest))
    for m in families(rows, ("geom_hash_norm",)):
        m = [u for u in m if ok(u)]
        if len(m) > 1 and len({sig[u]["geom_hash"] for u in m}) > 1 and (
                len({sig[u]["tex_hash"] for u in m}) == 1 or len({albedo(u) for u in m}) == 1):
            by_rule.append(("rescaled", m))
    for step, m in by_rule:
        by_emission = collections.defaultdict(list)
        for u in m:
            by_emission[(keys[u]["emissive"], keys[u]["emissive_factors"])].append(u)
        groups += [(step, g) for g in by_emission.values() if len(g) > 1]
    return sorted(((s, sorted(m)) for s, m in groups), key=lambda g: (STEPS.index(g[0]), g[1]))


def cmd_keys(a):
    shapes = candidates(load_signatures(a.list, a.signatures).to_dict("records"))
    if a.limit:
        shapes = shapes[:a.limit]
    print(f"candidates (share a geometry hash): {len(shapes)}", flush=True)
    meta = pd.read_parquet(a.metadata, columns=["glb_1k_path"])
    t0 = time.time()
    with open(a.out, "w", newline="") as f, ThreadPoolExecutor(a.workers) as ex:
        w = csv.writer(f)
        w.writerow(KEY_FIELDS)
        status = collections.Counter()
        for i, row in enumerate(ex.map(lambda u: keys_row(u, meta, a.roots), shapes), 1):
            w.writerow(row)
            status[row[1]] += 1
            if i % 500 == 0:
                print(f"{i}/{len(shapes)} {time.time() - t0:.0f}s", flush=True)
    print("status: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())))
    print(f"keys -> {a.out}")


def cmd_list(a):
    order = read_uuids(a.list)
    rows = load_signatures(a.list, a.signatures).to_dict("records")
    keys = pd.read_csv(a.keys, dtype=str, keep_default_na=False)
    if list(keys.columns) != KEY_FIELDS:
        sys.exit(f"{a.keys} has the columns {list(keys.columns)}, not {KEY_FIELDS}; rerun `keys`")
    keys = keys.set_index("uuid").to_dict("index")
    cand = candidates(rows)
    missing = set(cand) - set(keys)
    if missing:
        sys.exit(f"{len(missing)} candidates have no row in {a.keys}; rerun `keys`")
    bad = sum(keys[u]["status"] != "ok" for u in cand)
    print(f"candidates: {len(cand)}, unreadable (kept, never merged): {bad}")
    groups = copy_groups(rows, keys)
    removed = {u for _, m in groups for u in m[1:]}
    kept = [u for u in order if u not in removed]

    write_uuids(a.out, kept)
    with open(a.groups, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "step", "uuid", "kept"])
        for gi, (step, m) in enumerate(groups):
            for u in m:
                w.writerow([gi, step, u, int(u not in removed)])
    for step in STEPS:
        g = [m for s, m in groups if s == step]
        print(f"{step}: {len(g)} groups over {sum(map(len, g))} shapes, largest {max(map(len, g), default=0)}")
    print(f"removed {len(removed)}, kept {len(kept)} of {len(order)}")
    print(f"list -> {a.out}\ngroups -> {a.groups}")


def main():
    # `signatures` is the pass: the shared runner parses its flags (`--list`, `--out`,
    # `--workers`, `--limit`, ...).
    if sys.argv[1:2] == ["signatures"]:
        sys.argv[1:] = sys.argv[2:]
        run_check(check_one, SIG_FIELDS, (), INPUT_LIST, SIGNATURES, "", __doc__)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("signatures", add_help=False,
                   help="the signature pass, a resumable CSV; `signatures -h` lists its flags")
    k = sub.add_parser("keys", help="the candidates' albedo, emission and strict keys (reads their GLBs)")
    l = sub.add_parser("list", help="the copy groups and the pass list (reads no GLB)")
    for p in (k, l):
        p.add_argument("--list", default=str(INPUT_LIST), help="uuid list (default: the albedo-copy pass list)")
        p.add_argument("--signatures", default=str(SIGNATURES),
                       help="the `signatures` CSV of that list (default: what `signatures` writes)")
    k.add_argument("--metadata", default=str(DEFAULT_METADATA))
    k.add_argument("--roots", nargs="+", default=[str(r) for r in DEFAULT_GLB_ROOTS])
    k.add_argument("--workers", type=int, default=16)
    k.add_argument("--limit", type=int, default=0, help="first N candidates only (smoke test)")
    k.add_argument("--out", default=str(KEYS))
    l.add_argument("--keys", default=str(KEYS), help="the `keys` CSV (default: what `keys` writes)")
    l.add_argument("--out", default=str(PASS_LIST))
    l.add_argument("--groups", default=str(GROUPS))
    a = ap.parse_args()
    {"keys": cmd_keys, "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    main()
