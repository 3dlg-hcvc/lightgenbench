"""Metadata filtering: select the emissive shapes of TexVerse from the GLB JSON alone.

Three subcommands: `scan` reads every downloaded GLB's material declarations into a
resumable CSV, `list` applies the stage's two rules (declares emission; writes albedo,
metallic and roughness) to that CSV and writes the pass list, and `specgloss_rescan`
re-reads the spec/gloss shapes of a scan made before the scan read the spec/gloss keys.

    conda run --no-capture-output -n lightgen python -m \\
      data_processing.metadata_filtering.check_emission scan \\
      --out data_processing/metadata_filtering/out/gltf_emission_scan.csv --workers 32
    conda run --no-capture-output -n lightgen python -m \\
      data_processing.metadata_filtering.check_emission list \\
      --csv data_processing/metadata_filtering/out/gltf_emission_scan.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.check_runner import DEFAULT_GLB_ROOTS, DEFAULT_METADATA, DEFAULT_WORKERS, resume_done, write_as_completed  # noqa: E402
from utils.glb_io import find_glb  # noqa: E402
from utils.gltf_emission import material_emission_from_json, read_gltf_json  # noqa: E402
from utils.gltf_materials import CHANNEL_COUNT_FIELDS, materials_from_json, summarize_channels  # noqa: E402
from utils.metadata_io import write_uuids  # noqa: E402
from utils.paths import TEXVERSE_THUMBNAIL_ROOT  # noqa: E402

#: Where the metadata's `thumbnail_path` is relative to.
DEFAULT_THUMBNAIL_ROOT = TEXVERSE_THUMBNAIL_ROOT

#: Columns of the GLB index (`glb_index.py`) the scan reads. The shape id is the index.
DOWNLOADED_COLUMN = "if_downloaded_glb"   # bool: the GLB is on disk
THUMBNAIL_COLUMN = "thumbnail_path"       # relative to DEFAULT_THUMBNAIL_ROOT

#: One row per shape in the scan CSV. Order is the column order.
FIELDS = [
    "uuid",
    "status",                 # ok | no_file | err:<ExceptionName>
    "n_materials",
    "n_unused_materials",     # listed but referenced by no mesh primitive; excluded from the counts below
    "n_emissive_materials",   # USED materials that declare emission (texture OR factor > 0)
    "n_emissive_texture",     # USED materials that declare an emissive texture (content NOT checked here)
    "n_factor_positive",      # USED materials with emissiveFactor > 0 (the render-visible sense)
    "n_texture_only",         # USED materials with an emissive texture but a zero factor -> renders black in a strict renderer
    "max_strength",           # max KHR_materials_emissive_strength, 1.0 if absent
    "declares_strength",      # 1 if any material carried the extension
    "passes",                 # THE VERDICT (loose rule, declaration level): 1 if any used material declares an emissive texture, or has factor > 0 without one
    # --- material channel inventory (RECORD ONLY, never part of the verdict).
    # Counts of USED materials declaring each channel, from utils.gltf_materials.summarize_channels:
    "n_basecolor_tex", "n_mr_tex", "n_normal_tex", "n_occlusion_tex",
    "n_unlit", "n_specgloss", "n_nondefault_metal", "n_nondefault_rough",
    "n_alpha_blend", "n_alpha_mask",
    # "defined" = written in the file as a texture OR a number; a
    # material that writes nothing renders with glTF's defaults (metallic 1, roughness 1)
    "n_albedo_defined", "n_metallic_defined", "n_roughness_defined", "n_pbr_defined",
    "n_images",               # images in the file (file-level, not per material)
]
#: Fields that are blank on a failed row.
EMPTY_ON_FAILURE = FIELDS[2:]
#: The columns `specgloss_rescan` writes and `list --specgloss_rescan` overrides.
SPECGLOSS_FIELDS = ["uuid", "n_albedo_defined", "n_metallic_defined", "n_roughness_defined", "n_pbr_defined"]


def pbr_verdict(row) -> bool:
    """Second rule: some used material writes albedo, metallic and roughness, as a
    texture or a number (spec/gloss counts). From `n_pbr_defined`."""
    return row.get("status") == "ok" and int(row.get("n_pbr_defined") or 0) > 0


def declaration_verdict(row) -> bool:
    """The emission rule for one scan row, from the declaration counts.

    Read from `n_emissive_materials` rather than the `passes` column, so an older CSV
    whose `passes` folded in a texture decode still yields the declaration-level rule.
    For CSVs this scan writes the two readings are identical.
    """
    if row.get("status") != "ok":
        return False
    return int(row["n_emissive_materials"] or 0) > 0


def scan_one(sha: str, metadata: pd.DataFrame, roots) -> dict:
    """One shape's emission declaration + verdict. Never raises; failures land in `status`.

    One declaration-level verdict per used material; a shape passes when any used
    material passes. Nothing is decoded: whether a declared texture is all-black is
    the heuristic stage's test (`heuristic_filtering/nonzero_emission_check.py`).

    `passes`:
        an emissive texture is declared (factor ignored), or
        no texture and emissiveFactor has a channel > 0.
      A texture counts on its own because artists leave the factor unset on real
      emission maps (a screen's glow map with the factor absent); the cost is that
      exporter copies of the base-color image into the emissive slot also pass.
    """
    path = find_glb(sha, metadata, roots)
    if path is None:
        return {"uuid": sha, "status": "no_file", **{k: "" for k in EMPTY_ON_FAILURE}}
    try:
        js = read_gltf_json(path)          # parsed ONCE; both readers build from it
    except OSError as exc:
        return {"uuid": sha, "status": "err:" + type(exc).__name__, **{k: "" for k in EMPTY_ON_FAILURE}}
    mats = material_emission_from_json(js) if js is not None else []
    # Channel inventory over USED materials: record only, never part of the verdict.
    channels = (summarize_channels(materials_from_json(js), only_used=True) if js is not None
                else {k: 0 for k in CHANNEL_COUNT_FIELDS})
    if not mats:
        # Either an unparseable GLB or a file with no materials at all; both are
        # "nothing declared", and the two are distinguished by re-reading the JSON
        # only if someone needs it. Neither can be emissive.
        return {
            "uuid": sha, "status": "ok", "n_materials": 0, "n_unused_materials": 0,
            "n_emissive_materials": 0, "n_emissive_texture": 0, "n_factor_positive": 0,
            "n_texture_only": 0, "max_strength": 1.0, "declares_strength": 0,
            "passes": 0, **channels,
        }
    # Only materials some mesh primitive references can affect what is drawn.
    used = [m for m in mats if m.is_used]
    factor_pos_mats = [m for m in used if any(c > 0 for c in m.emissive_factor)]
    textured = [m for m in used if m.has_emissive_texture]
    tex_only = sum(1 for m in textured if m not in factor_pos_mats)

    passes = any(m.is_emissive for m in used)           # the rule: texture, or factor > 0 without one

    # Strength is reported over ALL materials, used or not, to stay identical to
    # the evaluator's parse (utils.gltf_emission.max_emissive_strength).
    declared = [m.strength for m in mats if m.declares_strength]
    return {
        "uuid": sha,
        "status": "ok",
        "n_materials": len(mats),
        "n_unused_materials": len(mats) - len(used),
        "n_emissive_materials": sum(1 for m in used if m.is_emissive),
        "n_emissive_texture": len(textured),
        "n_factor_positive": len(factor_pos_mats),
        "n_texture_only": tex_only,
        "max_strength": max(declared) if declared else 1.0,
        "declares_strength": int(bool(declared)),
        "passes": int(passes),
        **channels,
    }


def summarize(rows, paper_set=None) -> str:
    """Human-readable tally. `paper_set` (the uuids of the paper's shape set) adds the in/outside rows."""
    out = []
    status = collections.Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "ok"]
    out.append(f"rows: {len(rows)}   status: {dict(status)}")
    out.append(f"scanned ok: {len(ok)}")

    def _n(r, k):
        return int(r[k]) if r[k] != "" else 0

    declares = [r for r in ok if _n(r, "n_emissive_materials") > 0]
    factor = [r for r in ok if _n(r, "n_factor_positive") > 0]
    texonly = [r for r in declares if _n(r, "n_factor_positive") == 0]
    unused = sum(1 for r in ok if _n(r, "n_unused_materials") > 0)
    passing = [r for r in ok if _n(r, "passes") > 0]
    out.append(f"files with a listed material no mesh uses: {unused}  (such materials are excluded from every count below)")
    out.append("")
    out.append(f"declares emission (texture OR factor>0) : {len(declares)} ({100*len(declares)/max(len(ok),1):.2f}%)")
    out.append(f"  of which factor > 0 (renders lit)     : {len(factor)}")
    out.append(f"  of which texture only, zero factor    : {len(texonly)}  <- renders black in a strict renderer")
    out.append(f"PASS (the filter's verdict, loose)      : {len(passing)} ({100*len(passing)/max(len(ok),1):.2f}%)")
    out.append("  (whether a declared texture is all-black is heuristic_filtering's test, not this stage's)")

    strengths = collections.Counter(float(r["max_strength"]) for r in ok if r["declares_strength"] == "1" or r["declares_strength"] == 1)
    if strengths:
        out.append("")
        out.append("per-shape MAX KHR strength (shapes carrying the extension), top 10:")
        for v, c in strengths.most_common(10):
            out.append(f"  {v:8g}: {c:7d}")

    if paper_set is not None:
        inside = [r for r in passing if r["uuid"] in paper_set]
        missed = [r for r in passing if r["uuid"] not in paper_set]
        missed_lit = [r for r in missed if _n(r, "n_factor_positive") > 0]
        out.append("")
        out.append(f"paper set given: {len(paper_set)}")
        out.append(f"  passes AND in the paper set               : {len(inside)}")
        out.append(f"  passes but NOT in the paper set           : {len(missed)}")
        out.append(f"    ...and factor > 0 (genuinely missed) : {len(missed_lit)}")
        out.append("  NOTE: 'not in the paper set' mixes shapes the Blender scan missed with")
        out.append("  emissive shapes dropped later by the bake intersection and the")
        out.append("  held-out shapes of the pilot. Subtract those before quoting a miss rate.")
    return "\n".join(out)


def cmd_scan(a):
    cols = ["glb_1k_path"]
    extra = ([DOWNLOADED_COLUMN] if a.shas_file is None else []) + ([THUMBNAIL_COLUMN] if a.thumbnails_out else [])
    meta = pd.read_parquet(a.metadata, columns=cols + extra)
    if a.shas_file is None:
        if DOWNLOADED_COLUMN in meta.columns:
            meta = meta[meta[DOWNLOADED_COLUMN] == True]  # noqa: E712
        shas = [str(x) for x in meta.index]
    else:
        shas = [ln.strip() for ln in open(a.shas_file) if ln.strip()]
    thumbs = meta[THUMBNAIL_COLUMN] if a.thumbnails_out else None
    meta = meta[cols]

    # a killed scan can leave half a row (rows are flushed in batches) and a failed row may
    # read now: resume_done drops both, so those shapes are scanned again
    done, dropped = resume_done(a.out, FIELDS, shas, header_error=(
        f"{a.out} has a different column set than this scan writes (it writes no texture "
        "columns). Start a new --out, or derive lists from the old "
        "CSV with `list`, which reads the declaration counts."))
    if done or dropped:
        print(f"resuming: {len(done)} complete rows already in {a.out}"
              f"{f'; {dropped} failed, incomplete or repeated rows dropped, those shapes are scanned again' if dropped else ''}",
              flush=True)
    todo = [s for s in dict.fromkeys(shas) if s not in done]    # a uuid listed twice runs once
    if a.limit:
        todo = todo[: a.limit]
    print(f"shapes: {len(shas)} total, {len(todo)} to scan, {a.workers} workers", flush=True)

    new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    t0 = time.time()
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        written = write_as_completed(lambda s: scan_one(s, meta, a.roots), todo, w, fh, a.workers)

    with open(a.out) as f:
        rows = list(csv.DictReader(f))
    paper_set = None
    if a.paper_list:
        paper_set = {ln.strip() for ln in open(a.paper_list) if ln.strip()}
    text = summarize(rows, paper_set)
    print("\n" + text, flush=True)
    with open(a.out + ".summary.txt", "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {written} new rows to {a.out} in {time.time()-t0:.0f}s", flush=True)

    if a.thumbnails_out:
        passing = [r["uuid"] for r in rows if declaration_verdict(r)]
        os.makedirs(a.thumbnails_out, exist_ok=True)
        copied = missing = 0
        for u in passing:
            rel = thumbs.get(u)
            src = os.path.join(a.thumbnails_src, rel) if isinstance(rel, str) and rel else None
            dst = os.path.join(a.thumbnails_out, u + ".png")
            if src is None or not os.path.exists(src):
                missing += 1
                continue
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
            copied += 1
        print(f"thumbnails: {copied} present in {a.thumbnails_out}, {missing} passing shapes had none", flush=True)


def cmd_list(a):
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.csv))
    csv.field_size_limit(10**7)

    with open(a.csv) as f:
        rows = list(csv.DictReader(f))
    if a.specgloss_rescan:
        fix = {r["uuid"]: r for r in csv.DictReader(open(a.specgloss_rescan))}
        n = 0
        for r in rows:
            if r["uuid"] in fix:
                r.update({k: v for k, v in fix[r["uuid"]].items() if k != "uuid"})
                n += 1
        print(f"applied {a.specgloss_rescan}: {n} rows overridden")
    ok = [r for r in rows if r["status"] == "ok"]
    emission = [r for r in ok if declaration_verdict(r)]
    passing = [r for r in emission if pbr_verdict(r)]
    paper = {ln.strip() for ln in open(a.paper_list) if ln.strip()} if a.paper_list else set()

    print(f"rows total {len(rows)} | status counts {dict(collections.Counter(r['status'] for r in rows))}")
    print(f"scanned ok {len(ok)}")
    print(f"rule 1, declares emission            : {len(emission)}")
    print(f"rule 1 + 2, and PBR written somewhere: {len(passing)}   <- the pass list")
    if paper:
        i = sum(1 for r in passing if r["uuid"] in paper)
        coh = [r for r in ok if r["uuid"] in paper]
        rej = [r for r in coh if not (declaration_verdict(r) and pbr_verdict(r))]
        print(f"  in the paper set {i}   outside it {len(passing) - i}")
        print(f"  paper-set shapes scanned {len(coh)} | not passing {len(rej)} "
              f"(no emission {sum(1 for r in coh if not declaration_verdict(r))}, "
              f"no PBR {sum(1 for r in coh if declaration_verdict(r) and not pbr_verdict(r))})")

    write_uuids(os.path.join(out_dir, "emissive_pass.txt"), [r["uuid"] for r in passing])
    write_uuids(os.path.join(out_dir, "emission_only_pass.txt"), [r["uuid"] for r in emission])
    if paper:
        with open(os.path.join(out_dir, "paper_set_rejected.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["uuid", "sketchfab", "reason", "n_materials", "n_emissive_materials", "n_pbr_defined"])
            for r in rej:
                reason = "no_emission" if not declaration_verdict(r) else "no_pbr"
                w.writerow([r["uuid"], "https://sketchfab.com/models/" + r["uuid"], reason,
                            r["n_materials"], r["n_emissive_materials"], r.get("n_pbr_defined", "")])
    print(f"pass list: {len(passing)} uuids -> {out_dir}/emissive_pass.txt")


def specgloss_uuids(scan_csv) -> list:
    csv.field_size_limit(10**7)
    with open(scan_csv) as f:
        return [r["uuid"] for r in csv.DictReader(f) if int(r["n_specgloss"] or 0) > 0]


def cmd_specgloss_rescan(a):
    uuids = specgloss_uuids(a.csv)
    print(f"spec/gloss shapes in {a.csv}: {len(uuids)}", flush=True)
    if a.sample:
        uuids = sorted(random.Random(a.seed).sample(uuids, a.sample))
    meta = pd.read_parquet(a.metadata, columns=["glb_1k_path"])
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda s: scan_one(s, meta, a.roots), uuids))
    bad = [r["uuid"] for r in rows if r["status"] != "ok"]
    if bad:
        print(f"WARNING: {len(bad)} shapes not read ok, e.g. {bad[:3]}", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPECGLOSS_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="read every GLB's declarations into a resumable CSV")
    s.add_argument("--metadata", default=DEFAULT_METADATA, help="parquet with glb_1k_path (default: the GLB index)")
    s.add_argument("--shas_file", help="uuid list to scan (default: every downloaded GLB)")
    s.add_argument("--paper_list", help="uuid list of the paper's shape set, for the missed-shape tally")
    s.add_argument("--out", required=True, help="output CSV")
    s.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="thread pool size (I/O bound)")
    s.add_argument("--roots", nargs="*", default=list(DEFAULT_GLB_ROOTS), help="GLB mounts to search")
    s.add_argument("--limit", type=int, help="scan at most N shapes (smoke test)")
    s.add_argument("--thumbnails_out", help="copy the thumbnail of each shape that declares emission to <dir>/<uuid>.png")
    s.add_argument("--thumbnails_src", default=DEFAULT_THUMBNAIL_ROOT,
                   help="where the metadata's thumbnail_path is relative to")

    l = sub.add_parser("list", help="the pass list from one scan CSV, both rules")
    l.add_argument("--csv", required=True, help="scan CSV written by `scan`")
    l.add_argument("--paper_list", help="uuid list of the paper's shape set (adds the in/outside split)")
    l.add_argument("--specgloss_rescan", help="CSV of uuid + n_*_defined overrides (only for a scan that did not read the spec/gloss keys)")
    l.add_argument("--out_dir", help="where to write (default: beside the CSV)")

    r = sub.add_parser("specgloss_rescan", help="re-read the spec/gloss shapes of a scan that did not read the spec/gloss keys")
    r.add_argument("--csv", required=True, help="the scan CSV to repair (one that did not read the spec/gloss keys)")
    r.add_argument("--out", required=True, help="the override CSV to write")
    r.add_argument("--metadata", default=DEFAULT_METADATA)
    r.add_argument("--roots", nargs="*", default=list(DEFAULT_GLB_ROOTS))
    r.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    r.add_argument("--sample", type=int, help="re-read a random N of the shapes only (a check)")
    r.add_argument("--seed", type=int, default=0)

    a = ap.parse_args()
    {"scan": cmd_scan, "list": cmd_list, "specgloss_rescan": cmd_specgloss_rescan}[a.cmd](a)


if __name__ == "__main__":
    main()
