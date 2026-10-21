"""VLM filtering: judge every shape of the list of record on its eight review images.

The last filter of the dataset build (paper §4, VLM filtering). A shape is judged when all
eight review images exist (`render/`); its baked emission map must not be all zero, and
GPT-5.6 Luna answers, from the prompt below and the eight images, whether it is a good
emitter: emission that is correctly placed and actually visible when rendered. The verdict
is the VLM's (no whole-object rule); the pass list is the split stage's input.

The check, one subcommand per step, in run order:

  input       list of record + review-image coverage + category -> out/input.csv
  prewarm     the derived images (orbit grids, UV map) -> out/evidence/
  atlas       each judged shape's emitting texels -> out/atlas_emission.csv
  judge       the VLM over the judged shapes, resumable -> out/predictions/<name>.jsonl
  verdicts    predictions -> out/verdict.csv, out/vlm_pass.txt, out/verdict.meta.json

The tools the prompt was tuned with (their code is in `utils.py`):

  labels      freeze the human labels of both annotation DBs -> labels/labels_<date>.csv
  membership  the tune half, the test half and the other labeled shapes -> labels/membership_<date>.json
  score       a prediction file against the labels on one set -> out/scores/<set>_<name>.md
  errors      the shapes the VLM got wrong, with their images -> an HTML page

Usage (lightgen env, from the repo root):

  conda run --no-capture-output -n lightgen python -m data_processing.vlm_filtering.vlm_check <step> [-h]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data_processing.vlm_filtering.utils import (  # noqa: E402
    agreement, append_row, available_models, build_page, confusion, coverage,
    emitting_texels_from_png, git_sha, grid_from_webp, judge, load_cache, load_human_rows, load_labels,
    load_predictions, make_client, mcnemar_exact, membership, merge_rows, nonempty, per_flag_prf,
    resolve_model, scorecard_md, sha256_of_lines, sha256_of_text,
    terminate_partial_line, write_sidecar, write_snapshot)
from utils import atlas_io, review_images  # noqa: E402
from utils.metadata_io import read_uuids, sha256_of_file, write_uuids  # noqa: E402
from utils.paths import LABEL_DBS_ROOT, OPENAI_KEY_PATH, REPO_ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent

# --- inputs ---------------------------------------------------------------------------
#: The de-duplicated list of record: the heuristic stage's pass list minus the copies the
#: duplicate check removes.
INPUT_LIST = REPO_ROOT / "data_processing/heuristic_filtering/out/duplicate_pass.txt"
#: One row per uuid of the list of record, one boolean per review-image kind, plus `shard`.
COVERAGE_PARQUET = REPO_ROOT / "data_processing/render/out/review/coverage.parquet"
#: uuid -> `auto_category`, carried into `out/input.csv` for the
#: split stage; the prompt names no category. Optional: without it every category is `unknown`.
MANIFEST_PARQUET = REPO_ROOT / "data_processing/tools/human_annotation/_data/manifest.parquet"
#: The OpenAI key file, else the OPENAI_API_KEY environment variable. Never written anywhere.
KEY_CONFIG_PATH = OPENAI_KEY_PATH
#: The paper's random study sample in two halves: the tune half refines the prompt, the test
#: half measures it once.
TUNE_HALF_SOURCE = HERE / "labels/study1000_tune499.txt"
TEST_HALF_SOURCE = HERE / "labels/study1000_test501.txt"

# --- outputs (gitignored) -------------------------------------------------------------
OUT_DIR = HERE / "out"
INPUT_CSV = OUT_DIR / "input.csv"
EVIDENCE_DIR = OUT_DIR / "evidence"
SCORES_DIR = OUT_DIR / "scores"
VERDICT_CSV = OUT_DIR / "verdict.csv"
PASS_LIST = OUT_DIR / "vlm_pass.txt"
VERDICT_META = OUT_DIR / "verdict.meta.json"
PREWARM_UNUSABLE = OUT_DIR / "prewarm_unusable.txt"
#: Per judged shape, the emitting-texel count of its baked emission map (`atlas`).
ATLAS_STATS_CSV = OUT_DIR / "atlas_emission.csv"
#: Tracked: the frozen label snapshot and the tune/test membership.
LABELS_DIR = HERE / "labels"
#: The annotation DB copies `labels` reads unless `--local-db` names others; the first wins a conflict.
LABEL_DBS = {"v2": LABEL_DBS_ROOT / "annotations_v2.db",
             "canonical": LABEL_DBS_ROOT / "annotations_canonical.db"}
# --- the VLM --------------------------------------------------------------------------
#: The paper names gpt-5.6-luna. Nothing is substituted: if it is not available the run stops.
MODEL = "gpt-5.6-luna"
DETAIL = "low"
CONCURRENCY = 8
MAX_RETRIES = 4
PROGRESS_EVERY = 500

#: Mirrors `tools.human_annotation.utils.FLAGS` exactly, kept in sync by hand.
FLAG_VALUES = ("not_emissive_object", "wrong_emissive_maps",
               "zero_emission_map", "multiple_object")
Flag = Literal[FLAG_VALUES]  # type: ignore[valid-type]

#: The verdict values, the annotation tool's own words.
VERDICT_VALUES = ("no_issue", "has_issue")


# What the VLM returns, as OpenAI strict JSON. No `confidence`: the self-report was measured
# uncalibrated. No docstring: pydantic would send it to the VLM as the schema's description.
class Prediction(BaseModel):
    # extra='forbid' -> additionalProperties:false, required by OpenAI strict mode.
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["no_issue", "has_issue"]
    flags: list[Flag] = Field(description="Issue reasons; empty when verdict is no_issue.")  # type: ignore[valid-type]
    style: Literal["real", "artistic"]
    emission_complete: bool = Field(description="True if every part that should glow does.")
    rationale: str = Field(description="One short sentence justifying the decision.")


# --- the prompt -----------------------------------------------------------------------
#: The eight images in the order the prompt enumerates them; reorder one only with the other.
ORDER = ("glb", "emit", "brighter", "box_off", "box_on", "orbit_standard", "orbit_emit_only", "uv")

#: Tuned on the tune half, measured once on the test half; the text is
#: `prompt.txt`, where `{flags}` stands for the flag list.
PROMPT_TEMPLATE = (HERE / "prompt.txt").read_text(encoding="utf-8")
PROMPT = PROMPT_TEMPLATE.format(flags=", ".join(FLAG_VALUES))
#: Recorded in every prediction row and sidecar; a resumed run must have the same prompt.
PROMPT_SHA256 = sha256_of_text(PROMPT_TEMPLATE)


# --- input: the table of shapes -------------------------------------------------------
INPUT_COLUMNS = ["uuid", "shard", "category", "judged", "reason"]


def build_input(list_path, coverage_path, manifest_path) -> pd.DataFrame:
    """`uuid, shard, category, judged, reason`. A shape is judged when every review-image kind
    in the coverage table is present; otherwise its reason is `images_incomplete` and the
    stage says nothing about it. A uuid the coverage table does not know is an error: the
    coverage scan is run over the whole list, so a gap means the list moved on and the scan
    must be re-run first."""
    uuids = read_uuids(list_path)
    cov = pd.read_parquet(coverage_path)
    kinds = [c for c in cov.columns if c not in ("uuid", "shard") and cov[c].dtype == bool]
    cov = cov.set_index("uuid")
    missing = [u for u in uuids if u not in cov.index]
    if missing:
        raise SystemExit(f"{len(missing)} uuids of {list_path} are not in {coverage_path}, "
                         f"e.g. {missing[:3]}. Re-run the review coverage scan over the list.")
    if Path(manifest_path).is_file():
        cats = pd.read_parquet(manifest_path, columns=["uuid", "auto_category"]) \
            .set_index("uuid")["auto_category"]
    else:                                    # the category only stratifies a fresh split draw
        print(f"WARNING: no annotation manifest at {manifest_path}: every category is 'unknown', "
              f"so a fresh split draw is not stratified (--manifest names one)", flush=True)
        cats = pd.Series(dtype=str)
    rows = []
    for u in uuids:
        r = cov.loc[u]
        judged = bool(r[kinds].all())
        rows.append(dict(uuid=u, shard=str(r["shard"]),
                         category=str(cats.get(u, "unknown") or "unknown"),
                         judged=judged, reason="" if judged else "images_incomplete"))
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def read_input(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or INPUT_CSV, dtype=str, keep_default_na=False)
    df["judged"] = df["judged"].map({"True": True, "False": False})
    return df[INPUT_COLUMNS]


# --- the eight images -----------------------------------------------------------------
# Three stills, the dark-box pair (OFF then ON, so the difference reads as what the emission
# adds), the two orbit grids, the emission UV map. The stills and the pair are the files the
# render lanes (`render/`) rendered; the orbits are its webps packed into a 3-column grid of
# their six frames; the UV map is encoded from the bake's atlas by the annotation tool's own
# function (`utils.atlas_io.write_emission_png`), so the VLM sees the picture the annotators
# saw; the stills, the pair and the orbits are found through `utils.review_images`. The derived
# files are cached under `out/evidence/`; a 0-byte cache file (a failed write) is rebuilt,
# never served.

@dataclass
class Evidence:
    uuid: str
    images: list


def resolve_images(uuid: str, shard: str, evidence_dir=None):
    """All eight images, or None: a shape is judged on every image or not at all."""
    evidence_dir = Path(evidence_dir or EVIDENCE_DIR)
    images = []
    for kind in ("glb", "emit", "brighter"):
        p = review_images.still(kind, shard, uuid)
        if p is None:
            return None
        images.append(Path(p))
    on, off = review_images.box(uuid)
    if on is None or off is None:
        return None
    images += [Path(off), Path(on)]
    for preset, suffix in (("standard", "_mv.png"), ("emit_only", "_mv_emit.png")):
        w = review_images.webp(preset, uuid)
        if w is None:
            return None
        g = evidence_dir / f"{uuid}{suffix}"
        if not nonempty(g):
            grid_from_webp(w, g)
        images.append(g)
    npz = review_images.atlas(uuid)
    if npz is None:
        return None
    uv = evidence_dir / f"{uuid}_uv.png"
    if not nonempty(uv) and not atlas_io.write_emission_png(npz, uv):
        return None
    images.append(uv)
    assert len(images) == len(ORDER)
    return Evidence(uuid=uuid, images=images)


# --- atlas: the empty-atlas filter ----------------------------------------------------
ATLAS_FIELDS = ["uuid", "n_emitting", "n_pixels", "fraction"]


def emitting_texels(uuid: str, evidence_dir=None):
    """A baked emission map that is all zero inside the UV charts is a defective bake (missing
    UVs, a textureless material): the shape has no emission target, whatever the VLM says of
    its renders. The paper's bake-level zero rule, the one non-VLM source of a `reject`.
    Counted on the cached UV image (occupancy-masked and monotonically encoded, so any non-zero
    pixel is an emitting texel), else on the atlas itself."""
    png = Path(evidence_dir or EVIDENCE_DIR) / f"{uuid}_uv.png"
    if png.exists() and png.stat().st_size > 0:
        return emitting_texels_from_png(png)
    npz = review_images.atlas(uuid)
    return atlas_io.emitting_texels(npz) if npz is not None else (0, 0)


def load_atlas_stats(csv_path) -> dict:
    with open(csv_path, newline="") as f:
        return {r["uuid"]: dict(n_emitting=int(r["n_emitting"]), n_pixels=int(r["n_pixels"]),
                                fraction=float(r["fraction"])) for r in csv.DictReader(f)}


# --- judge: the resumable run ---------------------------------------------------------

async def judge_shapes(client, model, items, out_jsonl, concurrency, detail,
                       resolve=resolve_images, judge=judge):
    """One JSONL row per shape, appended as it is judged. Rows without an error are kept on a
    resume and error rows are retried. A truncated last line (a killed process) is terminated
    before anything is appended, so the next row is never glued to the fragment. A file whose
    rows come from another prompt, model or detail is refused: a sidecar must never describe
    judgments it did not make."""
    terminate_partial_line(out_jsonl)
    cache = load_cache(out_jsonl)
    done = {}
    for u, r in cache.items():
        if "error" in r or "verdict" not in r:
            continue
        got = (r.get("prompt_sha256"), r.get("model"), r.get("detail"))
        if got != (PROMPT_SHA256, model, detail):
            raise SystemExit(f"{out_jsonl} holds rows from another prompt/model/detail {got}; this run is "
                             f"{(PROMPT_SHA256, model, detail)}. A new prompt is a new file: choose another --out.")
        done[u] = r
    todo = [it for it in items if it["uuid"] not in done]
    # Only rows of the current item set: a larger cache must not leak into a smaller run.
    results = [done[it["uuid"]] for it in items if it["uuid"] in done]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n_done = 0

    async def one(it):
        nonlocal n_done
        u = it["uuid"]
        async with sem:
            try:
                ev = await asyncio.to_thread(resolve, u, it["shard"])
                n = 0 if ev is None else len(ev.images)
                if n != len(ORDER):
                    raise ValueError(f"{u}: resolved {n} of the {len(ORDER)} images the prompt "
                                     f"promises; judged on nothing rather than on partial evidence")
                pred = await judge(client, model, PROMPT, ev.images, Prediction, detail=detail,
                                   max_retries=MAX_RETRIES)
                row = {"uuid": u, **pred.model_dump(), "model": model,
                       "prompt_sha256": PROMPT_SHA256, "detail": detail, "n_images": n}
            except Exception as e:
                row = {"uuid": u, "error": repr(e)}
        async with lock:
            append_row(out_jsonl, row)
            results.append(row)
            n_done += 1
            if n_done % PROGRESS_EVERY == 0:
                errs = sum(1 for r in results if "error" in r)
                print(f"  {n_done}/{len(todo)} judged, {errs} errors", flush=True)

    await asyncio.gather(*(one(it) for it in todo))
    return results


# --- verdicts -------------------------------------------------------------------------
VERDICT_FIELDS = ["uuid", "verdict", "source", "flags", "rationale"]


def verdicts(input_df, preds: dict, labels: dict = None, atlas_stats: dict = None) -> list:
    """`pass` / `reject` from the VLM (`source = vlm`), after the empty-atlas filter: a shape
    whose baked emission map has no emitting texel is `reject` with `source =
    zero_emission_atlas`, whatever the VLM said. Else `not_judged`: no eight images
    (`images_incomplete`) or no successful prediction (`not_predicted`); whether those enter
    training is the split stage's decision. `labels` (off by default) makes a human label win
    where one exists (`source = human`)."""
    labels = labels or {}
    atlas_stats = atlas_stats or {}
    rows = []
    for r in input_df.itertuples():
        h = labels.get(r.uuid)
        z = atlas_stats.get(r.uuid)
        if h is not None:
            rows.append(dict(uuid=r.uuid, verdict="pass" if h["emissive"] else "reject",
                             source="human", flags=";".join(h["flags"]), rationale=""))
        elif z is not None and z["n_emitting"] == 0:
            rows.append(dict(uuid=r.uuid, verdict="reject", source="zero_emission_atlas",
                             flags="zero_emission_map", rationale="the baked emission map has no emitting texel"))
        elif not r.judged:
            rows.append(dict(uuid=r.uuid, verdict="not_judged",
                             source=r.reason or "images_incomplete", flags="", rationale=""))
        elif r.uuid not in preds:
            rows.append(dict(uuid=r.uuid, verdict="not_judged", source="not_predicted",
                             flags="", rationale=""))
        else:
            p = preds[r.uuid]
            rows.append(dict(uuid=r.uuid, verdict="pass" if p["verdict"] == "no_issue" else "reject",
                             source="vlm", flags=";".join(p.get("flags") or []),
                             rationale=p.get("rationale") or ""))
    return rows


def write_verdicts(rows, pred_path, input_path, labels_path, out_csv, pass_list, meta_path,
                   atlas_stats_path=None) -> dict:
    """`labels_path` is None for a purely VLM verdict; `atlas_stats_path` names the empty-atlas scan."""
    out_csv, pass_list, meta_path = Path(out_csv), Path(pass_list), Path(meta_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VERDICT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    passed = [r["uuid"] for r in rows if r["verdict"] == "pass"]
    sha = sha256_of_lines(passed)
    write_uuids(pass_list, passed, header=[
        f"VLM filtering pass list: {len(passed)} of {len(rows)} shapes "
        f"({'human label first, then the VLM' if labels_path else 'the VLM alone'}).",
        f"generated {datetime.now(timezone.utc).date()}   sha256(uuids) = {sha}"])
    pred_rows = [r for r in load_cache(pred_path).values() if "error" not in r]
    meta = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        input=str(Path(input_path).resolve()), input_sha256=sha256_of_file(input_path),
        predictions=str(Path(pred_path).resolve()), predictions_sha256=sha256_of_file(pred_path),
        labels=(str(Path(labels_path).resolve()) if labels_path else None),
        labels_sha256=(sha256_of_file(labels_path) if labels_path else None),
        atlas_stats=(str(Path(atlas_stats_path).resolve()) if atlas_stats_path else None),
        atlas_stats_sha256=(sha256_of_file(atlas_stats_path) if atlas_stats_path else None),
        prompt_sha256=sorted({r.get("prompt_sha256") for r in pred_rows} - {None}),
        models=sorted({r.get("model") for r in pred_rows} - {None}),
        n=len(rows), passed=len(passed), pass_sha256=sha,
        counts=dict(sorted(Counter(f"{r['verdict']}:{r['source']}" for r in rows).items())),
    )
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


# --- the subcommands ------------------------------------------------------------------

def cmd_input(a):
    df = build_input(a.list, a.coverage, a.manifest)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    n, j = len(df), int(df.judged.sum())
    print(f"{n} shapes: {j} judged, {n - j} not ({df.reason.value_counts().to_dict()}); "
          f"{int((df.category == 'unknown').sum())} unknown categories -> {a.out}")


def cmd_prewarm(a):
    """Packs the two grids and encodes the UV map of every judged shape up front, in a thread
    pool. Done inside the run it blocks the event loop (a cold shape is several times slower
    than a cached one), so the run's ceiling would not be the API. Here it is CPU and NFS
    bound, after which the run is API-bound. Idempotent;
    shapes that resolve fewer than eight images, or raise, are listed in
    `out/prewarm_unusable.txt` (the run would write an error row for each)."""
    df = read_input(a.input)
    df = df[df.judged]
    if a.uuids:
        df = df[df.uuid.isin(set(read_uuids(a.uuids)))]
    items = [(r.uuid, r.shard) for r in df.itertuples()]
    print(f"prewarming {len(items)} shapes with {a.workers} workers", flush=True)
    t0 = time.time()
    done = 0
    unusable = []

    def one(u, shard):
        try:
            ev = resolve_images(u, shard, a.evidence_dir)
            return u, (0 if ev is None else len(ev.images)), None
        except Exception as e:  # a broken source file must not stop the sweep
            return u, 0, repr(e)[:160]

    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, u, s) for u, s in items]):
            u, n, err = fut.result()
            done += 1
            if err or n != len(ORDER):
                unusable.append((u, n, err or ""))
            if done % a.report_every == 0:
                el = time.time() - t0
                print(f"  {done}/{len(items)}  {el/60:.1f} min  {done/el:.1f} shapes/s  "
                      f"eta {(len(items)-done)/(done/el)/60:.0f} min  unusable={len(unusable)}",
                      flush=True)
    el = time.time() - t0
    print(f"done in {el/60:.1f} min; resolved all {len(ORDER)}: {done - len(unusable)}; "
          f"unusable: {len(unusable)}")
    out = Path(a.unusable)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{u}\t{n}\t{e}\n" for u, n, e in unusable))
    for u, n, e in unusable[:10]:
        print(f"    {u}  {n} images  {e}")
    print(f"  -> {out}")


def cmd_atlas(a):
    df = read_input(a.input)
    uuids = list(df.uuid[df.judged])
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda u: emitting_texels(u, a.evidence_dir), uuids))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ATLAS_FIELDS)
        for u, (n, tot) in zip(uuids, res):
            w.writerow([u, n, tot, f"{(n / tot if tot else 0.0):.6f}"])
    zero = sum(1 for n, _ in res if n == 0)
    print(f"{len(uuids)} judged shapes scanned in {time.time() - t0:.0f} s: {zero} with an all-zero emission map -> {a.out}")


async def _judge_main(a):
    """The sidecar (`<out>.meta.json`) records the git SHA, the model, and the sha256 of the
    prompt, of the output schema, of the item list and of the predictions."""
    out = Path(a.out)
    if out.exists() and not a.resume:
        raise SystemExit(f"{out} exists. Refusing to append; pass --resume to continue this exact "
                         f"run, or choose another --out.")
    df = read_input(a.input)
    df = df[df.judged]
    if a.uuids:
        keep = set(read_uuids(a.uuids))
        unknown = keep - set(df.uuid)
        if unknown:
            raise SystemExit(f"{len(unknown)} uuids of {a.uuids} are not judged shapes of "
                             f"{a.input}, e.g. {sorted(unknown)[:3]}")
        df = df[df.uuid.isin(keep)]
    if a.limit:
        df = df.head(a.limit)
    items = [dict(uuid=r.uuid, shard=r.shard) for r in df.itertuples()]
    items_sha = sha256_of_lines([it["uuid"] for it in items])
    print(f"[1/3] {len(items)} items (sha256 {items_sha[:16]}), prompt sha256 {PROMPT_SHA256[:12]}")
    client = make_client(KEY_CONFIG_PATH)
    model = resolve_model(await available_models(client), MODEL)
    before = len(load_cache(out)) if out.exists() else 0
    print(f"[2/3] model {model}; {len(items) - before} to go at concurrency {a.concurrency}")
    rows = await judge_shapes(client, model, items, out, a.concurrency, a.detail,
                              resolve=lambda u, shard: resolve_images(u, shard, a.evidence_dir))
    errs = [r for r in rows if "error" in r]
    print(f"[3/3] {len(rows)} rows, {len(errs)} errors")
    for r in errs[:5]:
        print(f"      ERROR {r['uuid']}: {r['error'][:160]}")
    side = write_sidecar(
        out, created_utc=datetime.now(timezone.utc).isoformat(), git_sha=git_sha(), model=model,
        prompt_sha256=PROMPT_SHA256,
        schema_sha256=sha256_of_text(json.dumps(Prediction.model_json_schema(), sort_keys=True)),
        detail=a.detail, n_items=len(items), items_sha256=items_sha,
        uuids_file=(str(Path(a.uuids).resolve()) if a.uuids else None), n_rows=len(rows),
        n_errors=len(errs), predictions_sha256=sha256_of_file(out))
    print(f"      predictions -> {out}\n      sidecar     -> {side}")


def cmd_judge(a):
    asyncio.run(_judge_main(a))


def cmd_verdicts(a):
    """The meta names the prediction file (and the label file if used) with their sha256;
    `out/vlm_pass.txt` holds `pass` only, with a sha256 header."""
    preds = {u: r for u, r in load_cache(a.pred).items() if "error" not in r and "verdict" in r}
    stats = load_atlas_stats(a.atlas_stats) if a.atlas_stats else None
    rows = verdicts(read_input(a.input), preds, load_labels(a.labels) if a.labels else None, stats)
    meta = write_verdicts(rows, a.pred, a.input, a.labels, a.out, a.pass_list, a.meta,
                          atlas_stats_path=(a.atlas_stats or None))
    print(f"{meta['n']} shapes -> pass {meta['passed']}")
    for k, v in meta["counts"].items():
        print(f"  {k:<32} {v:>6}")
    print(f"  {a.out}\n  {a.pass_list}   sha256 {meta['pass_sha256'][:16]}\n  {a.meta}")


def cmd_labels(a):
    """Reads the annotation DB copies (`LABEL_DBS`, or `--local-db`), unions them by uuid with
    the first winning a conflict, and freezes the human rows whose uuid is in the list of record:
    `labels/labels_<date>.csv` + meta with its sha256."""
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    if a.local_db:
        dbs = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in a.local_db}
    else:
        dbs = dict(LABEL_DBS)
    missing = [str(p) for p in dbs.values() if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"no annotation DB at {missing}: copy the tool's DBs there "
                         f"(sqlite3 .backup) or pass --local-db NAME=PATH")
    names = list(dbs)
    per = {n: load_human_rows(dbs[n]) for n in names}
    rows, n_conflicts = per[names[0]], 0
    for n in names[1:]:
        rows, c = merge_rows(rows, per[n])
        n_conflicts += c
    out = Path(LABELS_DIR) / f"{a.name or f'labels_{date}'}.csv"
    meta = write_snapshot(rows, read_input(a.input), out, n_conflicts=n_conflicts,
                          sources={n: dict(db=str(dbs[n]), sha256=sha256_of_file(dbs[n]),
                                           n_human_rows=len(per[n])) for n in names})
    print(json.dumps(meta, indent=1)); print(f"-> {out}")


def cmd_membership(a):
    """The tune set is the study sample's tune half and the test set its other half, each
    restricted to labeled, judged shapes; `other` is every other labeled, judged shape (a
    secondary card, never a tuning decision). Never regenerated during tuning."""
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    labels = load_labels(a.labels)
    mem = membership(labels, read_input(a.input), read_uuids(a.tune_source),
                     read_uuids(a.test_source))
    out = Path(LABELS_DIR) / f"{a.name or f'membership_{date}'}.json"
    out.write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(), git_sha=git_sha(),
        labels=str(Path(a.labels)), labels_sha256=sha256_of_file(a.labels),
        tune_source=str(Path(a.tune_source)), tune_source_sha256=sha256_of_file(a.tune_source),
        test_source=str(Path(a.test_source)), test_source_sha256=sha256_of_file(a.test_source),
        n_tune=len(mem["tune"]), n_test=len(mem["test"]), n_other=len(mem["other"]),
        tune_sha256=sha256_of_lines(mem["tune"]), test_sha256=sha256_of_lines(mem["test"]),
        other_sha256=sha256_of_lines(mem["other"]),
        **mem), indent=1) + "\n")
    print(f"tune {len(mem['tune'])}  test {len(mem['test'])}  other {len(mem['other'])}  -> {out}")


def cmd_score(a):
    """Accuracy, sensitivity, specificity and precision with Wilson intervals, the 2×2 table,
    per-flag precision/recall (exploratory), and with `--against` the agreement with a second
    prediction file on the same items and McNemar's exact test on the discordant pairs (b:
    this file right, the other wrong; c: the reverse). The test set is read only with
    `--i-am-finished-tuning` and refuses a labeled member without a successful prediction:
    it is worth exactly as much as the number of times it has been looked at is small."""
    if a.set == "test" and not a.i_am_finished_tuning:
        raise SystemExit("Refusing to score the TEST set: it is the one unbiased estimate and it "
                         "survives exactly as long as nobody peeks. Pass --i-am-finished-tuning "
                         "only when the prompt is final and will not change from what you see.")
    uuids = json.loads(Path(a.membership).read_text())[a.set]
    labels = load_labels(a.labels)
    preds = load_predictions(a.pred)
    cov = coverage(uuids, labels, preds)
    if (a.set == "test" or a.require_complete) and cov["missing_predictions"]:
        raise SystemExit(f"{len(cov['missing_predictions'])} labeled members of the {a.set} set are missing a "
                         f"successful prediction in {a.pred} (e.g. {cov['missing_predictions'][:3]}); "
                         f"finish the run with --resume first, then score.")
    gt = {u: labels[u]["emissive"] for u in uuids if u in labels}
    gt_flags = {u: labels[u]["flags"] for u in gt}
    pe = {u: p["no_issue"] for u, p in preds.items()}
    r = confusion(gt, pe, uuids)
    bad = [u for u in uuids if u in gt and u in pe and not gt[u]]
    flags = per_flag_prf(gt_flags, {u: p["flags"] for u, p in preds.items()}, bad, FLAG_VALUES)
    against = None
    if a.against:
        other = {u: p["no_issue"] for u, p in load_predictions(a.against).items()}
        common = [u for u in uuids if u in gt and u in pe and u in other]
        b = sum(1 for u in common if pe[u] == gt[u] and other[u] != gt[u])
        c = sum(1 for u in common if pe[u] != gt[u] and other[u] == gt[u])
        against = dict(name=Path(a.against).name, agreement=agreement(pe, other, common), b=b, c=c,
                       p=mcnemar_exact(b, c), other_acc=confusion(gt, other, common)["acc"])
    title = f"{a.set} set, {Path(a.pred).name}, {datetime.now(timezone.utc).date()}"
    md = scorecard_md(title, r, flags, against, cov)
    out = Path(a.out) if a.out else Path(SCORES_DIR) / f"{a.set}_{Path(a.pred).stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"-> {out}")


def cmd_errors(a):
    uuids = json.loads(Path(a.membership).read_text())[a.set]
    df = read_input()
    meta = {r.uuid: dict(category=r.category, shard=r.shard) for r in df.itertuples()}
    page = build_page(uuids, load_labels(a.labels), load_predictions(a.pred), meta,
                      resolve=resolve_images, show_all=a.all)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"-> {out}  {out.stat().st_size / 1e6:.1f} MB")


#: `--evidence-dir`: a run in another output folder (`run.sh --vlm`) keeps its derived images
#: in its own folder.
EVIDENCE_HELP = "cache of the derived images: orbit grids, UV map (default out/evidence/)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("input", help="the table of shapes -> out/input.csv")
    s.add_argument("--list", default=str(INPUT_LIST))
    s.add_argument("--coverage", default=str(COVERAGE_PARQUET))
    s.add_argument("--manifest", default=str(MANIFEST_PARQUET))
    s.add_argument("--out", default=str(INPUT_CSV))

    s = sub.add_parser("prewarm", help="build the derived images up front")
    s.add_argument("--input", default=str(INPUT_CSV))
    s.add_argument("--uuids", default=None, help="restrict to these uuids (one per line)")
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--report-every", type=int, default=2000)
    s.add_argument("--evidence-dir", default=str(EVIDENCE_DIR), help=EVIDENCE_HELP)
    s.add_argument("--unusable", default=str(PREWARM_UNUSABLE),
                   help="the shapes that resolve fewer than eight images (default out/prewarm_unusable.txt)")

    s = sub.add_parser("atlas", help="the empty-atlas scan -> out/atlas_emission.csv")
    s.add_argument("--input", default=str(INPUT_CSV))
    s.add_argument("--out", default=str(ATLAS_STATS_CSV))
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--evidence-dir", default=str(EVIDENCE_DIR), help=EVIDENCE_HELP)

    s = sub.add_parser("judge", help="ask the VLM -> a prediction file and its sidecar")
    s.add_argument("--out", required=True)
    s.add_argument("--input", default=str(INPUT_CSV))
    s.add_argument("--uuids", default=None, help="restrict to these uuids (one per line)")
    s.add_argument("--detail", choices=("low", "high"), default=DETAIL)
    s.add_argument("--concurrency", type=int, default=CONCURRENCY)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--evidence-dir", default=str(EVIDENCE_DIR), help=EVIDENCE_HELP)

    s = sub.add_parser("verdicts", help="predictions -> verdict CSV, pass list, meta")
    s.add_argument("--pred", required=True)
    s.add_argument("--labels", default=None,
                   help="optional: a frozen snapshot whose human labels win where they exist (off = purely the VLM)")
    s.add_argument("--atlas-stats", default=str(ATLAS_STATS_CSV),
                   help="the empty-atlas scan (`atlas`); '' to skip")
    s.add_argument("--input", default=str(INPUT_CSV))
    s.add_argument("--out", default=str(VERDICT_CSV))
    s.add_argument("--pass-list", default=str(PASS_LIST))
    s.add_argument("--meta", default=str(VERDICT_META))

    s = sub.add_parser("labels", help="freeze the human labels -> labels/labels_<date>.csv")
    s.add_argument("--local-db", action="append", default=[], metavar="NAME=PATH",
                   help="DB copies to read instead of LABEL_DBS (first wins a conflict)")
    s.add_argument("--input", default=str(INPUT_CSV))
    s.add_argument("--name", default=None, help="file stem (default labels_<date>); a re-snapshot over a new list must not overwrite the day's frozen record")

    s = sub.add_parser("membership", help="the tune / test / other sets -> labels/membership_<date>.json")
    s.add_argument("--labels", required=True)
    s.add_argument("--input", default=str(INPUT_CSV))
    s.add_argument("--tune-source", default=str(TUNE_HALF_SOURCE))
    s.add_argument("--test-source", default=str(TEST_HALF_SOURCE))
    s.add_argument("--name", default=None, help="file stem (default membership_<date>)")

    s = sub.add_parser("score", help="a prediction file against the labels on one set")
    s.add_argument("--set", choices=("tune", "test", "other"), required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--membership", required=True)
    s.add_argument("--against", default=None)
    s.add_argument("--out", default=None, help="markdown scorecard (default out/scores/<set>_<pred>.md)")
    s.add_argument("--i-am-finished-tuning", action="store_true")
    s.add_argument("--require-complete", action="store_true",
                   help="refuse when a labeled member has no successful prediction (always on for --set test)")

    s = sub.add_parser("errors", help="the tune-set errors with their images -> an HTML page")
    s.add_argument("--pred", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--membership", required=True)
    s.add_argument("--set", choices=("tune", "other"), default="tune")   # never the test set
    s.add_argument("--out", required=True)
    s.add_argument("--all", action="store_true", help="also list the correctly judged shapes")

    a = ap.parse_args()
    {"input": cmd_input, "prewarm": cmd_prewarm, "atlas": cmd_atlas, "judge": cmd_judge,
     "verdicts": cmd_verdicts, "labels": cmd_labels, "membership": cmd_membership,
     "score": cmd_score, "errors": cmd_errors}[a.cmd](a)


if __name__ == "__main__":
    main()
