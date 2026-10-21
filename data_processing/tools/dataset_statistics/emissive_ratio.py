"""Emissive ratio (paper §4, *Data statistics*; supp §A.3, *Emissive ratio*).

A shape's emissive ratio is the fraction of its UV-occupied atlas texels whose emission is
non-zero in any channel, read from the bake (`occupancy`, `emission_color` of each shape's
`atlas.npz`).

    compute   one row per shape the VLM stage judged: uuid, n_occupied, n_emitting, error.
              About 1 MB read per shape; resumable (complete, error-free rows are kept).
    report    the paper's numbers: the mean before VLM filtering (the judged shapes with
              emission) and in the released dataset, its median and quartiles, the mean
              per split, and the share of the shapes with emission over at least half of
              their surface that VLM filtering removes.

Usage (lightgen env, repo root):

    conda run --no-capture-output -n lightgen python -m data_processing.tools.dataset_statistics.emissive_ratio compute
    conda run --no-capture-output -n lightgen python -m data_processing.tools.dataset_statistics.emissive_ratio report
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from data_processing.tools.dataset_statistics import config
from utils.atlas_io import emitting_texels
from utils.paths import REPRESENTATIONS_ROOT

RATIOS = config.OUT_DIR / "emissive_ratio.csv"
HEADER = ["uuid", "n_occupied", "n_emitting", "error"]

#: supp §A.3 and §4, percentages at one decimal (the 87 % is rounded to an integer there).
PAPER = {"before mean": 18.9, "after mean": 8.1, "after median": 1.4, "after p25": 0.3, "after p75": 7.0,
         "train mean": 8.1, "val mean": 8.7, "test mean": 7.4, "half or more removed": 87}


def one(uuid, root=REPRESENTATIONS_ROOT):
    """The VLM check's empty-atlas count (`utils.atlas_io.emitting_texels`), strict: an atlas
    missing `occupancy` or `emission_color` is an error row."""
    try:
        n_emitting, n_occupied = emitting_texels(f"{root}/{uuid}/atlas.npz", strict=True)
        return uuid, n_occupied, n_emitting, ""
    except Exception as e:  # recorded per row, never fatal
        return uuid, 0, 0, type(e).__name__


def compute(verdict_csv, out: Path, limit=None, workers=16):
    ids = [r["uuid"] for r in csv.DictReader(open(verdict_csv)) if r["verdict"] != "not_judged"]
    if limit:
        ids = ids[:limit]
    # Resume keeps only complete, error-free rows, and rewrites the file without the rest
    # first, so a failed read is retried and an interrupted last line cannot have the next
    # row appended onto it. A truncated last line lacks the trailing `error` field (None),
    # so require it present and empty: `uuid,100,2` may be a cut-off `uuid,100,25,`.
    kept = []
    if out.exists():
        for r in csv.DictReader(open(out)):
            if all(r.get(k) not in (None, "") for k in HEADER[:3]) and r.get("error") == "":
                kept.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(kept)
    tmp.replace(out)
    done = {r["uuid"] for r in kept}
    todo = [u for u in ids if u not in done]
    print(f"judged {len(ids)}, done {len(done)}, todo {len(todo)}", flush=True)
    with open(out, "a", newline="") as f, ThreadPoolExecutor(workers) as ex:
        w = csv.writer(f)
        for i, row in enumerate(ex.map(one, todo), 1):
            w.writerow(row)
            if i % 2000 == 0:
                f.flush()
                print(f"{i}/{len(todo)}", flush=True)
    print("done", flush=True)


def report(ratios_csv, verdict_csv, splits_json) -> dict:
    r = pd.read_csv(ratios_csv).set_index("uuid")
    assert (r["error"].isna() | (r["error"] == "")).all(), "unreadable atlases in the ratio file"
    r = r[r.n_occupied > 0]
    ratio = r.n_emitting / r.n_occupied
    before = ratio[r.n_emitting > 0]                 # the empty-atlas filter's survivors
    splits = json.load(open(splits_json))
    released = [u for s in ("train", "val", "test") for u in splits[s]]
    after = ratio.loc[released]
    verdict = pd.read_csv(verdict_csv, dtype=str).set_index("uuid")["verdict"]
    half = before[before >= 0.5]
    got = {
        "before mean": 100 * before.mean(), "after mean": 100 * after.mean(),
        "after median": 100 * after.median(), "after p25": 100 * after.quantile(.25),
        "after p75": 100 * after.quantile(.75),
        **{f"{s} mean": 100 * ratio.loc[splits[s]].mean() for s in ("train", "val", "test")},
        "half or more removed": 100 * (verdict.reindex(half.index) == "reject").mean(),
    }
    print(f"before VLM filtering n={len(before):,}; released n={len(after):,}; "
          f"emission over at least half the surface n={len(half):,}")
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--verdict", default=str(config.VLM_VERDICT_CSV))
    c.add_argument("--out", default=str(RATIOS))
    c.add_argument("--limit", type=int, help="first N judged shapes only (a check)")
    c.add_argument("--workers", type=int, default=16)
    r = sub.add_parser("report")
    r.add_argument("--ratios", default=str(RATIOS))
    r.add_argument("--verdict", default=str(config.VLM_VERDICT_CSV))
    r.add_argument("--splits", default=str(config.RELEASE_SPLITS))
    a = ap.parse_args()
    if a.cmd == "compute":
        compute(a.verdict, Path(a.out), a.limit, a.workers)
        return
    got = report(a.ratios, a.verdict, a.splits)
    match = True
    for k, paper in PAPER.items():
        shown = round(got[k]) if isinstance(paper, int) else round(got[k], 1)
        match &= shown == paper
        print(f"{k:22} {got[k]:6.2f}%   paper {paper}%   {'ok' if shown == paper else 'DIFFERS'}")
    print(f"matches the paper: {match}")


if __name__ == "__main__":
    main()
