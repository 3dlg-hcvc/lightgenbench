"""Category counts (paper §4 and its dataset table; supp §A.3, *Categories*).

Every candidate's category is the one `vlm_filtering/vlm_check.py input` wrote into
`vlm_filtering/out/input.csv` (from the category pipeline in `categories/`; shapes it never
reached are `unknown`, a placeholder that is not a category). Counted, with the paper's values
in parentheses:

  over the candidates          distinct categories (supp: 107), the share in `other` (4.5 %)
  over the released dataset    distinct categories (§4 and the table: 105), the share in
                               `other` (3.0 %), the five largest and their share (43.2 %),
                               categories with a hundred shapes or fewer (55), the smallest (6)
  the category run itself      fine-grained clusters (552) and the shapes no cluster took,
                               classified directly (supp: "~23k"), over the shapes that run
                               covered and over the candidates (with the backfill)

Usage (lightgen env, repo root):

    conda run --no-capture-output -n lightgen python -m data_processing.tools.dataset_statistics.category_counts
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

from data_processing.tools.dataset_statistics import config
from utils.paths import CATEGORIES_OUT_FULL

#: supp §A.3 and the dataset table; percentages at one decimal.
PAPER = {"candidates: categories": 107, "candidates: other %": 4.5, "release: categories": 105,
         "release: other %": 3.0, "release: top five %": 43.2, "release: categories <= 100 shapes": 55,
         "release: smallest": 6, "clusters": 552, "category run: classified directly (k)": 23}


def category_counts(input_csv, splits_json, out_full) -> dict:
    cand = pd.read_csv(input_csv, dtype=str)
    known = cand[cand.category != "unknown"]
    splits = json.load(open(splits_json))
    released = {u for s in ("train", "val", "test") for u in splits[s]}
    rel = cand[cand.uuid.isin(released)]
    assert len(rel) == len(released), "released shapes missing from input.csv"
    vc = rel[rel.category != "unknown"].category.value_counts()
    final = pd.read_json(out_full / "categories_final.jsonl", lines=True)
    backfill = pd.read_json(out_full / "categories_backfill.jsonl", lines=True)
    source = dict(zip(final.uuid, final.source))
    source.update(zip(backfill.uuid, backfill.source))
    over_cand = collections.Counter(source.get(u, "none") for u in cand.uuid)
    return {
        "candidates: categories": known.category.nunique(),
        "candidates: other %": 100 * (cand.category == "other").mean(),
        "release: categories": len(vc),
        "release: other %": 100 * (rel.category == "other").mean(),
        "release: top five %": 100 * vc.iloc[:5].sum() / len(rel),
        "release: categories <= 100 shapes": int((vc <= 100).sum()),
        "release: smallest": int(vc.min()),
        "clusters": len(json.load(open(out_full / "taxonomy.json"))),
        "category run: classified directly (k)": int((final.source != "cluster").sum()) / 1000,
        "_top five": list(vc.index[:5]), "_smallest": list(vc[vc == vc.min()].index),
        "_category run": {k: int(v) for k, v in final.source.value_counts().items()}, "_candidates by source": dict(over_cand),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(config.VLM_INPUT_CSV))
    ap.add_argument("--splits", default=str(config.RELEASE_SPLITS))
    ap.add_argument("--out_full", default=str(CATEGORIES_OUT_FULL))
    a = ap.parse_args()
    got = category_counts(a.input, a.splits, Path(a.out_full))
    match = True
    for k, paper in PAPER.items():
        shown = round(got[k]) if isinstance(paper, int) else round(got[k], 1)
        match &= shown == paper
        print(f"{k:40} {got[k]:>9.2f}   paper {paper}   {'ok' if shown == paper else 'DIFFERS'}")
    for k in ("_top five", "_smallest", "_category run", "_candidates by source"):
        print(f"{k[1:]}: {got[k]}")
    print(f"matches the paper: {match}")


if __name__ == "__main__":
    main()
