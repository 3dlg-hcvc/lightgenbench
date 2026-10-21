"""How often the copy check's rejects are bad emitters by human label (paper §4, *Heuristic
filtering*: "Of the discarded assets, 159 have a human label from our annotation, and 95.6%
of them are labeled as not emissive").

The rejects are the copy check's rows (`heuristic_filtering/out/albedo_copy_check.csv`) with
`reject == albedo_copy`, among the shapes that pass the zero-emission check. A human label is a
verified, non-automatic annotation with an `emissive` value in the annotation tool's database;
"not emissive" is `emissive == 0`. The default database is the canonical annotation database,
one of the two that `vlm_check.py labels` reads.

Usage (lightgen env, repo root):

    conda run --no-capture-output -n lightgen python -m data_processing.tools.dataset_statistics.copy_check_precision
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from data_processing.heuristic_filtering.albedo_copy_check import DEFAULT_OUT as COPY_ROWS, INPUT_LIST as ZERO_PASS
from utils.paths import LABEL_DBS_ROOT

LABELS_DB = LABEL_DBS_ROOT / "annotations_canonical.db"
PAPER = {"labeled": 159, "not emissive": 152, "share %": 95.6}


def human_labels(db) -> dict:
    # immutable: the snapshot is a record, so SQLite must not write -shm/-wal beside it
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    lab = dict(con.execute(
        "SELECT uuid, emissive FROM items WHERE verified=1 AND "
        "(annotator IS NULL OR annotator<>'auto') AND emissive IS NOT NULL"))
    con.close()
    return lab


def precision(copy_rows, zero_pass, db) -> dict:
    rows = pd.read_csv(copy_rows, dtype=str).fillna("")
    keep = {ln.strip() for ln in open(zero_pass) if ln.strip()}
    rejects = set(rows.loc[(rows.reject == "albedo_copy") & rows.uuid.isin(keep), "uuid"])
    lab = human_labels(db)
    labeled = rejects & set(lab)
    bad = sum(1 for u in labeled if lab[u] == 0)
    return {"rejects": len(rejects), "labels": len(lab), "labeled": len(labeled), "not emissive": bad,
            "share %": 100 * bad / len(labeled)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy_rows", default=str(COPY_ROWS))
    ap.add_argument("--zero_pass", default=str(ZERO_PASS))
    ap.add_argument("--labels_db", default=str(LABELS_DB))
    a = ap.parse_args()
    got = precision(a.copy_rows, a.zero_pass, a.labels_db)
    print(f"copy-check rejects {got['rejects']:,}; human labels in the database {got['labels']:,}")
    match = True
    for k, paper in PAPER.items():
        shown = round(got[k], 1) if isinstance(paper, float) else got[k]
        match &= shown == paper
        print(f"{k:13} {got[k]:8.1f}   paper {paper}   {'ok' if shown == paper else 'DIFFERS'}")
    print(f"matches the paper: {match}")


if __name__ == "__main__":
    main()
