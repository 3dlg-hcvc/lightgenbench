"""The auto filter: shapes that need no human review, marked in the annotation DB.

`compute` writes the zero-baked-emission table for the tool's set; `apply` marks its zero rows
in the DB.

compute: zero emission = occupancy-masked baked emission max == 0, read from the
prewarmed emission-cache PNGs (PNG.max()==0). With `--stats-parquet`, that parquet's
`#stat_emission_color_max` is recorded beside it as a cross-check only; it never decides.
Falls back to the source npz when a cache PNG is missing.

Scores whichever bake the tool serves (`utils.resolve_emission_source`) — so this
must run AFTER the prewarm, or it will score a partly filled cache.

apply: marks each zero-emission shape as has-issue + verified with the
`zero_emission_map` flag and annotator `auto`, only where verified=0
(never clobbers a human decision). Idempotent.

Run: conda run -n lightgen python -m data_processing.tools.human_annotation.auto_filter compute \
        [--stats-parquet PATH] [--out PATH] [--workers N]
     conda run -n lightgen python -m data_processing.tools.human_annotation.auto_filter apply \
        --table _data/auto_filter/zero_emission_filter.parquet --db <data dir>/annotations.db
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from . import utils as annotation_utils
from .store import AnnotationStore

STAT_COL = "#stat_emission_color_max"


# ---- compute ----------------------------------------------------------------------------

def png_max(path):
    """Max pixel value of a PNG (0..255), or -1 if it can't be read."""
    try:
        with Image.open(path) as im:
            return int(np.asarray(im).max())
    except Exception:
        return -1


def build_zero_table(uuids, stat_by_uuid, max_fn):
    """DataFrame [uuid, stat_emission_max, masked_png_max, is_zero_emission].

    max_fn(uuid) -> int occupancy-masked emission max (0 => zero emission).
    """
    rows = []
    for u in uuids:
        m = max_fn(u)
        rows.append({
            "uuid": u,
            "stat_emission_max": stat_by_uuid.get(u),
            "masked_png_max": m,
            "is_zero_emission": bool(m == 0),
        })
    return pd.DataFrame(
        rows, columns=["uuid", "stat_emission_max", "masked_png_max", "is_zero_emission"])


def _npz_masked_max(npz_path):
    """Fallback: occupancy-masked emission max straight from the npz, 0..255, or -1."""
    p = str(npz_path)
    try:
        z = np.load(p)
    except Exception:
        return -1
    if "emission_color" not in z:
        return -1
    em = z["emission_color"]
    if em.ndim == 3 and em.shape[2] >= 3:
        em = em[:, :, :3]
    if "occupancy" in z:
        occ = z["occupancy"]
        occ = occ[..., 0] if occ.ndim == 3 else occ
        em = em[occ.astype(bool)]
    return int(em.max()) if em.size else 0


def _cohort_max_fn(manifest_df):
    """max_fn(uuid): cache PNG max, falling back to the source npz if the PNG is missing.

    Routes through `utils.resolve_emission_source` so the value scored here comes from
    the same bake the tool serves — otherwise a shape could be judged on one bake's
    emission while annotators look at another's. `utils.atlas_io.write_emission_png` masks
    by occupancy before writing, so the PNG max IS the masked max. `manifest_df` is unused:
    the bake is addressable by uuid alone, so a uuid missing from the manifest simply has no
    atlas and scores -1 like any other unresolvable shape.
    """
    del manifest_df

    def fn(uuid):
        npz_path = annotation_utils.resolve_emission_source(uuid)
        if npz_path is None:
            return -1
        m = png_max(str(annotation_utils.cache_png_path(uuid)))
        if m == -1:
            return _npz_masked_max(npz_path)
        return m

    return fn


def run_compute(args):
    manifest = pd.read_parquet(annotation_utils.MANIFEST_PATH)
    uuids = list(manifest["uuid"])
    print(f"[1/3] cohort: {len(uuids)} uuids from {annotation_utils.MANIFEST_PATH}")

    if args.stats_parquet:
        stats = pd.read_parquet(args.stats_parquet, columns=[STAT_COL])
        stat_by = stats[STAT_COL].to_dict()  # index is ditem_id == uuid
        print(f"[2/3] stats: {len(stat_by)} rows from {args.stats_parquet}")
    else:
        stat_by = {}
        print("[2/3] stats: none (no --stats-parquet)")

    max_fn = _cohort_max_fn(manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        maxes = dict(zip(uuids, ex.map(max_fn, uuids)))
    df = build_zero_table(uuids, stat_by, lambda u: maxes[u])

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_parquet(out, index=False)

    n_zero = int(df["is_zero_emission"].sum())
    unreadable = int((df["masked_png_max"] == -1).sum())
    stat_zero = df["stat_emission_max"].fillna(-1).eq(0)
    disagree = int((df["is_zero_emission"] & ~stat_zero).sum())  # PNG says zero, stat did not
    print(f"[3/3] wrote {out}")
    print(f"      zero-emission: {n_zero} / {len(df)} ({100*n_zero/len(df):.1f}%)")
    print(f"      PNG-only zeros (stat>0, edge-bleed): {disagree}")
    print(f"      unreadable (png+npz both failed): {unreadable}")
    return 0


# ---- apply ------------------------------------------------------------------------------

def zero_uuids(df):
    return [str(u) for u in df.loc[df["is_zero_emission"].astype(bool), "uuid"]]


def apply(store, df):
    return store.mark_auto_zero_emission(zero_uuids(df))


def run_apply(args):
    df = pd.read_parquet(args.table)
    store = AnnotationStore(args.db)
    store.init_schema()
    res = apply(store, df)
    print(f"table: {args.table}  db: {args.db}")
    print(f"zero-emission uuids: {int(df['is_zero_emission'].astype(bool).sum())}")
    print(f"result: {res}")
    return 0


def main(argv=None):
    default_table = str(annotation_utils.DATA_DIR / "auto_filter" / "zero_emission_filter.parquet")
    ap = argparse.ArgumentParser(
        prog="python -m data_processing.tools.human_annotation.auto_filter",
        description="The zero-emission auto filter: compute the table, then apply it to a DB.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute", help="write the zero-emission table (run after the prewarm)")
    c.add_argument("--stats-parquet", default=None,
                   help="optional: a parquet indexed by uuid with #stat_emission_color_max, "
                        "recorded beside the measured max")
    c.add_argument("--out", default=default_table)
    c.add_argument("--workers", type=int, default=32)
    a = sub.add_parser("apply", help="mark the table's zero rows in a DB (never a human decision)")
    a.add_argument("--table", default=default_table)
    a.add_argument("--db", default=str(annotation_utils.DB_PATH))
    args = ap.parse_args(argv)
    return run_compute(args) if args.cmd == "compute" else run_apply(args)


if __name__ == "__main__":
    sys.exit(main())
