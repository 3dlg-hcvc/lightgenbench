"""Build the item manifest from the filtered set, seed the DB, and set membership.

Run: conda run -n lightgen python -m data_processing.tools.human_annotation.build_manifest
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from . import utils as annotation_utils
from .store import AnnotationStore

#: What a shape with no category is filed under. It must still be addressable — the
#: category backfill fills these in later — so it gets a bucket rather than being dropped.
UNKNOWN_CATEGORY = "unknown"

#: Source labels for the category report, in precedence order.
SRC_JSONL = "categories_final.jsonl"
SRC_BACKFILL = "categories_backfill.jsonl"
SRC_EXISTING = "the existing manifest"
SRC_UNKNOWN = "unknown"


class CategoryBlankingError(RuntimeError):
    """The build would write a manifest that loses categories the existing one holds.

    Three ways in (see `check_no_blanking`): a surviving row comes out `unknown`, a row
    the input list names disappears entirely, or the whole build produces nothing. Raised
    before anything is written: without categories_final.jsonl the existing manifest is the
    only copy of the categories, so overwriting it is expensive to undo, while refusing to
    write costs a re-run.
    """


def _is_real_category(value):
    """A category that carries information. `unknown`, empty and null do not.

    Applied to every tier, so an explicit `unknown` in one source falls through to the
    next instead of shadowing a real category the next source has.
    """
    return isinstance(value, str) and value != "" and value != UNKNOWN_CATEGORY


def load_categories(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append({"uuid": d["uuid"], "category": d["category"]})
    return out


def load_categories_if_present(path):
    """Tier 1 of the category chain: categories_final.jsonl, the source of record.

    Missing is a loud event, not a quiet fallback: a run that fell through to tier 2
    without saying so is how a stale category set gets frozen in unnoticed.
    """
    p = Path(path) if path is not None else None
    if p is not None and p.exists():
        return {r["uuid"]: r["category"] for r in load_categories(p)}
    print("!" * 78, file=sys.stderr)
    print(f"!! {SRC_JSONL} NOT FOUND at {p}", file=sys.stderr)
    print("!! Falling back to the categories already in the existing manifest.", file=sys.stderr)
    print("!! Shapes in neither get `unknown`. Recover the jsonl before trusting the",
          file=sys.stderr)
    print("!! category facet for anything new: an annotation DB keeps every row, so", file=sys.stderr)
    print('!! sqlite3 <db> "SELECT uuid, auto_category FROM items" recovers them.', file=sys.stderr)
    print("!" * 78, file=sys.stderr)
    return {}


def load_backfill_categories(path):
    """Tier 2: the categories `tools/categories/backfill.py` produced for shapes new to the set.

    Kept in its own file rather than appended to `categories_final.jsonl`, so the original
    stays the record of the clustered pass. Absence is normal and quiet — the backfill has
    simply not been run — unlike tier 1, whose absence is loud. Rows carrying `error`
    instead of `category` are failed VLM calls and are skipped, not treated as answers.
    """
    p = Path(path) if path is not None else None
    if p is None or not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("uuid") and d.get("category"):
            out[d["uuid"]] = d["category"]
    return out


def load_existing_categories(manifest_path):
    """Tier 3 of the category chain: the `auto_category` a previous manifest already holds.

    Absence is fine (a fresh checkout has no `_data/`) and reported. An older manifest can
    carry other columns, so read whatever columns are there and take the two that matter. A read that FAILS is not swallowed: silently returning nothing
    from an unreadable parquet is the same blanking this tier exists to prevent.
    """
    p = Path(manifest_path) if manifest_path is not None else None
    if p is None or not p.exists():
        print(f"      no existing manifest at {p}: tier 3 of the category chain is empty")
        return {}
    df = pd.read_parquet(p)
    if "uuid" not in df.columns or "auto_category" not in df.columns:
        raise CategoryBlankingError(
            f"{p} has columns {list(df.columns)}: no uuid/auto_category to fall back on")
    return {u: c for u, c in zip(df["uuid"], df["auto_category"]) if _is_real_category(c)}


def check_no_blanking(rows, existing_cats, input_uuids=None):
    """Refuse to hand back a manifest that loses a category the old one has.

    A category can be lost two ways, and comparing only the surviving rows sees just the
    first. Both raise before the caller writes anything, because without
    categories_final.jsonl the manifest is the surviving copy of the categories.

    1. The row survives but comes out `unknown`. A broken tier of the precedence chain —
       the chain is meant to make it impossible.
    2. The row does not survive at all: a uuid in the input list that the existing
       manifest has a real category for got dropped on the way (no `ditem_dir` in the
       metadata parquet, a duplicated index entry, a null cell). Nothing in the surviving
       rows records that it was ever there, which is why this needs `input_uuids`.

    An empty `rows` is refused outright: every way of producing one (an empty or
    unreadable list, a metadata parquet that matches nothing) is a bug, and writing it
    would replace the manifest with a header.
    """
    if not rows:
        raise CategoryBlankingError(
            "the build produced 0 rows. Refusing to write: this replaces the manifest — "
            "the surviving copy of the categories — with an empty one. Check the input "
            "list and the metadata parquet.")
    blanked = [r["uuid"] for r in rows
               if r["auto_category"] == UNKNOWN_CATEGORY and r["uuid"] in existing_cats]
    if blanked:
        raise CategoryBlankingError(
            f"{len(blanked)} uuid(s) would lose the category the existing manifest holds, "
            f"e.g. " + ", ".join(
                f"{u} ({existing_cats[u]} -> {UNKNOWN_CATEGORY})" for u in blanked[:5]) +
            ". Refusing to write: the manifest is the surviving copy of these categories.")
    if input_uuids is not None:
        kept = {r["uuid"] for r in rows}
        dropped = sorted(set(existing_cats) & set(input_uuids) - kept)
        if dropped:
            raise CategoryBlankingError(
                f"{len(dropped)} uuid(s) of the input list that the existing manifest has "
                f"a category for did not make it into the output rows, e.g. " + ", ".join(
                    f"{u} ({existing_cats[u]})" for u in dropped[:5]) +
                ". Refusing to write: their categories exist nowhere else.")


def write_manifest_atomically(mdf, path):
    """Back up, write beside, rename over. Never truncate the file we just read.

    `to_parquet(path)` opens the destination for writing: a crash, a full disk or a
    quota limit partway through leaves a truncated manifest and no copy of the
    categories. The backup is kept rather than rotated — it is small next to what losing
    it costs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        shutil.copy2(path, backup)
        print(f"      backed up the existing manifest to {backup}")
    tmp = path.with_name(path.name + ".tmp")
    try:
        mdf.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return backup


def report_dropped(n_list, absent, unusable):
    """Say how many of the input list did not become rows, and why."""
    print(f"      {absent:,} list uuids absent from metadata, "
          f"{unusable:,} with no usable ditem_dir "
          f"({n_list - absent - unusable:,} of {n_list:,} kept)")


def report_category_sources(counts):
    """Print where every row's category came from. The operator sees the shape of the run."""
    print("      categories: "
          f"{counts[SRC_JSONL]:,} from {SRC_JSONL}, "
          f"{counts[SRC_BACKFILL]:,} from {SRC_BACKFILL}, "
          f"{counts[SRC_EXISTING]:,} from {SRC_EXISTING}, "
          f"{counts[SRC_UNKNOWN]:,} {SRC_UNKNOWN}")


def distinct_categories(manifest_df):
    """The sidebar vocabulary: exactly the categories the served set uses.

    Read off the manifest, not off categories_final.jsonl, so `unknown` is listed (a bucket
    the sidebar omits is a bucket nobody can filter to) and a category no served shape
    carries is not offered as an empty facet.
    """
    return sorted(set(manifest_df["auto_category"]))


def build_manifest_df(list_path, metadata_path, categories_path,
                      existing_manifest_path=None, backfill_path=None):
    """One row per uuid of the filtered set, in list order.

    Rows come from the heuristic stage's pass list and their shard from the TexVerse
    metadata parquet, which has `ditem_dir` for every downloaded asset.

    The category comes from a precedence chain: categories_final.jsonl if present, else the
    backfill for shapes new to the set, else the `auto_category` of
    the manifest already on disk, else `unknown`. The manifest records which tier answered
    in `category_source`. Tier 3 keeps the categories when the jsonl is missing; without it
    a rebuild stamps every shape `unknown`.

    A uuid with no category in any tier gets `unknown` rather than being dropped: it must be
    addressable now, and the category backfill fills it in later.
    """
    uuids = [ln.strip() for ln in open(list_path) if ln.strip()]
    meta = pd.read_parquet(metadata_path, columns=["ditem_dir"])
    cats = load_categories_if_present(categories_path)
    backfill_cats = load_backfill_categories(backfill_path)
    existing_cats = load_existing_categories(existing_manifest_path)
    counts = {SRC_JSONL: 0, SRC_BACKFILL: 0, SRC_EXISTING: 0, SRC_UNKNOWN: 0}
    absent = unusable = 0
    rows = []
    for u in uuids:
        if u not in meta.index:
            absent += 1
            continue
        dd = meta.loc[u, "ditem_dir"]
        # A duplicated index entry makes .loc hand back a Series, not a path; so does a
        # null cell. Either way the shard is not knowable, so the shape is not addressable.
        if not isinstance(dd, str) or not dd:
            unusable += 1
            continue
        if _is_real_category(cats.get(u)):
            cat, src = cats[u], SRC_JSONL
        elif _is_real_category(backfill_cats.get(u)):
            cat, src = backfill_cats[u], SRC_BACKFILL
        elif _is_real_category(existing_cats.get(u)):
            cat, src = existing_cats[u], SRC_EXISTING
        else:
            cat, src = UNKNOWN_CATEGORY, SRC_UNKNOWN
        counts[src] += 1
        rows.append({"uuid": u, "auto_category": cat, "ditem_dir": dd,
                     "category_source": src})
    report_category_sources(counts)
    # A shrinking row set is otherwise invisible: the manifest just comes out smaller.
    report_dropped(len(uuids), absent, unusable)
    check_no_blanking(rows, existing_cats, uuids)
    return pd.DataFrame(
        rows, columns=["uuid", "auto_category", "ditem_dir", "category_source"])


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m data_processing.tools.human_annotation.build_manifest",
        description="Build the item manifest from the filtered set, seed the DB, and set "
                    "membership.",
        epilog="No options: the inputs and outputs are utils.py's ANN74K_* overrides "
               "It writes $ANN74K_DATA_DIR and its DB.")
    ap.parse_args(argv)
    annotation_utils.DATA_DIR.mkdir(parents=True, exist_ok=True)
    annotation_utils.EMISSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] building manifest from {annotation_utils.INPUT_LIST}")
    print(f"      metadata {annotation_utils.METADATA_PARQUET}, "
          f"categories {annotation_utils.CATEGORIES_FINAL}")
    print(f"      backfill {annotation_utils.CATEGORIES_BACKFILL}")
    print(f"      category fallback: {annotation_utils.MANIFEST_PATH}")
    mdf = build_manifest_df(annotation_utils.INPUT_LIST, annotation_utils.METADATA_PARQUET,
                            annotation_utils.CATEGORIES_FINAL,
                            existing_manifest_path=annotation_utils.MANIFEST_PATH,
                            backfill_path=annotation_utils.CATEGORIES_BACKFILL)
    unknown = int((mdf["auto_category"] == UNKNOWN_CATEGORY).sum())
    print(f"      manifest rows: {len(mdf)} ({unknown} with no category yet)")

    print("[2/4] writing manifest and category vocabulary")
    write_manifest_atomically(mdf, annotation_utils.MANIFEST_PATH)
    with open(annotation_utils.CATEGORIES_PATH, "w") as f:
        json.dump(distinct_categories(mdf), f)
    print(f"      wrote {annotation_utils.MANIFEST_PATH} and {annotation_utils.CATEGORIES_PATH}")

    print(f"[3/4] seeding DB {annotation_utils.DB_PATH}")
    store = AnnotationStore(annotation_utils.DB_PATH)
    store.init_schema()
    # INSERT OR IGNORE: new uuids are added, and no existing row — category, label,
    # annotator, notes — is touched. Re-running this over a worked DB is safe.
    store.seed_from_manifest(mdf.to_dict("records"))
    print(f"      seeded; progress={store.progress()}")

    # Rows the current list drops keep their row and their labels; they just stop being
    # served. Nothing here deletes anything.
    print("[4/4] setting membership from the manifest")
    print(f"      {store.set_membership(list(mdf['uuid']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
