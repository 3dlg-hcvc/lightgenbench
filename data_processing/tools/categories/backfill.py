"""Give a category to the shapes of the albedo-copy check's pass list that the clustered pass
(`categories_final.jsonl`) never saw.

No re-clustering and no new taxonomy: one closed-set call per uncategorized shape against the
existing final taxonomy, the same call `classify pass2` makes, over the same 2x2 composite.
Rows go to `categories_backfill.jsonl`, not into `categories_final.jsonl`, and `source` says
which route a row took. `build_manifest.load_categories` reads both, backfill second.

Run:
  python -m data_processing.tools.categories.backfill --concurrency 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: What `source` a row written by this module carries. `categories_final.jsonl`'s own
#: rows are the clustered pass; keeping the two distinguishable is the point of the column.
BACKFILL_SOURCE = "pass2_backfill"


class MissingCategoriesError(RuntimeError):
    """Raised when the clustered pass's output is not on disk.

    `uuids_to_backfill` treats an absent file as "nothing categorized yet", which is the
    right answer for a pure function and the right answer on a genuine first run. For the
    driver it is a trap: `categories_final.jsonl` holds the clustered pass's categories,
    so if it is merely MISSING rather than empty, the driver would re-categorize the whole
    set -- one paid VLM call per shape, and it would overwrite good categories with
    fresh guesses. Refuse instead, and make the operator say which file to read.
    """


def _categorized(categories_path) -> set:
    """Uuids that already hold a real category.

    A missing file means nothing has been categorized yet, which is a normal first-run
    state, not an error. A row carrying `error` instead of `category` is a FAILED call —
    `classify.run_pass2_async` writes those into the same file — and must not count as done, or the
    shape stays uncategorized forever across every later resume.
    """
    path = Path(categories_path)
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("uuid") and row.get("category"):
            out.add(row["uuid"])
    return out


def uuids_to_backfill(list_path, categories_path) -> list:
    """The set's uuids with no category yet, in list order, deduplicated.

    Order is kept because the composites and the VLM calls are made in this order and a
    stable order makes a resumed run comparable to the one before it.
    """
    have = _categorized(categories_path)
    seen, out = set(), []
    for line in Path(list_path).read_text().splitlines():
        uuid = line.strip()
        if not uuid or uuid in seen:
            continue
        seen.add(uuid)
        if uuid not in have:
            out.append(uuid)
    return out


def missing_composites(uuids, composites_dir) -> list:
    """Uuids with no composite on disk — reported, never guessed.

    A shape whose orbit views never rendered cannot be categorized from a picture. The
    honest outcome is to name it and leave it uncategorized, not to send the VLM a
    partial image and record whatever comes back.
    """
    d = Path(composites_dir)
    return [u for u in uuids if not (d / f"{u}.png").exists()]


def _normalize_source(out_path, source: str = BACKFILL_SOURCE) -> int:
    """Rewrite `source` on every row of the runner's output.

    `classify.run_pass2_async` stamps its own rows `"pass2"`. These rows are a
    backfill against the taxonomy that pass already produced, and the manifest carries
    the column precisely so the two are distinguishable, so they are restamped here.
    Error rows keep their shape and are left uncategorized.
    """
    path = Path(out_path)
    if not path.exists():
        return 0
    out, n = [], 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if isinstance(row, dict) and row.get("category"):
            row["source"] = source
            n += 1
        out.append(json.dumps(row))
    path.write_text("".join(f"{line}\n" for line in out))
    return n


def final_taxonomy(out_full, vocabulary=None) -> list:
    """The closed-set list the backfill asks over: the FINAL category names, with the
    taxonomy's definitions where a name is reachable.

    The clustered pass asked over the merged names and mapped them down afterwards
    (simplify_mapping, shapenet_mapping, then a hand cleanup that is not recorded: the two
    mappings alone miss some final names, `castle`, `ruin` and `other` among them). Asking
    over the final names directly puts every answer straight into the manifest's space,
    with no lossy hop. A name the mappings do reach carries the merged definition; the rest
    are described by their name alone. `other` is appended by build_taxonomy_block.
    """
    out_full = Path(out_full)
    vocab_path = Path(vocabulary) if vocabulary else (
        Path(__file__).resolve().parents[1] / "human_annotation" / "_data" / "categories.json")
    names = [n for n in json.loads(vocab_path.read_text()) if n not in ("unknown", "other")]
    merged = json.loads((out_full / "merged_taxonomy.json").read_text())
    simp = json.loads((out_full / "simplify_mapping.json").read_text())
    shp = json.loads((out_full / "shapenet_mapping.json").read_text())
    defs = {}
    for t in merged:                       # merged name -> final name, keep the first definition
        final = shp.get(simp.get(t["name"], t["name"]), simp.get(t["name"], t["name"]))
        defs.setdefault(final, t["definition"])
    return [{"name": n, "definition": defs.get(n, n.replace("_", " "))} for n in names]


def main() -> None:
    import pandas as pd

    from data_processing.tools.categories import composites
    from data_processing.tools.categories.utils import (
        COMPOSITES_FULL, MULTIVIEW_AXES, OUT_FULL_DIR, PANE_SIZE, THUMBNAILS_ROOT,
    )
    from utils import review_images
    from utils.metadata_io import shard_of
    from utils.paths import ALBEDO_COPY_PASS_LIST, TEXVERSE_METADATA_PARQUET

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", default=str(ALBEDO_COPY_PASS_LIST))
    ap.add_argument("--categories", default=str(OUT_FULL_DIR / "categories_final.jsonl"))
    ap.add_argument("--out", default=str(OUT_FULL_DIR / "categories_backfill.jsonl"))
    ap.add_argument("--composites", default=str(COMPOSITES_FULL))
    ap.add_argument("--metadata", default=str(TEXVERSE_METADATA_PARQUET))
    ap.add_argument("--vocabulary", default=None,
                    help="the manifest's categories.json (the final names); default "
                         "data_processing/tools/human_annotation/_data/categories.json")
    ap.add_argument("--concurrency", type=int, default=32, help="VLM calls in flight")
    ap.add_argument("--workers", type=int, default=16, help="composite-building threads")
    ap.add_argument("--allow_missing_categories", action="store_true",
                    help="proceed even with no categories file (re-categorizes everything)")
    ap.add_argument("--build_only", action="store_true",
                    help="build composites and report coverage, make no VLM calls")
    a = ap.parse_args()

    if not Path(a.categories).exists() and not a.allow_missing_categories:
        raise MissingCategoriesError(
            f"{a.categories} is not on disk. Every shape would look uncategorized and the "
            f"whole set would be re-categorized. The clustered categories also survive in "
            f"the annotation tool's manifest.parquet and in items.auto_category of its DB -- "
            f"point --categories at a jsonl "
            f"rebuilt from one of those, or pass --allow_missing_categories if you really "
            f"do mean to categorize everything.")

    composites_dir = Path(a.composites)
    todo = uuids_to_backfill(a.list, a.categories)
    # Resume: what this backfill already answered is not asked again (the closed-set call
    # is paid per shape). Error rows do not count as answered.
    already = _categorized(a.out)
    todo = [u for u in todo if u not in already]
    print(f"[backfill] {len(todo)} uuids without a category ({len(already)} already backfilled)",
          flush=True)
    if not todo:
        return

    # 2. Composites, from whichever orbit root has the shape's views.
    #    The bucket comes from the metadata's ditem_dir, never from the uuid: the shard
    #    tree keys on `000-000` of `<ditem_dir>`, and a uuid carries no such prefix.
    ditem = pd.read_parquet(a.metadata, columns=["ditem_dir"])["ditem_dir"].to_dict()
    index = composites.build_thumbnail_index(THUMBNAILS_ROOT)
    composites_dir.mkdir(parents=True, exist_ok=True)

    def _one(uuid):
        """Build one composite; returns 'built' | 'skipped' | 'failed'. IO-bound (four
        PNG reads over NFS), so a thread pool below runs many at once."""
        out_path = composites_dir / f"{uuid}.png"
        if out_path.exists():
            return "skipped"
        batch, dd = index.get(uuid), ditem.get(uuid)
        if batch is None or not isinstance(dd, str):
            return "failed"
        # The orbit views live in whichever root has them (`utils.review_images`:
        # LEGACY_ORBIT_ROOT or NEW_ORBIT_ROOT). Resolve one view and take its root; a
        # shape's six views always come from one root. The builder itself looks in ONE
        # root and returns False silently when a pane is missing, so its result must be
        # checked.
        shard = shard_of(dd)
        view = review_images.orbit_view("standard", MULTIVIEW_AXES[0], shard, uuid)
        if view is None:
            return "failed"
        root = Path(view).parents[4]            # <root>/standard/<axis>/<shard>/<uuid>/render.png
        try:
            ok = composites.build_composite_2x2(
                uuid=uuid, thumbnail_root=THUMBNAILS_ROOT, thumbnail_batch=batch,
                multiview_root=root, multiview_bucket=shard, out_path=out_path,
                pane_size=PANE_SIZE)          # 512, what the clustered pass used
            return "built" if ok else "failed"
        except Exception as e:                  # noqa: BLE001 - reported, not raised
            print(f"[backfill] composite {uuid}: {type(e).__name__}: {e}", flush=True)
            return "failed"

    from concurrent.futures import ThreadPoolExecutor
    counts = {"built": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(_one, todo), 1):
            counts[r] += 1
            if i % 1000 == 0:
                print(f"[backfill] composites {i}/{len(todo)} {counts}", flush=True)
    built, failed = counts["built"], counts["failed"]
    absent = missing_composites(todo, composites_dir)
    print(f"[backfill] composites built={built} failed={failed} "
          f"still absent={len(absent)}", flush=True)
    if a.build_only:
        return

    # 3. One closed-set call each, against the taxonomy the clustered pass already made.
    #    Same prompt, schema and runner as the noise pass -- this is that call, over a
    #    different uuid list, so it must not diverge from it.
    import asyncio

    from data_processing.tools.categories import classify
    from data_processing.tools.categories.classify import (
        build_pass2_schema, build_taxonomy_block,
    )

    taxonomy = final_taxonomy(OUT_FULL_DIR, a.vocabulary)
    prompt = ((Path(classify.__file__).parent / "prompts" / "pass2_closed.txt").read_text()
              .replace("{taxonomy_block}", build_taxonomy_block(taxonomy)))
    ready = [u for u in todo if u not in set(absent)]
    print(f"[backfill] categorizing {len(ready)} against {len(taxonomy)} categories",
          flush=True)
    asyncio.run(classify.run_pass2_async(ready, composites_dir, prompt,
                                         build_pass2_schema(taxonomy), Path(a.out), a.concurrency))
    n = _normalize_source(a.out)
    print(f"[backfill] {n} categorized rows -> {a.out}", flush=True)
    if absent:
        print(f"[backfill] {len(absent)} left uncategorized (no composite)", flush=True)


if __name__ == "__main__":
    main()
