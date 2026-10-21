"""The two VLM labeling passes over the composites (gpt-4o-mini, async).

  pass1   a free-form name for every shape (prompt `prompts/pass1_freeform_2x2.txt`), the text
          that `embed` and `cluster` group. Output `out_full/pass1_freeform.jsonl`.
  pass2   one category per shape in the merged taxonomy's names (`merged_taxonomy.json`, the
          names of `taxonomy merge` + `cleanup`). A shape in a named text cluster takes its
          cluster's merged name through `merge_mapping.json`, with no call; only the HDBSCAN
          noise (cluster_id -1) is asked, closed-set (prompt `prompts/pass2_closed.txt`, the
          answer an enum built from the taxonomy at runtime, plus `other`). Output
          `out_full/categories_final.jsonl`, one row per uuid:
          {"uuid": ..., "category": ..., "source": "cluster" | "pass2"}; `relabel` then puts
          the rows in the final names.

Both are single-process and resume-safe: a uuid already in the output is skipped. A call that
fails after 4 retries writes {"uuid": ..., "error": ...} so it does not block a rerun.

Usage:
    python -m data_processing.tools.categories.classify pass1 --concurrency 100
    python -m data_processing.tools.categories.classify pass2 --concurrency 100
"""
import argparse
import asyncio
import json
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Type

from pydantic import BaseModel, ConfigDict, Field

from data_processing.tools.categories.utils import (
    COMPOSITES_FULL, OUT_FULL_DIR, VLM_MODEL, AsyncVlmClient, Pass1Output,
    make_async_openai_client, run_subcommand,
)


def _load_existing_uuids(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    seen: Set[str] = set()
    for line in out_path.read_text().splitlines():
        if line.strip():
            seen.add(json.loads(line)["uuid"])
    return seen


# ---------------------------------------------------------------------------------------------
# pass1: free-form labels

async def _call_one(
    *,
    uuid: str,
    composite_path: Path,
    prompt: str,
    vlm: AsyncVlmClient,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        try:
            result: Pass1Output = await vlm.call_structured(
                prompt=prompt,
                image_paths=[composite_path],
                schema=Pass1Output,
            )
            return {"uuid": uuid, **result.model_dump()}
        except Exception as e:
            return {"uuid": uuid, "error": str(e)}


async def run_pass1_async(
    uuids: List[str],
    composites_dir: Path,
    prompt: str,
    out_path: Path,
    concurrency: int,
) -> None:
    sem = asyncio.Semaphore(concurrency)
    client = make_async_openai_client()
    vlm = AsyncVlmClient(client=client, model=VLM_MODEL)
    n = len(uuids)
    print(f"[pass1-async] {n} uuids to process; concurrency={concurrency}")
    t0 = time.time()
    done = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = out_path.open("a")

    async def _worker(uuid: str):
        composite = composites_dir / f"{uuid}.png"
        if not composite.exists():
            return None
        row = await _call_one(uuid=uuid, composite_path=composite,
                                prompt=prompt, vlm=vlm, sem=sem)
        return row

    tasks = [asyncio.create_task(_worker(u)) for u in uuids]
    for fut in asyncio.as_completed(tasks):
        row = await fut
        if row is None:
            done += 1
            continue
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
        done += 1
        if done % 200 == 0:
            elapsed = time.time() - t0
            rps = done / elapsed if elapsed > 0 else 0
            eta_min = (n - done) / rps / 60 if rps > 0 else float("inf")
            print(f"   [{done}/{n}] {rps:.1f} RPS  ETA {eta_min:.1f} min")
    out_fh.close()


def pass1_main(argv=None) -> None:
    """Free-form labels for every composite -> out_full/pass1_freeform.jsonl."""
    parser = argparse.ArgumentParser(prog="python -m data_processing.tools.categories.classify pass1")
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: only process the first N composites.")
    args = parser.parse_args(argv)

    composites_dir = COMPOSITES_FULL
    prompt_path = Path(__file__).parent / "prompts" / "pass1_freeform_2x2.txt"
    out_path = OUT_FULL_DIR / "pass1_freeform.jsonl"

    uuids = sorted(p.stem for p in composites_dir.glob("*.png"))
    existing = _load_existing_uuids(out_path)
    uuids = [u for u in uuids if u not in existing]
    if args.limit:
        uuids = uuids[: args.limit]
    print(f"[pass1-async] composites: {len(list(composites_dir.glob('*.png')))} ; "
          f"already done: {len(existing)} ; to process: {len(uuids)}")

    if not uuids:
        print("[pass1-async] nothing to do.")
        return

    prompt = prompt_path.read_text()
    asyncio.run(run_pass1_async(
        uuids=uuids,
        composites_dir=composites_dir,
        prompt=prompt,
        out_path=out_path,
        concurrency=args.concurrency,
    ))
    print(f"[pass1-async] done. wrote {out_path}")


# ---------------------------------------------------------------------------------------------
# pass2: closed-set categories (the backfill makes the same call over its own uuids)

def build_taxonomy_block(taxonomy: List[Dict]) -> str:
    lines = [f"  {t['name']} — {t['definition']}" for t in taxonomy]
    lines.append("  other — does not fit any of the above")
    return "\n".join(lines)


def build_pass2_schema(taxonomy: List[Dict]) -> Type[BaseModel]:
    names = [t["name"] for t in taxonomy] + ["other"]
    CategoryEnum = Enum("CategoryEnum", {n: n for n in names}, type=str)

    class Pass2Closed(BaseModel):
        model_config = ConfigDict(extra="forbid", use_enum_values=True)
        category: CategoryEnum = Field(...)

    return Pass2Closed


async def run_pass2_async(
    uuids: List[str],
    composites_dir: Path,
    prompt: str,
    schema_cls,
    out_path: Path,
    concurrency: int,
) -> None:
    sem = asyncio.Semaphore(concurrency)
    client = make_async_openai_client()
    vlm = AsyncVlmClient(client=client, model=VLM_MODEL)
    n = len(uuids)
    print(f"[pass2-noise] {n} noise uuids; concurrency={concurrency}")
    t0 = time.time()
    done = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = out_path.open("a")

    async def _worker(uuid: str):
        composite = composites_dir / f"{uuid}.png"
        if not composite.exists():
            return None
        async with sem:
            try:
                result = await vlm.call_structured(
                    prompt=prompt, image_paths=[composite], schema=schema_cls,
                )
                return {"uuid": uuid, "category": result.category, "source": "pass2"}
            except Exception as e:
                return {"uuid": uuid, "error": str(e), "source": "pass2"}

    tasks = [asyncio.create_task(_worker(u)) for u in uuids]
    for fut in asyncio.as_completed(tasks):
        row = await fut
        if row is None:
            done += 1
            continue
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
        done += 1
        if done % 500 == 0:
            elapsed = time.time() - t0
            rps = done / elapsed if elapsed > 0 else 0
            eta_min = (n - done) / rps / 60 if rps > 0 else float("inf")
            print(f"   [{done}/{n}] {rps:.1f} RPS  ETA {eta_min:.1f} min")
    out_fh.close()


def pass2_main(argv=None) -> None:
    """Cluster rows by lookup, the noise by a closed-set call -> out_full/categories_final.jsonl."""
    parser = argparse.ArgumentParser(prog="python -m data_processing.tools.categories.classify pass2")
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args(argv)

    # Load the merged taxonomy (the broad categories).
    merged_path = OUT_FULL_DIR / "merged_taxonomy.json"
    mapping_path = OUT_FULL_DIR / "merge_mapping.json"
    cluster_path = OUT_FULL_DIR / "cluster_text_assignments.jsonl"
    fine_tax_path = OUT_FULL_DIR / "taxonomy.json"
    out_path = OUT_FULL_DIR / "categories_final.jsonl"
    composites_dir = COMPOSITES_FULL

    merged = json.loads(merged_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    fine_tax = {e["name"]: e for e in json.loads(fine_tax_path.read_text()) if "name" in e}
    print(f"[pass2-noise] {len(merged)} broad categories")

    # Build {fine_cluster_id -> broad_name} so we can look up clustered uuids fast.
    # taxonomy.json has cluster_id; we map cluster -> fine_name -> broad_name via mapping.
    cluster_to_broad: Dict[int, str] = {}
    for entry in json.loads(fine_tax_path.read_text()):
        if "name" in entry and entry["name"] in mapping:
            cluster_to_broad[entry["cluster_id"]] = mapping[entry["name"]]

    # Pass 2 schema + prompt for noise items.
    schema_cls = build_pass2_schema(merged)
    block = build_taxonomy_block(merged)
    prompt_template = (Path(__file__).parent / "prompts" / "pass2_closed.txt").read_text()
    prompt = prompt_template.replace("{taxonomy_block}", block)

    # Resume support: skip uuids already in out_path.
    existing = _load_existing_uuids(out_path)

    # Walk the cluster assignments file:
    #   - If cluster_id != -1, write the broad name immediately ("cluster" source)
    #   - If cluster_id == -1, queue the uuid for Pass 2
    noise_uuids: List[str] = []
    fast_writes = 0
    out_fh = out_path.open("a")
    for line in cluster_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = row["uuid"]
        if uid in existing:
            continue
        cid = int(row["cluster_id"])
        if cid == -1:
            noise_uuids.append(uid)
        else:
            broad = cluster_to_broad.get(cid)
            if broad is not None:
                out_fh.write(json.dumps({"uuid": uid, "category": broad,
                                          "source": "cluster"}) + "\n")
                fast_writes += 1
            else:
                noise_uuids.append(uid)   # fallback if mapping missing
    out_fh.flush()
    out_fh.close()
    print(f"[pass2-noise] cluster-derived rows written: {fast_writes}")
    print(f"[pass2-noise] noise uuids needing Pass 2: {len(noise_uuids)}")

    if not noise_uuids:
        print("[pass2-noise] nothing to do.")
        return

    asyncio.run(run_pass2_async(
        uuids=noise_uuids,
        composites_dir=composites_dir,
        prompt=prompt,
        schema_cls=schema_cls,
        out_path=out_path,
        concurrency=args.concurrency,
    ))
    print(f"[pass2-noise] done. wrote {out_path}")


def main(argv=None) -> None:
    run_subcommand({"pass1": pass1_main, "pass2": pass2_main}, __doc__, argv,
                   prog="python -m data_processing.tools.categories.classify")


if __name__ == "__main__":
    main()
