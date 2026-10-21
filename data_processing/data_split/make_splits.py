"""The train / validation / test split of the released dataset.

  python -m data_processing.data_split.make_splits \
      --require data_processing/glb2representations/out/bake_pass.txt \
                data_processing/glb2representations/out/views_complete.txt \
                data_processing/data_split/out/thumbnail_pass.txt \
      [--pins-from <splits.json> | --pin-val <val.txt> --pin-test <test.txt>] \
      --out data_processing/data_split/out/splits.json

Input: the VLM filtering stage's pass list (`vlm_filtering/out/vlm_pass.txt`), kept only where
every `--require` list also holds the shape (a released shape has every file). Rules:

1. Without pins, val and test are drawn from it, `--target` each, category-stratified (seed 42,
   proportional largest-remainder allocation), and train is the rest.
2. With pins, val and test are those lists verbatim (they must be inside the filtered pass list)
   and train is the rest. `--pins-from` takes them from a split file: the released
   `splits.json` reproduces the released split, which a fresh draw does not. `--pin-val` /
   `--pin-test` take two uuid lists.
3. No family handling: the duplicate check already removed the copies.

Output: one file, `{"train": [...], "val": [...], "test": [...]}`, sorted uuids.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from utils.metadata_io import read_uuids, sha256_of_file
from utils.paths import REPO_ROOT

PASS_LIST = REPO_ROOT / "data_processing/vlm_filtering/out/vlm_pass.txt"
INPUT_CSV = REPO_ROOT / "data_processing/vlm_filtering/out/input.csv"          # uuid -> category
TARGET = 200
SEED = 42
KEYS = ("train", "val", "test")


def allocate(counts: dict, n: int) -> dict:
    """Proportional largest-remainder allocation of `n` draws over categories (no floor:
    with fewer draws than categories a floor would exceed the budget)."""
    total = sum(counts.values())
    if n <= 0 or total == 0:
        return {c: 0 for c in counts}
    exact = {c: n * k / total for c, k in counts.items()}
    alloc = {c: int(v) for c, v in exact.items()}
    left = n - sum(alloc.values())
    for c, _ in sorted(exact.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))[:left]:
        alloc[c] += 1
    for c in alloc:
        alloc[c] = min(alloc[c], counts[c])
    return alloc


def draw(pool: dict, n: int, rng: random.Random) -> list:
    """`pool`: uuid -> category. Category-stratified draw of `n` uuids, deterministic."""
    by_cat = defaultdict(list)
    for u in sorted(pool):
        by_cat[pool[u]].append(u)
    alloc = allocate({c: len(v) for c, v in by_cat.items()}, n)
    out = []
    for c in sorted(by_cat):
        members = by_cat[c][:]
        rng.shuffle(members)
        out += members[:alloc[c]]
    return sorted(out)


def build(passed: list, categories: dict, target: int, seed: int) -> dict:
    rng = random.Random(seed)
    pool = {u: categories.get(u, "unknown") for u in set(passed)}
    val = draw(pool, target, rng)
    pool = {u: c for u, c in pool.items() if u not in set(val)}
    test = draw(pool, target, rng)
    held = set(val) | set(test)
    train = sorted(u for u in passed if u not in held)
    return dict(train=train, val=sorted(val), test=sorted(test))


def build_pinned(passed: list, val: list, test: list) -> dict:
    ps = set(passed)
    outside = sorted((set(val) | set(test)) - ps)
    assert not outside, f"{len(outside)} pinned shapes not in the filtered pass list, e.g. {outside[:3]}"
    both = set(val) & set(test)
    assert not both, f"{len(both)} shapes pinned to both val and test"
    held = set(val) | set(test)
    return dict(train=sorted(u for u in ps if u not in held), val=sorted(val), test=sorted(test))


def filter_required(passed: list, required: list) -> list:
    """`passed` in order, kept where every set in `required` holds the uuid."""
    return [u for u in passed if all(u in r for r in required)]


def write_splits(path, splits: dict) -> str:
    data = {k: sorted(splits[k]) for k in KEYS}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data) + "\n")
    return sha256_of_file(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pass-list", default=str(PASS_LIST),
                    help="the VLM stage's pass list (default vlm_filtering/out/vlm_pass.txt; "
                         "`data_split/run.sh` passes its own input)")
    ap.add_argument("--require", nargs="*", default=[])
    ap.add_argument("--categories", default=str(INPUT_CSV),
                    help="uuid -> category for the fresh draw: the VLM stage's input.csv "
                         "(default vlm_filtering/out/input.csv; `data_split/run.sh` passes its run's)")
    ap.add_argument("--pins-from", help="a split JSON (e.g. the released splits.json): its val and "
                                        "test are the pins")
    ap.add_argument("--pin-val")
    ap.add_argument("--pin-test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    assert (a.pin_val is None) == (a.pin_test is None), "pass both --pin-val and --pin-test or neither"
    assert not (a.pins_from and a.pin_val), "pass --pins-from or --pin-val/--pin-test, not both"
    raw = read_uuids(a.pass_list)
    passed = filter_required(raw, [set(read_uuids(r)) for r in a.require])
    if a.pins_from:
        pins = json.loads(Path(a.pins_from).read_text())
        splits = build_pinned(passed, pins["val"], pins["test"])
    elif a.pin_val:
        splits = build_pinned(passed, read_uuids(a.pin_val), read_uuids(a.pin_test))
    else:
        if not Path(a.categories).is_file():
            raise SystemExit(f"no category CSV at {a.categories}: the fresh draw needs the VLM stage's "
                             f"input.csv (pass --categories), or pin the split")
        categories = {r["uuid"]: r["category"] for r in csv.DictReader(open(a.categories))}
        splits = build(passed, categories, a.target, a.seed)
    sha = write_splits(a.out, splits)
    print(f"pass {len(raw)} -> complete {len(passed)}: train {len(splits['train'])}  val {len(splits['val'])}  "
          f"test {len(splits['test'])}  -> {a.out}  sha256 {sha}")


if __name__ == "__main__":
    main()
