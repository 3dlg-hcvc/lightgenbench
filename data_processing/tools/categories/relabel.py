"""The last step of the category run: the merged names become the final ones.

`classify pass2` labels the shapes in the merged-name space of `merged_taxonomy.json`; this
relabel puts them in the final names by the lookup `shapenet_mapping[name]`, with `other` kept.
The merged-name rows are not kept, so `--check` checks the vocabulary only: the mapping's keys
are the merged names, and its values plus `other` are exactly the names of
`categories_final.jsonl`. (Going through `simplify_mapping.json` first, as
`backfill.final_taxonomy` does for its definitions, misses `castle` and `ruin`.)

Usage (lightgen env, repo root):

    conda run --no-capture-output -n lightgen python -m \\
      data_processing.tools.categories.relabel --check
    ... --in <rows in the merged-name space>.jsonl --out <rows in the final space>.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_processing.tools.categories.utils import OUT_FULL_DIR


def load_mapping(out_full: Path) -> dict:
    return json.loads((out_full / "shapenet_mapping.json").read_text())


def relabel(name: str, mapping: dict) -> str:
    return "other" if name == "other" else mapping[name]


def check(out_full: Path) -> bool:
    mapping = load_mapping(out_full)
    merged = {t["name"] for t in json.loads((out_full / "merged_taxonomy.json").read_text())}
    final = {json.loads(ln)["category"] for ln in open(out_full / "categories_final.jsonl")}
    keys_ok = set(mapping) == merged
    image = {relabel(n, mapping) for n in merged} | {"other"}
    print(f"mapping keys = the {len(merged)} merged names: {keys_ok}")
    print(f"mapping values + other = the {len(final)} names of categories_final.jsonl: {image == final}"
          f" (missing {sorted(final - image)}, extra {sorted(image - final)})")
    return keys_ok and image == final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_full", default=str(OUT_FULL_DIR))
    ap.add_argument("--check", action="store_true", help="check the mapping against the final vocabulary")
    ap.add_argument("--in", dest="inp", help="JSONL rows with `category` in the merged-name space")
    ap.add_argument("--out", help="where to write the relabelled rows")
    a = ap.parse_args()
    out_full = Path(a.out_full)
    if a.check:
        raise SystemExit(0 if check(out_full) else 1)
    if not (a.inp and a.out):
        ap.error("give --check, or both --in and --out")
    mapping = load_mapping(out_full)
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for ln in fi:
            r = json.loads(ln)
            r["category"] = relabel(r["category"], mapping)
            fo.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
