"""The source counts of the data filtering pipeline (paper §4, *Data Filtering Pipeline*;
supp §A.1, *Metadata Filtering*).

  records       TexVerse's models (its `metadata.json`): "858k 3D models from Sketchfab"
  downloaded    the GLBs the metadata scan read: "857k downloaded"
  PBR-labeled   TexVerse's own `pbrType` flag (metalness or specular): "only 158k"
  all three     shapes that write albedo, metallic and roughness on some used material,
                as a texture or a number: "530k assets"
  emission      shapes with a used material that declares emission: "103,762"
  candidates    both of the last two, the metadata stage's result: "83,124"

The last three use the metadata stage's own rules (`metadata_filtering/check_emission.py`
`declaration_verdict`, `pbr_verdict`) on the metadata scan, with its spec/gloss rows
overridden by `specgloss_rescan.csv` as `check_emission.py list --specgloss_rescan` does.
Without a rescan file (a fresh scan needs none) the scan's rows are counted as they are.

Usage (lightgen env, repo root; reads the 666 MB metadata.json):

    conda run --no-capture-output -n lightgen python -m data_processing.tools.dataset_statistics.source_counts
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from data_processing.tools.dataset_statistics import config
from data_processing.metadata_filtering.check_emission import declaration_verdict, pbr_verdict

#: The paper rounds to thousands in the main text; the supplement gives the last two exactly.
PAPER = {"records": "858k", "downloaded": "857k", "PBR-labeled": "158k", "all three": "530k",
         "emission": "103,762", "candidates": "83,124"}


def _shown(name: str, n: int) -> dict:
    """How the paper may print `n`: exactly, or in thousands rounded down or to the nearest.
    The paper is not consistent: "858k" and "158k" are rounded down, "857k" to the nearest."""
    if not PAPER[name].endswith("k"):
        return {"exact": f"{n:,}"}
    return {"rounded down": f"{n // 1000}k", "nearest": f"{round(n / 1000)}k"}


def source_counts(metadata_json, scan_csv, specgloss_rescan=None) -> dict:
    """`specgloss_rescan` None: no override (a scan whose scanner reads the spec/gloss keys
    needs none)."""
    meta = json.load(open(metadata_json))
    flagged = {u for u, m in meta.items() if m.get("pbrType") in ("metalness", "specular")}
    csv.field_size_limit(10**7)
    fix = {r["uuid"]: r for r in csv.DictReader(open(specgloss_rescan))} if specgloss_rescan else {}
    n_rows = 0
    all3, emission, candidates = set(), set(), set()
    with open(scan_csv) as f:
        for r in csv.DictReader(f):
            n_rows += 1
            if r["uuid"] in fix:
                r.update({k: v for k, v in fix[r["uuid"]].items() if k != "uuid"})
            if pbr_verdict(r):
                all3.add(r["uuid"])
            if declaration_verdict(r):
                emission.add(r["uuid"])
                if pbr_verdict(r):
                    candidates.add(r["uuid"])
    types = collections.Counter(m.get("pbrType") for m in meta.values())
    return {
        "records": len(meta), "downloaded": n_rows, "PBR-labeled": len(flagged),
        "all three": len(all3), "emission": len(emission), "candidates": len(candidates),
        "_pbrType": dict(types), "_all three, flagged": len(all3 & flagged),
        "_emission, flagged": len(emission & flagged),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata_json", default=str(config.TEXVERSE_METADATA_JSON))
    ap.add_argument("--scan", default=str(config.SCAN_OF_RECORD))
    ap.add_argument("--specgloss_rescan", default=None,
                    help=f"spec/gloss overrides of a pre-fix scan (default: {config.SPECGLOSS_RESCAN} "
                         f"when it exists, else none)")
    a = ap.parse_args()
    rescan = a.specgloss_rescan
    if rescan is None and Path(config.SPECGLOSS_RESCAN).is_file():
        rescan = str(config.SPECGLOSS_RESCAN)
    print(f"spec/gloss rescan: {rescan or 'none (the scan rows as they are)'}")
    got = source_counts(a.metadata_json, a.scan, rescan)
    match = True
    for name, paper in PAPER.items():
        how = [k for k, v in _shown(name, got[name]).items() if v == paper]
        match &= bool(how)
        print(f"{name:12} {got[name]:9,}   paper {paper:>8}   {', '.join(how) if how else 'DIFFERS'}")
    for name in ("_pbrType", "_all three, flagged", "_emission, flagged"):
        print(f"{name[1:]}: {got[name]}")
    print(f"matches the paper: {match}")


if __name__ == "__main__":
    main()
