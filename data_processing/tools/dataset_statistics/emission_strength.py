"""Emission strength over the metadata stage's candidates (supp §A.3, *Emission strength*).

`KHR_materials_emissive_strength` multiplies a material's emission; the paper drops it
(§4: "We discard the glTF emission strength, since most assets use the default value").
This counts, over the shapes that pass metadata filtering, the largest strength any of their
used materials declares (`max_strength` of the metadata scan; 1 when none is declared):
exactly 1, exactly 10, and anything else.

Usage (lightgen env, repo root):

    conda run --no-capture-output -n lightgen python -m data_processing.tools.dataset_statistics.emission_strength
"""

from __future__ import annotations

import argparse

import pandas as pd

from data_processing.tools.dataset_statistics import config

#: The supplement's emission-strength table (supp §A.3).
PAPER = {"1": 66_693, "10": 7_872, "other": 8_559}


def strength_counts(scan_csv, pass_list) -> dict:
    keep = {ln.strip() for ln in open(pass_list) if ln.strip()}
    df = pd.read_csv(scan_csv, usecols=["uuid", "max_strength"], dtype={"uuid": str})
    s = df.loc[df.uuid.isin(keep), "max_strength"]
    assert len(s) == len(keep) and s.notna().all(), "every candidate has one scan row"
    return {"1": int((s == 1).sum()), "10": int((s == 10).sum()), "other": int(((s != 1) & (s != 10)).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default=str(config.SCAN_OF_RECORD))
    ap.add_argument("--pass_list", default=str(config.METADATA_PASS))
    a = ap.parse_args()
    got = strength_counts(a.scan, a.pass_list)
    n = sum(got.values())
    for k, v in got.items():
        print(f"strength {k:>5}: {v:6,} ({100 * v / n:.1f}%)   paper {PAPER[k]:,}")
    print(f"total {n:,}; matches the paper: {got == PAPER}")


if __name__ == "__main__":
    main()
