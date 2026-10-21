"""Inputs of the dataset statistics: the lists and run rows the paper's numbers are counted from.

Each script in this folder prints one paragraph's numbers (paper §4 and supp §A.3) next to
the values the paper states, so a rerun shows at once whether they still hold. The inputs are
the stages' records.
"""

from __future__ import annotations

from pathlib import Path

from utils.paths import RELEASE_ROOT, REPO_ROOT, TEXVERSE_ROOT

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"

# --- metadata filtering -----------------------------------------------------------------
_META = REPO_ROOT / "data_processing/metadata_filtering/out"
#: The metadata scan over every downloaded GLB, as `check_emission scan --out` writes it.
SCAN_OF_RECORD = _META / "scan.csv"
#: Its spec/gloss rows re-read (`check_emission.py specgloss_rescan`), for a scan made before the
#: scanner read the spec/gloss keys.
SPECGLOSS_RESCAN = _META / "specgloss_rescan.csv"
#: The stage's result: the shapes that declare emission and write albedo, metallic, roughness.
METADATA_PASS = _META / "emissive_pass.txt"
#: TexVerse's own record per model (`pbrType` among the fields).
TEXVERSE_METADATA_JSON = TEXVERSE_ROOT / "metadata.json"

# --- VLM filtering and the release ------------------------------------------------------
_VLM = REPO_ROOT / "data_processing/vlm_filtering/out"
#: One row per shape the VLM stage takes in, judged or not, with its category (`build_input.py`).
VLM_INPUT_CSV = _VLM / "input.csv"
#: The VLM stage's verdict per shape (`emit_verdicts.py`).
VLM_VERDICT_CSV = _VLM / "verdict.csv"
#: The released split: train / val / test uuid lists.
RELEASE_SPLITS = RELEASE_ROOT / "splits.json"
