#!/usr/bin/env bash
# metadata filtering: the GLB index, the GLB JSON scan of a uuid list, then the pass list from it
#
#   bash data_processing/metadata_filtering/run.sh [--in <uuid list>] --out <dir> [--workers N]
#        [--rank K --world N] [--dry-run]
#   bash data_processing/metadata_filtering/run.sh merge [--in <uuid list>] --out <dir> --world N
#
# First it builds the GLB index every stage reads, data/texverse_index.parquet, from the
# download on disk (glb_index.py: the TexVerse id list, the TexVerse-1K tree, the thumbnail
# list), at every run, and warns when that download is not the released data's.
# Without --in it scans every downloaded GLB, a long scan for one machine, so split it over
# machines with --rank K --world N (K = 0..N-1, one per machine), then run `merge` once.
# Writes <dir>/metadata_filtering/ (with --rank: .../rank<K>/):
#   scan.csv                 one row per shape; the resume marker (a rerun skips its rows)
#   emissive_pass.txt        both rules: declares emission and writes albedo, metallic, roughness
#   emission_only_pass.txt   the emission rule alone
#   pass.txt                 emissive_pass.txt in input order: the stage's result
# --workers: thread pool of the scan (default 32). Env: lightgen.

source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"
parse_common "$@"
((${#REST[@]} == 0)) || die "unknown argument: ${REST[0]} (-h for the usage)"
stage_setup metadata_filtering
WORKERS=${WORKERS:-32}

downloads() {  # every downloaded GLB, the scan's own default set, in the index's order
  run_to "$1" "$CONDA" run --no-capture-output -n lightgen python -u -c '
import os, sys, pandas as pd
from data_processing.metadata_filtering.check_emission import DOWNLOADED_COLUMN as C
from utils.paths import TEXVERSE_METADATA_PARQUET as P
m = pd.read_parquet(P, columns=[C])
sys.stdout.write("".join(f"{u}\n" for u in m.index[m[C] == True]))
sys.stdout.flush(); os._exit(0)'    # os._exit: see glb2representations/run.sh
}

py lightgen -m data_processing.metadata_filtering.glb_index

if ((MERGE)); then
  order=$IN
  if [[ -z $order ]]; then order=$STAGE_DIR/downloads.txt; downloads "$order"; fi
  stage_merge "$order" emissive_pass.txt emission_only_pass.txt pass.txt
  stage_done
  exit 0
fi

src=$IN
if [[ -z $src ]]; then
  [[ -n $RANK ]] || log "WARNING: no --in, so every downloaded GLB (a long scan): consider --rank/--world"
  src=$W/downloads.txt
  downloads "$src"
fi
stage_input "$src"
if stage_empty; then : > "$W/emissive_pass.txt"; : > "$W/emission_only_pass.txt"; stage_finish "$INPUT"; exit 0; fi

py lightgen -m data_processing.metadata_filtering.check_emission scan \
  --shas_file "$INPUT" --out "$W/scan.csv" --workers "$WORKERS"
py lightgen -m data_processing.metadata_filtering.check_emission list \
  --csv "$W/scan.csv" --out_dir "$W"
stage_finish "$W/emissive_pass.txt"
