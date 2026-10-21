#!/usr/bin/env bash
# VLM filtering: by default the recorded verdicts; with --vlm the review images
# (../render/) and the VLM check itself.
#
#   bash data_processing/vlm_filtering/run.sh --in <uuid list> --out <dir> [--workers N]
#        [--rank K --world N] [--vlm [--manifest <parquet>] [--atlas_root <bake tree>]]
#        [--record <vlm_pass.txt>] [--dry-run]
#   bash data_processing/vlm_filtering/run.sh merge --in <uuid list> --out <dir> --world N
#
# Default: no renders and no API call. A listed shape passes when the recorded verdicts passed
# it (out/vlm_pass.txt, used only when the sha256 of its uuids is the one this script expects,
# so a regenerated copy with another dated header passes; --record names another, unchecked).
# A shape outside the recorded input has no recorded verdict and does not pass; the log warns
# with their count. --record also takes a split file (a .json such
# as the released splits.json): its train, val and test members are the shapes that pass, so a
# run from the download alone gives the released split back (the VLM passes the split left out
# lack a file, which the split stage drops anyway).
# --vlm renders the eight review images of the list into <dir>/vlm_filtering/render/ (stills,
# lit and emit-only orbits packed into webps, the dark-box pair; LIGHTGEN_REVIEW_ROOT makes that
# folder the only root read and written, so no other review-image root is reused or touched),
# then runs vlm_check.py input, prewarm, atlas, judge (gpt-5.6-luna via the OpenAI API: costs
# money, about $70 for the full list; needs api_key/key.config or OPENAI_API_KEY; skipped when no shape
# has all its images) and verdicts. A renderer that crashes stops the stage (rerun: it
# resumes); the shapes still missing images after the renders are counted in a warning. The
# atlas is read from --atlas_root, default <dir>/glb2representations/tree when it exists, else
# the default dataset tree. The renders are the slow part (CPU Cycles; the dark box on the GPU):
# split them over machines with --rank/--world, all writing one shared <dir>.
# --workers: render processes per lane and prewarm/atlas threads (default 4 and 16).
# Writes <dir>/vlm_filtering/ (with --rank: .../rank<K>/): vlm_pass.txt, pass.txt; with --vlm
# also render_work/ (coverage, work lists, shards, guards), coverage.parquet, input.csv,
# evidence/, atlas_emission.csv, predictions.jsonl, verdict.csv, verdict.meta.json. Env: lightgen.

source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"
parse_common "$@"
VLM=0 MANIFEST="" ATLAS_ROOT="" RECORD=""
set -- "${REST[@]}"
while (($#)); do
  case $1 in
    --vlm) VLM=1; shift ;;
    --manifest) MANIFEST=$(abspath "${2:?--manifest needs a file}"); shift 2 ;;
    --atlas_root) ATLAS_ROOT=$(abspath "${2:?--atlas_root needs a directory}"); shift 2 ;;
    --record) RECORD=$(abspath "${2:?--record needs a file}"); shift 2 ;;
    *) die "unknown argument: $1 (-h for the usage)" ;;
  esac
done
[[ -n $IN ]] || die "--in <uuid list> is required (the glb2representations stage's pass.txt)"
stage_setup vlm_filtering

if ((MERGE)); then
  stage_merge "$IN" vlm_pass.txt pass.txt
  stage_done
  exit 0
fi

stage_input "$IN"
if stage_empty; then : > "$W/vlm_pass.txt"; stage_finish "$INPUT"; exit 0; fi

if ! ((VLM)); then
  if [[ -z $RECORD ]]; then
    # sha256 of the recorded pass list's uuids joined by newlines (vlm_check's header hash), so the
    # dated `#` header does not enter it
    RECORD_SHA256=266465c6fe472887b7248c888504fcf4ad25891003e5e6f343628b22fe101df6
    for f in "$REPO/data_processing/vlm_filtering/out/vlm_pass.txt"; do
      [[ -f $f ]] || continue
      if [[ $(printf '%s' "$(uuids "$f")" | sha256sum) == "$RECORD_SHA256 "* ]]; then RECORD=$f; break; fi
      log "WARNING: $f is not the recorded list (the sha256 of its uuids differs): not used"
    done
  fi
  [[ -n $RECORD ]] || die "no recorded VLM pass list found: pass --record <vlm_pass.txt | the released splits.json>, or --vlm"
  if [[ $RECORD == *.json ]]; then          # a split file: its members are the shapes that pass
    run_to "$W/record_members.txt" "$CONDA" run --no-capture-output -n lightgen python -u -c '
import json, sys
d = json.load(open(sys.argv[1]))
sys.stdout.write("".join(f"{u}\n" for k in ("train", "val", "test") for u in d[k]))' "$RECORD"
    log "no renders, no API call: the members of the split $RECORD pass, every other shape does not"
    RECORD=$W/record_members.txt verdicts=""
  else
    log "no renders, no API call: the recorded verdicts of $RECORD"
    verdicts=$(dirname "$RECORD")/verdict.csv
  fi
  run_to "$W/vlm_pass.txt" keep_in_order "$INPUT" "$RECORD"
  # verdict.csv beside the record has a row for every shape of its input, judged or not
  if ! ((DRY)) && [[ -n $verdicts && -f $verdicts ]]; then
    log "recorded verdicts of the $(count "$INPUT") listed shapes: $(awk -F, 'FILENAME == ARGV[1] { k[$1]; nk++; next }
        FNR > 1 && ($1 in k) { n++; c[$2]++ } END { for (v in c) printf "%s %d, ", v, c[v]; printf "none (not in its input) %d", nk - n }' \
        "$INPUT" "$verdicts")"
    unjudged=$(awk -F, 'FILENAME == ARGV[1] { if (FNR > 1) v[$1]; next } !/^#/ && NF && !($1 in v)' "$verdicts" "$INPUT" | wc -l)
    ((unjudged == 0)) || log "WARNING: $unjudged listed shapes are outside the input of $RECORD: no verdict, dropped; --vlm judges them"
  fi
  stage_finish "$W/vlm_pass.txt"
  exit 0
fi

# --- --vlm: the review images, then the VLM check ------------------------------------
NW=${WORKERS:-4} NT=${WORKERS:-16}
R=$STAGE_DIR/render RW=$W/render_work
[[ -n $ATLAS_ROOT || ! -d $OUT/glb2representations/tree ]] || ATLAS_ROOT=$OUT/glb2representations/tree
ENVS=(LIGHTGEN_REVIEW_ROOT="$R")
[[ -z $ATLAS_ROOT ]] || ENVS+=(LIGHTGEN_ATLAS_ROOT="$ATLAS_ROOT")
pyr() { need_module "$2"; run env "${ENVS[@]}" "$CONDA" run --no-capture-output -n lightgen python -u "$@"; }
PIDS=()
pyr_bg() {  # pyr in the background (the dry run only prints); wait_bg collects them
  need_module "$2"
  show env "${ENVS[@]}" "$CONDA" run --no-capture-output -n lightgen python -u "$@"; printf ' &\n'
  ((DRY)) && return
  env "${ENVS[@]}" "$CONDA" run --no-capture-output -n lightgen python -u "$@" &
  PIDS+=($!)
}
wait_bg() {  # the renderers exit 0 after any per-shape failure, so a non-zero exit is a crash
  local p bad=0
  echo "+ wait"
  for p in "${PIDS[@]}"; do wait "$p" || bad=$((bad + 1)); done
  PIDS=()
  ((bad == 0)) || die "$bad renderer processes failed (log above); rerun to resume"
}
nonempty() { ((DRY)) || [[ -s $1 ]]; }

run mkdir -p "$R"/{glb,emit_only,emissive_brighter,multiview,webp,box_renders} "$RW"
pyr -m data_processing.render.utils coverage --list "$INPUT" --out "$RW/coverage_before.parquet" \
  --work_dir "$RW/missing"
run_to "$RW/missing/stills.txt" sort -u "$RW/missing/glb.txt" "$RW/missing/emit.txt" "$RW/missing/brighter.txt"
for lane in stills orbit_standard orbit_emit_only; do
  if nonempty "$RW/missing/$lane.txt"; then
    pyr -m data_processing.render.dark_box shards --list "$RW/missing/$lane.txt" --world_size "$NW" \
      --with_shard --prefix "$lane" --out_dir "$RW/shards"
    for ((r = 0; r < NW; r++)); do
      shard=$RW/shards/${lane}_rank$(printf %02d "$r").txt
      case $lane in
        stills) pyr_bg -m data_processing.render.single_view --shard "$shard" --rank "$r" ;;
        orbit_*) pyr_bg -m data_processing.render.orbit --shard "$shard" --preset "${lane#orbit_}" --rank "$r" ;;
      esac
    done
    wait_bg
  fi
done
if nonempty "$RW/missing/box.txt"; then
  pyr -m data_processing.render.dark_box shards --list "$RW/missing/box.txt" --world_size 1 \
    --prefix box --out_dir "$RW/shards"
  pyr -m data_processing.render.dark_box --shard "$RW/shards/box_rank00.txt" --out "$R/box_renders" --rank 0
fi
for preset in standard emit_only; do
  if nonempty "$RW/missing/orbit_$preset.txt"; then
    pyr -m data_processing.render.orbit webp --list "$RW/missing/orbit_$preset.txt" --preset "$preset"
  fi
done
# the coverage scan is the verdict on what the renders produced
pyr -m data_processing.render.utils coverage --list "$INPUT" --out "$W/coverage.parquet" \
  --work_dir "$RW/missing_after"
if ! ((DRY)); then
  for f in "$RW"/missing_after/*.txt; do
    [[ ! -s $f ]] || log "WARNING: $(count "$f") shapes still miss $(basename "$f" .txt) images: not judged"
  done
fi

V=(-m data_processing.vlm_filtering.vlm_check)
manifest=()
[[ -z $MANIFEST ]] || manifest=(--manifest "$MANIFEST")
pyr "${V[@]}" input --list "$INPUT" --coverage "$W/coverage.parquet" --out "$W/input.csv" "${manifest[@]}"
pyr "${V[@]}" prewarm --input "$W/input.csv" --evidence-dir "$W/evidence" \
  --unusable "$W/prewarm_unusable.txt" --workers "$NT"
pyr "${V[@]}" atlas --input "$W/input.csv" --out "$W/atlas_emission.csv" --evidence-dir "$W/evidence" \
  --workers "$NT"
resume=()
[[ -f $W/predictions.jsonl ]] && resume=(--resume)
n_judged=1
if ! ((DRY)); then
  n_judged=$("$CONDA" run --no-capture-output -n lightgen python -c '
import sys, pandas as pd
print(int((pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)["judged"] == "True").sum()))' "$W/input.csv")
fi
if ((n_judged > 0)); then
  pyr "${V[@]}" judge --out "$W/predictions.jsonl" --input "$W/input.csv" --evidence-dir "$W/evidence" "${resume[@]}"
else
  log "no shape has all its images: no API call"
  run touch "$W/predictions.jsonl"
fi
pyr "${V[@]}" verdicts --pred "$W/predictions.jsonl" --atlas-stats "$W/atlas_emission.csv" \
  --input "$W/input.csv" --out "$W/verdict.csv" --pass-list "$W/vlm_pass.txt" --meta "$W/verdict.meta.json"
stage_finish "$W/vlm_pass.txt"
