#!/usr/bin/env bash
# data split: the train / validation / test split of the listed shapes that have every file
#
#   bash data_processing/data_split/run.sh --in <uuid list> --out <dir>
#        [--pins-from <splits.json> | --pin-val <file> --pin-test <file>] [--dry-run]
#
# The input is the VLM stage's pass list. A shape is kept when this run's bake pass list and
# complete-views list (<dir>/glb2representations/{bake_pass,views_complete}.txt; missing, the
# stage stops: no other list stands in) and the thumbnail pass list hold it (make_splits
# --require). The thumbnail pass list is made here, by thumbnail_index from TexVerse's
# thumbnails_batch/batch_image_list.txt (seconds). Without pins val and test are drawn fresh;
# --pins-from takes them from a split file (the released splits.json reproduces the released
# split) and --pin-val/--pin-test from two lists. Only the pinned shapes in the input are
# pinned, so a subset gets the subset of the pins. Seconds; the split is one global step, so
# it takes no --rank/--world (run it once, after the other stages' merge). Writes
# <dir>/data_split/: thumbnail_pass.txt, pin_val.txt, pin_test.txt (the pins in the input),
# splits.json, pass.txt (every member). Env: lightgen.

source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"
parse_common "$@"
PIN_VAL="" PIN_TEST="" PINS_FROM=""
set -- "${REST[@]}"
while (($#)); do
  case $1 in
    --pin-val) PIN_VAL=$(abspath "${2:?--pin-val needs a file}"); shift 2 ;;
    --pin-test) PIN_TEST=$(abspath "${2:?--pin-test needs a file}"); shift 2 ;;
    --pins-from) PINS_FROM=$(abspath "${2:?--pins-from needs a split file}"); shift 2 ;;
    *) die "unknown argument: $1 (-h for the usage)" ;;
  esac
done
((MERGE == 0)) && [[ -z $RANK ]] || die "the split is one global step: no --rank/--world, no merge"
[[ -n $IN ]] || die "--in <uuid list> is required (the VLM stage's pass.txt)"
[[ -n $PIN_VAL || -z $PIN_TEST ]] && [[ -n $PIN_TEST || -z $PIN_VAL ]] || die "pass both --pin-val and --pin-test"
[[ -z $PINS_FROM || -z $PIN_VAL ]] || die "pass --pins-from or --pin-val/--pin-test, not both"
stage_setup data_split
((DRY)) || rm -f "$W"/{splits.json,members.txt,thumbnail_pass.txt,pin_val.txt,pin_test.txt,pin_val_all.txt,pin_test_all.txt}
stage_input "$IN"
if stage_empty; then
  printf '{"train": [], "val": [], "test": []}\n' > "$W/splits.json"; : > "$W/members.txt"
  stage_finish "$INPUT"; exit 0
fi

require=()   # this run's lists only: no stage output ships with the repo
for f in bake_pass views_complete; do
  f=$OUT/glb2representations/$f.txt
  if [[ ! -f $f ]]; then
    ((DRY)) || die "no $f: the split reads this run's glb2representations lists (run that stage, or its merge, into $OUT first)"
    log "dry run: --require reads $f once the glb2representations stage has run"
  fi
  require+=("$f")
done
py lightgen -m data_processing.data_split.thumbnail_index --out "$W/thumbnail_pass.txt"
((DRY)) || [[ -s $W/thumbnail_pass.txt ]] || die "thumbnail_index wrote no $W/thumbnail_pass.txt"
require+=("$W/thumbnail_pass.txt")

pin_args=()
if [[ -n $PINS_FROM || -n $PIN_VAL ]]; then
  for s in val test; do
    pin=$PIN_VAL; [[ $s == test ]] && pin=$PIN_TEST
    if [[ -n $PINS_FROM ]]; then
      pin=$W/pin_${s}_all.txt
      run_to "$pin" "$CONDA" run --no-capture-output -n lightgen python -u -c '
import json, sys
sys.stdout.write("".join(f"{u}\n" for u in json.load(open(sys.argv[1]))[sys.argv[2]]))' "$PINS_FROM" "$s"
    fi
    run_to "$W/pin_$s.txt" keep_in_order "$pin" "$INPUT"
    ((DRY)) || log "pins $s: $(count "$W/pin_$s.txt") of the $(count "$pin") in $pin are in the input"
  done
  pin_args=(--pin-val "$W/pin_val.txt" --pin-test "$W/pin_test.txt")
fi

cat_args=()   # the fresh draw's categories: this run's VLM stage, else make_splits' default
[[ -f $OUT/vlm_filtering/input.csv ]] && cat_args=(--categories "$OUT/vlm_filtering/input.csv")
py lightgen -m data_processing.data_split.make_splits --pass-list "$INPUT" --require "${require[@]}" \
  "${pin_args[@]}" "${cat_args[@]}" --out "$W/splits.json"
run_to "$W/members.txt" "$CONDA" run --no-capture-output -n lightgen python -u -c '
import json, sys
d = json.load(open(sys.argv[1]))
sys.stdout.write("".join(f"{u}\n" for k in ("train", "val", "test") for u in d[k]))' "$W/splits.json"
stage_finish "$W/members.txt"
