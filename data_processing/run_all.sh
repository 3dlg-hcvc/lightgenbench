#!/usr/bin/env bash
# Every stage of the dataset build, in run order, each stage's pass.txt the next one's input
# (the repository README, Dataset Build):
#   metadata_filtering -> heuristic_filtering -> glb2representations -> vlm_filtering -> data_split
# The bake runs before the VLM stage because the judge's eighth image is the baked atlas.
#
#   bash data_processing/run_all.sh [--list <uuid list>] --out <dir> [--workers N]
#        [--rank K --world N] [--from <stage>] [--to <stage>] [--vlm]
#        [--pins-from <splits.json> | --pin-val <file> --pin-test <file>] [--dry-run]
#   bash data_processing/run_all.sh --merge [--list <uuid list>] --out <dir> --world N
#        [--from <stage>] [--to <stage>] [--vlm] [--pins-from ... | --pin-val ... --pin-test ...]
#
# --list feeds the first stage that runs; without it metadata_filtering scans every download
# and a later --from stage reads the previous stage's <dir>/<stage>/pass.txt. The GLB index
# every stage reads (data/texverse_index.parquet) is built first, from the download on disk.
# --vlm, --pins-from, --pin-val/--pin-test go to their stage (see its run.sh -h; the split draws
# val and test fresh unless --pins-from <the released splits.json>); --workers to every stage
# (each has its own default). Without --vlm, --pins-from's file is also the VLM stage's --record:
# its members are the shapes that pass, so the run gives that split back with no API call.
# A --from stage without --list reads the previous stage's pass.txt only when its `done` exists.
# Many machines: run the same command with --rank K --world N on N machines (K = 0..N-1) with
# one shared <dir>. Each machine runs the stages on its share of the shapes
# (<dir>/<stage>/rank<K>/), feeding its own pass.txt forward, and stops after the heuristic
# stage's signatures, because the duplicate decision needs every shape: once all are done,
# `--merge --to heuristic_filtering` decides it, and `--from glb2representations` continues
# (sliced again, or not). A sliced run from glb2representations on stops before the split;
# `--merge` then combines every stage's lists into <dir>/<stage>/ in input order and runs it.

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"
STAGES=(metadata_filtering heuristic_filtering glb2representations vlm_filtering data_split)
LIST="" FROM=metadata_filtering TO=data_split VLM=0 PIN=() PINS_FROM=""
args=()
while (($#)); do
  case $1 in
    --list) LIST=$(abspath "${2:?--list needs a file}"); shift 2 ;;
    --from) FROM=${2:?--from needs a stage}; shift 2 ;;
    --to) TO=${2:?--to needs a stage}; shift 2 ;;
    --vlm) VLM=1; shift ;;
    --pins-from|--pin-val|--pin-test) PIN+=("$1" "$(abspath "${2:?$1 needs a file}")")
      [[ $1 != --pins-from ]] || PINS_FROM=${PIN[-1]}; shift 2 ;;
    --in) die "run_all takes --list, not --in" ;;
    *) args+=("$1"); shift ;;
  esac
done
parse_common "${args[@]}"
((${#REST[@]} == 0)) || die "unknown argument: ${REST[0]} (-h for the usage)"
[[ -z $LIST ]] || ((DRY)) || [[ -f $LIST ]] || die "no such list: $LIST"
index() { local i; for i in "${!STAGES[@]}"; do [[ ${STAGES[i]} == "$1" ]] && { echo "$i"; return; }; done
          die "unknown stage $1 (one of ${STAGES[*]})"; }
FI=$(index "$FROM")       # one per line: a failed substitution then stops the script
TI=$(index "$TO")
((FI <= TI)) || die "--from $FROM comes after --to $TO"
# Every stage finds its GLBs through the GLB index; metadata_filtering/run.sh builds it at every
# run, so a run that starts later builds it here.
if ((FI > 0)); then py lightgen -m data_processing.metadata_filtering.glb_index; fi

common=(--out "$OUT")
[[ -z $WORKERS ]] || common+=(--workers "$WORKERS")
if ((DRY)); then common+=(--dry-run); fi
summary=()
stopped=""
for ((i = FI; i <= TI; i++)); do
  s=${STAGES[i]}
  prev=""; ((i > 0)) && prev=${STAGES[i - 1]}
  # the input: --list for the first stage that runs, else the previous stage's pass list.
  # Sliced, a later stage reads this machine's own, just made; the first stage reads it only
  # when that stage finished on this machine after the last merge, else the merged list.
  in=""
  if ((i == FI)) && [[ -n $LIST ]]; then in=$LIST
  elif [[ -n $prev ]]; then
    in=$OUT/$prev/pass.txt
    own=$OUT/$prev/rank${RANK:-}/pass.txt
    if [[ -n $RANK ]] && { ((i > FI)) || { [[ -f $(dirname "$own")/done && -f $own ]] && [[ ! -f $in || $own -nt $in ]]; }; }; then
      in=$own
    fi
    if ((i == FI && ! DRY)); then
      [[ -f $(dirname "$in")/done && -f $in ]] || die "$prev has not finished: no $(dirname "$in")/done (run it, or its merge, first)"
    fi
  fi
  flags=()
  [[ -z $in ]] || flags+=(--in "$in")
  if [[ $s == vlm_filtering ]] && ((VLM)); then flags+=(--vlm); fi
  if [[ $s == vlm_filtering && -n $PINS_FROM ]] && ! ((VLM || MERGE)); then flags+=(--record "$PINS_FROM"); fi
  if [[ $s == data_split ]]; then flags+=("${PIN[@]}"); fi
  if [[ $s == data_split && -n $RANK ]]; then
    echo "== data_split: one global step; run it after the merge (run_all.sh --merge --world $WORLD)"
    break
  fi
  echo "== $s"
  t0=$SECONDS
  if ((MERGE)) && [[ $s != data_split ]]; then
    bash "data_processing/$s/run.sh" merge "${flags[@]}" "${common[@]}" --world "$WORLD"
  elif ((MERGE)); then
    bash "data_processing/$s/run.sh" "${flags[@]}" "${common[@]}"
  else
    bash "data_processing/$s/run.sh" "${flags[@]}" "${common[@]}" ${RANK:+--rank "$RANK" --world "$WORLD"}
  fi
  dir=$OUT/$s; [[ -n $RANK ]] && dir=$dir/rank$RANK
  summary+=("$(printf '%-20s %8s %8s %7s' "$s" "$(count "$dir/input.txt")" "$(count "$dir/pass.txt")" "$((SECONDS - t0))")")
  if [[ $s == heuristic_filtering && -n $RANK ]]; then
    stopped="the duplicate decision needs every shape: once all ranks are done, the same command with --merge --world $WORLD (no --rank) --from $FROM --to heuristic_filtering, then --from glb2representations"
    break
  fi
done
echo
printf '%-20s %8s %8s %7s\n' stage in kept "wall s"
printf '%s\n' "${summary[@]}"
[[ -z $stopped ]] || echo "stopped: $stopped"
