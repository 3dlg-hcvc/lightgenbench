#!/usr/bin/env bash
# glb2representations: the bake (UV atlas, o-voxels) and the six multiview views of every shape,
# one load per shape, then the bake's verdict and the views' check.
#
#   bash data_processing/glb2representations/run.sh --in <uuid list> --out <dir> [--workers N]
#        [--rank K --world N] [--timeout_s S] [--recovery_timeout_s S] [--dry-run]
#   bash data_processing/glb2representations/run.sh merge --in <uuid list> --out <dir> --world N
#
# The expensive stage (GPU, ~2 GB RAM a worker): worth splitting over machines with
# --rank K --world N, all writing one shared <dir>, then `merge` once. Writes
# <dir>/glb2representations/:
#   tree/<uuid>/             atlas.npz, the .vxz voxels, render_tex/ and render_cond/ (the views);
#                            shared by the ranks; a rerun skips the complete shapes
#   tree/summary_<time>[-redo|-slow]_rank<K>.parquet   one per bake run, read by `run collect`
# and in <dir>/glb2representations/ (with --rank: .../rank<K>/): run.json, summary.parquet,
# bake_pass.txt, no_verdict.txt, unattempted.txt (`run collect`), views_complete.txt,
# views_missing.txt (`run collect --views`), and pass.txt = bake_pass.txt ∩ views_complete.txt.
# The tracked records out/bake_pass.txt and out/views_complete.txt are never written
# (`collect --keep_record`). After the bake's collect: shapes whose summary was lost
# (no_verdict.txt: a hard kill before the driver wrote it) are re-baked with --overwrite,
# then the recovery pass re-runs the timeouts and dead workers under --recovery_timeout_s
# (default 3,600 s; 0 skips it; it can take hours), each followed by a new collect; the
# stage fails rather than finish on a collect that still has shapes without a verdict. A
# rerun skips the complete shapes and retries those without a complete directory, timeouts
# included (the driver's resume). --workers: shapes in parallel, both passes (default 8);
# --timeout_s: the bake's per-shape cap (default 1,200 s, the views' minimum).
# Envs: trellis2 (bake), lightgen (collect).

source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"
parse_common "$@"
TIMEOUT=1200 RECOVERY=3600
set -- "${REST[@]}"
while (($#)); do
  case $1 in
    --timeout_s) TIMEOUT=${2:?--timeout_s needs seconds}; shift 2 ;;
    --recovery_timeout_s) RECOVERY=${2:?--recovery_timeout_s needs seconds}; shift 2 ;;
    *) die "unknown argument: $1 (-h for the usage)" ;;
  esac
done
[[ $TIMEOUT =~ ^[1-9][0-9]*$ && $RECOVERY =~ ^[0-9]+$ ]] || die "--timeout_s needs a positive number, --recovery_timeout_s a number"
[[ -n $IN ]] || die "--in <uuid list> is required (the heuristic stage's pass.txt)"
stage_setup glb2representations
WORKERS=${WORKERS:-8}
TREE=$STAGE_DIR/tree
LISTS=(bake_pass.txt no_verdict.txt unattempted.txt views_complete.txt views_missing.txt)

if ((MERGE)); then
  stage_merge "$IN" "${LISTS[@]}" pass.txt
  stage_done
  exit 0
fi

stage_input "$IN"
if stage_empty; then
  for f in "${LISTS[@]}"; do : > "$W/$f"; done
  stage_finish "$INPUT"; exit 0
fi

# Every invocation writes its own summary (a resume's `skip` rows carry no verdict, so they
# must not replace the verdict rows of the run before). The redo and
# recovery runs are named <RUN>-redo, <RUN>-slow: they sort after <RUN>, so collect takes
# their rows.
RUN=$(date +%Y%m%d-%H%M%S)
IF_ANY=""; if ((DRY)); then IF_ANY=" (only when there are any)"; fi
bake() {  # bake <uuid list> <run name> <per-shape cap> [run.py flags...]
  local list=$1 name=$2 cap=$3; shift 3
  py trellis2 -m data_processing.glb2representations.run --list "$list" --out_root "$TREE" \
    --outputs atlas,voxels,views --workers "$WORKERS" --timeout_s "$cap" \
    --summary_name "summary_${name}_rank${RANK:-0}.parquet" "$@"
}
collect() {
  py lightgen -m data_processing.glb2representations.run collect --run_dir "$W" --out_root "$TREE" --keep_record
}
echo "+ printf '{\"input\": \"$INPUT\"}' > $W/run.json"
((DRY)) || printf '{"input": "%s"}\n' "$INPUT" > "$W/run.json"
bake "$INPUT" "$RUN" "$TIMEOUT"
collect
if ((DRY)) || [[ -s $W/no_verdict.txt ]]; then
  log "re-bake the complete shapes whose summary was lost (no_verdict.txt)$IF_ANY"
  bake "$W/no_verdict.txt" "$RUN-redo" "$TIMEOUT" --overwrite
  collect
fi
if ((RECOVERY > 0)); then
  # os._exit: pyarrow's threads can abort the interpreter's teardown under load
  run_to "$W/recovery.txt" "$CONDA" run --no-capture-output -n lightgen python -u -c '
import os, sys, pandas as pd
d = pd.read_parquet(sys.argv[1])
sys.stdout.write("".join(f"{u}\n" for u in d["sha"][d["stage"].isin(["timeout", "worker_died"])]))
sys.stdout.flush(); os._exit(0)' "$W/summary.parquet"
  if ((DRY)) || [[ -s $W/recovery.txt ]]; then
    log "recovery pass: $(count "$W/recovery.txt") timeouts and dead workers at $RECOVERY s a shape$IF_ANY"
    bake "$W/recovery.txt" "$RUN-slow" "$RECOVERY"
    collect
  fi
fi
((DRY)) || [[ ! -s $W/no_verdict.txt && ! -s $W/unattempted.txt ]] \
  || die "partial collect: $(count "$W/no_verdict.txt") shapes without a verdict, $(count "$W/unattempted.txt") unattempted ($W): rerun"
if ((DRY)) || [[ -s $W/bake_pass.txt ]]; then
  py lightgen -m data_processing.glb2representations.run collect --views --input "$W/bake_pass.txt" \
    --out_root "$TREE" --out "$W"
else
  log "no shape passed the bake: no views to check"
  : > "$W/views_complete.txt"; : > "$W/views_missing.txt"
fi
stage_finish "$W/views_complete.txt"
