#!/usr/bin/env bash
# heuristic filtering: the non-zero emission check, the albedo-copy check, the duplicate check
# on the GLB alone.
#
#   bash data_processing/heuristic_filtering/run.sh --in <uuid list> --out <dir> [--workers N]
#        [--rank K --world N] [--dry-run]
#   bash data_processing/heuristic_filtering/run.sh merge --in <uuid list> --out <dir> --world N
#
# Runs the two checks, then duplicate_check.py signatures (every GLB read whole), keys (the
# shapes that share a geometry hash with another, read again) and list. The duplicate decision
# is global: it finds only the copies inside the input list, so the released list needs the
# full set. With --rank/--world each machine runs the two checks and the signatures (the
# expensive part) on its share and stops; `merge` then joins the ranks' lists and signatures
# and runs keys and list once.
# Writes <dir>/heuristic_filtering/ (with --rank: .../rank<K>/): nonzero_emission_check.csv,
# albedo_copy_check.csv, signatures.csv (the resume markers), nonzero_emission_pass.txt,
# albedo_copy_pass.txt; without --rank (or in `merge`) also duplicate_keys.csv,
# duplicate_groups.csv, duplicate_pass.txt and pass.txt. --workers: threads per check
# (default 24). Env: lightgen.

source "$(dirname "${BASH_SOURCE[0]}")/../utils.sh"
parse_common "$@"
((${#REST[@]} == 0)) || die "unknown argument: ${REST[0]} (-h for the usage)"
[[ -n $IN ]] || die "--in <uuid list> is required (the metadata stage's pass.txt)"
stage_setup heuristic_filtering
WORKERS=${WORKERS:-24}
H=data_processing.heuristic_filtering

duplicates() {  # keys and list over W's albedo-copy pass list and signatures
  if ! ((DRY)) && [[ ! -s $W/albedo_copy_pass.txt ]]; then : > "$W/duplicate_pass.txt"; return; fi
  py lightgen -m $H.duplicate_check keys --list "$W/albedo_copy_pass.txt" \
    --signatures "$W/signatures.csv" --out "$W/duplicate_keys.csv" --workers "$WORKERS"
  py lightgen -m $H.duplicate_check list --list "$W/albedo_copy_pass.txt" \
    --signatures "$W/signatures.csv" --keys "$W/duplicate_keys.csv" \
    --out "$W/duplicate_pass.txt" --groups "$W/duplicate_groups.csv"
}

if ((MERGE)); then
  stage_merge "$IN" nonzero_emission_pass.txt albedo_copy_pass.txt
  # each rank's rows of its own albedo-copy passes only: a resumed signatures.csv can hold rows
  # of shapes outside the rank's current share, and a shape listed twice is its own copy
  echo "+ concatenate the rows of rank{0..$((WORLD - 1))}/albedo_copy_pass.txt in rank{0..$((WORLD - 1))}/signatures.csv > $STAGE_DIR/signatures.csv"
  if ! ((DRY)); then
    header=1
    for ((k = 0; k < WORLD; k++)); do     # a rank with nothing to sign wrote none
      f=$STAGE_DIR/rank$k/signatures.csv
      if [[ ! -s $f ]]; then
        [[ ! -s $STAGE_DIR/rank$k/albedo_copy_pass.txt ]] \
          || die "rank$k has albedo-copy passes but no signatures.csv: rerun it"
        continue
      fi
      if ((header)); then head -n 1 "$f"; header=0; fi
      awk -F, 'FILENAME == ARGV[1] { if (!/^#/ && NF) k[$1]; next } FNR > 1 && ($1 in k)' \
        "$STAGE_DIR/rank$k/albedo_copy_pass.txt" "$f"
    done > "$STAGE_DIR/signatures.csv"
    repeated=$(awk -F, 'FNR > 1 { print $1 }' "$STAGE_DIR/signatures.csv" | sort | uniq -d | wc -l)
    ((repeated == 0)) || die "$repeated shapes have a row in more than one rank's signatures.csv: rerun the ranks with one --world"
    unsigned=$(awk -F, 'FILENAME == ARGV[1] { if (FNR > 1) s[$1]; next } !/^#/ && NF && !($1 in s)' \
      "$STAGE_DIR/signatures.csv" "$STAGE_DIR/albedo_copy_pass.txt" | wc -l)
    ((unsigned == 0)) || die "$unsigned albedo-copy passes have no row in the ranks' signatures.csv: rerun those ranks"
  fi
  duplicates
  stage_finish "$W/duplicate_pass.txt"
  exit 0
fi

# no duplicate list until this run writes one
if ! ((DRY)); then rm -f "$W/duplicate_pass.txt"; fi
stage_input "$IN"
if stage_empty; then
  for f in nonzero_emission_pass albedo_copy_pass duplicate_pass; do : > "$W/$f.txt"; done
  stage_finish "$INPUT"; exit 0
fi

py lightgen -m $H.nonzero_emission_check --list "$INPUT" --out "$W/nonzero_emission_check.csv" \
  --pass_list "$W/nonzero_emission_pass.txt" --workers "$WORKERS"
py lightgen -m $H.albedo_copy_check --list "$W/nonzero_emission_pass.txt" \
  --out "$W/albedo_copy_check.csv" --pass_list "$W/albedo_copy_pass.txt" --workers "$WORKERS"
py lightgen -m $H.duplicate_check signatures --list "$W/albedo_copy_pass.txt" \
  --out "$W/signatures.csv" --workers "$WORKERS"
if [[ -n $RANK ]]; then
  stage_done
  log "rank $RANK done; the duplicate decision needs every rank: run \`merge --world $WORLD\`"
  exit 0
fi
log "duplicates over the $(count "$W/albedo_copy_pass.txt") albedo-copy passes: only the copies inside this list are found"
duplicates
stage_finish "$W/duplicate_pass.txt"
