# shellcheck shell=bash
# What the stage scripts (<stage>/run.sh) share; sourced, never run. The common flags, the
# dry-run printer, the conda call, a machine's slice of a uuid list, and the merge of the
# machines' lists.

set -euo pipefail
# bash 4.4+: older bash treats an empty "${array[@]}" as unset under set -u
((BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4))) \
  || { echo "error: needs bash 4.4 or newer (this is $BASH_VERSION)" >&2; exit 2; }
CALLER_PWD=$PWD
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
CONDA=${CONDA_EXE:-conda}

IN="" OUT="" WORKERS="" RANK="" WORLD="" DRY=0 MERGE=0
REST=()

usage() {  # the calling script's header comment
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
}
die() { echo "error: $*" >&2; exit 2; }
abspath() { case $1 in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$CALLER_PWD/$1" ;; esac; }
log() { printf '[%s] %s\n' "${STAGE:-run_all}" "$*"; }

# Print a command (quoting only the words that need it), then run it unless --dry-run.
show() {
  local w out="+"
  for w in "$@"; do
    if [[ $w =~ ^[A-Za-z0-9_./,:=+@%-]+$ ]]; then out+=" $w"; else out+=" $(printf '%q' "$w")"; fi
  done
  printf '%s' "$out"
}
run() {
  show "$@"; printf '\n'
  ((DRY)) || "$@"
}
run_to() {  # run_to <file> <command...>: the command's output into <file>
  local file=$1; shift
  show "$@"; printf ' > %s\n' "$file"
  ((DRY)) || "$@" > "$file"
}
need_module() {  # a `python -m` module must exist, checked even under --dry-run (catches a rename)
  local f=$REPO/${1//.//}
  [[ -f $f.py || -f $f/__main__.py ]] || die "no module $1 (no $f.py)"
}
py() {  # py <conda env> <python arguments...>
  local env=$1; shift
  [[ ${1-} != -m ]] || need_module "$2"
  run "$CONDA" run --no-capture-output -n "$env" python -u "$@"
}

# --- uuid lists ---------------------------------------------------------------------
uuids() { awk '!/^#/ && NF { print $1 }' "$@"; }    # comments and blank lines dropped
count() { if [[ -f $1 ]]; then uuids "$1" | wc -l; else echo "?"; fi; }
keep_in_order() {  # keep_in_order <list> <keep>: the uuids of <list> that <keep> holds, in <list>'s order
  awk 'FILENAME == ARGV[1] { if (!/^#/ && NF) k[$1] = 1; next }
       !/^#/ && NF && ($1 in k) { print $1 }' "$2" "$1"
}
# One machine's share: the uuids whose hash (of the uuid's characters, mod 1000003) is K mod N.
# A function of the uuid alone, so a later stage's slice of its (smaller) input is exactly the
# same machine's shapes, and a rank can run every stage on its own pass list. LC_ALL=C: every
# awk counts bytes, so all machines agree.
slice() {  # slice <K> <N> <list>
  LC_ALL=C awk -v k="$1" -v n="$2" '
    BEGIN { a = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_." }
    /^#/ || !NF { next }
    { h = 0
      for (i = 1; i <= length($1); i++) h = (h * 31 + index(a, substr($1, i, 1))) % 1000003
      if (h % n == k) print $1 }' "$3"
}

# --- the common flags ----------------------------------------------------------------
parse_common() {
  while (($#)); do
    case $1 in
      -h|--help) usage; exit 0 ;;
      merge|--merge) MERGE=1; shift ;;
      --in) IN=$(abspath "${2:?--in needs a file}"); shift 2 ;;
      --out) OUT=$(abspath "${2:?--out needs a directory}"); shift 2 ;;
      --workers) WORKERS=${2:?--workers needs a number}; shift 2 ;;
      --rank) RANK=${2:?--rank needs a number}; shift 2 ;;
      --world) WORLD=${2:?--world needs a number}; shift 2 ;;
      --dry-run) DRY=1; shift ;;
      *) REST+=("$1"); shift ;;
    esac
  done
  [[ -n $OUT ]] || die "--out <dir> is required (-h for the usage)"
  [[ -z $WORKERS || $WORKERS =~ ^[1-9][0-9]*$ ]] || die "--workers needs a positive number"
  if ((MERGE)); then
    [[ -z $RANK && $WORLD =~ ^[1-9][0-9]*$ ]] || die "merge needs --world N and no --rank"
  elif [[ -n $RANK$WORLD ]]; then
    [[ $RANK =~ ^[0-9]+$ && $WORLD =~ ^[1-9][0-9]*$ ]] && ((RANK < WORLD)) \
      || die "--rank K --world N needs 0 <= K < N"
  fi
  if [[ -n $IN ]] && ! ((DRY)); then [[ -f $IN ]] || die "no such list: $IN"; fi
  return 0
}

# --- a stage --------------------------------------------------------------------------
# STAGE_DIR = <out>/<stage>; W = this machine's folder (STAGE_DIR, or STAGE_DIR/rank<K>);
# INPUT = W/input.txt, the uuids this machine processes. The old result goes at the start, so
# a pass.txt only exists once its stage has finished (`done` is written right after it).
stage_setup() {  # stage_setup <stage>
  STAGE=$1 STAGE_DIR=$OUT/$1 W=$OUT/$1
  [[ -n $RANK ]] && W=$STAGE_DIR/rank$RANK
  INPUT=$W/input.txt T0=$SECONDS
  if ! ((DRY)); then mkdir -p "$W"; rm -f "$W/done" "$W/pass.txt"; fi
}
stage_input() {  # stage_input <uuid list>: this machine's input -> INPUT
  local src=$1
  if [[ -n $RANK ]]; then
    echo "+ slice rank $RANK of $WORLD of $src > $INPUT"
    ((DRY)) && return
    slice "$RANK" "$WORLD" "$src" > "$INPUT.tmp"
    echo "rank $RANK world $WORLD input $src" > "$W/slice.txt"
  else
    echo "+ uuids $src > $INPUT"
    ((DRY)) && return
    uuids "$src" > "$INPUT.tmp"
  fi
  mv "$INPUT.tmp" "$INPUT"
  log "input: $(count "$INPUT") uuids"
}
stage_empty() { ! ((DRY)) && [[ ! -s $INPUT ]]; }
stage_done() { ((DRY)) || date -u +%FT%TZ > "$W/done"; }    # written last: merge checks it
stage_finish() {  # stage_finish <kept list>: W/pass.txt (input order), the done marker, the tally
  if ((DRY)); then echo "+ keep_in_order $INPUT $1 > $W/pass.txt"; return; fi
  keep_in_order "$INPUT" "$1" > "$W/pass.txt"
  stage_done
  log "in $(count "$INPUT"), kept $(count "$W/pass.txt"), $((SECONDS - T0)) s -> $W/pass.txt"
}

# merge: every rank's lists -> STAGE_DIR, in the order of the stage's input <order list>.
# Refuses while a rank has no `done` marker, and when the ranks' inputs are not exactly the
# stage's input (another list, another --world).
stage_merge() {  # stage_merge <order list> <list names...>
  local order=$1 k name missing=()
  shift
  W=$STAGE_DIR INPUT=$STAGE_DIR/input.txt
  for ((k = 0; k < WORLD; k++)); do
    ((DRY)) || [[ -f $STAGE_DIR/rank$k/done ]] || missing+=("rank$k")
  done
  ((${#missing[@]} == 0)) || die "$STAGE: not done yet: ${missing[*]} (a machine writes $STAGE_DIR/rank<K>/done last)"
  echo "+ merge $STAGE_DIR/rank{0..$((WORLD - 1))}/{$*} in the order of $order"
  ((DRY)) && return
  mkdir -p "$STAGE_DIR"; rm -f "$STAGE_DIR/done" "$STAGE_DIR/pass.txt"
  uuids "$order" > "$INPUT.tmp"; mv "$INPUT.tmp" "$INPUT"     # <order> may be INPUT itself
  local n_in n_ranks n_cover
  for ((k = 0; k < WORLD; k++)); do uuids "$STAGE_DIR/rank$k/input.txt"; done | sort -u > "$STAGE_DIR/input.ranks"
  n_in=$(count "$INPUT")
  n_ranks=$(count "$STAGE_DIR/input.ranks")
  n_cover=$(keep_in_order "$INPUT" "$STAGE_DIR/input.ranks" | wc -l)
  rm -f "$STAGE_DIR/input.ranks"
  [[ $n_ranks == "$n_in" && $n_cover == "$n_in" ]] \
    || die "$STAGE: the $WORLD ranks' inputs ($n_ranks uuids) are not the stage's input $order ($n_in)"
  for name in "$@"; do
    for ((k = 0; k < WORLD; k++)); do uuids "$STAGE_DIR/rank$k/$name"; done > "$STAGE_DIR/$name.ranks"
    keep_in_order "$INPUT" "$STAGE_DIR/$name.ranks" > "$STAGE_DIR/$name"
    rm -f "$STAGE_DIR/$name.ranks"
    log "merged $name: $(count "$STAGE_DIR/$name") of $n_in"
  done
}
