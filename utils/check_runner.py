"""The CLI harness the heuristic filtering checks share: uuid list in, resumable CSV out.

A check module supplies `check_one(sha, metadata, roots) -> dict` (never raises; read
failures land in `status`), its CSV `fields`, its `reject_reasons`, and its default
paths; `run_check` does the rest:

  * `--list` uuid file (default: the check's input), `--out` CSV, `--metadata`,
    `--roots`, `--workers`, `--limit`, `--pass_list`;
  * resume (`resume_done`): rows already in the CSV are skipped, except a failed one (a
    missing or unreadable file), which is done again; a partial last line (a killed run) is
    dropped; an empty existing file gets a header;
  * rows are written as they complete (a slow NFS read must not hold back the rest)
    from a bounded submission window;
  * the summary and the pass list cover the INPUT list only, since the CSV may hold
    rows from a larger earlier run.

The write loop alone, `write_as_completed`, is shared with the metadata scan
(`data_processing/metadata_filtering/check_emission.py scan`).
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Sequence

import pandas as pd

from utils.metadata_io import write_uuids
from utils.paths import TEXVERSE_1K_ROOTS, TEXVERSE_METADATA_PARQUET

#: Metadata parquet with `glb_1k_path` (index = shape id) and the GLB mounts.
DEFAULT_METADATA = TEXVERSE_METADATA_PARQUET
DEFAULT_GLB_ROOTS = TEXVERSE_1K_ROOTS

# --- run parameters ---------------------------------------------------------------
DEFAULT_WORKERS = 24
PROGRESS_EVERY = 1_000


def write_as_completed(job: Callable, todo: Sequence, writer, fh, workers: int) -> int:
    """Run `job` over `todo` on `workers` threads and write each returned row with `writer`
    as it completes; returns the number of rows written.

    Rows go out as they complete, not in submission order: with an ordered map one slow NFS
    read at the head of the queue holds back every finished row behind it. Row order is
    irrelevant, since the CSV is keyed by uuid and resume is a set. Jobs are submitted in
    bounded windows so a long list does not create all its futures up front. `fh` is flushed with every progress line, so a
    killed run loses little and `wc -l` on the CSV is a live gauge.
    """
    t0, written = time.time(), 0
    window = max(4 * workers, 64)
    with ThreadPoolExecutor(workers) as ex:
        pending, it = set(), iter(todo)
        for s in it:
            pending.add(ex.submit(job, s))
            if len(pending) >= window:
                break
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                writer.writerow(fut.result())
                written += 1
                if written % PROGRESS_EVERY == 0:
                    fh.flush()
                    print(f"{written}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
            for s in it:
                pending.add(ex.submit(job, s))
                if len(pending) >= window:
                    break
    fh.flush()
    return written


def resume_done(path, fields: Sequence[str], retry: Sequence[str] = (), header_error: str = "") -> tuple:
    """(done, dropped) for resuming the CSV at `path`: the uuids that already have a complete
    row, and how many rows were dropped.

    A partial last line (a killed run) is dropped, and so is each row of a uuid in `retry`
    whose `status` is not `ok` (a missing file or a read error: the file may be readable
    now) or that lacks a field, so those shapes are done again; of two rows for one uuid only
    the last stays. The
    file is rewritten, beside it and renamed, only when something goes. With `header_error`,
    a CSV whose columns are not `fields` stops the run with that message before anything is
    read further or rewritten.
    """
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return set(), 0
    with open(path, newline="") as f:
        text = f.read()
    reader = csv.DictReader(io.StringIO(text))
    if header_error and reader.fieldnames != list(fields):
        raise SystemExit(header_error)
    cut = not text.endswith("\n")
    if cut:                                  # interrupted mid-row: drop the fragment
        reader = csv.DictReader(io.StringIO(text[: text.rfind("\n") + 1]))
    rows = list(reader)
    retry = set(retry)
    keep = {}
    for r in rows:
        complete = all(r.get(k) is not None for k in fields)
        if not (r["uuid"] in retry and (r.get("status") != "ok" or not complete)):
            keep.pop(r["uuid"], None)        # the last row of a uuid stays, in file order
            keep[r["uuid"]] = r
    dropped = len(rows) - len(keep)
    if cut or dropped:
        tmp = f"{path}.tmp"
        with open(tmp, "w", newline="") as f:
            if reader.fieldnames:
                w = csv.DictWriter(f, fieldnames=reader.fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(keep.values())
        os.replace(tmp, path)
    done = {u for u, r in keep.items() if all(r.get(k) is not None for k in fields)}
    return done, dropped


def summarize(rows, reject_reasons: Sequence[str]) -> str:
    n = len(rows)
    status = collections.Counter(r["status"] for r in rows)
    lines = [f"shapes: {n}", "status: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items()))]
    if reject_reasons:      # a pass that only fingerprints (duplicate_check signatures) has no verdict
        ok = [r for r in rows if r["status"] == "ok"]
        reasons = collections.Counter(r.get("reject", "") for r in ok)
        lines.append(f"pass: {reasons.get('', 0)}")
        for r in reject_reasons:
            lines.append(f"reject {r}: {reasons.get(r, 0)}")
    return "\n".join(lines)


def run_check(check_one: Callable, fields: Sequence[str], reject_reasons: Sequence[str],
              default_list, default_out, default_pass_list, description: str = "") -> None:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default=str(default_list), help="uuid list, one per line (default: the check's input)")
    ap.add_argument("--out", default=str(default_out), help="output CSV; also the resume marker")
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA), help="parquet with glb_1k_path")
    ap.add_argument("--roots", nargs="*", default=[str(r) for r in DEFAULT_GLB_ROOTS], help="GLB mounts to search")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, help="check at most N shapes (smoke test)")
    ap.add_argument("--pass_list", default=str(default_pass_list), help="write the passing uuids here, one per line (derived from the CSV after the run)")
    a = ap.parse_args()

    meta = pd.read_parquet(a.metadata, columns=["glb_1k_path"])
    shas = [ln.strip() for ln in open(a.list) if ln.strip()]

    done, dropped = resume_done(a.out, fields, shas)
    if done or dropped:
        print(f"resuming: {len(done)} complete rows already in {a.out}"
              f"{f'; {dropped} failed, incomplete or repeated rows dropped, those shapes are checked again' if dropped else ''}",
              flush=True)
    todo = [s for s in dict.fromkeys(shas) if s not in done]    # a uuid listed twice runs once
    if a.limit:
        todo = todo[: a.limit]
    print(f"shapes: {len(shas)} total, {len(todo)} to check, {a.workers} workers", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    t0 = time.time()
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        if new_file:
            w.writeheader()
        written = write_as_completed(lambda s: check_one(s, meta, a.roots), todo, w, fh, a.workers)
    print(f"wrote {written} rows in {time.time() - t0:.0f}s", flush=True)

    wanted = set(shas)
    with open(a.out) as f:
        rows = [r for r in csv.DictReader(f) if r["uuid"] in wanted]
    text = summarize(rows, reject_reasons)
    print(text)
    with open(a.out + ".summary.txt", "w") as f:
        f.write(text + "\n")
    if a.pass_list:
        passing = [r["uuid"] for r in rows if r["status"] == "ok" and not r.get("reject", "")]
        write_uuids(a.pass_list, passing)
        print(f"pass list: {len(passing)} uuids -> {a.pass_list}", flush=True)
