"""The stage's one command: every shape of a list becomes its atlas, its voxels and,
when asked, its six views, each shape in a fresh process.

    python -m data_processing.glb2representations.run --list <uuids> [--outputs atlas,voxels,views] ...
    python -m data_processing.glb2representations.run collect --run_dir out/runs/<RUN> [--out_root <tree>]
    python -m data_processing.glb2representations.run collect --views --out <dir> [--input <uuids>] [--out_root <tree>]

The driver (`run_main`) bounds parallelism, enforces wall-clock deadlines, cleans partial
staging output, and continues after a timeout or native crash. Every shape runs in a fresh
`run worker` subprocess (`process_one`): one load and normalize of the GLB, then the
requested outputs from it, written to a driver-owned staging directory that the driver
promotes only after the worker exits with a valid completion record. `collect` turns a
run's shard summaries into the stage's result list; `collect --views` lists the shapes
whose six views are complete on disk.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_processing.glb2representations.multiview import CAMERAS, VIEW_MAPS
from data_processing.glb2representations.utils import FP_GATE
from utils.glb_io import find_glb, resolve_glb
from utils.metadata_io import read_uuids, write_uuids
from utils.paths import (
    ALBEDO_COPY_PASS_LIST,
    REPRESENTATIONS_ROOT,
    TEXVERSE_1K_ROOTS,
    TEXVERSE_METADATA_PARQUET,
)

HERE = Path(__file__).resolve().parent

# --- inputs -----------------------------------------------------------------------
#: The heuristic stage's result: the shapes that declare emission, write albedo,
#: metallic and roughness, whose emission can show, and whose emission is not a copy of
#: the base color. One uuid per line. The bake's input.
INPUT_LIST = ALBEDO_COPY_PASS_LIST
#: Metadata parquet with `glb_1k_path` for every downloaded GLB (index = uuid).
DEFAULT_METADATA = TEXVERSE_METADATA_PARQUET
DEFAULT_GLB_ROOTS = TEXVERSE_1K_ROOTS

# --- outputs ----------------------------------------------------------------------
#: The dataset tree: one directory per shape, holding every representation this stage
#: writes. The default `--out_root` of `collect`.
DEFAULT_OUT_ROOT = REPRESENTATIONS_ROOT

#: Local, gitignored except the result lists and one run's bookkeeping (see `.gitignore`).
#: Everything this stage writes locally goes in one of the named subfolders below.
OUT_DIR = HERE / "out"
#: Run records, one directory per run: the `--run_dir` of `collect`.
RUNS_DIR = OUT_DIR / "runs"
#: The stage's result: uuids whose bake passed the gate, one per line (written by
#: `collect`).
DEFAULT_PASS_LIST = OUT_DIR / "bake_pass.txt"

# --- fixtures ---------------------------------------------------------------------
#: Bake-mechanics regression shapes: simple (single textured mesh), multipart (two
#: meshes that must stay aligned), emissive (saturated text). The first two are
#: albedo-copy rejects and are NOT in INPUT_LIST; they are baked with `--shas`.
FIXTURE_SHAS = (
    "00001dbc99db48efb16f81945dcc9999",
    "b613063b55db49e7b2d968d2cc43b254",
    "4159a2a7539d4db1b9146455a60c1887",
)

# --- run parameters ---------------------------------------------------------------
#: Per-shape wall-clock cap, fresh-interpreter and CUDA start-up included.
DEFAULT_TIMEOUT_S = 600.0
#: A `--outputs ...,views` run pays six 512x512 renders per shape on top of a bake that
#: already lost heavy meshes at the default cap, so the driver warns below this and the
#: node wants fewer workers than a bake-only run.
VIEWS_TIMEOUT_HINT_S = 1200.0


# --- the output set --------------------------------------------------------
#: What one worker can produce for a shape, in the order the stages run. `atlas` and
#: `voxels` are the bake; `views` is the six-camera multiview render, shaded from the
#: atlas (`multiview.py`).
OUTPUTS = ("atlas", "voxels", "views")
#: The bake alone: the default of `--outputs`, so `bake_pass.txt` keeps its
#: meaning.
DEFAULT_OUTPUTS = ("atlas", "voxels")


def parse_outputs(text):
    """`"views,atlas"` -> `("atlas", "views")`: the requested outputs in OUTPUTS order.

    Spacing and request order do not matter; an empty or unknown name is a ValueError,
    so a typo stops the run instead of quietly baking less than asked.
    """
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    if not names:
        raise ValueError(f"--outputs is empty: pick from {','.join(OUTPUTS)}")
    unknown = [name for name in names if name not in OUTPUTS]
    if unknown:
        raise ValueError(
            f"unknown output(s) {','.join(unknown)}: pick from {','.join(OUTPUTS)}"
        )
    return tuple(name for name in OUTPUTS if name in names)


# --- the driver -----------------------------------------------------------------

_POLL_S = 0.1
_TERMINATE_GRACE_S = 2.0
_STDERR_LIMIT = 1500
_WORK_DIR_NAME = ".uv_voxel_work"


def select_shas(list_path=None, shas=None, rank=0, world_size=1):
    """Shapes to process.

    `shas` (a comma-joined list, for smokes) wins; otherwise the newline-delimited
    `list_path`, default `INPUT_LIST` (the heuristic stage's pass list). The
    `rank`/`world_size` slice then applies to whichever selection is active. `--shas`
    is not filtered by the list: any sha with a resolvable GLB bakes.
    """
    if shas:
        selected = [s for s in shas.split(",") if s]
    else:
        path = str(list_path) if list_path else str(INPUT_LIST)
        with open(path) as handle:
            selected = [line.strip() for line in handle if line.strip()]
    n = len(selected)
    return selected[n * rank // world_size: n * (rank + 1) // world_size]


def _fail(sha, stage, out_dir, e, wall_s=None, timings=None):
    rec = {
        "sha": sha,
        "fp_voxel": 0.0,
        "fp_atlas": 0.0,
        "passed": False,
        "error": f"{stage} failed: {e}\n{traceback.format_exc()[-1500:]}",
        "stage": stage,
        "out_dir": out_dir,
    }
    if wall_s is not None:
        rec["wall_s"] = wall_s
    if timings is not None:
        rec["timings"] = timings
    return rec


def _abort(sha, stage, out_dir, msg, wall_s):
    """Record a shape killed by the parent (timeout / dead worker)."""
    return {
        "sha": sha,
        "fp_voxel": 0.0,
        "fp_atlas": 0.0,
        "passed": False,
        "error": msg,
        "stage": stage,
        "out_dir": out_dir,
        "wall_s": wall_s,
    }


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _physical_cores():
    """Physical core count (xatlas/OpenMP scale with physical, not logical), capped by
    this process's CPU affinity.

    Both psutil and `os.cpu_count()` report the whole machine, so under a Slurm
    `--cpus-per-task` cgroup they hand every worker several times the threads the job
    owns, and xatlas starves into timeouts. The affinity mask is what the job holds.
    """
    n = None
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
    except Exception:
        pass
    if not n:
        n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except AttributeError:   # not Linux
        pass
    return max(1, n)


def _pin_threads(n):
    """Cap this process's intra-op parallelism to `n` threads.

    xatlas (OpenMP) and torch each default to about one thread per core, so P unpinned
    workers oversubscribe to P*cores threads, which slows the run and causes timeouts.
    The env vars must be set before numpy/BLAS/OpenMP first load; the parent sets them
    for each spawned worker. torch reads its thread count at runtime, so the worker pins
    it explicitly too.
    """
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(n)


def _child_env(threads):
    """Return an environment that pins native-library threads before imports."""
    env = os.environ.copy()
    for var in _THREAD_ENV_VARS:
        env[var] = str(threads)
    env["UV_VOXEL_THREADS"] = str(threads)
    return env


def artifact_paths(out_dir, sha, outputs=DEFAULT_OUTPUTS, resolution=256):
    """The load-bearing files the requested `outputs` leave in `out_dir`.

    The one definition of the set: the driver promotes a staging dir only when the
    requested files exist, `shape_complete` resumes on the same list, and `collect`
    decides a baked shape is complete from the same list with the default set.
    """
    paths = []
    if "atlas" in outputs:
        paths.append(os.path.join(out_dir, "atlas.npz"))
    if "voxels" in outputs:
        paths += [
            os.path.join(out_dir, f"{sha}.coords.npz"),
            os.path.join(out_dir, f"emission_voxels_{resolution}", f"{sha}.vxz"),
            os.path.join(out_dir, f"pbr_voxels_{resolution}", f"{sha}.vxz"),
        ]
    if "views" in outputs:
        render_tex = os.path.join(out_dir, "render_tex")
        paths += [
            os.path.join(render_tex, f"{cam['file_stem']}_{name}.png")
            for cam in CAMERAS
            for name in VIEW_MAPS
        ]
        paths.append(os.path.join(render_tex, "transforms.json"))
        paths.append(os.path.join(out_dir, "render_cond", "000.png"))
    return paths


def _shape_dir_complete(out_dir, sha, outputs=DEFAULT_OUTPUTS, resolution=256):
    return all(os.path.isfile(p) for p in artifact_paths(out_dir, sha, outputs, resolution))


def _seed_staging(staging_dir, final_dir, sha, outputs, resolution=256):
    """Hardlink what the shape already has into its fresh staging directory, minus
    what this run writes. Returns the number of seeded files.

    `_promote_staging` swaps whole directories, so without this a run that produces
    only some of the outputs would promote those alone and delete the rest. A link
    costs no bytes and the worker never rewrites a file in place (every writer opens a
    fresh path), so a seeded artifact reaches the promoted directory unchanged.

    Files this run writes are skipped, and so are the whole view directories when views
    are requested: opening a hardlinked path for writing would truncate the shared
    inode, which is the shape's own artifact.
    """
    if not os.path.isdir(final_dir):
        return 0
    produced = set(artifact_paths(staging_dir, sha, outputs, resolution))
    skip_dirs = {"render_tex", "render_cond"} if "views" in outputs else set()
    seeded = 0
    for root, dirs, files in os.walk(final_dir):
        relative_root = os.path.relpath(root, final_dir)
        if relative_root == ".":
            relative_root = ""
            dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            destination = os.path.join(staging_dir, relative_root, name)
            if destination in produced:
                continue
            source = os.path.join(root, name)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                # a different device (or a filesystem without links): copy instead
                shutil.copy2(source, destination)
            seeded += 1
    return seeded


def _promote_staging(staging_dir, final_dir):
    """Replace final_dir only after staging contains a complete shape."""
    backup = f"{final_dir}.previous.{uuid.uuid4().hex}"
    had_final = os.path.exists(final_dir)
    if had_final:
        os.replace(final_dir, backup)
    try:
        os.replace(staging_dir, final_dir)
    except BaseException:
        if had_final and os.path.exists(backup) and not os.path.exists(final_dir):
            os.replace(backup, final_dir)
        raise
    if had_final:
        shutil.rmtree(backup, ignore_errors=True)


@dataclass
class ShapeJob:
    sha: str
    glb_path: str
    out_root: str
    final_dir: str
    staging_dir: str
    result_path: str
    stderr_path: str
    process: subprocess.Popen | None
    stderr_handle: object | None
    started: float


class ShapeProcessPool:
    """Bounded scheduler that launches a fresh subprocess for every shape."""

    def __init__(self, n_workers, timeout_s, threads=None, command_builder=None,
                 outputs=DEFAULT_OUTPUTS):
        self.n_workers = max(1, n_workers)
        self.timeout_s = timeout_s
        if threads is None:
            threads = max(1, _physical_cores() // self.n_workers)
        self.threads = threads
        self.command_builder = command_builder
        self.outputs = tuple(outputs)
        self.active = []

    def _default_command(self, job):
        return [
            sys.executable,
            "-m",
            "data_processing.glb2representations.run",
            "worker",
            "--sha",
            job.sha,
            "--glb_path",
            job.glb_path,
            "--staging_dir",
            job.staging_dir,
            "--final_out_dir",
            job.final_dir,
            "--result_path",
            job.result_path,
            "--outputs",
            ",".join(self.outputs),
        ]

    def _launch(self, sha, glb_path, out_root):
        work_root = os.path.join(out_root, _WORK_DIR_NAME)
        os.makedirs(work_root, exist_ok=True)
        token = f"{sha}.{os.getpid()}.{uuid.uuid4().hex}"
        final_dir = os.path.join(out_root, sha)
        staging_dir = os.path.join(work_root, f"{token}.staging")
        result_path = os.path.join(work_root, f"{token}.result.json")
        stderr_path = os.path.join(work_root, f"{token}.stderr.log")
        job = ShapeJob(
            sha=sha,
            glb_path=glb_path,
            out_root=out_root,
            final_dir=final_dir,
            staging_dir=staging_dir,
            result_path=result_path,
            stderr_path=stderr_path,
            process=None,
            stderr_handle=None,
            started=time.perf_counter(),
        )
        command = (
            self.command_builder(job)
            if self.command_builder is not None
            else self._default_command(job)
        )
        # seed before the worker starts: promotion swaps whole directories, so
        # staging must hold everything the shape keeps, not only what this run writes
        os.makedirs(staging_dir, exist_ok=True)
        try:
            _seed_staging(staging_dir, final_dir, sha, self.outputs)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        stderr_handle = open(stderr_path, "w+", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                env=_child_env(self.threads),
                stderr=stderr_handle,
                start_new_session=True,
            )
        except BaseException:
            stderr_handle.close()
            for path in (result_path, stderr_path):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        job.process = process
        job.stderr_handle = stderr_handle
        job.started = time.perf_counter()
        self.active.append(job)

    @staticmethod
    def _stop(job):
        if job.process.poll() is not None:
            return
        try:
            os.killpg(job.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            job.process.wait(timeout=_TERMINATE_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(job.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        job.process.wait()

    @staticmethod
    def _stderr(job):
        try:
            job.stderr_handle.flush()
            job.stderr_handle.seek(0)
            return job.stderr_handle.read()[-_STDERR_LIMIT:]
        except Exception:
            return ""

    @staticmethod
    def _cleanup(job, remove_staging=True):
        try:
            job.stderr_handle.close()
        except Exception:
            pass
        if remove_staging:
            shutil.rmtree(job.staging_dir, ignore_errors=True)
        for path in (job.result_path, job.stderr_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _promotion_gate(self):
        """The outputs staging must hold before a shape is promoted.

        A failed render never voids the bake (the views are extra files in the same
        directory), so views drop out of the gate -- unless they are all this run was
        asked for, in which case the worker reports a failed render as a stage failure
        and nothing is promoted at all.
        """
        gate = tuple(name for name in self.outputs if name != "views")
        return gate or ("views",)

    def _finished_record(self, job):
        wall = time.perf_counter() - job.started
        stderr = self._stderr(job)
        if job.process.returncode != 0:
            rec = _abort(
                job.sha,
                "worker_died",
                job.final_dir,
                f"worker exited with code {job.process.returncode}"
                + (f"\n{stderr}" if stderr else ""),
                wall,
            )
            self._cleanup(job)
            return rec
        try:
            with open(job.result_path, encoding="utf-8") as handle:
                rec = json.load(handle)
            if rec.get("sha") != job.sha:
                raise ValueError(
                    f"result sha {rec.get('sha')!r} does not match {job.sha!r}"
                )
        except Exception as error:
            rec = _abort(
                job.sha,
                "worker_died",
                job.final_dir,
                f"missing or malformed worker result: {error}"
                + (f"\n{stderr}" if stderr else ""),
                wall,
            )
            self._cleanup(job)
            return rec

        if rec.get("stage"):
            self._cleanup(job)
            return rec
        if not _shape_dir_complete(job.staging_dir, job.sha, self._promotion_gate()):
            rec = _abort(
                job.sha,
                "worker_died",
                job.final_dir,
                "worker reported completion without all required artifacts",
                wall,
            )
            self._cleanup(job)
            return rec
        try:
            _promote_staging(job.staging_dir, job.final_dir)
        except Exception as error:
            rec = _abort(
                job.sha,
                "promote",
                job.final_dir,
                f"failed to promote staging output: {error}",
                wall,
            )
            self._cleanup(job)
            return rec
        self._cleanup(job, remove_staging=False)
        return rec

    def _poll(self, job):
        returncode = job.process.poll()
        wall = time.perf_counter() - job.started
        if returncode is not None:
            return self._finished_record(job)
        if wall < self.timeout_s:
            return None
        self._stop(job)
        stderr = self._stderr(job)
        rec = _abort(
            job.sha,
            "timeout",
            job.final_dir,
            f"exceeded --timeout_s={self.timeout_s:g}s"
            + (f"\n{stderr}" if stderr else ""),
            wall,
        )
        self._cleanup(job)
        return rec

    def imap_unordered(self, jobs):
        """Yield records as fresh one-shape child processes finish."""
        pending = iter(jobs)
        exhausted = False
        while self.active or not exhausted:
            launch_failures = []
            while len(self.active) < self.n_workers and not exhausted:
                try:
                    sha, glb_path, out_root = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                try:
                    self._launch(sha, glb_path, out_root)
                except Exception as error:
                    launch_failures.append(
                        _abort(
                            sha,
                            "worker_died",
                            os.path.join(out_root, sha),
                            f"failed to launch worker: {error}",
                            0.0,
                        )
                    )
            for rec in launch_failures:
                yield rec

            progressed = bool(launch_failures)
            for job in list(self.active):
                rec = self._poll(job)
                if rec is not None:
                    self.active.remove(job)
                    progressed = True
                    yield rec
            if not progressed and (self.active or not exhausted):
                time.sleep(_POLL_S)

    def close(self):
        """Terminate all active children, used on driver interruption."""
        for job in list(self.active):
            self._stop(job)
            self._cleanup(job)
            self.active.remove(job)


def shape_complete(out_root, sha, resolution=256, outputs=DEFAULT_OUTPUTS):
    """True iff every artifact of the requested `outputs` for `sha` exists under out_root.

    Partial dirs (e.g. atlas.npz without .vxz after a kill) return False so a
    resume re-runs them. Matches the on-disk contract of atlas.py + voxels.py, and of
    multiview.py for the views. A shape baked without views is incomplete
    for a views run, so `--outputs views` picks it up instead of skipping it forever.
    """
    out_dir = os.path.join(out_root, sha)
    return _shape_dir_complete(out_dir, sha, outputs, resolution)


def _skip_rec(sha, out_root):
    return {
        "sha": sha,
        "fp_voxel": float("nan"),
        "fp_atlas": float("nan"),
        "passed": True,
        "stage": "skip",
        "out_dir": os.path.join(out_root, sha),
        "error": None,
    }


def run_main(argv=None):
    ap = argparse.ArgumentParser(
        description="UV+voxel generator with one fresh subprocess per shape",
        epilog="subcommands: `collect --run_dir <dir>` writes a run's verdict and "
               "`collect --views --out <dir>` the list of complete views (see `collect -h`); "
               "`worker` bakes one shape and is launched by the driver.",
    )
    ap.add_argument(
        "--list", "--shas_file", dest="list_path", default=None,
        help="newline-delimited uuid list (default: INPUT_LIST, the heuristic "
             "stage's pass list). --shas_file is the old spelling.",
    )
    ap.add_argument("--shas", help="comma-joined uuids; wins over --list; not filtered by it")
    ap.add_argument(
        "--metadata", default=str(DEFAULT_METADATA),
        help="parquet with glb_1k_path per uuid (default: the TexVerse metadata)",
    )
    ap.add_argument(
        "--glb_root", default=None,
        help="mount to join glb_1k_path against; without it, the GLB is searched under "
             "DEFAULT_GLB_ROOTS per shape",
    )
    ap.add_argument("--out_root", default=str(OUT_DIR / "bake"))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--timeout_s", type=float, default=DEFAULT_TIMEOUT_S,
        help="per-shape wall-clock cap (default %(default)gs). Timed-out shapes "
             "are dropped and recorded stage=timeout. 0 disables (no cap).",
    )
    ap.add_argument(
        "--workers", type=int, default=1,
        help="parallel shape workers (default 1). Shapes are independent CPU-bound "
             "jobs; N processes scale near-linearly to the physical core count.",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="re-process shapes that already have complete artifacts under "
             "out_root. Default is to skip them (resume-friendly).",
    )
    ap.add_argument(
        "--outputs", default=",".join(DEFAULT_OUTPUTS),
        help="comma-joined outputs this run produces, from atlas,voxels,views "
             "(default %(default)s, the bake). A run produces exactly these: what a "
             "shape already has is hardlinked into staging and kept. `views` renders "
             "the six Hunyuan cameras from the same load, shaded from this run's atlas "
             "or the shape's atlas.npz.",
    )
    ap.add_argument(
        "--summary_name", default=None,
        help="summary parquet name under out_root (default summary_rank{rank}.parquet); "
             "a sharded run gives every submission its own name",
    )
    a = ap.parse_args(argv)

    try:
        outputs = parse_outputs(a.outputs)
    except ValueError as error:
        ap.error(str(error))
    outputs_text = ",".join(outputs)
    if outputs != tuple(DEFAULT_OUTPUTS):
        # a default run's log is unchanged; anything else says so on its first line
        print(
            f"outputs: {outputs_text} (default {','.join(DEFAULT_OUTPUTS)}); what a "
            f"shape already has is hardlinked into staging and kept",
            flush=True,
        )
    if "views" in outputs and 0 < a.timeout_s < VIEWS_TIMEOUT_HINT_S:
        print(
            f"warning: --outputs {outputs_text} renders six 512x512 views per shape on "
            f"top of the bake; --timeout_s={a.timeout_s:g} is below "
            f"{VIEWS_TIMEOUT_HINT_S:g}s and will drop heavy meshes. Raise the cap and "
            f"run fewer workers per node.",
            flush=True,
        )

    def _stamp(rec):
        """Every summary row says which outputs the run asked for."""
        rec.setdefault("outputs", outputs_text)
        return rec

    shas = select_shas(a.list_path, a.shas, a.rank, a.world_size)
    if a.limit:
        shas = shas[: a.limit]
    meta = pd.read_parquet(a.metadata, columns=["glb_1k_path"])

    def _resolve(sha):
        if a.glb_root:
            # resolve_glb does not stat the path; check here so a missing file is
            # recorded stage="resolve" instead of costing a worker launch and a
            # stage="load" failure.
            path = resolve_glb(sha, a.glb_root, meta)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{sha}: {path} does not exist")
            return path
        path = find_glb(sha, meta, DEFAULT_GLB_ROOTS)
        if path is None:
            raise FileNotFoundError(
                f"{sha}: no metadata row or no GLB under {[str(r) for r in DEFAULT_GLB_ROOTS]}"
            )
        return path

    # resolve glbs up front; a shape whose glb path is unresolvable is recorded
    # as a failure rather than crashing the shard. Complete on-disk shapes are
    # skipped by default so a killed run can resume without redoing work.
    records = []
    jobs = []
    n_skip = 0
    for sha in shas:
        if not a.overwrite and shape_complete(a.out_root, sha, outputs=outputs):
            records.append(_stamp(_skip_rec(sha, a.out_root)))
            n_skip += 1
            continue
        try:
            jobs.append((sha, _resolve(sha), a.out_root))
        except Exception as e:
            records.append(_stamp(_fail(sha, "resolve", os.path.join(a.out_root, sha), e)))

    total = len(shas)
    done = [len(records)]
    if n_skip:
        print(
            f"skip: {n_skip}/{total} already complete under {a.out_root} "
            f"({len(jobs)} remaining)",
            flush=True,
        )

    def _report(rec):
        done[0] += 1
        records.append(_stamp(rec))
        wall = rec.get("wall_s")
        wall_s = f" wall={wall:.1f}s" if wall is not None else ""
        t = rec.get("timings") or {}
        xatlas_s = f" xatlas={t['xatlas']:.1f}s" if "xatlas" in t else ""
        print(
            f"[{done[0]}/{total}] {rec['sha']} "
            f"fp_voxel={rec.get('fp_voxel'):.3f} "
            f"fp_atlas={rec.get('fp_atlas', 0):.3f} "
            f"passed={rec.get('passed')}"
            f"{wall_s}{xatlas_s}"
            + (f" stage={rec['stage']}" if rec.get("stage") else ""),
            flush=True,
        )

    # timeout_s == 0 means no cap. Every job still gets a fresh subprocess.
    cap = a.timeout_s if a.timeout_s > 0 else float("inf")
    pool = None
    if jobs:
        worker_count = min(max(a.workers, 1), len(jobs))
        pool = ShapeProcessPool(worker_count, cap, outputs=outputs)
        print(
            f"pool: up to {pool.n_workers} fresh one-shape processes x "
            f"{pool.threads} threads ({_physical_cores()} physical cores)",
            flush=True,
        )
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        if pool is not None:
            for rec in pool.imap_unordered(jobs):
                _report(rec)
    finally:
        # always land a summary -- an interrupt mid-shard must not lose the
        # records for everything already processed
        if pool is not None:
            pool.close()
        if records:
            name = a.summary_name or f"summary_rank{a.rank}.parquet"
            out = os.path.join(a.out_root, name)
            os.makedirs(a.out_root, exist_ok=True)
            # whole or not at all: another rank's `collect` globs summary_*rank*.parquet in
            # the same tree, and must never read a half-written file
            tmp = out + ".tmp"      # the glob does not match it
            pd.DataFrame(records).to_parquet(tmp)
            os.replace(tmp, out)
            n_pass = sum(bool(r["passed"]) for r in records)
            n_timeout = sum(r.get("stage") == "timeout" for r in records)
            n_skip_done = sum(r.get("stage") == "skip" for r in records)
            print(
                f"DONE rank{a.rank}: {n_pass}/{len(records)} passed "
                f"fp>={FP_GATE} ({n_timeout} timed out, {n_skip_done} skipped); "
                f"wrote {out}",
                flush=True,
            )
        signal.signal(signal.SIGTERM, previous_sigterm)


# --- one shape ------------------------------------------------------------------
#
# `run worker`: one shape in a fresh interpreter. The requested `--outputs` all come
# from one load and normalize of the GLB and are written into the driver's staging
# directory, which already holds hardlinks to what the shape had.

def _detail(error: BaseException | str) -> str:
    """The error text plus the tail of the active traceback, when there is one."""
    detail = str(error)
    active_traceback = traceback.format_exc()
    if active_traceback.strip() != "NoneType: None":
        detail = f"{detail}\n{active_traceback[-1500:]}"
    return detail


def failure_record(
    sha: str,
    stage: str,
    out_dir: str,
    error: BaseException | str,
    *,
    wall_s: float | None = None,
    timings: dict[str, float] | None = None,
    outputs: str | None = None,
) -> dict[str, Any]:
    """Build the established per-shape failure record."""
    detail = _detail(error)
    record: dict[str, Any] = {
        "sha": sha,
        "fp_voxel": 0.0,
        "fp_atlas": 0.0,
        "passed": False,
        "error": f"{stage} failed: {detail}",
        "stage": stage,
        "out_dir": out_dir,
        "worker_pid": os.getpid(),
    }
    if wall_s is not None:
        record["wall_s"] = wall_s
    if timings is not None:
        record["timings"] = timings
    if outputs is not None:
        record["outputs"] = outputs
    return record


def _default_render(bundle, atlas, sha: str, out_dir: str):
    """Render the six Hunyuan views into `out_dir`, shaded from `atlas` (a path to an
    `atlas.npz` or the loaded arrays).

    The rasterizer context is this worker's own: one process per shape, so there is
    nothing to share and nothing to leak between shapes.
    """
    import nvdiffrast.torch as dr

    from data_processing.glb2representations.multiview import render_from_bundle

    return render_from_bundle(
        bundle, atlas, sha, out_dir, dr.RasterizeCudaContext(), CAMERAS
    )


def _atlas_for_views(staging_dir: str, final_out_dir: str) -> str:
    """The atlas the views are shaded from: this run's, if it baked one, else the
    shape's own `atlas.npz` (hardlinked into staging by the driver, or still in the
    output tree). Refuses with a clear message rather than rendering nothing."""
    for directory in (staging_dir, final_out_dir):
        path = os.path.join(directory, "atlas.npz")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"no atlas.npz in {staging_dir} or {final_out_dir}: the views are shaded from "
        f"an atlas, so add `atlas` to --outputs or run over a tree that has one"
    )


def process_one(
    sha: str,
    glb_path: str,
    staging_dir: str,
    final_out_dir: str,
    *,
    outputs=DEFAULT_OUTPUTS,
    load_fn=None,
    atlas_fn=None,
    voxelize_fn=None,
    validate_fn=None,
    render_fn=None,
) -> dict[str, Any]:
    """Run the requested `outputs` for one shape, from a single load of the GLB.

    Stages, in order: load, atlas, voxelization, validation, views. Validation needs
    both an atlas and a coords file in the staging directory (this run's or the
    driver's hardlink seed); without them the row carries no alignment number. A failed
    render never voids a bake this run produced -- the record keeps its verdict and
    carries `views_error` -- but a views-only run reports it as the stage failure it is.
    """
    outputs = tuple(outputs)
    outputs_text = ",".join(outputs)
    if load_fn is None:
        from data_processing.glb2representations.utils import load_and_merge

        load_fn = load_and_merge
    if atlas_fn is None and "atlas" in outputs:
        from data_processing.glb2representations.atlas import bake_atlas

        atlas_fn = bake_atlas
    if voxelize_fn is None and "voxels" in outputs:
        from data_processing.glb2representations.voxels import voxelize_bundle

        voxelize_fn = voxelize_bundle
    if validate_fn is None:
        from data_processing.glb2representations.utils import validate_shape

        validate_fn = validate_shape
    if render_fn is None:
        render_fn = _default_render

    threads = os.environ.get("UV_VOXEL_THREADS")
    if threads:
        import torch

        torch.set_num_threads(int(threads))
    os.makedirs(staging_dir, exist_ok=True)
    started = time.perf_counter()
    timings = None

    try:
        bundle = load_fn(glb_path)
    except Exception as error:
        return failure_record(
            sha,
            "load",
            final_out_dir,
            error,
            wall_s=time.perf_counter() - started,
            outputs=outputs_text,
        )

    atlas = None
    if "atlas" in outputs:
        try:
            atlas = atlas_fn(bundle, staging_dir, sha)
            timings = atlas.timings
        except Exception as error:
            return failure_record(
                sha,
                "atlas",
                final_out_dir,
                error,
                wall_s=time.perf_counter() - started,
                timings=timings,
                outputs=outputs_text,
            )

    voxels = None
    if "voxels" in outputs:
        try:
            voxels = voxelize_fn(bundle, staging_dir, sha)
        except Exception as error:
            return failure_record(
                sha,
                "voxelize",
                final_out_dir,
                error,
                wall_s=time.perf_counter() - started,
                timings=timings,
                outputs=outputs_text,
            )

    # alignment compares atlas texels with voxel coordinates, so it needs both:
    # produced by this run, or seeded into staging from what the shape already had
    atlas_path = (
        atlas.atlas_path if atlas is not None else os.path.join(staging_dir, "atlas.npz")
    )
    coords_path = (
        voxels.coords_path
        if voxels is not None
        else os.path.join(staging_dir, f"{sha}.coords.npz")
    )
    has_atlas = atlas is not None or os.path.isfile(atlas_path)
    has_voxels = voxels is not None or os.path.isfile(coords_path)
    validation = None
    if has_atlas and has_voxels:
        try:
            validation = validate_fn(
                staging_dir,
                sha,
                coords_path,
                atlas_path,
                bundle,
            )
        except Exception as error:
            return failure_record(
                sha,
                "validate",
                final_out_dir,
                error,
                wall_s=time.perf_counter() - started,
                timings=timings,
                outputs=outputs_text,
            )

    views_error = None
    if "views" in outputs:
        try:
            render_fn(
                bundle,
                _atlas_for_views(staging_dir, final_out_dir),
                sha,
                staging_dir,
            )
        except Exception as error:
            if outputs == ("views",):
                # nothing else to keep: the render is what this run was asked for
                return failure_record(
                    sha,
                    "views",
                    final_out_dir,
                    error,
                    wall_s=time.perf_counter() - started,
                    timings=timings,
                    outputs=outputs_text,
                )
            views_error = _detail(error)

    record: dict[str, Any] = {
        "sha": sha,
        "fp_voxel": validation.fp_voxel if validation is not None else float("nan"),
        "fp_atlas": validation.fp_atlas if validation is not None else float("nan"),
        "passed": validation.passed if validation is not None else views_error is None,
    }
    if voxels is not None:
        record["num_voxels"] = voxels.num_voxels
    if atlas is not None:
        record["occupied_texels"] = atlas.occupied_texels
    record.update({
        "wall_s": time.perf_counter() - started,
        "timings": timings,
        "out_dir": final_out_dir,
        "worker_pid": os.getpid(),
        "outputs": outputs_text,
    })
    if "views" in outputs:
        record["views_error"] = views_error
    return record


def _json_default(value: Any) -> Any:
    """Convert numpy-like scalar values without importing numpy eagerly."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_result(path: str, record: dict[str, Any]) -> None:
    """Atomically write a worker result so the driver never reads partial JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_processing.glb2representations.run worker",
        description="Generate UV+voxel data for one shape",
    )
    parser.add_argument("--sha", required=True)
    parser.add_argument("--glb_path", required=True)
    parser.add_argument("--staging_dir", required=True)
    parser.add_argument("--final_out_dir", required=True)
    parser.add_argument("--result_path", required=True)
    parser.add_argument(
        "--outputs", default=",".join(DEFAULT_OUTPUTS),
        help="comma-joined outputs to produce, from atlas,voxels,views "
             "(default %(default)s)",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        outputs = parse_outputs(args.outputs)
    except ValueError as error:
        parser.error(str(error))
    try:
        record = process_one(
            args.sha,
            args.glb_path,
            args.staging_dir,
            args.final_out_dir,
            outputs=outputs,
        )
    except BaseException as error:
        record = failure_record(
            args.sha,
            "unknown",
            args.final_out_dir,
            error,
            wall_s=time.perf_counter() - started,
            outputs=",".join(outputs),
        )
    write_result(args.result_path, record)
    return 0


# --- collect --------------------------------------------------------------------
#
# `collect` merges every ``summary_*rank*.parquet`` in the tree into one bake-verdict row
# per input uuid (latest run, then highest rank) and writes ``summary.parquet``,
# ``bake_pass.txt``, ``no_verdict.txt`` and ``unattempted.txt`` into ``--run_dir``.
# A verdict row is a non-``skip`` row from a run that produced both ``atlas`` and
# ``voxels``: only then did ``validate_shape`` run. A pass also needs the four artifacts
# with size > 0 (a failed NFS write leaves a 0-byte file). ``bake_pass.txt`` is copied to
# ``DEFAULT_PASS_LIST`` only when the other two lists are empty and ``--keep_record`` is
# not set.
#
# ``collect --views`` reads no summary: it checks each uuid of ``--input`` on disk and
# writes ``views_complete.txt`` and ``views_missing.txt`` into ``--out``. The checks are
# NFS round trips, so they run on a thread pool.

_SUMMARY_RE = re.compile(r"summary_(?:(.+)_)?rank([0-9]+)\.parquet")


def bake_artifact_paths(root, sha, resolution=256):
    """The four load-bearing files of a baked shape, under ``<root>/<sha>/``.

    The set itself is ``artifact_paths`` — the one the driver promotes on — so the
    collect's idea of a complete shape cannot drift from the bake's. Views are not part
    of it: ``bake_pass.txt`` means a complete, gate-passing bake whatever else a run
    produced.
    """
    return artifact_paths(os.path.join(root, sha), sha, resolution=resolution)


def complete(root, sha):
    """True iff every artifact exists with size > 0."""
    return all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in bake_artifact_paths(root, sha))


def row_outputs(value):
    """The outputs the run that wrote a summary row produced, as a tuple.

    A missing or empty value is the default bake set (a summary written without an
    ``outputs`` column comes from a run that baked atlas and voxels).
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return tuple(DEFAULT_OUTPUTS)
    text = str(value).strip()
    if not text:
        return tuple(DEFAULT_OUTPUTS)
    return parse_outputs(text)


def outputs_column(df):
    """`df`'s outputs as a Series of tuples, one per row (see `row_outputs`)."""
    if "outputs" not in df.columns:
        return pd.Series([tuple(DEFAULT_OUTPUTS)] * len(df), index=df.index, dtype=object)
    return df["outputs"].map(row_outputs)


def is_bake_verdict(outputs):
    """True iff a run with these outputs validated the shape, so its `passed` is the
    alignment gate's answer: alignment compares atlas texels with voxel coordinates,
    so it needs both."""
    return all(name in outputs for name in DEFAULT_OUTPUTS)


def _latest(df):
    """One row per sha: the highest run, then the highest rank."""
    return df.sort_values(["run", "rank"]).groupby("sha", as_index=False).tail(1)


def _output_tally(real, outputs):
    """Per output: input uuids with a row from a run that produced it, and how many of
    those latest rows succeeded (the gate for atlas/voxels; for views, a row that reached
    the end of its run and recorded no `views_error`).
    """
    tally = {}
    for name in OUTPUTS:
        # a boolean Series, not a list: `df[[]]` on an empty frame selects columns
        rows = _latest(real[outputs.map(lambda o, name=name: name in o).astype(bool)])
        if rows.empty:
            continue
        if name == "views":
            # Two ways a views run can fail, and both must count as a failure. A render
            # that failed while the bake stood records `views_error` and leaves `passed`
            # to the gate. A views-only run whose render failed, and every driver-side
            # abort or timeout, write a failure record instead: it carries a `stage` and
            # no `views_error` at all, so the column alone would score it OK -- and when
            # every views row is one, the column is missing from the frame entirely.
            ok = rows["stage"].isna()
            if "views_error" in rows.columns:
                ok &= rows["views_error"].isna()
            n_ok = int(ok.sum())
        else:
            n_ok = int((rows["passed"] == True).sum())   # noqa: E712
        tally[name] = {"rows": int(len(rows)), "ok": n_ok}
    return tally


def load_summaries(root):
    """Every shard summary under `root`, with `run` and `rank` columns and a `stage`
    column that exists even when no record had one (a shard of successes only)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(root, "summary_*rank*.parquet"))):
        match = _SUMMARY_RE.fullmatch(os.path.basename(path))
        if match is None:
            continue
        df = pd.read_parquet(path)
        df["run"] = match.group(1) or ""     # runs are timestamps, so they sort oldest first
        df["rank"] = int(match.group(2))
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["stage"] = df["stage"] if "stage" in df.columns else None
    return df


def collect(run_dir, out_root=None, input_path=None, keep_record=False):
    """Merge the summaries and write the four result files; return the counts.

    `keep_record`: never copy the verdict to `DEFAULT_PASS_LIST`, the tracked record, even
    when the collect is complete (`run.sh` sets it)."""
    run_dir = str(run_dir)
    with open(os.path.join(run_dir, "run.json")) as handle:
        run = json.load(handle)
    root = str(out_root) if out_root else str(DEFAULT_OUT_ROOT)
    inp = read_uuids(input_path if input_path else run["input"])
    wanted = set(inp)

    df = load_summaries(root)
    if df is None:
        sys.exit("no summaries yet")

    # Real rows only: a resume's skip row is never a verdict. A partial run's row is
    # not one either (it never validated the shape). Highest run/rank wins.
    real = df[df["stage"].fillna("") != "skip"]
    real = real[real["sha"].isin(wanted)]
    outputs = outputs_column(real)
    tally = _output_tally(real, outputs)
    rows = _latest(real[outputs.map(is_bake_verdict).astype(bool)])
    rows = rows.assign(complete=[complete(root, s) for s in rows["sha"]])
    rows.to_parquet(os.path.join(run_dir, "summary.parquet"))

    ok = (rows["passed"] == True) & rows["complete"]   # noqa: E712
    passed_set = set(rows.loc[ok, "sha"])
    seen = set(rows["sha"])
    passed = [s for s in inp if s in passed_set]
    no_verdict = [s for s in inp if s not in seen and complete(root, s)]
    unattempted = [s for s in inp if s not in seen and not complete(root, s)]

    write_uuids(os.path.join(run_dir, "bake_pass.txt"), passed)
    write_uuids(os.path.join(run_dir, "no_verdict.txt"), no_verdict)
    write_uuids(os.path.join(run_dir, "unattempted.txt"), unattempted)

    bad = rows.loc[~ok].copy()
    bad["stage"] = bad["stage"].fillna("gate")
    bad.loc[bad["passed"] == True, "stage"] = "incomplete"   # noqa: E712  (passed, artifacts missing or empty)
    stages = bad["stage"].value_counts().to_dict()

    full = not no_verdict and not unattempted
    if full and keep_record:
        tail = f"-> {run_dir}/bake_pass.txt (--keep_record: {DEFAULT_PASS_LIST} untouched)"
    elif full:
        target = str(DEFAULT_PASS_LIST)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        shutil.copyfile(os.path.join(run_dir, "bake_pass.txt"), target)
        tail = f"-> {run_dir}/bake_pass.txt copied to {target}"
    else:
        tail = f"-> {run_dir}/bake_pass.txt (partial: {DEFAULT_PASS_LIST} untouched)"
    # a bake-only tree's line is unchanged; a tree with partial runs in it says what
    # each output has
    per_output = ""
    if tally and (
        set(tally) != set(DEFAULT_OUTPUTS)
        or any(t["rows"] != len(rows) for t in tally.values())
    ):
        per_output = "  outputs " + ", ".join(
            f"{name} {t['ok']}/{t['rows']}" for name, t in tally.items()
        )
    print(
        f"input {len(inp)}  rows {len(rows)}  passed {len(passed)}  failed {len(bad)} {stages}"
        f"  no_verdict {len(no_verdict)}  unattempted {len(unattempted)}{per_output}  {tail}"
    )
    return {
        "input": len(inp), "rows": len(rows), "passed": len(passed), "failed": len(bad),
        "stages": stages, "no_verdict": len(no_verdict), "unattempted": len(unattempted),
        "outputs": tally, "published": full and not keep_record,
    }


#: Threads for `collect --views`' per-shape checks (I/O-bound NFS round trips).
VIEWS_CHECK_THREADS = 32


def views_complete(root, sha):
    """True iff every file of the shape's six views exists: `shape_complete`'s views rule.

    Sizes are not checked, unlike the bake's `complete`: that is the rule that reproduces
    the tracked `views_complete.txt` (a size check would drop shapes with a 0-byte
    `render_cond/000.png` from it). The files come from one `scandir` of `render_tex/`
    and `render_cond/` rather than a stat per path, which is much faster on NFS. A
    missing render dir means the views never ran; any other OSError (EIO, ESTALE on
    NFS) is raised, so the collect fails instead of writing a wrong list.
    """
    shape_dir = os.path.join(root, sha)
    want = set(artifact_paths(shape_dir, sha, ("views",)))
    present = set()
    try:
        for sub in ("render_tex", "render_cond"):
            with os.scandir(os.path.join(shape_dir, sub)) as entries:
                present.update(e.path for e in entries if e.path in want and e.is_file())
    except (FileNotFoundError, NotADirectoryError):
        return False
    return want <= present


def collect_views(out_dir, out_root=None, input_path=None):
    """Split the input list into complete and missing views; write both lists into
    `out_dir`; return the counts."""
    root = str(out_root) if out_root else str(DEFAULT_OUT_ROOT)
    list_path = str(input_path) if input_path else str(DEFAULT_PASS_LIST)
    inp = read_uuids(list_path)
    if not inp:
        sys.exit(f"{list_path} is empty")
    with ThreadPoolExecutor(max_workers=VIEWS_CHECK_THREADS) as pool:
        ok = list(pool.map(lambda s: views_complete(root, s), inp))
    done = [s for s, good in zip(inp, ok) if good]
    missing = [s for s, good in zip(inp, ok) if not good]
    os.makedirs(out_dir, exist_ok=True)
    write_uuids(os.path.join(out_dir, "views_complete.txt"), done)
    write_uuids(os.path.join(out_dir, "views_missing.txt"), missing)
    print(
        f"input {len(inp)} ({list_path})  complete {len(done)}  missing {len(missing)}  "
        f"root {root}  -> {out_dir}/views_complete.txt, views_missing.txt"
    )
    return {"input": len(inp), "complete": len(done), "missing": len(missing)}


def collect_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m data_processing.glb2representations.run collect",
        description="The `collect` step of the bake: a run's shard summaries -> the stage's "
                    "result list. With --views: the list of shapes whose six views are complete.",
    )
    ap.add_argument("--run_dir", default=None,
                    help="out/runs/<RUN>, holding run.json (required without --views)")
    ap.add_argument("--out_root", default=None,
                    help="tree with the summaries and shape dirs (default DEFAULT_OUT_ROOT)")
    ap.add_argument("--input", default=None,
                    help="uuid list (default run.json's input; with --views, out/bake_pass.txt)")
    ap.add_argument("--views", action="store_true",
                    help="check each uuid's six views on disk (the 36 named PNGs, "
                         "transforms.json and render_cond/000.png exist) and write "
                         "views_complete.txt and views_missing.txt into --out")
    ap.add_argument("--out", default=None,
                    help="--views only: directory for the two lists (default --run_dir). The "
                         "tracked records are views_{complete,missing}.txt in "
                         "data_processing/glb2representations/out/; they are rewritten only when "
                         "--out names that folder")
    ap.add_argument("--keep_record", action="store_true",
                    help="never copy the verdict to out/bake_pass.txt (the tracked record), even "
                         "when the collect is complete; the run's own lists stay in --run_dir")
    a = ap.parse_args(argv)
    if a.views:
        out_dir = a.out or a.run_dir
        if not out_dir:
            ap.error("--views needs --out (or --run_dir)")
        if a.out is None and Path(out_dir).resolve() == OUT_DIR.resolve():
            ap.error("--run_dir names out/, which holds the tracked records; "
                     "pass --out to rewrite them")
        collect_views(out_dir, a.out_root, a.input)
        return
    if not a.run_dir:
        ap.error("--run_dir is required (or pass --views)")
    if a.out:
        ap.error("--out goes with --views; the bake's verdict is written into --run_dir")
    collect(a.run_dir, a.out_root, a.input, keep_record=a.keep_record)

def main(argv=None):
    """`collect` and `worker` are subcommands; any other argument list is a run."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["collect"]:
        return collect_main(argv[1:])
    if argv[:1] == ["worker"]:
        return worker_main(argv[1:])
    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
