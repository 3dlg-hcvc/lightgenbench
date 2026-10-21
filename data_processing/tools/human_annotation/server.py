"""Threaded stdlib HTTP server: frontend + image serving + annotation API.

Its two offline jobs are subcommands: `prewarm` fills the emission cache the server reads, and
`export` writes what `GET /api/export` returns.

Run: conda run -n lightgen python -m data_processing.tools.human_annotation.server [port]
     conda run -n lightgen python -m data_processing.tools.human_annotation.server prewarm [workers]
     conda run -n lightgen python -m data_processing.tools.human_annotation.server export [out.json]
"""
import argparse
import hmac
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import pandas as pd

from utils import review_images

from . import admin, api
from . import utils as annotation_utils
from .store import AnnotationStore

_CT = {".html": "text/html", ".js": "application/javascript", ".css": "text/css",
       # the guide's screenshots, served through the same /static/ route
       ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
       ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml"}
_IMG_CACHE = "public, max-age=31536000, immutable"  # uuid-keyed images never change


def load_manifest_paths(manifest_path):
    df = pd.read_parquet(manifest_path)
    out = {}
    for r in df.itertuples(index=False):
        # `ditem_dir` alone: the manifest's columns are uuid, auto_category, ditem_dir
        # (`build_manifest.build_manifest_df`). Naming any other column here makes the
        # server die on startup the next time the manifest is rebuilt.
        out[r.uuid] = {"ditem_dir": r.ditem_dir}
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: a reverse proxy reuses backend conns instead
                                   # of opening one per request (much faster for a page of
                                   # ~30 images). Safe because every response sets Content-Length.
    store = None
    manifest_paths = {}
    categories = []
    frontend_dir = None

    def log_message(self, *a):
        pass

    def _send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, status, data, ctype, no_cache=False, cache=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if no_cache:  # frontend HTML/JS/CSS must revalidate so edits reach annotators
            self.send_header("Cache-Control", "no-cache, must-revalidate")
        elif cache:   # immutable uuid-keyed images -> browser caches hard, no re-download
            self.send_header("Cache-Control", cache)
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _send_file(self, path, ctype, no_cache=False, cache=None):
        # None is what the render builders answer when neither root holds the image
        # (`utils.py`); it means the same thing to a browser as an unreadable file, so
        # both land on the card's "not rendered" placeholder rather than a traceback.
        if path is None:
            return self._send_bytes(404, b"not found", "text/plain")
        try:
            data = path.read_bytes()
        except OSError:
            return self._send_bytes(404, b"not found", "text/plain")
        self._send_bytes(200, data, ctype, no_cache=no_cache, cache=cache)

    def _admin_ok(self, params):
        """Constant-time check of the admin token.

        Accepts the X-Admin-Token header (what admin.js sends) or ?key= (what a browser
        navigating to /admin can send, since a navigation can't set headers). An unset
        ANN74K_ADMIN_TOKEN disables the admin surface outright, so a deploy that forgets
        to configure it is closed rather than wide open.
        """
        token = annotation_utils.ADMIN_TOKEN
        if not token:
            return False
        supplied = self.headers.get("X-Admin-Token") or params.get("key") or ""
        return hmac.compare_digest(
            supplied.encode("utf-8", "ignore"), token.encode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        params = {k: v[0] for k, v in parse_qs(u.query).items()}

        if u.path == "/" or u.path == "/index.html":
            return self._send_file(self.frontend_dir / "index.html", "text/html", no_cache=True)
        if parts and parts[0] == "static" and len(parts) == 2:
            f = self.frontend_dir / parts[1]
            return self._send_file(f, _CT.get(f.suffix, "application/octet-stream"), no_cache=True)
        if u.path == "/api/study/ping":
            return self._send_json(200, {"ok": True})
        if u.path == "/guide":
            return self._send_file(annotation_utils.GUIDE_PATH, "text/markdown; charset=utf-8",
                                   no_cache=True)

        # Admin routes answer 404 — not 403 — when the token is missing or wrong, so the
        # dashboard is indistinguishable from a route that was never built.
        if u.path == "/admin":
            if not self._admin_ok(params):
                return self._send_bytes(404, b"not found", "text/plain")
            return self._send_file(self.frontend_dir / "admin.html", "text/html", no_cache=True)

        if parts[:2] == ["api", "admin"]:
            if not self._admin_ok(params):
                return self._send_json(404, {"error": "not found"})
            # Re-parse keeping blank values: `?name=` selects the empty-string annotator,
            # a real value in the DB, and the default parse_qs would drop the key entirely.
            aparams = {k: v[0] for k, v in parse_qs(u.query, keep_blank_values=True).items()}
            if parts == ["api", "admin", "overview"]:
                return self._send_json(*admin.handle_overview(self.store))
            if parts == ["api", "admin", "annotators"]:
                return self._send_json(*admin.handle_annotators(self.store))
            if parts == ["api", "admin", "annotator"]:
                return self._send_json(*admin.handle_annotator(self.store, aparams))
            if parts == ["api", "admin", "items"]:
                return self._send_json(*admin.handle_browse(self.store, aparams))
            if parts == ["api", "admin", "categories"]:
                return self._send_json(*admin.handle_categories(self.store))
            if parts == ["api", "admin", "activity"]:
                return self._send_json(*admin.handle_activity(self.store, aparams))
            return self._send_json(404, {"error": "unknown admin api"})

        if parts[:1] == ["api"]:
            if parts == ["api", "items"]:
                return self._send_json(*api.handle_items(self.store, params))
            if parts == ["api", "progress"]:
                return self._send_json(*api.handle_progress(self.store))
            if parts == ["api", "export"]:
                return self._send_json(*api.handle_export(self.store))
            if parts == ["api", "categories"]:
                return self._send_json(*api.handle_categories(self.categories))
            if parts == ["api", "category_progress"]:
                return self._send_json(*api.handle_category_progress(self.store))
            if parts == ["api", "batch"]:
                return self._send_json(*api.handle_batch(self.store, params))
            if parts == ["api", "claimable"]:
                return self._send_json(*api.handle_claimable(self.store))
            if parts == ["api", "sessions"]:
                return self._send_json(*api.handle_sessions(self.store, params))
            return self._send_json(404, {"error": "unknown api"})

        if parts[:1] == ["img"]:
            return self._serve_image(parts)

        return self._send_bytes(404, b"not found", "text/plain")

    def _serve_image(self, parts):
        # /img/thumb/<uuid> | /img/emission/<uuid> | /img/render/<kind>/<uuid>
        #   | /img/mv/<preset>/<uuid>  (animated multiview orbit webp)
        if len(parts) == 3 and parts[1] == "thumb":
            return self._send_file(annotation_utils.thumb_path(parts[2]), "image/png",
                                   cache=_IMG_CACHE)
        if len(parts) == 4 and parts[1] == "mv":
            preset, uuid = parts[2], parts[3]
            if preset not in annotation_utils.MULTIVIEW_PRESETS:
                return self._send_bytes(404, b"no preset", "text/plain")
            return self._send_file(annotation_utils.multiview_webp_path(preset, uuid), "image/webp",
                                   cache=_IMG_CACHE)
        if len(parts) == 3 and parts[1] == "emission":
            uuid = parts[2]
            info = self.manifest_paths.get(uuid)
            if not info:
                return self._send_bytes(404, b"no item", "text/plain")
            p = annotation_utils.get_emission_cached(uuid)
            if p is None:
                return self._send_bytes(404, b"no emission", "text/plain")
            return self._send_file(p, "image/png", cache=_IMG_CACHE)
        if len(parts) == 4 and parts[1] == "render" and parts[2] in ("box", "boxoff"):
            # dark-box still pair: flat by uuid, no manifest entry needed
            fn = (annotation_utils.box_render_path if parts[2] == "box"
                  else annotation_utils.box_render_off_path)
            return self._send_file(fn(parts[3]), "image/png", cache=_IMG_CACHE)
        if len(parts) == 4 and parts[1] == "render":
            kind, uuid = parts[2], parts[3]
            info = self.manifest_paths.get(uuid)
            # Checked against the kinds the resolver knows: an unknown kind is a KeyError
            # inside `review_images.still`, i.e. a 500.
            if not info or kind not in review_images.STILL_KINDS:
                return self._send_bytes(404, b"no render", "text/plain")
            return self._send_file(annotation_utils.render_path(kind, info["ditem_dir"], uuid),
                                   "image/png", cache=_IMG_CACHE)
        return self._send_bytes(404, b"bad image path", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/study/submit":
            length = int(self.headers.get("Content-Length", 0))
            if length > api.STUDY_MAX_BYTES:
                return self._send_json(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json(400, {"error": "bad json"})
            return self._send_json(*api.handle_study_submit(
                body, annotation_utils.STUDY_RESULTS_DIR, length))
        if u.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json(400, {"error": "bad json"})
            return self._send_json(*api.handle_save(self.store, body))
        if u.path in ("/api/claim", "/api/release"):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json(400, {"error": "bad json"})
            handler = api.handle_claim if u.path == "/api/claim" else api.handle_release
            return self._send_json(*handler(self.store, body))
        if u.path == "/api/annotator":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json(400, {"error": "bad json"})
            return self._send_json(*api.handle_register(self.store, body))
        if u.path == "/api/view":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json(400, {"error": "bad json"})
            return self._send_json(*api.handle_view(self.store, body))
        if u.path == "/api/session":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json(400, {"error": "bad json"})
            return self._send_json(*api.handle_session(self.store, body))
        return self._send_json(404, {"error": "unknown api"})


class _Server(ThreadingHTTPServer):
    daemon_threads = True      # kept-alive connection threads must not block shutdown
    request_queue_size = 128   # larger listen backlog for bursts of concurrent image requests


def make_server(host, port, store, manifest_paths, categories, frontend_dir):
    handler = type("BoundHandler", (Handler,), {
        "store": store, "manifest_paths": manifest_paths,
        "categories": list(categories), "frontend_dir": frontend_dir,
    })
    return _Server((host, port), handler)


def serve(port):
    host = os.environ.get("ANN74K_HOST", "0.0.0.0")  # set to 127.0.0.1 behind a reverse proxy
    store = AnnotationStore(annotation_utils.DB_PATH)
    store.init_schema()
    manifest_paths = load_manifest_paths(annotation_utils.MANIFEST_PATH)
    categories = json.loads(annotation_utils.CATEGORIES_PATH.read_text())
    srv = make_server(host, port, store, manifest_paths, categories, annotation_utils.FRONTEND_DIR)
    print(f"annotate74k serving {len(manifest_paths)} items on {host}:{port} "
          f"(progress={store.progress()})")
    srv.serve_forever()


# ---- prewarm ------------------------------------------------------------------------------

def _work(uuid):
    return annotation_utils.get_emission_cached(uuid) is not None


def prewarm(workers):
    """Pre-extract all emission PNGs into the cache so the "more" overlay is instant
    (otherwise each emission is extracted lazily on first view: load atlas.npz + mask).

    Set the CACHE dir (`ANN74K_EMISSION_CACHE_DIR`) the server reads, not just the data dir:
    a deploy that overrides it reads that directory, not the default one under `$ANN74K_DATA_DIR`.
    The shape list is `MANIFEST_PATH` — `$ANN74K_DATA_DIR/manifest.parquet`, i.e. the repo's own
    `_data/` unless that is set too.
    """
    df = pd.read_parquet(annotation_utils.MANIFEST_PATH)
    annotation_utils.EMISSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(df["uuid"])
    print(f"prewarming {len(rows)} emissions -> {annotation_utils.EMISSION_CACHE_DIR} "
          f"({workers} workers)", flush=True)
    ok = miss = 0
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(_work, rows, chunksize=32)):
            ok += bool(res)
            miss += not res
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(rows) - (i + 1)) / rate / 60
                print(f"  {i+1}/{len(rows)} ok={ok} miss={miss} "
                      f"{rate:.0f}/s eta={eta:.0f}min", flush=True)
    print(f"done: ok={ok} miss={miss} in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


# ---- export -------------------------------------------------------------------------------

def export(out):
    """Dump the SQLite annotation store to a JSON file."""
    out = out or str(annotation_utils.DATA_DIR / "annotations_export.json")
    store = AnnotationStore(annotation_utils.DB_PATH)
    _, body = api.handle_export(store)
    with open(out, "w") as f:
        json.dump(body, f, indent=2)
    print(f"wrote {out}: {len(body['items'])} items")
    return 0


_PROG = "python -m data_processing.tools.human_annotation.server"


def main(argv=None):
    """`[port]` serves; `prewarm` and `export` are the offline jobs."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("prewarm", "export"):
        ap = argparse.ArgumentParser(prog=_PROG)
        sub = ap.add_subparsers(dest="cmd", required=True)
        p = sub.add_parser("prewarm", help="pre-extract every emission PNG into the cache")
        p.add_argument("workers", nargs="?", type=int, default=8,
                       help="worker processes (default 8)")
        e = sub.add_parser("export", help="dump the annotation DB to JSON")
        e.add_argument("out", nargs="?", default=None,
                       help="output file (default $ANN74K_DATA_DIR/annotations_export.json)")
        args = ap.parse_args(argv)
        return prewarm(args.workers) if args.cmd == "prewarm" else export(args.out)
    ap = argparse.ArgumentParser(
        prog=_PROG, description="Serve the annotation tool.",
        epilog="offline jobs: `prewarm [workers]` fills the emission cache, `export [out.json]` "
               "dumps the DB; `<job> -h` for each.")
    ap.add_argument("port", nargs="?", type=int, default=annotation_utils.PORT,
                    help="port (default $ANN74K_PORT or 8074)")
    serve(ap.parse_args(argv).port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
