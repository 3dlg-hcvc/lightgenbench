"""Pure request handlers: (store, params/body) -> (status, dict). No HTTP."""
import json

from . import utils as annotation_utils

_EDITABLE = ("category", "emissive", "flags", "verified", "annotator", "notes",
             "style", "emission_complete")

_SESSION_FIELDS = ("annotator", "planned_minutes", "started_at", "ended_at",
                   "active_seconds", "annotated_count", "status")


def _as_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes")


def handle_items(store, params):
    try:
        offset = int(params.get("offset", 0))
        limit = min(int(params.get("limit", 60)), 200)
    except ValueError:
        return 400, {"error": "bad offset/limit"}
    # "mine": only the batch this person has claimed. `me` is the viewer's identity;
    # `annotator` below stays a FILTER ("annotated by X"), used by the auto-zero view —
    # conflating the two makes the query exclude exactly the rows it should return.
    claimed_by = params["me"] if (_as_bool(params.get("mine")) and params.get("me")) else None
    verified = _as_bool(params.get("verified")) if "verified" in params else None
    emissive = _as_bool(params.get("emissive")) if "emissive" in params else None
    if claimed_by:
        # A batch is returned whole, decided shapes included, so the pager's positions
        # cannot move under it. A status filter would drop a shape the instant it was
        # decided and `offset += 1` would skip the next one. app.js already omits the
        # filter in batch mode; enforcing it here means a stale cached client cannot
        # bring it back.
        verified = emissive = None
    rows, total = store.query(
        category=params.get("category") or None,
        bucket=params.get("bucket") or None,
        flag=params.get("flag") or None,
        verified=verified,
        emissive=emissive,
        annotator=params.get("annotator") or None,
        q=params.get("q") or None,
        offset=offset, limit=limit, claimed_by=claimed_by,
    )
    body = {"items": rows, "total": total, "offset": offset, "limit": limit}
    if claimed_by:
        body["batch"] = store.claim_progress(claimed_by)
    return 200, body


def handle_save(store, body):
    uuid = body.get("uuid")
    if not uuid:
        return 400, {"error": "uuid required"}
    # Claiming exists so two people never annotate the same shape. The frontend alone
    # cannot enforce it — a stale cached page could save straight over a live batch — so
    # refuse that here. Deliberately narrow: only a LIVE claim held by SOMEONE ELSE
    # blocks a write. Unheld rows stay writable (the auto-filter never claims), and so do
    # lapsed ones — a batch whose TTL expires mid-session is still on its annotator's
    # screen, and refusing their save would throw away work already done. Names are
    # self-declared, so this stops accidental and stale-client overlap, not spoofing.
    fields = {k: body[k] for k in _EDITABLE if k in body}
    holder = store.upsert_unless_held_by_other(uuid, body.get("annotator"), **fields)
    if holder is not None:
        return 409, {"error": "This shape is claimed by someone else.",
                     "claimed_by": holder, "uuid": uuid}
    # Deciding a shape proves the batch is being worked, so push its TTL back. Otherwise
    # a batch spanning more than 48 hours lapses mid-session and its rows vanish from
    # under the pager, changing both the list and every position in it.
    store.touch_claims(body.get("annotator"))
    return 200, {"ok": True, "uuid": uuid}


def handle_session(store, body):
    session_id = body.get("session_id")
    if not session_id:
        return 400, {"error": "session_id required"}
    fields = {k: body[k] for k in _SESSION_FIELDS if k in body}
    try:
        store.upsert_session(session_id, **fields)
    except (ValueError, TypeError):
        return 400, {"error": "bad field value"}
    return 200, {"ok": True, "session_id": session_id}


def handle_claim(store, body):
    """Claim a batch of shapes for one annotator."""
    name = body.get("annotator")
    if not name:
        return 400, {"error": "annotator required"}
    try:
        count = int(body.get("count", 50))
    except (TypeError, ValueError):
        return 400, {"error": "bad count"}
    if not 1 <= count <= 2000:
        return 400, {"error": "count must be 1..2000"}
    # The claim panel sends `categories` (several ticked, or none at all for a mix).
    # `category` is still accepted so a cached older client keeps working.
    cats = body.get("categories")
    if cats is None:
        cats = body.get("category")
    got = store.claim_batch(name, count, category=cats or None)
    got["batch"] = store.claim_progress(name)
    return 200, got


def handle_claimable(store):
    """What is free to claim per category — the claim panel's live counts."""
    return 200, {"categories": store.claimable_by_category()}


def handle_batch(store, params):
    """Progress of the caller's current batch (empty if they hold none)."""
    return 200, store.claim_progress(params.get("annotator"))


def handle_release(store, body):
    """Give unfinished claims back to the pool."""
    name = body.get("annotator")
    if not name:
        return 400, {"error": "annotator required"}
    return 200, {"released": store.release_claims(name)}


def handle_register(store, body):
    """Register an annotator and report their share of the set."""
    reg = store.register_annotator(body.get("name"))
    if reg is None:
        return 400, {"error": "a real annotator name is required"}
    return 200, reg


def handle_view(store, body):
    """Record how long an annotator stayed on one shape.

    An implausible duration is dropped by the store rather than rejected: the
    client sends these from pagehide/beacon paths where a 4xx is invisible, and
    losing one timing is preferable to the client retrying or logging errors.
    """
    uuid = body.get("uuid")
    if not uuid:
        return 400, {"error": "uuid required"}
    patch = body.get("patch")
    recorded = store.record_view(
        uuid, body.get("annotator"), body.get("dwell_ms"),
        saves=body.get("saves", 0), patch=patch if isinstance(patch, dict) else None)
    # Looking at shapes is activity too, so it renews the batch's TTL just as deciding
    # one does. Someone reading carefully through a batch without saving for a long
    # stretch would otherwise have it lapse out from under them.
    store.touch_claims(body.get("annotator"))
    return 200, {"ok": True, "recorded": recorded}


def handle_sessions(store, params):
    try:
        limit = max(0, min(int(params.get("limit", 500)), 2000))
    except ValueError:
        return 400, {"error": "bad limit"}
    return 200, {"sessions": store.list_sessions(
        annotator=params.get("annotator") or None, limit=limit)}


def handle_progress(store):
    return 200, store.progress()


def handle_export(store):
    return 200, store.export_dict()


def handle_categories(categories):
    return 200, {"categories": list(categories), "flags": annotation_utils.FLAGS,
                 "styles": annotation_utils.STYLES}


def handle_category_progress(store):
    return 200, {"categories": store.category_progress()}


# ---- user study ---------------------------------------------------------------------
STUDY_MAX_BYTES = 2_000_000


def handle_study_submit(body, results_dir, raw_len=None):
    """Store one study submission (partial after each page, final at the end).

    Writes <results_dir>/<participant-slug>_<utc>_<partial|final>.json atomically and appends a
    one-line record to <results_dir>/submissions.jsonl. Returns (status, json) like the other
    handlers. Never trusts the client: only a dict with a non-empty `participant` string and a
    `results` list is accepted, and the body is capped at STUDY_MAX_BYTES."""
    import datetime
    import re
    from pathlib import Path
    if raw_len is not None and raw_len > STUDY_MAX_BYTES:
        return 413, {"error": "too large"}
    if not isinstance(body, dict):
        return 400, {"error": "expected a json object"}
    name = body.get("participant")
    if not isinstance(name, str) or not name.strip():
        return 400, {"error": "participant missing"}
    if not isinstance(body.get("results"), list):
        return 400, {"error": "results missing"}
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())[:48] or "anon"
    kind = "final" if body.get("finished_at") else "partial"
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    fname = f"{slug}_{ts}_{kind}.json"
    tmp = d / (fname + ".part")
    tmp.write_text(json.dumps(body, indent=1))
    tmp.replace(d / fname)
    n_ranked = sum(1 for r in body["results"]
                   if isinstance(r, dict) and isinstance(r.get("ranks"), dict) and r["ranks"])
    with open(d / "submissions.jsonl", "a") as f:
        f.write(json.dumps({"file": fname, "participant": name.strip(), "kind": kind,
                            "n_objects": body.get("n_objects"), "n_ranked": n_ranked,
                            "received_utc": ts}) + "\n")
    return 200, {"ok": True, "saved": fname, "kind": kind, "n_ranked": n_ranked}
