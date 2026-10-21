"""Pure admin request handlers: (store, params) -> (status, dict). No HTTP.

A read-only progress dashboard over the annotation DB — no writes, no schema of its own.
Token checking lives in server.py; by the time a handler here runs the caller is already
authorized.

Kept separate from api.py so the annotator-facing API stays small and obviously safe:
nothing here is reachable without the admin token, and nothing in api.py depends on it.
"""

MAX_ITEM_PAGE = 200
MAX_SESSION_PAGE = 500
MAX_ACTIVITY_DAYS = 400
DEFAULT_BROWSE_PAGE = 48


def _as_bool(v):
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes")


def _projection(overview):
    """Observed human throughput and the implied effort left for one full pass.

    Numerator and denominator both come from the sessions table (annotated_count over
    active_seconds) so they describe the same work; mixing session time with item-table
    counts would divide by a window the items were never annotated in.

    The sample can be small and include test sessions, so callers must show
    sample_items/sample_seconds next to the estimate rather than presenting it as a
    measurement.
    """
    seconds = overview.get("active_seconds") or 0
    done = overview.get("session_items") or 0
    remaining = overview.get("remaining") or 0
    if seconds <= 0 or done <= 0:
        return {"items_per_hour": None, "remaining_hours": None,
                "sample_items": done, "sample_seconds": seconds}
    per_hour = done / (seconds / 3600.0)
    return {
        "items_per_hour": round(per_hour, 1),
        "remaining_hours": round(remaining / per_hour, 1) if per_hour > 0 else None,
        "sample_items": done,
        "sample_seconds": seconds,
    }


def handle_overview(store):
    overview = store.admin_overview()
    overview["projection"] = _projection(overview)
    return 200, overview


def handle_annotators(store):
    return 200, {"annotators": store.annotator_stats()}


def handle_annotator(store, params):
    """One annotator's detail page.

    `name` is taken literally, so `?name=` selects the empty-string annotator. Rows with a
    NULL annotator are unreachable that way, so they get an explicit `?unattributed=1`.
    """
    if _as_bool(params.get("unattributed")):
        name = None
    elif "name" in params:
        name = params["name"]
    else:
        return 400, {"error": "name required (or unattributed=1)"}
    try:
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 50))
    except (TypeError, ValueError):
        return 400, {"error": "bad offset/limit"}
    if offset < 0 or limit < 1:
        return 400, {"error": "bad offset/limit"}
    return 200, store.annotator_detail(name, offset=offset, limit=min(limit, MAX_ITEM_PAGE))


def handle_browse(store, params):
    """A filtered page of the whole set — the dashboard's shape browser.

    The annotator-facing tool deliberately shows nobody more than their own claimed
    batch, so this is the only view of the full set. Read-only, like everything
    else here: it hands back rows, and changing one still means going through the
    annotation tool.

    An unrecognized `status`/`order` is a 400 rather than a silent fallback — a typo
    that quietly widened the filter to the whole set would look like a real result.
    """
    try:
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", DEFAULT_BROWSE_PAGE))
    except (TypeError, ValueError):
        return 400, {"error": "bad offset/limit"}
    if offset < 0 or limit < 1:
        return 400, {"error": "bad offset/limit"}
    # Same rule as handle_annotator: `?annotator=` is a filter selecting the
    # empty-string annotator, an absent key is no filter, and NULL needs its own flag.
    unattributed = _as_bool(params.get("unattributed"))
    annotator = params["annotator"] if ("annotator" in params and not unattributed) else None
    try:
        page = store.admin_browse(
            status=params.get("status") or None,
            bucket=params.get("bucket") or None,
            flag=params.get("flag") or None,
            annotator=annotator,
            unattributed=unattributed,
            q=params.get("q") or None,
            claimed=_as_bool(params.get("claimed")),
            order=params.get("order") or "uuid",
            offset=offset, limit=min(limit, MAX_ITEM_PAGE),
            # The only way to see rows a later filter dropped. They keep their labels and
            # are never handed to an annotator, so nothing else in the tool shows them.
            include_out_of_set=_as_bool(params.get("include_out_of_set")),
        )
    except ValueError as e:
        return 400, {"error": str(e)}
    return 200, page


def handle_categories(store):
    return 200, {"categories": store.category_progress_split()}


def handle_activity(store, params):
    try:
        days = int(params.get("days", 60))
        limit = int(params.get("limit", 50))
    except (TypeError, ValueError):
        return 400, {"error": "bad days/limit"}
    if days < 1 or limit < 1:
        return 400, {"error": "bad days/limit"}
    return 200, {
        "daily": store.daily_activity(days=min(days, MAX_ACTIVITY_DAYS)),
        "sessions": store.recent_sessions(limit=min(limit, MAX_SESSION_PAGE)),
    }
