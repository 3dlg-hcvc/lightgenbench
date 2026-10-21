"""SQLite-backed annotation store. Fresh connection per op; WAL mode."""
import json
import zlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

EDITABLE = ("category", "emissive", "flags", "verified", "annotator", "notes",
            "style", "emission_complete")
STYLES = ("artistic", "real")
SESSION_FIELDS = ("annotator", "planned_minutes", "started_at", "ended_at",
                  "active_seconds", "annotated_count", "status")
SESSION_STATUSES = ("active", "paused", "completed", "stopped")


def _now():
    return datetime.now(timezone.utc).isoformat()


def shard_key(uuid):
    """Stable, evenly-spread integer for a uuid.

    crc32 rather than the uuid's own hex prefix so any id shape works, and
    rather than Python's hash() because that is salted per process — the same
    uuid must land in the same shard on every server start."""
    return zlib.crc32(uuid.encode("utf-8")) & 0x7FFFFFFF


# --- admin/reporting predicates ---------------------------------------------------
# Every item is in exactly one of these three states, so the counts always sum to the
# table total. `annotator<>'auto'` alone would silently drop rows
# with a NULL annotator, because SQL evaluates NULL<>'auto' as NULL rather than true —
# hence the explicit IS NULL arm.
_AUTO_SQL = "verified=1 AND annotator='auto'"
_HUMAN_SQL = "verified=1 AND (annotator IS NULL OR annotator<>'auto')"
_REMAINING_SQL = "verified IS NULL OR verified<>1"

# The order a claimed batch is worked in, in ONE place: query() pages by position and
# claim_progress() reports a position, so the two drifting apart would send the pager to
# the wrong shape. By claim_seq first, so a later batch is appended after the work
# already in hand instead of scattering through it by uuid.
_BATCH_ORDER = "claim_seq ASC, uuid ASC"

# A row is served only while it is in the current filtered set. Rows dropped by a later
# filter keep their labels and stay readable through include_out_of_set=True; they are
# never deleted, and they are never handed to an annotator — their images may not exist.
# Every "how much of the dataset is X" count carries this; the who-did-what reports
# (annotator_stats, annotator_detail, daily_activity, views, export_dict) deliberately do
# not, because work somebody did is still work they did after the shape left the set.
_IN_SET_SQL = "in_set=1"

# Free to claim: in the current set, undecided, and not held by a live claim (anyone's,
# including your own — what you hold is already yours). Shared by claim_batch() and
# claimable_by_category() so the panel's "left to claim" counts are what a claim actually
# hands out.
_CLAIMABLE_SQL = ("(verified IS NULL OR verified<>1) "
                  "AND (claimed_by IS NULL OR claimed_at IS NULL OR claimed_at<?) "
                  "AND " + _IN_SET_SQL)

# The dashboard's shape browser. Statuses are built from the predicates above rather
# than re-spelled, so "show me the auto rows" selects exactly the rows the auto
# tile counts. `issue`/`good` cut the same table a different way (by decision, not by
# who made it) and so overlap the first three on purpose.
_BROWSE_STATUS_SQL = {
    "todo": _REMAINING_SQL,
    "human": _HUMAN_SQL,
    "auto": _AUTO_SQL,
    "issue": "emissive=0",
    "good": "emissive=1",
}
# Whitelisted because it is interpolated into the SQL. `ts DESC` puts NULLs last in
# SQLite, which is what we want: never-touched rows are not recent work. uuid breaks
# ties so a page boundary can't hand back the same shape twice.
_BROWSE_ORDER_SQL = {"uuid": "uuid ASC", "recent": "ts DESC, uuid ASC"}


def _like_escape(s):
    """Neutralize LIKE's wildcards for a literal substring search."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _as_category_list(category):
    """One category or several — the claim panel sends a list, older callers a string."""
    if not category:
        return []
    if isinstance(category, str):
        return [category]
    return [c for c in category if c]


def _display_name(name):
    """Label for an annotator value. '' and NULL are both real states in the DB."""
    if name is None:
        return "(unattributed)"
    if name == "":
        return "(blank)"
    return name


def _edit_distance(a, b):
    """Levenshtein distance. Only ever called on short annotator names."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_subsequence(short, long):
    it = iter(long)
    return all(ch in it for ch in short)


def _looks_like_same_person(a, b):
    """Advisory duplicate-name heuristic — never mutates anything.

    Catches two shapes: a typo/suffix variant ('test' vs 'test1') and an abbreviation
    (a short name that is neither a prefix nor within edit distance of the long one — but
    IS a subsequence of it).
    """
    x, y = a.strip().lower(), b.strip().lower()
    if not x or not y or x == y:
        return bool(x) and x == y
    if x.startswith(y) or y.startswith(x):
        return True
    short, long = (x, y) if len(x) <= len(y) else (y, x)
    if len(short) <= 3 and _is_subsequence(short, long):
        return True
    return len(short) >= 4 and _edit_distance(x, y) <= 2


# A live session heartbeats every 30s, so an 'active' row that hasn't been touched in
# this long was abandoned (tab closed, laptop slept) rather than genuinely running.
STALE_ACTIVE_MINUTES = 30


def _parse_ts(value):
    """Parse either ISO layout the app has written ('...+00:00' and '...Z'), or None."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _annotate_session(row, now=None):
    """Tag a session dict with `stale` so the UI can stop counting zombies as live."""
    row = dict(row)
    stale = False
    if row.get("status") == "active" and not row.get("ended_at"):
        ts = _parse_ts(row.get("ts")) or _parse_ts(row.get("started_at"))
        if ts is not None:
            reference = now or datetime.now(timezone.utc)
            if reference.tzinfo is None:      # a naive `now` from a caller would otherwise
                reference = reference.replace(tzinfo=timezone.utc)   # raise on subtraction
            stale = (reference - ts).total_seconds() > STALE_ACTIVE_MINUTES * 60
    row["stale"] = stale
    return row


def _flag_counts(c):
    """Flag histogram, parsed in Python.

    Deliberately not SQLite's json_each(): the server may run on a stock Python 3.8 build
    and we don't want to depend on JSON1 being compiled in. Only the flagged rows are read.
    """
    counts = {}
    for row in c.execute("SELECT flags FROM items WHERE flags IS NOT NULL AND flags<>'[]' "
                         "AND " + _IN_SET_SQL):
        try:
            vals = json.loads(row[0] or "[]")
        except ValueError:
            continue
        for f in vals:
            counts[f] = counts.get(f, 0) + 1
    return [{"flag": k, "count": v}
            for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


class AnnotationStore:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _conn_ro(self):
        """Connection for the admin dashboard's read paths — issues no statement at all.

        Deliberately skips `PRAGMA journal_mode=WAL`: that pragma is not a SELECT and
        would *convert* a database that wasn't already in WAL mode, which is a write to a
        file the caller believes it is only reading. WAL is a persistent property of the
        database header, so reads don't need it set per-connection.

        Not a `file:...?mode=ro` URI either — opening a WAL database strictly read-only
        fails when the -shm file is absent and can't be created, which is exactly the case
        for a freshly copied snapshot.
        """
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def init_schema(self):
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS items (
                    uuid TEXT PRIMARY KEY,
                    auto_category TEXT,
                    category TEXT,
                    emissive INTEGER,
                    flags TEXT DEFAULT '[]',
                    verified INTEGER DEFAULT 0,
                    annotator TEXT,
                    notes TEXT,
                    style TEXT,
                    emission_complete INTEGER,
                    ts TEXT
                )"""
            )
            cols = [r[1] for r in c.execute("PRAGMA table_info(items)").fetchall()]
            if "notes" not in cols:  # migrate DBs created before the notes column
                c.execute("ALTER TABLE items ADD COLUMN notes TEXT")
            if "style" not in cols:
                c.execute("ALTER TABLE items ADD COLUMN style TEXT")
            if "emission_complete" not in cols:
                c.execute("ALTER TABLE items ADD COLUMN emission_complete INTEGER")
            if "shard_key" not in cols:   # even split of work between annotators
                c.execute("ALTER TABLE items ADD COLUMN shard_key INTEGER")
            if "claimed_by" not in cols:  # batch claiming: who is working on this, since when
                c.execute("ALTER TABLE items ADD COLUMN claimed_by TEXT")
            if "claimed_at" not in cols:
                c.execute("ALTER TABLE items ADD COLUMN claimed_at TEXT")
            if "in_set" not in cols:     # membership in the CURRENT filtered set
                # Default 1 so an existing DB keeps serving everything until
                # set_membership() is run against the new list.
                c.execute("ALTER TABLE items ADD COLUMN in_set INTEGER DEFAULT 1")
            if "claim_seq" not in cols:   # which claim handed this out; orders a batch
                c.execute("ALTER TABLE items ADD COLUMN claim_seq INTEGER")
                # Number claims that predate the column by when they were made, so a
                # batch in flight keeps a sane order instead of interleaving with the
                # next one. Restricted to claimed rows, of which there are a handful.
                c.execute(
                    "UPDATE items SET claim_seq=(SELECT COUNT(DISTINCT i2.claimed_at) "
                    "FROM items i2 WHERE i2.claimed_at IS NOT NULL "
                    "AND i2.claimed_at<=items.claimed_at) "
                    "WHERE claimed_by IS NOT NULL AND claimed_at IS NOT NULL")
            c.execute("CREATE INDEX IF NOT EXISTS idx_claimed ON items(claimed_by)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_in_set ON items(in_set)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_category ON items(category)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_auto_category ON items(auto_category)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_verified ON items(verified)")
            c.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    annotator TEXT,
                    planned_minutes REAL,
                    started_at TEXT,
                    ended_at TEXT,
                    active_seconds INTEGER DEFAULT 0,
                    annotated_count INTEGER DEFAULT 0,
                    status TEXT,
                    ts TEXT
                )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_annotator "
                      "ON sessions(annotator)")
            # One row per visit to a shape (the tool shows one shape per page), so
            # effort can be read per shape and per person instead of per timed
            # session. Append-only: revisits add rows rather than overwriting, and
            # nothing here feeds the annotation state itself.
            c.execute(
                """CREATE TABLE IF NOT EXISTS views (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL,
                    annotator TEXT,
                    dwell_ms INTEGER NOT NULL,
                    saves INTEGER DEFAULT 0,
                    patch TEXT,
                    ts TEXT
                )"""
            )
            vcols = [r[1] for r in c.execute("PRAGMA table_info(views)").fetchall()]
            if "saves" not in vcols:   # migrate views created before decisions were tracked
                c.execute("ALTER TABLE views ADD COLUMN saves INTEGER DEFAULT 0")
            if "patch" not in vcols:
                c.execute("ALTER TABLE views ADD COLUMN patch TEXT")
            c.execute("CREATE INDEX IF NOT EXISTS idx_views_uuid ON views(uuid)")
            # Registered annotators, in join order. The index defines which shard of
            # the set a person gets; adding someone reshards only what nobody has
            # annotated yet, so a reshuffle can never lose or duplicate finished work.
            c.execute(
                """CREATE TABLE IF NOT EXISTS annotators (
                    name TEXT PRIMARY KEY,
                    idx INTEGER NOT NULL,
                    registered_at TEXT
                )"""
            )
            self._backfill_shard_keys(c)
            c.execute("CREATE INDEX IF NOT EXISTS idx_views_annotator ON views(annotator)")

    def _backfill_shard_keys(self, c):
        missing = [r[0] for r in c.execute("SELECT uuid FROM items WHERE shard_key IS NULL")]
        if missing:
            c.executemany("UPDATE items SET shard_key=? WHERE uuid=?",
                          [(shard_key(u), u) for u in missing])
        return len(missing)

    # A claim that goes untouched this long returns to the pool, so a batch taken by
    # someone who then disappears cannot strand work forever.
    CLAIM_TTL_HOURS = 48

    def _claim_cutoff(self):
        return (datetime.now(timezone.utc)
                - timedelta(hours=self.CLAIM_TTL_HOURS)).isoformat()

    def claim_batch(self, annotator, count, category=None):
        """Hand `annotator` up to `count` shapes nobody else is working on.

        Only shapes they do NOT already hold are counted. Re-selecting their own
        unfinished rows would make "claim 50 more" with 40 in hand add 10 and report
        50. Those rows have their TTL renewed instead, so the earliest shapes of a long
        sitting cannot lapse back to the pool while it is still being worked.

        Every call stamps its rows with one increasing `claim_seq` (see _BATCH_ORDER),
        so a later batch is appended after the work in hand rather than scattered
        through it. `category` takes one name or several. SQLite serialises writers, so
        two people claiming at once cannot be handed the same shape.
        """
        if not annotator or annotator == "auto":
            return {"claimed": 0, "category": category}
        try:
            count = max(0, int(count))
        except (TypeError, ValueError):
            return {"claimed": 0, "category": category}
        cats = _as_category_list(category)
        where = [_CLAIMABLE_SQL]
        params = [self._claim_cutoff()]
        if cats:
            where.append("auto_category IN (%s)" % ",".join("?" * len(cats)))
            params.extend(cats)
        now = _now()
        with self._conn() as c:
            # Renew what is still live before selecting, since those rows are excluded
            # from the select below. Claims that already lapsed are deliberately NOT
            # renewed — they belong to the pool again, and rescuing them here would let
            # someone who walked away for three days jump the queue for their own old
            # work. They stay eligible for the select like anyone else's.
            self._renew(c, annotator, now)
            rows = c.execute(
                f"SELECT uuid FROM items WHERE {' AND '.join(where)} LIMIT ?",
                params + [count],
            ).fetchall()
            seq = c.execute("SELECT COALESCE(MAX(claim_seq), 0) + 1 FROM items").fetchone()[0]
            c.executemany(
                "UPDATE items SET claimed_by=?, claimed_at=?, claim_seq=? WHERE uuid=?",
                [(annotator, now, seq, r[0]) for r in rows])
        return {"claimed": len(rows), "category": category}

    def _renew(self, c, annotator, now):
        """Push the TTL back on `annotator`'s still-live claims.

        Lapsed claims are left alone: they are the pool's again, and this must not take
        them back. Decided shapes ARE renewed, unlike everywhere else that skips them —
        they keep their place in the batch, so letting their claim lapse would shrink
        the batch and move every position after them, which is exactly the drift the
        stable ordering exists to prevent. Nothing is at risk in holding them: a decided
        shape fails the claimable test anyway, so no one else was ever going to get it.
        """
        cur = c.execute(
            "UPDATE items SET claimed_at=? WHERE claimed_by=? AND claimed_at>=?",
            (now, annotator, self._claim_cutoff()))
        return cur.rowcount

    def touch_claims(self, annotator):
        """Renew the caller's batch because they are demonstrably still working on it.

        Called on every save. Without it a batch worked across more than 48 hours lapses
        out from under the person annotating it: rows drop out of query() and
        claim_progress() mid-session, so the pager's list — and its positions — change
        underneath them. Returns how many rows were renewed.
        """
        if not annotator or annotator == "auto":
            return 0
        with self._conn() as c:
            return self._renew(c, annotator, _now())

    def claimable_by_category(self):
        """How much work is free to claim right now, per category.

        Shares claim_batch's own predicate (_CLAIMABLE_SQL), so the number the panel
        shows is the number a claim hands out. Every category is listed, including ones
        with nothing left — "0 left" tells an annotator it is finished, whereas dropping
        the row would just look like the category had gone missing.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT auto_category AS category, "
                "SUM(CASE WHEN " + _CLAIMABLE_SQL + " THEN 1 ELSE 0 END) AS claimable "
                "FROM items WHERE " + _IN_SET_SQL +
                " GROUP BY auto_category ORDER BY auto_category",
                (self._claim_cutoff(),),
            ).fetchall()
        return [{"category": r["category"], "claimable": r["claimable"]} for r in rows]

    def release_claims(self, annotator):
        with self._conn() as c:
            cur = c.execute(
                "UPDATE items SET claimed_by=NULL, claimed_at=NULL "
                "WHERE claimed_by=? AND (verified IS NULL OR verified<>1)", (annotator,))
        return cur.rowcount

    def claim_progress(self, annotator):
        """How the current batch is going: total claimed, done, still to do.

        `next_index` is where the pager should open: the position of the first
        undecided shape in the batch's stable uuid ordering (see query()). Decided
        shapes keep their place in a batch, so without this someone resuming a
        half-finished batch lands on shape 1 and pages past all their own work. A
        finished batch has no undecided shape and reports 0.

        Counts the in-set rows of the batch, because that is what query() hands the
        pager: if the list a filter narrowed and the count taken over the whole batch
        disagreed, every position after the first dropped shape would be off by one —
        the same drift the stable ordering exists to prevent.
        """
        if not annotator:
            return {"total": 0, "done": 0, "remaining": 0, "next_index": 0}
        cutoff = self._claim_cutoff()
        with self._conn() as c:
            r = c.execute(
                "SELECT COUNT(*) AS total, "
                "COALESCE(SUM(CASE WHEN verified=1 THEN 1 ELSE 0 END), 0) AS done "
                "FROM items WHERE claimed_by=? AND claimed_at>=? AND " + _IN_SET_SQL,
                (annotator, cutoff),
            ).fetchone()
            # Walk the batch in the very ordering query() pages through, so the index
            # reported here cannot drift from the one the pager will land on. A batch is
            # at most a couple of thousand rows, and a fully decided one falls through
            # the loop to 0.
            decided = c.execute(
                "SELECT verified FROM items WHERE claimed_by=? AND claimed_at>=? AND "
                + _IN_SET_SQL + " ORDER BY " + _BATCH_ORDER, (annotator, cutoff)).fetchall()
        out = dict(r)
        out["remaining"] = out["total"] - out["done"]
        out["next_index"] = next((i for i, row in enumerate(decided) if row[0] != 1), 0)
        return out

    def register_annotator(self, name):
        """Register a person and return their {idx, total}; None for non-people.

        Idempotent: an existing name keeps its index, so someone's shard does not
        move because they reopened the tool.
        """
        if not name or name == "auto":
            return None
        with self._conn() as c:
            row = c.execute("SELECT idx FROM annotators WHERE name=?", (name,)).fetchone()
            if row is None:
                nxt = c.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM annotators").fetchone()[0]
                c.execute("INSERT INTO annotators (name, idx, registered_at) VALUES (?,?,?)",
                          (name, nxt, _now()))
                idx = nxt
            else:
                idx = row[0]
            total = c.execute("SELECT COUNT(*) FROM annotators").fetchone()[0]
        return {"name": name, "idx": idx, "total": total}

    def list_annotators(self):
        with self._conn() as c:
            return [dict(r) for r in
                    c.execute("SELECT * FROM annotators ORDER BY idx")]

    def seed_from_manifest(self, rows):
        """Add rows the DB does not have yet. Never updates one it does.

        INSERT OR IGNORE, deliberately: a re-seed after the manifest's source changed
        adds the new uuids and leaves every existing row exactly as it is — label,
        annotator, notes, category, claim. A worked DB holds human decisions, so this
        has to stay a pure insert. `in_set` is left to its default
        (1) here and set exactly by set_membership().
        """
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO items(uuid, auto_category, category, flags, verified, shard_key) "
                "VALUES (?,?,?,'[]',0,?)",
                [(r["uuid"], r["auto_category"], r["auto_category"], shard_key(r["uuid"]))
                 for r in rows],
            )

    def set_membership(self, uuids):
        """Mark exactly `uuids` as in the current set; everything else out of it.

        Not a delete: a row that leaves the set keeps its human label, which is the
        record of a decision somebody made and is still worth reading."""
        with self._conn() as c:
            c.execute("CREATE TEMP TABLE IF NOT EXISTS _cur(uuid TEXT PRIMARY KEY)")
            c.execute("DELETE FROM _cur")
            c.executemany("INSERT OR IGNORE INTO _cur(uuid) VALUES (?)",
                          [(u,) for u in uuids])
            c.execute("UPDATE items SET in_set = "
                      "CASE WHEN uuid IN (SELECT uuid FROM _cur) THEN 1 ELSE 0 END")
            n_in = c.execute("SELECT COUNT(*) FROM items WHERE in_set=1").fetchone()[0]
            n_out = c.execute("SELECT COUNT(*) FROM items WHERE in_set=0").fetchone()[0]
        return {"in_set": n_in, "out_of_set": n_out}

    def _assignments(self, fields):
        """The SET clause shared by upsert() and upsert_unless_held_by_other()."""
        sets, vals = [], []
        for k in EDITABLE:
            if k in fields:
                v = fields[k]
                if k == "flags":
                    v = json.dumps(list(v))
                elif k in ("emissive", "verified", "emission_complete") and v is not None:
                    v = int(v)
                elif k == "style" and v is not None:
                    v = str(v) if v in STYLES else None
                sets.append(f"{k}=?")
                vals.append(v)
        sets.append("ts=?")
        vals.append(_now())
        return sets, vals

    def upsert(self, uuid, **fields):
        sets, vals = self._assignments(fields)
        vals.append(uuid)
        with self._conn() as c:
            c.execute(f"UPDATE items SET {','.join(sets)} WHERE uuid=?", vals)

    def upsert_unless_held_by_other(self, uuid, claimant, **fields):
        """upsert(), but refuse if a live claim on `uuid` belongs to someone other than
        `claimant`. Named `claimant`, not `annotator`, because `annotator` is itself one
        of the EDITABLE columns and would collide with **fields.

        The ownership test is part of the UPDATE's WHERE clause rather than a SELECT
        before it: `_conn()` hands out a fresh connection per call, so a check and a write
        issued separately are two transactions, and a save that passed the check could
        land after another annotator had claimed *and* decided the row — overwriting them.
        As one statement the test and the write cannot be interleaved.

        `claimed_by IS ?` (not `=`) so an anonymous save matches only a row whose claim is
        NULL; `= NULL` is never true and would refuse writes to unclaimed shapes.

        Returns the blocking holder's name if the write was refused, else None. An unknown
        uuid writes nothing and returns None, which is what plain upsert() does too.
        """
        sets, vals = self._assignments(fields)
        vals += [uuid, self._claim_cutoff(), claimant]
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE items SET {','.join(sets)} WHERE uuid=? AND "
                "(claimed_by IS NULL OR claimed_at IS NULL OR claimed_at<? "
                " OR claimed_by IS ?)", vals)
            if cur.rowcount:
                return None
            row = c.execute(
                "SELECT claimed_by FROM items WHERE uuid=? AND claimed_by IS NOT NULL "
                "AND claimed_at IS NOT NULL AND claimed_at>=?",
                (uuid, self._claim_cutoff())).fetchone()
        return row["claimed_by"] if row else None

    def mark_auto_zero_emission(self, uuids, annotator="auto", flag="zero_emission_map"):
        """Auto-annotate zero-emission shapes as has-issue+verified.

        Non-clobber: only rows currently verified!=1 (i.e., verified=0 or NULL) are updated,
        so any human-reviewed row is left exactly as-is. Idempotent — a re-run skips
        rows this method already marked (they are now verified=1).
        Returns {"marked", "skipped_reviewed", "not_in_db"}.
        """
        uuids = list(dict.fromkeys(uuids))  # dedupe, preserving order
        ts = _now()
        flags_json = json.dumps([flag])
        with self._conn() as c:
            verified_by = dict(c.execute("SELECT uuid, verified FROM items").fetchall())
            to_mark = [u for u in uuids if u in verified_by and verified_by[u] != 1]
            c.executemany(
                "UPDATE items SET emissive=0, verified=1, flags=?, annotator=?, ts=? "
                "WHERE uuid=? AND verified IS NOT 1",
                [(flags_json, annotator, ts, u) for u in to_mark],
            )
        return {
            "marked": len(to_mark),
            "skipped_reviewed": sum(1 for u in uuids if verified_by.get(u) == 1),
            "not_in_db": sum(1 for u in uuids if u not in verified_by),
        }

    def upsert_session(self, session_id, **fields):
        cols, vals = ["session_id"], [session_id]
        for k in SESSION_FIELDS:
            if k in fields:
                v = fields[k]
                if k in ("active_seconds", "annotated_count") and v is not None:
                    v = int(v)
                elif k == "planned_minutes" and v is not None:
                    v = float(v)
                elif k == "status" and v not in SESSION_STATUSES:
                    v = None
                cols.append(k)
                vals.append(v)
        cols.append("ts")
        vals.append(_now())
        placeholders = ",".join("?" for _ in cols)
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "session_id")
        with self._conn() as c:
            c.execute(
                f"INSERT INTO sessions ({','.join(cols)}) VALUES ({placeholders}) "
                f"ON CONFLICT(session_id) DO UPDATE SET {updates}",
                vals,
            )

    # Anything outside this range is a client bug, not effort: negatives come from
    # clock skew, and the ceiling is well past any real look at one shape — a tab
    # left open overnight would otherwise dominate every average.
    MIN_DWELL_MS = 100
    MAX_DWELL_MS = 30 * 60 * 1000

    def record_view(self, uuid, annotator, dwell_ms, saves=0, patch=None):
        """Append one visit. Silently drops implausible durations.

        `saves` counts the writes made during the visit (0 = browsed past) and
        `patch` is what those writes set, merged. Together they are the only
        record of how a decision was reached — `items` keeps just the final state.
        """
        try:
            dwell_ms = int(dwell_ms)
        except (TypeError, ValueError):
            return False
        if not uuid or not (self.MIN_DWELL_MS <= dwell_ms <= self.MAX_DWELL_MS):
            return False
        try:
            saves = max(0, int(saves))
        except (TypeError, ValueError):
            saves = 0
        blob = json.dumps(patch, sort_keys=True) if isinstance(patch, dict) and patch else None
        with self._conn() as c:
            c.execute(
                "INSERT INTO views (uuid, annotator, dwell_ms, saves, patch, ts) "
                "VALUES (?,?,?,?,?,?)",
                (uuid, annotator, dwell_ms, saves, blob, _now()),
            )
        return True

    def list_views(self, uuid=None, annotator=None, limit=500):
        where, params = [], []
        if uuid:
            where.append("uuid=?"); params.append(uuid)
        if annotator:
            where.append("annotator=?"); params.append(annotator)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM views{clause} ORDER BY id DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [dict(r) for r in rows]

    def view_stats(self):
        """Totals, split by whether the visit produced a decision.

        The split matters: most visits are someone paging past, so a mean over
        all visits describes scrolling speed, not annotation effort.
        """
        with self._conn() as c:
            r = c.execute(
                "SELECT COUNT(*) AS total_views, COALESCE(SUM(dwell_ms), 0) AS total_dwell_ms, "
                "COUNT(DISTINCT uuid) AS shapes_viewed, "
                "COALESCE(SUM(CASE WHEN saves > 0 THEN 1 ELSE 0 END), 0) AS decided_views, "
                "COALESCE(SUM(CASE WHEN saves > 0 THEN dwell_ms ELSE 0 END), 0) AS decided_dwell_ms "
                "FROM views"
            ).fetchone()
        return dict(r)

    def list_sessions(self, annotator=None, limit=500):
        where, params = "", []
        if annotator:
            where = " WHERE annotator=?"
            params.append(annotator)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM sessions{where} ORDER BY started_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        return [dict(r) for r in rows]

    def _row_to_dict(self, r):
        d = dict(r)
        d["flags"] = json.loads(d.get("flags") or "[]")
        return d

    def query(self, category=None, bucket=None, flag=None, verified=None, emissive=None,
              annotator=None, q=None, offset=0, limit=60, claimed_by=None,
              include_out_of_set=False):
        where, params = [], []
        if not include_out_of_set:
            where.append(_IN_SET_SQL)
        if claimed_by:
            where.append("claimed_by=? AND claimed_at>=?")
            params += [claimed_by, self._claim_cutoff()]
        if category:
            where.append("category=?"); params.append(category)
        if bucket:
            where.append("auto_category=?"); params.append(bucket)
        if flag:
            where.append("flags LIKE ?"); params.append(f'%"{flag}"%')
        if verified is not None:
            where.append("verified=?"); params.append(1 if verified else 0)
        if emissive is not None:
            where.append("emissive=?"); params.append(1 if emissive else 0)
        if annotator:
            where.append("annotator=?"); params.append(annotator)
        if q:
            where.append("uuid LIKE ?"); params.append(f"%{q}%")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        # A claimed batch is a fixed work list, so it uses the stable _BATCH_ORDER. The
        # default `verified ASC` lifts a shape out of the undecided run the instant it
        # is decided, sliding every later shape one place forward — and the pager
        # addresses shapes by position, so that shift skips the next one.
        # Browsing without a batch keeps to-do-first, which is the point of that view.
        order = _BATCH_ORDER if claimed_by else "verified ASC, uuid ASC"
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM items{clause}", params).fetchone()[0]
            rows = c.execute(
                f"SELECT * FROM items{clause} ORDER BY {order} LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
        return [self._row_to_dict(r) for r in rows], total

    def category_progress(self):
        """Per-bucket progress keyed by the stable original auto_category.

        Counts the current set only, like every other progress number: a bucket's total
        is what is left to be looked at, and a shape the filter dropped is never handed
        out, so counting it would make a finished bucket read as unfinished forever.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT auto_category AS category, COUNT(*) AS total, "
                "COALESCE(SUM(verified), 0) AS verified FROM items "
                "WHERE " + _IN_SET_SQL + " GROUP BY auto_category ORDER BY auto_category"
            ).fetchall()
        return [
            {"category": r["category"], "total": r["total"], "verified": r["verified"]}
            for r in rows
        ]

    def progress(self):
        """The tool's header numbers, over the current set (see _IN_SET_SQL)."""
        with self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) FROM items WHERE " + _IN_SET_SQL).fetchone()[0]
            verified = c.execute(
                "SELECT COUNT(*) FROM items WHERE verified=1 AND " + _IN_SET_SQL).fetchone()[0]
            emis = c.execute(
                "SELECT COUNT(*) FROM items WHERE emissive=1 AND " + _IN_SET_SQL).fetchone()[0]
        return {"total": total, "verified": verified, "emissive_true": emis}

    def export_dict(self):
        # The export is the record of the whole table, dropped rows included: a label
        # somebody wrote must not fall out of the backup because the filter moved on.
        rows, _ = self.query(limit=10_000_000, include_out_of_set=True)
        return {"items": rows, "views": self.list_views(limit=10_000_000),
                "sessions": self.list_sessions(limit=10_000_000),
                "exported_at": _now()}

    # --- admin dashboard -----------------------------------------------------------
    # Read-only: every method below goes through _conn_ro() and issues SELECTs only —
    # not even a pragma. test_admin_reads_never_write_to_the_database traces the SQL and
    # fails if that ever stops being true.

    def admin_overview(self):
        """Dataset-wide progress, with the human pass separated from the auto-filter.

        The zero-emission auto-filter marks shapes verified without a human ever seeing
        them. Collapsing that into one "verified" number makes the human pass look much
        further along than it is, so the three states are always reported separately and
        never pre-summed.

        Every count here is over the current set (_IN_SET_SQL), which keeps the three
        states summing to `total` and keeps `remaining` a number that can reach zero —
        rows the filter dropped are never handed out, so counting them would park the
        dashboard permanently short of done, and disagree with the claim panel, which
        counts what a claim actually gives you. `annotator_stats` and `daily_activity`
        below stay table-wide: those report effort, not how much of the set is left.
        """
        with self._conn_ro() as c:
            def count(where):
                return c.execute("SELECT COUNT(*) FROM items WHERE (" + where + ") AND "
                                 + _IN_SET_SQL).fetchone()[0]

            total = c.execute(
                "SELECT COUNT(*) FROM items WHERE " + _IN_SET_SQL).fetchone()[0]
            emissive = self._group_counts(c, "emissive")
            style = self._group_counts(c, "style")
            complete = self._group_counts(c, "emission_complete")
            sessions_total = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            active_seconds = c.execute(
                "SELECT COALESCE(SUM(active_seconds), 0) FROM sessions "
                "WHERE annotator IS NULL OR annotator<>'auto'").fetchone()[0]
            session_items = c.execute(
                "SELECT COALESCE(SUM(annotated_count), 0) FROM sessions "
                "WHERE annotator IS NULL OR annotator<>'auto'").fetchone()[0]
            out = {
                "total": total,
                "human_verified": count(_HUMAN_SQL),
                "auto_verified": count(_AUTO_SQL),
                "remaining": count("(" + _REMAINING_SQL + ")"),
                "emissive": {"good": emissive.get(1, 0), "issue": emissive.get(0, 0),
                             "todo": emissive.get(None, 0)},
                "style": {"artistic": style.get("artistic", 0), "real": style.get("real", 0),
                          "unset": style.get(None, 0)},
                "emission_complete": {"complete": complete.get(1, 0),
                                      "incomplete": complete.get(0, 0),
                                      "unset": complete.get(None, 0)},
                "sessions": sessions_total,
                "active_seconds": active_seconds,
                "session_items": session_items,
                "flags": _flag_counts(c),
            }
        return out

    def _group_counts(self, c, column):
        """value -> count for one column over the current set, NULL as the None key."""
        rows = c.execute(
            "SELECT {0} AS v, COUNT(*) AS n FROM items WHERE {1} GROUP BY {0}".format(
                column, _IN_SET_SQL)).fetchall()
        return dict((r["v"], r["n"]) for r in rows)

    def annotator_stats(self):
        """Per-annotator rollup over items, merged with their session totals.

        Names are reported raw (no aliasing) — a DB can hold 'test' and 'test1', an
        abbreviation and its full name, an empty string and NULL, and merging them is a
        judgment call for a human, not this query. Likely duplicates are flagged advisorily.
        """
        with self._conn_ro() as c:
            item_rows = c.execute(
                "SELECT annotator AS name, COUNT(*) AS items, "
                "SUM(CASE WHEN verified=1 THEN 1 ELSE 0 END) AS verified, "
                "SUM(CASE WHEN emissive=1 THEN 1 ELSE 0 END) AS good, "
                "SUM(CASE WHEN emissive=0 THEN 1 ELSE 0 END) AS issue, "
                "MIN(ts) AS first_ts, MAX(ts) AS last_ts "
                "FROM items WHERE annotator IS NOT NULL OR verified=1 "
                "GROUP BY annotator").fetchall()
            sess_rows = c.execute(
                "SELECT annotator AS name, COUNT(*) AS sessions, "
                "COALESCE(SUM(active_seconds), 0) AS active_seconds, "
                "COALESCE(SUM(annotated_count), 0) AS session_items "
                "FROM sessions GROUP BY annotator").fetchall()

        by = {}

        def blank(name):
            return {"annotator": name, "display": _display_name(name), "items": 0,
                    "verified": 0, "good": 0, "issue": 0, "first_ts": None, "last_ts": None,
                    "sessions": 0, "active_seconds": 0, "session_items": 0}

        for r in item_rows:
            d = by.setdefault(r["name"], blank(r["name"]))
            d.update(items=r["items"], verified=r["verified"], good=r["good"],
                     issue=r["issue"], first_ts=r["first_ts"], last_ts=r["last_ts"])
        for r in sess_rows:
            d = by.setdefault(r["name"], blank(r["name"]))
            d.update(sessions=r["sessions"], active_seconds=r["active_seconds"],
                     session_items=r["session_items"])

        out = []
        for d in by.values():
            d["is_auto"] = d["annotator"] == "auto"
            hours = d["active_seconds"] / 3600.0
            d["items_per_hour"] = round(d["items"] / hours, 1) if hours > 0 else None
            # Items attributed but no session at all: the signature of a mid-session
            # rename (the session row keeps the old name, so the items land on a name
            # that owns no session row).
            d["no_sessions"] = (not d["is_auto"]) and d["items"] > 0 and d["sessions"] == 0
            out.append(d)

        names = [d["annotator"] for d in out
                 if d["annotator"] and not d["is_auto"]]
        for d in out:
            dupes = []
            if d["annotator"] and not d["is_auto"]:
                dupes = [n for n in names
                         if n != d["annotator"] and _looks_like_same_person(d["annotator"], n)]
            d["possible_duplicates"] = sorted(dupes)

        out.sort(key=lambda d: (-d["items"], d["display"]))
        return out

    def annotator_detail(self, annotator, offset=0, limit=50):
        """One annotator's sessions, a page of their items, and per-day item counts.

        Matched exactly, including the empty string — a real value in the DB (rows
        saved before the name gate existed). None matches a NULL annotator. Neither case
        can go through query()/list_sessions(), which treat a falsy annotator as
        "no filter at all" and would silently return the whole table.
        """
        if annotator is None:
            pred, params = "annotator IS NULL", []
        else:
            pred, params = "annotator=?", [annotator]
        with self._conn_ro() as c:
            total = c.execute(
                "SELECT COUNT(*) FROM items WHERE " + pred, params).fetchone()[0]
            rows = c.execute(
                "SELECT * FROM items WHERE " + pred + " ORDER BY ts DESC, uuid ASC "
                "LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
            sessions = c.execute(
                "SELECT * FROM sessions WHERE " + pred + " ORDER BY started_at DESC "
                "LIMIT 500", params).fetchall()
            daily = c.execute(
                "SELECT substr(ts, 1, 10) AS day, COUNT(*) AS n FROM items "
                "WHERE ts IS NOT NULL AND " + pred + " GROUP BY day ORDER BY day",
                params).fetchall()
        return {
            "annotator": annotator,
            "display": _display_name(annotator),
            "total": total, "offset": offset, "limit": limit,
            "items": [self._row_to_dict(r) for r in rows],
            "sessions": [_annotate_session(dict(s)) for s in sessions],
            "daily": [{"day": r["day"], "count": r["n"]} for r in daily],
        }

    def admin_browse(self, status=None, bucket=None, flag=None, annotator=None,
                     unattributed=False, q=None, claimed=False, order="uuid",
                     offset=0, limit=48, include_out_of_set=False):
        """A filtered page of the whole set, for the dashboard's shape browser.

        Deliberately NOT query(): that one goes through _conn() (which runs a
        journal-mode pragma, so it is not a read) and it orders by `verified ASC`,
        which reshuffles as work lands. Nothing here writes and the order is stable.

        `status` reuses the same three predicates the progress tiles are built from,
        so a browse count can never disagree with the number displayed above it.
        `annotator` is matched exactly — including the empty string, a real name in
        the DB — so the caller passes `unattributed=True` for NULL rather than a
        falsy name that every other query path reads as "no filter".

        Shows the current set by default, so a browse total still matches the tile above
        it. `include_out_of_set=True` is the way to read the rows a later filter dropped:
        they keep their labels and are never deleted, and this is the only surface that
        can show them.
        """
        where, params = [], []
        if not include_out_of_set:
            where.append(_IN_SET_SQL)
        if status and status != "all":
            pred = _BROWSE_STATUS_SQL.get(status)
            if pred is None:
                raise ValueError("unknown status: %r" % (status,))
            where.append("(" + pred + ")")
        if bucket:
            where.append("auto_category=?"); params.append(bucket)
        if flag:
            where.append("flags LIKE ?"); params.append(f'%"{flag}"%')
        if unattributed:
            where.append("annotator IS NULL")
        elif annotator is not None:
            where.append("annotator=?"); params.append(annotator)
        if q:
            # The box searches for a substring, so a typed % or _ is a character to
            # find and not a wildcard — unescaped, a stray '%' silently returns the
            # whole set and reads as "the filter did nothing".
            where.append("uuid LIKE ? ESCAPE '\\'")
            params.append("%" + _like_escape(q) + "%")
        if claimed:
            where.append("claimed_by IS NOT NULL AND claimed_at>=?")
            params.append(self._claim_cutoff())
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        sort = _BROWSE_ORDER_SQL.get(order or "uuid")
        if sort is None:
            raise ValueError("unknown order: %r" % (order,))
        with self._conn_ro() as c:
            total = c.execute(f"SELECT COUNT(*) FROM items{clause}", params).fetchone()[0]
            rows = c.execute(
                f"SELECT * FROM items{clause} ORDER BY {sort} LIMIT ? OFFSET ?",
                params + [limit, offset]).fetchall()
        return {"items": [self._row_to_dict(r) for r in rows], "total": total,
                "offset": offset, "limit": limit}

    def flag_counts(self):
        with self._conn_ro() as c:
            return _flag_counts(c)

    def category_progress_split(self):
        """category_progress(), but with the human pass separated from the auto-filter."""
        with self._conn_ro() as c:
            rows = c.execute(
                "SELECT auto_category AS category, COUNT(*) AS total, "
                "SUM(CASE WHEN " + _AUTO_SQL + " THEN 1 ELSE 0 END) AS auto_verified, "
                "SUM(CASE WHEN " + _HUMAN_SQL + " THEN 1 ELSE 0 END) AS human_verified "
                "FROM items WHERE " + _IN_SET_SQL +
                " GROUP BY auto_category ORDER BY auto_category").fetchall()
        out = []
        for r in rows:
            out.append({
                "category": r["category"], "total": r["total"],
                "auto_verified": r["auto_verified"], "human_verified": r["human_verified"],
                "remaining": r["total"] - r["auto_verified"] - r["human_verified"],
            })
        return out

    def daily_activity(self, days=60):
        """Items touched per calendar day, split human vs auto.

        Uses substr(ts,1,10) rather than SQLite's date(): ts is written in two different
        ISO layouts over the app's history, and both start with YYYY-MM-DD.
        """
        with self._conn_ro() as c:
            rows = c.execute(
                "SELECT substr(ts, 1, 10) AS day, "
                "SUM(CASE WHEN annotator='auto' THEN 1 ELSE 0 END) AS auto, "
                "SUM(CASE WHEN annotator IS NULL OR annotator<>'auto' THEN 1 ELSE 0 END) "
                "AS human FROM items WHERE ts IS NOT NULL "
                "GROUP BY day ORDER BY day DESC LIMIT ?", [days]).fetchall()
        out = [{"day": r["day"], "auto": r["auto"], "human": r["human"]} for r in rows]
        out.reverse()
        return out

    def recent_sessions(self, limit=50):
        # Own query rather than list_sessions(), which goes through _conn() and would
        # reintroduce the journal-mode pragma on an otherwise read-only path.
        with self._conn_ro() as c:
            rows = c.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", [limit]).fetchall()
        return [_annotate_session(dict(r)) for r in rows]
