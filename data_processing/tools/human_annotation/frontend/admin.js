// Admin dashboard for the annotation tool. Read-only: it issues GETs and never writes.
// Vanilla + relative URLs, same as app.js, so it works unchanged behind a reverse proxy
// at any path (see <base href="./">).

// The page is reached as /admin?key=<token>. Stash the token for the API calls (which
// send it as a header instead) and scrub it out of the address bar so it isn't left
// sitting in history or copy-pasted into a chat along with the URL.
const TOKEN = (() => {
  const fromUrl = new URLSearchParams(location.search).get("key");
  if (fromUrl) {
    sessionStorage.setItem("ann74k_admin_token", fromUrl);
    // Keep the hash — it selects an annotator, and dropping it here would silently
    // break every deep link that also carries the token.
    history.replaceState(null, "", location.pathname + location.hash);
  }
  return sessionStorage.getItem("ann74k_admin_token") || "";
})();

const el = (id) => document.getElementById(id);
const fmt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US"));
const pct = (n, total) => (total ? (100 * n / total) : 0);
const pctText = (n, total) => pct(n, total).toFixed(pct(n, total) < 1 && n > 0 ? 2 : 1) + "%";

function dur(seconds) {
  if (!seconds) return "–";
  const h = Math.floor(seconds / 3600), m = Math.round((seconds % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

function shortDate(ts) {
  if (!ts) return "–";
  const d = new Date(ts);
  return isNaN(d) ? String(ts).slice(0, 10)
    : d.toLocaleDateString("en-CA") + " " + d.toTimeString().slice(0, 5);
}

async function api(path) {
  const r = await fetch("api/admin/" + path, { headers: { "X-Admin-Token": TOKEN } });
  if (!r.ok) {
    // The status is carried so callers can tell "this route does not exist yet" from a
    // real failure: the frontend is served no-cache and goes live the moment it is
    // rsynced, while a new backend route only exists after the server is restarted.
    const err = new Error(`${path} → HTTP ${r.status}`);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

function fail(msg) {
  const box = el("error");
  box.hidden = false;
  box.textContent = msg + " — if this persists, reopen the page with ?key=<token>.";
}

// --- tooltip ---------------------------------------------------------------------
const tip = el("tip");
function hoverable(node, text) {
  node.addEventListener("mouseenter", () => { tip.textContent = text; tip.hidden = false; });
  node.addEventListener("mousemove", (e) => {
    tip.style.left = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (e.clientY + 16) + "px";
  });
  node.addEventListener("mouseleave", () => { tip.hidden = true; });
  return node;
}

function div(cls, text) {
  const d = document.createElement("div");
  if (cls) d.className = cls;
  if (text != null) d.textContent = text;
  return d;
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// --- deep links -------------------------------------------------------------------
// The hash carries two independent selections now — which annotator is open, and how the
// shape browser is filtered — so it is read and written as URL params rather than as one
// opaque string. Old links still work: `#annotator=xg` parses as the same key, and the
// bare `#unattributed` parses as that key with an empty value.
function hashParams() {
  return new URLSearchParams(location.hash.replace(/^#/, ""));
}

function writeHash(p) {
  const s = p.toString();
  // replaceState, not `location.hash =`: filtering the browser would otherwise stack up
  // a history entry per keystroke and make Back useless.
  history.replaceState(null, "", location.pathname + (s ? "#" + s : ""));
}

function setHashParams(patch) {
  const p = hashParams();
  for (const [k, v] of Object.entries(patch)) {
    if (v == null) p.delete(k); else p.set(k, v);
  }
  writeHash(p);
}

// --- overall progress -------------------------------------------------------------
function renderTiles(o) {
  const tiles = el("tiles");
  tiles.innerHTML = "";
  const spec = [
    ["human", "Reviewed by a person", o.human_verified, pctText(o.human_verified, o.total)],
    ["auto", "Auto-filtered", o.auto_verified, pctText(o.auto_verified, o.total)],
    ["", "Not yet reviewed", o.remaining, pctText(o.remaining, o.total)],
    ["", "Shapes in total", o.total, "the full cohort"],
  ];
  for (const [cls, label, value, sub] of spec) {
    const t = div("tile " + cls);
    t.append(div("tile-label", label), div("tile-value", fmt(value)), div("tile-sub", sub));
    tiles.appendChild(t);
  }
}

function stackBar(segments, total, opts) {
  // segments: [{cls, label, value}]. Every segment is direct-labelled when it has room,
  // and repeated in the legend below — the amber fill sits just under 3:1 on white, so
  // color alone is never the only way to read a number here.
  const compact = opts && opts.compact;
  const stack = div("stack");
  for (const s of segments) {
    if (!s.value) continue;
    const seg = div("seg " + s.cls);
    seg.style.flex = `0 1 ${pct(s.value, total)}%`;
    const share = pct(s.value, total);
    if (!compact && share > 7) seg.textContent = fmt(s.value);
    hoverable(seg, `${s.label}: ${fmt(s.value)} (${pctText(s.value, total)})`);
    stack.appendChild(seg);
  }
  if (compact) return stack;

  const wrap = div("");
  const legend = div("legend");
  for (const s of segments) {
    const item = div("legend-item");
    const sw = div("swatch " + s.cls);
    const num = div("legend-num", fmt(s.value));
    item.append(sw, document.createTextNode(s.label), num,
      document.createTextNode(`(${pctText(s.value, total)})`));
    legend.appendChild(item);
  }
  wrap.append(stack, legend);
  return wrap;
}

function renderOverall(o) {
  const host = el("overall-bar");
  host.innerHTML = "";
  host.appendChild(stackBar([
    { cls: "human", label: "reviewed by a person", value: o.human_verified },
    { cls: "auto", label: "auto-filtered", value: o.auto_verified },
    { cls: "remaining", label: "not yet reviewed", value: o.remaining },
  ], o.total));
}

function renderProjection(o) {
  const host = el("projection");
  host.innerHTML = "";
  const p = o.projection || {};
  if (p.items_per_hour == null) {
    host.appendChild(div("empty", "No completed sessions yet — nothing to estimate from."));
    return;
  }
  const row = div("proj");
  row.append(
    div("proj-value", fmt(Math.round(p.remaining_hours))),
    div("proj-unit", `more hours of annotation to finish the remaining ${fmt(o.remaining)} shapes`));
  const caveat = div("proj-caveat");
  caveat.textContent =
    `Based on ${fmt(p.sample_items)} items annotated across ${dur(p.sample_seconds)} of ` +
    `active session time (${p.items_per_hour} items/hour). That is a small sample, and it ` +
    `includes sessions recorded under placeholder names — treat this as an order of ` +
    `magnitude, not a deadline.`;
  host.append(row, caveat);
}

function hbars(host, rows, total) {
  host.innerHTML = "";
  if (!rows.length) { host.appendChild(div("empty", "Nothing recorded yet.")); return; }
  const max = Math.max(total || 0, ...rows.map((r) => r.value)) || 1;
  for (const r of rows) {
    const line = div("hbar");
    const track = div("hbar-track");
    const fill = div("hbar-fill " + (r.cls || ""));
    fill.style.width = pct(r.value, max) + "%";
    track.appendChild(fill);
    hoverable(track, `${r.label}: ${fmt(r.value)}`);
    line.append(div("hbar-label", r.label), track, div("hbar-value", fmt(r.value)));
    host.appendChild(line);
  }
}

function renderDecisions(o) {
  const e = o.emissive;
  // Scaled to the largest row, not to the full set — against the full total every
  // decision bar collapses to a sliver and the block says nothing.
  hbars(el("decisions"), [
    { label: "✓ no issue", value: e.good, cls: "good" },
    { label: "✗ has issue", value: e.issue, cls: "issue" },
    { label: "not decided", value: e.todo, cls: "todo" },
  ]);
  hbars(el("flags"), (o.flags || []).map((f) => ({
    label: f.flag.replace(/_/g, " "),
    value: f.count,
    cls: f.flag === "zero_emission_map" ? "auto" : "issue",
  })));
}

// --- annotators -------------------------------------------------------------------
let annotators = [];
let sort = { key: "items", asc: false };
let selected = null;

function sortRows(rows) {
  const k = sort.key;
  return rows.slice().sort((a, b) => {
    let x = a[k], y = b[k];
    if (x == null && y == null) return 0;
    if (x == null) return 1;          // nulls (e.g. no items/hr) always sort last
    if (y == null) return -1;
    if (typeof x === "string") { const c = x.localeCompare(y); return sort.asc ? c : -c; }
    return sort.asc ? x - y : y - x;
  });
}

function renderAnnotators() {
  const tbody = document.querySelector("#annotators tbody");
  tbody.innerHTML = "";
  for (const a of sortRows(annotators)) {
    const tr = document.createElement("tr");
    if (a.is_auto) tr.className = "is-auto";
    else { tr.className = "clickable"; tr.onclick = () => selectAnnotator(a); }
    if (selected && selected.annotator === a.annotator && !a.is_auto) tr.classList.add("selected");

    const name = document.createElement("td");
    name.appendChild(document.createTextNode(a.display));
    if (a.is_auto) name.appendChild(badge("auto", "machine", "Marked by the zero-emission auto-filter, not a person"));
    if (a.no_sessions) {
      name.appendChild(badge("warn", "no session",
        "Items are attributed to this name but it owns no session row — usually the result of " +
        "renaming mid-session, which logged the session under the previous name."));
    }
    if (a.possible_duplicates && a.possible_duplicates.length) {
      name.appendChild(badge("dup", "≈ " + a.possible_duplicates.join(", "),
        "Looks like it could be the same person as: " + a.possible_duplicates.join(", ") +
        ". Nothing has been merged — this is only a hint."));
    }

    tr.appendChild(name);
    for (const v of [fmt(a.items), fmt(a.good), fmt(a.issue), fmt(a.sessions)]) {
      const td = document.createElement("td"); td.className = "num"; td.textContent = v;
      tr.appendChild(td);
    }
    const act = document.createElement("td"); act.className = "num";
    act.textContent = dur(a.active_seconds); tr.appendChild(act);
    const rate = document.createElement("td"); rate.className = "num";
    rate.textContent = a.items_per_hour == null ? "–" : a.items_per_hour; tr.appendChild(rate);
    const last = document.createElement("td"); last.className = "dim";
    last.textContent = shortDate(a.last_ts); tr.appendChild(last);
    tbody.appendChild(tr);
  }
}

function badge(cls, text, title) {
  const b = document.createElement("span");
  b.className = "badge " + cls;
  b.textContent = text;
  if (title) b.title = title;
  return b;
}

document.querySelectorAll("#annotators th[data-sort]").forEach((th) => {
  th.onclick = () => {
    const k = th.dataset.sort;
    sort = { key: k, asc: sort.key === k ? !sort.asc : (k === "display") };
    document.querySelectorAll("#annotators th").forEach((h) => h.classList.remove("sorted", "asc"));
    th.classList.add("sorted");
    if (sort.asc) th.classList.add("asc");
    renderAnnotators();
  };
});

// --- annotator detail -------------------------------------------------------------
async function selectAnnotator(a, offset) {
  selected = a;
  // Deep-linkable so one person's page can be shared. The token is not in the hash —
  // whoever opens it still needs their own ?key=. The browser's own filters live in the
  // same hash, so only this pair of keys is touched.
  setHashParams(a.annotator == null
    ? { unattributed: "", annotator: null }
    : { annotator: a.annotator, unattributed: null });
  renderAnnotators();
  el("detail-card").hidden = false;
  el("detail-name").textContent = a.display;
  const host = el("detail");
  host.textContent = "loading…";
  // An empty-string name is a real annotator; ?name= must survive as a blank value, and
  // a NULL annotator (the untouched backlog) has its own selector.
  const q = a.annotator == null
    ? "unattributed=1"
    : "name=" + encodeURIComponent(a.annotator);
  try {
    const d = await api(`annotator?${q}&offset=${offset || 0}&limit=25`);
    renderDetail(a, d);
  } catch (e) {
    host.textContent = "";
    fail("Could not load annotator detail (" + e.message + ")");
  }
}

function renderDetail(a, d) {
  const host = el("detail");
  host.innerHTML = "";

  host.appendChild(div("detail-sub", `Sessions (${d.sessions.length})`));
  if (!d.sessions.length) {
    host.appendChild(div("empty",
      "No sessions recorded for this name. Their items were logged under a different name's session."));
  } else {
    host.appendChild(table(
      ["Started", "Planned", "Active", "Items", "Status"],
      d.sessions.map((s) => [
        shortDate(s.started_at),
        s.planned_minutes ? s.planned_minutes + " min" : "–",
        dur(s.active_seconds),
        fmt(s.annotated_count),
        s.stale ? [s.status, badge("stale", "abandoned",
          "Still marked active but hasn't checked in for a long time — the tab was closed " +
          "rather than the session being finished.")] : s.status,
      ]), [false, false, true, true, false]));
  }

  host.appendChild(div("detail-sub", `Items (${fmt(d.total)})`));
  if (!d.items.length) {
    host.appendChild(div("empty", "No items."));
  } else {
    host.appendChild(table(
      ["Shape", "Category", "Outcome", "Reasons", "Note", "When"],
      d.items.map((it) => [
        mono(it.uuid),
        it.category || "–",
        it.emissive === 1 ? "✓ no issue" : it.emissive === 0 ? "✗ has issue" : "–",
        (it.flags || []).join(", ").replace(/_/g, " ") || "–",
        it.notes || "–",
        shortDate(it.ts),
      ])));
    const pager = div("pager-row");
    const prev = button("← previous", d.offset <= 0,
      () => selectAnnotator(a, Math.max(0, d.offset - d.limit)));
    const next = button("next →", d.offset + d.limit >= d.total,
      () => selectAnnotator(a, d.offset + d.limit));
    pager.append(prev, div("dim",
      `${fmt(d.offset + 1)}–${fmt(Math.min(d.offset + d.limit, d.total))} of ${fmt(d.total)}`), next);
    host.appendChild(pager);
  }
}

function button(label, disabled, onclick) {
  const b = document.createElement("button");
  b.textContent = label;
  b.disabled = !!disabled;
  b.onclick = onclick;
  return b;
}

function mono(text) {
  const s = document.createElement("span");
  s.className = "mono";
  s.textContent = text;
  return s;
}

function table(headers, rows, numeric) {
  const t = document.createElement("table");
  t.className = "adm-table";
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  headers.forEach((h, i) => {
    const th = document.createElement("th");
    th.textContent = h;
    if (numeric && numeric[i]) th.className = "num";
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    row.forEach((cell, i) => {
      const td = document.createElement("td");
      if (numeric && numeric[i]) td.className = "num";
      if (Array.isArray(cell)) cell.forEach((c) =>
        td.append(typeof c === "string" ? document.createTextNode(c + " ") : c));
      else if (cell instanceof Node) td.appendChild(cell);
      else td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  t.append(thead, tbody);
  const wrap = div("adm-scroll");
  wrap.appendChild(t);
  return wrap;
}

// --- activity ---------------------------------------------------------------------
function renderActivity(days) {
  const host = el("activity");
  host.innerHTML = "";
  if (!days.length) { host.appendChild(div("empty", "No dated activity yet.")); return; }
  // Two charts rather than one stacked chart with a shared axis: the auto-filter's bulk
  // day is orders of magnitude larger than any human day, and a shared scale would
  // render every human bar as a single invisible pixel.
  host.appendChild(chart("Reviewed by a person", days, "human", "human"));
  host.appendChild(chart("Auto-filtered", days, "auto", "auto"));
}

function chart(title, days, key, cls) {
  const max = Math.max(...days.map((d) => d[key])) || 1;
  const c = div("chart");
  c.append(div("chart-title", title), div("chart-max", `peak ${fmt(max)} on one day`));
  const bars = div("bars");
  for (const d of days) {
    const b = div("bar " + cls);
    b.style.height = Math.max(1, Math.round(100 * d[key] / max)) + "%";
    if (!d[key]) b.style.opacity = ".15";
    hoverable(b, `${d.day}: ${fmt(d[key])} items`);
    bars.appendChild(b);
  }
  c.appendChild(bars);
  const axis = div("chart-axis");
  axis.append(div("", days[0].day), div("", days[days.length - 1].day));
  c.appendChild(axis);
  return c;
}

// --- categories -------------------------------------------------------------------
function renderCategories(cats) {
  const tbody = document.querySelector("#categories tbody");
  tbody.innerHTML = "";
  // Most-remaining first: the question this table answers is "where is the work left",
  // not "what does the alphabet say".
  for (const c of cats.slice().sort((a, b) => b.remaining - a.remaining)) {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = c.category == null ? "(none)" : c.category;
    tr.appendChild(name);
    for (const v of [c.total, c.human_verified, c.auto_verified, c.remaining]) {
      const td = document.createElement("td");
      td.className = "num";
      td.textContent = fmt(v);
      tr.appendChild(td);
    }
    const barCell = document.createElement("td");
    barCell.className = "bar-col";
    barCell.appendChild(stackBar([
      { cls: "human", label: "reviewed by a person", value: c.human_verified },
      { cls: "auto", label: "auto-filtered", value: c.auto_verified },
      { cls: "remaining", label: "not yet reviewed", value: c.remaining },
    ], c.total, { compact: true }));
    tr.appendChild(barCell);
    tbody.appendChild(tr);
  }
}

function openFromHash() {
  const p = hashParams();
  let want;
  if (p.has("unattributed")) want = null;
  else if (p.has("annotator")) want = p.get("annotator");
  else return;
  const match = annotators.find((a) => a.annotator === want && !a.is_auto);
  if (match) selectAnnotator(match);
}

// --- browse all shapes ------------------------------------------------------------
// The annotator UI shows nobody more than their own claimed batch on purpose: a
// browse-everything view there would hand unclaimed shapes to whoever looked, which is
// exactly what claiming exists to prevent. The dashboard is token-gated and read-only,
// so the whole-cohort view lives here instead.

// One image per card: [key, label, url, aspect ratio]. Version tags mirror app.js and
// are not decoration — images are served `immutable`, so a re-burnt render under the
// same uuid would keep showing the browser's old copy without them.
// The ratio is carried per view because the sources genuinely differ: thumbnails are
// 1920x1080 Sketchfab-style previews while every render and the emission atlas is 512
// square. One shared square slot letterboxed the default view down to about half the
// card height.
const BROWSE_VIEWS = [
  ["thumb", "thumbnail", (u) => `img/thumb/${u}`, "16 / 9"],
  ["glb", "rendered object", (u) => `img/render/glb/${u}`, "1 / 1"],
  ["brighter", "emissive-brightened", (u) => `img/render/brighter/${u}`, "1 / 1"],
  ["box", "dark-box (emission on)", (u) => `img/render/box/${u}?v=filmic`, "1 / 1"],
  ["boxoff", "dark-box (emission off)", (u) => `img/render/boxoff/${u}?v=filmic`, "1 / 1"],
  ["emit", "emit-only rendering", (u) => `img/render/emit/${u}`, "1 / 1"],
  // ?v=73432 must match app.js: the emission map moved to the texture-alone bake and
  // the browser caches these `immutable`. Change the two together.
  ["emission", "emission map · uv", (u) => `img/emission/${u}?v=73432`, "1 / 1"],
];
// The lightbox shows everything, in the annotator card's own order — object, brightened,
// then the dark-box pair off-before-on so the eye reads the difference as what the
// emission adds.
const LIGHTBOX_VIEWS = ["glb", "brighter", "boxoff", "box", "emit", "emission"];
const BROWSE_STATUSES = [
  ["all", "everything"],
  ["todo", "not yet reviewed"],
  ["human", "reviewed by a person"],
  ["auto", "auto-filtered"],
  ["issue", "✗ has issue"],
  ["good", "✓ no issue"],
];
const viewSpec = (key) => BROWSE_VIEWS.find((v) => v[0] === key) || BROWSE_VIEWS[0];

const browse = {
  // annotator: null = no filter at all; "" = the blank-string annotator, a real name in
  // the live DB. That distinction is why the dropdown carries indices, not names.
  status: "all", bucket: "", flag: "", annotator: null, unattributed: false,
  q: "", claimed: false, order: "uuid", view: "thumb",
  offset: 0, limit: 48, total: 0, items: [],
};
let annotatorOptions = [];

function browseQuery() {
  const p = new URLSearchParams();
  if (browse.status !== "all") p.set("status", browse.status);
  if (browse.bucket) p.set("bucket", browse.bucket);
  if (browse.flag) p.set("flag", browse.flag);
  if (browse.unattributed) p.set("unattributed", "1");
  else if (browse.annotator != null) p.set("annotator", browse.annotator);
  if (browse.q) p.set("q", browse.q);
  if (browse.claimed) p.set("claimed", "1");
  if (browse.order !== "uuid") p.set("order", browse.order);
  p.set("offset", browse.offset);
  p.set("limit", browse.limit);
  return p.toString();
}

function saveBrowseHash() {
  setHashParams({
    b_status: browse.status === "all" ? null : browse.status,
    b_bucket: browse.bucket || null,
    b_flag: browse.flag || null,
    // Two keys rather than one with a sentinel value: annotator names are free text, so
    // any "no annotator" marker is a name somebody could have typed. b_who therefore
    // only ever holds a literal name — including the empty string, a real one.
    b_who: (browse.unattributed || browse.annotator == null) ? null : browse.annotator,
    b_unattributed: browse.unattributed ? "1" : null,
    b_q: browse.q || null,
    b_claimed: browse.claimed ? "1" : null,
    b_order: browse.order === "uuid" ? null : browse.order,
    b_view: browse.view === "thumb" ? null : browse.view,
    b_offset: browse.offset ? String(browse.offset) : null,
    b_limit: browse.limit === 48 ? null : String(browse.limit),
  });
}

function readBrowseHash() {
  const p = hashParams();
  const num = (k, dflt) => {
    const n = parseInt(p.get(k), 10);
    return Number.isFinite(n) && n >= 0 ? n : dflt;
  };
  if (BROWSE_STATUSES.some(([k]) => k === p.get("b_status"))) browse.status = p.get("b_status");
  browse.bucket = p.get("b_bucket") || "";
  browse.flag = p.get("b_flag") || "";
  // A blank b_who is the blank-string annotator, a real name — so the key's presence is
  // tested, not its truthiness.
  if (p.get("b_unattributed") === "1") { browse.unattributed = true; browse.annotator = null; }
  else if (p.has("b_who")) { browse.unattributed = false; browse.annotator = p.get("b_who"); }
  browse.q = p.get("b_q") || "";
  browse.claimed = p.get("b_claimed") === "1";
  if (p.get("b_order") === "recent") browse.order = "recent";
  if (BROWSE_VIEWS.some(([k]) => k === p.get("b_view"))) browse.view = p.get("b_view");
  browse.offset = num("b_offset", 0);
  const lim = num("b_limit", 48);
  if ([24, 48, 96].includes(lim)) browse.limit = lim;
}

function fillSelect(select, options, value) {
  select.innerHTML = "";
  for (const [v, label] of options) {
    const o = document.createElement("option");
    o.value = v; o.textContent = label;
    select.appendChild(o);
  }
  select.value = value;
}

function initBrowseOptions(overview, cats) {
  fillSelect(el("b-status"), BROWSE_STATUSES, browse.status);
  fillSelect(el("b-view"), BROWSE_VIEWS.map(([k, label]) => [k, label]), browse.view);
  fillSelect(el("b-bucket"),
    [["", "any category"]].concat(cats
      .filter((c) => c.category != null)
      .slice().sort((a, b) => b.total - a.total)
      .map((c) => [c.category, `${c.category} (${fmt(c.total)})`])),
    browse.bucket);
  fillSelect(el("b-flag"),
    [["", "any reason"]].concat((overview.flags || [])
      .map((f) => [f.flag, `${f.flag.replace(/_/g, " ")} (${fmt(f.count)})`])),
    browse.flag);

  // Names are indices here because two of the selectable values are not names: "anyone"
  // (no filter) and the unattributed backlog (a NULL annotator). Encoding those as
  // sentinel strings would collide with a real annotator who typed the sentinel.
  annotatorOptions = [{ label: "anyone", annotator: null, unattributed: false }]
    .concat(annotators.map((a) => ({
      label: `${a.display} (${fmt(a.items)})`, annotator: a.annotator, unattributed: false })))
    .concat([{ label: "(unattributed)", annotator: null, unattributed: true }]);
  const current = annotatorOptions.findIndex((o) => browse.unattributed
    ? o.unattributed
    : (!o.unattributed && o.annotator === browse.annotator));
  fillSelect(el("b-annotator"), annotatorOptions.map((o, i) => [String(i), o.label]),
    String(current < 0 ? 0 : current));

  el("b-order").value = browse.order;
  el("b-limit").value = String(browse.limit);
  el("b-q").value = browse.q;
  el("b-claimed").checked = browse.claimed;
}

const decisionCls = (e) => (e === 1 ? " good" : e === 0 ? " issue" : " todo");
const decisionText = (e) => (e === 1 ? "✓ no issue" : e === 0 ? "✗ has issue" : "not reviewed");

// Responses are not guaranteed to arrive in the order they were asked for, and the grid
// is shared state. Without this, switching from "auto-filtered" to "not yet reviewed"
// while the first request is slow paints auto rows under a filter bar — and a hash —
// that both say todo. Only the newest request may touch the grid.
let browseSeq = 0;

async function loadBrowse() {
  saveBrowseHash();
  const seq = ++browseSeq;
  const grid = el("b-grid");
  grid.classList.add("loading");
  try {
    const page = await api("items?" + browseQuery());
    if (seq !== browseSeq) return;
    browse.total = page.total;
    browse.items = page.items;
    // A filter can shrink the result set past the cursor (e.g. narrowing while on page
    // 9). Rather than render a blank grid, step back to the last page that exists.
    if (!page.items.length && page.total > 0 && browse.offset >= page.total) {
      browse.offset = Math.max(0, (Math.ceil(page.total / browse.limit) - 1) * browse.limit);
      return loadBrowse();
    }
    renderBrowse();
  } catch (e) {
    if (seq !== browseSeq) return;
    grid.innerHTML = "";
    // Everything else on this page loaded, so a 404 here is not the token — it is a
    // deploy that has the new frontend but not yet the new backend route.
    grid.appendChild(div("empty", e.status === 404
      ? "This view needs the server restarted: the page has been updated but "
        + "/api/admin/items is not being served yet. Everything else above is live."
      : "Could not load shapes (" + e.message + ")."));
  } finally {
    // The retry above owns the spinner from here; clearing it would flash the grid.
    if (seq === browseSeq) grid.classList.remove("loading");
  }
}

function renderBrowse() {
  const grid = el("b-grid");
  grid.innerHTML = "";
  const [, , url, ratio] = viewSpec(browse.view);
  if (!browse.items.length) {
    grid.appendChild(div("empty", "No shapes match these filters."));
  }
  browse.items.forEach((it, i) => {
    const card = div("bcard" + decisionCls(it.emissive));
    card.appendChild(shotFor(url(it.uuid), it.uuid, null, ratio));

    const line = div("bline");
    line.append(mono(it.uuid.slice(0, 12) + "…"));
    if (it.annotator === "auto") line.appendChild(badge("auto", "auto", "Marked by the zero-emission auto-filter, not a person"));
    if (it.claimed_by) line.appendChild(badge("dup", "claimed", "Currently claimed by " + it.claimed_by));
    card.appendChild(line);

    const flags = (it.flags || []).join(", ").replace(/_/g, " ");
    card.appendChild(div("bmeta", decisionText(it.emissive) + (flags ? " · " + flags : "")));
    card.appendChild(div("bmeta dim",
      [it.auto_category || "–", it.annotator || "", it.ts ? shortDate(it.ts).slice(0, 10) : ""]
        .filter(Boolean).join(" · ")));

    card.tabIndex = 0;
    card.onclick = () => lightbox.open(i);
    card.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); lightbox.open(i); } };
    hoverable(card, it.uuid + " — click for every render");
    grid.appendChild(card);
  });

  const first = browse.total ? browse.offset + 1 : 0;
  const last = Math.min(browse.offset + browse.limit, browse.total);
  el("b-range").textContent = browse.total
    ? `${fmt(first)}–${fmt(last)} of ${fmt(browse.total)} shapes`
    : "no matches";
  el("b-prev").disabled = browse.offset <= 0;
  el("b-next").disabled = last >= browse.total;
}

function shotFor(src, uuid, caption, ratio) {
  // A missing render is normal, not an error: the dark-box pair does not cover every shape
  // and a partial burn is expected, so the slot says so rather than showing a broken icon.
  const shot = div("bshot");
  if (ratio) shot.style.aspectRatio = ratio;
  const img = document.createElement("img");
  img.loading = "lazy";
  img.decoding = "async";
  img.alt = (caption || "render") + " — " + uuid;
  img.onerror = () => { img.remove(); shot.classList.add("missing"); shot.appendChild(div("bnone", "not rendered")); };
  img.src = src;
  shot.appendChild(img);
  if (caption) shot.appendChild(div("bcap", caption));
  return shot;
}

function browseReload(patch, keepOffset) {
  Object.assign(browse, patch);
  if (!keepOffset) browse.offset = 0;   // a narrower filter makes page 9 meaningless
  loadBrowse();
}

el("b-status").onchange = (e) => browseReload({ status: e.target.value });
el("b-bucket").onchange = (e) => browseReload({ bucket: e.target.value });
el("b-flag").onchange = (e) => browseReload({ flag: e.target.value });
el("b-order").onchange = (e) => browseReload({ order: e.target.value });
el("b-claimed").onchange = (e) => browseReload({ claimed: e.target.checked });
el("b-limit").onchange = (e) => browseReload({ limit: parseInt(e.target.value, 10) });
el("b-annotator").onchange = (e) => {
  const o = annotatorOptions[parseInt(e.target.value, 10)] || annotatorOptions[0];
  browseReload({ annotator: o.annotator, unattributed: o.unattributed });
};
// Re-rendering is enough for a view switch — same rows, different image URL — so it does
// not go back to the server and does not reset the page.
el("b-view").onchange = (e) => { browse.view = e.target.value; saveBrowseHash(); renderBrowse(); };
el("b-q").addEventListener("input", debounce((e) => browseReload({ q: e.target.value.trim() }), 300));
el("b-prev").onclick = () => browseReload({ offset: Math.max(0, browse.offset - browse.limit) }, true);
el("b-next").onclick = () => browseReload({ offset: browse.offset + browse.limit }, true);
el("b-reset").onclick = () => {
  Object.assign(browse, { status: "all", bucket: "", flag: "", annotator: null,
    unattributed: false, q: "", claimed: false, order: "uuid", offset: 0 });
  el("b-status").value = "all"; el("b-bucket").value = ""; el("b-flag").value = "";
  el("b-annotator").value = "0"; el("b-order").value = "uuid";
  el("b-q").value = ""; el("b-claimed").checked = false;
  loadBrowse();
};

// --- lightbox ---------------------------------------------------------------------
const lightbox = (() => {
  let index = -1;

  function open(i) {
    const it = browse.items[i];
    if (!it) return;
    index = i;
    el("lb-uuid").textContent = it.uuid;
    el("lb-sf").href = "https://sketchfab.com/models/" + it.uuid;
    el("lb-pos").textContent = `${i + 1} of ${browse.items.length} on this page`;
    el("lb-prev").disabled = i <= 0;
    el("lb-next").disabled = i >= browse.items.length - 1;

    const badges = el("lb-badges");
    badges.innerHTML = "";
    // Its own colour, not the auto-filter's amber: on this page amber already means
    // "machine did it", and reusing it for a human "✓ no issue" would misattribute work.
    badges.appendChild(badge(decisionCls(it.emissive).trim(), decisionText(it.emissive)));
    if (it.annotator === "auto") badges.appendChild(badge("auto", "auto-filter"));
    if (it.claimed_by) badges.appendChild(badge("dup", "claimed by " + it.claimed_by));

    const grid = el("lb-grid");
    grid.innerHTML = "";
    for (const key of LIGHTBOX_VIEWS) {
      const [, label, url, ratio] = viewSpec(key);
      grid.appendChild(shotFor(url(it.uuid), it.uuid, label, ratio));
    }

    const meta = el("lb-meta");
    meta.innerHTML = "";
    const rows = [
      ["Category", it.auto_category || "–"],
      ["Reasons flagged", (it.flags || []).join(", ").replace(/_/g, " ") || "–"],
      ["Annotator", it.annotator == null ? "(unattributed)" : (it.annotator || "(blank)")],
      ["Last touched", shortDate(it.ts)],
      ["Style", it.style || "–"],
      ["Note", it.notes || "–"],
    ];
    for (const [k, v] of rows) {
      const row = div("lb-row");
      row.append(div("lb-key", k), div("lb-val", v));
      meta.appendChild(row);
    }
    el("lb").hidden = false;
    el("lb-close").focus();
  }

  function close() { el("lb").hidden = true; index = -1; }
  function step(d) { if (browse.items[index + d]) open(index + d); }

  el("lb-close").onclick = close;
  el("lb-prev").onclick = () => step(-1);
  el("lb-next").onclick = () => step(1);
  // Backdrop only — a click inside the panel must not close it.
  el("lb").onclick = (e) => { if (e.target === el("lb")) close(); };
  document.addEventListener("keydown", (e) => {
    if (el("lb").hidden) return;              // inert while closed, so typing in the
    if (e.key === "Escape") close();          // uuid box is never intercepted
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
  });
  return { open, close };
})();

// --- boot -------------------------------------------------------------------------
async function load() {
  try {
    const [overview, annos, cats, activity] = await Promise.all([
      api("overview"), api("annotators"), api("categories"), api("activity?days=90"),
    ]);
    renderTiles(overview);
    renderOverall(overview);
    renderProjection(overview);
    renderDecisions(overview);
    annotators = annos.annotators;
    renderAnnotators();
    openFromHash();
    renderCategories(cats.categories);
    renderActivity(activity.daily);
    // The browser's dropdowns are built from data already fetched above — the categories,
    // the flag counts and the annotator list — so it costs no extra request.
    initBrowseOptions(overview, cats.categories);
    loadBrowse();
    el("error").hidden = true;
    el("refreshed").textContent = "updated " + new Date().toTimeString().slice(0, 5);
  } catch (e) {
    fail("Could not load the dashboard (" + e.message + ")");
  }
}

el("goto-browse").onclick = () =>
  el("browse-card").scrollIntoView({ behavior: "smooth", block: "start" });
el("refresh").onclick = load;
readBrowseHash();   // a shared link arrives with its filters in the hash
load();
