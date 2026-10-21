const RENDER_KINDS = [                                       // shown on each card: [kind, caption]
  // row 1 — what the object looks like; row 2 — the dark-box pair, same scene
  // and same faint fill, differing ONLY in emission, off before on so the eye
  // reads the difference as "what the emission adds"
  ["glb", "rendered object"],
  ["brighter", "emissive-brightened rendering"],
  ["boxoff", "dark-box (emission off)"],
  ["box", "dark-box (emission on)"],
];
// Renders are served `immutable`, so a re-render under the same uuid would keep
// showing the browser's year-old copy. Bump the tag whenever a kind is re-burnt.
// filmic = the dark-box renders with Filmic at exposure 0.
const RENDER_VERSION = { box: "filmic", boxoff: "filmic" };
const ALL_VIEWS = [                     // "more" overlay — the 4 card renders are omitted (already on the card)
  ["emit-only rendering", u => `img/render/emit/${u}`],
  // ?v=73432 busts the `immutable` image cache: emission moved to the texture-alone
  // uv_voxel_73432 bake, so the same URL yields a different image. Bump this tag on any
  // future change to emission generation.
  ["emission map · uv", u => `img/emission/${u}?v=73432`],
  ["multi-view orbit", u => `img/mv/standard/${u}`],
  ["multi-view orbit (emit-only)", u => `img/mv/emit/${u}`],
];
const NON_EMISSIVE_REASONS = [                               // "why not emissive?" (stored in flags)
  ["not_emissive_object", "not emissive object"],
  ["wrong_emissive_maps", "wrong emissive maps"],
  ["zero_emission_map", "zero emission map"],
  ["multiple_object", "multiple object"],
];
const STYLE_OPTIONS = [                                      // "artistic vs real" (stored in style)
  ["artistic", "artistic"],
  ["real", "real"],
];
// One shape per page: the dark-box pair only reads at size, and judging emission
// is a per-object decision anyway. Arrow keys page through (see keyboard nav).
// The grid only ever shows shapes from your OWN claimed batch — there is no view of the
// whole cohort. Claimed shapes are invisible to everyone else, so two people never land
// on the same one; a browse-everything escape hatch would hand out unclaimed shapes to
// whoever looked, which is exactly what claiming exists to prevent. Before you claim,
// the grid is empty.
const state = { offset: 0, limit: 1, total: 0 };

function annotator() { return localStorage.getItem("ann74k_annotator") || ""; }
function setAnnotator() {
  // Renaming mid-run splits your work across two names, but nothing pins a name
  // for a block of time any more, so this is a warning rather than a lock: the
  // dwell rows already written keep the name they were written with.
  const n = prompt("Your annotator name:", annotator());
  if (n) {
    dwell.flush();      // bank the current shape under the OLD name before switching
    localStorage.setItem("ann74k_annotator", n);
    document.getElementById("annotator-name").textContent = n;
    updateGate();
    syncStatusFilter();   // a name is half of what puts you in batch mode
    // Invalidate everything already in flight for the OLD name, synchronously and before
    // the round-trip. A load() that started as Alice would otherwise still satisfy its own
    // `seq === loadSeq` check when it lands, painting Alice's batch onto a screen that now
    // says Bob — and the next decision made there saves under Bob, onto Alice's shape.
    loadSeq++; resumeSeq++;
    state.offset = 0; state.current = null;
    document.getElementById("grid").innerHTML = "";
    fetch("api/annotator", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: n }) })
      .catch(() => {})    // registering the name is bookkeeping; the batch still loads
      .then(async () => {
        await resumeAtFirstUndecided();   // this name may already hold a part-done batch
        load();
      });
  }
}

// Annotation gate: a name is all that's required — it's what makes the work
// attributable. (It used to also demand a running timer session.)
function annotationAllowed() {
  return !!annotator();
}
// The status dropdown and a claimed batch are mutually exclusive views (see filters()),
// so inside a batch it is disabled and says why, rather than sitting there looking live.
// Since the grid became batch-only there is no view left that it applies to, so it is
// disabled throughout; it stays in the markup because restoring a cohort-wide view (the
// `auto: zero-emission` list in particular) is a matter of re-enabling it.
function syncStatusFilter() {
  const sel = document.getElementById("f-status");
  if (!sel) return;
  sel.disabled = true;
  sel.title = "Your batch is shown whole, decided shapes included — there is nothing to filter.";
}

function updateGate() {
  const ok = annotationAllowed();
  document.body.classList.toggle("annotation-locked", !ok);
  const banner = document.getElementById("gate-banner");
  if (!banner) return;
  if (ok) { banner.hidden = true; banner.textContent = ""; return; }
  banner.hidden = false;
  banner.textContent = "🔒 Set your annotator name (top bar) to start annotating.";
}

async function init() {
  document.getElementById("annotator-name").textContent = annotator() || "?";
  document.getElementById("set-annotator").onclick = setAnnotator;
  dwell.init();
  updateGate();   // annotation stays locked until a name is set
  syncStatusFilter();
  // filters auto-apply on change so the grid updates without an extra click
  document.getElementById("f-status").onchange = () => { state.offset = 0; load(); };
  document.getElementById("apply").onclick = () => { state.offset = 0; load(); };
  document.getElementById("f-q").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { state.offset = 0; load(); } });
  // Going back is always a plain step: deciding a shape only shifts the ones AFTER it,
  // so everything before the cursor is still exactly where it was.
  document.getElementById("prev").onclick = () => {
    if (state.offset > 0) { state.offset -= state.limit; load(); } };
  document.getElementById("next").onclick = () => {
    // A filter like "to-do" drops a shape the moment you decide it, so the list has
    // already shifted one place and the next shape has slid into the current position.
    // Advancing as well would step straight over it.
    if (!currentStillMatchesFilter()) { load(); return; }
    if (state.offset + state.limit < state.total) { state.offset += state.limit; load(); }
  };
  document.getElementById("viewer-close").onclick = () => closeViews();
  document.getElementById("viewer").onclick = (e) => { if (e.target.id === "viewer") closeViews(); };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeViews(); sketchfab.close(); claimPanel.close(); } });
  // Arrow keys page through — with one shape per page you turn the page ten times
  // as often, so reaching for the mouse every time is the wrong ergonomics. Typing
  // in a field, an open overlay, or a browser shortcut all keep their own meaning.
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable) return;
    if (document.getElementById("viewer").classList.contains("show")) return;
    if (claimPanel.isOpen()) return;   // the panel is a dialog; paging behind it is not
    if (e.key === "ArrowLeft") { e.preventDefault(); document.getElementById("prev").click(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); document.getElementById("next").click(); }
  });
  claimPanel.init();
  document.getElementById("claim-btn").onclick = claimPanel.open;
  document.getElementById("open-guide").onclick = openGuide;
  // Filters and uuid search are needed rarely, so they hide behind ⚙ instead of taking
  // permanent space away from the shape — which is the only thing on screen that matters.
  const toolsBtn = document.getElementById("toggle-tools");
  toolsBtn.onclick = () => {
    const tools = document.getElementById("tools");
    tools.hidden = !tools.hidden;
    toolsBtn.classList.toggle("active", !tools.hidden);
  };
  const pv = document.getElementById("preview");
  pv.addEventListener("mouseenter", () => clearTimeout(previewHideTimer));
  pv.addEventListener("mouseleave", scheduleHidePreview);
  await resumeAtFirstUndecided();
  await load();
}

// Decided shapes keep their place in a batch, so opening at position 0 lands on shape 1
// — already done — and you would page past all of your own previous work to reach the
// front. Ask the server where the first undecided shape is and open there instead.
// next_index is a position in the WHOLE batch, so it is meaningless once a uuid search
// narrows the view — resuming into a filtered subset would land past its end.
let resumeSeq = 0;
function resumable() {
  return inBatch() && !document.getElementById("f-q").value.trim();
}
async function resumeAtFirstUndecided() {
  if (!resumable()) return;
  const seq = ++resumeSeq, from = state.offset, who = annotator();
  try {
    const b = await (await fetch(
      "api/batch?annotator=" + encodeURIComponent(who))).json();
    // Re-check everything, not just what held when we asked: the user can page, type a
    // uuid, rename themselves or leave batch mode while this is in flight. Their action
    // wins — a resume decided against a view that no longer exists would yank them
    // somewhere they never asked to go. `who` is checked because a position inside one
    // person's batch means nothing inside another's.
    if (seq !== resumeSeq || state.offset !== from || !resumable() || annotator() !== who) return;
    if (b && b.total && b.remaining > 0) state.offset = b.next_index || 0;
  } catch (e) { /* opening at the start is a fine fallback */ }
}

function renderShare(batch) {
  const el = document.getElementById("share-info");
  const btn = document.getElementById("claim-btn");
  if (!el) return;
  if (!batch || !batch.total) {
    el.textContent = "no batch claimed";
    btn.textContent = "＋ claim a batch";
    return;
  }
  el.textContent = `batch: ${batch.done} / ${batch.total} done`;
  btn.textContent = batch.remaining === 0 ? "✓ batch done — claim another" : "＋ claim more";
}

// Claim panel: pick a size, tick categories, each showing how much work is still free
// in it. It replaces two native prompt()s — the category one printed 40 names as a
// single comma-separated line and expected one typed back exactly.
const claimPanel = (() => {
  const PRESETS = [25, 50, 100, 200];
  let free = [];                     // [{category, claimable}] from /api/claimable
  const picked = new Set();
  let count = 50;
  const el = id => document.getElementById(id);

  // Ticking nothing means a mix of everything, so the pool is then every category.
  function available() {
    const rows = picked.size ? free.filter(f => picked.has(f.category)) : free;
    return rows.reduce((a, f) => a + f.claimable, 0);
  }

  function paintCounts() {
    for (const b of el("claim-counts").querySelectorAll(".count-btn"))
      b.classList.toggle("active", Number(b.dataset.n) === count);
  }

  // Says what this claim will actually hand out, not what was asked for: the two differ
  // whenever a category has less work left than the size picked.
  function paintSummary() {
    const avail = available(), take = Math.min(count, avail);
    const where = picked.size === 0 ? "all categories"
                : picked.size === 1 ? `“${[...picked][0]}”`
                : `${picked.size} categories`;
    el("claim-summary").textContent = avail === 0
      ? `nothing left to claim in ${where}`
      : `claiming ${take.toLocaleString()} · ${avail.toLocaleString()} free in ${where}`;
    el("claim-go").disabled = avail === 0 || count < 1;
  }

  function buildCounts() {
    const box = el("claim-counts");
    box.innerHTML = "";
    for (const n of PRESETS) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "count-btn"; b.textContent = n; b.dataset.n = n;
      b.onclick = () => {
        count = n; el("claim-custom").value = n; paintCounts(); paintSummary();
      };
      box.appendChild(b);
    }
    const custom = document.createElement("input");
    custom.id = "claim-custom"; custom.type = "number"; custom.min = "1"; custom.max = "2000";
    custom.className = "count-custom"; custom.value = count; custom.title = "1 – 2000";
    custom.oninput = () => {
      const v = parseInt(custom.value, 10);
      count = v > 0 ? Math.min(v, 2000) : 0;   // the API caps a claim at 2000
      paintCounts(); paintSummary();
    };
    box.appendChild(custom);
    paintCounts();
  }

  function renderCats() {
    const q = el("claim-search").value.trim().toLowerCase();
    const list = el("claim-cat-list");
    list.innerHTML = "";
    const rows = free.filter(f => f.category && (!q || f.category.toLowerCase().includes(q)));
    if (!rows.length) {
      const none = document.createElement("div");
      none.className = "claim-cat empty"; none.textContent = "no category matches";
      list.appendChild(none); return;
    }
    for (const f of rows) {
      const row = document.createElement("label");
      row.className = "claim-cat" + (f.claimable ? "" : " empty");
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = picked.has(f.category);
      cb.disabled = !f.claimable;      // nothing free there — ticking it would be a no-op
      cb.onchange = () => {
        if (cb.checked) picked.add(f.category); else picked.delete(f.category);
        paintSummary();
      };
      const nm = document.createElement("span");
      nm.className = "claim-cat-name"; nm.textContent = f.category; nm.title = f.category;
      const n = document.createElement("span");
      n.className = "claim-cat-n"; n.textContent = f.claimable.toLocaleString() + " left";
      row.append(cb, nm, n);
      list.appendChild(row);
    }
  }

  function close() { el("claim-panel").hidden = true; }
  function isOpen() { return !el("claim-panel").hidden; }

  let openSeq = 0;
  async function open() {
    if (!annotator()) { alert("Set your annotator name first."); return; }
    picked.clear();
    el("claim-search").value = "";
    el("claim-panel").hidden = false;
    el("claim-cat-list").textContent = "loading…";
    el("claim-summary").textContent = "";
    el("claim-go").disabled = true;
    buildCounts();
    // Close-and-reopen starts a second request; whichever was asked for LAST is the one
    // whose counts belong on screen, not whichever happens to answer last.
    const seq = ++openSeq;
    let got;
    try {
      got = (await (await fetch("api/claimable")).json()).categories || [];
    } catch (e) { got = []; }
    if (seq !== openSeq) return;
    free = got;
    free.sort((a, b) => b.claimable - a.claimable);   // where the work is, first
    renderCats(); paintSummary();
  }

  async function submit() {
    el("claim-go").disabled = true;
    let res;
    try {
      res = await (await fetch("api/claim", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotator: annotator(), count,
                               categories: picked.size ? [...picked] : null }),
      })).json();
    } catch (e) {
      el("claim-summary").textContent = "claim failed — try again";
      el("claim-go").disabled = false; return;
    }
    if (res.error) {
      el("claim-summary").textContent = res.error;
      el("claim-go").disabled = false; return;
    }
    close();
    if (!res.claimed) alert("Nothing was free to claim there — try another category.");
    // Land on work still to do, not on shapes decided in an earlier sitting.
    state.offset = (res.batch && res.batch.remaining > 0) ? (res.batch.next_index || 0) : 0;
    await load();
  }

  function init() {
    el("claim-close").onclick = close;
    el("claim-cancel").onclick = close;
    el("claim-go").onclick = submit;
    el("claim-search").oninput = renderCats;
    el("claim-panel").onclick = e => { if (e.target.id === "claim-panel") close(); };
  }
  return { init, open, close, isOpen };
})();

// Working a batch means working ALL of it, so the status filter is not applied there.
// Filtering to "to-do" drops a shape out of the list the moment you decide it, and the
// pager addresses shapes by position — every decision slid the list one place under the
// cursor, so → skipped the next shape and then overran the list at the batch's midpoint.
// Leaving decided shapes in place also lets you page back and fix one.
// A name is now the only thing standing between you and your batch, so "in a batch" and
// "has a name" are the same condition — the grid has no other mode to be in.
function inBatch() { return !!annotator(); }

// Does the shape on screen still belong in the list it was fetched from? Deciding one
// under a status filter answers no — the server would no longer return it, so the list
// shortens and everything after it moves up a place. The pager reads this to tell
// "the list moved for me" apart from "I need to move myself". A batch is never
// filtered (see filters()), so nothing there can drop out.
function currentStillMatchesFilter() {
  const it = state.current;
  if (!it || inBatch()) return true;
  switch (document.getElementById("f-status").value) {
    case "todo": return it.verified !== 1;
    case "emissive": return it.emissive === 1;
    case "not_emissive": return it.emissive === 0;
    case "finished": return it.verified === 1;
    case "auto_zero": return it.annotator === "auto";
    default: return true;      // "all" holds everything, whatever you decide
  }
}

function filters() {
  const p = new URLSearchParams();
  const status = inBatch() ? "all" : document.getElementById("f-status").value;
  const q = document.getElementById("f-q").value.trim();
  if (status === "todo") p.set("verified", "false");          // not yet reviewed
  else if (status === "emissive") p.set("emissive", "true");  // good
  else if (status === "not_emissive") p.set("emissive", "false");  // has issue
  else if (status === "finished") p.set("verified", "true");  // reviewed (good + has issue)
  else if (status === "auto_zero") p.set("annotator", "auto");  // auto-filtered zero-emission
  if (q) p.set("q", q);
  // `me` is who is looking (drives the shard); `annotator` above is a filter.
  // Always sent when a name is set — omitting it is what used to return the whole cohort.
  // load() refuses to fetch at all without a name, so the unscoped query is unreachable.
  if (annotator()) { p.set("mine", "1"); p.set("me", annotator()); }
  p.set("offset", state.offset); p.set("limit", state.limit);
  return p;
}

let loadSeq = 0;
async function load() {
  const seq = ++loadSeq;
  // Clear the grid BEFORE fetching: this cancels the previous page's still-downloading
  // images so they stop hogging the browser's ~6 connections. Otherwise switching category
  // (this api/items request) stalls behind ~30 in-flight image downloads.
  const grid = document.getElementById("grid");
  sketchfab.close();   // the open panel's card is about to be cleared — tear it down
  grid.innerHTML = "";
  // No name means no batch, and there is no cohort-wide view to fall back to, so there is
  // nothing to ask the server for. Returning here also keeps the unscoped /api/items query
  // (no `mine`, no `me` — the whole set) from ever being issued by this page.
  if (!annotator()) {
    state.total = 0; state.current = null; state.offset = 0;
    dwell.flush();
    renderEmpty("Set your annotator name in the top bar to get started.", false);
    document.getElementById("page-info").textContent = "0 / 0";
    renderShare(null);
    return refreshProgress();
  }
  const who = annotator();
  const data = await (await fetch("api/items?" + filters().toString())).json();
  // A newer load started (filter changed, page turned) — drop the stale grid. The name is
  // checked as well as the sequence: cards fetched for one annotator must never be shown
  // to another, since a decision on screen is saved under whoever is named at that moment.
  if (seq !== loadSeq || annotator() !== who) return;
  state.total = data.total;
  // The result set shrinks as you decide shapes under a status filter, so an offset that
  // was valid a moment ago can now point past the end. Step back to the last shape rather
  // than showing a blank page, which looks like a dead → key.
  if (!data.items.length && state.total > 0 && state.offset > 0) {
    state.offset = Math.max(0, state.total - state.limit);
    return load();
  }
  // One shape per page makes a range redundant — "17 / N", not "17-17 / N"
  const from = state.total ? state.offset + 1 : 0;
  const to = Math.min(state.offset + state.limit, state.total);
  document.getElementById("page-info").textContent =
    `${from === to ? from : `${from}-${to}`} / ${state.total.toLocaleString()}`;
  state.current = data.items.length === 1 ? data.items[0] : null;
  // An empty batch is the normal state for someone who has just set their name, not an
  // error — say what to do about it instead of leaving a blank page. A uuid search that
  // matches nothing inside the batch lands here too, and needs the opposite advice.
  if (!data.items.length) {
    renderEmpty(document.getElementById("f-q").value.trim()
      ? "No shape in your batch matches that uuid — clear the search in ⚙ to go back."
      : "You have no shapes to work on yet — claim a batch to get started.",
      !document.getElementById("f-q").value.trim());
  }
  for (const it of data.items) grid.appendChild(card(it));
  // One shape per page, so the page IS the dwell unit; entering banks the
  // previous shape's time. A multi-item page has no single subject — just flush.
  if (data.items.length === 1) dwell.enter(data.items[0].uuid); else dwell.flush();
  renderShare(data.batch);
  refreshProgress();
}

// The empty grid is where a new annotator spends their first minute, so it carries the
// next action rather than just stating the absence.
function renderEmpty(msg, withClaimButton) {
  const box = document.createElement("div");
  box.className = "grid-empty";
  const p = document.createElement("p");
  p.textContent = msg;
  box.appendChild(p);
  if (withClaimButton) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "claim-btn empty-claim";
    b.textContent = "＋ claim a batch";
    b.onclick = claimPanel.open;
    box.appendChild(b);
  }
  document.getElementById("grid").appendChild(box);
}

async function refreshProgress() {
  const p = await (await fetch("api/progress")).json();
  document.getElementById("progress").textContent =
    `reviewed ${p.verified} / ${p.total} · issues ${p.verified - p.emissive_true}`;
}

function lazyImg(src, label) {
  const fig = document.createElement("figure");
  const img = document.createElement("img");
  img.dataset.src = src; img.alt = label;
  img.onerror = () => { img.style.opacity = .25; };
  const cap = document.createElement("figcaption"); cap.textContent = label;
  fig.append(img, cap);
  observer.observe(img);
  return fig;
}

const observer = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) { e.target.src = e.target.dataset.src; observer.unobserve(e.target); }
  }
}, { rootMargin: "200px" });

function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

function card(it) {
  const el = document.createElement("div");
  el.className = "card" + decisionClass(it.emissive);
  // Header first, then a clean 2x2 of images. "more" used to sit as a fifth
  // grid cell, which forced a third row and pushed the decision controls off
  // the bottom of the screen; as a header button it costs no vertical space.
  const uuidRow = document.createElement("div"); uuidRow.className = "uuid-row";
  const uuid = document.createElement("span"); uuid.className = "uuid"; uuid.textContent = it.uuid;
  const more = document.createElement("button"); more.className = "more-btn"; more.type = "button";
  more.textContent = "⤢ more"; more.title = "hover to preview: emit-only + emission map + multi-view";
  more.addEventListener("mouseenter", () => showPreview(it.uuid, more));
  more.addEventListener("mouseleave", scheduleHidePreview);
  const sfBtn = document.createElement("button"); sfBtn.className = "sf-btn"; sfBtn.type = "button";
  sfBtn.textContent = "⬡ Sketchfab";
  sfBtn.title = "View the source model on Sketchfab (loads on click; not preloaded)";
  sfBtn.setAttribute("aria-expanded", "false");
  sfBtn.onclick = () => sketchfab.toggle(it.uuid, sfBtn);
  uuidRow.append(uuid, more, sfBtn);
  el.appendChild(uuidRow);

  const imgs = document.createElement("div"); imgs.className = "imgs";
  for (const [k, label] of RENDER_KINDS) {
    const v = RENDER_VERSION[k] ? `?v=${RENDER_VERSION[k]}` : "";
    imgs.appendChild(lazyImg(`img/render/${k}/${it.uuid}${v}`, label));
  }
  el.appendChild(imgs);

  const saved = document.createElement("span"); saved.className = "saved";
  let saveChain = Promise.resolve(); let pending = {};
  const flush = debounce(() => {
    // Attribute the save to the SESSION's annotator (frozen at start), not the
    // live name — so every item in a session shares one name even if the stored
    // name is somehow changed. Falls back to the stored name if no session (saves
    // are gated on a running timer, so this is just belt-and-suspenders).
    const patch = { ...pending, uuid: it.uuid, annotator: annotator() }; pending = {};
    dwell.noteSave(patch);          // this visit ended in a decision — record which
    // A write that did not land must not leave the card looking decided: the button
    // handlers mutate `it` and repaint BEFORE saving, so silence here would show a green
    // card over a database row that has nothing. Put the patch back so the next save
    // retries it — under anything newer, which is the annotator's more recent intent.
    const fail = (msg, gone) => {
      const { uuid: _u, annotator: _a, ...sent } = patch;
      pending = { ...sent, ...pending };
      saved.textContent = msg;
      saved.classList.add("save-failed");
      // 409: the shape belongs to someone else now, so no retry can succeed and the card
      // on screen is stale. Drop the patch and re-read the batch — what the server has is
      // the truth, and it is better seen than quietly diverged from.
      if (gone) { pending = {}; load(); }
    };
    saveChain = saveChain.then(async () => {   // serialize this card's writes; fields are merged so they don't clobber
      const r = await fetch("api/save", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch) });
      // The response has to be read: the server refuses a write to a shape someone else
      // now holds (409), which happens for real when a batch lapses after 48 idle hours
      // and is handed on while this page stays open. Printing "saved ✓" regardless would
      // be a lie about work that was dropped, and the annotator would keep going.
      if (!r.ok) return fail(r.status === 409
        ? "not saved — someone else now has this shape"
        : "not saved — please reload", r.status === 409);
      saved.classList.remove("save-failed");
      saved.textContent = "saved ✓"; refreshProgress();
    }).catch(() => fail("not saved — connection lost", false));
  }, 250);
  const save = fields => { Object.assign(pending, fields); flush(); };

  // decision: "No issue" (good) or "Has issue" (doesn't pass) — mutually exclusive.
  const row = document.createElement("div"); row.className = "row decision";
  const okBtn = document.createElement("button"); okBtn.className = "dec em"; okBtn.textContent = "✓ No issue";
  const issueBtn = document.createElement("button"); issueBtn.className = "dec no"; issueBtn.textContent = "✗ Has issue";

  // style (shown only while No-issue is active) — stored in it.style
  const styleRow = document.createElement("div"); styleRow.className = "row styles";
  const styleLbl = document.createElement("span"); styleLbl.className = "why-lbl"; styleLbl.textContent = "style:";
  const styleBtns = STYLE_OPTIONS.map(([key, label]) => {
    const b = document.createElement("button"); b.className = "style-btn";
    b.textContent = label; b.dataset.key = key;
    b.onclick = () => pickStyle(key);
    return b;
  });
  styleRow.append(styleLbl, ...styleBtns);

  // emission completeness (shown only while No-issue is active) — stored in it.emission_complete
  const completeRow = document.createElement("div"); completeRow.className = "row complete";
  const completeLbl = document.createElement("label"); completeLbl.className = "complete-lbl";
  const completeCb = document.createElement("input"); completeCb.type = "checkbox";
  // null/undefined = unset (treat as complete once No-issue is chosen); 0/false = incomplete
  completeCb.checked = it.emission_complete == null ? true : !!it.emission_complete;
  completeCb.onchange = () => {
    if (!annotationAllowed()) { completeCb.checked = !completeCb.checked; return; }  // revert the toggle
    it.emission_complete = completeCb.checked ? 1 : 0;
    save({ emission_complete: completeCb.checked });
  };
  const completeTxt = document.createElement("span");
  completeTxt.textContent = "emission complete (all parts that should glow do)";
  completeTxt.title = "Uncheck if emission is missing expected parts — e.g. a car with only the dashboard lit, but headlights/taillights should also emit.";
  completeLbl.append(completeCb, completeTxt);
  completeRow.appendChild(completeLbl);

  // reason (shown only while Has-issue is active) — stored in it.flags
  const reasonRow = document.createElement("div"); reasonRow.className = "row reasons";
  const why = document.createElement("span"); why.className = "why-lbl"; why.textContent = "why:";
  const reasonBtns = NON_EMISSIVE_REASONS.map(([key, label]) => {
    const b = document.createElement("button"); b.className = "reason";
    b.textContent = label; b.dataset.key = key;
    b.onclick = () => pickReason(key);
    return b;
  });
  reasonRow.append(why, ...reasonBtns);

  // free-form note (shown only while Has-issue is active) — stored in it.notes
  const notesRow = document.createElement("div"); notesRow.className = "row notes";
  const notesBox = document.createElement("textarea"); notesBox.className = "notes-box"; notesBox.rows = 2;
  notesBox.placeholder = "note (optional) — e.g. object too small to verify";
  notesBox.value = it.notes || "";
  notesBox.oninput = () => { if (!annotationAllowed()) return; it.notes = notesBox.value; save({ notes: notesBox.value }); };
  notesRow.appendChild(notesBox);

  const hasReason = (key) => Array.isArray(it.flags) && it.flags.includes(key);
  function paintReasons() {
    for (const b of reasonBtns) b.classList.toggle("active", hasReason(b.dataset.key));
  }
  function paintStyles() {
    for (const b of styleBtns) b.classList.toggle("active", b.dataset.key === it.style);
  }
  function render() {
    const ok = it.emissive === 1;
    const issue = it.emissive === 0;
    el.className = "card" + decisionClass(it.emissive);
    okBtn.classList.toggle("active", ok);
    issueBtn.classList.toggle("active", issue);
    styleRow.style.display = ok ? "flex" : "none";
    completeRow.style.display = ok ? "flex" : "none";
    reasonRow.style.display = issue ? "flex" : "none";
    notesRow.style.display = issue ? "block" : "none";
    if (ok) {
      completeCb.checked = it.emission_complete == null ? true : !!it.emission_complete;
    }
    paintReasons();
    paintStyles();
  }
  function clearGoodFields() {
    it.style = null; it.emission_complete = null;
    completeCb.checked = true;
  }
  function clearIssueFields() {
    it.flags = []; it.notes = ""; notesBox.value = "";
  }
  function toggleOk() {
    if (!annotationAllowed()) return;
    if (it.emissive === 1) {                       // un-mark → back to to-do
      it.emissive = null; it.verified = 0; clearGoodFields();
      render(); save({ emissive: null, verified: false, style: null, emission_complete: null });
    } else {                                       // mark as "no issue"
      it.emissive = 1; it.verified = 1; clearIssueFields();
      if (it.emission_complete == null) it.emission_complete = 1;  // default: complete
      completeCb.checked = it.emission_complete !== 0;
      render();
      save({ emissive: true, verified: true, flags: [], notes: "",
             emission_complete: !!completeCb.checked });
    }
  }
  function toggleIssue() {
    if (!annotationAllowed()) return;
    if (it.emissive === 0) {                       // un-flag → back to to-do
      it.emissive = null; it.verified = 0; clearIssueFields();
      render(); save({ emissive: null, verified: false, flags: [], notes: "" });
    } else {                                       // flag as "has issue"
      it.emissive = 0; it.verified = 1; clearGoodFields();
      render();
      save({ emissive: false, verified: true, style: null, emission_complete: null });
    }
  }
  function pickStyle(key) {
    if (!annotationAllowed()) return;
    it.style = (it.style === key) ? null : key;    // click again to unset
    paintStyles(); save({ style: it.style });
  }
  function pickReason(key) {                          // multi-select: toggle each reason in/out
    if (!annotationAllowed()) return;
    const flags = Array.isArray(it.flags) ? it.flags.slice() : [];
    const i = flags.indexOf(key);
    if (i >= 0) flags.splice(i, 1); else flags.push(key);
    it.flags = flags;
    paintReasons(); save({ flags: it.flags });
  }
  okBtn.onclick = toggleOk;
  issueBtn.onclick = toggleIssue;
  render();
  row.append(okBtn, issueBtn, saved); el.appendChild(row);
  el.appendChild(styleRow); el.appendChild(completeRow);
  el.appendChild(reasonRow); el.appendChild(notesRow);
  return el;
}

function decisionClass(val) {
  if (val === 1) return " is-emissive";
  if (val === 0) return " not-emissive";
  return "";
}

// hover preview: a small floating 2x2 panel next to the card (no full-page backdrop)
let previewHideTimer = null;
let previewUuid = null;
function showPreview(uuid, anchor) {
  clearTimeout(previewHideTimer);
  const pv = document.getElementById("preview");
  if (previewUuid !== uuid) {
    previewUuid = uuid;
    const grid = document.createElement("div"); grid.className = "preview-grid";
    for (const [label, url] of ALL_VIEWS) {
      const fig = document.createElement("figure");
      const img = document.createElement("img"); img.src = url(uuid); img.alt = label;
      img.onerror = () => { img.style.opacity = .25; };
      const cap = document.createElement("figcaption"); cap.textContent = label;
      fig.append(img, cap); grid.appendChild(fig);
    }
    pv.innerHTML = ""; pv.appendChild(grid);
  }
  pv.style.display = "block";
  const r = anchor.getBoundingClientRect();
  let left = r.right + 8;
  if (left + pv.offsetWidth > window.innerWidth - 8) left = r.left - pv.offsetWidth - 8;  // flip to the left
  left = Math.max(8, left);
  const top = Math.min(Math.max(8, r.top), window.innerHeight - pv.offsetHeight - 8);
  pv.style.left = left + "px"; pv.style.top = top + "px";
}
function scheduleHidePreview() {
  clearTimeout(previewHideTimer);
  previewHideTimer = setTimeout(() => {
    document.getElementById("preview").style.display = "none"; previewUuid = null;
  }, 200);
}

function closeViews() {
  document.getElementById("viewer").classList.remove("show");
}

// Lazy Sketchfab embed: the shape's uuid IS its Sketchfab model UID, so the source
// model is just https://sketchfab.com/models/<uuid>. A single floating panel is
// mounted only on click and torn down on close / re-click / Escape, so the heavy
// WebGL viewer is never preloaded and never lingers across cards.
const sketchfab = (() => {
  let panel = null, activeBtn = null;
  function close() {
    if (panel) { panel.remove(); panel = null; }
    if (activeBtn) { activeBtn.setAttribute("aria-expanded", "false"); activeBtn = null; }
  }
  function open(uuid, btn) {
    close();
    activeBtn = btn;
    panel = document.createElement("div"); panel.id = "sf-panel";
    const head = document.createElement("div"); head.className = "sf-head";
    head.innerHTML =
      "Sketchfab · " + uuid.slice(0, 12) +
      ' <a href="https://sketchfab.com/models/' + uuid + '" target="_blank" rel="noopener">open full page ↗</a>' +
      '<button type="button" title="close">✕</button>';
    const frame = document.createElement("iframe");
    frame.title = "Sketchfab model " + uuid;
    frame.setAttribute("allow", "autoplay; fullscreen; xr-spatial-tracking");
    frame.setAttribute("allowfullscreen", "");
    frame.src = "https://sketchfab.com/models/" + uuid + "/embed?ui_theme=dark&autospin=0.2&ui_infos=0";
    panel.append(head, frame);
    document.body.appendChild(panel);
    head.querySelector("button").addEventListener("click", close);
    btn.setAttribute("aria-expanded", "true");
  }
  function toggle(uuid, btn) {
    if (panel && activeBtn === btn) { close(); return; }  // re-click same card → close
    open(uuid, btn);
  }
  return { toggle, close };
})();

// minimal markdown -> html for the in-app guide
// (headers / bold / italic / code / images / links / lists)
function mdToHtml(md) {
  // `"` is escaped too, not just the tag characters: alt and href text goes into a quoted
  // HTML attribute, and a guide line like `![A "No issue" decision](…)` would otherwise
  // close the attribute early and spill its own words into the markup as bare tokens.
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
  const inline = s => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    // Images BEFORE links: `![alt](src)` also matches the link pattern, and running that
    // one first would leave a stray "!" in front of an anchor instead of a picture.
    .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" loading="lazy">')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const out = []; let para = []; let items = null; let listTag = "ul";
  const flushPara = () => { if (para.length) { out.push("<p>" + inline(para.join(" ")) + "</p>"); para = []; } };
  const flushList = () => {
    if (items) {
      out.push(`<${listTag}>` + items.map(t => "<li>" + inline(t) + "</li>").join("") + `</${listTag}>`);
      items = null;
    }
  };
  for (const raw of md.split(/\r?\n/)) {
    if (!raw.trim()) { flushPara(); flushList(); continue; }
    let m = raw.match(/^(#{1,4})\s+(.*)$/);
    if (m) { flushPara(); flushList(); out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); continue; }
    // Bulleted AND numbered lists. Without the numbered case the Setup section — which is
    // "1. 2. 3." — collapsed into a single run-on paragraph, steps and screenshots alike.
    let tag = null, text = null;
    if ((m = raw.match(/^\s*[-*]\s+(.*)$/))) { tag = "ul"; text = m[1]; }
    else if ((m = raw.match(/^\s*\d+\.\s+(.*)$/))) { tag = "ol"; text = m[1]; }
    if (tag) {
      flushPara();
      if (items && listTag !== tag) flushList();   // a different kind of list starts a new one
      if (!items) { items = []; listTag = tag; }
      items.push(text); continue;
    }
    if (items && /^\s+\S/.test(raw)) { items[items.length - 1] += " " + raw.trim(); continue; }  // wrapped li
    flushList(); para.push(raw.trim());
  }
  flushPara(); flushList();
  return out.join("\n");
}

async function openGuide() {
  const content = document.getElementById("viewer-content");
  content.className = "viewer-content guide";
  content.innerHTML = "<p>loading…</p>";
  document.getElementById("viewer").classList.add("show");
  try {
    content.innerHTML = mdToHtml(await (await fetch("guide")).text());
    // The screenshots are full browser captures squeezed into a 760px column, so their
    // red callouts end up small. Clicking opens the original at full size in a new tab.
    for (const img of content.querySelectorAll("img"))
      img.onclick = () => window.open(img.src, "_blank", "noopener");
  } catch (e) {
    content.innerHTML = "<p>Could not load the guide.</p>";
  }
}

// ---- session timer ------------------------------------------------------
// Per-page dwell: how long an annotator actually looked at this shape. Replaces
// the timed-session model — with one shape per page, the page IS the unit of
// work, so effort is measured where it is spent instead of in fixed blocks.
// Hidden-tab time is excluded: a tab left open over lunch would otherwise swamp
// every average.
const dwell = (() => {
  const MIN_MS = 100;
  let uuid = null, since = 0, acc = 0, saves = 0, patch = null;
  const visible = () => document.visibilityState === "visible";

  function accumulate() {
    if (uuid && since) { acc += Date.now() - since; since = 0; }
  }
  // Called on every save so a visit records not just its length but whether it
  // ended in a decision, and which one. `items` keeps only the final state, so a
  // merged patch per visit is the only trace of how that state was arrived at.
  function noteSave(p) {
    saves++;
    const { uuid: _u, annotator: _a, ...fields } = p;
    patch = { ...(patch || {}), ...fields };
  }
  function send(u, ms) {
    const body = JSON.stringify({ uuid: u, annotator: annotator(), dwell_ms: ms,
                                  saves, patch });
    // sendBeacon survives the page going away; a normal fetch is cancelled
    if (navigator.sendBeacon) {
      navigator.sendBeacon("api/view", new Blob([body], { type: "application/json" }));
    } else {
      fetch("api/view", { method: "POST", keepalive: true,
        headers: { "Content-Type": "application/json" }, body });
    }
  }
  function flush() {
    accumulate();
    if (uuid && acc >= MIN_MS) send(uuid, Math.round(acc));
    uuid = null; acc = 0; since = 0; saves = 0; patch = null;
  }
  function enter(u) {
    flush();                                   // close out the previous shape first
    uuid = u; acc = 0; since = visible() ? Date.now() : 0; saves = 0; patch = null;
  }
  function init() {
    document.addEventListener("visibilitychange", () => {
      if (visible()) { if (uuid && !since) since = Date.now(); } else accumulate();
    });
    window.addEventListener("pagehide", flush);
  }
  return { init, enter, flush, noteSave };
})();

init();
