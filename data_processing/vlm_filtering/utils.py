"""Helpers of the VLM check (`vlm_check.py`) and the tools the prompt was tuned with.

Nothing here knows the stage's paths or settings: `vlm_check.py` holds them and passes them
in. Sections: files and hashes; the OpenAI client; the derived images; the human labels;
scoring; the error page. What other stages share lives in `utils/`: the uuid list reader and
writer and the file hash in `utils/metadata_io.py`, the atlas's UV image and its texel count
in `utils/atlas_io.py`.
"""
from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
from openai import AsyncOpenAI
from PIL import Image, ImageSequence

from utils.metadata_io import sha256_of_file


# --- files and hashes -----------------------------------------------------------------

def nonempty(p: Path) -> bool:
    try:
        return p.stat().st_size > 0
    except OSError:
        return False


def atomic_save(im: Image.Image, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    im.save(tmp, format="PNG")
    os.replace(tmp, out_path)
    return out_path


def load_cache(out_jsonl) -> dict:
    out_jsonl = Path(out_jsonl)
    if not out_jsonl.exists():
        return {}
    cache = {}
    with open(out_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated final line from a killed process
            if "uuid" in row:
                cache[row["uuid"]] = row  # last write wins: a success overrides an earlier error
    return cache


def append_row(out_jsonl, row: dict) -> None:
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "a") as f:
        f.write(json.dumps(row) + "\n")


def terminate_partial_line(out_jsonl) -> None:
    """A killed process can leave an unterminated last line; appending after it would glue the
    next row onto the fragment and lose both. End the fragment; `load_cache` then skips it."""
    p = Path(out_jsonl)
    if p.exists() and p.stat().st_size and not p.read_bytes().endswith(b"\n"):
        with open(p, "ab") as f:
            f.write(b"\n")


def sha256_of_lines(lines) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, timeout=10).stdout.strip()
    except Exception:
        return "(unavailable)"


def write_sidecar(out, **fields) -> Path:
    out = Path(out)
    side = out.with_name(out.name + ".meta.json")
    side.write_text(json.dumps(fields, indent=2, sort_keys=True) + "\n")
    return side


# --- the OpenAI client ----------------------------------------------------------------
# The key is read, used, and never written anywhere.

#: Reasoning-family models reject `temperature`; everything else gets 0.0.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def load_api_key(key_path) -> str:
    if Path(key_path).exists():
        return Path(key_path).read_text().strip()
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    raise SystemExit(f"No API key: tried {key_path} and OPENAI_API_KEY.")


def make_client(key_path) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=load_api_key(key_path))


async def available_models(client) -> set:
    page = await client.models.list()
    return {m.id for m in page.data}


def resolve_model(available: set, model: str) -> str:
    """The paper names the model; a run on anything else is a different stage."""
    if model not in available:
        raise SystemExit(
            f"{model} is not available to this key (sample of what is: "
            f"{sorted(available)[:20]}). Nothing is substituted; stop and ask.")
    return model


def supports_temperature(model: str) -> bool:
    return not model.startswith(_REASONING_PREFIXES)


def image_part(path, detail: str) -> dict:
    """Every image the stage sends is a PNG (the stills and the UV map are PNGs, the
    orbits are packed into PNG grids first), so the data URL's type is always right."""
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}", "detail": detail}}


async def judge(client, model: str, prompt_text: str, image_paths: Iterable, schema,
                detail: str, max_retries: int, backoff_base: float = 1.0):
    """One user message: the prompt, then the images in order; strict JSON back."""
    content = [{"type": "text", "text": prompt_text}]
    content += [image_part(p, detail) for p in image_paths]
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_schema", "json_schema": {
            "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}},
    )
    if supports_temperature(model):
        kwargs["temperature"] = 0.0
    last = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(**kwargs)
            return schema.model_validate(json.loads(resp.choices[0].message.content))
        except Exception as e:  # transient API or validation errors
            last = e
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff_base * (2 ** attempt) * (1 + random.random() * 0.5))
    raise RuntimeError(f"judge failed after {max_retries} attempts: {last}")


# --- the derived images ---------------------------------------------------------------

def grid_from_webp(webp_path, out_path, cols: int = 3, cell: int = 512, n_frames: int = 6) -> Path:
    """Exactly the six orbit views, each a `cell`² tile, three per row: a five-frame webp is
    a broken render and must not pass as complete evidence."""
    frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(Image.open(webp_path))]
    if len(frames) != n_frames:
        raise ValueError(f"{webp_path}: {len(frames)} frames, expected {n_frames}")
    rows = n_frames // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    for i, fr in enumerate(frames):
        if fr.size != (cell, cell):
            fr = fr.resize((cell, cell))
        grid.paste(fr, ((i % cols) * cell, (i // cols) * cell))
    return atomic_save(grid, Path(out_path))


def emitting_texels_from_png(png_path) -> tuple:
    a = np.asarray(Image.open(png_path).convert("RGB"))
    lit = a.max(axis=-1) > 0
    return int(lit.sum()), int(lit.size)


# --- the human labels -----------------------------------------------------------------
# The labels come from the annotation tool's SQLite DBs; `labels` reads copies of them. A human
# row is the dashboard's definition: `verified=1`, not the auto filter, with a decision.

HUMAN_SQL = ("SELECT uuid, emissive, flags, style, emission_complete, annotator FROM items "
             "WHERE verified=1 AND (annotator IS NULL OR annotator<>'auto') AND emissive IS NOT NULL")
CSV_FIELDS = ["uuid", "emissive", "flags", "style", "emission_complete", "annotator"]


def load_human_rows(db_path) -> list:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(HUMAN_SQL).fetchall()
    finally:
        con.close()
    out = []
    for u, e, fl, st, ec, an in rows:
        try:
            flags = json.loads(fl or "[]")
        except (ValueError, TypeError):
            flags = []
        out.append(dict(uuid=u, emissive=bool(e), flags=flags, style=st,
                        emission_complete=None if ec is None else bool(ec), annotator=an))
    return out


def merge_rows(primary: list, secondary: list):
    """Union by uuid. A uuid in both keeps the primary row; a differing `emissive` is a conflict."""
    by = {r["uuid"]: r for r in primary}
    n_conflicts = 0
    for r in secondary:
        if r["uuid"] in by:
            if by[r["uuid"]]["emissive"] != r["emissive"]:
                n_conflicts += 1
        else:
            by[r["uuid"]] = r
    return list(by.values()), n_conflicts


def write_snapshot(rows, input_df, out_csv, sources=None, n_conflicts: int = 0) -> dict:
    """Keeps the rows whose uuid is in the list of record, frozen with its sha256."""
    in_set = dict(zip(input_df.uuid, input_df.judged))
    kept = sorted((r for r in rows if r["uuid"] in in_set), key=lambda r: r["uuid"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in kept:
            w.writerow(dict(uuid=r["uuid"], emissive=int(r["emissive"]), flags=";".join(r["flags"]),
                            style=r["style"] or "", annotator=r["annotator"] or "",
                            emission_complete="" if r["emission_complete"] is None else int(r["emission_complete"])))
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), git_sha=git_sha(),
                sources=sources or {}, n_conflicts=n_conflicts,
                n_human_rows=len(rows), n_in_input=len(kept),
                n_judged=sum(1 for r in kept if in_set[r["uuid"]]),
                counts={"pass": sum(1 for r in kept if r["emissive"]),
                        "reject": sum(1 for r in kept if not r["emissive"])},
                labels_sha256=sha256_of_file(out_csv))
    out_csv.with_name(out_csv.name + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def load_labels(csv_path) -> dict:
    out = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["uuid"]] = dict(emissive=r["emissive"] == "1",
                                  flags=[x for x in r["flags"].split(";") if x],
                                  annotator=r["annotator"] or None)
    return out


def membership(labels: dict, input_df, tune_source_uuids, test_source_uuids) -> dict:
    """The paper's protocol: the random study sample's halves tune and verify; every other
    label is a purposive sample and only ever a secondary card."""
    judged = set(input_df.uuid[input_df.judged])
    usable = set(labels) & judged
    tune_src, test_src = set(tune_source_uuids), set(test_source_uuids)
    if tune_src & test_src:
        raise SystemExit(f"the tune and test sources overlap on {len(tune_src & test_src)} uuids")
    return dict(tune=sorted(usable & tune_src), test=sorted(usable & test_src),
                other=sorted(usable - tune_src - test_src))


# --- scoring --------------------------------------------------------------------------
# "Has issue" is the positive class.

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def confusion(gt: dict, pred: dict, uuids) -> dict:
    """`gt[u]` and `pred[u]` are True for no issue; has issue is the positive class."""
    u = [x for x in uuids if x in gt and x in pred]
    tp = sum(1 for x in u if not gt[x] and not pred[x])
    tn = sum(1 for x in u if gt[x] and pred[x])
    fp = sum(1 for x in u if gt[x] and not pred[x])
    fn = sum(1 for x in u if not gt[x] and pred[x])
    n, nb, ng = len(u), tp + fn, tn + fp
    return dict(n=n, bad_rate=(nb / n if n else 0.0), acc=((tp + tn) / n if n else 0.0),
                acc_ci=wilson(tp + tn, n), sens=(tp / nb if nb else 0.0), sens_ci=wilson(tp, nb),
                spec=(tn / ng if ng else 0.0), spec_ci=wilson(tn, ng),
                prec=(tp / (tp + fp) if tp + fp else 0.0), prec_ci=wilson(tp, tp + fp),
                tp=tp, tn=tn, fp=fp, fn=fn)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant counts (binomial, p = 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def agreement(pred_a: dict, pred_b: dict, uuids) -> dict:
    u = [x for x in uuids if x in pred_a and x in pred_b]
    agree = sum(1 for x in u if pred_a[x] == pred_b[x])
    return dict(n=len(u), agree=agree, rate=(agree / len(u) if u else 0.0), ci=wilson(agree, len(u)))


def per_flag_prf(gt_flags: dict, pred_flags: dict, uuids, flag_values) -> dict:
    out = {}
    for flag in flag_values:
        tp = fp = fn = 0
        for x in uuids:
            g, p = flag in gt_flags.get(x, []), flag in pred_flags.get(x, [])
            tp += g and p
            fp += p and not g
            fn += g and not p
        out[flag] = dict(tp=tp, fp=fp, fn=fn, precision=(tp / (tp + fp) if tp + fp else 0.0),
                         recall=(tp / (tp + fn) if tp + fn else 0.0))
    return out


def load_predictions(jsonl) -> dict:
    """The successful rows of a prediction file, `no_issue` read from `verdict`."""
    out = {}
    for u, r in load_cache(jsonl).items():
        if "error" in r or "verdict" not in r:
            continue
        out[u] = dict(no_issue=r["verdict"] == "no_issue", flags=list(r.get("flags") or []),
                      rationale=r.get("rationale") or "")
    return out


def coverage(uuids, labels: dict, preds: dict) -> dict:
    """What the membership set is missing on either side; the card prints it and the sealed
    scoring refuses missing predictions."""
    return dict(n_set=len(uuids), n_labeled=sum(1 for u in uuids if u in labels),
                n_predicted=sum(1 for u in uuids if u in preds),
                missing_labels=[u for u in uuids if u not in labels],
                missing_predictions=[u for u in uuids if u in labels and u not in preds])


def _pct(x):
    return f"{100 * x:.1f}"


def _ci(ci):
    return f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def scorecard_md(title, r, flags, against=None, cov=None) -> str:
    lines = [f"# {title}", ""]
    if cov:
        lines += [f"set {cov['n_set']}, labeled {cov['n_labeled']}, predicted {cov['n_predicted']}; "
                  f"missing labels {len(cov['missing_labels'])}, missing predictions "
                  f"{len(cov['missing_predictions'])}", ""]
    lines += [f"n = {r['n']}, bad-rate {_pct(r['bad_rate'])} %", "",
              "| metric | value | Wilson 95 % |", "|---|---|---|",
              f"| accuracy | {_pct(r['acc'])} | {_ci(r['acc_ci'])} |",
              f"| sensitivity (defects caught) | {_pct(r['sens'])} | {_ci(r['sens_ci'])} |",
              f"| specificity (good kept) | {_pct(r['spec'])} | {_ci(r['spec_ci'])} |",
              f"| precision (of has-issue calls) | {_pct(r['prec'])} | {_ci(r['prec_ci'])} |", "",
              "| | human: has issue | human: no issue |", "|---|---|---|",
              f"| VLM: has issue | {r['tp']} | {r['fp']} |",
              f"| VLM: no issue | {r['fn']} | {r['tn']} |", "",
              "| flag (exploratory) | tp | fp | fn | precision | recall |", "|---|---|---|---|---|---|"]
    lines += [f"| {f} | {v['tp']} | {v['fp']} | {v['fn']} | {_pct(v['precision'])} | {_pct(v['recall'])} |"
              for f, v in flags.items()]
    if against:
        a = against
        lines += ["", f"## Against `{a['name']}`", "",
                  f"agreement {a['agreement']['agree']}/{a['agreement']['n']} = "
                  f"{_pct(a['agreement']['rate'])} % {_ci(a['agreement']['ci'])}",
                  f"this right / other wrong b = {a['b']}, the reverse c = {a['c']}, "
                  f"McNemar exact p = {a['p']:.4f}",
                  f"other file's accuracy on the same items: {_pct(a['other_acc'])} %"]
    return "\n".join(lines) + "\n"


# --- the error page -------------------------------------------------------------------
# The tune-set scorecard plus every shape the VLM got wrong, with the images that decide the
# call and both sides' reasoning: the numbers never say WHICH kinds of asset fail, and that is
# what decides the next edit. Thumbnails are downscaled JPEGs embedded in the page (a
# page of full-size images is too large to open).

#: (label, index into `vlm_check.ORDER`)
VIEWS = [("常规渲染", 0), ("暗箱 OFF", 3), ("暗箱 ON", 4), ("发光轨道", 6), ("发光 UV", 7)]

CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b66;--line:#e3e3df;--card:#fff;--bad:#b4342c;--good:#2f6f4f}
@media(prefers-color-scheme:dark){:root{--bg:#161615;--fg:#eceae4;--mut:#9a978f;--line:#2f2f2c;--card:#1e1e1c;--bad:#e2726a;--good:#6fbf95}}
*{box-sizing:border-box}
body{margin:0;padding:32px 24px 80px;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto} h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:40px 0 12px;padding-top:18px;border-top:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 28px}
table{border-collapse:collapse;margin:0 0 8px} td,th{border:1px solid var(--line);padding:7px 13px;text-align:right}
th{font-weight:600;text-align:left} td:first-child,th:first-child{text-align:left}
.m{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 6px}
.m div{border:1px solid var(--line);border-radius:9px;padding:9px 15px;background:var(--card);min-width:132px}
.m b{display:block;font-size:21px} .m span{color:var(--mut);font-size:12.5px}
.card{border:1px solid var(--line);border-radius:11px;background:var(--card);padding:15px;margin:0 0 16px}
.hd{display:flex;flex-wrap:wrap;gap:9px;align-items:baseline;margin-bottom:9px}
.uuid{font-family:ui-monospace,Menlo,monospace;font-size:13px}
.tag{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:2px 9px;color:var(--mut)}
.tag.h{color:var(--bad);border-color:currentColor} .tag.a{color:var(--good);border-color:currentColor}
.rat{color:var(--mut);font-size:13.5px;margin:2px 0 11px}
.imgs{display:flex;gap:9px;overflow-x:auto;padding-bottom:4px}
figure{margin:0;flex:0 0 auto} figure img{height:240px;border-radius:6px;display:block;background:#000}
figcaption{font-size:11.5px;color:var(--mut);margin-top:4px;text-align:center}
"""


def _thumb(path, thumb_h) -> str:
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return ""
    if im.height > thumb_h:
        im = im.resize((max(1, round(im.width * thumb_h / im.height)), thumb_h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _views(uuid, shard, resolve, thumb_h):
    ev = resolve(uuid, shard)
    if ev is None:
        return [(label, "") for label, _ in VIEWS]
    return [(label, _thumb(ev.images[i], thumb_h)) for label, i in VIEWS]


def build_page(uuids, labels, preds, meta, resolve, thumb_h=240, show_all=False) -> str:
    gt = {u: labels[u]["emissive"] for u in uuids if u in labels}
    pe = {u: preds[u]["no_issue"] for u in uuids if u in preds}
    r = confusion(gt, pe, uuids)
    fn = sorted(u for u in uuids if u in gt and u in pe and not gt[u] and pe[u])
    fp = sorted(u for u in uuids if u in gt and u in pe and gt[u] and not pe[u])
    tp = sorted(u for u in uuids if u in gt and u in pe and not gt[u] and not pe[u]) if show_all else []
    tn = sorted(u for u in uuids if u in gt and u in pe and gt[u] and pe[u]) if show_all else []
    shown = fn + fp + tp + tn
    with ThreadPoolExecutor(12) as ex:
        thumbs = dict(zip(shown, ex.map(lambda u: _views(u, meta[u]["shard"], resolve, thumb_h), shown)))

    def card(u):
        g, p = labels[u], preds[u]
        imgs = "".join(f'<figure><img src="{src}" alt="{lab}"><figcaption>{lab}</figcaption></figure>'
                       for lab, src in thumbs[u] if src)
        hflags = ", ".join(g["flags"]) or "（未填 flag）"
        aflags = ", ".join(p["flags"]) or "（无 flag）"
        return (f'<div class="card"><div class="hd"><span class="uuid">{u[:12]}</span>'
                f'<span class="tag">{html.escape(meta[u]["category"])}</span>'
                f'<span class="tag">{html.escape(str(g.get("annotator")))}</span>'
                f'<span class="tag h">人：{"有问题 · " + html.escape(hflags) if not g["emissive"] else "没问题"}</span>'
                f'<span class="tag a">VLM：{"有问题 · " + html.escape(aflags) if not p["no_issue"] else "没问题"}</span></div>'
                f'<div class="rat">VLM 的理由：{html.escape(p.get("rationale") or "")}</div>'
                f'<div class="imgs">{imgs}</div></div>')

    body = ['<div class="wrap"><h1>调优集错例总览</h1>',
            f'<p class="sub">{r["n"]} 个有标注且有判断的样本 · VLM 原始判断（无整体发光规则，不含人工优先）</p>',
            '<table><tr><th></th><th>人说有问题</th><th>人说没问题</th></tr>'
            f'<tr><th>VLM 说有问题</th><td>{r["tp"]}</td><td>{r["fp"]} 误报</td></tr>'
            f'<tr><th>VLM 说没问题</th><td>{r["fn"]} 漏报</td><td>{r["tn"]}</td></tr></table>',
            '<div class="m">'
            f'<div><b>{r["acc"]:.3f}</b><span>准确率</span></div>'
            f'<div><b>{r["sens"]:.3f}</b><span>敏感度 · 缺陷抓到多少</span></div>'
            f'<div><b>{r["spec"]:.3f}</b><span>特异度 · 好的没被打扰</span></div>'
            f'<div><b>{r["prec"]:.3f}</b><span>精确率 · 报警有多准</span></div>'
            f'<div><b>{r["bad_rate"]:.3f}</b><span>bad 率</span></div></div>',
            f'<h2>漏报 · {len(fn)} 个（人说有问题，VLM 放过了）</h2>']
    body += [card(u) for u in fn]
    body += [f'<h2>误报 · {len(fp)} 个（人说没问题，VLM 报警了）</h2>']
    body += [card(u) for u in fp]
    if show_all:
        body += [f'<h2>判对 · 有问题 · {len(tp)} 个（人和 VLM 都说有问题）</h2>']
        body += [card(u) for u in tp]
        body += [f'<h2>判对 · 没问题 · {len(tn)} 个（人和 VLM 都说没问题）</h2>']
        body += [card(u) for u in tn]
    body.append("</div>")
    return ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>调优集错例总览</title><style>" + CSS + "</style>" + "".join(body))
