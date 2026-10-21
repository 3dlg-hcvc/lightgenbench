"""Point-sampled emission evaluation: score one folder of predictions on the released dataset.

    python -m evaluation.pointsampled.pipeline --pred_root <dir> --pred_format voxels|atlas \
        --dataset_root <released dataset> [--split test | --shas_file <uuids>] \
        --metadata <the GLB index> --glb_root <TexVerse-1K> --output_dir <dir>

For every shape, 50,000 points are sampled on the source GLB and the reference emission is read at each
point through its own part's material, as the dataset was baked (the texture alone, no factor, no strength).
The prediction is read at the same points by a nearest-neighbor lookup in its own representation:
`voxels` -- `<pred_root>/<uuid>.npz` with `coords` [N,3] on the 256^3 grid and `pred` [N,3] in [0,1], read
at the nearest occupied voxel; `atlas` -- `<pred_root>/<uuid>.png` or `<pred_root>/<uuid>/pred_emission.png`
on the dataset's UV atlas, read at the nearest atlas texel by 3D position (`<dataset_root>/<uuid>/atlas.npz`).
Six-view predictions are first back-projected to an atlas with `evaluation.converters.multiview`.

Writes `pointsampled_eval.parquet` (one row per shape: IoU of the emissive masks at 1/255, precision,
recall, MAE and PSNR over all points, MAE and PSNR on the reference's emissive points) and
`summary_eval.json` (their means over the scored shapes, as the paper computes them: `summarize`).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.converters.atlas import load_posocc, sample_atlas_emission
from evaluation.converters.voxels import (
    apply_voxel_transform, compute_voxel_transform, occupancy_overlap, sample_voxel_emission,
)
from evaluation.pointsampled.metrics import (
    baked_to_canonical_linear, emission_mask, point_iou, point_mae, point_precision, point_psnr,
    point_recall,
)
from evaluation.pointsampled.load_gt import load_parts, part_gt_linear, sample_parts

EPS_DEFAULT = 1.0 / 255.0
PRED_FORMATS = ("voxels", "atlas")


def _atomic_write(path, write_fn):
    """Write via `<path>.tmp` and os.replace, so a killed process never leaves a partial file.

    os.replace is atomic within a filesystem, and the temp file sits beside the destination, so
    the destination is either the old complete file or the new complete one -- never a truncated
    parquet in a run dir the table is built from.
    """
    tmp = f"{path}.tmp.{os.getpid()}"   # per-process: two writers must not share one temp name
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_pred_png(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0


def evaluate_one_shape(glb_path, pred_path, pred_format, atlas_npz=None, n_points: int = 50000,
                       eps: float = EPS_DEFAULT, seed: int = 0) -> dict:
    """Score one prediction of one shape at `n_points` surface points.

    The points are drawn on the GLB's kept parts and the reference is read at each through its own
    part's material (`load_gt.py`); both are mapped into the voxel frame of those parts' bounding box.
    The prediction is read at the same points by its converter: ``voxels`` (an npz with ``coords``
    and ``pred``) at the nearest occupied voxel, with ``align_fp``, the share of points that land on
    a predicted voxel; ``atlas`` (a PNG on the dataset's atlas) at the nearest atlas texel through
    `atlas_npz`'s position map. Everything is compared in the canonical linear space (`metrics.py`).
    The precision/recall, GT-mask and count columns come from the same two masks as the IoU.
    """
    if pred_format not in PRED_FORMATS:
        raise ValueError(f"pred_format must be one of {PRED_FORMATS}, got {pred_format!r}")
    parts = load_parts(glb_path)
    sample = sample_parts(parts, n_points, seed=seed)
    center, scale = compute_voxel_transform(parts.vertices)
    pvf = apply_voxel_transform(sample.points, center, scale)
    gt_lin = part_gt_linear(parts, sample)
    gt_mask = emission_mask(gt_lin, eps)

    if pred_format == "voxels":
        npz = np.load(pred_path)
        coords = npz["coords"]
        align_fp = occupancy_overlap(pvf, coords)[0]
        pred, miss = sample_voxel_emission(pvf, coords, npz["pred"])
    else:
        position_map, occ = load_posocc(atlas_npz)
        pred, miss = sample_atlas_emission(pvf, position_map, occ, _load_pred_png(pred_path))
    pred_lin = baked_to_canonical_linear(pred)
    pred_mask = emission_mask(pred_lin, eps)

    rec = {
        "iou": point_iou(gt_mask, pred_mask),
        "mae": point_mae(pred_lin, gt_lin),
        "psnr_surface": point_psnr(pred_lin, gt_lin),
        "psnr_emission": point_psnr(pred_lin, gt_lin, mask=(gt_mask | pred_mask)),
        "n_points": int(n_points), "n_gt_on": int(gt_mask.sum()),
        "miss_frac": miss,
    }
    if pred_format == "voxels":
        rec["align_fp"] = float(align_fp)
    # Counts are stored as float so they share the finite-mean summary; they are exact in float64.
    rec.update({
        "prec": point_precision(pred_mask, gt_mask),
        "rec": point_recall(pred_mask, gt_mask),
        "mae_gton": point_mae(pred_lin, gt_lin, mask=gt_mask),
        "psnr_gton": point_psnr(pred_lin, gt_lin, mask=gt_mask),
        "n_pred_on": float(pred_mask.sum()),
        "n_both": float(np.logical_and(pred_mask, gt_mask).sum()),
    })
    return rec


PSNR_CAP_DB = 40.0


def summarize(scored: pd.DataFrame) -> dict:
    """The paper's statistic for one run: each metric's mean over its finite values on the scored shapes,
    PSNR capped at 40 dB, and the GT-mask columns (`*_gton`) over the shapes whose reference has emissive
    points. A table cell is the mean and std (ddof=1) of these over seeds 0-4."""
    gton = scored[scored["n_gt_on"] > 0] if "n_gt_on" in scored.columns else scored
    out = {}
    for c in scored.columns:
        if scored[c].dtype.kind != "f":
            continue
        v = (gton if c.endswith("_gton") else scored)[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]  # NaN where a column has no value (mae_gton on an empty reference)
        if c.startswith("psnr"):
            v = np.minimum(v, PSNR_CAP_DB)  # point_psnr already caps at 40 dB; the paper's rule, restated
        out[c] = float(v.mean()) if v.size else float("nan")
    out["n_gt_on_shapes"] = int(len(gton))
    return out


def _shape_list(args) -> list:
    if args.shas_file:
        with open(args.shas_file) as f:
            return [line.strip() for line in f if line.strip()]
    with open(os.path.join(args.dataset_root, "splits.json")) as f:
        return json.load(f)[args.split]


def _glb_paths(metadata: str) -> pd.Series:
    """uuid -> glb_1k_path. Only that column is read: the index has a row per TexVerse id.
    A uuid listed twice keeps its first row."""
    glbs = pd.read_parquet(metadata, columns=["glb_1k_path"])["glb_1k_path"]
    return glbs[~glbs.index.duplicated(keep="first")]


def _prediction(pred_root: str, pred_format: str, uuid: str):
    if pred_format == "voxels":
        p = os.path.join(pred_root, f"{uuid}.npz")
        return p if os.path.exists(p) else None
    for p in (os.path.join(pred_root, f"{uuid}.png"), os.path.join(pred_root, uuid, "pred_emission.png")):
        if os.path.exists(p):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_root", required=True, help="one folder of predictions, one file per shape")
    ap.add_argument("--pred_format", choices=PRED_FORMATS, required=True)
    ap.add_argument("--dataset_root", required=True,
                    help="the released dataset: splits.json and <uuid>/atlas.npz")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--shas_file", default=None, help="score these uuids (one per line) instead of the split")
    ap.add_argument("--metadata", required=True, help="the GLB index (data/texverse_index.parquet): glb_1k_path per uuid")
    ap.add_argument("--glb_root", required=True, help="TexVerse-1K root: the reference is read from the GLBs")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_points", type=int, default=50000)
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT)
    ap.add_argument("--align_threshold", type=float, default=0.5,
                    help="voxels: a shape whose surface points land on predicted voxels less often than "
                         "this is left out of the summary (a GLB / voxel frame mismatch)")
    args = ap.parse_args(argv)

    uuids = _shape_list(args)
    glbs = _glb_paths(args.metadata)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[pointsampled] {len(uuids)} shapes, {args.pred_format} predictions from {args.pred_root}")

    records = []
    for uuid in uuids:
        rel = glbs.get(uuid)
        pred = _prediction(args.pred_root, args.pred_format, uuid)
        atlas = args.pred_format == "atlas"
        posmap = os.path.join(args.dataset_root, uuid, "atlas.npz")
        if rel is None or pd.isna(rel):
            records.append({"uuid": uuid, "error": "no GLB path in the metadata"})
            continue
        if pred is None:
            records.append({"uuid": uuid, "error": "no prediction"})
            continue
        if atlas and not os.path.exists(posmap):  # the atlas reader would score it NaN, not fail
            records.append({"uuid": uuid, "error": "no atlas.npz in the dataset"})
            continue
        try:
            rec = evaluate_one_shape(os.path.join(args.glb_root, rel), pred, args.pred_format,
                                     atlas_npz=posmap if atlas else None,
                                     n_points=args.n_points, eps=args.eps)
            records.append({"uuid": uuid, **rec})
        except Exception as e:  # skip + record failures, don't abort the batch
            records.append({"uuid": uuid, "error": repr(e)})

    df = pd.DataFrame(records)
    out_parquet = os.path.join(args.output_dir, "pointsampled_eval.parquet")
    # Written via a temp file + os.replace, so a killed run never leaves a truncated parquet.
    _atomic_write(out_parquet, lambda tmp: df.to_parquet(tmp))

    scored = df[df["error"].isna()] if "error" in df.columns else df
    n_err = len(df) - len(scored)
    if args.pred_format == "voxels" and "align_fp" in scored.columns:
        aligned = scored[scored["align_fp"] >= args.align_threshold]
        print(f"[pointsampled] {len(aligned)} scored | {len(scored) - len(aligned)} misaligned "
              f"(align_fp < {args.align_threshold}) | {n_err} without a prediction or failed")
        scored = aligned
    else:
        print(f"[pointsampled] {len(scored)} scored | {n_err} without a prediction or failed")

    summary = summarize(scored)
    summary.update({"n_shapes_summarized": int(len(scored)), "n_without_prediction_or_failed": n_err,
                    "pred_format": args.pred_format, "gt": "texture_only, per part"})

    def _dump_summary(tmp):
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
    _atomic_write(os.path.join(args.output_dir, "summary_eval.json"), _dump_summary)
    print(f"[pointsampled] wrote {out_parquet}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
