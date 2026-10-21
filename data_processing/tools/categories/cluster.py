"""HDBSCAN over the cached embeddings of the pass 1 labels: the text clusters.

Loads the embeddings `embed` cached at the chosen dimension and clusters them, writing
    out_full/cluster_text_assignments.jsonl   {"uuid", "label", "cluster_id"} (-1 = noise)

Default dim is 256: HDBSCAN is fast at this dim and the Matryoshka-truncated
text-embedding-3-small still keeps the category structure. Use --dim 1536 to cluster on the
full embedding (slow on a full-size dataset).

Usage:
    python -m data_processing.tools.categories.cluster
"""
import argparse
import json
import time

import hdbscan
import numpy as np

from data_processing.tools.categories.utils import OUT_FULL_DIR


def run_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    """Return per-row cluster labels (-1 = noise). Approximates cosine distance
    by L2-normalizing then using euclidean (HDBSCAN doesn't accept cosine directly).

    cluster_selection_epsilon: if > 0, forces clusters whose centroid distance
    is below this threshold to merge — useful for collapsing fine-grained
    micro-clusters into broader categories. Default 0 = standard HDBSCAN.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = features / norms
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
    )
    return clusterer.fit_predict(normed)


def cluster_labels(
    features: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    return run_hdbscan(
        features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256, choices=[256, 1536],
                        help="Cached embedding dimension to cluster on.")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    args = parser.parse_args()

    pass1_path = OUT_FULL_DIR / "pass1_freeform.jsonl"
    cache_path = OUT_FULL_DIR / f"embeddings_{args.dim}.npz"
    out_path = OUT_FULL_DIR / "cluster_text_assignments.jsonl"

    if not cache_path.exists():
        raise SystemExit(
            f"missing {cache_path}; run embed first."
        )

    labels_by_uuid: dict[str, str] = {}
    for line in pass1_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "category_freeform" in r:
            labels_by_uuid[r["uuid"]] = r["category_freeform"]
    ordered_uuids = sorted(labels_by_uuid)
    label_strings = [labels_by_uuid[u] for u in ordered_uuids]

    print(f"[cluster-text-full] loading cached embeddings ({cache_path}) ...")
    data = np.load(cache_path, allow_pickle=False)
    cached_uuids = list(data["uuids"])
    feats_all = data["features"]
    if cached_uuids != ordered_uuids:
        raise SystemExit(
            "cached uuid list does not match pass1_freeform.jsonl order; "
            "regenerate the cache."
        )
    print(f"   features: {feats_all.shape} ({feats_all.dtype})")

    print(f"[cluster-text-full] HDBSCAN (dim={args.dim}, "
          f"min_cluster_size={args.min_cluster_size}, "
          f"min_samples={args.min_samples}, "
          f"epsilon={args.cluster_selection_epsilon}) ...")
    t0 = time.time()
    cluster_ids = cluster_labels(
        feats_all,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
    )
    n_clusters = len(set(cluster_ids) - {-1})
    n_noise = int((cluster_ids == -1).sum())
    print(f"   clusters: {n_clusters}  noise: {n_noise}  ({time.time() - t0:.1f}s)")

    with out_path.open("w") as fh:
        for uid, lbl, cid in zip(ordered_uuids, label_strings, cluster_ids):
            fh.write(json.dumps({"uuid": uid, "label": lbl, "cluster_id": int(cid)}) + "\n")
    print(f"[cluster-text-full] wrote {out_path}")


if __name__ == "__main__":
    main()
