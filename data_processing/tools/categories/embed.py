"""Embed the pass 1 labels with text-embedding-3-small, at two sizes, into .npz caches.

Reads `out_full/pass1_freeform.jsonl` (`classify pass1`), embeds each label at dim=1536 (the
default) and dim=256 (the Matryoshka truncation text-embedding-3-* models support), and writes:

    out_full/embeddings_shards_1536/batch_NNNN.npz   (incremental, resume-safe)
    out_full/embeddings_1536.npz                       (final concat)
    out_full/embeddings_shards_256/batch_NNNN.npz
    out_full/embeddings_256.npz

Once both finals exist, `cluster` can pick either dimension without paying the API again.
It has no flags; it calls the API from the start.

Usage:
    python -m data_processing.tools.categories.embed
"""
import argparse
import json
import time
from typing import List

import numpy as np

from data_processing.tools.categories.utils import OUT_FULL_DIR, TEXT_EMBED_MODEL, make_openai_client


BATCH = 256


def _embed_chunk(client, model: str, labels: List[str], dimensions: int | None) -> np.ndarray:
    """One API call. Pass dimensions=None for the model's default (1536)."""
    kwargs = {"model": model, "input": list(labels)}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    resp = client.embeddings.create(**kwargs)
    return np.stack([np.asarray(d.embedding, dtype="float32") for d in resp.data])


def cache_at_dim(*, label_strings: List[str], ordered_uuids: List[str], dim: int) -> None:
    """Embed all labels at the given dim. Shard cache + final concat .npz."""
    out_dim = dim
    shards_dir = OUT_FULL_DIR / f"embeddings_shards_{out_dim}"
    shards_dir.mkdir(parents=True, exist_ok=True)
    final_path = OUT_FULL_DIR / f"embeddings_{out_dim}.npz"

    # If the final .npz already exists and matches the uuid list, skip entirely.
    if final_path.exists():
        try:
            data = np.load(final_path, allow_pickle=False)
            if (len(data["uuids"]) == len(ordered_uuids)
                    and (data["uuids"] == np.array(ordered_uuids)).all()
                    and data["features"].shape[1] == out_dim):
                print(f"[dim={out_dim}] final cache already complete: {final_path}")
                return
        except Exception:
            pass

    n = len(label_strings)
    n_batches = (n + BATCH - 1) // BATCH
    api_dim = None if dim == 1536 else dim
    print(f"[dim={out_dim}] embedding {n} labels in {n_batches} shards "
          f"(api dim arg = {api_dim}); cache dir {shards_dir}")

    client = make_openai_client()
    t0 = time.time()
    n_called = 0
    n_skipped = 0
    for bi in range(n_batches):
        shard_path = shards_dir / f"batch_{bi:04d}.npz"
        if shard_path.exists():
            n_skipped += 1
            continue
        chunk = label_strings[bi * BATCH:(bi + 1) * BATCH]
        feats = _embed_chunk(client, TEXT_EMBED_MODEL, chunk, dimensions=api_dim)
        np.savez_compressed(shard_path, features=feats.astype(np.float32))
        n_called += 1
        if n_called % 16 == 0:
            elapsed = time.time() - t0
            rps = n_called * BATCH / elapsed if elapsed > 0 else 0
            done = (bi + 1) * BATCH
            print(f"   batch {bi + 1}/{n_batches}: {min(done, n)}/{n} "
                  f"({rps:.0f} labels/s; {n_skipped} resumed-skipped)")

    print(f"[dim={out_dim}] all shards present ({n_called} new, {n_skipped} resumed). "
          f"stitching final array ...")
    chunks: List[np.ndarray] = []
    for bi in range(n_batches):
        chunks.append(np.load(shards_dir / f"batch_{bi:04d}.npz")["features"])
    feats_all = np.vstack(chunks).astype(np.float32)
    np.savez_compressed(final_path,
                         uuids=np.array(ordered_uuids),
                         features=feats_all)
    print(f"[dim={out_dim}] wrote {final_path}  shape={feats_all.shape}")


def main() -> None:
    # No flags: the parser is here so that -h prints this help instead of starting the run.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    pass1_path = OUT_FULL_DIR / "pass1_freeform.jsonl"
    labels_by_uuid: dict[str, str] = {}
    for line in pass1_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "category_freeform" in r:
            labels_by_uuid[r["uuid"]] = r["category_freeform"]
    print(f"[cache_embeddings] {len(labels_by_uuid)} labels loaded")

    ordered_uuids = sorted(labels_by_uuid)
    label_strings = [labels_by_uuid[u] for u in ordered_uuids]

    # Run BOTH dimensions sequentially. Resume-safe per batch.
    cache_at_dim(label_strings=label_strings, ordered_uuids=ordered_uuids, dim=1536)
    cache_at_dim(label_strings=label_strings, ordered_uuids=ordered_uuids, dim=256)
    print("[cache_embeddings] done. embeddings_1536.npz and embeddings_256.npz are ready.")


if __name__ == "__main__":
    main()
