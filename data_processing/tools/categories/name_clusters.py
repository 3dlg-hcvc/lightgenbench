"""Name every text cluster with the VLM: one name and one definition per cluster.

Reads `out_full/cluster_text_assignments.jsonl` (`cluster`) and, for each cluster but the
noise, shows gpt-4o-mini up to 5 exemplar composites and up to 30 of its pass 1 labels (prompt
`prompts/cluster_name.txt`). Writes
    out_full/taxonomy.json   the fine-grained taxonomy, one entry per cluster
It has no flags; it calls the API from the start.

Usage:
    python -m data_processing.tools.categories.name_clusters
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from data_processing.tools.categories.utils import (
    COMPOSITES_FULL, OUT_FULL_DIR, SAMPLING_SEED, VLM_MODEL, ClusterNamingOutput, VlmClient,
    make_openai_client,
)


def pick_exemplars(member_uuids: List[str], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    if len(member_uuids) <= n:
        return list(member_uuids)
    return rng.sample(member_uuids, n)


def name_cluster(
    *,
    exemplar_uuids: List[str],
    member_labels: Optional[List[str]],
    composites_dir: Path,
    prompt: str,
    vlm: VlmClient,
) -> ClusterNamingOutput:
    image_paths = [composites_dir / f"{uid}.png" for uid in exemplar_uuids
                   if (composites_dir / f"{uid}.png").exists()]
    full_prompt = prompt
    if member_labels:
        labels_block = "\n".join(f"  - {l}" for l in member_labels)
        full_prompt += f"\n\nFree-form labels in this cluster:\n{labels_block}\n"
    return vlm.call_structured(
        prompt=full_prompt, image_paths=image_paths, schema=ClusterNamingOutput,
    )


def name_all_clusters(
    *,
    assignments: Dict[str, int],
    labels_by_uuid: Dict[str, str],
    composites_dir: Path,
    prompt_path: Path,
    vlm: VlmClient,
    n_exemplars: int,
    max_member_labels: int,
    seed: int,
) -> List[Dict]:
    prompt = prompt_path.read_text()
    members_per_cluster: Dict[int, List[str]] = {}
    for uid, cid in assignments.items():
        members_per_cluster.setdefault(cid, []).append(uid)

    taxonomy: List[Dict] = []
    for cid, members in sorted(members_per_cluster.items()):
        if cid == -1:
            continue
        exemplar_uuids = pick_exemplars(sorted(members), n=n_exemplars, seed=seed + cid)
        member_labels = [labels_by_uuid[uid] for uid in members if uid in labels_by_uuid]
        if max_member_labels and len(member_labels) > max_member_labels:
            rng = random.Random(seed + cid + 1)
            member_labels = rng.sample(member_labels, max_member_labels)
        try:
            named: ClusterNamingOutput = name_cluster(
                exemplar_uuids=exemplar_uuids, member_labels=member_labels or None,
                composites_dir=composites_dir, prompt=prompt, vlm=vlm,
            )
        except Exception as e:
            taxonomy.append({"cluster_id": int(cid), "error": str(e)})
            continue
        taxonomy.append({
            "cluster_id": int(cid),
            "name": named.name,
            "definition": named.definition,
            "exemplar_uuids": exemplar_uuids,
            "member_count": len(members),
            "member_labels": member_labels,
        })
    return taxonomy


def main() -> None:
    # No flags: the parser is here so that -h prints this help instead of starting the run.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    assignments_path = OUT_FULL_DIR / "cluster_text_assignments.jsonl"
    taxonomy_path = OUT_FULL_DIR / "taxonomy.json"

    assignments: dict[str, int] = {}
    labels_by_uuid: dict[str, str] = {}
    for line in assignments_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        assignments[r["uuid"]] = int(r["cluster_id"])
        if "label" in r:
            labels_by_uuid[r["uuid"]] = r["label"]
    print(f"[name-clusters-full] {len(assignments)} uuids; "
          f"{len(set(assignments.values()) - {-1})} non-noise clusters to name")

    composites_dir = COMPOSITES_FULL
    vlm = VlmClient(client=make_openai_client(), model=VLM_MODEL)
    taxonomy = name_all_clusters(
        assignments=assignments,
        labels_by_uuid=labels_by_uuid,
        composites_dir=composites_dir,
        prompt_path=Path(__file__).parent / "prompts" / "cluster_name.txt",
        vlm=vlm,
        n_exemplars=5,
        max_member_labels=30,
        seed=SAMPLING_SEED,
    )
    taxonomy_path.write_text(json.dumps(taxonomy, indent=2))
    print(f"[name-clusters-full] wrote {len(taxonomy)} categories to {taxonomy_path}")


if __name__ == "__main__":
    main()
