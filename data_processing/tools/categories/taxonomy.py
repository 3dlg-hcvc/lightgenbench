"""The taxonomy merges: the fine-grained cluster names become the broad categories.

Four LLM steps, each a subcommand; all read and write `out_full/`. `merge` runs first, then
`cleanup`; `simplify` and `shapenet` are two alternative renames of cleanup's broad names, and each
rewrites `merge_mapping.json` (its values are the names of whichever of the two ran last).

  merge     the named clusters of `taxonomy.json` (`name_clusters`) -> broad categories
            (the prompt aims at 80-150) in one call (default gpt-4o-mini): `merged_taxonomy.json` and the
            per-original-name `merge_mapping.json`. Every original name must land in exactly one
            group; the call is checked and the drops are reported.
  cleanup   one call (gpt-5) gives every original name that `merge` dropped or put in two groups
            exactly one of the broad names; rewrites `merged_taxonomy.json` and
            `merge_mapping.json` in place.
  simplify  strips the cosmetic adjectives from the broad names (futuristic_weapon -> weapon,
            keeping table_lamp, robot_character): `simplified_taxonomy.json`, `simplify_mapping.json`
            (broad -> simplified), and `merge_mapping.json` rewritten to point at the simplified
            names.
  shapenet  the alternative to `simplify`: the broad names in ShapeNet's naming
            style (short, lowercase, one common noun): `shapenet_taxonomy.json`,
            `shapenet_mapping.json` (broad -> ShapeNet-style), and `merge_mapping.json` rewritten.
            `relabel` maps the pass 2 rows through `shapenet_mapping.json` to the final names.

`classify pass2` labels in the broad-name space of `merge` + `cleanup`. The prompts are the
`*_PROMPT` constants below.

Usage:
    python -m data_processing.tools.categories.taxonomy merge [--model gpt-4o-mini]
    python -m data_processing.tools.categories.taxonomy cleanup [--model gpt-5]
    python -m data_processing.tools.categories.taxonomy simplify [--model gpt-5]
    python -m data_processing.tools.categories.taxonomy shapenet [--model gpt-5]
"""
import argparse
import json
from collections import defaultdict
from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from data_processing.tools.categories.utils import (
    OUT_FULL_DIR, VlmClient, make_openai_client, run_subcommand,
)


# ---------------------------------------------------------------------------------------------
# merge: the fine-grained clusters -> broad categories

class MergedCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    definition: str
    merged_from: List[str]


class MergedTaxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    categories: List[MergedCategory]


MERGE_PROMPT = """\
You are an expert data taxonomist. Below is a flat list of {N} fine-grained
3D-object categories discovered by clustering. Your job is to merge them
into BROADER categories that group semantically equivalent or closely
related categories.

HARD RULES on the merged-category `name`:
- It MUST be a SINGLE noun phrase representing ONE concept.
- ABSOLUTELY NO compound names containing "and", "&", "+", "or", commas,
  or two unrelated objects glued together.
- BAD names (DO NOT produce these):
    "cartoon_characters_and_animals", "robot_and_animatronic_figures",
    "drones_and_uavs", "fireplaces_and_fountains",
    "vases_and_jewelry", "anatomy_and_mortuary_objects"
- GOOD names: "robot_figure", "fantasy_weapon", "lamp", "vehicle",
  "drone", "vase".
- If two source clusters are genuinely different concepts that don't
  share a clean parent noun, KEEP THEM AS SEPARATE merged categories —
  don't invent a joint name to force them together.

OTHER RULES:
- Every original `name` MUST appear in exactly one merged group's
  `merged_from` list. Don't drop any; don't put any in two groups.
- Provide a one-sentence `definition`.
- Preserve meaningful theme qualifiers when they cleanly map to a single
  noun: "futuristic_weapon" / "fantasy_weapon" can both exist as separate
  categories; "vintage_car" / "classic_car" / "sports_car" can be split
  by era/style.
- Aim for 80-150 merged categories.
- Examples of merges that SHOULD happen:
    All "robot_*" / "robotic_*" toy/figurine variants → "robot_figure"
    All "cartoon_*" character variants → "cartoon_character"
    All sword/axe/dagger/spear under fantasy → "fantasy_weapon"
    Exact-name duplicates (two "sports_car" clusters) → one cluster

Original {N} categories (name | member_count | definition):

{listing}

Output a JSON object with a "categories" list, each containing
{{"name": ..., "definition": ..., "merged_from": [...]}}.
"""


def _format_listing(entries: List[Dict]) -> str:
    lines = []
    for e in entries:
        defn = (e.get("definition") or "")[:90]
        lines.append(f"  - {e['name']} | {e['member_count']} | {defn}")
    return "\n".join(lines)


def merge_main(argv=None) -> None:
    """One call merges taxonomy.json's names -> merged_taxonomy.json, merge_mapping.json."""
    parser = argparse.ArgumentParser(prog="python -m data_processing.tools.categories.taxonomy merge")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Use gpt-4o for stronger reasoning; gpt-4o-mini is the cheap default.")
    args = parser.parse_args(argv)

    tax_path = OUT_FULL_DIR / "taxonomy.json"
    out_path = OUT_FULL_DIR / "merged_taxonomy.json"
    mapping_path = OUT_FULL_DIR / "merge_mapping.json"

    raw = json.loads(tax_path.read_text())
    entries = [e for e in raw if "name" in e]
    print(f"[merge] {len(entries)} input categories")

    prompt = MERGE_PROMPT.format(N=len(entries), listing=_format_listing(entries))
    print(f"[merge] prompt size: ~{len(prompt) // 4} tokens (rough)")

    vlm = VlmClient(client=make_openai_client(), model=args.model)
    print(f"[merge] calling {args.model} ...")
    # gpt-5 only accepts temperature=1; gpt-4o-mini accepts 0.0.
    temp = 1.0 if args.model.startswith("gpt-5") else 0.0
    result: MergedTaxonomy = vlm.call_structured(
        prompt=prompt, image_paths=[], schema=MergedTaxonomy, temperature=temp,
    )
    print(f"[merge] got {len(result.categories)} merged categories")

    # Verify every original name is accounted for exactly once.
    original = {e["name"]: e["member_count"] for e in entries}
    seen: Dict[str, int] = {}
    for cat in result.categories:
        for n in cat.merged_from:
            seen[n] = seen.get(n, 0) + 1

    missing = sorted(set(original) - set(seen))
    duplicated = {n: c for n, c in seen.items() if c > 1}
    extra = sorted(set(seen) - set(original))

    print(f"\n[merge] verification:")
    print(f"   originals:     {len(original)}")
    print(f"   covered:       {len(seen)}")
    print(f"   missing:       {len(missing)}")
    print(f"   duplicated:    {len(duplicated)}")
    print(f"   extra (hallucinated): {len(extra)}")
    if missing:
        print(f"   first 10 missing: {missing[:10]}")
    if duplicated:
        print(f"   first 10 duplicated: {list(duplicated.items())[:10]}")
    if extra:
        print(f"   first 10 extra: {extra[:10]}")

    # Write merged taxonomy with member-count rollup for each merged category.
    merged_with_counts = []
    for cat in result.categories:
        total = sum(original.get(n, 0) for n in cat.merged_from)
        merged_with_counts.append({
            "name": cat.name,
            "definition": cat.definition,
            "merged_from": cat.merged_from,
            "member_count": total,
        })
    merged_with_counts.sort(key=lambda c: -c["member_count"])
    out_path.write_text(json.dumps(merged_with_counts, indent=2))
    print(f"\n[merge] wrote {out_path}")

    # Per-original-name mapping (one entry per original, even unmapped ones).
    mapping = {}
    for cat in result.categories:
        for n in cat.merged_from:
            mapping[n] = cat.name
    for n in missing:
        mapping[n] = "__unmapped__"
    mapping_path.write_text(json.dumps(mapping, indent=2))
    print(f"[merge] wrote {mapping_path}")


# ---------------------------------------------------------------------------------------------
# cleanup: every original name in exactly one broad category

class Pick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    original_name: str
    broad_name: str


class PickList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    picks: List[Pick]


CLEANUP_PROMPT = """\
You previously merged 552 fine-grained 3D-object categories into the
following {N_BROAD} broad categories (single-noun names, no compounds):

{broad_listing}

The categories below need cleanup: each line is one ORIGINAL fine-grained
category that either was missed entirely OR was placed in multiple broad
groups. For each, pick EXACTLY ONE broad name from the list above —
whichever fits best semantically. Use only names from the list; do not
invent new ones.

Items needing a canonical broad assignment ({N_ITEMS}) (name | member_count | definition | currently_assigned_to):

{items_listing}

Output JSON: {{"picks": [{{"original_name": ..., "broad_name": ...}}, ...]}}
Every original_name above MUST appear exactly once in the output.
"""


def _format_broad(merged: List[Dict]) -> str:
    return "\n".join(
        f"  - {c['name']} | {c['member_count']} members | {c['definition'][:90]}"
        for c in merged
    )


def _format_items(orig_lookup: Dict[str, Dict], items: List[str],
                   current_assignments: Dict[str, List[str]]) -> str:
    lines = []
    for n in items:
        e = orig_lookup.get(n, {})
        cnt = e.get("member_count", 0)
        defn = (e.get("definition") or "")[:80]
        assigned = current_assignments.get(n, [])
        ass_str = ", ".join(assigned) if assigned else "(unmapped)"
        lines.append(f"  - {n} | {cnt} | {defn} | currently in: {ass_str}")
    return "\n".join(lines)


def cleanup_main(argv=None) -> None:
    """The unmapped and duplicated names get one broad name each; rewrites both files in place."""
    parser = argparse.ArgumentParser(prog="python -m data_processing.tools.categories.taxonomy cleanup")
    parser.add_argument("--model", default="gpt-5")
    args = parser.parse_args(argv)

    tax_path = OUT_FULL_DIR / "taxonomy.json"
    merged_path = OUT_FULL_DIR / "merged_taxonomy.json"
    mapping_path = OUT_FULL_DIR / "merge_mapping.json"

    originals = {e["name"]: e for e in json.loads(tax_path.read_text()) if "name" in e}
    merged = json.loads(merged_path.read_text())
    mapping = json.loads(mapping_path.read_text())

    # Discover state by re-reading the merged_from lists rather than trusting
    # the (lossy) flat mapping.
    multi_assignments: Dict[str, List[str]] = defaultdict(list)
    for cat in merged:
        for n in cat["merged_from"]:
            multi_assignments[n].append(cat["name"])

    unmapped = sorted(set(originals) - set(multi_assignments))
    duplicated = sorted([n for n, cats in multi_assignments.items() if len(cats) > 1])
    items = unmapped + duplicated
    print(f"[cleanup] originals: {len(originals)}")
    print(f"[cleanup] unmapped: {len(unmapped)}")
    print(f"[cleanup] duplicated: {len(duplicated)}")
    print(f"[cleanup] items to resolve: {len(items)}")
    if not items:
        print("[cleanup] nothing to do.")
        return

    valid_broad = {c["name"] for c in merged}
    prompt = CLEANUP_PROMPT.format(
        N_BROAD=len(merged),
        N_ITEMS=len(items),
        broad_listing=_format_broad(merged),
        items_listing=_format_items(originals, items, multi_assignments),
    )
    print(f"[cleanup] prompt size: ~{len(prompt) // 4} tokens")

    vlm = VlmClient(client=make_openai_client(), model=args.model)
    temp = 1.0 if args.model.startswith("gpt-5") else 0.0
    print(f"[cleanup] calling {args.model} ...")
    result: PickList = vlm.call_structured(
        prompt=prompt, image_paths=[], schema=PickList, temperature=temp,
    )

    # Validate.
    canonical: Dict[str, str] = {}
    invalid_broad = []
    for p in result.picks:
        if p.broad_name in valid_broad and p.original_name in set(items):
            canonical[p.original_name] = p.broad_name
        else:
            invalid_broad.append((p.original_name, p.broad_name))
    still_missing = sorted(set(items) - set(canonical))

    print(f"[cleanup] resolved: {len(canonical)} / {len(items)}")
    print(f"[cleanup] invalid_broad: {len(invalid_broad)}; still_missing: {len(still_missing)}")
    if invalid_broad:
        print(f"   first 5 invalid: {invalid_broad[:5]}")
    if still_missing:
        print(f"   first 5 still missing: {still_missing[:5]}")

    # Rebuild merged_from lists from scratch using canonical assignments
    # for items, and the existing single assignments for others.
    final_assignment: Dict[str, str] = {}
    for n, cats in multi_assignments.items():
        if n in canonical:
            final_assignment[n] = canonical[n]
        elif len(cats) == 1:
            final_assignment[n] = cats[0]
    for n in unmapped:
        if n in canonical:
            final_assignment[n] = canonical[n]
    # Items still missing remain unmapped.

    by_name: Dict[str, Dict] = {c["name"]: {**c, "merged_from": [], "member_count": 0}
                                 for c in merged}
    for n, broad in final_assignment.items():
        by_name[broad]["merged_from"].append(n)
        by_name[broad]["member_count"] += originals.get(n, {}).get("member_count", 0)

    new_merged = sorted(by_name.values(), key=lambda c: -c["member_count"])
    merged_path.write_text(json.dumps(new_merged, indent=2))

    new_mapping: Dict[str, str] = {n: final_assignment.get(n, "__unmapped__")
                                    for n in originals}
    mapping_path.write_text(json.dumps(new_mapping, indent=2))

    n_unmapped_after = sum(1 for v in new_mapping.values() if v == "__unmapped__")
    n_empty_cats = sum(1 for c in new_merged if c["member_count"] == 0)
    print(f"[cleanup] post-cleanup:")
    print(f"   total originals:  {len(new_mapping)}")
    print(f"   still unmapped:   {n_unmapped_after}")
    print(f"   empty merged cats: {n_empty_cats}")
    print(f"   wrote {merged_path}")
    print(f"   wrote {mapping_path}")


# ---------------------------------------------------------------------------------------------
# simplify and shapenet: two renames of the broad names (shared schema)

class Rename(BaseModel):
    model_config = ConfigDict(extra="forbid")
    old_name: str
    new_name: str


class RenameList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    renames: List[Rename]


SIMPLIFY_PROMPT = """\
You will simplify the names of {N} broad 3D-object categories. For each
input name, strip STRICTLY-COSMETIC adjectives but keep every other
qualifier (including type/genre/material qualifiers).

WHAT TO STRIP (style/era/aesthetic only — these don't change WHAT the
object is, only how it looks):
    futuristic, fantasy, vintage, classic, retro, modern, medieval,
    steampunk, cyberpunk, ornate, ruined, decorative, glowing, magical,
    mystical, sci_fi, stylized.

  Examples (DO this):
    "futuristic_weapon"          -> "weapon"
    "fantasy_creature"           -> "fantasy_creature"   (keep — fantasy IS a type here? NO actually strip:) -> "creature"
    "vintage_television"         -> "television"
    "futuristic_device"          -> "device"
    "decorative_sculpture"       -> "sculpture"
    "stylized_character"         -> "character"
    "ruined_building"            -> "building"

WHAT TO KEEP — these qualifiers are ESSENTIAL because they define the
TYPE/GENRE/MATERIAL/PURPOSE of the object, not just style:
    robot, robotic, cartoon, anime, animated, animatronic, horror,
    fantasy (when no clear non-fantasy alternative), human, alien,
    architectural, anatomical, urban, planetary, spacecraft, kitchen,
    interior, table, floor, pendant, ceiling, computer, gaming, arcade,
    treasure, candle, audio, video_game, retro_computer (retro_computer
    is a specific genre, keep it as "retro_computer"), athletic,
    musical, jack_o_lantern, christmas, pixel, virtual_reality,
    space, fire, towel, military, police, religious, geologic, fish,
    tree, board, magic.

  Examples (DO NOT collapse these):
    "robot_character"            -> "robot_character"    (robot is essential type, NOT style)
    "cartoon_character"          -> "cartoon_character"  (cartoon is essential type)
    "anime_character"            -> "anime_character"
    "animatronic_character"      -> "animatronic_character"
    "robot_figure"               -> "robot_figure"
    "robot_head"                 -> "robot_head"
    "architectural_model"        -> "architectural_model" (architectural defines what it's a model OF)
    "spacecraft_model"           -> "spacecraft_model"
    "tree_model"                 -> "tree_model"
    "interior_scene"             -> "interior_scene"
    "table_lamp"                 -> "table_lamp"
    "computer_monitor"           -> "computer_monitor"
    "horror_character"           -> "horror_character"

SPECIAL CASES:
- "fantasy_*" — strip "fantasy" UNLESS the base noun "weapon"/"creature"
  has no specific non-fantasy version in the broader corpus. Default:
  strip "fantasy".
- "decorative_*" — strip "decorative" because it's purely stylistic.
- Single-noun names already stay unchanged: weapon, drone, helmet, table.
- "_model" suffix — keep the qualifier (DO NOT strip to bare "model").
  If the result would be the bare word "model", instead keep the full
  original name.

If two input names collapse to the same simplified name (e.g.
"futuristic_weapon" and "fantasy_weapon" both become "weapon"), that's
expected.

Input categories (one per line):
{listing}

Output JSON: {{"renames": [{{"old_name": ..., "new_name": ...}}, ...]}}
Every input name MUST appear exactly once in the output.
"""


def simplify_main(argv=None) -> None:
    """Cosmetic adjectives off the broad names -> simplified_taxonomy.json, simplify_mapping.json."""
    parser = argparse.ArgumentParser(prog="python -m data_processing.tools.categories.taxonomy simplify")
    parser.add_argument("--model", default="gpt-5")
    args = parser.parse_args(argv)

    merged_path = OUT_FULL_DIR / "merged_taxonomy.json"
    mapping_path = OUT_FULL_DIR / "merge_mapping.json"
    simplified_path = OUT_FULL_DIR / "simplified_taxonomy.json"
    rename_map_path = OUT_FULL_DIR / "simplify_mapping.json"

    merged = json.loads(merged_path.read_text())
    mapping = json.loads(mapping_path.read_text())

    print(f"[simplify] {len(merged)} broad categories")
    listing = "\n".join(f"  - {c['name']}" for c in merged)
    prompt = SIMPLIFY_PROMPT.format(N=len(merged), listing=listing)

    vlm = VlmClient(client=make_openai_client(), model=args.model)
    temp = 1.0 if args.model.startswith("gpt-5") else 0.0
    print(f"[simplify] calling {args.model} ...")
    result: RenameList = vlm.call_structured(
        prompt=prompt, image_paths=[], schema=RenameList, temperature=temp,
    )

    # Build rename map and verify.
    rename_map: Dict[str, str] = {}
    for r in result.renames:
        rename_map[r.old_name] = r.new_name
    missing = sorted(set(c["name"] for c in merged) - set(rename_map))
    extra = sorted(set(rename_map) - set(c["name"] for c in merged))
    print(f"[simplify] renamed: {len(rename_map)}; missing: {len(missing)}; extra: {len(extra)}")
    if missing:
        print(f"   first 10 missing: {missing[:10]}")
        # Default: keep the original name for any missing ones.
        for n in missing:
            rename_map[n] = n
    if extra:
        print(f"   first 10 extra: {extra[:10]}")

    # Show how many distinct simplified names result.
    new_names = set(rename_map.values())
    n_collapses = sum(1 for old, new in rename_map.items() if old != new)
    print(f"[simplify] distinct simplified names: {len(new_names)}")
    print(f"[simplify] # categories renamed (old != new): {n_collapses}")

    # Re-merge: group original entries by simplified name.
    by_new: Dict[str, Dict] = {}
    for c in merged:
        new_name = rename_map[c["name"]]
        if new_name not in by_new:
            by_new[new_name] = {
                "name": new_name,
                "definition": c["definition"],
                "merged_from": [],
                "member_count": 0,
                "_source_broads": [],
            }
        # Concatenate
        by_new[new_name]["merged_from"].extend(c["merged_from"])
        by_new[new_name]["member_count"] += c["member_count"]
        by_new[new_name]["_source_broads"].append(c["name"])
    new_merged = sorted(by_new.values(), key=lambda c: -c["member_count"])

    simplified_path.write_text(json.dumps(new_merged, indent=2))
    rename_map_path.write_text(json.dumps(rename_map, indent=2))

    # Also update merge_mapping (original -> simplified) to chain through.
    new_mapping: Dict[str, str] = {}
    for original, broad in mapping.items():
        if broad == "__unmapped__":
            new_mapping[original] = "__unmapped__"
        else:
            new_mapping[original] = rename_map.get(broad, broad)
    mapping_path.write_text(json.dumps(new_mapping, indent=2))

    print(f"\n[simplify] wrote:")
    print(f"   {simplified_path}")
    print(f"   {rename_map_path}")
    print(f"   updated {mapping_path}")
    print()
    # Show collapses:
    collapse_groups = defaultdict(list)
    for old, new in rename_map.items():
        if old != new:
            collapse_groups[new].append(old)
    print(f"[simplify] {len(collapse_groups)} simplified names that absorbed other categories:")
    for new, olds in sorted(collapse_groups.items(), key=lambda kv: -len(kv[1])):
        print(f"   {new:32s}  <-  {', '.join(olds)}")


SHAPENET_REFERENCE = (
    "airplane, bag, basket, bathtub, bed, bench, bicycle, birdhouse, "
    "bookshelf, bottle, bowl, bus, cabinet, camera, can, cap, car, chair, "
    "clock, keyboard, dishwasher, display, earphone, faucet, file, guitar, "
    "helmet, jar, knife, lamp, laptop, loudspeaker, mailbox, microphone, "
    "microwave, monitor, motorcycle, mug, piano, pillow, pistol, pot, "
    "printer, remote, rifle, rocket, skateboard, sofa, stove, table, "
    "telephone, tower, train, vessel, washer"
)


SHAPENET_PROMPT = """\
You are renaming {N} 3D-object category labels to follow the ShapeNet
labeling convention. ShapeNet uses short, lowercase, single common-noun
labels for categories.

REFERENCE — these 55 ShapeNet category labels show the exact style:
  {shapenet_ref}

REQUIREMENTS for every output `new_name`:
- Lowercase only.
- Prefer ONE common noun: "car", "lamp", "rifle", "sofa", "chair",
  "monitor", "robot", "drone", "weapon", "creature", "character",
  "figurine", "sculpture", "boat", "vase", "container", "helmet".
- AVOID adjectives, even ones I previously kept: NO "passenger_car",
  "animatronic_character", "robot_character", "decorative_sphere",
  "cartoon_character". Use "car", "character", "robot", "sphere".
- AVOID modifier-noun compounds unless the compound IS the canonical name
  (e.g. ShapeNet has "remote" alone, not "remote_control"; uses "vessel"
  not "fishing_vessel"). Use "lamp" not "table_lamp"; "monitor" not
  "computer_monitor".
- Where the input clearly maps to one of the 55 ShapeNet labels above,
  USE THAT EXACT LABEL.
- For things NOT in ShapeNet (fantasy_creature, drone, robot, neon_sign,
  spacecraft, etc.), invent a ShapeNet-style label following the same
  convention: single short common noun.
- Examples:
    "passenger_car"          -> "car"
    "futuristic_firearm"     -> "rifle"  (or "pistol" if smaller; pick the better fit)
    "fantasy_weapon"         -> "weapon"
    "robot_character"        -> "robot"
    "cartoon_character"      -> "character"
    "animatronic_character"  -> "character"
    "character_figurine"     -> "figurine"
    "interior_scene"         -> "scene"
    "table_lamp"             -> "lamp"
    "computer_monitor"       -> "monitor"
    "kitchen_appliance"      -> "appliance"
    "control_console"        -> "console"
    "robotic_arm"            -> "robot"   (or "arm" — pick more general)
    "potion_bottle"          -> "bottle"
    "treasure_chest"         -> "chest"
    "spacecraft_model"       -> "spacecraft"
    "geologic_formation"     -> "rock"

Multiple input names will collapse to the same new_name — that's expected.

Input categories (one per line):
{listing}

Output JSON: {{"renames": [{{"old_name": ..., "new_name": ...}}, ...]}}
Every input name MUST appear exactly once.
"""


def shapenet_main(argv=None) -> None:
    """The broad names in ShapeNet style -> shapenet_taxonomy.json, shapenet_mapping.json."""
    parser = argparse.ArgumentParser(prog="python -m data_processing.tools.categories.taxonomy shapenet")
    parser.add_argument("--model", default="gpt-5")
    args = parser.parse_args(argv)

    merged_path = OUT_FULL_DIR / "merged_taxonomy.json"
    mapping_path = OUT_FULL_DIR / "merge_mapping.json"
    out_tax_path = OUT_FULL_DIR / "shapenet_taxonomy.json"
    out_map_path = OUT_FULL_DIR / "shapenet_mapping.json"

    merged = json.loads(merged_path.read_text())
    mapping = json.loads(mapping_path.read_text())

    print(f"[shapenet] {len(merged)} input categories")
    listing = "\n".join(f"  - {c['name']}" for c in merged)
    prompt = SHAPENET_PROMPT.format(N=len(merged), listing=listing,
                            shapenet_ref=SHAPENET_REFERENCE)

    vlm = VlmClient(client=make_openai_client(), model=args.model)
    temp = 1.0 if args.model.startswith("gpt-5") else 0.0
    print(f"[shapenet] calling {args.model} ...")
    result: RenameList = vlm.call_structured(
        prompt=prompt, image_paths=[], schema=RenameList, temperature=temp,
    )

    rename_map: Dict[str, str] = {r.old_name: r.new_name for r in result.renames}
    missing = sorted(set(c["name"] for c in merged) - set(rename_map))
    extra = sorted(set(rename_map) - set(c["name"] for c in merged))
    print(f"[shapenet] renamed: {len(rename_map)}; missing: {len(missing)}; extra: {len(extra)}")
    if missing:
        for n in missing:
            rename_map[n] = n   # default: keep
    new_names = sorted(set(rename_map.values()))
    print(f"[shapenet] distinct ShapeNet-style names: {len(new_names)}")

    # Re-merge: group input categories by simplified name.
    by_new: Dict[str, Dict] = {}
    for c in merged:
        new_name = rename_map[c["name"]]
        if new_name not in by_new:
            by_new[new_name] = {
                "name": new_name,
                "definition": c["definition"],
                "merged_from": [],
                "member_count": 0,
                "_source_broads": [],
            }
        by_new[new_name]["merged_from"].extend(c["merged_from"])
        by_new[new_name]["member_count"] += c["member_count"]
        by_new[new_name]["_source_broads"].append(c["name"])
    new_merged = sorted(by_new.values(), key=lambda c: -c["member_count"])

    out_tax_path.write_text(json.dumps(new_merged, indent=2))
    out_map_path.write_text(json.dumps(rename_map, indent=2))

    new_mapping: Dict[str, str] = {}
    for original, broad in mapping.items():
        if broad == "__unmapped__":
            new_mapping[original] = "__unmapped__"
        else:
            new_mapping[original] = rename_map.get(broad, broad)
    mapping_path.write_text(json.dumps(new_mapping, indent=2))

    print(f"\n[shapenet] wrote:")
    print(f"   {out_tax_path}")
    print(f"   {out_map_path}")
    print(f"   updated {mapping_path}")
    print()
    # Show top collapses:
    collapse_groups = defaultdict(list)
    for old, new in rename_map.items():
        if old != new:
            collapse_groups[new].append(old)
    print(f"[shapenet] {len(collapse_groups)} ShapeNet labels that absorbed multiple inputs:")
    for new, olds in sorted(collapse_groups.items(), key=lambda kv: -len(kv[1])):
        print(f"   {new:20s}  <-  {', '.join(olds)}")


def main(argv=None) -> None:
    run_subcommand({"merge": merge_main, "cleanup": cleanup_main, "simplify": simplify_main,
                    "shapenet": shapenet_main}, __doc__, argv,
                   prog="python -m data_processing.tools.categories.taxonomy")


if __name__ == "__main__":
    main()
