"""Prepare/check human make, opinion, and aspect labels from confirmed positives.

This development sample does not read relevance blind annotations or infer
sentiment. Multi-make comments may add target_index=2 and corresponding aspects.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

try:
    from .compare_relevance_models import load_training
    from .prepare_make_sentiment_pilot import ROOT, _normalized_words
    from .prepare_relevance_round import SOURCE_FIELDS, digest, file_hash, read_csv, text_hash, write_csv, write_json
except ImportError:
    from compare_relevance_models import load_training
    from prepare_make_sentiment_pilot import ROOT, _normalized_words
    from prepare_relevance_round import SOURCE_FIELDS, digest, file_hash, read_csv, text_hash, write_csv, write_json

DEFAULT_ROUND = ROOT / "YOUTUBE/outputs/relevance_round_01_reviewed"
ASPECTS = ("reliability", "maintenance_cost", "price_value", "fuel_economy_range", "performance",
           "comfort", "safety", "technology", "design")
POLARITIES = {"positive", "negative", "neutral", "mixed"}
BASES = {"firsthand_owner", "secondhand", "speculation", "unclear"}
SAMPLE_FIELDS = (*SOURCE_FIELDS, "candidate_makes", "label_pool", "normalized_text_sha256")
TARGET_FIELDS = (*SAMPLE_FIELDS, "target_index", "target_resolution_status", "Make", "opinion_expressed", "labeler_notes")
ASPECT_FIELDS = (*SAMPLE_FIELDS, "target_index", "aspect", "aspect_present", "polarity", "evidence_quote",
                 "reliability_claim_basis", "labeler_notes")


def normalized_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def source_check(reference, row):
    for field in SAMPLE_FIELDS:
        a, b = reference[field], row.get(field, "")
        if field == "text":
            a, b = normalized_newlines(a), normalized_newlines(b)
        if a != b:
            raise ValueError(f"Changed frozen source/context: {reference['comment_id']}, {field}")


def prepare(directory, output, size=100, seed=44, toyota_sample=None, toyota_feedback=None):
    if output.exists():
        raise FileExistsError(f"Choose a new aspect-round output directory: {output}")
    if size < 1:
        raise ValueError("Positive sample size required")
    reviewed = load_training(directory)  # It never reads blind annotations.
    manifest = json.loads((directory / "sample_manifest.json").read_text(encoding="utf-8"))
    makes = manifest.get("vpic", {}).get("makes", [])
    if not makes:
        raise ValueError("The source sample manifest must contain canonical vPIC makes")
    blind = read_csv(directory / "blind_sample.csv")
    protected_videos = {r["video_id"] for r in blind}
    protected_ids = {r["comment_id"] for r in blind}
    protected_hashes = {text_hash(r["text"]) for r in blind}
    candidates = [{**{k: r[k] for k in SOURCE_FIELDS}, "candidate_makes": r["candidate_makes"],
                   "label_pool": "reviewed_round_01", "normalized_text_sha256": text_hash(r["text"])}
                  for r in reviewed if r["vehicle_relevant"] == "yes"]
    inputs = {"train_labels.csv": file_hash(directory / "train_labels.csv"),
              "sample_manifest.json": file_hash(directory / "sample_manifest.json"),
              "blind_sample.csv": file_hash(directory / "blind_sample.csv")}
    if (toyota_sample is None) != (toyota_feedback is None):
        raise ValueError("Supply both Toyota sample and confirmed feedback, or neither")
    if toyota_sample is not None:
        feedback_rows = read_csv(toyota_feedback)
        feedback = {r["comment_id"]: r["vehicle_relevant"] for r in feedback_rows}
        if len(feedback) != len(feedback_rows) or any(v not in {"yes", "no"} for v in feedback.values()):
            raise ValueError("Invalid/duplicate Toyota relevance feedback")
        terms = {m: _normalized_words(m) for m in makes}

        @lru_cache(maxsize=20_000)
        def make_context(title):
            words = " " + _normalized_words(title) + " "
            return " | ".join(m for m, t in terms.items() if " " + t + " " in words)

        for row in read_csv(toyota_sample):
            if feedback.get(row["comment_id"]) == "yes":
                candidates.append({**{k: row[k] for k in SOURCE_FIELDS},
                                   "candidate_makes": make_context(row["video_title"]),
                                   "label_pool": "toyota_confirmed", "normalized_text_sha256": text_hash(row["text"])})
        inputs.update(toyota_sample=file_hash(toyota_sample), toyota_feedback=file_hash(toyota_feedback))
    # Round-robin across tentative make contexts, keeping exact texts distinct.
    groups = {}
    for row in candidates:
        if row["comment_id"] in protected_ids or row["video_id"] in protected_videos or \
                row["normalized_text_sha256"] in protected_hashes:
            continue
        groups.setdefault(row["candidate_makes"] or "unmatched", []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: digest(f"{seed}|{r['comment_id']}"))
    selected, ids, hashes = [], set(), set()
    while len(selected) < size and any(groups.values()):
        for key in sorted(groups):
            if not groups[key] or len(selected) >= size:
                continue
            row = groups[key].pop()
            if row["comment_id"] in ids or row["normalized_text_sha256"] in hashes:
                continue
            ids.add(row["comment_id"]); hashes.add(row["normalized_text_sha256"])
            selected.append(row)
    if not selected:
        raise ValueError("No confirmed relevant comments available outside the blind sources")
    selected.sort(key=lambda r: digest(f"{seed}|display|{r['comment_id']}"))
    output.mkdir(parents=True)
    write_csv(output / "comment_sample.csv", SAMPLE_FIELDS, selected)
    write_csv(output / "canonical_makes.csv", ("Make",), [{"Make": m} for m in sorted(makes)])
    write_csv(output / "target_labels.csv", TARGET_FIELDS, [{**r, "target_index": "1"} for r in selected])
    write_csv(output / "aspect_labels.csv", ASPECT_FIELDS,
              [{**r, "target_index": "1", "aspect": aspect} for r in selected for aspect in ASPECTS])
    result = {"schema_version": "make_aspect_annotation_v1", "requested_comments": size,
              "sampled_comments": len(selected), "aspect_template_rows": len(selected) * len(ASPECTS),
              "purpose": "Development annotation only; no sentiment holdout is created or consumed.",
              "input_sha256": inputs, "source": manifest.get("source", {}), "seed": seed,
              "sample_sha256": {name: file_hash(output / name) for name in ("comment_sample.csv", "canonical_makes.csv")},
              "label_pool_counts": dict(Counter(r["label_pool"] for r in selected)),
              "candidate_make_counts": dict(Counter(r["candidate_makes"] or "unmatched" for r in selected)),
              "script_sha256": file_hash(Path(__file__)),
              "note": "Only human relevance judgments selected candidates. Title makes are context, not labels. "
                      "Blank target/aspect annotations mean unfinished. Reserve fresh videos before sentiment model development."}
    write_json(output / "annotation_manifest.json", result)
    return {k: result[k] for k in ("sampled_comments", "aspect_template_rows", "label_pool_counts", "purpose")}


def check_labels(directory, output):
    if output.exists():
        raise FileExistsError(f"Choose a new validation output directory: {output}")
    manifest = json.loads((directory / "annotation_manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["sample_sha256"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError(f"Frozen annotation reference changed: {name}")
    reference = {r["comment_id"]: r for r in read_csv(directory / "comment_sample.csv")}
    makes = {r["Make"] for r in read_csv(directory / "canonical_makes.csv")}
    targets, exported_targets, exported_aspects = {}, [], []
    seen_makes = set()
    for r in read_csv(directory / "target_labels.csv"):
        cid = r["comment_id"]
        if cid not in reference:
            raise ValueError(f"Unknown annotation comment: {cid}")
        source_check(reference[cid], r)
        index = r["target_index"]
        if not index.isdigit() or int(index) < 1 or str(int(index)) != index:
            raise ValueError("target_index must be a positive integer without leading zeros")
        key = (cid, index)
        if key in targets:
            raise ValueError(f"Duplicate target index: {key}")
        status, make, opinion = r["target_resolution_status"], r["Make"], r["opinion_expressed"]
        if status not in {"", "resolved", "ambiguous", "unresolved"} or opinion not in {"", "yes", "no", "unclear"}:
            raise ValueError(f"Invalid target status/opinion: {key}")
        if status == "resolved":
            if make not in makes:
                raise ValueError(f"Use an exact canonical Make: {key}, {make}")
            if (cid, make) in seen_makes:
                raise ValueError(f"Duplicate confirmed (comment_id, Make): {cid}, {make}")
            seen_makes.add((cid, make))
            if opinion:
                exported_targets.append({**reference[cid], "Make": make, "opinion_expressed": opinion})
        elif make or opinion:
            raise ValueError(f"Clear Make/opinion on pending or unresolved targets: {key}")
        targets[key] = r
    if {cid for cid, _ in targets} != set(reference):
        raise ValueError("Every frozen comment needs at least one target annotation row")
    aspect_rows = {}
    for r in read_csv(directory / "aspect_labels.csv"):
        key = (r["comment_id"], r["target_index"])
        if key not in targets:
            raise ValueError(f"Aspect row has no target: {key}")
        source_check(reference[key[0]], r)
        aspect = r["aspect"]
        if aspect not in ASPECTS or (*key, aspect) in aspect_rows:
            raise ValueError(f"Invalid/duplicate aspect row: {key}, {aspect}")
        present, polarity, quote, basis = (r[k] for k in ("aspect_present", "polarity", "evidence_quote", "reliability_claim_basis"))
        if present not in {"", "yes", "no", "unclear"}:
            raise ValueError(f"Invalid aspect_present: {key}, {aspect}")
        target = targets[key]
        if any((present, polarity, quote, basis)) and (target["target_resolution_status"] != "resolved" or
                                                     target["opinion_expressed"] != "yes"):
            raise ValueError(f"Aspect labels require a resolved make with opinion=yes: {key}")
        if present == "yes":
            if polarity not in POLARITIES or not quote or normalized_newlines(quote) not in normalized_newlines(reference[key[0]]["text"]):
                raise ValueError(f"Present aspect needs polarity and verbatim source evidence: {key}, {aspect}")
            if (aspect == "reliability" and basis not in BASES) or (aspect != "reliability" and basis):
                raise ValueError(f"Invalid reliability basis: {key}, {aspect}")
            exported_aspects.append({**reference[key[0]], "Make": target["Make"], "aspect": aspect,
                                     "polarity": polarity, "evidence_quote": quote, "reliability_claim_basis": basis})
        elif polarity or quote or basis:
            raise ValueError(f"Only present aspects carry polarity/evidence/basis: {key}, {aspect}")
        aspect_rows[(*key, aspect)] = r
    finished = set()
    for cid in reference:
        complete = True
        for key, target in targets.items():
            if key[0] != cid:
                continue
            status, opinion = target["target_resolution_status"], target["opinion_expressed"]
            if not status or (status == "resolved" and not opinion):
                complete = False
            if status == "resolved" and opinion == "yes":
                if any(not aspect_rows.get((*key, a), {}).get("aspect_present") for a in ASPECTS):
                    complete = False
        if complete:
            finished.add(cid)
    output.mkdir(parents=True)
    write_csv(output / "labeled_targets.csv", (*SAMPLE_FIELDS, "Make", "opinion_expressed"), exported_targets)
    write_csv(output / "labeled_aspects.csv", (*SAMPLE_FIELDS, "Make", "aspect", "polarity", "evidence_quote",
                                             "reliability_claim_basis"), exported_aspects)
    result = {"comments": len(reference), "annotation_complete_comments": len(finished),
              "unfinished_comments": len(reference) - len(finished), "resolved_targets_with_opinion_label": len(exported_targets),
              "opinion_yes_targets": sum(r["opinion_expressed"] == "yes" for r in exported_targets),
              "aspect_opinion_records": len(exported_aspects),
              "aspect_counts": dict(Counter(r["aspect"] for r in exported_aspects)),
              "polarity_counts": dict(Counter(r["polarity"] for r in exported_aspects)),
              "input_labels_sha256": {name: file_hash(directory / name) for name in ("target_labels.csv", "aspect_labels.csv")},
              "ready_for_sentiment_training": False,
              "note": "Valid partial human annotations exported. Fresh video holdout, independent review, and adequate per-aspect "
                      "support are still required before sentiment model development. Unfinished/absent aspects are not neutral."}
    write_json(output / "annotation_progress.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Create blank make/aspect templates from human-confirmed positives")
    prep.add_argument("--round-dir", type=Path, default=DEFAULT_ROUND)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--sample-size", type=int, default=100)
    prep.add_argument("--seed", type=int, default=44)
    prep.add_argument("--toyota-sample", type=Path)
    prep.add_argument("--toyota-feedback", type=Path)
    check = sub.add_parser("check-labels", help="Check partial human labels and export resolved long-form records")
    check.add_argument("--round-dir", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.round_dir, args.output_dir, args.sample_size, args.seed, args.toyota_sample, args.toyota_feedback)
    else:
        result = check_labels(args.round_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
