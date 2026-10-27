"""Prepare Phase 1 relevance labels; never train or score the source comments.

The blind CSV stays out of model selection. Inputs are read-only and each run
uses a new output directory. Run `prepare --help` or `check-labels --help`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq

try:
    from .prepare_make_sentiment_pilot import ROOT, prefilter_reason, _normalized_words
except ImportError:
    from prepare_make_sentiment_pilot import ROOT, prefilter_reason, _normalized_words

SOURCE_FIELDS = ("comment_id", "video_id", "video_title", "published_at", "text")
FIELDS = (*SOURCE_FIELDS, "candidate_makes", "split", "sample_stratum",
          "prefilter_reason", "normalized_text_sha256", "vehicle_relevant",
          "negative_subtype", "labeler_notes")
NEGATIVE_TYPES = {"creator_video", "spam", "generic_reaction", "other"}
HARD_CANDIDATE = re.compile(
    r"\b(video|review|reviewer|channel|subscribe|subscribed|presenter|camera|"
    r"editing|music|sponsor|giveaway|telegram|whatsapp)\b", re.I,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text_hash(text: str | None) -> str:
    # Normalized exact duplicates only; this does not detect semantic paraphrases.
    return digest(_normalized_words(text or ""))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields, rows) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def file_hash(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def iter_rows(path: Path, fields=SOURCE_FIELDS):
    source = pq.ParquetFile(path)
    missing = set(fields) - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing source fields: {sorted(missing)}")
    for batch in source.iter_batches(batch_size=25_000, columns=list(fields)):
        yield from batch.to_pylist()


@lru_cache(maxsize=20_000)
def date_value(value: str | None) -> str | None:
    if not value:
        return None
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def offer(heap: list, limit: int, row: dict, seed: int) -> None:
    rank = int(digest(f"{seed}|{row['comment_id']}"), 16)
    entry = (-rank, row["comment_id"], row)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)


def ordered(heap: list) -> list[dict]:
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def prepare(source: Path, vpic: Path, existing: Path, predictions: Path,
            output: Path, seed: int = 42, blind_n: int = 300,
            train_n: int = 400, reserve_fraction: float = 0.2) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new --output-dir; {output} already exists")
    if min(blind_n, train_n) < 1 or not 0 < reserve_fraction < 1:
        raise ValueError("Positive sample counts and 0 < --reserve-fraction < 1 required")
    pq.ParquetFile(source)
    source_stat = (source.stat().st_size, source.stat().st_mtime_ns)
    makes = sorted({str(row["Make"]).strip() for row in iter_rows(vpic, ("Make",))
                    if row["Make"] and str(row["Make"]).strip()})
    terms = {make: _normalized_words(make) for make in makes}

    @lru_cache(maxsize=20_000)
    def candidates(title: str | None) -> str:
        words = f" {_normalized_words(title or '')} "
        return " | ".join(make for make, term in terms.items() if f" {term} " in words)

    old = read_csv(existing)
    prior_videos = {row["video_id"] for row in old}
    prior_hashes = {text_hash(row["text"]) for row in old}
    prior_ids = {row["comment_id"] for row in old}
    errors = [row for row in read_csv(predictions)
              if row["vehicle_relevant"] != row["predicted_vehicle_relevant"]]

    def reserved(video_id: str) -> bool:
        return (video_id not in prior_videos and
                int(digest(f"{seed}|video|{video_id}")[:16], 16) / 2**64 < reserve_fraction)

    blind_heap = []
    seen_ids, hashes = set(), set()
    videos, dates, lengths, flags = Counter(), Counter(), Counter(), Counter()
    title_coverage = Counter()
    total = duplicate_ids = duplicate_text = empty = invalid_dates = 0
    blind_population = 0
    min_date = max_date = None
    for row in iter_rows(source):
        total += 1
        cid, vid, text = row["comment_id"], row["video_id"], row["text"]
        if not cid or not vid:
            raise ValueError(f"Missing comment/video ID at source row {total}")
        if cid in seen_ids:
            duplicate_ids += 1
            continue
        seen_ids.add(cid)
        videos[vid] += 1
        normalized_hash = text_hash(text)
        duplicate_text += normalized_hash in hashes
        hashes.add(normalized_hash)
        empty += not bool(text and text.strip())
        length = len(text or "")
        lengths["0" if length == 0 else "1-39" if length < 40 else
                "40-199" if length < 200 else "200-999" if length < 1000 else "1000+"] += 1
        reason = prefilter_reason(text)
        flags[reason or "none"] += 1
        date = date_value(row["published_at"])
        if date:
            dates[date[:7]] += 1
            min_date = min(min_date, date) if min_date else date
            max_date = max(max_date, date) if max_date else date
        else:
            invalid_dates += 1
        make_context = candidates(row["video_title"])
        title_coverage[make_context or "unmatched"] += 1
        if reserved(vid) and cid not in prior_ids and normalized_hash not in prior_hashes:
            blind_population += 1
            offer(blind_heap, blind_n, {
                **row, "candidate_makes": make_context, "split": "blind",
                "sample_stratum": "comment_uniform_reserved_videos",
                "prefilter_reason": reason, "normalized_text_sha256": normalized_hash,
            }, seed)
        if total % 250_000 == 0:
            print(f"Audit: {total:,} rows", flush=True)
    if duplicate_ids:
        raise ValueError(f"Source has {duplicate_ids} repeated comment IDs; resolve before sampling")
    blind = ordered(blind_heap)
    if len(blind) != blind_n:
        raise ValueError(f"Only {len(blind)} blind candidates; reduce --blind-size or reserve more videos")
    blind_hashes = {row["normalized_text_sha256"] for row in blind}
    del seen_ids, hashes
    pools = {name: [] for name in ("creator_spam_candidate", "short_nonfiller", "general_nonfiller")}
    for row in iter_rows(source):
        if reserved(row["video_id"]) or row["comment_id"] in prior_ids:
            continue
        text = row["text"] or ""
        norm = text_hash(text)
        if prefilter_reason(text) or norm in prior_hashes or norm in blind_hashes:
            continue
        stratum = ("creator_spam_candidate" if HARD_CANDIDATE.search(text) else
                   "short_nonfiller" if len(text.strip()) < 40 else "general_nonfiller")
        offer(pools[stratum], train_n * 10, {
            **row, "candidate_makes": candidates(row["video_title"]),
            "split": "train", "sample_stratum": stratum, "prefilter_reason": "",
            "normalized_text_sha256": norm,
        }, seed + 1)
    train, train_hashes, video_counts = [], set(), Counter()
    def take(rows, target):
        for row in rows:
            norm, vid = row["normalized_text_sha256"], row["video_id"]
            if len(train) >= target:
                break
            if norm in train_hashes or video_counts[vid] >= 4:
                continue
            train.append(row)
            train_hashes.add(norm)
            video_counts[vid] += 1
    for name, fraction in (("creator_spam_candidate", .30), ("short_nonfiller", .20),
                           ("general_nonfiller", .50)):
        take(ordered(pools[name]), min(train_n, len(train) + round(train_n * fraction)))
    take(ordered([entry for pool in pools.values() for entry in pool]), train_n)
    if len(train) != train_n:
        raise ValueError(f"Only {len(train)} training candidates; reduce --train-size or use another seed")
    for rows in (train, blind):
        rows.sort(key=lambda row: digest(f"{seed}|display|{row['comment_id']}"))
    second = [row for rows in (train, blind) for row in
              sorted(rows, key=lambda row: digest(f"{seed}|second|{row['comment_id']}"))
              [:max(1, round(len(rows) * .15))]]
    versions = {}
    import importlib.metadata
    for package in ("pyarrow", "polars", "scikit-learn"):
        versions[package] = importlib.metadata.version(package)
    if source_stat != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError("Source changed during preparation; use a stable Parquet snapshot")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "label_schema_version": "relevance_v2",
        "seed": seed, "reserve_fraction": reserve_fraction,
        "source": {"path": str(source.resolve()), "sha256": file_hash(source),
                   "bytes": source.stat().st_size, "rows": total},
        "vpic": {"path": str(vpic.resolve()), "sha256": file_hash(vpic), "makes": makes},
        "prior_labels_sha256": file_hash(existing), "prior_predictions_sha256": file_hash(predictions),
        "dependencies": versions, "blind_rows": len(blind), "training_rows": len(train),
        "preparation_script_sha256": file_hash(Path(__file__)),
        "prefilter_script_sha256": file_hash(ROOT / "YOUTUBE/relevance/prepare_make_sentiment_pilot.py"),
        "reserved_video_count": sum(reserved(vid) for vid in videos),
        "blind_eligible_population": blind_population,
        "training_strata": dict(Counter(row["sample_stratum"] for row in train)),
        "note": "Blind rows are comment-uniform within randomly reserved videos, excluding prior-label videos/text. "
                "Prevalence describes that eligible population, not an unbiased census estimate. Training is enriched; "
                "candidate strata are not labels. Only normalized exact duplicate leakage is blocked; "
                "audit paraphrases before modeling. No automated language identification is performed.",
    }
    audit = {"rows": total, "videos": len(videos), "empty_text": empty,
             "duplicate_comment_id_rows": duplicate_ids, "normalized_duplicate_text_rows": duplicate_text,
             "invalid_or_missing_publication_dates": invalid_dates,
             "publication_date_min": min_date, "publication_date_max": max_date,
             "publication_month_counts": dict(sorted(dates.items())), "length_counts": dict(lengths),
             "prefilter_counts": dict(flags), "title_candidate_make_counts": dict(title_coverage),
             "top_20_videos": videos.most_common(20),
             "language_audit": "Not inferred. Inspect non-English examples and label unclear with notes when needed."}
    output.mkdir(parents=True)
    for name, rows in (("train", train), ("blind", blind)):
        write_csv(output / f"{name}_sample.csv", FIELDS, rows)
        write_csv(output / f"{name}_labels.csv", FIELDS, rows)
    write_csv(output / "second_reviewer_labels.csv", FIELDS, second)
    write_csv(output / "toyota_error_review.csv", (*errors[0].keys(), "review_notes") if errors else
              ("comment_id", "review_notes"), errors)
    write_json(output / "source_audit.json", audit)
    manifest["sample_sha256"] = {name: file_hash(output / name) for name in
                                  ("train_sample.csv", "blind_sample.csv")}
    write_json(output / "sample_manifest.json", manifest)
    print(f"Created {len(train)} training and {len(blind)} blind rows in {output}", flush=True)
    return manifest


def check_labels(directory: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((directory / "sample_manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["sample_sha256"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError(f"Frozen sample changed: {name}")
    split_rows = {}
    for split in ("train", "blind"):
        original = {row["comment_id"]: row for row in read_csv(directory / f"{split}_sample.csv")}
        rows = read_csv(directory / f"{split}_labels.csv")
        if len(rows) != len(original) or {row["comment_id"] for row in rows} != set(original):
            raise ValueError(f"{split}: rows removed, added, or duplicated")
        for row in rows:
            for field in FIELDS[:-3]:
                if row[field] != original[row["comment_id"]][field]:
                    raise ValueError(f"{split}: changed source/sampling field {field} for {row['comment_id']}")
            if row["vehicle_relevant"] not in {"", "yes", "no", "unclear"}:
                raise ValueError(f"Invalid relevance label for {row['comment_id']}")
            if row["negative_subtype"] not in NEGATIVE_TYPES | {""}:
                raise ValueError(f"Invalid negative subtype for {row['comment_id']}")
            if row["vehicle_relevant"] == "no" and not row["negative_subtype"]:
                raise ValueError(f"Add negative_subtype for {row['comment_id']}")
            if row["vehicle_relevant"] != "no" and row["negative_subtype"]:
                raise ValueError(f"Negative subtype on a non-negative label for {row['comment_id']}")
        split_rows[split] = rows
    train, blind = split_rows["train"], split_rows["blind"]
    if {r["video_id"] for r in train} & {r["video_id"] for r in blind} or \
       {r["normalized_text_sha256"] for r in train} & {r["normalized_text_sha256"] for r in blind}:
        raise ValueError("Training/blind split leakage")
    counts = Counter(r["vehicle_relevant"] or "unlabeled" for r in train)
    hard_negatives = sum(r["vehicle_relevant"] == "no" for r in train)
    second = read_csv(directory / "second_reviewer_labels.csv")
    train_by_id = {r["comment_id"]: r for r in train}
    originals = {r["comment_id"]: r for r in train + blind}
    seen_second = set()
    agreed = compared = 0
    for row in second:
        if row["comment_id"] not in originals or row["comment_id"] in seen_second:
            raise ValueError("Unknown or duplicate second-reviewer comment ID")
        seen_second.add(row["comment_id"])
        for field in FIELDS[:-3]:
            if row[field] != originals[row["comment_id"]][field]:
                raise ValueError(f"Changed second-reviewer field {field}")
        if row["vehicle_relevant"] not in {"", "yes", "no", "unclear"}:
            raise ValueError(f"Invalid second-reviewer label for {row['comment_id']}")
        original = train_by_id.get(row["comment_id"])
        if original and original["vehicle_relevant"] and row["vehicle_relevant"]:
            compared += 1
            agreed += row["vehicle_relevant"] == original["vehicle_relevant"]
    result = {"training_label_counts": dict(counts), "additional_nonfiller_negatives": hard_negatives,
              "training_negative_subtypes": dict(Counter(r["negative_subtype"] for r in train
                                                        if r["vehicle_relevant"] == "no")),
              "training_candidate_make_counts": dict(Counter(r["candidate_makes"] or "unmatched" for r in train)),
              "training_double_labeled": compared,
              "training_disagreement_rate": 1 - agreed / compared if compared else None,
              "blind_completion": {"rows": len(blind),
                                   "unlabeled": sum(not r["vehicle_relevant"] for r in blind)},
              "ready_for_model_comparison": counts["unlabeled"] == 0 and counts["yes"] > 0 and
                                           hard_negatives >= 100 and compared >= round(len(train) * .1) and
                                           agreed == compared and all(r["vehicle_relevant"] for r in blind),
              "note": "Blind label distribution is intentionally hidden until models/thresholds are locked. "
                      "Unclear labels remain for review and are excluded from binary training. "
                      "Readiness checks label coverage only, not production model performance. "
                      "Reconcile disagreements while preserving independent annotations."}
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Audit Parquet and create protected labeling samples")
    prep.add_argument("--source", type=Path, default=ROOT / "DATA/CAR_DATA_FINAL/Youtube_Comments.parquet")
    prep.add_argument("--vpic", type=Path, default=ROOT / "DATA/CAR_DATA_FINAL/NHTSA_vPIC.parquet")
    prep.add_argument("--existing-labels", type=Path, default=ROOT / "YOUTUBE/outputs/TOYOTA/comment_label_sample.csv")
    prep.add_argument("--predictions", type=Path, default=ROOT / "YOUTUBE/outputs/TOYOTA/relevance_baseline_v1/relevance_cv_predictions.csv")
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--blind-size", type=int, default=300)
    prep.add_argument("--train-size", type=int, default=400)
    prep.add_argument("--reserve-fraction", type=float, default=.2)
    check = sub.add_parser("check-labels", help="Validate labels without revealing blind-test prevalence")
    check.add_argument("--round-dir", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.vpic, args.existing_labels, args.predictions,
                         args.output_dir, args.seed, args.blind_size, args.train_size, args.reserve_fraction)
    else:
        result = check_labels(args.round_dir, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
