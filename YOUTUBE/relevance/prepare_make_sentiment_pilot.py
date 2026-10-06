"""Prepare a make-level YouTube sentiment pilot without scoring comments.

Title matches provide candidate make context, not a video eligibility gate.
Comment-level labels determine relevance. Parquet and SQLite inputs are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "DATA" / "CAR_DATA_FINAL" / "Youtube_Comments.parquet"
DEFAULT_OUTPUT_ROOT = ROOT / "YOUTUBE" / "outputs"
COMMENT_COLUMNS = ("comment_id", "video_id", "video_title", "text", "published_at")
ASPECTS = (
    "reliability", "maintenance_cost", "price_value", "fuel_economy_range",
    "performance", "comfort", "safety", "technology", "design",
)
COVERAGE_FIELDS = (
    "video_id", "video_title", "playlist_id", "comment_count", "empty_comments",
    "title_versions", "matched_title_term", "candidate_Make",
)
SAMPLE_FIELDS = (
    "comment_id", "video_id", "video_title", "published_at", "text", "Make",
    "sample_stratum", "prefilter_reason", "comment_topic", "opinion_expressed",
    "referenced_make", *ASPECTS, "reliability_claim_basis", "evidence_quote",
    "labeler_notes",
)
EXACT_FILLER = {
    "first", "first comment", "great video", "nice video", "good video",
    "awesome video", "great review", "nice review", "good review",
    "love your videos", "love your channel", "thanks for the video",
    "thanks for the review", "subscribed", "new subscriber",
}


def _hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def prefilter_reason(text: str | None) -> str:
    """Only flag text that clearly contains no vehicle opinion.

    These flags are audit candidates, not automatic training labels. Short but
    meaningful remarks such as 'CVT sucks' intentionally pass this gate.
    """
    if text is None or not text.strip():
        return "empty"
    if re.fullmatch(r"\s*https?://\S+\s*", text, flags=re.IGNORECASE):
        return "url_only"
    if _normalized_words(text) in EXACT_FILLER:
        return "exact_creator_filler"
    return ""


def _sqlite_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def video_summary(source: Path) -> list[dict]:
    if source.suffix.lower() == ".db":
        with closing(_sqlite_connection(source)) as conn:
            rows = conn.execute("""
                SELECT video_id, MIN(video_title), MIN(playlist_id), COUNT(*),
                       SUM(text IS NULL OR TRIM(text) = ''),
                       COUNT(DISTINCT video_title)
                FROM youtube_comments_sentiment
                GROUP BY video_id
            """).fetchall()
        return [dict(zip(("video_id", "video_title", "playlist_id", "comment_count",
                          "empty_comments", "title_versions"), row)) for row in rows]
    if source.suffix.lower() != ".parquet":
        raise ValueError("--source must be a .parquet file or .db SQLite file")
    if not source.is_file():
        raise FileNotFoundError(source)
    import polars as pl

    frame = (
        pl.scan_parquet(source)
        .group_by("video_id")
        .agg(
            pl.col("video_title").min().alias("video_title"),
            pl.col("playlist_id").min().alias("playlist_id"),
            pl.len().alias("comment_count"),
            (pl.col("text").is_null() | (pl.col("text").str.strip_chars() == ""))
            .sum().alias("empty_comments"),
            pl.col("video_title").n_unique().alias("title_versions"),
        )
        .collect()
    )
    return frame.to_dicts()


def _title_term(title: str | None, terms: list[str]) -> str:
    words = f" {_normalized_words(title or '')} "
    for term in terms:
        if f" {_normalized_words(term)} " in words:
            return term
    return ""


def _write_csv_new(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(source: Path, make: str, aliases: list[str], output_dir: Path) -> dict:
    make = make.strip().upper()
    if not make:
        raise ValueError("--make cannot be empty")
    summaries = video_summary(source)
    candidates = []
    for row in summaries:
        term = _title_term(row["video_title"], [make, *aliases])
        if term:
            candidates.append({
                **row, "matched_title_term": term, "candidate_Make": make,
            })
    candidates.sort(key=lambda row: (-row["comment_count"], row["video_id"] or ""))
    coverage_path = output_dir / "video_make_coverage.csv"
    summary_path = output_dir / "make_coverage_summary.json"
    if coverage_path.exists() or summary_path.exists():
        raise FileExistsError(f"Pilot output exists in {output_dir}; choose a new --output-dir")
    summary = {
        "source": str(source.resolve()),
        "make": make,
        "title_terms": [make, *aliases],
        "source_videos": len(summaries),
        "source_comments": sum(row["comment_count"] for row in summaries),
        "title_candidate_videos": len(candidates),
        "title_candidate_comments": sum(row["comment_count"] for row in candidates),
        "title_candidate_empty_comments": sum(row["empty_comments"] for row in candidates),
        "note": "Title matches supply candidate make context; comment relevance is labeled separately.",
    }
    _write_csv_new(coverage_path, COVERAGE_FIELDS, candidates)
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def iter_comments(source: Path, video_ids: set[str]) -> Iterator[dict]:
    if source.suffix.lower() == ".db":
        with closing(_sqlite_connection(source)) as conn:
            ids = sorted(video_ids)
            for start in range(0, len(ids), 800):
                part = ids[start:start + 800]
                marks = ",".join("?" for _ in part)
                columns = ",".join(f'"{col}"' for col in COMMENT_COLUMNS)
                cursor = conn.execute(
                    f"SELECT {columns} FROM youtube_comments_sentiment "
                    f"WHERE video_id IN ({marks})", part,
                )
                for row in cursor:
                    yield dict(zip(COMMENT_COLUMNS, row))
        return
    if source.suffix.lower() != ".parquet":
        raise ValueError("--source must be a .parquet file or .db SQLite file")
    import polars as pl

    frame = (
        pl.scan_parquet(source)
        .select(COMMENT_COLUMNS)
        .filter(pl.col("video_id").is_in(sorted(video_ids)))
        .collect()
    )
    yield from frame.iter_rows(named=True)


def sample(
    source: Path, make: str, aliases: list[str], output_path: Path,
    max_comments: int, per_video: int, filler_audit: int, seed: int,
) -> dict:
    if min(max_comments, per_video) < 1 or filler_audit < 0:
        raise ValueError("Sample counts must be positive; --filler-audit may be zero")
    if output_path.exists():
        raise FileExistsError(output_path)
    candidate_videos = {
        row["video_id"] for row in video_summary(source)
        if _title_term(row["video_title"], [make, *aliases])
    }
    if not candidate_videos:
        raise ValueError(f"No video titles matched {make!r} or its aliases")
    by_video: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    filler_by_video: dict[str, tuple[str, dict]] = {}
    scanned = 0
    for row in iter_comments(source, candidate_videos):
        scanned += 1
        if not row["comment_id"] or not row["text"] or not row["text"].strip():
            continue
        video_id = row["video_id"]
        rank = _hash(row["comment_id"], seed)
        bucket = by_video[video_id]
        bucket.append((rank, row))
        bucket.sort(key=lambda item: item[0])
        del bucket[per_video:]
        if prefilter_reason(row["text"]):
            previous = filler_by_video.get(video_id)
            if previous is None or rank < previous[0]:
                filler_by_video[video_id] = (rank, row)

    ordered_videos = sorted(by_video, key=lambda video_id: _hash(video_id, seed))
    chosen: list[tuple[str, dict]] = []
    seen_ids = set()
    for position in range(per_video):
        for video_id in ordered_videos:
            if len(chosen) >= max_comments:
                break
            if len(by_video[video_id]) > position:
                row = by_video[video_id][position][1]
                if row["comment_id"] not in seen_ids:
                    chosen.append(("video_uniform", row))
                    seen_ids.add(row["comment_id"])
        if len(chosen) >= max_comments:
            break
    filler_candidates = sorted(
        filler_by_video.items(), key=lambda item: _hash(item[0], seed + 1)
    )
    added_filler = 0
    for _, (_, row) in filler_candidates:
        if added_filler >= filler_audit:
            break
        if row["comment_id"] not in seen_ids:
            chosen.append(("prefilter_audit", row))
            seen_ids.add(row["comment_id"])
            added_filler += 1
    output_rows = []
    for stratum, row in chosen:
        output_rows.append({
            **row, "Make": make.upper(), "sample_stratum": stratum,
            "prefilter_reason": prefilter_reason(row["text"]),
        })
    _write_csv_new(output_path, SAMPLE_FIELDS, output_rows)
    return {
        "candidate_videos": len(candidate_videos), "scanned_comments": scanned,
        "video_uniform_sample": sum(row["sample_stratum"] == "video_uniform" for row in output_rows),
        "prefilter_audit_sample": added_filler,
        "output": str(output_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="Export make title coverage")
    audit_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    audit_parser.add_argument("--make", required=True, help="Exact vPIC Make, such as TOYOTA")
    audit_parser.add_argument("--alias", action="append", default=[], help="Additional title spelling")
    audit_parser.add_argument("--output-dir", type=Path)
    sample_parser = subparsers.add_parser("sample", help="Sample comments across make title matches")
    sample_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    sample_parser.add_argument("--make", required=True)
    sample_parser.add_argument("--alias", action="append", default=[], help="Additional title spelling")
    sample_parser.add_argument("--output", type=Path)
    sample_parser.add_argument("--max-comments", type=int, default=200)
    sample_parser.add_argument("--per-video", type=int, default=2)
    sample_parser.add_argument("--filler-audit", type=int, default=40)
    sample_parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "audit":
        output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.make.upper()
        result = audit(args.source, args.make, args.alias, output_dir)
    else:
        output = args.output or DEFAULT_OUTPUT_ROOT / args.make.upper() / "comment_label_sample.csv"
        result = sample(
            args.source, args.make, args.alias, output,
            args.max_comments, args.per_video, args.filler_audit, args.seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
