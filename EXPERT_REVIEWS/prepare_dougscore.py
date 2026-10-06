"""Import the audited expert snapshot and prepare unapproved linkage evidence.

Never assigns comment sentiment, approves vehicle/video matches, or changes inputs.
Outputs require fresh directories to preserve annotations and earlier snapshots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "5b93a406ad014c7b473b3dcef69360d69c93249bf89815416899b373030f1600"
WEEKEND = ("styling", "acceleration", "handling", "fun_factor", "cool_factor")
DAILY = ("features", "comfort", "quality", "practicality", "value")
TOTALS = ("weekend_total", "daily_total", "dougscore")
COLUMNS = ("model_year_raw", "make_raw", "model_raw", *WEEKEND, "weekend_total",
           *DAILY, "daily_total", "dougscore", "video_link_raw", "filming_city_raw",
           "filming_region_raw", "vehicle_country_raw")
MAPPING_VERSION = "dougscore-community-v1-proposed"
MAPPING = {"design": ("styling",), "performance": ("acceleration", "handling", "fun_factor"),
           "comfort": ("comfort",), "technology": ("features",), "price_value": ("value",)}
CROSSWALK_FIELDS = ("review_record_id", "source_snapshot_sha256", "source_row_number",
                    "model_year_raw", "make_raw", "model_raw", "match_candidate_index",
                    "ModelYear", "Make", "Model", "Trim", "match_level", "match_status",
                    "match_evidence", "reviewed_by", "reviewed_at", "video_id", "review_url",
                    "review_published_at", "video_match_status", "video_match_evidence",
                    "score_available_at", "score_version_evidence")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(value):
    return " ".join(str(value or "").split()).casefold()


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parse_snapshot(source, expected_sha256):
    data = source.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != expected_sha256:
        raise ValueError("Snapshot hash changed; audit the new source and specify its expected SHA-256.")
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    if (len(rows) < 4 or any(len(r) != 20 for r in rows)
            or rows[0][0] != "Welcome to the DougScore!"
            or rows[1][3:8] != ["Styling", "Acceleration", "Handling", "Fun Factor", "Cool Factor"]
            or rows[1][9:14] != ["Features", "Comfort", "Quality", "Practicality", "Value"]
            or any(v.strip() for v in rows[2])):
        raise ValueError("Unexpected source layout.")
    records, keys = [], set()
    ingested = datetime.now(timezone.utc).isoformat()
    for row_number, row in enumerate(rows[3:], 4):
        raw = dict(zip(COLUMNS, row))
        if not re.fullmatch(r"\d{4}", raw["model_year_raw"].strip()) or not all(
                raw[k].strip() for k in ("make_raw", "model_raw")):
            raise ValueError(f"Missing/invalid identity at logical row {row_number}.")
        key = tuple(norm(raw[k]) for k in COLUMNS[:3])
        if key in keys:
            raise ValueError(f"Duplicate normalized identity at logical row {row_number}.")
        keys.add(key)
        scores = {k: int(raw[k]) for k in (*WEEKEND, *DAILY, *TOTALS)}
        if (not all(1 <= scores[k] <= 10 for k in (*WEEKEND, *DAILY))
                or sum(scores[k] for k in WEEKEND) != scores["weekend_total"]
                or sum(scores[k] for k in DAILY) != scores["daily_total"]
                or scores["weekend_total"] + scores["daily_total"] != scores["dougscore"]):
            raise ValueError(f"Invalid score or total at logical row {row_number}.")
        records.append({**raw, **{f"{k}_raw": raw[k] for k in scores}, **scores,
                        "review_record_id": f"{sha}:{row_number}", "reviewer": "Doug DeMuro",
                        "source_snapshot_sha256": sha, "source_row_number": row_number,
                        "ModelYear": int(raw["model_year_raw"]), "make_normalized": norm(raw["make_raw"]),
                        "model_normalized": norm(raw["model_raw"]), "ingested_at": ingested,
                        "video_id": None, "review_url": None, "review_published_at": None,
                        "score_available_at": None, "score_version_evidence": None})
    return records


def import_snapshot(source, output, expected_sha256=EXPECTED_SHA256):
    records = parse_snapshot(source, expected_sha256)
    aspects = []
    for r in records:
        for aspect, components in MAPPING.items():
            aspects.append({"review_record_id": r["review_record_id"], "aspect": aspect,
                            "aspect_mapping_version": MAPPING_VERSION, "mapping_status": "proposed",
                            "expert_categories": list(components),
                            "expert_component_scores": [r[k] for k in components],
                            "expert_index_name": "equal_weight_performance_mean" if len(components) > 1 else components[0],
                            "expert_score": sum(r[k] for k in components) / len(components),
                            "partial_proxy": aspect == "technology",
                            "source_snapshot_sha256": r["source_snapshot_sha256"]})
    output.mkdir(parents=True, exist_ok=False)
    # Explicit null strings remain typed strings when metadata is unavailable.
    table = pa.Table.from_pylist(records)
    for name in ("video_id", "review_url", "review_published_at", "score_available_at", "score_version_evidence"):
        table = table.set_column(table.schema.get_field_index(name), name, pa.array([None] * len(records), type=pa.string()))
    pq.write_table(table, output / "expert_review_scores.parquet")
    pq.write_table(pa.Table.from_pylist(aspects), output / "expert_aspect_scores.parquet")
    crosswalk = [{**{k: r[k] for k in CROSSWALK_FIELDS[:6]}, "match_candidate_index": 1,
                  "match_status": "pending", "video_match_status": "pending"} for r in records]
    write_csv(output / "vehicle_video_crosswalk.csv", CROSSWALK_FIELDS, crosswalk)
    manifest = {"state": "imported_linkage_pending", "source": str(source.resolve()),
                "source_snapshot_sha256": expected_sha256, "review_rows": len(records), "aspect_rows": len(aspects),
                "ingested_at": records[0]["ingested_at"], "script_sha256": digest(Path(__file__)),
                "mapping_version": MAPPING_VERSION, "approved_vehicle_matches": 0, "verified_videos": 0,
                "community_comparison_generated": False, "price_model_consumes_expert_scores": False,
                "files_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "import_manifest.json", manifest)
    return manifest


def coverage(snapshot, vpic, output):
    manifest = json.loads((snapshot / "import_manifest.json").read_text(encoding="utf-8"))
    for name in ("expert_review_scores.parquet", "expert_aspect_scores.parquet"):
        if digest(snapshot / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Changed imported artifact: {name}")
    records = pq.read_table(snapshot / "expert_review_scores.parquet").to_pylist()
    cohorts = set()
    # Project only identity fields; do not load listing rows or comment text.
    source_hashes = {"vpic": digest(vpic)}
    for batch in pq.ParquetFile(vpic).iter_batches(columns=["ModelYear", "Make", "Model", "Trim"]):
        for r in batch.to_pylist():
            cohorts.add(tuple(r[k] for k in ("ModelYear", "Make", "Model", "Trim")))
    index = defaultdict(list)
    for c in cohorts:
        index[(norm(c[0]), norm(c[1]))].append(c)
    candidates, counts = [], Counter()
    for r in records:
        matches = sorted([c for c in index[(norm(r["ModelYear"]), r["make_normalized"])]
                          if norm(c[2]) and (r["model_normalized"] == norm(c[2])
                          or r["model_normalized"].startswith(norm(c[2]) + " "))], key=str)
        counts["reviews_with_text_candidates" if matches else "reviews_without_text_candidates"] += 1
        for i, c in enumerate(matches, 1):
            candidates.append({"review_record_id": r["review_record_id"], "match_candidate_index": i,
                               **dict(zip(("ModelYear", "Make", "Model", "Trim"), c)),
                               "match_status": "pending", "candidate_method": "year_make_exact_model_prefix",
                               "model_raw": r["model_raw"], "match_evidence": "Text candidate only; verify reviewed variant and trim."})
    if source_hashes != {"vpic": digest(vpic)}:
        raise ValueError("Coverage source changed during scan.")
    output.mkdir(parents=True, exist_ok=False)
    fields = ("review_record_id", "match_candidate_index", "ModelYear", "Make", "Model", "Trim",
              "match_status", "candidate_method", "model_raw", "match_evidence")
    write_csv(output / "vehicle_candidate_queue.csv", fields, candidates)
    report = {"state": "candidate_coverage_only", "review_rows": len(records), **dict(counts),
              "candidate_rows": len(candidates), "distinct_vpic_cohorts": len(cohorts),
              "approved_vehicle_matches": 0, "video_linkage": "deferred_by_user",
              "source_paths": {"vpic": str(vpic.resolve())},
              "source_sha256": source_hashes, "import_manifest_sha256": digest(snapshot / "import_manifest.json"),
              "script_sha256": digest(Path(__file__)), "executed_at": datetime.now(timezone.utc).isoformat(),
              "candidate_method": "Exact whitespace/case-normalized year and make, raw variant equals or starts with canonical model plus a space; no aliases or trim inference.",
              "limitations": "Candidates are not verified overlap. No video is linked from title or time-formatted CSV text. No comments attributed or scored.",
              "files_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "coverage_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-snapshot")
    imp.add_argument("--source", type=Path, default=Path("C:/Users/danie/Downloads/DougScore - DougScore.csv"))
    imp.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    imp.add_argument("--output-dir", type=Path, required=True)
    cov = sub.add_parser("coverage")
    cov.add_argument("--snapshot-dir", type=Path, required=True)
    cov.add_argument("--vpic", type=Path, default=ROOT / "DATA/CAR_DATA_FINAL/NHTSA_vPIC.parquet")
    cov.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = (import_snapshot(args.source, args.output_dir, args.expected_sha256) if args.command == "import-snapshot"
              else coverage(args.snapshot_dir, args.vpic, args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
