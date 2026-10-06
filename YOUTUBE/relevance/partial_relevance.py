"""Recover a separate annotation snapshot and run a training-only diagnostic.

Blank annotations stay unlabeled unless an explicit yes/no meaning is supplied
for a bounded, user-confirmed reviewed range. Blind labels never enter evaluation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

try:
    from .prepare_relevance_round import FIELDS, NEGATIVE_TYPES, file_hash, read_csv, write_csv, write_json
    from .evaluate_comment_relevance import evaluate
except ImportError:
    from prepare_relevance_round import FIELDS, NEGATIVE_TYPES, file_hash, read_csv, write_csv, write_json
    from evaluate_comment_relevance import evaluate

EDITABLE = FIELDS[-3:]


def index_rows(rows: list[dict]) -> dict:
    indexed = {r["comment_id"]: r for r in rows}
    if len(indexed) != len(rows) or "" in indexed:
        raise ValueError("Missing or duplicate comment ID")
    return indexed


def verify_source(originals: list[dict], rows: list[dict]) -> int:
    reference, supplied = index_rows(originals), index_rows(rows)
    if set(reference) != set(supplied):
        raise ValueError("Source rows were added or removed")
    newline_changes = 0
    for cid, row in supplied.items():
        for field in FIELDS[:-3]:
            before, after = reference[cid][field], row[field]
            if before == after:
                continue
            if field == "text" and before.replace("\r\n", "\n").replace("\r", "\n") == \
                    after.replace("\r\n", "\n").replace("\r", "\n"):
                newline_changes += 1
                continue
            raise ValueError(f"Substantive source edit: {cid}, {field}")
    return newline_changes


def snapshot(round_dir: Path, output: Path, reviewed_train: int = 0,
             reviewed_blind: int = 0, reviewed_blanks: str = "keep") -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if min(reviewed_train, reviewed_blind) < 0:
        raise ValueError("Reviewed spreadsheet row limits cannot be negative")
    if reviewed_blanks != "keep" and not (reviewed_train or reviewed_blind):
        raise ValueError("Bounded reviewed rows are required to fill blanks")
    manifest = json.loads((round_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    recovered, sources, changes, filled = {}, {}, {}, {}
    for split, through in (("train", reviewed_train), ("blind", reviewed_blind)):
        expected = manifest["sample_sha256"][f"{split}_sample.csv"]
        candidates = [round_dir / f"{split}_sample.csv", round_dir / f"{split}_labels.csv"]
        clean = next((p for p in candidates if file_hash(p) == expected), None)
        if clean is None:
            raise ValueError(f"No pristine {split} template matches the original manifest")
        originals = read_csv(clean)
        if through > len(originals) + 1:
            raise ValueError(f"{split}: reviewed row exceeds spreadsheet bounds")
        merged = {r["comment_id"]: dict(r) for r in originals}
        changes[split] = 0
        for path in candidates:
            rows = read_csv(path)
            changes[split] += verify_source(originals, rows)
            if through and [r["comment_id"] for r in rows] != [r["comment_id"] for r in originals]:
                raise ValueError("Reviewed row bounds require the original spreadsheet ordering")
            for row in rows:
                dest = merged[row["comment_id"]]
                for field in EDITABLE:
                    if row[field]:
                        if dest[field] and dest[field] != row[field]:
                            raise ValueError(f"Conflicting annotations: {row['comment_id']}, {field}")
                        dest[field] = row[field]
        filled[split] = 0
        for spreadsheet_row, original in enumerate(originals, 2):
            row = merged[original["comment_id"]]
            if row["vehicle_relevant"] not in {"", "yes", "no", "unclear"}:
                raise ValueError(f"Invalid relevance label: {row['comment_id']}")
            if reviewed_blanks != "keep" and spreadsheet_row <= through and not row["vehicle_relevant"]:
                row["vehicle_relevant"] = reviewed_blanks
                row["labeler_notes"] = (row["labeler_notes"] +
                                       f" | User confirmed blank={reviewed_blanks} in reviewed range").strip(" |")
                filled[split] += 1
        recovered[split] = [merged[r["comment_id"]] for r in originals]
        sources[split] = clean
    train, blind = recovered["train"], recovered["blind"]
    if {r["video_id"] for r in train} & {r["video_id"] for r in blind} or \
            {r["normalized_text_sha256"] for r in train} & {r["normalized_text_sha256"] for r in blind}:
        raise ValueError("Training/blind leakage")
    output.mkdir(parents=True)
    backup = output / "input_backups"
    backup.mkdir()
    inputs = [f"{split}_{kind}.csv" for split in ("train", "blind") for kind in ("sample", "labels")]
    inputs.append("second_reviewer_labels.csv")
    input_hashes = {}
    for name in inputs:
        shutil.copyfile(round_dir / name, backup / name)
        input_hashes[name] = file_hash(backup / name)
    for split in ("train", "blind"):
        shutil.copyfile(sources[split], output / f"{split}_sample.csv")
        write_csv(output / f"{split}_labels.csv", FIELDS, recovered[split])
    shutil.copyfile(round_dir / "second_reviewer_labels.csv", output / "second_reviewer_labels.csv")
    write_json(output / "sample_manifest.json", manifest)
    recovery = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_round": str(round_dir.resolve()), "input_backup_sha256": input_hashes,
        "reviewed_blanks": reviewed_blanks, "reviewed_train_through": reviewed_train,
        "reviewed_blind_through": reviewed_blind, "filled_reviewed_training_blanks": filled["train"],
        "source_line_endings_restored": changes,
        "training_labels": dict(Counter(r["vehicle_relevant"] or "unlabeled" for r in train)),
        "blind_completion": {"rows": len(blind), "unlabeled": sum(not r["vehicle_relevant"] for r in blind)},
        "note": "Original input files preserved; immutable source fields restored from checksum-verified templates. "
                "Blind labels retained only for later locked evaluation. Spreadsheet rows include the header. "
                "Invalid/missing negative subtypes remain unchanged for human review.",
    }
    write_json(output / "recovery_manifest.json", recovery)
    return recovery


def evaluate_partial(round_dir: Path, output: Path, folds: int = 5, seed: int = 42) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if folds < 2:
        raise ValueError("Need at least two CV folds")
    manifest = json.loads((round_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["sample_sha256"].items():
        if file_hash(round_dir / name) != checksum:
            raise ValueError(f"Frozen template changed: {name}")
    originals = read_csv(round_dir / "train_sample.csv")
    rows = read_csv(round_dir / "train_labels.csv")
    verify_source(originals, rows)
    canonical = index_rows(originals)
    # Only read the blank blind template for source identity isolation, never its annotations.
    blind_template = read_csv(round_dir / "blind_sample.csv")
    blind_videos = {r["video_id"] for r in blind_template}
    blind_hashes = {r["normalized_text_sha256"] for r in blind_template}
    if any(r["video_id"] in blind_videos or r["normalized_text_sha256"] in blind_hashes for r in rows):
        raise ValueError("Training overlaps blind source identity")
    if len({r["normalized_text_sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training text: resolve/group before CV")
    labels, pending, issues = [], [], []
    for number, supplied in enumerate(rows, 2):
        row = {**canonical[supplied["comment_id"]], **{k: supplied[k] for k in EDITABLE}, "csv_row": number}
        label = row["vehicle_relevant"]
        if label not in {"", "yes", "no", "unclear"}:
            raise ValueError(f"Invalid label at training CSV row {number}")
        if label == "no" and row["negative_subtype"] not in NEGATIVE_TYPES:
            issues.append({**row, "issue": "Review missing/unsupported negative subtype; binary no label preserved"})
        if label != "no" and row["negative_subtype"]:
            issues.append({**row, "issue": "Negative subtype on non-negative annotation"})
        if label in {"yes", "no"}:
            labels.append({**row, "comment_topic": "vehicle_specific" if label == "yes" else "not_vehicle_specific",
                           "vehicle_relevant": label == "yes"})
        else:
            pending.append(row)
    if len(labels) < 10 or len({r["vehicle_relevant"] for r in labels}) != 2:
        raise ValueError("Need at least ten explicit yes/no labels with both classes")
    metrics = evaluate(labels, folds, seed)
    metrics["note"] = (
        "Exploratory video-grouped CV on partially human-labeled, enriched training data. "
        "Blank/unclear rows excluded. Blind annotations are not read, trained on, scored, or used for tuning. "
        "The 0.5 cutoff is diagnostic only. No automatic rejection or production release is authorized."
    )
    metrics["training_label_counts"] = dict(Counter("yes" if r["vehicle_relevant"] else "no" for r in labels))
    metrics["training_videos"] = len({r["video_id"] for r in labels})
    metrics["pending_training_rows"] = len(pending)
    metrics["annotation_issues"] = len(issues)
    metrics["production_ready"] = False
    for model_metrics in (metrics["tfidf_char"], metrics["exact_prefilter"]):
        for value in model_metrics.values():
            tn, fn = value["true_negative"], value["false_negative"]
            value["rejection_precision"] = tn / (tn + fn) if tn + fn else None
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100_000, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    model.fit([r["text"] for r in labels], [int(r["vehicle_relevant"]) for r in labels])
    if pending:
        scores = model.predict_proba([r["text"] for r in pending])[:, 1]
        for row, score in zip(pending, scores):
            row["prob_vehicle_relevant"] = float(score)
            row["predicted_vehicle_relevant"] = "yes" if score >= .5 else "no"
            row["distance_from_half"] = abs(float(score) - .5)
        pending.sort(key=lambda r: (r["distance_from_half"], r["comment_id"]))
    prediction_rows = [{**r, "vehicle_relevant": "yes" if r["vehicle_relevant"] else "no",
                        "predicted_vehicle_relevant": "yes" if r["predicted_vehicle_relevant"] else "no"}
                       for r in labels]
    prediction_fields = (*FIELDS, "csv_row", "prob_vehicle_relevant", "predicted_vehicle_relevant", "cv_fold")
    errors = [r for r in prediction_rows if r["vehicle_relevant"] != r["predicted_vehicle_relevant"]]
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_labels_sha256": file_hash(round_dir / "train_labels.csv"),
        "script_sha256": file_hash(Path(__file__)), "source_snapshot": manifest["source"],
        "seed": seed, "folds": folds,
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("scikit-learn", "numpy", "pyarrow")},
        "prediction_role": "CV is out-of-fold; pending queue uses a model fit on all explicit training labels. "
                           "Scores are uncalibrated suggestions, not gold labels or production decisions.",
    }
    output.mkdir(parents=True)
    write_json(output / "relevance_cv_metrics.json", metrics)
    write_json(output / "run_manifest.json", provenance)
    write_csv(output / "relevance_cv_predictions.csv", prediction_fields, prediction_rows)
    write_csv(output / "cv_error_review.csv", (*prediction_fields, "review_notes"), errors)
    write_csv(output / "training_review_queue.csv", (*FIELDS, "csv_row", "prob_vehicle_relevant",
                                                    "predicted_vehicle_relevant", "distance_from_half"), pending)
    write_csv(output / "annotation_issues.csv", (*FIELDS, "csv_row", "issue"), issues)
    return metrics


def apply_reviews(round_dir: Path, review_dir: Path, output: Path) -> dict:
    """Apply only user changes; unchanged fields in review exports are stale copies."""
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    manifest = json.loads((round_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["sample_sha256"].items():
        if file_hash(round_dir / name) != checksum:
            raise ValueError(f"Frozen template changed: {name}")
    originals = read_csv(round_dir / "train_sample.csv")
    current = read_csv(round_dir / "train_labels.csv")
    verify_source(originals, current)
    canonical, baseline = index_rows(originals), index_rows(read_csv(review_dir / "relevance_cv_predictions.csv"))
    merged = {r["comment_id"]: {**canonical[r["comment_id"]], **{k: r[k] for k in EDITABLE}} for r in current}
    proposals, audit = {}, []
    for name in ("cv_error_review.csv", "annotation_issues.csv"):
        rows = read_csv(review_dir / name)
        index_rows(rows)
        for row in rows:
            cid = row["comment_id"]
            if cid not in baseline or cid not in merged:
                raise ValueError(f"Unknown reviewed comment ID: {cid}")
            verify_source([canonical[cid]], [row])
            updates = {key: row[key] for key in EDITABLE if row[key] != baseline[cid][key]}
            note = row.get("review_notes", "").strip().lower()
            if note in {"yes", "no", "unclear"}:
                if "vehicle_relevant" in updates and updates["vehicle_relevant"] != note:
                    raise ValueError(f"Conflicting relevance field/review note: {cid}")
                updates["vehicle_relevant"] = note
            for key, value in updates.items():
                pair = (cid, key)
                if pair in proposals and proposals[pair] != value:
                    raise ValueError(f"Conflicting review corrections: {cid}, {key}")
                if merged[cid][key] not in {baseline[cid][key], value}:
                    raise ValueError(f"Working annotation conflicts with review: {cid}, {key}")
                proposals[pair] = value
                if merged[cid][key] != value:
                    audit.append({"comment_id": cid, "field": key, "old": merged[cid][key],
                                  "new": value, "review_file": name})
                merged[cid][key] = value
            if note:
                merged[cid]["labeler_notes"] = (merged[cid]["labeler_notes"] +
                                                f" | User cv review_notes: {row['review_notes']}").strip(" |")
    for cid, row in merged.items():
        if row["vehicle_relevant"] not in {"", "yes", "no", "unclear"}:
            raise ValueError(f"Invalid reviewed relevance label: {cid}")
        if row["vehicle_relevant"] != "no" and row["negative_subtype"]:
            audit.append({"comment_id": cid, "field": "negative_subtype", "old": row["negative_subtype"],
                          "new": "", "review_file": "clear subtype after user relevance correction"})
            row["negative_subtype"] = ""
    output.mkdir(parents=True)
    backups = output / "input_backups"
    backups.mkdir()
    for name in ("train_sample.csv", "blind_sample.csv", "blind_labels.csv", "second_reviewer_labels.csv",
                 "sample_manifest.json"):
        shutil.copyfile(round_dir / name, output / name)
    shutil.copyfile(round_dir / "train_labels.csv", backups / "train_labels_before_review.csv")
    for name in ("cv_error_review.csv", "annotation_issues.csv", "relevance_cv_predictions.csv"):
        shutil.copyfile(review_dir / name, backups / name)
    write_csv(output / "train_labels.csv", FIELDS, [merged[r["comment_id"]] for r in originals])
    write_csv(output / "review_changes.csv", ("comment_id", "field", "old", "new", "review_file"), audit)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_round": str(round_dir.resolve()),
        "review_dir": str(review_dir.resolve()),
        "input_sha256": {p.name: file_hash(p) for p in backups.iterdir()},
        "changes_by_field": dict(Counter(r["field"] for r in audit)),
        "training_labels": dict(Counter(r["vehicle_relevant"] or "unlabeled" for r in merged.values())),
        "remaining_training_subtype_issues": sum(r["vehicle_relevant"] == "no" and
                                                  r["negative_subtype"] not in NEGATIVE_TYPES for r in merged.values()),
        "note": "Exact yes/no/unclear review notes interpreted as user relevance corrections; unchanged review "
                "export fields do not override newer corrections. Blind annotation file copied unchanged and "
                "not used for learning. Revised CV after error review is descriptive, not an independent test.",
    }
    write_json(output / "review_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    snap = subs.add_parser("snapshot", help="Recover a new working copy while preserving input files")
    snap.add_argument("--round-dir", type=Path, required=True)
    snap.add_argument("--output-dir", type=Path, required=True)
    snap.add_argument("--reviewed-train-through", type=int, default=0)
    snap.add_argument("--reviewed-blind-through", type=int, default=0)
    snap.add_argument("--reviewed-blanks", choices=("keep", "yes", "no"), default="keep")
    evaluation = subs.add_parser("evaluate", help="Evaluate explicit training labels and queue remaining training rows")
    evaluation.add_argument("--round-dir", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--folds", type=int, default=5)
    evaluation.add_argument("--seed", type=int, default=42)
    review = subs.add_parser("apply-reviews", help="Create a corrected snapshot from user-reviewed training CSVs")
    review.add_argument("--round-dir", type=Path, required=True)
    review.add_argument("--review-dir", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        result = snapshot(args.round_dir, args.output_dir, args.reviewed_train_through,
                          args.reviewed_blind_through, args.reviewed_blanks)
    elif args.command == "apply-reviews":
        result = apply_reviews(args.round_dir, args.review_dir, args.output_dir)
    else:
        result = evaluate_partial(args.round_dir, args.output_dir, args.folds, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
