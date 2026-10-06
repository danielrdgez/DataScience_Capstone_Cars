"""Evaluate a local vehicle-relevance baseline on human-labeled comments.

This is a diagnostic, not a production filter. Splits are grouped by video so
comments from one review cannot appear in both training and validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline


POSITIVE_TOPICS = {"vehicle_specific", "target_make", "other_vehicle"}
NEGATIVE_TOPICS = {"not_vehicle_specific", "video_creator", "off_topic"}
PREDICTION_FIELDS = (
    "csv_row", "comment_id", "video_id", "sample_stratum", "prefilter_reason",
    "vehicle_relevant", "predicted_vehicle_relevant", "prob_vehicle_relevant",
    "cv_fold", "text",
)


def read_labels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen_ids: set[str] = set()
    labeled = []
    for number, row in enumerate(rows, start=2):
        topic = row["comment_topic"].strip()
        if topic not in POSITIVE_TOPICS | NEGATIVE_TOPICS:
            continue
        comment_id = row["comment_id"]
        if not comment_id or comment_id in seen_ids or not row["video_id"]:
            raise ValueError(f"Missing or repeated comment/video ID at CSV row {number}")
        seen_ids.add(comment_id)
        labeled.append({**row, "csv_row": number, "vehicle_relevant": topic in POSITIVE_TOPICS})
    if len(labeled) < 10 or len({row["vehicle_relevant"] for row in labeled}) != 2:
        raise ValueError("Need at least ten labeled comments from both relevance classes")
    return labeled


def confusion(rows: list[dict], key: str) -> dict[str, int | float | None]:
    tp = int(sum(row["vehicle_relevant"] and row[key] for row in rows))
    fn = int(sum(row["vehicle_relevant"] and not row[key] for row in rows))
    tn = int(sum(not row["vehicle_relevant"] and not row[key] for row in rows))
    fp = int(sum(not row["vehicle_relevant"] and row[key] for row in rows))
    return {
        "n": len(rows), "true_positive": tp, "false_negative": fn,
        "true_negative": tn, "false_positive": fp,
        "relevant_recall": tp / (tp + fn) if tp + fn else None,
        "irrelevant_recall": tn / (tn + fp) if tn + fp else None,
    }


def evaluate(rows: list[dict], folds: int, seed: int) -> dict:
    texts = [row["text"] or "" for row in rows]
    labels = [int(row["vehicle_relevant"]) for row in rows]
    video_ids = [row["video_id"] for row in rows]
    if len(set(video_ids)) < folds:
        raise ValueError("Need at least as many distinct videos as CV folds")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(texts, labels, video_ids), start=1
    ):
        train_videos = {video_ids[i] for i in train_indices}
        test_videos = {video_ids[i] for i in test_indices}
        if train_videos & test_videos:
            raise AssertionError("A video appears in both training and validation")
        if len({labels[i] for i in train_indices}) != 2:
            raise ValueError(f"Fold {fold} training set lacks one relevance class")
        model = make_pipeline(
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                max_features=100_000, sublinear_tf=True,
            ),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        model.fit([texts[i] for i in train_indices], [labels[i] for i in train_indices])
        probabilities = model.predict_proba([texts[i] for i in test_indices])[:, 1]
        for index, probability in zip(test_indices, probabilities):
            rows[index]["prob_vehicle_relevant"] = float(probability)
            rows[index]["predicted_vehicle_relevant"] = probability >= 0.5
            rows[index]["cv_fold"] = fold
    if any("cv_fold" not in row for row in rows):
        raise AssertionError("A labeled row did not receive an out-of-fold prediction")
    for row in rows:
        row["rule_predicted_vehicle_relevant"] = not bool(row["prefilter_reason"])
    strata = sorted({row["sample_stratum"] for row in rows})
    return {
        "source_labels": dict(Counter(row["comment_topic"] for row in rows)),
        "folds": folds, "seed": seed, "threshold": 0.5,
        "note": (
            "Exploratory grouped cross-validation on a deliberately sampled label set. "
            "The 40 filler-audit negatives make this sample easier than the full comment pool. "
            "Do not use these scores as production performance estimates."
        ),
        "tfidf_char": {
            "all": confusion(rows, "predicted_vehicle_relevant"),
            **{stratum: confusion(
                [row for row in rows if row["sample_stratum"] == stratum],
                "predicted_vehicle_relevant",
            ) for stratum in strata},
        },
        "exact_prefilter": {
            "all": confusion(rows, "rule_predicted_vehicle_relevant"),
            **{stratum: confusion(
                [row for row in rows if row["sample_stratum"] == stratum],
                "rule_predicted_vehicle_relevant",
            ) for stratum in strata},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    output = args.output_dir
    metrics_path = output / "relevance_cv_metrics.json"
    predictions_path = output / "relevance_cv_predictions.csv"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError(f"Evaluation output already exists in {output}")
    rows = read_labels(args.sample)
    metrics = evaluate(rows, args.folds, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "vehicle_relevant": "yes" if row["vehicle_relevant"] else "no",
                "predicted_vehicle_relevant": (
                    "yes" if row["predicted_vehicle_relevant"] else "no"
                ),
            })
    with metrics_path.open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
