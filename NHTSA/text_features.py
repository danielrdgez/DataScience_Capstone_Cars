"""Prepare and compare NHTSA text features, with optional semantic-label review.

Run with ``python -m NHTSA.text_features --help`` from the project root.
Source snapshots and the production model feature sets are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from EDA.DATA_CLEANING import FINAL_DATASET_PATH, scan_final_table
from MODELS import Price_ML_Models as price


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "MODEL_OUTPUT/nhtsa_text/pilot_v1"
KEYS = ["ModelYear", "Make", "Model"]
VERSION = "nhtsa-text-v1"
COMPARISON_FEATURE_SETS = (
    "base", "vpic_enriched", "complaints", "recalls", "safety", "nhtsa_structured",
    "components", "text_support", "interpretable", "tfidf", "embeddings",
    "enriched", "nhtsa_text", "nhtsa_text_youtube", "nhtsa_text_youtube_coverage",
    "nhtsa_text_youtube_topics",
)
FULL_COMPARISON_SETS = ["base", "enriched", "nhtsa_text", "nhtsa_text_youtube"]
CHANNELS = {
    "complaint": ("complaint", "summary"),
    "defect": ("recall", "summary"),
    "hazard": ("recall", "consequence"),
    "remedy": ("recall", "remedy"),
}
RULES = {
    "stall": ("complaint", r"\b(stall(?:ed|ing|s)?|engine (?:died|shut off)|lost (?:engine )?power)\b"),
    "oil_leak": ("complaint", r"\b(oil leak\w*|leak\w* oil|oil starvation|lost all (?:of )?its oil)\b"),
    "brake_loss": ("complaint", r"\b(brake(?:s)? fail\w*|lost (?:the )?brakes|loss of braking|brakes? (?:locked|did not work))\b"),
    "steering_loss": ("complaint", r"\b(lost (?:power )?steering|steering (?:locked|fail\w*)|loss of steering)\b"),
    "transmission_failure": ("complaint", r"\b(transmission (?:fail\w*|slip\w*|jerk\w*)|unable to shift)\b"),
    "electrical_failure": ("complaint", r"\b(electrical (?:fail\w*|short\w*|problem)|battery drain\w*|dead battery)\b"),
    "towing": ("complaint", r"\b(towed|tow truck)\b"),
    "repeat_failure": ("complaint", r"\b(failed again|same problem|recurring|problem (?:returned|recurred)|happened (?:several|multiple) times)\b"),
    "repair_delay": ("complaint", r"\b(back.?order\w*|parts? (?:were |was |are |is )?(?:not available|unavailable)|waiting for parts)\b"),
    "unsuccessful_repair": ("complaint", r"\b(repair\w* (?:did not|didn't|failed to) (?:fix|resolve)|problem (?:persisted|continued) after)\b"),
    "leak": ("defect", r"\b(leak\w*|fluid loss)\b"),
    "corrosion": ("defect", r"\b(corro\w*|rust\w*)\b"),
    "overheat": ("defect", r"\b(overheat\w*|thermal runaway)\b"),
    "software_fault": ("defect", r"\b(software|programming|calibration)\b"),
    "separation": ("defect", r"\b(separat\w*|detach\w*|fractur\w*)\b"),
    "fire_risk": ("hazard", r"\b(fire|ignite|ignition|combustion)\b"),
    "braking_risk": ("hazard", r"\b(loss of brak\w*|brak\w* performance|stopping distance|unable to stop)\b"),
    "propulsion_risk": ("hazard", r"\b(stall\w*|loss of (?:motive power|propulsion)|engine (?:shut|stop)\w*)\b"),
    "control_risk": ("hazard", r"\b(loss of (?:control|steering)|lose control|steering control)\b"),
    "injury_risk": ("hazard", r"\b(injur\w*|death|fatal\w*)\b"),
    "replacement": ("remedy", r"\b(replace\w*|replacement)\b"),
    "repair": ("remedy", r"\b(repair\w*|tighten\w*|reseal\w*|install\w*)\b"),
    "inspection": ("remedy", r"\b(inspect\w*)\b"),
    "software_update": ("remedy", r"\b(reprogram\w*|software update|update\w* (?:the )?software|recalibrat\w*)\b"),
    "remedy_unavailable": ("remedy", r"\b(remedy (?:is |was )?(?:not yet available|unavailable|under development))\b"),
}
COMPONENTS = {
    "powertrain": r"ENGINE|POWER TRAIN",
    "brakes": r"BRAKE",
    "steering": r"STEERING",
    "electrical": r"ELECTRICAL",
    "airbags": r"AIR BAG",
    "fuel": r"FUEL SYSTEM",
    "structure": r"STRUCTURE",
    "visibility": r"VISIBILITY|EXTERIOR LIGHTING",
}
EDITABLE = {"review_status", "human_labels", "language", "evidence", "notes"}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_keys(frame):
    return frame.with_columns(
        pl.col("ModelYear").cast(pl.Int64, strict=False),
        *[pl.col(c).str.strip_chars().str.to_uppercase().str.replace_all(r"\s+", " ") for c in KEYS[1:]],
    )


def rule_labels(row):
    labels = []
    for label, (channel, pattern) in RULES.items():
        source, field = CHANNELS[channel]
        if row["source"] != source:
            continue
        text = (row.get(field) or "").lower()
        for match in re.finditer(pattern, text):
            context = text[max(0, match.start() - 45):match.start()]
            context = re.split(r"[.;!?]", context)[-1]
            negated = re.search(r"\b(no|not|never|without)\b(?:\W+\w+){0,3}\W*$", context)
            hypothetical = source == "complaint" and re.search(
                r"\b(could|may|might|would|will|risk of|worried about)\b", context,
            )
            if label in {"repair_delay", "unsuccessful_repair", "remedy_unavailable"} or not (negated or hypothetical):
                labels.append(label)
                break
    return labels


def prepare_records(as_of, families):
    outputs, audit = [], {}
    for source, table, identifier, dated in (
        ("complaint", "nhtsa_complaints", "odiNumber", "dateComplaintFiled"),
        ("recall", "nhtsa_recalls", "NHTSACampaignNumber", "ReportReceivedDate"),
    ):
        names = (["summary", "components", "products", "crash", "fire", "numberOfInjuries", "numberOfDeaths"]
                 if source == "complaint" else ["Summary", "Consequence", "Remedy", "Component", "ModelYear", "Make", "Model"])
        raw = scan_final_table(table).select("model_year", "make", "model", identifier, dated, *names).collect()
        eligible = raw.filter(pl.col(dated).is_not_null() & (pl.col(dated) <= as_of)
                              & pl.col(identifier).is_not_null())
        if source == "complaint":
            eligible = eligible.filter(pl.struct("products", "model_year", "make", "model")
                                       .map_elements(price._complaint_product_matches, return_dtype=pl.Boolean))
            eligible = eligible.rename({"model_year": "ModelYear", "make": "Make", "model": "Model"})
        else:
            returned = normalized_keys(eligible.select(KEYS))
            query = normalized_keys(eligible.select(pl.col("model_year").alias("ModelYear"),
                                                   pl.col("make").alias("Make"), pl.col("model").alias("Model")))
            eligible = eligible.filter(pl.Series("applicable", (returned == query).to_numpy().all(axis=1)))
            eligible = eligible.drop("model_year", "make", "model")
            eligible = eligible.rename({"Summary": "summary", "Consequence": "consequence", "Remedy": "remedy", "Component": "components"})
        eligible = normalized_keys(eligible).join(families, on=KEYS, how="inner", validate="m:1")
        conflicts = ["summary", "components", "crash", "fire", "numberOfInjuries", "numberOfDeaths"] if source == "complaint" else ["summary", "consequence", "remedy", "components"]
        unique = eligible.group_by(*KEYS, identifier).agg(
            pl.col(dated).min().alias("evidence_date"),
            *[pl.col(c).first() for c in conflicts],
            pl.any_horizontal([pl.col(c).n_unique() > 1 for c in conflicts]).alias("conflict"),
        )
        audit[source] = {"source_rows": raw.height, "eligible_mapped_rows": eligible.height,
                         "distinct_family_ids": unique.height, "conflicting_family_ids": unique["conflict"].sum()}
        unique = unique.filter(~pl.col("conflict"))
        frame = unique.select(*KEYS, pl.lit(source).alias("source"),
                              pl.col(identifier).cast(pl.String).alias("source_id"), "evidence_date", "summary", "components",
                              *[pl.col(c) if c in unique.columns else pl.lit(None, dtype=pl.String).alias(c) for c in ["consequence", "remedy"]])
        frame = frame.with_columns(*[pl.col(c).fill_null("").str.strip_chars() for c in ["summary", "consequence", "remedy", "components"]])
        outputs.append(frame)
    records = pl.concat(outputs)
    records = records.with_columns(pl.struct("summary", "consequence", "remedy").map_elements(
        lambda r: digest("\n".join(" ".join(r[c].lower().split()) for c in ["summary", "consequence", "remedy"])),
        return_dtype=pl.String).alias("text_hash"))
    records = records.with_columns((pl.col("source") + ":" + pl.col("source_id") + ":" + pl.col("text_hash")).alias("document_id"))
    return records.sort(*KEYS, "source", "source_id"), audit


def prepare(args):
    workspace = args.workspace
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("Workspace already contains files. Use a new --workspace; existing labels are preserved.")
    if (args.sample_size != 0 and args.sample_size < 100) or args.complaint_labels < 1 or args.recall_labels < 1:
        raise ValueError("Use at least 100 listings (or 0 for all) and positive annotation sample sizes.")
    paths = [FINAL_DATASET_PATH / name for name in ["used_cars.parquet", "NHTSA_vPIC.parquet", "NHTSA_Complaints.parquet", "NHTSA_Recalls.parquet", "NHTSA_Safety.parquet"]]
    hashes = {p.name: price.file_sha256(p) for p in paths}
    print("Preparing matching listings, split assignments, and eligible NHTSA text...", flush=True)
    listings, info = price.load_model_frame("enriched", args.sample_size or None, issue_as_of=args.as_of)
    groups = listings["_cohort_group"].to_numpy() if args.split_strategy == "unseen_family" else listings["vin"].to_numpy()
    fit, validation, test, split_info = price.split_by_vehicle_cohort(groups)
    if not len(validation):
        raise ValueError("A separate validation partition is required; increase --sample-size.")
    partition = np.full(listings.height, "fit", dtype=object)
    partition[validation], partition[test] = "validation", "test"
    listings = listings.with_columns(pl.Series("partition", partition), pl.Series("split_group", groups))
    families = normalized_keys(listings.select(KEYS)).unique().drop_nulls()
    records, audit = prepare_records(args.as_of, families)
    fit_families = normalized_keys(listings.filter(pl.col("partition") == "fit").select(KEYS)).unique().drop_nulls()
    candidates = records.join(fit_families, on=KEYS, how="inner").filter(pl.col("summary").str.len_chars() >= 20)
    candidates = candidates.sort("document_id").unique(subset=["source", "source_id"], maintain_order=True)
    if args.exclude_sample:
        excluded = pl.read_parquet(args.exclude_sample)
        candidates = candidates.join(excluded.select("source", "source_id").unique(), on=["source", "source_id"], how="anti")
        candidates = candidates.join(excluded.select("text_hash").unique(), on="text_hash", how="anti")
    candidates = candidates.with_columns(
        pl.when(pl.col("text_hash").hash(seed=43) % 5 == 0).then(pl.lit("holdout"))
        .otherwise(pl.lit("development")).alias("annotation_split"),
        pl.col("text_hash").hash(seed=42).alias("sample_order"),
    )
    samples = []
    for source, count in [("complaint", args.complaint_labels), ("recall", args.recall_labels)]:
        for split, size in [("development", count - count // 5), ("holdout", count // 5)]:
            samples.append(candidates.filter((pl.col("source") == source) & (pl.col("annotation_split") == split))
                           .sort("sample_order").head(size))
    sample = pl.concat(samples).drop("sample_order").sort("source", "annotation_split", "document_id")
    sample = sample.with_columns(pl.struct(sample.columns).map_elements(
        lambda r: digest(json.dumps(r, sort_keys=True, default=str)), return_dtype=pl.String).alias("row_signature"))
    sample = sample.with_columns(pl.struct("source", "summary", "consequence", "remedy")
                                .map_elements(lambda r: "|".join(rule_labels(r)), return_dtype=pl.String).alias("suggested_labels"))
    if {p.name: price.file_sha256(p) for p in paths} != hashes:
        raise RuntimeError("Source files changed during preparation. Retry against a stable snapshot.")
    workspace.mkdir(parents=True, exist_ok=True)
    listings.write_parquet(workspace / "listings.parquet")
    records.write_parquet(workspace / "records.parquet")
    sample.write_parquet(workspace / "annotation_sample.parquet")
    editable = sample.with_columns(*[pl.lit("").alias(c) for c in sorted(EDITABLE)])
    editable.write_csv(workspace / "labels.csv", include_bom=True)
    manifest = {"version": VERSION, "as_of_date": args.as_of.isoformat(), "interpretation": "retrospective_association",
                "price_observation_date_verified": False, "historical_text_versions_verified": False,
                "sample_size_requested": args.sample_size, "listing_rows": listings.height,
                "excluded_annotation_sample": str(args.exclude_sample) if args.exclude_sample else None,
                "excluded_annotation_sample_sha256": price.file_sha256(args.exclude_sample) if args.exclude_sample else None,
                "base_features": info["feature_columns"], "split_strategy": args.split_strategy,
                "split_details": split_info, "source_sha256": hashes,
                "script_sha256": price.file_sha256(Path(__file__)),
                "price_script_sha256": price.file_sha256(Path(price.__file__)), "data_info": info, "records_audit": audit,
                "label_definitions": {k: {"channel": channel, "pattern": pattern} for k, (channel, pattern) in RULES.items()},
                "annotation_counts": sample.group_by("source", "annotation_split").len().to_dicts(),
                "files_sha256": {name: price.file_sha256(workspace / name) for name in ["listings.parquet", "records.parquet", "annotation_sample.parquet"]}}
    write_json(workspace / "workflow.json", manifest)
    print(json.dumps({"workspace": str(workspace), "annotation_counts": manifest["annotation_counts"], "audit": audit}, indent=2))


def checked_workspace(workspace):
    manifest = read_json(workspace / "workflow.json")
    if manifest["version"] != VERSION:
        raise ValueError("Prepared data format changed. Prepare a new workspace.")
    if manifest["price_script_sha256"] != price.file_sha256(Path(price.__file__)):
        raise ValueError("Price-model helpers changed. Prepare a new workspace to preserve provenance.")
    for name, expected in manifest["files_sha256"].items():
        if price.file_sha256(workspace / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return manifest


def read_labels(workspace):
    original = pl.read_parquet(workspace / "annotation_sample.parquet").to_dicts()
    reference = {row["document_id"]: row for row in original}
    with (workspace / "labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(original) or len({r["document_id"] for r in rows}) != len(rows):
        raise ValueError("labels.csv must preserve every original document exactly once.")
    reviewed, errors = [], []
    for row in rows:
        frozen = reference.get(row["document_id"])
        if frozen is None or any(str(frozen[c] if frozen[c] is not None else "") != row.get(c, "") for c in frozen):
            errors.append({"document_id": row["document_id"], "error": "Source/sample columns changed"})
            continue
        status, language = row["review_status"].strip().lower(), row["language"].strip().lower()
        if status not in {"", "reviewed", "unclear"} or language not in {"", "english", "other", "unclear"}:
            errors.append({"document_id": row["document_id"], "error": "Invalid status or language"})
            continue
        if status != "reviewed":
            continue
        allowed = {label for label, (channel, _) in RULES.items() if CHANNELS[channel][0] == row["source"]}
        value = row["human_labels"].strip().lower()
        labels = {label.strip() for label in value.split("|")} if value and value != "none" else set()
        if not value or not labels <= allowed or not language or (labels and not row["evidence"].strip()):
            errors.append({"document_id": row["document_id"], "error": "Reviewed rows need valid labels (or none), language, and evidence for positives"})
            continue
        text = " ".join(" ".join(str(row.get(c, "")).lower().split()) for c in ["summary", "consequence", "remedy"])
        quotes = [" ".join(quote.lower().split()) for quote in row["evidence"].split("||") if quote.strip()]
        if labels and any(quote not in text for quote in quotes):
            errors.append({"document_id": row["document_id"], "error": "Evidence must be verbatim text; separate excerpts with ||"})
            continue
        if language == "english":
            reviewed.append({**frozen, "labels": labels})
    summary = {"sample_rows": len(rows), "reviewed_english_rows": len(reviewed), "pending_or_nonenglish_rows": len(rows) - len(reviewed),
               "counts": pd.DataFrame(reviewed).groupby(["source", "annotation_split"]).size().to_dict() if reviewed else {}, "errors": errors}
    summary["counts"] = {"/".join(key): int(value) for key, value in summary["counts"].items()}
    return reviewed, summary


def model_text(row, field):
    text = row.get(field) or ""
    for name in [row.get("Make"), row.get("Model")]:
        if name and len(str(name)) >= 3:
            text = re.sub(r"\b" + re.escape(str(name)) + r"\b", "vehicle", text, flags=re.I)
    text = re.sub(r"\b\d{2}[VE]\d{6}\b|\b(?:19|20)\d{2}\b", " ", text, flags=re.I)
    text = re.sub(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", " ", text)
    text = re.sub(r"\b(?=[A-HJ-NPR-Z0-9]{11,17}\b)(?=[A-HJ-NPR-Z0-9]*\d)[A-HJ-NPR-Z0-9]{11,17}\b", " ", text, flags=re.I)
    return " ".join(text.lower().split())


def fit_extractors(reviewed):
    extractors = {}
    for channel, (source, field) in CHANNELS.items():
        rows = [r for r in reviewed if r["source"] == source and r["annotation_split"] == "development"]
        texts = [model_text(r, field) for r in rows]
        if not rows or not any(texts):
            continue
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, sublinear_tf=True, dtype=np.float32)
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            continue
        models = {}
        for label, (label_channel, _) in RULES.items():
            if label_channel != channel:
                continue
            target = np.array([label in r["labels"] for r in rows], dtype=int)
            if min(target.sum(), len(target) - target.sum()) >= 5:
                models[label] = LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=42).fit(matrix, target)
        extractors[channel] = (vectorizer, models)
    return extractors


def predict_labels(rows, method, extractors=None):
    predictions = {label: np.zeros(len(rows), dtype=np.int8) for label in RULES}
    if method == "rules":
        for i, row in enumerate(rows):
            for label in rule_labels(row):
                predictions[label][i] = 1
    else:
        for channel, (vectorizer, models) in (extractors or {}).items():
            source, field = CHANNELS[channel]
            indices = [i for i, row in enumerate(rows) if row["source"] == source]
            if not indices:
                continue
            matrix = vectorizer.transform([model_text(rows[i], field) for i in indices])
            for label, model in models.items():
                predictions[label][indices] = model.predict(matrix)
    return predictions


def evaluate(workspace, method):
    checked_workspace(workspace)
    reviewed, summary = read_labels(workspace)
    if summary["errors"]:
        raise ValueError(json.dumps(summary["errors"][:10], indent=2))
    extractors = fit_extractors(reviewed) if method == "supervised" else {}
    metrics, accepted = [], []
    for split in ["development", "holdout"]:
        rows = [r for r in reviewed if r["annotation_split"] == split]
        predictions = predict_labels(rows, method, extractors)
        for label, (channel, _) in RULES.items():
            source = CHANNELS[channel][0]
            indices = [i for i, row in enumerate(rows) if row["source"] == source]
            if not indices:
                continue
            actual = [label in rows[i]["labels"] for i in indices]
            predicted = predictions[label][indices]
            precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", zero_division=0)
            positives, negatives = sum(actual), len(actual) - sum(actual)
            qualifies = split == "holdout" and positives >= 5 and negatives >= 5 and precision >= .8 and recall >= .5
            metrics.append({"source": source, "label": label, "split": split, "rows": len(actual),
                            "positives": positives, "negatives": negatives, "precision": float(precision),
                            "recall": float(recall), "f1": float(f1), "qualified": qualifies})
            if qualifies:
                accepted.append(label)
    ready = all(summary["counts"].get(f"{source}/holdout", 0) >= 20 and
                any(CHANNELS[RULES[label][0]][0] == source for label in accepted) for source in ["complaint", "recall"])
    report = {"method": method, "label_sha256": price.file_sha256(workspace / "labels.csv"),
              "summary": summary, "qualified_labels": accepted, "ready_for_price_experiment": ready,
              "qualification_rule": "At least 20 reviewed English holdout records per source; label has >=5 positives, >=5 negatives, precision >=0.80, recall >=0.50; at least one qualified label per source.",
              "interpretation": "Pilot acceptance check; development scores are descriptive, and holdout must not be used to tune rules or thresholds.",
              "metrics": metrics}
    return report, extractors


def aggregate_features(records, qualified):
    components, narrative = [], []
    for source in ["complaint", "recall"]:
        frame = records.filter(pl.col("source") == source)
        component_values = []
        for label, pattern in COMPONENTS.items():
            indicator = pl.col("components").str.to_uppercase().str.contains(pattern).cast(pl.Float64)
            value = indicator.mean() if source == "complaint" else indicator.sum()
            suffix = "share" if source == "complaint" else "campaign_count"
            component_values.append(value.alias(f"{source}_component_{label}_{suffix}"))
        components.append(frame.group_by(KEYS).agg(*component_values))
        useful = pl.col("summary").str.len_chars() >= 20
        fields = ["summary"] if source == "complaint" else ["summary", "consequence", "remedy"]
        expressions = [pl.len().alias(f"{source}_eligible_record_count"), useful.sum().alias(f"{source}_usable_summary_count"),
                       useful.mean().alias(f"{source}_usable_summary_share"),
                       (useful.sum() < 20).cast(pl.Int8).alias(f"{source}_low_text_support")]
        for field in fields:
            available = pl.col(field).str.len_chars() >= 20
            for label in qualified:
                channel, _ = RULES[label]
                if CHANNELS[channel] != (source, field):
                    continue
                count = pl.when(available).then(pl.col(label)).otherwise(0).sum()
                value = count / available.sum() if source == "complaint" else count
                expressions.append(pl.when(available.sum() > 0).then(value).otherwise(None)
                                   .alias(f"{source}_{label}_{'share' if source == 'complaint' else 'campaign_count'}"))
        narrative.append(frame.group_by(KEYS).agg(*expressions))
    component_frame = components[0].join(components[1], on=KEYS, how="full", coalesce=True, validate="1:1")
    narrative_frame = narrative[0].join(narrative[1], on=KEYS, how="full", coalesce=True, validate="1:1")
    return component_frame.join(narrative_frame, on=KEYS, how="full", coalesce=True, validate="1:1")


def build(args):
    manifest = checked_workspace(args.workspace)
    destination = args.workspace / args.build_name
    if destination.exists():
        raise ValueError("Feature build exists. Use a new --build-name to preserve its evaluation and lineage.")
    records = pl.read_parquet(args.workspace / "records.parquet")
    if args.method == "automatic":
        report = {"qualified_labels": [], "label_sha256": None}
        tagged = records
    else:
        report, extractors = evaluate(args.workspace, args.method)
        write_json(args.workspace / f"evaluation_{args.method}.json", report)
        if not report["ready_for_price_experiment"]:
            raise ValueError("Extraction is not qualified. Complete labels or use --method automatic for an unlabeled experiment.")
        scored = []
        for batch in records.iter_slices(10_000):
            values = predict_labels(batch.to_dicts(), args.method, extractors)
            scored.append(batch.with_columns(*[pl.Series(label, values[label]) for label in report["qualified_labels"]]))
        tagged = pl.concat(scored)
    features = aggregate_features(tagged, report["qualified_labels"]).with_columns(pl.lit(manifest["as_of_date"]).alias("as_of_date"))
    destination.mkdir()
    features.write_parquet(destination / "family_features.parquet")
    if args.method != "automatic":
        joblib.dump(extractors, destination / "extractor.joblib", compress=3)
        write_json(destination / "evaluation.json", report)
    write_json(destination / "feature_manifest.json", {
        "method": args.method, "as_of_date": manifest["as_of_date"], "qualified_labels": report["qualified_labels"],
        "component_columns": [c for c in features.columns if "_component_" in c],
        "text_columns": [c for c in features.columns if c not in KEYS + ["as_of_date"] and "_component_" not in c],
        "features_sha256": price.file_sha256(destination / "family_features.parquet"),
        "label_sha256": report["label_sha256"], "script_sha256": price.file_sha256(Path(__file__)),
        "preparation_script_sha256": manifest["script_sha256"],
        "semantic_label_accuracy": "not_evaluated" if args.method == "automatic" else "reviewed_english_holdout",
        "interpretation": "Family associations from submitted reports; counts and shares are not population failure probabilities.",
    })
    print(f"Built {features.height:,} family rows using {args.method}. Features: {destination}")
    if report["qualified_labels"]:
        print(f"Qualified labels: {', '.join(report['qualified_labels'])}")


def vector_features(records, training_families, args):
    from sklearn.decomposition import PCA, TruncatedSVD

    embedding_model = None
    if args.representation == "embeddings":
        if not args.embedding_model:
            raise ValueError("Embeddings require --embedding-model pointing to a downloaded local sentence-transformers model.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ValueError("Install the optional sentence-transformers dependency before using embeddings.") from error
        embedding_model = SentenceTransformer(str(args.embedding_model), local_files_only=True)
    result, encoders = None, {}
    for channel, (source, field) in CHANNELS.items():
        available = records.filter((pl.col("source") == source) & (pl.col(field).str.len_chars() >= 20))
        documents = available.unique("document_id", maintain_order=True)
        training = available.join(training_families, on=KEYS, how="inner", validate="m:1").unique("document_id")
        training = training.with_columns(pl.col("document_id").hash(seed=42).alias("_order")).sort("_order").head(args.vector_training_documents)
        if training.height < 3:
            raise ValueError(f"Too few training documents for {channel} vectors.")
        texts = [model_text(row, field) for row in training.to_dicts()]
        print(f"{channel}: fitting {args.representation} on {training.height:,} training documents", flush=True)
        if args.representation == "tfidf":
            encoder = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, min_df=2, sublinear_tf=True, dtype=np.float32)
            matrix = encoder.fit_transform(texts)
            dimensions = min(args.vector_dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1)
            if dimensions < 1:
                raise ValueError(f"No usable vector vocabulary for {channel}.")
            reducer = TruncatedSVD(n_components=dimensions, random_state=42).fit(matrix)
        else:
            encoder = None
            matrix = embedding_model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
            dimensions = min(args.vector_dimensions, matrix.shape[0] - 1, matrix.shape[1])
            reducer = PCA(n_components=dimensions, random_state=42).fit(matrix)
        if dimensions < 1:
            raise ValueError(f"No usable vector vocabulary for {channel}.")
        names = [f"{args.representation}_{channel}_{i:02d}" for i in range(dimensions)]
        blocks = []
        for batch in documents.iter_slices(5000):
            batch_text = [model_text(row, field) for row in batch.to_dicts()]
            encoded = encoder.transform(batch_text) if encoder is not None else embedding_model.encode(
                batch_text, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            vectors = reducer.transform(encoded).astype(np.float32)
            blocks.append(batch.select("document_id").hstack(pl.DataFrame(vectors, schema=names)))
        mapping = pl.concat(blocks)
        pooled = available.select(*KEYS, "document_id").join(mapping, on="document_id", validate="m:1")
        pooled = pooled.group_by(KEYS).agg(*[pl.col(c).mean() for c in names])
        result = pooled if result is None else result.join(pooled, on=KEYS, how="full", coalesce=True, validate="1:1")
        encoders[channel] = {"vectorizer": encoder, "reducer": reducer, "columns": names,
                             "training_documents": training.height, "field": field, "source": source}
        print(f"{channel}: fitted {dimensions} vector dimensions on {training.height:,} training documents", flush=True)
    return result, encoders


def join_family_features(listings, features):
    identities = normalized_keys(listings.select(KEYS)).with_row_index("_row")
    extra = identities.join(features.drop("as_of_date") if "as_of_date" in features.columns else features,
                            on=KEYS, how="left", validate="m:1", maintain_order="left")
    extra = extra.sort("_row").drop(*KEYS, "_row")
    if extra.height != listings.height:
        raise ValueError("Feature join changed the listing grain.")
    return listings.hstack(extra)


def comparison_feature_sets(manifest, feature_info, vector_columns):
    base = list(price.BASE_FEATURE_COLUMNS)
    vpic = base + manifest["data_info"]["vpic_feature_columns"]
    structured = manifest["base_features"]
    components = feature_info["component_columns"]
    text = feature_info["text_columns"]
    all_text = structured + components + text
    groups = {"base": base, "vpic_enriched": vpic, "nhtsa_structured": structured, "enriched": structured,
              "components": structured + components, "text_support": all_text}
    if feature_info["method"] != "automatic":
        groups["interpretable"] = all_text
    for source, values in [("complaint", price.COMPLAINT_VALUE_FEATURES + price.COMPLAINT_COVERAGE_FEATURES),
                           ("recall", price.RECALL_VALUE_FEATURES + price.RECALL_COVERAGE_FEATURES)]:
        additional = [c for c in components + text if c.startswith(source + "_")]
        channels = [channel for channel, (record_source, _) in CHANNELS.items() if record_source == source]
        vectors = [c for c in vector_columns if any(c.startswith(f"{method}_{channel}_")
                   for method in ["tfidf", "embeddings"] for channel in channels)]
        groups[source + "s"] = vpic + list(values) + additional + vectors
    groups["safety"] = vpic + list(price.SAFETY_FEATURE_COLUMNS)
    for method in ["tfidf", "embeddings"]:
        vectors = [c for c in vector_columns if c.startswith(method + "_")]
        if vectors:
            groups[method] = all_text + vectors
    return {name: list(dict.fromkeys(columns)) for name, columns in groups.items()}


def load_youtube_features(directory):
    directory = Path(directory)
    info = read_json(directory / "price_feature_manifest.json")
    if info["state"] != "complete" or info["attribution_scope"] != "explicit_comment" or info["window"] != "cumulative":
        raise ValueError("YouTube comparison needs a complete explicit-comment/cumulative price export.")
    frame = pl.read_parquet(directory / "make_features.parquet")
    if price.file_sha256(directory / "make_features.parquet") != info["features_sha256"]:
        raise ValueError("Selected YouTube features changed after export.")
    columns = [c for block in info["feature_blocks"].values() for c in block]
    if len(columns) != len(set(columns)) or set(frame.columns) != {"Make", *columns}:
        raise ValueError("Selected YouTube predictor schema differs from its manifest.")
    if any(not c.startswith("yt_") for c in columns):
        raise ValueError("Only declared YouTube predictors may enter this feature block.")
    frame = frame.with_columns(pl.col("Make").str.strip_chars().str.to_uppercase().str.replace_all(r"\s+", " "))
    if frame["Make"].null_count() or frame["Make"].n_unique() != frame.height:
        raise ValueError("YouTube feature makes are missing or duplicated.")
    info = {**info, "manifest_sha256": price.file_sha256(directory / "price_feature_manifest.json")}
    return frame, info


def join_youtube_features(listings, features):
    identities = listings.select(pl.col("Make").cast(pl.String).str.strip_chars().str.to_uppercase()
                                 .str.replace_all(r"\s+", " ")).with_row_index("_row")
    joined = identities.join(features, on="Make", how="left", validate="m:1", maintain_order="left")
    extra = joined.sort("_row").drop("Make", "_row")
    if extra.height != listings.height:
        raise ValueError("YouTube join changed the listing grain.")
    flags = [c for c in extra.columns if c.endswith("_missing") or c == "yt_support_suppressed"]
    extra = extra.with_columns(pl.col("yt_feature_row_present").fill_null(0),
                               *[pl.col(c).fill_null(True) for c in flags])
    return listings.hstack(extra)


def write_feature_comparison(output, phase, overall, segments):
    """Matched incremental and baseline comparisons on the same evaluation rows."""
    metrics = ["mae", "rmse", "r2", "wape_pct", "mean_signed_error"]
    rows = [{"variant": entry["variant"], "model": entry["model"], "price_weighting": entry["price_weighting"],
             "dimension": "overall", "segment": "all",
             **{key: entry.get(f"validation_{key}" if phase == "validation" else key) for key in metrics}}
            for entry in overall]
    frame = pd.DataFrame(rows + segments)
    pairs = []
    links = [("enriched", "base"), ("nhtsa_text", "enriched"), ("nhtsa_text", "base"),
             ("nhtsa_text_youtube", "nhtsa_text"), ("nhtsa_text_youtube", "base"),
             ("nhtsa_text_youtube_coverage", "nhtsa_text"),
             ("nhtsa_text_youtube_topics", "nhtsa_text_youtube_coverage"),
             ("nhtsa_text_youtube", "nhtsa_text_youtube_topics")]
    keys = ["model", "price_weighting", "dimension", "segment"]
    for variant, reference in links:
        current, baseline = frame[frame.variant == variant], frame[frame.variant == reference]
        if current.empty or baseline.empty:
            continue
        pair = current.merge(baseline, on=keys, suffixes=("", "_reference"), validate="one_to_one")
        for key in metrics:
            pair[key + "_change"] = pair[key] - pair[key + "_reference"]
        pairs.append(pair)
    if pairs:
        pd.concat(pairs, ignore_index=True).to_csv(output / f"{phase}_feature_comparison.csv", index=False)


def write_weighting_comparison(output, phase, overall, segments):
    """Pair unweighted evaluation scores from both training treatments."""
    rows = []
    for entry in overall:
        rows.append({"variant": entry["variant"], "model": entry["model"],
                     "price_weighting": entry["price_weighting"], "dimension": "overall", "segment": "all",
                     **{metric: entry.get(f"validation_{metric}" if phase == "validation" else metric)
                        for metric in ["mae", "rmse", "r2", "wape_pct", "mean_signed_error"]}})
    rows.extend(segments)
    frame = pd.DataFrame(rows)
    keys = ["variant", "model", "dimension", "segment"]
    regular = frame[frame.price_weighting == "none"]
    weighted = frame[frame.price_weighting == "high_price"]
    pairs = regular.merge(weighted, on=keys, suffixes=("_none", "_high_price"), validate="one_to_one")
    for metric in ["mae", "rmse", "r2", "wape_pct", "mean_signed_error"]:
        pairs[f"{metric}_change"] = pairs[f"{metric}_high_price"] - pairs[f"{metric}_none"]
    pairs.to_csv(output / f"{phase}_weighting_comparison.csv", index=False)


def compare(args):
    manifest = checked_workspace(args.workspace)
    directory = args.workspace / args.build_name
    if args.evaluate_test and (args.workspace / "test_evaluated.json").exists():
        raise ValueError("This workspace already exposed its test set. Preserve that result; do not retune against it.")
    configuration = {key: str(getattr(args, key)) if key == "embedding_model" else getattr(args, key)
                     for key in ["models", "max_trees", "tuning_rows", "representation", "vector_dimensions", "vector_training_documents", "embedding_model"]}
    requested = args.feature_sets or ["base", "vpic_enriched", args.representation if args.representation != "none" else "nhtsa_structured"]
    if len(requested) != len(set(requested)):
        raise ValueError("Feature sets must be unique.")
    configuration["feature_sets"] = requested
    configuration["price_weightings"] = args.price_weightings
    youtube_directory = getattr(args, "youtube_features", None)
    youtube_frame, youtube_info = load_youtube_features(youtube_directory) if youtube_directory else (None, None)
    if youtube_info:
        configuration["youtube_features_sha256"] = youtube_info["features_sha256"]
        configuration["youtube_manifest_sha256"] = youtube_info["manifest_sha256"]
    configuration["comparison_version"] = 4
    prior = read_json(directory / "selected_comparison.json") if args.evaluate_test else None
    if prior is not None and prior["configuration"] != configuration:
        raise ValueError("Test evaluation must use the exact settings of the selected validation comparison.")
    feature_info = read_json(directory / "feature_manifest.json")
    if feature_info["features_sha256"] != price.file_sha256(directory / "family_features.parquet"):
        raise ValueError("Feature file changed after building.")
    if feature_info["label_sha256"] is not None and feature_info["label_sha256"] != price.file_sha256(args.workspace / "labels.csv"):
        raise ValueError("Labels changed after feature building. Create a new feature build.")
    if feature_info["preparation_script_sha256"] != manifest["script_sha256"]:
        raise ValueError("Feature build refers to a different prepared workspace.")
    frame = pl.read_parquet(args.workspace / "listings.parquet")
    records = pl.read_parquet(args.workspace / "records.parquet")
    features = pl.read_parquet(directory / "family_features.parquet")
    frame = join_family_features(frame, features)
    if youtube_frame is not None:
        frame = join_youtube_features(frame, youtube_frame)
        if not youtube_info["directional_sentiment_available"]:
            warnings.warn(youtube_info["sentiment_limitation"], UserWarning)
    encoders = {}
    if args.representation != "none" and not args.evaluate_test:
        training_families = normalized_keys(frame.filter(pl.col("partition") == "fit").select(KEYS)).unique().drop_nulls()
        vectors, encoders = vector_features(records, training_families, args)
        frame = join_family_features(frame, vectors)
    groups = comparison_feature_sets(manifest, feature_info, [c for c in frame.columns if c.startswith(("tfidf_", "embeddings_"))])
    if args.representation in groups:
        groups["nhtsa_text"] = groups[args.representation]
        if youtube_info:
            coverage = youtube_info["feature_blocks"]["coverage"]
            topics = youtube_info["feature_blocks"]["topics"]
            sentiment = youtube_info["feature_blocks"]["sentiment"]
            groups["nhtsa_text_youtube_coverage"] = groups["nhtsa_text"] + coverage
            groups["nhtsa_text_youtube_topics"] = groups["nhtsa_text"] + coverage + topics
            groups["nhtsa_text_youtube"] = groups["nhtsa_text"] + coverage + topics + sentiment
    if not args.evaluate_test and any(name not in groups for name in requested):
        raise ValueError("Requested features are unavailable. Vector sets need a matching --representation; interpretable needs reviewed labels.")
    candidates = {name: groups[name] for name in requested} if not args.evaluate_test else {}
    categorical = [c for c in manifest["base_features"] if frame.schema[c] == pl.String]
    data = frame.to_pandas()
    all_makes = sorted(data["Make"].dropna().unique())
    fit = np.flatnonzero(data["partition"].eq("fit"))
    validation = np.flatnonzero(data["partition"].eq("validation"))
    test = np.flatnonzero(data["partition"].eq("test"))
    rng = np.random.default_rng(42)
    tuning_fit = np.sort(rng.choice(fit, min(args.tuning_rows, len(fit)), replace=False))
    tuning_validation = np.sort(rng.choice(validation, min(args.tuning_rows, len(validation)), replace=False))
    y = data["price"].to_numpy(dtype=float)
    y_log = np.log(y)
    diagnostics, selections, validation_segments, validation_year_make = [], [], [], []
    validation_make_price, validation_focus = [], []
    output = directory / (prior["output_directory"] if prior else datetime.now(timezone.utc).strftime("comparison_%Y%m%dT%H%M%S%fZ"))
    if not args.evaluate_test:
        output.mkdir()
    else:
        previous = read_json(output / "run_metadata.json")
        if previous["script_sha256"] != price.file_sha256(Path(__file__)):
            raise ValueError("Comparison code changed after validation. Rerun validation before exposing the test.")
        if previous["features_sha256"] != feature_info["features_sha256"] or previous["label_sha256"] != feature_info["label_sha256"]:
            raise ValueError("Frozen validation selection does not match the feature build.")
        candidates = previous["feature_sets"]
        if any(c.startswith("yt_") for columns in candidates.values() for c in columns) and youtube_frame is None:
            raise ValueError("The frozen comparison requires the same --youtube-features export.")
        selections = read_json(output / "validation_selection.json")["candidates"]
        encoders = joblib.load(output / "validation_text_encoders.joblib") if args.representation != "none" else {}
    for variant, columns in (candidates.items() if not args.evaluate_test else []):
        variant_categorical = [c for c in categorical if c in columns]
        numbers = [c for c in columns if c not in variant_categorical]
        preprocessor = price.PricePreprocessor(variant_categorical, numbers)
        x_fit = preprocessor.fit_transform(data.iloc[tuning_fit][columns])
        x_validation = preprocessor.transform(data.iloc[tuning_validation][columns])
        for weighting in args.price_weightings:
            weights = price.training_price_weights(y[tuning_fit], weighting)
            for model_name in args.models:
                print(f"Comparing {variant} / {model_name} / {weighting}: {len(columns)} inputs, {len(tuning_fit):,} fit rows", flush=True)
                local_diagnostics, evaluation = [], {}
                iterations, cap_limited = None, False
                if model_name in price.TREE_MODEL_NAMES:
                    iterations, score, parameters, cap_limited = price._fit_with_early_stopping(
                        model_name, x_fit, y_log[tuning_fit], x_validation, y_log[tuning_validation],
                        variant_categorical, args.max_trees, local_diagnostics,
                        sample_weight=weights, selected_evaluation=evaluation)
                else:
                    _, parameters, score = price._fit_classical_model(
                        model_name, x_fit, y_log[tuning_fit], x_validation, y_log[tuning_validation],
                        variant_categorical, [c for c in x_fit.columns if c not in variant_categorical],
                        local_diagnostics, sample_weight=weights, selected_evaluation=evaluation)
                selections.append({"variant": variant, "model": model_name, "price_weighting": weighting,
                                   **{k: v for k, v in evaluation.items() if k != "predictions"},
                                   "iterations": iterations, "parameters": parameters, "cap_limited": cap_limited})
                diagnostics.extend({"variant": variant, "price_weighting": weighting, **entry} for entry in local_diagnostics)
                receipt = data.iloc[tuning_validation][["vin", *price.COHORT_COLUMNS, "make_name", "is_new",
                          "cohort_complete", "split_identity_source", "mileage_review_flag", "price_review_flag", "split_group"]].copy()
                receipt["family_block"] = data.iloc[tuning_validation]["_cohort_group"].to_numpy()
                receipt["actual_price"], receipt["prediction"] = y[tuning_validation], evaluation["predictions"]
                tags = {"variant": variant, "price_weighting": weighting}
                validation_segments.extend({**row, **tags} for row in price.segment_metrics(receipt, model_name))
                validation_year_make.extend({**row, **tags} for row in price.year_make_metrics(receipt, model_name))
                validation_make_price.extend({**row, **tags} for row in price.make_price_metrics(receipt, model_name, makes=all_makes))
                validation_focus.extend({**row, **tags} for row in price.make_price_metrics(receipt, model_name, focus_only=True))
                print(f"{variant} / {model_name} / {weighting}: validation MAE ${score:,.2f}; "
                      f"R²={evaluation['validation_r2']}; tree cap limited={cap_limited}", flush=True)
    selected = min(selections, key=lambda row: row["validation_mae"])
    if not args.evaluate_test:
        write_json(output / "validation_selection.json", {"winner": selected, "candidates": selections, "diagnostics": diagnostics})
        pd.DataFrame([{**row, "parameters": json.dumps(row["parameters"])} for row in selections]).to_csv(output / "validation_metrics.csv", index=False)
        pd.DataFrame(validation_segments).to_csv(output / "validation_segment_metrics.csv", index=False)
        pd.DataFrame(validation_year_make).to_csv(output / "validation_year_make_metrics.csv", index=False)
        pd.DataFrame(validation_make_price).to_csv(output / "validation_make_price_metrics.csv", index=False)
        pd.DataFrame(validation_focus).to_csv(output / "validation_focus_make_metrics.csv", index=False)
        write_weighting_comparison(output, "validation", selections, validation_segments)
        write_feature_comparison(output, "validation", selections, validation_segments)
        write_json(directory / "selected_comparison.json", {"output_directory": output.name, "configuration": configuration})
    metadata = {"status": "validation_complete", "interpretation": manifest["interpretation"],
                "historical_prediction_qualified": False, "as_of_date": manifest["as_of_date"],
                "source_sha256": manifest["source_sha256"], "split_strategy": manifest["split_strategy"],
                "split_fingerprint": price.split_fingerprint(data.rename(columns={"_cohort_group": "family_block"})),
                "features_sha256": feature_info["features_sha256"], "label_sha256": feature_info["label_sha256"],
                "script_sha256": price.file_sha256(Path(__file__)), "feature_build_script_sha256": feature_info["script_sha256"],
                "preparation_script_sha256": manifest["script_sha256"], "configuration": configuration,
                "feature_method": feature_info["method"], "winner_selected_on": "validation_mae_usd",
                "test_used_for_selection": False, "representation": args.representation,
                "youtube_features": youtube_info,
                "price_reporting": {"band_width_usd": price.PRICE_BAND_WIDTH, "interval": "[lower, upper)",
                                    "high_price_threshold_usd": price.HIGH_PRICE_THRESHOLD,
                                    "make_source": "NHTSA_vPIC.Make", "all_makes_reported": True},
                "embedding_model": str(args.embedding_model) if args.embedding_model else None,
                "price_weighting_policy": {"reference_price_usd": price.PRICE_WEIGHT_REFERENCE,
                                           "max_training_weight": price.PRICE_WEIGHT_CAP,
                                           "formula": "clip(training_price / reference_price, 1, cap)",
                                           "evaluation_weighting": "none"},
                "feature_sets": candidates}
    if encoders and not args.evaluate_test:
        joblib.dump(encoders, output / "validation_text_encoders.joblib", compress=3)
    write_json(output / "run_metadata.json", metadata)
    support_columns = ["partition", "vin", "price", "ModelYear", "Make", "Model", "Trim", "_cohort_group"]
    support = frame.select(support_columns).with_columns(
        ((pl.col("price") / price.PRICE_BAND_WIDTH).floor() * price.PRICE_BAND_WIDTH).cast(pl.Int64).alias("price_band_lower_usd"),
        (pl.col("price") >= price.HIGH_PRICE_THRESHOLD).alias("high_price"))
    support.group_by("partition", "Make", "price_band_lower_usd", "high_price").agg(
        pl.len().alias("rows"), pl.col("_cohort_group").n_unique().alias("family_blocks")
    ).sort("partition", "Make", "price_band_lower_usd").write_csv(output / "partition_make_price_support.csv")
    if youtube_info:
        frame.group_by("partition", "Make").agg(
            pl.len().alias("rows"), pl.col("yt_feature_row_present").sum().alias("matched_rows"),
            (~pl.col("yt_support_suppressed")).sum().alias("supported_topic_rows"),
            *[(~pl.col(c)).sum().alias(c.replace("_missing", "_supported_rows"))
              for c in frame.columns if c.startswith("yt_") and c.endswith("_missing")]
        ).sort("partition", "Make").write_csv(output / "youtube_join_coverage.csv")
    if not args.evaluate_test:
        print(f"Validation winner: {selected['variant']} / {selected['model']} / {selected['price_weighting']}. Review {output}; test remains unscored.")
        return
    from sklearn.pipeline import Pipeline

    outer = np.sort(np.concatenate([fit, validation]))
    write_json(args.workspace / "test_evaluated.json", {"selected_comparison": str(output), "status": "started", "configuration": configuration})
    if encoders:
        outer_families = normalized_keys(frame[outer].select(KEYS)).unique().drop_nulls()
        final_vectors, final_encoders = vector_features(records, outer_families, args)
        vector_columns = [c for c in frame.columns if c.startswith(args.representation + "_")]
        data = join_family_features(frame.drop(vector_columns), final_vectors).to_pandas()
        joblib.dump(final_encoders, output / "final_text_encoders.joblib", compress=3)
    frozen = selections
    metrics, segments, year_make, make_price, focus = [], [], [], [], []
    receipt_columns = ["vin", *price.COHORT_COLUMNS, "make_name", "is_new", "cohort_complete", "split_identity_source",
                       "mileage_review_flag", "price_review_flag", "split_group"]
    for candidate in frozen:
        variant, model_name, weighting = candidate["variant"], candidate["model"], candidate["price_weighting"]
        columns = candidates[variant]
        variant_categorical = [c for c in categorical if c in columns]
        preprocessor = price.PricePreprocessor(variant_categorical, [c for c in columns if c not in variant_categorical])
        x_train = preprocessor.fit_transform(data.iloc[outer][columns])
        x_test = preprocessor.transform(data.iloc[test][columns])
        weights = price.training_price_weights(y[outer], weighting)
        if model_name in price.TREE_MODEL_NAMES:
            estimator = price._make_tree_model(model_name, candidate["iterations"], early_stopping=False, parameters=candidate["parameters"])
            kwargs = {"cat_features": variant_categorical, "verbose": False} if model_name == "catboost" else {"categorical_feature": variant_categorical} if model_name == "lightgbm" else {"verbose": False}
            kwargs["sample_weight"] = weights
        else:
            estimator = price._make_classical_pipeline(model_name, variant_categorical,
                [c for c in x_train.columns if c not in variant_categorical]).set_params(**candidate["parameters"])
            kwargs = {"model__sample_weight": weights}
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(x_train, y_log[outer], **kwargs)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError(f"{variant} / {model_name} / {weighting} final refit did not converge; preserve test receipts")
        predicted = np.exp(estimator.predict(x_test))
        model_key = f"{variant}_{model_name}_{weighting}"
        tags = {"variant": variant, "price_weighting": weighting}
        metrics.append({"model": model_name, **tags, **price.prediction_metrics(y[test], predicted, data.iloc[test]["split_group"].to_numpy())})
        receipt = data.iloc[test][receipt_columns].copy()
        receipt["actual_price"], receipt["prediction"] = y[test], predicted
        receipt["family_block"] = data.iloc[test]["_cohort_group"].to_numpy()
        receipt.to_parquet(output / f"{model_key}_test_predictions.parquet", index=False)
        segments.extend({**row, **tags} for row in price.segment_metrics(receipt, model_name))
        year_make.extend({**row, **tags} for row in price.year_make_metrics(receipt, model_name))
        make_price.extend({**row, **tags} for row in price.make_price_metrics(receipt, model_name, makes=all_makes))
        focus.extend({**row, **tags} for row in price.make_price_metrics(receipt, model_name, focus_only=True))
        bundle = Pipeline([("input", preprocessor), ("estimator", price.ExpLogPricePredictor(estimator))])
        joblib.dump(bundle, output / f"{model_key}_predictor.joblib", compress=3)
    pd.DataFrame(metrics).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(segments).to_csv(output / "segment_metrics.csv", index=False)
    pd.DataFrame(year_make).to_csv(output / "year_make_metrics.csv", index=False)
    pd.DataFrame(make_price).to_csv(output / "make_price_metrics.csv", index=False)
    pd.DataFrame(focus).to_csv(output / "focus_make_metrics.csv", index=False)
    write_weighting_comparison(output, "test", metrics, segments)
    write_feature_comparison(output, "test", metrics, segments)
    metadata["status"] = "complete"
    metadata["test_candidates"] = frozen
    metadata["predictor_input"] = "Raw assembled numeric/categorical feature columns; vector runs additionally require the saved encoders and local embedding model."
    write_json(output / "run_metadata.json", metadata)
    write_json(args.workspace / "test_evaluated.json", {"selected_comparison": str(output), "status": "complete", "configuration": configuration})
    print(f"All {len(frozen)} declared validation-selected candidates evaluated once: {output}")


def run_full(args):
    """Prepare all eligible listings, compare, and optionally refit/score once."""
    if args.workspace.exists() and any(args.workspace.iterdir()):
        raise ValueError("Full run needs a new workspace. Existing runs and their tests are preserved.")
    if not args.youtube_features.exists():
        from YOUTUBE.text.features import export_price_features
        export_price_features(args.youtube_source_features, args.youtube_features)
    _, youtube_info = load_youtube_features(args.youtube_features)
    print(f"YouTube: {youtube_info['rows']} makes; retrospective cutoff {youtube_info['as_of']}", flush=True)
    if args.coverage_ablations:
        args.feature_sets = list(dict.fromkeys(args.feature_sets +
            ["nhtsa_text_youtube_coverage", "nhtsa_text_youtube_topics"]))
    args.method = "automatic"
    evaluate_test = args.evaluate_test
    args.evaluate_test = False
    prepare(args)
    build(args)
    compare(args)
    if evaluate_test:
        args.evaluate_test = True
        compare(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["prepare", "check-labels", "evaluate", "build", "compare", "run-full"]:
        child = commands.add_parser(command)
        child.add_argument("--workspace", type=Path,
                           default=ROOT / "MODEL_OUTPUT/nhtsa_text" / datetime.now(timezone.utc).strftime("full_youtube_%Y%m%dT%H%M%S%fZ")
                           if command == "run-full" else DEFAULT_WORKSPACE)
        if command in {"prepare", "run-full"}:
            child.add_argument("--as-of", type=date.fromisoformat, default=price.COMPLAINT_RECALL_AS_OF)
            child.add_argument("--sample-size", type=int, default=0 if command == "run-full" else 50_000,
                               help="VIN-hash sample size; 0 uses all eligible listings.")
            child.add_argument("--split-strategy", choices=["unseen_family", "known_family"], default="unseen_family")
            child.add_argument("--complaint-labels", type=int, default=400)
            child.add_argument("--recall-labels", type=int, default=200)
            child.add_argument("--exclude-sample", type=Path, help="Prior annotation_sample.parquet to exclude its IDs and exact text from a new review round.")
        if command == "evaluate":
            child.add_argument("--method", choices=["rules", "supervised"], default="rules")
        if command == "build":
            child.add_argument("--method", choices=["automatic", "rules", "supervised"], default="automatic")
        if command in {"build", "compare", "run-full"}:
            child.add_argument("--build-name", default="features_automatic")
        if command in {"compare", "run-full"}:
            child.add_argument("--models", nargs="+", choices=price.PILOT_MODEL_NAMES,
                               default=list(price.PILOT_MODEL_NAMES) if command == "run-full" else ["catboost"])
            child.add_argument("--price-weightings", nargs="+", choices=price.PRICE_WEIGHTING_MODES,
                               default=list(price.PRICE_WEIGHTING_MODES) if command == "run-full" else ["none"],
                               help="Compare none and high_price on identical rows; evaluation is always unweighted.")
            child.add_argument("--feature-sets", nargs="+", choices=COMPARISON_FEATURE_SETS,
                               default=FULL_COMPARISON_SETS.copy() if command == "run-full" else None,
                               help="Default: base, vpic_enriched, and nhtsa_structured (or the selected text representation).")
            child.add_argument("--max-trees", type=int, default=6400 if command == "run-full" else 800)
            child.add_argument("--tuning-rows", type=int, default=100_000 if command == "run-full" else 20_000)
            child.add_argument("--evaluate-test", action="store_true")
            child.add_argument("--representation", choices=["none", "tfidf", "embeddings"],
                               default="tfidf" if command == "run-full" else "none")
            child.add_argument("--vector-dimensions", type=int, default=16)
            child.add_argument("--vector-training-documents", type=int, default=100_000 if command == "run-full" else 20_000)
            child.add_argument("--embedding-model", type=Path)
            child.add_argument("--youtube-features", type=Path,
                               default=ROOT / "YOUTUBE/outputs/price_features_v1" if command == "run-full" else None,
                               help="Frozen make export created with YOUTUBE.text_pipeline price-export.")
        if command == "run-full":
            child.add_argument("--youtube-source-features", type=Path, default=ROOT / "YOUTUBE/outputs/text_features_v1")
            child.add_argument("--coverage-ablations", action="store_true",
                               help="Also isolate YouTube coverage and semantic/topic predictors from sentiment.")
    args = parser.parse_args()
    if args.command in {"compare", "run-full"} and min(args.max_trees, args.tuning_rows, args.vector_dimensions, args.vector_training_documents) < 1:
        parser.error("Training limits must be positive.")
    if args.command in {"compare", "run-full"} and (len(args.models) != len(set(args.models)) or
                                     len(args.price_weightings) != len(set(args.price_weightings))):
        parser.error("Models and price weightings must be unique.")
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "check-labels":
            checked_workspace(args.workspace)
            _, summary = read_labels(args.workspace)
            print(json.dumps(summary, indent=2))
            return int(bool(summary["errors"]))
        elif args.command == "evaluate":
            report, _ = evaluate(args.workspace, args.method)
            write_json(args.workspace / f"evaluation_{args.method}.json", report)
            print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2))
        elif args.command == "build":
            build(args)
        elif args.command == "run-full":
            run_full(args)
        else:
            compare(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
