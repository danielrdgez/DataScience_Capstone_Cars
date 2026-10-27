"""EPA-first discovery annotations and source-trained, family-held-out predictions.

This independently versioned layer preserves the frozen specification fits and
peer rankings. Human review is optional; source agreement is not human accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier

from MODELS import vehicle_classifiers_v5 as learning
from MODELS import vehicle_profiles as r
from MODELS import vehicle_sources_v5 as sources
from MODELS import vehicle_taxonomy as taxonomy
from MODELS import vehicle_similarity_v3 as similarity
from MODELS import vehicle_similarity_v4 as saved

REFERENCES = sources.CLASSIFICATION_REFERENCES
TABLE_URL = "https://en.wikipedia.org/wiki/Car_classification#Summary_of_classifications"
THRESHOLD = .70
DIMENSIONS = ("epa_group", "size_segment", "market_positioning")

# Illustrative model names in the supplied reference table, not year-scoped EPA
# assignments. Exact normalized family names only; no make-wide extrapolation.
REFERENCE_EXAMPLES = {
    "City car": [("Chevrolet", "Spark"), ("Fiat", "500"), ("Kia", "Picanto")],
    "Subcompact executive": [("Acura", "ILX"), ("Audi", "A3"), ("Lexus", "CT")],
    "Compact executive / entry-level luxury": [("Alfa Romeo", "Giulia"), ("Audi", "A4"), ("Lexus", "IS"), ("Mercedes-Benz", "C-Class")],
    "Mid-size luxury": [("Audi", "A6"), ("Cadillac", "CT5"), ("Mercedes-Benz", "E-Class"), ("Tesla", "Model S")],
    "Full-size luxury": [("Genesis", "G90"), ("Jaguar", "XJ"), ("Mercedes-Benz", "S-Class"), ("Porsche", "Panamera")],
    "Roadster sports car": [("BMW", "Z4"), ("Lotus", "Elise"), ("Mazda", "MX-5"), ("Porsche", "Boxster"), ("Mercedes-Benz", "SLK")],
    "Compact MPV": [("Ford", "C-Max"), ("Volkswagen", "Touran")],
    "People mover": [("Chrysler", "Pacifica"), ("Kia", "Carnival"), ("Toyota", "Sienna")],
    "Mini SUV": [("Ford", "EcoSport"), ("Jeep", "Renegade")],
    "Compact SUV": [("Tesla", "Model Y"), ("Toyota", "RAV4"), ("Ford", "Escape"), ("Honda", "CR-V"), ("Kia", "Sportage")],
    "Mid-size SUV": [("Ford", "Edge"), ("Hyundai", "Santa Fe"), ("Jeep", "Grand Cherokee"), ("Volkswagen", "Touareg"), ("Volvo", "XC90")],
    "Full-size SUV": [("Lincoln", "Navigator"), ("Land Rover", "Range Rover"), ("Chevrolet", "Suburban"), ("Toyota", "Land Cruiser"), ("Mercedes-Benz", "GLS-Class")],
}


def example_segment(row):
    normalize = sources.vocabulary.normalized
    identity = normalize(row.Make), normalize(row.market_family)
    return next((term for term, pairs in REFERENCE_EXAMPLES.items()
        if identity in {(normalize(make), normalize(model)) for make, model in pairs}), "")


def epa_group(value):
    """Collapse drive suffixes only; retain EPA wagon/truck/van distinctions."""
    return re.sub(r"\s*[-–]?\s*[24]WD\s*$", "", str(value), flags=re.I).strip()


def common_segment(size, positioning):
    """Terminology crosswalk, never an inference of an EPA regulatory class."""
    luxury = positioning in ("luxury", "ultra_luxury", "premium")
    if luxury and size == "compact":
        return "Subcompact executive / premium compact"
    if luxury and size == "midsize":
        return "Compact executive / entry-level luxury"
    if luxury and size == "large":
        return "Large luxury (executive/full-size unspecified)"
    return {"minicompact": "City car", "subcompact": "Supermini / subcompact",
        "compact": "Small family / compact", "midsize": "Large family / mid-size",
        "large": "Full-size car", "small_suv": "Mini / compact SUV (unspecified)",
        "midsize_suv": "Mid-size SUV", "large_suv": "Full-size SUV",
        "small_pickup": "Small pickup", "full_size_pickup": "Full-size pickup"}.get(size, "")


def fit_classifier(kind, x, y, cats, weights):
    if kind == "logistic":
        return learning.fit_model(kind, x, y, cats, weights)
    model = learning.estimator("logistic", cats)
    model.steps[-1] = ("trees", ExtraTreesClassifier(n_estimators=150, min_samples_leaf=3,
        max_features=.8, random_state=42, n_jobs=2))
    model.fit(x, y, trees__sample_weight=weights)
    return model


def source_predictions(data, labels, target):
    """Every emitted prediction excludes its entire family fold from fitting."""
    predictions, metrics = [], []
    for dimension in DIMENSIONS:
        selected = labels[labels.dimension.eq(dimension) & labels.status.isin(["observed", "source_supported"])]
        known = selected.groupby("profile_id").label.agg(lambda x: ";".join(sorted(set(x))))
        truth = data.profile_id.map(known).replace("", np.nan)
        truth = truth.mask(truth.fillna("").str.contains(";", regex=False))
        for fold in sorted(data.fold.unique()):
            train = data.fold.ne(fold) & truth.notna() & data.identity_complete
            support = data.loc[train].assign(label=truth[train]).groupby("label").family_block.nunique()
            supported = support[support.ge(3)].index
            train &= truth.isin(supported)
            if truth[train].nunique() < 2:
                metrics.append({"dimension": dimension, "fold": fold, "status": "insufficient_source_classes"})
                continue
            # Select using development families only; the query fold is never
            # consulted for estimator choice, fitting, or probability cutoff.
            calibration_family = data.family_block.map(lambda value: int(r.digest("hierarchy-select|" + value)[:8], 16) % 5 == 0)
            fit, calibration = train & ~calibration_family, train & calibration_family
            best, best_candidate = None, None
            for kind, with_make in (("logistic", False), ("extra_trees", True)):
                x, cats = learning.features(data, with_make)
                score = -1.
                if truth[fit].nunique() >= 2 and calibration.any():
                    candidate = fit_classifier(kind, x.loc[fit], truth[fit], cats, similarity.family_weights(data[fit]))
                    predicted = np.asarray(candidate.predict(x.loc[calibration])).ravel()
                    score = f1_score(truth[calibration], predicted, average="macro", sample_weight=similarity.family_weights(data[calibration]), zero_division=0)
                if best is None or score > best[0]:
                    best = score, kind, with_make
                    best_candidate = candidate if score >= 0 else None
            _, kind, with_make = best
            x, cats = learning.features(data, with_make)
            label_support = {}
            if best_candidate is not None:
                held = data.loc[calibration, ["profile_id", "family_block"]].copy()
                probabilities = best_candidate.predict_proba(x.loc[calibration])
                held["label"] = np.asarray(best_candidate.classes_)[probabilities.argmax(axis=1)]
                held["probability"] = probabilities.max(axis=1)
                held["truth"] = truth[calibration]
                for label, group in held[held.probability.ge(THRESHOLD)].groupby("label"):
                    group = group.sort_values("profile_id").drop_duplicates("family_block")
                    agreement = float(group.label.eq(group.truth).mean())
                    label_support[label] = {"families": len(group), "agreement": agreement,
                        "enabled": len(group) >= 3 and agreement >= .8}
            active_labels = {label for label, support_row in label_support.items() if support_row["enabled"]}
            model = fit_classifier(kind, x.loc[train], truth[train], cats, similarity.family_weights(data[train]))
            query = data.fold.eq(fold)
            probability = model.predict_proba(x.loc[query])
            confidence = probability.max(axis=1)
            assigned = np.asarray(model.classes_)[probability.argmax(axis=1)]
            rows = data.loc[query].copy()
            rows["label"], rows["probability"] = assigned, confidence
            rows["truth"] = truth[query]
            training_families = sorted(data.loc[train, "family_block"].unique())
            if set(training_families) & set(rows.family_block):
                raise ValueError("Classifier family leakage")
            model_name = f"{dimension}-fold-{fold}.joblib"
            joblib.dump({"model": model, "training_families": training_families, "excluded_fold": int(fold),
                "label_basis": "source_labels_without_human_review", "threshold": THRESHOLD,
                "development_label_support": label_support, "active_labels": sorted(active_labels)}, target / model_name)
            for label in model.classes_:
                test = rows[rows.truth.notna() & rows.label.eq(label) & rows.probability.ge(THRESHOLD)]
                # One deterministic representative per held-out family.
                test = test.sort_values("profile_id").drop_duplicates("family_block")
                metrics.append({"dimension": dimension, "fold": fold, "label": label,
                    "training_families": int(support[label]), "heldout_emitted_families": len(test),
                    "source_agreement": float(test.label.eq(test.truth).mean()) if len(test) else None,
                    "status": "experimental_source_trained", "human_reviewed": False,
                    "method": kind, "make_feature": with_make, "development_macro_f1": best[0],
                    "development_emitted_families": label_support.get(label, {}).get("families", 0),
                    "development_source_agreement": label_support.get(label, {}).get("agreement"),
                    "enabled": label in active_labels})
            unknown = rows.truth.isna() & rows.identity_complete & rows.probability.ge(THRESHOLD) & rows.label.isin(active_labels)
            if dimension == "epa_group":
                patterns = {"passenger_car": "Cars|Wagons|Two Seaters|Two-Seaters", "suv_crossover": "Sport Utility",
                    "pickup": "Pickup", "van_minivan": "Vans|Minivans"}
                unknown &= pd.Series([bool(re.search(patterns.get(body, r"(?!)"), label, re.I))
                    for body, label in zip(rows.body_family, rows.label)], index=rows.index)
            for row in rows[unknown].itertuples():
                predictions.append({"profile_id": row.profile_id, "dimension": dimension, "label": row.label,
                    "probability": row.probability, "policy": "source_trained", "status": "predicted",
                    "outer_fold": int(fold), "model_uri": str((target / model_name).resolve())})
        print(data.iloc[0].grain, dimension, "predictions", sum(p["dimension"] == dimension for p in predictions), flush=True)
    return pd.DataFrame(predictions, columns=["profile_id", "dimension", "label", "probability", "policy", "status", "outer_fold", "model_uri"]), pd.DataFrame(metrics)


def hierarchy_rows(data, labels, predictions, include_predictions=True):
    """One EPA parent and one common-term child per profile; unknowns stay visible."""
    result = data[["profile_id"]].copy()
    for dimension in DIMENSIONS:
        values = labels[labels.dimension.eq(dimension)].groupby("profile_id").label.agg(lambda x: ";".join(sorted(set(x))))
        result[dimension] = result.profile_id.map(values).fillna("")
        result[dimension + "_state"] = np.where(result[dimension].ne(""), "recorded", "unknown")
        if include_predictions and not predictions.empty:
            predicted = predictions[predictions.dimension.eq(dimension)].set_index("profile_id")
            fill = result[dimension].eq("") & result.profile_id.isin(predicted.index)
            result.loc[fill, dimension] = result.loc[fill, "profile_id"].map(predicted.label)
            result.loc[fill, dimension + "_state"] = "predicted"
    result["common_segment"] = [common_segment(s, p) for s, p in zip(result.size_segment, result.market_positioning)]
    result["common_segment_state"] = np.where(result.common_segment.eq(""), "unknown",
        np.where(result.size_segment_state.eq("predicted") | result.market_positioning_state.eq("predicted"), "predicted", "rule_derived"))
    examples = data.apply(example_segment, axis=1)
    has_example = examples.ne("")
    result.loc[has_example, "common_segment"] = examples[has_example]
    result.loc[has_example, "common_segment_state"] = "reference_example_year_unverified"
    result["common_segment_source"] = TABLE_URL
    result["hierarchy_path"] = result.epa_group.replace("", "Unresolved EPA class") + " → " + result.common_segment.replace("", "Unresolved common segment")
    return result


def build(run, source_dir, output, reuse_layer=None):
    run, source_dir, output = Path(run), Path(source_dir), Path(output)
    receipt = taxonomy.verify_artifacts(run)
    if receipt.get("status") != "completed" or receipt.get("reduced_budget", True):
        raise ValueError("A completed full discovery run is required")
    taxonomy.verify_artifacts(source_dir)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "vehicle_category_hierarchy.py")
    epa = pd.read_parquet(source_dir / "epa_vehicles.parquet")
    index = sources.epa_index(epa)
    if reuse_layer:
        reused = taxonomy.verify_artifacts(Path(reuse_layer))
        if reused.get("base_manifest_sha256") != r.file_hash(run / "manifest.json") or reused.get("source_manifest_sha256") != r.file_hash(source_dir / "manifest.json"):
            raise ValueError("Reusable EPA parent labels belong to different inputs")
    for grain in ("cohort", "model"):
        path = output / grain
        path.mkdir()
        taxonomy.verify_artifacts(run / grain)
        data = pd.read_parquet(run / grain / "variant_profiles.parquet")
        labels = pd.read_parquet(run / grain / "profile_labels.parquet")
        additions = []
        for row in ([] if reuse_layer else data[data.identity_complete].to_dict("records")):
            candidates = sources.epa_candidates(row, epa, index)
            parents = candidates.VClass.dropna().map(epa_group).unique()
            if len(parents) == 1 and candidates.VClass.notna().all():
                additions.append({"profile_id": row["profile_id"], "dimension": "epa_group", "label": parents[0],
                    "status": "source_supported", "namespace": "epa_size_parent", "source_url": f"https://www.fueleconomy.gov/feg/byclass/{int(row['ModelYear'])}ClassList.shtml",
                    "evidence": "Compatible EPA variants agree after drive-suffix removal; IDs: " + ";".join(candidates.id.astype(str)),
                    "scope": "Matched year/identity/specifications; drivetrain suffix collapsed only", "mapping_id": "epa-size-parent-v1",
                    "probability": 1., "policy": "verified"})
        if reuse_layer:
            reused_labels = pd.read_parquet(Path(reuse_layer) / grain / "profile_labels.parquet")
            additions = reused_labels[reused_labels.dimension.eq("epa_group")].to_dict("records")
        labels = pd.concat([labels, pd.DataFrame(additions)], ignore_index=True)
        with threadpool_limits(limits=2):
            predictions, metrics = source_predictions(data, labels, path)
        labels.to_parquet(path / "profile_labels.parquet", index=False)
        predictions.to_parquet(path / "predictions.parquet", index=False)
        metrics.to_csv(path / "classifier_metrics.csv", index=False)
        hierarchy = hierarchy_rows(data, labels, predictions)
        hierarchy.to_parquet(path / "hierarchy.parquet", index=False)
        counts = hierarchy.merge(data[["profile_id", "vin_count"]], validate="1:1")
        counts.groupby(["epa_group", "epa_group_state", "common_segment", "common_segment_state"], dropna=False).agg(profiles=("profile_id", "size"), listings=("vin_count", "sum")).reset_index().to_csv(path / "hierarchy_coverage.csv", index=False)
        if len(hierarchy) != len(data) or hierarchy.profile_id.duplicated().any() or counts.vin_count.sum() != data.vin_count.sum():
            raise ValueError("Hierarchy membership/count reconciliation failed")
    r.write_json(output / "classification_policy.json", {"human_review_required": False, "peer_review_required": False,
        "training_basis": "source labels", "prediction_status": "experimental, not independently reviewed",
        "probability": "uncalibrated model probability", "threshold": THRESHOLD,
        "model_selection": "Development-family macro F1: logistic without make versus ExtraTrees with make; query fold excluded",
        "automated_label_support": "At least three development emitted-label families and >=80% source agreement; no query-family evidence used",
        "epa_primary_groups": "EPA VClass with drive suffixes collapsed", "common_segments": "Terminology crosswalk from supported/predicted market size and positioning; not EPA equivalence",
        "references": REFERENCES, "table_reference": TABLE_URL})
    r.finish_stage(output, "completed", schema_version="epa-hierarchy-1", base_run=str(run.resolve()),
        base_manifest_sha256=r.file_hash(run / "manifest.json"), source_directory=str(source_dir.resolve()),
        source_manifest_sha256=r.file_hash(source_dir / "manifest.json"), human_review_required=False)
    return output


def promote(layer, output):
    """Deployment check against outer held-out source families, without human review.

    These held-out results decide deployment; they are not an untouched estimate
    after promotion. No held-out labels fit or select the underlying estimators.
    """
    layer, output = Path(layer), Path(output)
    receipt = taxonomy.verify_artifacts(layer)
    shutil.copytree(layer, output, ignore=shutil.ignore_patterns("manifest.json"))
    shutil.copyfile(__file__, output / "vehicle_category_hierarchy.py")
    for grain in ("cohort", "model"):
        path = output / grain
        metrics = pd.read_csv(path / "classifier_metrics.csv")
        evaluated = metrics[metrics.enabled.fillna(False).astype(bool)].copy()
        evaluated["correct"] = evaluated.heldout_emitted_families * evaluated.source_agreement
        summary = evaluated.groupby(["dimension", "label"])[["correct", "heldout_emitted_families"]].sum().reset_index()
        summary["source_agreement"] = summary.correct / summary.heldout_emitted_families.replace(0, np.nan)
        summary["deployment_enabled"] = summary.heldout_emitted_families.ge(10) & summary.source_agreement.ge(.8)
        summary.to_csv(path / "deployment_support.csv", index=False)
        allowed = set(summary.loc[summary.deployment_enabled, ["dimension", "label"]].itertuples(index=False, name=None))
        predictions = pd.read_parquet(path / "predictions.parquet")
        predictions = predictions.loc[[key in allowed for key in predictions[["dimension", "label"]].itertuples(index=False, name=None)]].copy()
        predictions["model_uri"] = predictions.model_uri.map(lambda value: str((path / Path(value).name).resolve()))
        predictions.to_parquet(path / "predictions.parquet", index=False)
        data = pd.read_parquet(Path(receipt["base_run"]) / grain / "variant_profiles.parquet")
        labels = pd.read_parquet(path / "profile_labels.parquet")
        hierarchy = hierarchy_rows(data, labels, predictions)
        hierarchy.to_parquet(path / "hierarchy.parquet", index=False)
        counts = hierarchy.merge(data[["profile_id", "vin_count"]], validate="1:1")
        counts.groupby(["epa_group", "epa_group_state", "common_segment", "common_segment_state"], dropna=False).agg(
            profiles=("profile_id", "size"), listings=("vin_count", "sum")).reset_index().to_csv(path / "hierarchy_coverage.csv", index=False)
        if hierarchy.profile_id.duplicated().any() or len(hierarchy) != len(data) or counts.vin_count.sum() != data.vin_count.sum():
            raise ValueError("Deployment population reconciliation failed")
        r.write_json(path / "classifier_status.json", {"status": "active_source_trained", "human_review_required": False,
            "active_predictions": len(predictions), "active_labels": len(allowed), "source_agreement_is_not_human_accuracy": True})
    policy = r.read_json(output / "classification_policy.json")
    policy["deployment_support"] = "At least ten outer held-out emitted-label families and >=80% source agreement across development-enabled folds; held-out results used for deployment decisions"
    r.write_json(output / "classification_policy.json", policy)
    r.finish_stage(output, "completed", schema_version="epa-hierarchy-1", base_run=receipt["base_run"],
        base_manifest_sha256=receipt["base_manifest_sha256"], source_directory=receipt["source_directory"],
        source_manifest_sha256=receipt["source_manifest_sha256"], human_review_required=False, deployment_checked=True,
        candidate_layer=str(layer.resolve()), candidate_manifest_sha256=r.file_hash(layer / "manifest.json"))
    return output


def activate(output, registry):
    output, registry = Path(output), Path(registry)
    receipt = taxonomy.verify_artifacts(output)
    if receipt.get("status") != "completed" or receipt.get("schema_version") != "epa-hierarchy-1":
        raise ValueError("Completed hierarchy required")
    if not receipt.get("deployment_checked"):
        output = promote(output, output.with_name(output.name + "-validated"))
        receipt = taxonomy.verify_artifacts(output)
    original = registry.read_bytes()
    config = json.loads(original.decode("utf-8-sig"))
    if config["similarity_manifest_sha256"] != receipt["base_manifest_sha256"]:
        raise ValueError("Hierarchy does not belong to active similarity run")
    backup = output.parent / ("hierarchy-activation-" + output.name)
    backup.mkdir(exist_ok=False)
    (backup / "previous_registry.json").write_bytes(original)
    config.update(category_hierarchy_run=str(output.resolve().relative_to(r.ROOT / "MODEL_OUTPUT")).replace("\\", "/"),
        category_hierarchy_manifest_sha256=r.file_hash(output / "manifest.json"))
    if registry.read_bytes() != original:
        raise ValueError("Registry changed concurrently")
    saved.atomic_json(registry, config)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--reuse-layer", type=Path)
    args = parser.parse_args()
    result = build(args.run, args.sources, args.output, args.reuse_layer)
    if args.activate:
        result = activate(result, r.ROOT / "car_app/artifact_registry.json")
    print(result)
