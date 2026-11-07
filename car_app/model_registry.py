"""Inventory, compare and explicitly activate completed local price artifacts.

Run ``python -m car_app.model_registry --help`` after a training suite finishes.
Training directories are read-only; reports always use a fresh directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from car_app.data import ROOT
from car_app.artifacts import hash_file, stat_key

CONFIG = ROOT / "car_app/artifact_registry.json"
VARIANTS = ("base", "enhanced", "enhanced_safety", "full_youtube", "full_youtube_sentiment",
            "enhanced_safety_expert", "enhanced_safety_expert_coverage",
            "full_youtube_expert_coverage", "full")
COMMON_SOURCES = ("used_cars.parquet", "NHTSA_vPIC.parquet", "NHTSA_Complaints.parquet",
                  "NHTSA_Recalls.parquet", "NHTSA_Safety.parquet")
COMPARISON_KEYS = ("split_fingerprint", "split_strategy", "preprocessing_version", "code_fingerprint",
                   "package_versions", "tuning_rows", "max_trees", "price_weighting")
INFERENCE_VERIFICATION_VERSION = 1


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hash_file(str(path), stat_key(path))


def inference_code_fingerprint():
    from MODELS.Price_ML_Models import PRICE_CODE_FILES
    code = {name: digest(ROOT / name) for name in PRICE_CODE_FILES}
    return hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()


@st.cache_data(max_entries=8, show_spinner=False)
def prediction_inputs(vins, variant, expert_directory, youtube_directory, source_signature, code_fingerprint):
    """Prepare actual VIN rows once for estimators sharing a source/feature contract."""
    from MODELS.Price_ML_Models import prepare_prediction_frame
    requested = pl.DataFrame({"vin": list(vins)})
    raw = pl.scan_parquet(ROOT / "DATA/CAR_DATA_FINAL/used_cars.parquet").filter(pl.col("vin").is_in(vins)).collect()
    raw = requested.join(raw, on="vin", how="left", validate="1:1", maintain_order="left")
    frame = prepare_prediction_frame(variant, raw,
        expert_feature_dir=Path(expert_directory) if expert_directory else None,
        youtube_feature_dir=Path(youtube_directory) if youtube_directory else None)
    if frame.height != len(vins) or frame["vin"].to_list() != list(vins):
        raise ValueError("Prediction rows lost their listing identity/order")
    return frame


def verify_inference_replay(path, metadata, model, code_fingerprint):
    """Accept changed source files only after reproducing frozen inputs and USD predictions."""
    import joblib
    path = Path(path)
    required = [metadata.get(key) for key in ("shap_inputs", "shap_receipt", "shap_background")]
    frozen_hashes = metadata.get("shap_input_sha256", {})
    if any(not name or name not in frozen_hashes for name in required):
        raise ValueError("Changed inference code requires verified frozen reference inputs")
    if any(digest(path / name) != frozen_hashes[name] for name in required):
        raise ValueError("Frozen inference reference changed")
    predictor_path = path / model / "predictor.joblib"
    prediction_path = path / f"{model}_test_predictions.parquet"
    contract = {key: metadata.get(key) for key in ("model_type", "raw_feature_columns", "feature_columns",
        "preprocessing_version", "source_file_sha256", "package_versions", "shap_input_sha256", "split_fingerprint")}
    signature = {"version": INFERENCE_VERIFICATION_VERSION, "run": str(path.resolve()), "model": model,
        "current_code_fingerprint": code_fingerprint, "training_code_fingerprint": metadata.get("code_fingerprint"),
        "contract": contract, "predictor_sha256": digest(predictor_path), "test_predictions_sha256": digest(prediction_path)}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    receipt_path = ROOT / "MODEL_OUTPUT/diagnostics/inference_compatibility" / (key + ".json")
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("signature") == signature and receipt.get("status") == "passed":
            return receipt
    reference = pd.read_parquet(path / metadata["shap_receipt"])
    frozen = pd.read_parquet(path / metadata["shap_inputs"])
    if reference.empty or len(reference) != len(frozen) or reference.vin.isna().any() or reference.vin.duplicated().any():
        raise ValueError("Frozen reference requires aligned, unique VIN rows")
    sources = tuple((name, str(source), stat_key(source)) for name, source in price_sources(metadata).items())
    frame = prediction_inputs(tuple(reference.vin), metadata["model_type"],
        metadata.get("expert_features", {}).get("directory", ""),
        metadata.get("youtube_features", {}).get("directory", ""), sources, code_fingerprint)
    predictor = joblib.load(predictor_path)
    raw = frame.select(metadata["raw_feature_columns"]).to_pandas()
    prepared = predictor.named_steps["input"].transform(raw)
    try:
        pd.testing.assert_frame_equal(prepared.reset_index(drop=True), frozen.reset_index(drop=True),
            check_dtype=False, check_categorical=False, rtol=1e-12, atol=1e-12)
        saved = pl.scan_parquet(prediction_path).filter(pl.col("vin").is_in(reference.vin.tolist())).select("vin", "prediction").collect().to_pandas()
        if saved.vin.duplicated().any() or set(saved.vin) != set(reference.vin):
            raise ValueError("Saved predictions do not cover the unique frozen reference VINs")
        expected = saved.set_index("vin").loc[reference.vin, "prediction"].to_numpy(float)
        predicted = np.asarray(predictor.predict(raw), dtype=float)
        if not np.isfinite(predicted).all() or (predicted <= 0).any():
            raise ValueError("Frozen replay produced invalid USD predictions")
        np.testing.assert_allclose(predicted, expected, rtol=1e-7, atol=.01)
    except AssertionError as exc:
        raise ValueError("Current inference does not reproduce the frozen training inputs/predictions") from exc
    if inference_code_fingerprint() != code_fingerprint or digest(predictor_path) != signature["predictor_sha256"]:
        raise ValueError("Inference inputs changed during verification")
    receipt = {"status": "passed", "signature": signature, "rows": len(reference),
               "maximum_absolute_usd_difference": float(np.max(np.abs(predicted - expected))),
               "comparison": "all frozen test inputs and VIN-aligned saved USD predictions; rtol=1e-7, atol=$0.01"}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix("." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    temporary.replace(receipt_path)
    return receipt


def compatibility_group(metadata):
    if any(key not in metadata for key in COMPARISON_KEYS):
        raise ValueError("Missing comparison provenance")
    sources = metadata.get("source_file_sha256", {})
    if any(name not in sources for name in COMMON_SOURCES):
        raise ValueError("Missing common source hashes")
    contract = {key: metadata[key] for key in COMPARISON_KEYS}
    contract["sources"] = {name: sources[name] for name in COMMON_SOURCES}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:16]


def latest_inventory(base=ROOT / "MODEL_OUTPUT"):
    """Account for every variant, including its newer unfinished run and pilot fallback."""
    rows, candidates = [], []
    for variant in VARIANTS:
        runs = []
        for file in sorted((Path(base) / variant).glob("*/run_metadata.json"), reverse=True):
            try:
                metadata = read_json(file)
                runs.append((file.parent, metadata))
            except (OSError, ValueError):
                rows.append({"variant": variant, "run": file.parent.name, "status": "unreadable"})
        latest = runs[0] if runs else None
        full = next(((p, m) for p, m in runs if m.get("status") == "completed"
                     and p.name.endswith("-full") and m.get("requested_sample_size") is None), None)
        row = {"variant": variant, "latest_run": latest[0].name if latest else None,
               "latest_status": latest[1].get("status", "legacy") if latest else "missing",
               "completed_full_run": full[0].name if full else None,
               "status": "completed_full" if full else "awaiting_completed_full"}
        if full:
            path, metadata = full
            try:
                from MODELS import Price_ML_Models as price
                if (variant in price.YOUTUBE_MODEL_TYPES
                        and metadata.get("raw_feature_columns") != list(price.MODEL_FEATURE_SETS[variant])):
                    raise ValueError("Retired YouTube feature definition; retrain with sentiment v2")
                group = compatibility_group(metadata)
                metrics = pd.read_csv(path / "validation_metrics.csv")
                test = pd.read_csv(path / "metrics.csv") if (path / "metrics.csv").is_file() else pd.DataFrame()
                for model, receipt in metadata.get("models", {}).items():
                    predictor = path / model / "predictor.joblib"
                    validation = metrics[metrics.model.eq(model)]
                    if receipt.get("status") != "completed" or not predictor.is_file() or validation.empty:
                        continue
                    mae = float(validation.iloc[0].validation_mae)
                    if not np.isfinite(mae):
                        continue
                    test_row = test[test.model.eq(model)] if not test.empty else pd.DataFrame()
                    candidates.append({"variant": variant, "run": path.relative_to(base).as_posix(), "model": model,
                        "group": group, "validation_mae": mae,
                        "test_mae": None if test_row.empty else test_row.iloc[0].get("mae"),
                        "test_rmse": None if test_row.empty else test_row.iloc[0].get("rmse"),
                        "elapsed_seconds": receipt.get("elapsed_seconds"),
                        "feature_count": len(metadata.get("raw_feature_columns", [])),
                        "source_hashes": json.dumps(metadata["source_file_sha256"], sort_keys=True)})
                row["group"] = group
                row["completed_estimators"] = sum(c["run"] == path.relative_to(base).as_posix() for c in candidates)
                row["unavailable_estimators"] = ", ".join(model for model in metadata.get("requested_models", [])
                    if metadata.get("models", {}).get(model, {}).get("status") != "completed"
                    or not (path / model / "predictor.joblib").is_file())
            except (ValueError, OSError, KeyError, pd.errors.ParserError) as exc:
                row.update(status="comparison_unavailable", reason=str(exc))
        if latest and not full and "pilot" in latest[0].name:
            row["status"] = "pilot_only"
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(candidates)


def validate_price_run(path, model=None):
    """Validate all inputs before loading executable estimator state."""
    from MODELS import Price_ML_Models as price
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "MODEL_OUTPUT").resolve()):
        raise ValueError("Predictor must be a project model artifact")
    metadata = read_json(path / "run_metadata.json")
    if (not path.name.endswith("-full") or metadata.get("requested_sample_size") is not None
            or (metadata.get("status") != "completed"
                and metadata.get("models", {}).get(model, {}).get("status") != "completed")):
        raise ValueError("Price run is not a completed full-budget run")
    variant = metadata["model_type"]
    if metadata.get("preprocessing_version") != price.PREPROCESSING_VERSION:
        raise ValueError("Preprocessing version differs")
    if list(metadata.get("raw_feature_columns", [])) != list(price.MODEL_FEATURE_SETS[variant]):
        raise ValueError("Raw feature definition/order differs")
    fingerprint = inference_code_fingerprint()
    for package, version in metadata.get("package_versions", {}).items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Package differs: {package}")
    sources = metadata["source_file_sha256"]
    for name, source in price_sources(metadata).items():
        if digest(source) != sources[name]:
            raise ValueError(f"Source changed: {name}")
    if hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest() != metadata.get("dataset_fingerprint"):
        raise ValueError("Dataset fingerprint differs")
    if fingerprint != metadata.get("code_fingerprint"):
        if model is None or metadata.get("models", {}).get(model, {}).get("status") != "completed":
            raise ValueError("Changed inference code requires a completed estimator for frozen replay")
        verify_inference_replay(path, metadata, model, fingerprint)
    return metadata


def price_sources(metadata):
    paths = {}
    for name in metadata["source_file_sha256"]:
        if name.startswith("expert/"):
            directory = metadata.get("expert_features", {}).get("directory")
            if not directory:
                raise ValueError("Expert source directory missing from receipt")
            source = Path(directory) / Path(name).name
        elif name in COMMON_SOURCES:
            source = ROOT / "DATA/CAR_DATA_FINAL" / name
        else:
            from NHTSA.text_features import load_youtube_features

            directory = metadata.get("youtube_features", {}).get("directory")
            if not directory:
                raise ValueError("YouTube source directory missing from receipt; retrain with sentiment v2")
            load_youtube_features(directory)
            source = Path(directory) / name
        paths[name] = source
    return paths


def registry():
    config = read_json(CONFIG)
    if config.get("price_run_selection") is not None:
        from car_app.artifacts import diagnostic_inventory

        _, candidates, _, _ = diagnostic_inventory(config["price_run_selection"])
        config["active_price"] = None
        if not candidates.empty:
            choice = candidates.sort_values(["validation_mae", "feature_set", "model"]).iloc[0]
            path = ROOT / "MODEL_OUTPUT" / choice.run
            config["active_price"] = {"run": choice.run, "model": choice.model,
                "metadata_sha256": digest(path / "run_metadata.json"),
                "predictor_sha256": digest(path / choice.model / "predictor.joblib"),
                "selection_metric": "validation_mae_usd"}
    return config


@st.cache_resource(max_entries=2, show_spinner=False)
def load_predictor(run, model, metadata_hash, predictor_hash, source_signature=(), code_fingerprint=""):
    import joblib
    path = ROOT / "MODEL_OUTPUT" / run
    if digest(path / "run_metadata.json") != metadata_hash:
        raise ValueError("Pinned model metadata changed")
    metadata = validate_price_run(path, model)
    if metadata.get("models", {}).get(model, {}).get("status") != "completed":
        raise ValueError("Estimator is not completed")
    predictor_path = path / model / "predictor.joblib"
    if digest(predictor_path) != predictor_hash:
        raise ValueError("Pinned predictor changed")
    predictor = joblib.load(predictor_path)
    saved = getattr(predictor, "raw_feature_columns_", None)
    if saved is None and hasattr(predictor, "named_steps") and "input" in predictor.named_steps:
        transform = predictor.named_steps["input"]
        saved = [*transform.categorical_columns, *transform.numeric_columns]
    if saved is None or set(saved) != set(metadata["raw_feature_columns"]):
        raise ValueError("Predictor feature contract differs")
    return predictor, metadata


def price_estimates(listings: pl.DataFrame, config=None):
    """Supplement returned listings; never change their eligibility or order."""
    config = registry() if config is None else config
    active = config.get("active_price")
    if not active or listings.is_empty():
        return listings, "No price model is activated."
    try:
        from MODELS.Price_ML_Models import MODEL_FEATURE_SETS
        receipt = read_json(ROOT / "MODEL_OUTPUT" / active["run"] / "run_metadata.json")
        source_signature = tuple((name, stat_key(path)) for name, path in price_sources(receipt).items())
        code_fingerprint = inference_code_fingerprint()
        predictor, metadata = load_predictor(active["run"], active["model"], active["metadata_sha256"], active["predictor_sha256"], source_signature, code_fingerprint)
        frame = prediction_inputs(tuple(listings["vin"].to_list()), metadata["model_type"],
            metadata.get("expert_features", {}).get("directory", ""),
            metadata.get("youtube_features", {}).get("directory", ""), source_signature, code_fingerprint)
        values = np.asarray(predictor.predict(frame.select(MODEL_FEATURE_SETS[metadata["model_type"]]).to_pandas()))
        if len(values) != listings.height or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Invalid USD predictions")
        return listings.with_columns(pl.Series("predicted_price_usd", values),
            pl.lit(metadata["model_type"] + "-" + active["model"]).alias("price_model")), None
    except Exception as exc:
        return listings, f"Price estimate unavailable: {type(exc).__name__}: {exc}"


def build_report(output, base=ROOT / "MODEL_OUTPUT", explain_rows=0):
    """Reuse strict comparison routines within matched groups; never train."""
    from MODELS.Price_ML_Models import compare_saved_runs
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inventory, candidates = latest_inventory(base)
    inventory.to_csv(output / "inventory.csv", index=False)
    comparisons, paired_rows, explanations, coverage = [], [], [], []
    if not candidates.empty:
        candidates = candidates.sort_values(["group", "validation_mae", "run", "model"])
        candidates.to_csv(output / "candidates.csv", index=False)
        for name in candidates.run.unique():
            metadata = read_json(Path(base) / name / "run_metadata.json")
            for field in ("rows_after_filters", "listing_rows_without_vpic_match", "complaint_recall_coverage",
                          "youtube_join_coverage", "expert_listing_coverage"):
                if field in metadata:
                    coverage.append({"run": name, "field": field, "value": json.dumps(metadata[field])})
            youtube = metadata.get("youtube_features", {})
            for field in ("supported_topic_makes", "directional_sentiment_available", "sentiment_limitation", "as_of"):
                if field in youtube:
                    coverage.append({"run": name, "field": "youtube_" + field, "value": json.dumps(youtube[field])})
        pd.DataFrame(coverage).to_csv(output / "source_coverage.csv", index=False)
        for group, part in candidates.groupby("group"):
            paths = [Path(base) / name for name in part.run.unique()]
            try:
                report = compare_saved_runs(paths)
                comparisons.append({"group": group, "status": "completed", "directory": str(report.resolve()),
                                    "artifacts": {p.name: digest(p) for p in report.iterdir() if p.is_file()}})
                from MODELS.Price_ML_Models import paired_expert_validation
                pairs = [("base", "enhanced"), ("enhanced", "enhanced_safety"), ("enhanced_safety", "full_youtube")]
                variants = {read_json(p / "run_metadata.json")["model_type"]: p for p in paths}
                for left, right in pairs:
                    if left not in variants or right not in variants:
                        continue
                    for lp in variants[left].glob("*_validation_predictions.parquet"):
                        rp = variants[right] / lp.name
                        if not rp.is_file():
                            continue
                        a, b = pd.read_parquet(lp), pd.read_parquet(rp)
                        if "doug_source_level" not in b:
                            b["doug_source_level"] = "missing"
                        overall = paired_expert_validation(a, b)[0]
                        paired_rows.append({"group": group, "left_variant": left, "right_variant": right,
                            "model": lp.name.removesuffix("_validation_predictions.parquet"), **overall})
                if explain_rows:
                    best = part.sort_values("validation_mae").iloc[0]
                    source = Path(base) / best.run
                    destination = output / "explanations" / (best.variant + "-" + source.name)
                    destination.mkdir(parents=True)
                    metadata = read_json(source / "run_metadata.json")
                    for filename in ["run_metadata.json", metadata["shap_inputs"], metadata["shap_background"], metadata["shap_receipt"]]:
                        shutil.copyfile(source / filename, destination / filename)
                    (destination / best.model).mkdir()
                    shutil.copyfile(source / best.model / "predictor.joblib", destination / best.model / "predictor.joblib")
                    from MODELS.Price_ML_Models import explain_saved_run
                    explained = explain_saved_run(destination, explain_rows, [best.model])
                    lineage = {"source_run": str(source.resolve()), "metadata_sha256": digest(source / "run_metadata.json"),
                               "predictor_sha256": digest(source / best.model / "predictor.joblib"), "results": explained}
                    (destination / "lineage.json").write_text(json.dumps(lineage, indent=2))
                    explanations.append({"group": group, "model": best.model, "directory": str(destination.resolve()), **explained[best.model]})
            except (ValueError, OSError, KeyError) as exc:
                comparisons.append({"group": group, "status": "unavailable", "reason": str(exc)})
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(output / "incremental_feature_comparison.csv", index=False)
    receipt = {"status": "completed", "comparisons": comparisons, "explanations": explanations,
               "awaiting_variants": inventory.loc[~inventory.status.eq("completed_full"), "variant"].tolist(),
               "selection_metric": "validation_mae_usd", "test_used_for_selection": False,
               "artifacts": {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*") if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2))
    return inventory, candidates, receipt


def verified_report(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT.resolve()):
        raise ValueError("Report must be inside the project")
    receipt = read_json(directory / "manifest.json")
    for name, expected in receipt.get("artifacts", {}).items():
        if digest(directory / name) != expected:
            raise ValueError(f"Comparison report changed: {name}")
    for comparison in receipt.get("comparisons", []):
        if comparison["status"] == "completed":
            for name, expected in comparison.get("artifacts", {}).items():
                if digest(Path(comparison["directory"]) / name) != expected:
                    raise ValueError(f"Saved comparison changed: {name}")
    return receipt


def activate(run, model, report_directory):
    verified_report(report_directory)
    path = ROOT / "MODEL_OUTPUT" / run
    metadata = validate_price_run(path)
    if metadata["models"].get(model, {}).get("status") != "completed":
        raise ValueError("Cannot activate incomplete estimator")
    config = registry() if CONFIG.is_file() else {
        "schema_version": 1, "similarity_run": "vehicle_similarity_v3/20261024-full"}
    config.setdefault("similarity_manifest_sha256", digest(ROOT / "MODEL_OUTPUT" / config["similarity_run"] / "manifest.json"))
    config["active_price"] = {"run": run, "model": model,
        "metadata_sha256": digest(path / "run_metadata.json"),
        "predictor_sha256": digest(path / model / "predictor.joblib"),
        "selection_metric": "validation_mae_usd"}
    config["comparison_report"] = str(Path(report_directory).resolve().relative_to(ROOT))
    config["comparison_manifest_sha256"] = digest(Path(report_directory) / "manifest.json")
    temporary = CONFIG.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2))
    temporary.replace(CONFIG)
    return config


def publish_report(directory):
    verified_report(directory)
    config = registry() if CONFIG.is_file() else {"schema_version": 1, "similarity_run": "vehicle_similarity_v3/20261024-full"}
    config.setdefault("similarity_manifest_sha256", digest(ROOT / "MODEL_OUTPUT" / config["similarity_run"] / "manifest.json"))
    config.update(comparison_report=str(Path(directory).resolve().relative_to(ROOT)),
                  comparison_manifest_sha256=digest(Path(directory) / "manifest.json"))
    temporary = CONFIG.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2))
    temporary.replace(CONFIG)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="Fresh comparison report directory")
    mode.add_argument("--report", type=Path, help="Existing verified report for explicit activation")
    parser.add_argument("--activate", help="Explicit variant/run/estimator to activate after checks")
    parser.add_argument("--explain-rows", type=int, default=0,
                        help="Explain each comparison-group validation winner on this many saved rows in report copies")
    parser.add_argument("--publish", action="store_true", help="Show this report in the app without changing its price model")
    args = parser.parse_args()
    report = args.output or args.report
    if args.output:
        _, candidates, receipt = build_report(report, explain_rows=args.explain_rows)
    else:
        receipt = verified_report(report)
        candidates = pd.read_csv(report / "candidates.csv")
        candidates["run"] = candidates.run.str.replace("\\", "/", regex=False)
    print(json.dumps(receipt, indent=2))
    if args.activate:
        run, model = args.activate.rsplit("/", 1)
        if candidates.empty or not ((candidates.run == run) & (candidates.model == model)).any():
            parser.error("Activation must name an eligible latest completed full-run estimator")
        activate(run, model, report)
        print("Activated:", args.activate)
    elif args.publish:
        publish_report(report)


if __name__ == "__main__":
    main()
