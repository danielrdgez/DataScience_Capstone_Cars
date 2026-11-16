"""Read saved evaluations without loading or retraining estimators."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from car_app.data import ROOT, COHORT, Filters, filter_expression, snapshot, EXCLUDED_LISTING_VINS

DEFAULT_BASE = "base/20261009T171630407257Z-full"
DEFAULT_ENRICHED = "enhanced/20261011T050517300002Z-full"
DEFAULT_PCR = "full/20261024T154353886346Z-full"


@dataclass
class Run:
    path: Path
    metadata: dict
    legacy: bool = False

    @property
    def name(self):
        return self.path.relative_to(ROOT / "MODEL_OUTPUT").as_posix()

    def table(self, name):
        if name not in {"metrics.csv", "validation_metrics.csv", "segment_metrics.csv",
                        "year_make_metrics.csv", "validation_segment_metrics.csv",
                        "validation_year_make_metrics.csv", "validation_make_price_metrics.csv", "validation_focus_make_metrics.csv",
                        "validation_feature_comparison.csv", "validation_weighting_comparison.csv",
                        "youtube_join_coverage.csv"}:
            raise ValueError("Unsupported report")
        file = self.path / name
        try:
            return pd.read_csv(file) if file.is_file() and file.stat().st_size > 2 else pd.DataFrame()
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # CSV checkpoints are rewritten before metadata is published.
            return pd.DataFrame()

    def prediction_path(self, model, partition="test"):
        if partition not in {"test", "validation"}:
            raise ValueError("Unsupported evaluation partition")
        if model not in set(self.table("metrics.csv").get("model", [])):
            return None
        file = self.path / f"{model}_{partition}_predictions.parquet"
        return file if file.is_file() else None


@st.cache_data(max_entries=4, ttl=900, show_spinner=False)
def inventory(refresh=0):
    runs, ignored = [], []
    for file in sorted((ROOT / "MODEL_OUTPUT").rglob("run_metadata.json")):
        try:
            meta = json.loads(file.read_text(encoding="utf-8-sig"))
            status = meta.get("status")
            if status not in (None, "completed", "validation_complete", "test_complete"):
                ignored.append((str(file.parent.relative_to(ROOT / "MODEL_OUTPUT")), status))
                continue
            legacy = status is None or (meta.get("model_type") is not None
                                         and meta.get("preprocessing_version") != "advertised-price-log-v4")
            run = Run(file.parent, meta, legacy)
            if not run.table("metrics.csv").empty or not run.table("validation_metrics.csv").empty:
                runs.append(run)
        except (ValueError, OSError, pd.errors.ParserError):
            ignored.append((str(file.parent.relative_to(ROOT / "MODEL_OUTPUT")), "unreadable"))
    return runs, ignored


def errors(frame: pd.DataFrame):
    if frame.empty:
        return {"rows": 0, "family_blocks": 0, "support": "no_rows",
                **{k: None for k in ["mae", "rmse", "r2", "wape_pct", "mean_signed_error", "p95_absolute_error"]}}
    actual = frame.actual_price.to_numpy(dtype=float)
    predicted = frame.prediction.to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation contains non-finite prices")
    delta = predicted - actual
    families = frame["family_block"].nunique(dropna=False) if "family_block" in frame else None
    return {
        "rows": len(frame), "family_blocks": families,
        "support": "supported" if len(frame) >= 100 and families is not None and families >= 5 else "low_support",
        "mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "r2": float(1 - np.sum(delta ** 2) / np.sum((actual - actual.mean()) ** 2))
              if len(actual) > 1 and np.ptp(actual) > 0 else None,
        "wape_pct": float(np.abs(delta).sum() / actual.sum() * 100) if actual.sum() > 0 else None,
        "mean_signed_error": float(delta.mean()),
        "p95_absolute_error": float(np.quantile(np.abs(delta), .95)),
    }


@st.cache_data(max_entries=4, ttl=900, show_spinner=False)
def selected_vins(source_identity: str, filters: Filters):
    source = snapshot()
    if source.identity != source_identity:
        raise ValueError("Snapshot changed; refresh the app")
    return source.catalog(filters).select("vin").collect()


@st.cache_data(max_entries=24, ttl=900, show_spinner=False)
def evaluation(file: str, signature: tuple, source_identity: str, filters: Filters):
    predictions = pl.scan_parquet(file).filter(~pl.col("vin").is_in(sorted(EXCLUDED_LISTING_VINS)))
    if filters.active():
        # Filter historical attributes directly, including the original target.
        # Only filters absent from the receipt need the current source snapshot.
        columns = {"makes": "Make", "models": "Model", "years": "ModelYear",
                   "trims": "Trim", "body_types": "body_type", "fuels": "fuel_type",
                   "drivetrains": "wheel_system", "condition": "is_new",
                   "min_price": "actual_price", "max_price": "actual_price", "max_mileage": "mileage"}
        schema = predictions.collect_schema()
        saved = {key: value for key, value in filters.active().items() if columns[key] in schema}
        current = {key: value for key, value in filters.active().items() if columns[key] not in schema}
        if saved:
            predictions = predictions.with_columns(pl.col("actual_price").alias("price")).filter(
                filter_expression(Filters(**saved))).drop("price")
        if current:
            run_path = Path(file).parent
            meta = json.loads((run_path / "run_metadata.json").read_text())
            validate_sources(meta)
            selected = selected_vins(source_identity, Filters(**current))
            predictions = predictions.join(selected.lazy(), on="vin", how="semi")
    frame = predictions.collect().to_pandas()
    if frame.vin.duplicated().any():
        raise ValueError("Duplicate evaluation VINs")
    return frame


def evaluation_with_fallback(file, signature, source_identity, filters, minimum_rows=100):
    """Broaden display scope explicitly; never alter saved predictions or sidebar state."""
    scopes = [("Selected filters", filters)]
    if filters.models:
        scopes.append(("Make and model", Filters(makes=filters.makes, models=filters.models)))
    if filters.makes:
        scopes.append(("Make", Filters(makes=filters.makes)))
    seen = set()
    first_rows = None
    used_label = "Selected filters"
    for label, scope in scopes:
        if scope in seen:
            continue
        seen.add(scope)
        used_label = label
        frame = evaluation(file, signature, source_identity, scope)
        if first_rows is None:
            first_rows = len(frame)
        if len(frame) >= minimum_rows:
            return frame, label, first_rows
    return frame, used_label, first_rows


def paired_comparison(left: Run, right: Run, filters, source_identity):
    from MODELS.Price_ML_Models import compatible_run_metadata
    if not compatible_run_metadata(left.metadata, right.metadata, allow_feature_set=True):
        raise ValueError("Run fingerprints, split strategies, or training settings differ")
    models = sorted(set(left.table("metrics.csv").model) & set(right.table("metrics.csv").model))
    rows = []
    for model in models:
        lp, rp = left.prediction_path(model), right.prediction_path(model)
        if lp is None or rp is None:
            continue
        a = evaluation(str(lp), stat_key(lp), source_identity, Filters())
        b = evaluation(str(rp), stat_key(rp), source_identity, Filters())
        paired = a.merge(b, on="vin", validate="one_to_one", suffixes=("_a", "_b"))
        if len(paired) != len(a) or len(a) != len(b):
            raise ValueError("Evaluation populations differ")
        for column in ["actual_price", *COHORT, "split_group"]:
            aa, bb = paired[f"{column}_a"], paired[f"{column}_b"]
            if not (aa.eq(bb) | (aa.isna() & bb.isna())).all():
                raise ValueError(f"Evaluation {column} values differ")
        af = evaluation(str(lp), stat_key(lp), source_identity, filters)
        bf = evaluation(str(rp), stat_key(rp), source_identity, filters)
        ea, eb = errors(af), errors(bf)
        rows.append({"model": model, "rows": ea["rows"], "base_mae": ea["mae"],
                     "enriched_mae": eb["mae"],
                     "enriched_minus_base_mae": eb["mae"] - ea["mae"] if ea["rows"] else None})
    return pd.DataFrame(rows)


def stat_key(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


@st.cache_data(max_entries=128, ttl=900, show_spinner=False)
def hash_file(path: str, signature: tuple):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(metadata):
    from car_app.model_registry import COMMON_SOURCES, price_sources

    hashes = metadata.get("source_file_sha256")
    if not hashes:
        raise ValueError("No snapshot hashes recorded; filtered evaluation/inference unavailable")
    for name in hashes:
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts or (len(parts) > 1 and
                not (len(parts) == 2 and parts[0] == "expert")):
            raise ValueError("Invalid source reference in artifact metadata")
    source = snapshot()
    paths = price_sources(metadata)
    for name, expected in hashes.items():
        path = source.directory / name if name in COMMON_SOURCES else paths[name]
        if not path.is_file():
            raise ValueError("Invalid source reference in artifact metadata")
        if hash_file(str(path), stat_key(path)) != expected:
            raise ValueError(f"Source fingerprint changed: {name}")
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if fingerprint != metadata.get("dataset_fingerprint"):
        raise ValueError("Dataset fingerprint does not match recorded source hashes")


def saved_shap(run, model):
    """Read exact-model notebook caches, including forest impurity and coefficients."""
    if model not in set(run.table("metrics.csv").get("model", [])):
        return pd.DataFrame()
    cache_root = ROOT / "MODEL_OUTPUT/diagnostics/model_evaluations/explanations"
    for receipt in sorted(cache_root.glob("*/status.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            status = json.loads(receipt.read_text(encoding="utf-8"))
            if (status.get("status") != "completed" or status.get("model") != model
                    or Path(status.get("run", "")).resolve() != run.path.resolve()
                    or status.get("frozen_hashes") != run.metadata.get("shap_input_sha256")):
                continue
            predictor = run.path / model / "predictor.joblib"
            if hash_file(str(predictor), stat_key(predictor)) != status.get("predictor_sha256"):
                continue
            result = pd.read_csv(receipt.parent / "importance.csv")
            result.attrs.update(method=status["method"], scale=status["scale"], sample_rows=status.get("sample_rows"))
            return result
        except (OSError, ValueError, KeyError):
            continue
    files = list(run.path.glob(f"{model}/{model}_shap_global.csv"))
    result = pd.read_csv(files[0]) if files else pd.DataFrame()
    result.attrs.update(method="Saved SHAP importance", scale=run.metadata.get("shap_output_scale", "unrecorded"))
    return result


def refresh_diagnostic_explanations(rows=100):
    """Build missing exact-checkpoint caches offline from frozen inputs, without fitting."""
    from MODELS.price_model_diagnostics import explain_checkpoint

    if rows < 1:
        raise ValueError("Explanation rows must be positive")
    available, choices, _, _ = diagnostic_inventory()
    results = []
    for choice in choices.itertuples():
        run = available[choice.run]
        if saved_shap(run, choice.model).empty:
            print(f"Explaining {choice.feature_set} / {choice.model}", flush=True)
            explain_checkpoint(run.path, run.metadata, choice.model,
                               ROOT / "MODEL_OUTPUT/diagnostics/model_evaluations/explanations", rows=rows)
        results.append({"feature_set": choice.feature_set, "model": choice.model,
                        "features": len(saved_shap(run, choice.model))})
    return pd.DataFrame(results)


def continuation_run(base, reference, policy, model=None):
    """Find only the continuation command's matching full-data training run."""
    original = json.loads((base / reference / "run_metadata.json").read_text(encoding="utf-8-sig"))
    keys = ("raw_feature_columns", "source_file_sha256", "split_fingerprint",
            "preprocessing_version", "package_versions", "tuning_rows", "max_trees",
            "price_weighting", "split_strategy", "device_request", "full_validation")
    for file in sorted((base / "full").glob("*-full/run_metadata.json"), reverse=True):
        if file.parent.name <= policy["after"]:
            continue
        try:
            metadata = json.loads(file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        requested = set(metadata.get("requested_models", []))
        expected_device = "cpu" if model == "lightgbm" else original.get("device_request")
        matching_keys = [key for key in keys if model is None or key != "device_request"]
        command_matches = (requested == set(policy["models"]) if model is None else
                           model in requested and requested <= set(policy["models"]))
        if (metadata.get("model_type") == "full"
                and metadata.get("requested_sample_size") is None
                and command_matches
                and (model is None or metadata.get("device_request") == expected_device)
                and (model != "lightgbm" or not policy.get("lightgbm_serialization")
                     or metadata.get("lightgbm_serialization") == policy["lightgbm_serialization"])
                and all(key in original and metadata.get(key) == original[key] for key in matching_keys)):
            name = file.parent.relative_to(base).as_posix()
            if model is not None:
                # A newer unfinished retry must not hide an existing checkpoint.
                _, choices, _, _ = diagnostic_inventory({"runs": {"full": name}, "models": [model]})
                if choices.empty:
                    continue
            return name
    return None


def youtube_price_contract(root=ROOT):
    """Bind the YouTube view to the configured Full runs, including pending fits."""
    from car_app.data import youtube_feature_source

    root = Path(root)
    config = json.loads((root / "car_app/artifact_registry.json").read_text(encoding="utf-8-sig"))
    selection = config.get("price_run_selection") or {}
    reference = selection.get("runs", {}).get("full")
    if not reference:
        raise ValueError("No Full pricing run is selected")
    base = root / "MODEL_OUTPUT"
    path = (base / reference).resolve()
    if path.parent != (base / "full").resolve() or not path.name.endswith("-full"):
        raise ValueError("Invalid selected Full pricing run")
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("model_type") != "full":
        raise ValueError("Selected pricing receipt is not a Full model")
    columns = [c for c in metadata["raw_feature_columns"] if c.startswith("yt_")]
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Full pricing receipt has missing or duplicate YouTube inputs")
    runs = [reference]
    policy = selection.get("full_continuation")
    continuation = continuation_run(base, reference, policy) if policy else None
    if continuation:
        runs.append(continuation)
    receipt = metadata.get("youtube_features") or {}
    if not receipt.get("directory"):
        raise ValueError("Full pricing receipt has no YouTube source directory")
    directory = Path(receipt["directory"])
    if not directory.is_absolute():
        directory = root / directory
    _, source, signature = youtube_feature_source(directory)
    for key in ("manifest_sha256", "features_sha256", "as_of"):
        if not receipt.get(key) or receipt[key] != source.get(key):
            raise ValueError("YouTube source differs from the Full pricing receipt: " + key)
    return {"directory": str(directory.resolve()), "columns": columns, "runs": runs,
            "as_of": source["as_of"], "source_signature": signature}


def diagnostic_inventory(selection=None):
    """Read completed estimators from pinned runs, or discover latest runs when unconfigured."""
    from car_app.model_registry import VARIANTS
    config = ROOT / "car_app/artifact_registry.json"
    if selection is None and config.is_file():
        selection = json.loads(config.read_text(encoding="utf-8-sig")).get("price_run_selection")
    selected, inspected, statuses = {}, [], []
    base = (ROOT / "MODEL_OUTPUT").resolve()
    if selection is not None:
        if not selection.get("runs") or not selection.get("models"):
            raise ValueError("Price run selection requires explicit runs and estimators")
        files = []
        for variant, name in selection["runs"].items():
            path = (base / name).resolve()
            if variant not in VARIANTS or path.parent != base / variant or not path.name.endswith("-full"):
                raise ValueError("Invalid selected price run: " + str(name))
            policy = selection.get("full_continuation") if variant == "full" else None
            if policy:
                files.append((variant, path / "run_metadata.json", ["catboost"]))
                groups = [[model] for model in policy["models"]] if policy.get("per_estimator") else [policy["models"]]
                for models in groups:
                    continuation = continuation_run(base, name, policy,
                                                    models[0] if policy.get("per_estimator") else None)
                    files.append((variant, base / continuation / "run_metadata.json" if continuation else None,
                                  models))
            else:
                files.append((variant, path / "run_metadata.json", None))
    else:
        files = [(variant, file, None) for variant in VARIANTS
                 for file in sorted((base / variant).glob("*-full/run_metadata.json"))]
    for variant, file, run_models in files:
        requested = run_models if run_models is not None else ([model for model in selection["models"] if model != "pcr" or variant == "full"] if selection else [])
        if file is None:
            statuses.extend({"feature_set": variant, "model": model, "run": "Awaiting matching Full continuation",
                             "status": "pending", "run_status": "awaiting run"} for model in requested)
            continue
        run_name = file.parent.relative_to(base).as_posix()
        pending = {model: {"feature_set": variant, "model": model, "run": run_name,
                          "status": "pending", "run_status": "awaiting checkpoint"} for model in requested}
        try:
            if selection is not None and not file.is_file():
                statuses.extend(pending.values())
                continue
            metadata = json.loads(file.read_text(encoding="utf-8-sig"))
            if metadata.get("requested_sample_size") is not None or metadata.get("model_type") != variant:
                raise ValueError("Selected run is not a full-data run for " + variant)
            run = Run(file.parent, metadata)
            scores = run.table("metrics.csv")
            validation = run.table("validation_metrics.csv")
            for row in pending.values():
                row.update(run_status=metadata.get("status"), feature_count=len(metadata.get("raw_feature_columns", [])))
                if metadata.get("status") == "failed":
                    row["status"] = "unavailable after run failure"
            for model, receipt in metadata.get("models", {}).items():
                if selection is not None and model not in pending:
                    continue
                required = [file.parent / model / "predictor.joblib",
                            file.parent / f"{model}_test_predictions.parquet",
                            file.parent / f"{model}_validation_predictions.parquet"]
                score = scores[scores.model.eq(model)] if "model" in scores else pd.DataFrame()
                ready = receipt.get("status") == "completed" and len(score) == 1 and all(p.is_file() for p in required)
                if selection is not None:
                    validation_score = validation[validation.model.eq(model)] if "model" in validation else pd.DataFrame()
                    ready = (ready and len(validation_score) == 1
                             and all(p.stat().st_size > 0 for p in required)
                             and all(np.isfinite(float(score.iloc[0].get(metric, np.nan)))
                                     for metric in ("mae", "rmse", "validation_mae")))
                    pending[model]["status"] = "completed" if ready else "awaiting completed artifacts"
                if ready:
                    selected[variant, model] = (run, score.iloc[0])
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            inspected.append({"run": run_name, "reason": str(exc)})
            for row in pending.values():
                row["status"] = "unavailable: " + str(exc)
        statuses.extend(pending.values())
    runs, rows = {}, []
    for (variant, model), (run, score) in selected.items():
        runs[run.name] = run
        rows.append({"feature_set": variant, "model": model, "run": run.name,
                     "label": f"{variant}-{model}", "run_status": run.metadata.get("status"),
                     "validation_mae": score.get("validation_mae"),
                     "test_mae": score.get("mae"), "test_rmse": score.get("rmse"),
                     "test_r2": score.get("r2")})
    return runs, pd.DataFrame(rows), pd.DataFrame(statuses if selection is not None else rows), pd.DataFrame(inspected)
