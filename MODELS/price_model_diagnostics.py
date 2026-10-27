"""Review completed price-model checkpoints without changing training runs."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from MODELS.Price_ML_Models import (
    PCR_COMPONENTS, PCR_FLAG_COLUMNS, PCR_PROTOCOL_VERSION, file_sha256, regression_metrics,
    execution_protocol_matches,
)

FEATURE_SETS = ("base", "enhanced", "enhanced_safety", "full")
MODELS = ("catboost", "lightgbm", "xgboost", "random_forest",
          "multiple_linear_regression", "lasso", "ridge", "elastic_net", "pcr")
TREE_MODELS = MODELS[:4]
MATCH_KEYS = ("split_fingerprint", "code_fingerprint", "preprocessing_version",
              "package_versions", "tuning_rows", "max_trees", "split_strategy",
              "price_weighting", "target_transform")


def models_for_variant(variant):
    return MODELS if variant == "full" else MODELS[:-1]


def pcr_matches_reference(metadata, reference):
    """Admit only PCR across implementation hashes; keep data/row protocol exact.

    PCR is a new estimator, so its code differs from the frozen original suite.
    The tree budget is irrelevant; every shared data and evaluation control still
    matches a full-feature reference. Other estimators retain strict code matching.
    """
    if reference is None or not metadata.get("code_fingerprint") or metadata.get("model_type") != "full":
        return False
    protocol = metadata.get("pcr_protocol", {})
    if (protocol.get("version") != PCR_PROTOCOL_VERSION or protocol.get("n_components") != PCR_COMPONENTS
            or protocol.get("whiten") is not False):
        return False
    if not all(metadata.get(k) == reference.get(k) for k in MATCH_KEYS if k not in {"code_fingerprint", "max_trees"}):
        return False
    if not execution_protocol_matches(metadata, reference):
        return False
    schema_keys = ("raw_feature_columns", "feature_columns", "numeric_features", "categorical_features")
    if any(reference.get(k) is None or metadata.get(k) != reference[k] for k in schema_keys):
        return False
    expected_numbers = [c for c in reference["numeric_features"]
                        if c not in PCR_FLAG_COLUMNS and not c.endswith("__missing")]
    if protocol.get("numerical_candidates") != expected_numbers:
        return False
    return bool(reference.get("source_file_sha256")) and metadata.get("source_file_sha256") == reference["source_file_sha256"]


def discover_checkpoints(output_root: Path, run_overrides=None):
    """Match the latest completed full-data base protocol; allow split restarts.

    Full adds external sources, so require matching hashes for every shared
    source rather than equality of the combined dataset fingerprint.
    """
    output_root = Path(output_root)
    candidates = []
    for variant in FEATURE_SETS:
        paths = ((run_overrides or {}).get(variant)
                 or sorted((output_root / variant).glob("*-full")))
        for path in paths:
            path = Path(path)
            metadata_path = path / "run_metadata.json"
            if metadata_path.exists():
                candidates.append((variant, path, json.loads(metadata_path.read_text(encoding="utf-8"))))
    anchors = [m for v, p, m in candidates if v == "base" and m.get("status") == "completed"]
    if not anchors:
        raise ValueError("No completed full-data base run. Supply matching RUN_OVERRIDES.")
    anchor = anchors[-1]
    if any(anchor.get(k) is None for k in MATCH_KEYS):
        raise ValueError("Base run lacks protocol metadata; legacy runs cannot be auto-combined")
    full_references = [m for v, _, m in candidates if v == "full"
                       and m.get("model_type") == "full"
                       and all(m.get(k) == anchor[k] for k in MATCH_KEYS)
                       and execution_protocol_matches(m, anchor)
                       and all(m.get("source_file_sha256", {}).get(k) == h
                               for k, h in anchor.get("source_file_sha256", {}).items())]
    full_reference = full_references[-1] if full_references else None
    selected, inventory = {}, []
    for variant, path, metadata in candidates:
        anchor_sources = anchor.get("source_file_sha256", {})
        candidate_sources = metadata.get("source_file_sha256", {})
        protocol_matches = (all(metadata.get(k) == anchor[k] for k in MATCH_KEYS)
                            and execution_protocol_matches(metadata, anchor))
        sources_match = bool(anchor_sources) and all(
            candidate_sources.get(k) == value for k, value in anchor_sources.items())
        legacy_compatible = protocol_matches and sources_match and metadata.get("model_type") == variant
        pcr_compatible = variant == "full" and pcr_matches_reference(metadata, full_reference)
        compatible = legacy_compatible or pcr_compatible
        inventory.append({"feature_set": variant, "run": str(path),
                          "run_status": metadata.get("status"), "compatible": compatible,
                          "dataset_fingerprint": metadata.get("dataset_fingerprint"),
                          "split_fingerprint": metadata.get("split_fingerprint"),
                          "source_files": len(metadata.get("source_file_sha256", {})),
                          "code_fingerprint": metadata.get("code_fingerprint"),
                          "reference_code_fingerprint": anchor["code_fingerprint"],
                          "reason": ("matched protocol and shared sources" if legacy_compatible else
                                     "PCR-only supplement: matched full sources/schema/splits; estimator code and tree budget may differ"
                                     if pcr_compatible else "protocol/source mismatch")})
        if not compatible:
            continue
        scores_path = path / "metrics.csv"
        if not scores_path.exists():
            continue
        scores = pd.read_csv(scores_path)
        for name in models_for_variant(variant):
            if (name == "pcr" and not pcr_compatible) or (name != "pcr" and not legacy_compatible):
                continue
            entry = metadata.get("models", {}).get(name, {})
            required = (path / name / "predictor.joblib", path / f"{name}_test_predictions.parquet",
                        path / f"{name}_validation_predictions.parquet")
            if entry.get("status") == "completed" and name in set(scores.model) and all(p.exists() for p in required):
                selected[variant, name] = (path, metadata, scores.set_index("model").loc[name])
    status = pd.DataFrame([
        {"feature_set": variant, "model": name,
         "status": "completed checkpoint" if (variant, name) in selected else "pending / unavailable",
         "run": str(selected[variant, name][0]) if (variant, name) in selected else None}
        for variant in FEATURE_SETS for name in models_for_variant(variant)])
    return selected, status, pd.DataFrame(inventory)


def receipt_identity(frame):
    """Bind scores to identical held-out VINs, labels, decoded makes and families."""
    columns = ["vin", "actual_price", "Make", "family_block"]
    if frame.vin.isna().any() or frame.vin.duplicated().any():
        raise ValueError("Prediction receipt requires one non-null row per lowercase vin")
    if not np.isfinite(frame[["actual_price", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite evaluation labels or predictions")
    ordered = frame[columns].sort_values("vin").reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).values.tobytes()).hexdigest()


def summarize_makes(frame):
    """All decoded makes, including missing identity, on the test population."""
    rows = []
    for make, part in frame.groupby("Make", dropna=False, observed=True):
        actual = part.actual_price.to_numpy(dtype=float)
        predicted = part.prediction.to_numpy(dtype=float)
        families = part.family_block.nunique(dropna=False)
        scores = regression_metrics(actual, predicted)
        rows.append({"Make": make if pd.notna(make) else "(missing decoded Make)",
                     "rows": len(part), "family_blocks": families,
                     "mean_actual_price": actual.mean(), "median_actual_price": np.median(actual),
                     "mse": np.mean((predicted - actual) ** 2), **scores,
                     "support_status": "supported" if len(part) >= 100 and families >= 5 else "low_support"})
    return pd.DataFrame(rows)


def collect_evaluations(selected, output_dir):
    """Recompute USD metrics and reconcile every completed model to saved scores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not selected:
        raise ValueError("No completed checkpoints match the selected full-data base protocol")
    identities, metrics, makes, evaluations = {}, [], [], []
    for (variant, name), (run, metadata, saved) in selected.items():
        for partition in ("validation", "test"):
            frame = pd.read_parquet(run / f"{name}_{partition}_predictions.parquet")
            identity = receipt_identity(frame)
            if partition in identities and identities[partition] != identity:
                raise ValueError(f"Unmatched {partition} VINs/labels/Make/families: {variant}/{name}")
            identities[partition] = identity
            scores = regression_metrics(frame.actual_price, frame.prediction)
            for key in ("mae", "rmse", "r2"):
                saved_key = f"validation_{key}" if partition == "validation" else key
                np.testing.assert_allclose(scores[key], saved[saved_key], rtol=1e-8, atol=1e-7)
            row = {"feature_set": variant, "model": name, "partition": partition,
                   "rows": len(frame), "mse": float(np.mean((frame.prediction - frame.actual_price) ** 2)),
                   **scores, "run": str(run), "receipt_identity": identity}
            metrics.append(row)
            if partition == "test":
                by_make = summarize_makes(frame)
                assert by_make.rows.sum() == len(frame)
                by_make.insert(0, "model", name)
                by_make.insert(0, "feature_set", variant)
                makes.append(by_make)
                baseline = pd.read_parquet(run / "training_median_baseline_test_predictions.parquet")
                if receipt_identity(baseline) != identity:
                    raise ValueError(f"Baseline test population differs: {run}")
                baseline_mae = np.mean(np.abs(baseline.prediction - baseline.actual_price))
                entry = metadata["models"][name]
                evaluations.append({"feature_set": variant, "model": name,
                                    "test_mae": scores["mae"],
                                    "mae_reduction_vs_median_pct": 100 * (1 - scores["mae"] / baseline_mae),
                                    "test_r2": scores["r2"], "bias_usd": scores["mean_signed_error"],
                                    "p95_error_usd": scores["p95_absolute_error"],
                                    "mae_family_bootstrap_low": saved.get("mae_group_bootstrap_low"),
                                    "mae_family_bootstrap_high": saved.get("mae_group_bootstrap_high"),
                                    "tree_cap_reached": entry.get("tree_cap_reached"),
                                    "selected_parameters": json.dumps(entry.get("selected_parameters", {})),
                                    "evaluation": "Positive held-out R²" if scores["r2"] > 0 else "Weak held-out fit (R² ≤ 0)"})
            del frame
    metrics, makes, evaluations = pd.DataFrame(metrics), pd.concat(makes, ignore_index=True), pd.DataFrame(evaluations)
    for name, table in [("metrics", metrics), ("make_metrics", makes), ("evaluations", evaluations)]:
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return metrics, makes, evaluations


def explain_checkpoint(run, metadata, name, cache_root, *, rows=100, background_rows=20,
                       forest_method="impurity"):
    """Top ten signed coefficients or SHAP from frozen inputs; no refits.

    Boosters use path-dependent SHAP; forests default to fitted global impurity
    importance. Optional exact forest SHAP averages individual tree values.
    """
    import joblib
    from scipy import sparse
    if forest_method not in {"impurity", "shap"}:
        raise ValueError("forest_method must be 'impurity' or 'shap'")
    if name == "random_forest" and forest_method == "shap":
        import shap

    run, cache_root = Path(run), Path(cache_root)
    predictor = run / name / "predictor.joblib"
    signature = {"method_version": 3 if name == "random_forest" else 1, "run": str(run.resolve()), "model": name,
                 "predictor_sha256": file_sha256(predictor), "rows": rows,
                 "background_rows": background_rows, "frozen_hashes": metadata.get("shap_input_sha256")}
    if name == "random_forest":
        signature["forest_method"] = forest_method
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    cache = cache_root / key
    if (cache / "status.json").exists():
        status = json.loads((cache / "status.json").read_text())
        if name not in TREE_MODELS:
            status["sample_rows"] = None
        return pd.read_csv(cache / "importance.csv"), status
    cache.mkdir(parents=True, exist_ok=True)
    for filename, expected in metadata.get("shap_input_sha256", {}).items():
        if file_sha256(run / filename) != expected:
            raise ValueError(f"Frozen explanation input changed: {filename}")
    x = pd.read_parquet(run / metadata["shap_inputs"]).head(rows)
    if list(x.columns) != metadata["feature_columns"]:
        raise ValueError("Frozen feature schema differs from fitted model")
    print(f"Loading saved {name} predictor", flush=True)
    bundle = joblib.load(predictor)
    print(f"Loaded saved {name} predictor", flush=True)
    estimator = bundle.named_steps["estimator"].estimator
    features, values, base, predicted = list(x.columns), None, None, None
    method = "native path-dependent TreeSHAP"
    is_impurity = name == "random_forest" and forest_method == "impurity"
    if is_impurity:
        features = list(estimator.named_steps["preprocess"].get_feature_names_out())
        fitted_importance = estimator.named_steps["model"].feature_importances_
        importance = pd.DataFrame({"feature": features, "value": fitted_importance})
        importance["magnitude"] = importance.value
        importance = importance.sort_values("magnitude", ascending=False)
        method = "fitted global normalized impurity importance (log-price squared-error reduction)"
    elif name not in TREE_MODELS:
        transformer = estimator.named_steps["preprocess"]
        linear = estimator.named_steps["model"]
        features = list(transformer.get_feature_names_out())
        coefficients = np.asarray(linear.coef_).reshape(-1)
        if len(features) != len(coefficients):
            raise ValueError("Coefficient names do not align with estimator inputs")
        importance = pd.DataFrame({"feature": features, "value": coefficients})
        importance["magnitude"] = importance.value.abs()
        importance = importance.sort_values("magnitude", ascending=False)
        method = ("signed coefficient (31 unwhitened PCs / standardized flags / one-hot categories)"
                  if name == "pcr" else "signed coefficient (standardized numeric / one-hot categorical)")
    else:
        if name == "random_forest":
            print("Preparing per-tree Random Forest SHAP", flush=True)
            transformer = estimator.named_steps["preprocess"]
            forest = estimator.named_steps["model"]
            transformed = transformer.transform(x)
            transformed = transformed.toarray() if sparse.issparse(transformed) else transformed
            features = list(transformer.get_feature_names_out())
            # SHAP is linear in the ensemble output. Explain one tree at a time
            # to avoid padding hundreds of large trees into one dense tensor.
            values = np.zeros((len(x), len(features)), dtype=float)
            base = np.zeros(len(x), dtype=float)
            for index, tree in enumerate(forest.estimators_, start=1):
                explainer = shap.TreeExplainer(tree, feature_perturbation="tree_path_dependent")
                values += explainer.shap_values(transformed, check_additivity=False) / len(forest.estimators_)
                base += float(np.asarray(explainer.expected_value).reshape(-1)[0]) / len(forest.estimators_)
                if index % 25 == 0:
                    print(f"Random Forest SHAP: {index}/{len(forest.estimators_)} trees", flush=True)
                del explainer
            predicted = forest.predict(transformed)
            method = "path-dependent TreeSHAP (mean of exact per-tree contributions)"
        elif name == "catboost":
            from catboost import Pool
            raw = estimator.get_feature_importance(
                Pool(x, cat_features=metadata["categorical_features"]), type="ShapValues", thread_count=2)
            values, base = raw[:, :-1], raw[:, -1]
            predicted = estimator.predict(x, thread_count=2)
        elif name == "xgboost":
            import xgboost as xgb
            estimator.get_booster().set_param({"nthread": 2})
            raw = estimator.get_booster().predict(xgb.DMatrix(x, enable_categorical=True), pred_contribs=True)
            values, base, predicted = raw[:, :-1], raw[:, -1], estimator.predict(x)
        else:
            raw = estimator.predict(x, pred_contrib=True, num_threads=2)
            values, base, predicted = raw[:, :-1], raw[:, -1], estimator.predict(x, num_threads=2)
        values = np.asarray(values)
        if values.shape != (len(x), len(features)):
            raise ValueError("SHAP shape does not match model inputs")
        np.testing.assert_allclose(base + values.sum(axis=1), predicted, rtol=1e-4, atol=1e-4)
        importance = pd.DataFrame({"feature": features, "value": np.abs(values).mean(axis=0)})
        importance["magnitude"] = importance.value
        importance = importance.sort_values("magnitude", ascending=False)
        receipt = pd.read_parquet(run / metadata["shap_receipt"]).head(rows)
        if len(receipt) != len(x):
            raise ValueError("SHAP sample receipt length differs")
        row_values = pd.DataFrame(values, columns=features)
        row_values.insert(0, "base_log_price", base)
        row_values.insert(0, "predicted_log_price", predicted)
        row_values.insert(0, "vin", receipt.vin.to_numpy())
        row_values.to_parquet(cache / "shap_values.parquet", index=False)
    importance = importance.head(10).reset_index(drop=True)
    importance.to_csv(cache / "importance.csv", index=False)
    status = {**signature, "status": "completed", "method": method,
              "scale": "share of total training impurity reduction" if is_impurity else "ln(price in USD)",
              "sample_rows": len(x) if name in TREE_MODELS and not is_impurity else None,
              "cache": str(cache)}
    (cache / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    del bundle, estimator
    gc.collect()
    return importance, status


def collect_importance(selected, cache_root, rows=100, forest_rows=20, forest_method="impurity"):
    tables, statuses = [], []
    for (variant, name), (run, metadata, _) in selected.items():
        print(f"Explaining {variant} / {name}", flush=True)
        try:
            sample_rows = forest_rows if name == "random_forest" else rows
            table, status = explain_checkpoint(run, metadata, name, cache_root, rows=sample_rows,
                                               forest_method=forest_method)
            table.insert(0, "model", name)
            table.insert(0, "feature_set", variant)
            tables.append(table)
        except Exception as exc:
            status = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        statuses.append({"feature_set": variant, "model": name, **status})
    empty = pd.DataFrame(columns=["feature_set", "model", "feature", "value", "magnitude"])
    return pd.concat(tables, ignore_index=True) if tables else empty, pd.DataFrame(statuses)
