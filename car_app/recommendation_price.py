"""Recommendation prices from selected checkpoints or an explicitly activated OOF bundle."""
from __future__ import annotations

import json
import importlib.metadata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from car_app.data import ROOT
from car_app.artifacts import stat_key
from MODELS import price_oof as oof


def bundle_path(selection):
    if not selection or selection.get("kind") != oof.KIND or not selection.get("run"):
        raise ValueError("Full CatBoost OOF pricing is pending training, validation and activation.")
    path = (ROOT / "MODEL_OUTPUT" / selection["run"]).resolve()
    if not path.is_relative_to((ROOT / "MODEL_OUTPUT").resolve()):
        raise ValueError("Recommendation price path escapes MODEL_OUTPUT")
    return path


def bundle_signature(selection):
    path = bundle_path(selection)
    manifest = oof.read_json(path / "manifest.json")
    files = [path / "manifest.json", path / "validation.json",
             *[oof.checked_path(path, f) for f in manifest["files"]],
             *oof.source_paths(manifest["options"]),
             *[ROOT / f for f in manifest["environment"]["code"]]]
    return (*((str(f), stat_key(f)) for f in files),
            ("packages", tuple((p, importlib.metadata.version(p)) for p in oof.PACKAGES)))


@lru_cache(maxsize=2)
def _load(selection_json, signature):
    selection = json.loads(selection_json)
    path = bundle_path(selection)
    if oof.digest(path / "manifest.json") != selection["manifest_sha256"]:
        raise ValueError("Pinned recommendation manifest changed")
    if oof.digest(path / "validation.json") != selection["validation_sha256"]:
        raise ValueError("Pinned recommendation validation changed")
    receipt = oof.read_json(path / "validation.json")
    if not receipt.get("passed") or receipt.get("manifest_sha256") != selection["manifest_sha256"] or receipt.get("code") != oof.code_hashes():
        raise ValueError("Recommendation validation does not match current code and bundle")
    manifest = oof.validate_bundle(path, audit=False)
    if manifest["reduced_budget"]:
        raise ValueError("Reduced-budget bundles cannot supply production recommendations")
    frame = pd.read_parquet(path / "oof_predictions.parquet",
                            columns=["vin", "prediction", "fold", "predictor_sha256"])
    if signature != bundle_signature(selection):
        raise ValueError("Recommendation bundle changed during loading")
    return path, manifest, frame


def load_bundle(selection):
    bundle_path(selection)
    selection = {k: selection[k] for k in ("kind", "run", "manifest_sha256", "validation_sha256")}
    return _load(json.dumps(selection, sort_keys=True), bundle_signature(selection))


def select_model(config):
    if (config.get("recommendation_price") or {}).get("kind") == "best_tree_linear_v1":
        return select_checkpoint_models(config)
    result = {"catboost": None, "primary": None, "unavailable": []}
    selection = config.get("recommendation_price")
    try:
        path, manifest, _ = load_bundle(selection)
        receipt = {**selection, "model": "catboost", "feature_set": "full",
                   "oof_mae_usd": oof.read_json(path / "metrics.json")["pooled"]["mae"],
                   "folds": manifest["options"]["folds"]}
        result.update(catboost=receipt, primary=receipt)
    except (OSError, ValueError, KeyError, AssertionError, pl.exceptions.PolarsError) as exc:
        result["unavailable"].append("Full CatBoost price evidence unavailable: " + str(exc))
    return result


def heldout_prices(selection):
    """Return a copy so ranking cannot mutate the shared cached prediction bank."""
    if selection and selection.get("kind") == "saved_checkpoint_v1":
        return checkpoint_prices(selection)
    _, _, rows = load_bundle(selection)
    return rows.rename(columns={"prediction": "predicted_price_usd"}).copy()


def price_evidence(listings, models):
    if "tree" in models or "linear" in models:
        return checkpoint_evidence(listings, models)
    result = {"listing:" + str(v): {} for v in listings["vin"].to_list()}
    warnings = list(models["unavailable"])
    selection = models.get("catboost")
    if not selection:
        return result, warnings
    try:
        path, manifest, bank = load_bundle(selection)
        requested = listings["vin"].to_list()
        matched = bank.loc[bank.vin.isin(requested)]
        for row in matched.itertuples():
            result["listing:" + str(row.vin)]["catboost"] = {
                **selection, "predicted_price_usd": float(row.prediction), "fold": int(row.fold),
                "predictor_sha256": row.predictor_sha256, "prediction_kind": "out_of_fold",
                "use": "Held-out historical price estimate; eligible for model-relative price, not a verified discount."}
        remaining = set(requested) - set(matched.vin)
        if remaining:
            training = set(pl.read_parquet(path / "folds.parquet", columns=["vin"])["vin"].to_list())
            if remaining & training:
                raise ValueError("A training VIN has no OOF prediction; final-model substitution is forbidden")
            # Catalog requests contain IDs; recover their observed listing inputs before inference.
            raw = pl.scan_parquet(ROOT / "DATA/CAR_DATA_FINAL/used_cars.parquet").filter(
                pl.col("vin").is_in(sorted(remaining))).collect()
            if raw.height != len(remaining):
                warnings.append("No saved input row for some requested VINs; use the new-listing scoring command with actual inputs.")
            if not raw.is_empty():
                predictions = oof.inference_rows(path, manifest, raw)
                for row in predictions.to_dicts():
                    result["listing:" + row["vin"]]["catboost"] = {
                        **selection, **{k: row[k] for k in ("predicted_price_usd", "prediction_kind", "predictor_sha256", "missing_feature_count", "decoded_identity_missing")},
                        "use": row["limitation"]}
    except (OSError, ValueError, KeyError, AssertionError, pl.exceptions.PolarsError) as exc:
        # Do not return evidence from a partially failed or changed bundle.
        result = {"listing:" + str(v): {} for v in listings["vin"].to_list()}
        warnings.append("Full CatBoost price evidence unavailable: " + str(exc))
    return result, warnings


def activate(run):
    from car_app.model_registry import CONFIG

    run = Path(run).resolve()
    manifest = oof.validate_bundle(run, audit=False)
    receipt = oof.read_json(run / "validation.json")
    if manifest["reduced_budget"] or not receipt.get("passed") or receipt.get("manifest_sha256") != oof.digest(run / "manifest.json") or receipt.get("code") != oof.code_hashes():
        raise ValueError("Activation requires a matching full-budget replay validation")
    selection = {"kind": oof.KIND, "run": run.relative_to(ROOT / "MODEL_OUTPUT").as_posix(),
                 "manifest_sha256": oof.digest(run / "manifest.json"),
                 "validation_sha256": oof.digest(run / "validation.json")}
    config = oof.read_json(CONFIG)
    config["recommendation_price"] = selection
    config.pop("recommendation_price_selection", None)
    config["recommendation_policy_version"] = 4
    oof.write_json(CONFIG, config)
    return selection


TREE_MODELS = {"catboost", "lightgbm", "xgboost", "random_forest"}
LINEAR_MODELS = {"multiple_linear_regression", "lasso", "ridge", "elastic_net", "pcr"}


def select_checkpoint_models(config):
    """Select each family's lowest validation MAE among configured completed checkpoints."""
    from car_app.artifacts import diagnostic_inventory
    from car_app.model_registry import digest
    result = {"tree": None, "linear": None, "primary": None, "unavailable": []}
    _, candidates, _, _ = diagnostic_inventory(config.get("price_run_selection"))
    for family, names in (("tree", TREE_MODELS), ("linear", LINEAR_MODELS)):
        if candidates.empty:
            choices = candidates
        else:
            choices = candidates.loc[candidates.model.isin(names)].copy()
            choices = choices.loc[choices.validation_mae.notna() & choices.validation_mae.ge(0)]
            choices = choices.sort_values(["validation_mae", "feature_set", "model", "run"])
        if choices.empty:
            result["unavailable"].append(f"No completed {family} price checkpoint is available.")
            continue
        choice = choices.iloc[0]
        path = ROOT / "MODEL_OUTPUT" / choice.run
        selection = {"kind": "saved_checkpoint_v1", "run": choice.run, "model": choice.model,
            "feature_set": choice.feature_set, "validation_mae_usd": float(choice.validation_mae),
            "selection_metric": "validation_mae_usd", "metadata_sha256": digest(path / "run_metadata.json"),
            "predictor_sha256": digest(path / choice.model / "predictor.joblib")}
        try:
            check_checkpoint(selection)
        except (OSError, ValueError, KeyError, AssertionError, pl.exceptions.PolarsError) as exc:
            result["unavailable"].append(f"{family.capitalize()} price checkpoint unavailable: {exc}")
            continue
        result[family] = selection
    result["primary"] = result["tree"] or result["linear"]
    return result


def check_checkpoint(selection):
    from car_app.model_registry import digest, validate_price_run
    path = (ROOT / "MODEL_OUTPUT" / selection["run"]).resolve()
    if not path.is_relative_to((ROOT / "MODEL_OUTPUT").resolve()):
        raise ValueError("Recommendation checkpoint escapes MODEL_OUTPUT")
    if (digest(path / "run_metadata.json") != selection["metadata_sha256"]
            or digest(path / selection["model"] / "predictor.joblib") != selection["predictor_sha256"]):
        raise ValueError("Selected recommendation checkpoint changed")
    validate_price_run(path, selection["model"])
    return path


def checkpoint_prices(selection):
    """Only saved, unique test VINs may support model-relative historical ranking."""
    path = check_checkpoint(selection)
    rows = pd.read_parquet(path / (selection["model"] + "_test_predictions.parquet"), columns=["vin", "prediction"])
    split = pd.read_parquet(path / "split_manifest.parquet")
    test_vins = set(split.loc[split.partition.eq("test"), "vin"])
    if (rows.vin.isna().any() or rows.vin.duplicated().any() or not set(rows.vin).issubset(test_vins)
            or not np.isfinite(rows.prediction).all() or not rows.prediction.gt(0).all()):
        raise ValueError("Historical price ranking requires unique valid held-out test predictions")
    return rows.rename(columns={"prediction": "predicted_price_usd"})


def linear_contributions(model, prepared, background):
    """Center the fitted linear terms on the mean encoded training reference."""
    from scipy import sparse

    encoder = model.named_steps["preprocess"]
    regression = model.named_steps["model"]
    encoded = encoder.transform(prepared)
    reference = np.asarray(encoder.transform(background).mean(axis=0)).ravel()
    values = encoded.toarray() if sparse.issparse(encoded) else encoded
    weights = regression.coef_
    baseline = float(regression.intercept_ + reference @ weights)
    return list(encoder.get_feature_names_out()), (values - reference) * weights, np.full(len(prepared), baseline)


def tree_contributions(name, model, prepared, metadata):
    """Return signed TreeSHAP terms in the fitted estimator's feature space."""
    if name == "catboost":
        from catboost import Pool

        inputs = Pool(prepared, cat_features=metadata["categorical_features"])
        values = model.get_feature_importance(inputs, type="ShapValues", thread_count=4)
    elif name == "xgboost":
        import xgboost as xgb

        inputs = xgb.DMatrix(prepared, enable_categorical=True)
        values = model.get_booster().predict(inputs, pred_contribs=True)
    elif name == "lightgbm":
        values = model.predict(prepared, pred_contrib=True)
    elif name == "random_forest":
        import shap
        from scipy import sparse

        encoder = model.named_steps["preprocess"]
        inputs = encoder.transform(prepared)
        explainer = shap.TreeExplainer(model.named_steps["model"])
        values = explainer.shap_values(inputs.toarray() if sparse.issparse(inputs) else inputs)
        return list(encoder.get_feature_names_out()), values, np.full(len(prepared), float(np.asarray(explainer.expected_value).item()))
    else:
        raise ValueError("Unsupported tree explanation")
    return list(prepared.columns), values[:, :-1], values[:, -1]


def checkpoint_contributions(selection, family, estimates):
    """Explain only candidate rows whose contributions reconcile to the delivered price."""
    from car_app.model_registry import checkpoint_prediction_inputs, digest

    predictor, metadata, raw = checkpoint_prediction_inputs(estimates, selection)
    if metadata.get("target_transform") != "natural_log":
        raise ValueError("Expected a natural-log price checkpoint")
    prepared = predictor.named_steps["input"].transform(raw)
    model = predictor.named_steps["estimator"].estimator
    if family == "linear":
        filename = metadata["shap_background"]
        background_path = ROOT / "MODEL_OUTPUT" / selection["run"] / filename
        if digest(background_path) != metadata["shap_input_sha256"][filename]:
            raise ValueError("Saved training reference changed")
        background = pd.read_parquet(background_path)
        if background.empty or list(background.columns) != list(prepared.columns):
            raise ValueError("Saved training reference does not match checkpoint inputs")
        features, values, bases = linear_contributions(model, prepared, background)
        method = "baseline-centered linear contributions"
        reference = "mean encoded features from the frozen training sample"
    else:
        features, values, bases = tree_contributions(selection["model"], model, prepared, metadata)
        method = "TreeSHAP"
        reference = "TreeSHAP expected prediction"

    if values.shape != (len(estimates), len(features)) or bases.shape != (len(estimates),):
        raise ValueError("Contribution dimensions do not match candidate rows and features")
    log_prices = np.asarray(model.predict(prepared), dtype=float)
    if log_prices.shape != (len(estimates),):
        raise ValueError("Predictions do not match candidate rows")
    reconstructed = bases + values.sum(axis=1)
    valid = (np.isfinite(values).all(axis=1) & np.isfinite(bases)
             & np.isclose(reconstructed, log_prices, rtol=0, atol=1e-5)
             & np.isclose(np.exp(log_prices), list(estimates.values()), rtol=1e-7, atol=.01))
    result = {}
    for index, vin in enumerate(estimates):
        if not valid[index]:
            continue
        terms = values[index]
        top = np.argsort(-np.abs(terms), kind="stable")[:4]
        result[vin] = {
            "method": method, "reference": reference, "scale": "ln(price in USD)",
            "baseline_log_price": float(bases[index]), "predicted_log_price": float(log_prices[index]),
            "contributions": [{"feature": features[i], "signed_log_price": float(terms[i])}
                              for i in top if terms[i] != 0],
            "other_features_log_price": float(terms.sum() - terms[top].sum()),
        }
    return result


def checkpoint_evidence(listings, models):
    from car_app.model_registry import price_estimates
    result = {"listing:" + str(v): {} for v in listings["vin"].to_list()}
    warnings = list(models["unavailable"])
    for family in ("tree", "linear"):
        selection = models.get(family)
        if not selection:
            continue
        try:
            bank = checkpoint_prices(selection)
            matched = bank.loc[bank.vin.isin(listings["vin"].to_list())]
            for row in matched.itertuples():
                result["listing:" + str(row.vin)][family] = {**selection,
                    "predicted_price_usd": float(row.predicted_price_usd), "prediction_kind": "held_out_test",
                    "use": "Saved held-out historical estimate; not a verified discount."}
            remaining = listings.filter(~pl.col("vin").is_in(matched.vin.tolist()))
            if not remaining.is_empty():
                estimated, warning = price_estimates(remaining, {"active_price": selection})
                if warning:
                    warnings.append(family + ": " + warning)
                else:
                    for row in estimated.iter_rows(named=True):
                        result["listing:" + str(row["vin"])][family] = {**selection,
                            "predicted_price_usd": float(row["predicted_price_usd"]), "prediction_kind": "supplemental_inference",
                            "use": "May include fitting VINs; cannot support held-out value ranking or evaluation."}
        except (OSError, ValueError, KeyError, AssertionError, pl.exceptions.PolarsError) as exc:
            for evidence in result.values():
                evidence.pop(family, None)
            warnings.append(family + " price evidence unavailable: " + str(exc))
            continue
        estimates = {vin: result["listing:" + vin][family]["predicted_price_usd"]
                     for vin in listings["vin"].to_list() if family in result["listing:" + vin]}
        if not estimates:
            continue
        try:
            explanations = checkpoint_contributions(selection, family, estimates)
            for vin, explanation in explanations.items():
                result["listing:" + vin][family]["explanation"] = explanation
            if len(explanations) != len(estimates):
                warnings.append(f"{family.capitalize()} price explanations failed to reproduce some displayed estimates.")
        except Exception as exc:
            warnings.append(f"{family.capitalize()} price explanations unavailable ({type(exc).__name__}).")
    return result, warnings
