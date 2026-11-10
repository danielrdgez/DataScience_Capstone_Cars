"""Recommendation prices from selected checkpoints or an explicitly activated OOF bundle."""
from __future__ import annotations

import json
import importlib.metadata
from functools import lru_cache
from pathlib import Path

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
        result[family] = {"kind": "saved_checkpoint_v1", "run": choice.run, "model": choice.model,
            "feature_set": choice.feature_set, "validation_mae_usd": float(choice.validation_mae),
            "selection_metric": "validation_mae_usd", "metadata_sha256": digest(path / "run_metadata.json"),
            "predictor_sha256": digest(path / choice.model / "predictor.joblib")}
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
    import numpy as np
    path = check_checkpoint(selection)
    rows = pd.read_parquet(path / (selection["model"] + "_test_predictions.parquet"), columns=["vin", "prediction"])
    split = pd.read_parquet(path / "split_manifest.parquet")
    test_vins = set(split.loc[split.partition.eq("test"), "vin"])
    if (rows.vin.isna().any() or rows.vin.duplicated().any() or not set(rows.vin).issubset(test_vins)
            or not np.isfinite(rows.prediction).all() or not rows.prediction.gt(0).all()):
        raise ValueError("Historical price ranking requires unique valid held-out test predictions")
    return rows.rename(columns={"prediction": "predicted_price_usd"})


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
    return result, warnings
