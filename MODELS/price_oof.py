"""Full CatBoost cross-fitted prices and a separate all-data inference predictor.

Run training explicitly; importing this module never trains or activates a model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

from MODELS import Price_ML_Models as price
from EDA.DATA_CLEANING import DEFAULT_COHORT_REVIEW_DIR, FINAL_DATASET_PATH

KIND = "catboost_full_oof_v1"
ROOT = price.PROJECT_ROOT
FEATURES = list(price.MODEL_FEATURE_SETS["full"])
PACKAGES = ("catboost", "numpy", "pandas", "polars", "scikit-learn", "pyarrow")
LOG = logging.getLogger(__name__)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    price._write_json_checkpoint(Path(path), data)


@lru_cache(maxsize=512)
def _digest(path, size, modified):
    return price.file_sha256(Path(path))


def digest(path):
    path = Path(path).resolve()
    info = path.stat()
    return _digest(str(path), info.st_size, info.st_mtime_ns)


def code_hashes():
    return {name: digest(ROOT / name) for name in (*price.PRICE_CODE_FILES, "MODELS/price_oof.py")}


def source_paths(options):
    paths = [FINAL_DATASET_PATH / name for name in (
        "used_cars.parquet", "NHTSA_vPIC.parquet", "NHTSA_Complaints.parquet",
        "NHTSA_Recalls.parquet", "NHTSA_Safety.parquet", "manifest.json")]
    youtube = Path(options["youtube_feature_dir"])
    expert = Path(options["expert_feature_dir"])
    paths += [youtube / name for name in ("make_features.parquet", "price_feature_manifest.json")]
    paths += [expert / name for name in ("cohort_features.parquet", "make_features.parquet",
                                        "expert_feature_manifest.json", "review_resolution.parquet")]
    reviews = sorted(DEFAULT_COHORT_REVIEW_DIR.glob("*/cohort_outlier_candidates.csv"))
    if not reviews:
        raise ValueError("The selected cohort-exclusion review is missing")
    return [*paths, reviews[-1]]


def environment(options):
    return {"sources": {str(p.resolve()): digest(p) for p in source_paths(options)},
            "code": code_hashes(),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
            "features": FEATURES, "preprocessing": price.PREPROCESSING_VERSION}


def checked_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escapes its run")
    return path


def verify_files(root, files):
    for name, expected in files.items():
        if digest(checked_path(root, name)) != expected:
            raise ValueError(f"Artifact changed: {name}")


def stratum_positions(frame):
    keys = frame[list(price.COHORT_COLUMNS[:3])].copy()
    for column in ("Make", "Model"):
        keys[column] = keys[column].astype("string").str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    return keys.groupby(list(keys.columns), dropna=False, sort=True).indices.values()


def hash_order(frame, positions, salt):
    vins = frame.vin.to_numpy()
    return np.asarray(sorted(positions, key=lambda i: hashlib.sha256(
        f"{price.RANDOM_SEED}:{salt}:{vins[i]}".encode()).digest()), dtype=np.int64)


def assign_folds(frame, folds):
    """Assign balanced VIN folds in already-eligible decoded Y/M/M strata."""
    if frame.vin.isna().any() or frame.vin.duplicated().any() or frame.vin.astype(str).str.strip().eq("").any():
        raise ValueError("OOF requires unique, nonmissing VINs")
    assignment = np.full(len(frame), -1, dtype=np.int16)
    for positions in stratum_positions(frame):
        if len(positions) < folds:
            raise ValueError("Each retained stratum must support every outer fold")
        ordered = hash_order(frame, positions, "outer")
        assignment[ordered] = np.arange(len(ordered)) % folds
    return assignment


def inner_split(frame, training, salt):
    """Split only the training portion; never reapply outer eligibility cutoffs."""
    subset = frame[["vin", *price.COHORT_COLUMNS[:3]]].iloc[training].reset_index(drop=True)
    fit, validation = [], []
    for positions in stratum_positions(subset):
        if len(positions) < 2:
            raise ValueError("Inner selection needs at least two training VINs per stratum")
        ordered = hash_order(subset, positions, salt)
        count = max(1, len(ordered) // 5)
        validation.extend(training[ordered[:count]])
        fit.extend(training[ordered[count:]])
    return np.sort(fit), np.sort(validation)


@contextmanager
def run_lock(run):
    """OS lock releases on crashes; the harmless lock file may remain."""
    import os
    with (run / ".training.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fit_stage(run, name, frame, training, heldout, options, data_info, categorical, device):
    stage = run / name
    stage.mkdir(exist_ok=True)
    fit, validation = inner_split(frame, training, name)
    groups = frame._cohort_group.to_numpy()
    tune_fit, _ = price.cohort_covered_sample(fit, groups, options["tuning_rows"], 20, frame.vin)
    tune_val, _ = price.cohort_covered_sample(validation, groups, options["tuning_rows"], 5, frame.vin)
    # Store roles for the full population, including the forbidden outer fold.
    roles = frame[["vin"]].copy()
    roles["role"] = "outer_holdout"
    roles.loc[fit, "role"] = "inner_fit"
    roles.loc[validation, "role"] = "inner_validation"
    roles["tuning_sample"] = False
    roles.loc[np.concatenate([tune_fit, tune_val]), "tuning_sample"] = True
    roles.to_parquet(stage / "selection_split.parquet", index=False)
    numeric = [c for c in FEATURES if c not in categorical]
    x = frame[FEATURES]
    y_log = np.log(frame[price.TARGET_COLUMN].to_numpy(float))
    tuning = price.PricePreprocessor(categorical, numeric)
    diagnostics, selected = [], {}
    LOG.info("%s: selecting settings using %s fit and %s validation rows", name, len(tune_fit), len(tune_val))
    iterations, _, parameters, cap_limited = price._fit_with_early_stopping(
        "catboost", tuning.fit_transform(x.iloc[tune_fit]), y_log[tune_fit],
        tuning.transform(x.iloc[tune_val]), y_log[tune_val], categorical,
        options["max_trees"], diagnostics, selected_evaluation=selected,
        progress=lambda: write_json(stage / "tuning_progress.json", diagnostics), device=device)
    write_json(stage / "selection.json", {"parameters": parameters, "iterations": iterations,
        "sample_validation_mae": selected["validation_mae"], "cap_limited": cap_limited})
    del tuning, selected
    gc.collect()
    # Reproduce the existing full-validation measurement before the outer refit.
    validation_preprocessor = price.PricePreprocessor(categorical, numeric)
    validation_model = price._make_tree_model("catboost", iterations, early_stopping=False,
                                             parameters=parameters, device=device)
    validation_model.fit(validation_preprocessor.fit_transform(x.iloc[fit]), y_log[fit],
                         cat_features=categorical, verbose=False)
    validation_prediction = np.exp(validation_model.predict(validation_preprocessor.transform(x.iloc[validation])))
    full_validation_mae = price.regression_metrics(np.exp(y_log[validation]), validation_prediction)["mae"]
    del validation_model, validation_preprocessor, validation_prediction
    gc.collect()
    LOG.info("%s: refitting %s trees on %s rows", name, iterations, len(training))
    transform = price.PricePreprocessor(categorical, numeric)
    model = price._make_tree_model("catboost", iterations, early_stopping=False,
                                  parameters=parameters, device=device)
    model.fit(transform.fit_transform(x.iloc[training]), y_log[training], cat_features=categorical, verbose=False)
    predictor = Pipeline([("input", transform), ("estimator", price.ExpLogPricePredictor(model))])
    predictor.raw_feature_columns_ = FEATURES
    predictor.expert_feature_info_ = data_info["expert_features"]
    predictor.youtube_feature_info_ = data_info["youtube_features"]
    joblib.dump(predictor, stage / "predictor.joblib", compress=3)
    model.save_model(str(stage / "model.cbm"))
    files = ["selection_split.parquet", "selection.json", "predictor.joblib", "model.cbm"]
    if len(heldout):
        columns = list(dict.fromkeys(["vin", *price.COHORT_COLUMNS, "is_new", "mileage", "_cohort_group"]))
        predictions = frame.iloc[heldout][columns].copy().reset_index(drop=True)
        predictions = predictions.rename(columns={"_cohort_group": "family_block"})
        predictions["actual_price"] = np.exp(y_log[heldout])
        predictions["prediction"] = predictor.predict(x.iloc[heldout])
        ensure_prices(predictions.prediction)
        predictions["residual"] = predictions.actual_price - predictions.prediction
        predictions["fold"] = int(name.split("_")[-1])
        predictions["predictor_sha256"] = digest(stage / "predictor.joblib")
        predictions.to_parquet(stage / "predictions.parquet", index=False)
        files.append("predictions.parquet")
    # Frozen replay references are not evaluation claims for the all-data model.
    reference = heldout if len(heldout) else training
    positions = hash_order(frame, reference, "replay")[:32]
    raw = x.iloc[positions].reset_index(drop=True)
    raw.to_parquet(stage / "replay_raw.parquet", index=False)
    transform.transform(raw).to_parquet(stage / "replay_prepared.parquet", index=False)
    pd.DataFrame({"vin": frame.iloc[positions].vin.to_numpy(),
                  "prediction": predictor.predict(raw)}).to_parquet(stage / "replay_predictions.parquet", index=False)
    files += ["replay_raw.parquet", "replay_prepared.parquet", "replay_predictions.parquet"]
    result = {"status": "completed", "training_rows": len(training), "heldout_rows": len(heldout),
              "validation_mae": full_validation_mae, "iterations": iterations,
              "files": {f: digest(stage / f) for f in files}}
    write_json(stage / "stage.json", result)
    LOG.info("%s completed", name)
    del predictor, model, transform, x
    gc.collect()
    return result


def ensure_prices(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Predictions must be finite positive USD values")


def train(args):
    run = args.output.resolve()
    if not run.is_relative_to((ROOT / "MODEL_OUTPUT").resolve()):
        raise ValueError("Training output must be under MODEL_OUTPUT")
    if args.resume:
        manifest = read_json(run / "manifest.json")
        if manifest.get("kind") != KIND:
            raise ValueError("Resume requires a Full CatBoost OOF run")
        options = manifest["options"]
        for key in options:
            supplied = getattr(args, key, None)
            if supplied is not None and str(supplied) != str(options[key]):
                raise ValueError(f"Resume option differs: {key}; omit it to reuse the saved configuration")
    else:
        options = {"folds": args.folds if args.folds is not None else 10,
                   "tuning_rows": args.tuning_rows if args.tuning_rows is not None else 200000,
                   "max_trees": args.max_trees if args.max_trees is not None else 0,
                   "sample_size": args.sample_size if args.sample_size is not None else 0,
                   "device": args.device or "auto",
                   "youtube_feature_dir": str((args.youtube_feature_dir or price.YOUTUBE_FEATURE_DIRECTORY).resolve()),
                   "expert_feature_dir": str((args.expert_feature_dir or price.EXPERT_FEATURE_DIRECTORY).resolve())}
        if not 2 <= options["folds"] <= 10 or options["tuning_rows"] < 1 or min(options["max_trees"], options["sample_size"]) < 0:
            raise ValueError("Require 2-10 folds, positive tuning rows, and nonnegative tree/sample limits")
        run.mkdir(parents=True, exist_ok=False)
        manifest = {"kind": KIND, "status": "running", "model": "catboost", "feature_set": "full",
                    "created_at": datetime.now(timezone.utc).isoformat(), "options": options, "stages": {},
                    "reduced_budget": bool(options["sample_size"] or options["max_trees"] or options["folds"] != 10)}
    with run_lock(run):
        current = environment(options)
        if args.resume and manifest["environment"] != current:
            raise ValueError("Source, code, packages or feature contract changed; start a fresh run")
        manifest["environment"] = current
        if manifest["status"] == "completed":
            verify_files(run, manifest["files"])
            print(f"Already completed: {run}")
            return
        write_json(run / "manifest.json", manifest)
        frame, data_info = price.load_model_frame("full", options["sample_size"] or None,
            youtube_feature_dir=Path(options["youtube_feature_dir"]), expert_feature_dir=Path(options["expert_feature_dir"]))
        categorical = [c for c in FEATURES if frame.schema[c] == pl.String]
        frame, _, _, _, _, audit = price.split_stratified_cohorts(frame.to_pandas())
        frame = frame.sort_values("vin").reset_index(drop=True)
        actual = frame[price.TARGET_COLUMN].to_numpy(float)
        ensure_prices(actual)
        folds = assign_folds(frame, options["folds"])
        if environment(options) != current:
            raise ValueError("Inputs changed during preparation")
        assignments = frame[["vin", *price.COHORT_COLUMNS]].copy()
        assignments["fold"] = folds
        assignments["actual_price"] = actual
        if args.resume and (run / "folds.parquet").exists():
            pd.testing.assert_frame_equal(assignments, pd.read_parquet(run / "folds.parquet"))
        else:
            assignments.to_parquet(run / "folds.parquet", index=False)
            audit.to_csv(run / "eligibility.csv", index=False)
        devices = price.resolve_training_devices(["catboost"], options["device"])
        if manifest.get("training_devices") and manifest["training_devices"] != devices:
            raise ValueError("Resolved training device changed; resume on the original device")
        manifest.update(training_devices=devices, data_info=data_info, rows=len(frame),
                        folds_sha256=digest(run / "folds.parquet"))
        write_json(run / "manifest.json", manifest)
        all_rows = np.arange(len(frame))
        stages = [(f"fold_{i:02d}", all_rows[folds != i], all_rows[folds == i]) for i in range(options["folds"])]
        stages.append(("final", all_rows, np.array([], dtype=int)))
        for name, training, heldout in stages:
            if name in manifest["stages"]:
                verify_files(run / name, manifest["stages"][name]["files"])
                continue
            if environment(options) != current:
                raise ValueError("Inputs changed during training")
            manifest["stages"][name] = fit_stage(run, name, frame, training, heldout, options,
                data_info, categorical, devices["catboost"]["device"])
            write_json(run / "manifest.json", manifest)
        oof = pd.concat([pd.read_parquet(run / f"fold_{i:02d}/predictions.parquet")
                         for i in range(options["folds"])], ignore_index=True).sort_values("vin").reset_index(drop=True)
        validate_oof_rows(assignments, oof)
        oof.to_parquet(run / "oof_predictions.parquet", index=False)
        metrics = {"scope": "nested inner-holdout selection; outer VIN CV within known Y/M/M strata",
                   "pooled": price.regression_metrics(oof.actual_price, oof.prediction),
                   "folds": [{"fold": int(i), "rows": len(part), **price.regression_metrics(part.actual_price, part.prediction)}
                             for i, part in oof.groupby("fold")],
                   "final_predictor_independently_tested": False,
                   "limitations": "Conditional on selected features and filtered historical population; retrospective source availability."}
        write_json(run / "metrics.json", metrics)
        pd.DataFrame(price.segment_metrics(oof, "catboost")).to_csv(run / "segment_metrics.csv", index=False)
        pd.DataFrame(price.year_make_metrics(oof, "catboost")).to_csv(run / "year_make_metrics.csv", index=False)
        files = ["folds.parquet", "eligibility.csv", "oof_predictions.parquet", "metrics.json",
                 "segment_metrics.csv", "year_make_metrics.csv"]
        manifest["files"] = {name: digest(run / name) for name in files}
        for name, receipt in manifest["stages"].items():
            manifest["files"].update({f"{name}/{file}": sha for file, sha in receipt["files"].items()})
        if environment(options) != current:
            raise ValueError("Inputs changed before completion")
        manifest.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
        write_json(run / "manifest.json", manifest)
        print(f"Completed: {run}\nOOF MAE: ${metrics['pooled']['mae']:,.2f}\nValidate before activation.")


def validate_oof_rows(assignments, oof):
    if assignments.empty or assignments.vin.isna().any() or assignments.vin.duplicated().any():
        raise ValueError("Invalid eligible VIN population")
    if oof.vin.isna().any() or oof.vin.duplicated().any() or set(oof.vin) != set(assignments.vin):
        raise ValueError("OOF must cover every eligible VIN exactly once")
    ensure_prices(assignments.actual_price)
    aligned = oof.set_index("vin").loc[assignments.vin]
    if not np.array_equal(aligned.fold, assignments.fold):
        raise ValueError("OOF prediction belongs to the wrong fold")
    np.testing.assert_allclose(aligned.actual_price, assignments.actual_price, rtol=1e-12, atol=1e-8)
    ensure_prices(aligned.prediction)


def validate_bundle(run, *, replay=False, audit=True):
    run = Path(run).resolve()
    if not run.is_relative_to((ROOT / "MODEL_OUTPUT").resolve()):
        raise ValueError("Bundle must be under project MODEL_OUTPUT")
    manifest = read_json(run / "manifest.json")
    if manifest.get("kind") != KIND or manifest.get("status") != "completed" or manifest.get("model") != "catboost" or manifest.get("feature_set") != "full":
        raise ValueError("Require a completed Full CatBoost OOF bundle")
    options = manifest["options"]
    reduced = bool(options["sample_size"] or options["max_trees"] or options["folds"] != 10)
    if manifest["reduced_budget"] != reduced:
        raise ValueError("Bundle budget receipt differs from its configuration")
    current = environment(manifest["options"])
    for key in ("sources", "packages", "features", "preprocessing"):
        if current[key] != manifest["environment"][key]:
            raise ValueError(f"Bundle {key} differs from the current environment")
    if not replay and current["code"] != manifest["environment"]["code"]:
        receipt = read_json(run / "validation.json")
        if not receipt.get("passed") or receipt.get("manifest_sha256") != digest(run / "manifest.json") or receipt.get("code") != current["code"]:
            raise ValueError("Changed code requires a new successful frozen replay validation")
    verify_files(run, manifest["files"])
    # Runtime consumers verify the hash-bound audit receipt instead of repeating
    # the full population/fold audit inside an interactive recommendation turn.
    if not audit and not replay:
        receipt = read_json(run / "validation.json")
        if not receipt.get("passed") or receipt.get("manifest_sha256") != digest(run / "manifest.json") or receipt.get("code") != current["code"]:
            raise ValueError("Run the validation command before consuming this bundle")
        return manifest
    assignments = pd.read_parquet(run / "folds.parquet")
    oof = pd.read_parquet(run / "oof_predictions.parquet")
    validate_oof_rows(assignments, oof)
    if not np.array_equal(assignments.fold, assign_folds(assignments, manifest["options"]["folds"])):
        raise ValueError("Saved fold assignments differ from the deterministic split")
    expected = {f"fold_{i:02d}" for i in range(manifest["options"]["folds"])} | {"final"}
    if set(manifest["stages"]) != expected:
        raise ValueError("Bundle is missing a fold or the final predictor")
    if len(assignments) != manifest["rows"] or digest(run / "folds.parquet") != manifest["folds_sha256"]:
        raise ValueError("Eligible population changed")
    eligibility = pd.read_csv(run / "eligibility.csv")
    if eligibility.loc[eligibility.status.eq("retained"), "eligible_vins"].sum() != len(assignments):
        raise ValueError("Eligible population does not reconcile to the source cohort audit")
    for name in sorted(expected):
        stage = run / name
        receipt = manifest["stages"][name]
        if receipt["status"] != "completed":
            raise ValueError("Incomplete stage")
        verify_files(stage, receipt["files"])
        roles = pd.read_parquet(stage / "selection_split.parquet")
        if roles.vin.duplicated().any() or set(roles.vin) != set(assignments.vin):
            raise ValueError("Invalid inner selection population")
        if not set(roles.role).issubset({"inner_fit", "inner_validation", "outer_holdout"}):
            raise ValueError("Invalid selection role")
        holdout = set() if name == "final" else set(assignments.loc[assignments.fold.eq(int(name[-2:])), "vin"])
        if set(roles.loc[roles.role.eq("outer_holdout"), "vin"]) != holdout:
            raise ValueError("Outer fold leaked into training or selection")
        if roles.loc[roles.role.eq("outer_holdout"), "tuning_sample"].any():
            raise ValueError("Outer fold leaked into tuning")
        if receipt["training_rows"] != len(assignments) - len(holdout) or receipt["heldout_rows"] != len(holdout):
            raise ValueError("Stage row counts differ")
        if not {"inner_fit", "inner_validation"}.issubset(set(roles.role)):
            raise ValueError("Selection must have disjoint fit and validation rows")
        training = np.flatnonzero(~assignments.vin.isin(holdout))
        fit, validation = inner_split(assignments, training, name)
        if (set(roles.loc[roles.role.eq("inner_fit"), "vin"]) != set(assignments.iloc[fit].vin)
                or set(roles.loc[roles.role.eq("inner_validation"), "vin"]) != set(assignments.iloc[validation].vin)):
            raise ValueError("Inner selection roles differ from the recorded split protocol")
        if name != "final":
            part = pd.read_parquet(stage / "predictions.parquet")
            pd.testing.assert_frame_equal(part.sort_values("vin").reset_index(drop=True),
                oof.loc[oof.fold.eq(int(name[-2:]))].sort_values("vin").reset_index(drop=True))
            if not part.predictor_sha256.eq(digest(stage / "predictor.joblib")).all():
                raise ValueError("OOF predictor provenance differs")
            reference = pd.read_parquet(stage / "replay_predictions.parquet")
            if reference.vin.duplicated().any() or not set(reference.vin).issubset(holdout):
                raise ValueError("Fold replay references must belong to its excluded fold")
            np.testing.assert_allclose(reference.prediction,
                part.set_index("vin").loc[reference.vin, "prediction"], rtol=1e-7, atol=.01)
        if replay:
            replay_stage(stage, manifest)
    report = read_json(run / "metrics.json")
    check_metrics(report["pooled"], oof)
    if len(report["folds"]) != options["folds"] or {r["fold"] for r in report["folds"]} != set(range(options["folds"])):
        raise ValueError("Fold metric coverage differs")
    for row in report["folds"]:
        part = oof.loc[oof.fold.eq(row["fold"])]
        if row["rows"] != len(part):
            raise ValueError("Fold metric row count differs")
        check_metrics(row, part)
    if environment(manifest["options"]) != current:
        raise ValueError("Inputs changed during validation")
    return manifest


def check_metrics(metrics, predictions):
    measured = price.regression_metrics(predictions.actual_price, predictions.prediction)
    for key, value in measured.items():
        if value is None:
            if metrics[key] is not None:
                raise ValueError(f"Undefined metric differs: {key}")
        else:
            np.testing.assert_allclose(metrics[key], value, rtol=1e-10, atol=1e-8)


def replay_stage(stage, manifest):
    reference = pd.read_parquet(stage / "replay_predictions.parquet")
    if reference.empty or reference.vin.isna().any() or reference.vin.duplicated().any():
        raise ValueError("Frozen replay needs unique, nonempty VIN references")
    requested = pl.DataFrame({"vin": reference.vin.tolist()})
    raw = pl.scan_parquet(FINAL_DATASET_PATH / "used_cars.parquet").filter(pl.col("vin").is_in(reference.vin.tolist())).collect()
    raw = requested.join(raw, on="vin", how="left", validate="1:1", maintain_order="left")
    prepared = price.prepare_prediction_frame("full", raw,
        youtube_feature_dir=Path(manifest["options"]["youtube_feature_dir"]),
        expert_feature_dir=Path(manifest["options"]["expert_feature_dir"]))
    if prepared["vin"].to_list() != reference.vin.tolist():
        raise ValueError("Replay VIN order changed")
    inputs = prepared.select(FEATURES).to_pandas()
    pd.testing.assert_frame_equal(inputs, pd.read_parquet(stage / "replay_raw.parquet"),
                                  check_dtype=False, check_categorical=False, rtol=1e-12, atol=1e-12)
    predictor = joblib.load(stage / "predictor.joblib")
    if list(predictor.raw_feature_columns_) != FEATURES:
        raise ValueError("Saved predictor feature order differs")
    pd.testing.assert_frame_equal(predictor.named_steps["input"].transform(inputs),
        pd.read_parquet(stage / "replay_prepared.parquet"), check_dtype=False, check_categorical=False, rtol=1e-12, atol=1e-12)
    predictions = predictor.predict(inputs)
    ensure_prices(predictions)
    np.testing.assert_allclose(predictions, reference.prediction, rtol=1e-7, atol=.01)
    del predictor
    gc.collect()


def validate(run):
    manifest = validate_bundle(run, replay=True)
    receipt = {"passed": True, "kind": KIND, "manifest_sha256": digest(run / "manifest.json"),
               "code": code_hashes(), "rows": manifest["rows"], "folds": manifest["options"]["folds"],
               "replay": "All saved fold/final reference rows; raw and transformed inputs; USD rtol=1e-7, atol=0.01"}
    write_json(run / "validation.json", receipt)
    print(f"Validated: {run}")


def inference_rows(run, manifest, listings):
    if listings.is_empty() or "vin" not in listings.columns or listings["vin"].null_count() or listings["vin"].n_unique() != listings.height:
        raise ValueError("Scoring requires unique nonmissing lowercase vin rows")
    frame = price.prepare_prediction_frame("full", listings,
        youtube_feature_dir=Path(manifest["options"]["youtube_feature_dir"]),
        expert_feature_dir=Path(manifest["options"]["expert_feature_dir"]))
    if frame.height != listings.height or frame["vin"].to_list() != listings["vin"].to_list():
        raise ValueError("Prediction VIN identity/order changed")
    predictor = joblib.load(run / "final/predictor.joblib")
    if list(predictor.raw_feature_columns_) != FEATURES:
        raise ValueError("Final predictor feature contract differs")
    values = predictor.predict(frame.select(FEATURES).to_pandas())
    ensure_prices(values)
    missing = frame.select(pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int32) for c in FEATURES])).to_series()
    vpic_missing = frame.select(pl.any_horizontal([pl.col(c).is_null() for c in price.COHORT_COLUMNS[:3]])).to_series()
    return frame.select("vin", *price.COHORT_COLUMNS).with_columns(
        pl.Series("predicted_price_usd", values),
        pl.lit("supplemental_inference").alias("prediction_kind"),
        pl.lit(digest(run / "final/predictor.joblib")).alias("predictor_sha256"),
        missing.alias("missing_feature_count"),
        vpic_missing.alias("decoded_identity_missing"),
        pl.lit("Historical-price model; saved enrichment only; missing fields use fitted imputation. Not a held-out value signal.").alias("limitation"))


def predict(run, input_path, output):
    manifest = validate_bundle(run, audit=False)
    listings = pl.read_parquet(input_path)
    if output.exists():
        raise ValueError("Prediction output already exists")
    training = pl.scan_parquet(run / "folds.parquet").select("vin").collect()["vin"]
    if "vin" not in listings.columns or listings["vin"].is_in(training.implode()).any():
        raise ValueError("New-vehicle scoring requires VINs outside training; use saved OOF predictions for training VINs")
    result = inference_rows(run, manifest, listings)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output)
    print(f"Saved {result.height} supplemental historical-price estimates: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", action="store_true")
    for name in ("folds", "tuning-rows", "max-trees", "sample-size"):
        training.add_argument("--" + name, type=int)
    training.add_argument("--device", choices=("auto", "cpu", "gpu"))
    training.add_argument("--youtube-feature-dir", type=Path)
    training.add_argument("--expert-feature-dir", type=Path)
    validation = commands.add_parser("validate")
    validation.add_argument("--run", type=Path, required=True)
    scoring = commands.add_parser("predict")
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--input", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "train":
        train(args)
    elif args.command == "validate":
        validate(args.run.resolve())
    else:
        predict(args.run.resolve(), args.input, args.output)


if __name__ == "__main__":
    main()
