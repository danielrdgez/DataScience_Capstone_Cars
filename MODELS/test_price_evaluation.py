"""Focused checks for dollar-scale metrics and matched training weights."""

import contextlib
import io
import json
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import polars as pl

from MODELS import Price_ML_Models as price
from NHTSA import text_features as text


class PriceEvaluationTests(unittest.TestCase):
    def test_full_suite_forwards_gpu_sampling_and_unlimited_tree_options(self):
        from unittest.mock import MagicMock

        args = Namespace(model_types=["base", "enhanced", "enhanced_safety", "full"],
                         price_weightings=["none"], models=list(price.AVAILABLE_MODEL_NAMES),
                         sample_size=0, tuning_rows=200000, max_trees=0,
                         split_strategy="cohort_stratified", device="auto",
                         full_validation=True, validation_only=False, shap_rows=0)
        process = MagicMock()
        process.stdout = []
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory(dir=price.PROJECT_ROOT) as directory, \
             patch.object(price, "MODEL_OUTPUT_DIR", Path(directory)), \
             patch("subprocess.Popen", return_value=process) as launch, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(price.run_price_forecast_suite(args), 0)
        self.assertEqual(launch.call_count, 4)
        for call in launch.call_args_list:
            command = call.args[0]
            for flag, expected in (("--sample-size", "0"), ("--max-trees", "0"),
                                   ("--tuning-rows", "200000"), ("--device", "auto")):
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertIn("--full-validation", command)
            variant = command[command.index("--model-type") + 1]
            self.assertEqual("pcr" in command, variant == "full")

    def test_cohort_tuning_quotas_cover_rare_cohorts_and_expand(self):
        groups = np.array(["rare"] * 15 + ["middle"] * 100 + ["large"] * 600)
        vins = np.array([f"vin-{i}" for i in range(len(groups))])
        indices = np.arange(len(groups))
        selected, info = price.cohort_covered_sample(indices, groups, 100, 20, vins)
        counts = pd.Series(groups[selected]).value_counts()
        self.assertEqual(len(selected), 100)
        self.assertEqual(counts["rare"], 15)
        self.assertGreaterEqual(counts.min(), 15)
        self.assertGreater(counts["large"], counts["middle"])
        expanded, info = price.cohort_covered_sample(indices, groups, 10, 20, vins)
        self.assertEqual(len(expanded), 55)
        self.assertTrue(info["expanded_for_coverage"])
        self.assertEqual(info["cohorts"], 3)
        order = np.random.default_rng(7).permutation(indices)
        reordered, _ = price.cohort_covered_sample(indices, groups[order], 100, 20, vins[order])
        self.assertEqual(set(vins[selected]), set(vins[order][reordered]))
        subset = indices[::2]
        bounded, _ = price.cohort_covered_sample(subset, groups, 100, 5, vins)
        self.assertTrue(set(bounded) <= set(subset))

    def test_gpu_parameters_and_explicit_fallback(self):
        self.assertEqual(price._make_tree_model("catboost", 2, device="gpu").get_params()["task_type"], "GPU")
        self.assertEqual(price._make_tree_model("xgboost", 2, device="gpu").get_params()["device"], "cuda:0")
        self.assertEqual(price._make_tree_model("lightgbm", 2, device="gpu").get_params()["device_type"], "gpu")
        with patch.object(price, "_make_tree_model", side_effect=RuntimeError("unsupported build")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = price.resolve_training_devices(["lightgbm", "ridge"], "auto")
            self.assertEqual(result["lightgbm"]["device"], "cpu")
            self.assertIn("unsupported build", result["lightgbm"]["reason"])
            self.assertEqual(result["ridge"]["device"], "cpu")
            with self.assertRaisesRegex(RuntimeError, "unsupported build"):
                price.resolve_training_devices(["lightgbm"], "gpu")

    def test_boosting_unlimited_budget_expands_until_patience_completes(self):
        from unittest.mock import MagicMock

        budgets = []
        def make_model(name, iterations, **kwargs):
            budgets.append(iterations)
            model = MagicMock()
            model.best_iteration_ = 2 if iterations == 2 else 1
            model.evals_result_ = {"valid_0": {"l1": [1.] * iterations}}
            model.predict.return_value = np.full(5, 9.)
            return model
        diagnostics = []
        with patch.object(price, "_make_tree_model", side_effect=make_model), \
             patch.object(price, "PILOT_MAX_TREES", 2), \
             patch.object(price, "TREE_EARLY_STOPPING_ROUNDS", 2), \
             patch.dict(price.TREE_PARAMETER_CANDIDATES, {"lightgbm": [{}]}), \
             patch.dict(price.TREE_REFINEMENT_OVERRIDES, {"lightgbm": []}):
            result = price._fit_with_early_stopping("lightgbm", np.ones((5, 1)), np.full(5, 9.),
                np.ones((5, 1)), np.full(5, 9.), [], max_trees=0, diagnostics=diagnostics)
        self.assertEqual(budgets, [2, 4])
        self.assertFalse(result[3])
        self.assertEqual(len(diagnostics[-1]["budget_attempts"]), 2)

    def test_adaptive_forest_stops_at_validation_plateau(self):
        from unittest.mock import MagicMock

        forest = MagicMock()
        def set_params(**params):
            for key, value in params.items():
                setattr(forest, key, value)
            return forest
        forest.set_params.side_effect = set_params
        forest.fit.side_effect = lambda *args, **kwargs: setattr(forest, "estimators_", [None] * forest.n_estimators)
        forest.predict.return_value = np.full(5, 9.)
        candidate = MagicMock()
        candidate.named_steps = {"preprocess": MagicMock(), "model": forest}
        history = []
        count = price._fit_adaptive_forest(candidate, np.ones((5, 1)), np.full(5, 9.),
            np.ones((5, 1)), np.full(5, 9.), None, history, None)
        self.assertEqual(count, 100)
        self.assertEqual([row["trees"] for row in history], [100, 200, 300, 400])
        self.assertEqual(len(forest.estimators_), 100)
        self.assertFalse(forest.warm_start)

    def test_cohort_stratification_support_membership_and_reproducibility(self):
        rows = []
        for model, count in (("A", 25), ("B", 29), ("C", 24), (None, 30)):
            for i in range(count):
                rows.append({"vin": f"{model}-{i}", "ModelYear": 2020,
                             "Make": "Example" if i % 2 else " EXAMPLE ",
                             "Model": model, "Trim": None if i % 2 else "Sport"})
        source = pd.DataFrame(rows)
        retained, fit, validation, test, details, audit = price.split_stratified_cohorts(source)
        self.assertEqual(len(retained), 54)
        self.assertEqual(details["excluded_rows"], 54)
        self.assertEqual(set(retained.Model), {"A", "B"})
        self.assertEqual((len(fit), len(validation), len(test)), (34, 10, 10))
        sets = [set(retained.iloc[index].vin) for index in (fit, validation, test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(retained.vin))
        for index in (fit, validation, test):
            self.assertEqual(set(retained.iloc[index].Model), {"A", "B"})
        shuffled, *results = price.split_stratified_cohorts(source.sample(frac=1, random_state=7))
        self.assertEqual(sets, [set(shuffled.iloc[index].vin) for index in results[:3]])
        self.assertEqual(set(audit.status), {"retained", "fewer_than_25_vins", "missing_decoded_identity"})
        with self.assertRaisesRegex(ValueError, "at least 25"):
            price.split_stratified_cohorts(source[source.Model == "C"])
        with self.assertRaisesRegex(ValueError, "unique"):
            price.split_stratified_cohorts(pd.concat([source, source.iloc[:1]]))

    def test_pcr_fixed_components_passthrough_and_saved_predictions(self):
        from sklearn.pipeline import Pipeline

        rng = np.random.default_rng(31)
        numeric = [f"measurement_{i}" for i in range(36)]
        raw = pd.DataFrame(rng.normal(size=(180, len(numeric))), columns=numeric)
        raw["constant"], raw["empty"] = 4., np.nan
        raw["option_navigation"] = np.arange(len(raw)) % 2
        raw["make_name"] = np.where(np.arange(len(raw)) % 2, "Toyota", "Ferrari")
        numeric += ["constant", "empty", "option_navigation"]
        input_transformer = price.PricePreprocessor(["make_name"], numeric).fit(raw)
        prepared = input_transformer.transform(raw)
        model = price._make_classical_pipeline("pcr", ["make_name"],
            numeric + [f"{c}__missing" for c in numeric])
        model.fit(prepared, 9 + .1 * raw.measurement_0)
        transformer = model.named_steps["preprocess"]
        components = transformer.named_transformers_["components"]
        self.assertEqual(components.pca_.n_components_, 31)
        self.assertEqual(set(components.excluded_constant_features_), {"constant", "empty"})
        self.assertNotIn("option_navigation", components.feature_names_in_)
        self.assertFalse(any(c.endswith("__missing") for c in components.feature_names_in_))
        names = transformer.get_feature_names_out().tolist()
        self.assertEqual(sum(n.startswith("components__PC") for n in names), 31)
        self.assertIn("numeric__option_navigation", names)
        self.assertIn("numeric__empty__missing", names)
        means = components.scaler_.mean_.copy()
        unseen = raw.iloc[:5].copy()
        unseen["make_name"], unseen["measurement_0"] = "Unseen", 1e6
        bundle = Pipeline([("input", input_transformer), ("estimator", price.ExpLogPricePredictor(model))])
        # Keep predictions numerically bounded while testing held-out transformation.
        components.transform(input_transformer.transform(unseen)[components.feature_names_in_])
        np.testing.assert_array_equal(means, components.scaler_.mean_)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pcr.joblib"
            joblib.dump(bundle, path)
            np.testing.assert_allclose(joblib.load(path).predict(raw.iloc[:5]), bundle.predict(raw.iloc[:5]))
        with self.assertRaisesRegex(ValueError, "31 training rows and varying"):
            price.FixedNumericPCA(31).fit(raw[["measurement_0"]])
        with self.assertRaisesRegex(ValueError, "only --model-type full"):
            price.run_price_model_pilot(device="cpu", model_type="base", model_names=["pcr"])

    def test_pcr_saved_run_and_component_coefficient_diagnostics(self):
        from EDA import DATA_CLEANING
        from MODELS.price_model_diagnostics import explain_checkpoint

        rng = np.random.default_rng(93)
        n = 240
        columns = [f"measurement_{i}" for i in range(36)]
        values = rng.normal(size=(n, len(columns)))
        frame = pl.DataFrame({c: values[:, i] for i, c in enumerate(columns)}).with_columns(
            pl.Series("vin", [f"vin{i}" for i in range(n)]),
            pl.Series("price", np.exp(9 + .1 * values[:, 0])),
            pl.lit(1000.).alias("mileage"), pl.lit("Toyota").alias("make_name"),
            pl.lit(False).alias("is_new"), pl.lit(2020).alias("ModelYear"),
            pl.lit("TOYOTA").alias("Make"), pl.lit("BASE").alias("Trim"),
            pl.Series("Model", [f"MODEL{i // 30}" for i in range(n)]),
            pl.Series("_cohort_group", [f"family{i // 30}" for i in range(n)]),
            pl.lit(True).alias("cohort_complete"), pl.lit("vpic").alias("split_identity_source"),
            pl.lit(False).alias("mileage_review_flag"), pl.lit(False).alias("price_review_flag"))
        with tempfile.TemporaryDirectory(dir=price.PROJECT_ROOT) as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name in ("used_cars", "NHTSA_vPIC", "NHTSA_Complaints", "NHTSA_Recalls", "NHTSA_Safety"):
                (source / f"{name}.parquet").write_bytes(b"synthetic")
            with patch.object(DATA_CLEANING, "FINAL_DATASET_PATH", source), \
                 patch.object(price, "MODEL_OUTPUT_DIR", root / "runs"), \
                 patch.object(price, "EXPERT_MODEL_TYPES", set()), \
                 patch.object(price, "YOUTUBE_MODEL_TYPES", set()), \
                 patch.object(price, "load_model_frame", return_value=(frame, {"feature_columns": columns + ["make_name"]})), \
                 contextlib.redirect_stdout(io.StringIO()):
                run = price.run_price_model_pilot(device="cpu", model_type="full", sample_size=n,
                    model_names=["pcr"], tuning_rows=80, full_validation=True)
            metadata = json.loads((run / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(set(metadata["models"]), {"pcr"})
            self.assertEqual(metadata["pcr_protocol"]["n_components"], 31)
            self.assertEqual(metadata["split_strategy"], "cohort_stratified")
            self.assertTrue(metadata["full_validation"])
            self.assertEqual(metadata["validation_evaluation_rows"], 48)
            self.assertEqual(len(pd.read_parquet(run / "pcr_validation_predictions.parquet")), 48)
            self.assertTrue((pd.read_parquet(run / "tuning_manifest.parquet").groupby("Model").partition.nunique() == 2).all())
            audit = pd.read_csv(run / "cohort_split_audit.csv")
            self.assertTrue((audit.eligible_vins == 30).all())
            self.assertEqual(tuple(audit[["fit_vins", "validation_vins", "test_vins"]].sum()), (144, 48, 48))
            manifest = pd.read_parquet(run / "split_manifest.parquet")
            self.assertTrue((manifest.groupby("Model").partition.nunique() == 3).all())
            summary = json.loads((run / "pcr/pca_summary.json").read_text())
            self.assertEqual(len(summary["explained_variance_ratio"]), 31)
            predictor = joblib.load(run / "pcr/predictor.joblib")
            receipt = pd.read_parquet(run / "pcr_test_predictions.parquet")
            rows = frame.to_pandas().set_index("vin").loc[receipt.vin, columns + ["make_name"]]
            np.testing.assert_allclose(predictor.predict(rows), receipt.prediction)
            _, status = explain_checkpoint(run, metadata, "pcr", root / "cache", rows=5)
            self.assertIn("31 unwhitened PCs", status["method"])
            self.assertTrue(pd.read_csv(run / "metrics.csv").r2.notna().all())

    def test_pcr_diagnostics_supplement_is_scoped_and_source_checked(self):
        from MODELS.price_model_diagnostics import discover_checkpoints, MATCH_KEYS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {key: "same" for key in MATCH_KEYS}
            sources = {"used_cars.parquet": "listings", "expert/make_features.parquet": "expert"}
            schema = {"raw_feature_columns": ["value", "category"],
                      "feature_columns": ["value", "category", "value__missing"],
                      "numeric_features": ["value", "value__missing"], "categorical_features": ["category"]}
            for variant, stamp, name, extra in [
                ("base", "01", "ridge", {}), ("full", "02", "ridge", {}),
                ("full", "03", "pcr", {"code_fingerprint": "new-pcr-code", "max_trees": 1,
                    "pcr_protocol": {"version": price.PCR_PROTOCOL_VERSION, "n_components": 31,
                                     "whiten": False, "numerical_candidates": ["value"]}})]:
                run = root / variant / f"{stamp}-full"
                (run / name).mkdir(parents=True)
                metadata = {**protocol, **schema, "model_type": variant, "status": "completed",
                    "source_file_sha256": sources, "models": {name: {"status": "completed"}}, **extra}
                (run / "run_metadata.json").write_text(json.dumps(metadata))
                pd.DataFrame([{"model": name, "mae": 1}]).to_csv(run / "metrics.csv", index=False)
                for file in [f"{name}/predictor.joblib", f"{name}_test_predictions.parquet", f"{name}_validation_predictions.parquet"]:
                    (run / file).touch()
            selected, status, inventory = discover_checkpoints(root)
            self.assertEqual(set(selected), {("base", "ridge"), ("full", "ridge"), ("full", "pcr")})
            self.assertEqual(status.query("model == 'pcr'").feature_set.tolist(), ["full"])
            self.assertIn("PCR-only", inventory.iloc[-1].reason)
            pcr_path = root / "full/03-full/run_metadata.json"
            original = json.loads(pcr_path.read_text())
            for key, changed in [("source_file_sha256", {**sources, "expert/make_features.parquet": "changed"}),
                                 ("raw_feature_columns", ["different"]), ("target_transform", "changed"),
                                 ("package_versions", "changed")]:
                pcr_path.write_text(json.dumps({**original, key: changed}))
                self.assertNotIn(("full", "pcr"), discover_checkpoints(root)[0], key)

    def test_streamed_forest_shap_matches_whole_ensemble(self):
        import shap
        from sklearn.pipeline import Pipeline
        from MODELS.price_model_diagnostics import explain_checkpoint

        inputs = pd.DataFrame({"value": np.linspace(0, 1, 40)})
        model = price._make_classical_pipeline("random_forest", [], ["value"])
        model.set_params(model__n_estimators=3, model__max_depth=3, model__min_samples_leaf=2)
        model.fit(inputs, 9 + inputs.value ** 2)
        encoded = model.named_steps["preprocess"].transform(inputs)
        reference = shap.TreeExplainer(model.named_steps["model"],
                                       feature_perturbation="tree_path_dependent").shap_values(encoded[:10])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "random_forest").mkdir()
            bundle = Pipeline([("estimator", price.ExpLogPricePredictor(model))])
            joblib.dump(bundle, run / "random_forest" / "predictor.joblib")
            inputs.to_parquet(run / "inputs.parquet")
            inputs.iloc[:20].to_parquet(run / "background.parquet")
            pd.DataFrame({"vin": [f"v{i}" for i in range(40)]}).to_parquet(run / "receipt.parquet")
            metadata = {"shap_inputs": "inputs.parquet", "shap_background": "background.parquet",
                        "shap_receipt": "receipt.parquet", "feature_columns": ["value"]}
            importance, status = explain_checkpoint(run, metadata, "random_forest", run / "cache", rows=10,
                                                    forest_method="shap")
            saved = pd.read_parquet(Path(status["cache"]) / "shap_values.parquet")
            # SHAP's native ensemble scaling introduces float32 rounding.
            np.testing.assert_allclose(saved[["numeric__value"]], reference, rtol=1e-5, atol=1e-6)
            self.assertEqual(status["status"], "completed")
            self.assertEqual(importance.feature.tolist(), ["numeric__value"])
            global_importance, global_status = explain_checkpoint(
                run, metadata, "random_forest", run / "cache", rows=10, forest_method="impurity")
            self.assertAlmostEqual(global_importance.value.iloc[0], 1)
            self.assertIn("impurity", global_status["method"])
            self.assertIsNone(global_status["sample_rows"])

    def test_diagnostics_make_metrics_preserve_denominators_and_undefined_r2(self):
        from MODELS.price_model_diagnostics import receipt_identity, summarize_makes

        receipt = pd.DataFrame({
            "vin": ["a", "b", "c", "d"], "Make": ["TOYOTA", "TOYOTA", "FERRARI", None],
            "family_block": ["t1", "t2", "f1", "unknown"],
            "actual_price": [100., 200., 1000., 50.], "prediction": [90., 220., 800., 60.],
        })
        scores = summarize_makes(receipt).set_index("Make")
        self.assertEqual(scores.rows.sum(), len(receipt))
        self.assertEqual(scores.loc["TOYOTA", "mean_actual_price"], 150)
        self.assertEqual(scores.loc["TOYOTA", "mse"], 250)
        self.assertAlmostEqual(scores.loc["TOYOTA", "r2"], .9)
        self.assertTrue(pd.isna(scores.loc["FERRARI", "r2"]))
        self.assertIn("(missing decoded Make)", scores.index)
        self.assertEqual(receipt_identity(receipt), receipt_identity(receipt.iloc[::-1]))
        changed = receipt.copy()
        changed.loc[0, "actual_price"] += 1
        self.assertNotEqual(receipt_identity(receipt), receipt_identity(changed))
        with self.assertRaisesRegex(ValueError, "lowercase vin"):
            receipt_identity(pd.concat([receipt, receipt.iloc[:1]]))

    def test_diagnostics_discover_completed_checkpoint_inside_running_run(self):
        from MODELS.price_model_diagnostics import discover_checkpoints, MATCH_KEYS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {key: "same" for key in MATCH_KEYS}
            for variant, run_status, extra_source in [("base", "completed", False), ("full", "running", True)]:
                run = root / variant / "20261024-full"
                (run / "ridge").mkdir(parents=True)
                sources = {"used_cars.parquet": "listing-hash", "NHTSA_vPIC.parquet": "vpic-hash"}
                if extra_source:
                    sources["expert/make_features.parquet"] = "expert-hash"
                metadata = {**protocol, "model_type": variant, "status": run_status,
                            "dataset_fingerprint": variant, "source_file_sha256": sources,
                            "models": {"ridge": {"status": "completed"}, "lasso": {"status": "running"}}}
                (run / "run_metadata.json").write_text(json.dumps(metadata))
                pd.DataFrame([{"model": "ridge", "mae": 1}]).to_csv(run / "metrics.csv", index=False)
                for file in ["ridge/predictor.joblib", "ridge_test_predictions.parquet", "ridge_validation_predictions.parquet"]:
                    (run / file).touch()
            selected, status, inventory = discover_checkpoints(root)
            self.assertEqual(set(selected), {("base", "ridge"), ("full", "ridge")})
            self.assertTrue(inventory.compatible.all())
            self.assertEqual(status.query("feature_set == 'full' and model == 'lasso'").status.iloc[0], "pending / unavailable")
            run = root / "full" / "20261024-full"
            metadata["source_file_sha256"]["used_cars.parquet"] = "changed"
            (run / "run_metadata.json").write_text(json.dumps(metadata))
            selected, _, _ = discover_checkpoints(root)
            self.assertNotIn(("full", "ridge"), selected)

    def test_dollar_r2_and_undefined_segments(self):
        result = price.regression_metrics([100, 200, 300], [100, 150, 250])
        self.assertAlmostEqual(result["r2"], .75)
        self.assertAlmostEqual(result["wape_pct"], 100 / 600 * 100)
        self.assertAlmostEqual(result["mean_signed_error"], -100 / 3)
        self.assertIsNone(price.regression_metrics([100], [90])["r2"])
        self.assertIsNone(price.regression_metrics([100, 100], [90, 90])["r2"])
        self.assertLess(price.regression_metrics([100, 200], [500, 500])["r2"], 0)

    def test_capped_weights_and_high_price_segments(self):
        actual = np.array([10_000, 50_000, 100_000, 200_000, 1_000_000.])
        original = actual.copy()
        self.assertIsNone(price.training_price_weights(actual, "none"))
        np.testing.assert_array_equal(price.training_price_weights(actual, "high_price"), [1, 1, 2, 4, 5])
        np.testing.assert_array_equal(actual, original)
        with self.assertRaises(ValueError):
            price.training_price_weights([np.nan], "high_price")
        receipt = pd.DataFrame({"actual_price": actual, "prediction": actual * .8,
                                "make_name": ["Ferrari"] * 5, "split_group": list("abcde")})
        segments = price.segment_metrics(receipt, "model")
        self.assertIn("1000k-1025k", {r["segment"] for r in segments if r["dimension"] == "price_band"})
        make = next(r for r in segments if r["dimension"] == "make_name")
        self.assertEqual(make["support_status"], "low_support")
        self.assertAlmostEqual(make["wape_pct"], 20)
        self.assertIn("r2", make)

    def test_price_band_boundaries_and_all_make_reports(self):
        actual = [0, 24_999, 25_000, 50_000, 75_000, 99_999, 100_000, 124_999, 125_000, 3_299_995]
        receipt = pd.DataFrame({"actual_price": actual, "prediction": actual,
                                "Make": ["TOYOTA", "HONDA"] * 5, "split_group": list(range(10))})
        dimensions = price.evaluation_dimensions(receipt)
        self.assertEqual(dimensions.price_band.tolist(), ["0k-25k", "0k-25k", "25k-50k", "50k-75k", "75k-100k",
                                                        "75k-100k", "100k-125k", "100k-125k", "125k-150k", "3275k-3300k"])
        self.assertEqual((dimensions.price_tier == "100k+").sum(), 4)
        scores = price.make_price_metrics(receipt, "ridge", makes=["TOYOTA", "HONDA", "BMW"])
        self.assertEqual({row["Make"] for row in scores}, {"TOYOTA", "HONDA", "BMW"})
        absent = [row for row in scores if row["Make"] == "BMW"]
        self.assertEqual(len(absent), 3)
        self.assertTrue(all(row["support_status"] == "no_rows" and row["mae"] is None for row in absent))
        high = [row for row in scores if row["dimension"] == "price_tier" and row["price_tier"] == "100k+"]
        self.assertEqual(sum(row["rows"] for row in high), 4)
        self.assertEqual(receipt.columns.tolist(), ["actual_price", "prediction", "Make", "split_group"])

    def test_youtube_join_preserves_listing_grain_missingness_and_identity(self):
        listings = pl.DataFrame({"vin": ["a", "b", "c", "d"], "Make": ["Toyota", " TOYOTA ", "HONDA", None],
                                 "ModelYear": [2020] * 4, "Model": ["X"] * 4, "Trim": ["BASE"] * 4})
        features = pl.DataFrame({"Make": ["TOYOTA"], "yt_feature_row_present": [1], "yt_comment_count": [100],
                                "yt_support_suppressed": [False], "yt_semantic_missing": [False], "yt_semantic_01": [.2]})
        joined = text.join_youtube_features(listings, features)
        self.assertEqual(joined.height, listings.height)
        self.assertEqual(joined["vin"].to_list(), listings["vin"].to_list())
        self.assertEqual(joined["Make"].to_list(), listings["Make"].to_list())
        self.assertEqual(joined["yt_feature_row_present"].to_list(), [1, 1, 0, 0])
        self.assertEqual(joined["yt_support_suppressed"].to_list(), [False, False, True, True])
        self.assertEqual(joined["yt_comment_count"].to_list(), [100, 100, None, None])
        with self.assertRaises(pl.exceptions.ComputeError):
            text.join_youtube_features(listings, pl.concat([features, features]))

    def test_all_eight_models_both_weights_and_frozen_test(self):
        # Exercise orchestration and real estimator APIs with small search grids.
        rng = np.random.default_rng(42)
        n = 240
        value = rng.uniform(0, 1, n)
        partitions = np.repeat(["fit", "validation", "test"], [120, 60, 60])
        groups = [f"{part}_{i // 4}" for i, part in enumerate(partitions)]
        frame = pl.DataFrame({
            "vin": [f"vin{i:014d}" for i in range(n)], "value": value,
            "price": np.exp(9 + 4 * value + rng.normal(0, .1, n)),
            "make_name": ["Ferrari" if i % 3 == 0 else "Toyota" for i in range(n)],
            "ModelYear": [2020] * n, "Make": ["FERRARI" if i % 3 == 0 else "TOYOTA" for i in range(n)],
            "Model": ["MODEL"] * n, "Trim": ["BASE"] * n,
            "is_new": [False] * n, "cohort_complete": [True] * n,
            "split_identity_source": ["vpic"] * n, "mileage_review_flag": [False] * n,
            "price_review_flag": [False] * n, "partition": partitions,
            "split_group": groups, "_cohort_group": groups,
        })
        tree_grid = {name: candidates[:1] for name, candidates in price.TREE_PARAMETER_CANDIDATES.items()}
        classical_grid = {name: {} for name in price.CLASSICAL_MODEL_NAMES}
        classical_grid["random_forest"] = {"model__n_estimators": [5]}
        with tempfile.TemporaryDirectory(prefix="price-evaluation-", dir=price.PROJECT_ROOT) as temporary:
            workspace = Path(temporary).resolve()
            self.assertEqual(workspace.parent, price.PROJECT_ROOT.resolve())
            build = workspace / "features_automatic"
            build.mkdir()
            frame.write_parquet(workspace / "listings.parquet")
            pl.DataFrame({"unused": [1]}).write_parquet(workspace / "records.parquet")
            features = text.normalized_keys(frame.select(text.KEYS)).unique().with_columns(pl.lit(1).alias("synthetic_feature"))
            features.write_parquet(build / "family_features.parquet")
            manifest = {"version": text.VERSION, "script_sha256": price.file_sha256(Path(text.__file__)),
                        "price_script_sha256": price.file_sha256(Path(price.__file__)),
                        "files_sha256": {name: price.file_sha256(workspace / name) for name in ["listings.parquet", "records.parquet"]},
                        "base_features": ["value", "make_name"], "data_info": {"vpic_feature_columns": []},
                        "interpretation": "synthetic", "as_of_date": "2020-12-31", "source_sha256": {},
                        "split_strategy": "unseen_family"}
            text.write_json(workspace / "workflow.json", manifest)
            text.write_json(build / "feature_manifest.json", {
                "features_sha256": price.file_sha256(build / "family_features.parquet"), "label_sha256": None,
                "preparation_script_sha256": manifest["script_sha256"], "script_sha256": manifest["script_sha256"],
                "component_columns": [], "text_columns": [], "method": "automatic"})
            args = Namespace(workspace=workspace, build_name=build.name, evaluate_test=False,
                             models=list(price.PILOT_MODEL_NAMES), max_trees=12, tuning_rows=1000,
                             representation="none", vector_dimensions=16, vector_training_documents=100,
                             embedding_model=None, feature_sets=["base"], price_weightings=["none", "high_price"])
            with patch.dict(price.TREE_PARAMETER_CANDIDATES, tree_grid), \
                 patch.dict(price.TREE_REFINEMENT_OVERRIDES, {name: [] for name in price.TREE_MODEL_NAMES}), \
                 patch.dict(price.CLASSICAL_PARAMETER_GRIDS, classical_grid), \
                 patch.object(text, "comparison_feature_sets", return_value={"base": ["value", "make_name"]}), \
                 contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                text.compare(args)
                chosen = json.loads((build / "selected_comparison.json").read_text())
                output = build / chosen["output_directory"]
                scores = pd.read_csv(output / "validation_metrics.csv")
                self.assertEqual(len(scores), 16)
                self.assertTrue(scores.validation_r2.notna().all())
                self.assertFalse((workspace / "test_evaluated.json").exists())
                pairs = pd.read_csv(output / "validation_weighting_comparison.csv")
                self.assertEqual(len(pairs[pairs.dimension == "overall"]), 8)
                args.evaluate_test = True
                args.price_weightings = ["none"]
                with self.assertRaisesRegex(ValueError, "exact settings"):
                    text.compare(args)
                self.assertFalse((workspace / "test_evaluated.json").exists())
                args.price_weightings = ["none", "high_price"]
                text.compare(args)
                test_scores = pd.read_csv(output / "test_metrics.csv")
                self.assertEqual(len(test_scores), 16)
                self.assertTrue(test_scores.r2.notna().all())
                test_frame = frame.filter(pl.col("partition") == "test").to_pandas()
                for model in price.PILOT_MODEL_NAMES:
                    for mode in price.PRICE_WEIGHTING_MODES:
                        key = f"base_{model}_{mode}"
                        predictor = joblib.load(output / f"{key}_predictor.joblib")
                        predictions = predictor.predict(test_frame[["value", "make_name"]])
                        saved = pd.read_parquet(output / f"{key}_test_predictions.parquet")
                        np.testing.assert_allclose(predictions, saved.prediction)
                self.assertEqual(json.loads((workspace / "test_evaluated.json").read_text())["status"], "complete")
                with self.assertRaisesRegex(ValueError, "already exposed"):
                    text.compare(args)

                # Exercise all six nested feature blocks, frozen YouTube inputs,
                # saved predictors, and the one-time refit path together.
                youtube_workspace = workspace / "youtube_comparison"
                youtube_build = youtube_workspace / build.name
                youtube_build.mkdir(parents=True)
                for name in ["listings.parquet", "records.parquet", "workflow.json"]:
                    (youtube_workspace / name).write_bytes((workspace / name).read_bytes())
                for name in ["family_features.parquet", "feature_manifest.json"]:
                    (youtube_build / name).write_bytes((build / name).read_bytes())
                selected_youtube = youtube_workspace / "selected_youtube"
                selected_youtube.mkdir()
                youtube_frame = pl.DataFrame({"Make": ["TOYOTA", "FERRARI"], "yt_feature_row_present": [1, 1],
                    "yt_support_suppressed": [False, False], "yt_semantic_01": [.3, .7],
                    "yt_reliability_video_balanced_net": [None, 0.]} )
                youtube_frame.write_parquet(selected_youtube / "make_features.parquet")
                youtube_manifest = {"state": "complete", "attribution_scope": "explicit_comment", "window": "cumulative",
                    "as_of": "2026-10-01", "features_sha256": price.file_sha256(selected_youtube / "make_features.parquet"),
                    "feature_blocks": {"coverage": ["yt_feature_row_present", "yt_support_suppressed"],
                                       "topics": ["yt_semantic_01"], "sentiment": ["yt_reliability_video_balanced_net"]},
                    "directional_sentiment_available": False, "sentiment_limitation": "Synthetic constant sentiment"}
                text.write_json(selected_youtube / "price_feature_manifest.json", youtube_manifest)
                vectors = features.select(text.KEYS).with_columns(pl.lit(.5).alias("tfidf_complaint_00"))
                youtube_args = Namespace(**vars(args))
                youtube_args.workspace, youtube_args.youtube_features = youtube_workspace, selected_youtube
                youtube_args.evaluate_test, youtube_args.models, youtube_args.price_weightings = False, ["ridge"], ["none"]
                youtube_args.representation = "tfidf"
                youtube_args.feature_sets = text.FULL_COMPARISON_SETS + ["nhtsa_text_youtube_coverage", "nhtsa_text_youtube_topics"]
                with patch.object(text, "comparison_feature_sets", return_value={
                    "base": ["value", "make_name"], "enriched": ["value", "make_name"],
                    "tfidf": ["value", "make_name", "tfidf_complaint_00"]}), \
                     patch.object(text, "vector_features", return_value=(vectors, {"synthetic": True})), \
                     warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    text.compare(youtube_args)
                    selected_run = text.read_json(youtube_build / "selected_comparison.json")
                    youtube_output = youtube_build / selected_run["output_directory"]
                    validation_scores = pd.read_csv(youtube_output / "validation_metrics.csv")
                    self.assertEqual(set(validation_scores.variant), set(youtube_args.feature_sets))
                    self.assertEqual(len(validation_scores), 6)
                    youtube_args.evaluate_test = True
                    tampered = {**youtube_manifest, "as_of": "2026-10-02"}
                    text.write_json(selected_youtube / "price_feature_manifest.json", tampered)
                    with self.assertRaisesRegex(ValueError, "exact settings"):
                        text.compare(youtube_args)
                    self.assertFalse((youtube_workspace / "test_evaluated.json").exists())
                    text.write_json(selected_youtube / "price_feature_manifest.json", youtube_manifest)
                    text.compare(youtube_args)
                    self.assertEqual(len(pd.read_csv(youtube_output / "test_metrics.csv")), 6)
                    self.assertTrue((youtube_output / "test_feature_comparison.csv").exists())
                    all_make_report = pd.read_csv(youtube_output / "make_price_metrics.csv")
                    self.assertEqual(set(all_make_report.Make), {"TOYOTA", "FERRARI"})
                    spotlight = pd.read_csv(youtube_output / "focus_make_metrics.csv")
                    self.assertTrue((spotlight[spotlight.Make == "LAMBORGHINI"].support_status == "no_rows").all())
                    assembled = text.join_youtube_features(text.join_family_features(frame, vectors), youtube_frame).to_pandas()
                    for variant in youtube_args.feature_sets:
                        predictor = joblib.load(youtube_output / f"{variant}_ridge_none_predictor.joblib")
                        saved = pd.read_parquet(youtube_output / f"{variant}_ridge_none_test_predictions.parquet")
                        np.testing.assert_allclose(predictor.predict(assembled[assembled.partition == "test"]), saved.prediction)

                # The separate full-price runner must pass the same weights
                # through tuning/refitting and publish validation reports too.
                from EDA import DATA_CLEANING
                source = workspace / "source"
                source.mkdir()
                for name in ["used_cars", "NHTSA_vPIC", "NHTSA_Complaints", "NHTSA_Recalls", "NHTSA_Safety"]:
                    (source / f"{name}.parquet").write_bytes(b"synthetic source fingerprint")
                model_frame = frame.with_columns(pl.lit(1000).alias("mileage"))
                with patch.object(DATA_CLEANING, "FINAL_DATASET_PATH", source), \
                     patch.object(price, "MODEL_OUTPUT_DIR", workspace / "price_runs"), \
                     patch.object(price, "load_model_frame", return_value=(model_frame,
                         {"feature_columns": ["value", "make_name"], "rows_after_filters": n})):
                    regular_run = price.run_price_model_pilot(device="cpu", sample_size=n, model_names=["catboost", "ridge"],
                        shap_rows=0, tuning_rows=1000, max_trees=12, price_weighting="none")
                    weighted_run = price.run_price_model_pilot(device="cpu", sample_size=n, model_names=["catboost", "ridge"],
                        shap_rows=0, tuning_rows=1000, max_trees=12, price_weighting="high_price")
                regular_meta = json.loads((regular_run / "run_metadata.json").read_text())
                weighted_meta = json.loads((weighted_run / "run_metadata.json").read_text())
                self.assertFalse(price.compatible_run_metadata(regular_meta, weighted_meta))
                self.assertTrue(price.compatible_run_metadata(regular_meta, weighted_meta, allow_price_weighting=True))
                self.assertEqual(regular_meta["split_fingerprint"], weighted_meta["split_fingerprint"])
                report = pd.read_csv(weighted_run / "validation_metrics.csv")
                self.assertEqual(len(report), 3)
                self.assertTrue(report.validation_r2.notna().all())
                weighted_scores = pd.read_csv(weighted_run / "metrics.csv")
                self.assertTrue(weighted_scores.r2.notna().all())
                self.assertIn("r2", pd.read_csv(weighted_run / "validation_segment_metrics.csv"))
                # Reuse the actual notebook cell against synthetic completed runs.
                notebook = json.loads((price.PROJECT_ROOT / "MODEL_OUTPUT.ipynb").read_text(encoding="utf-8"))
                cell = next(c for c in notebook["cells"] if c.get("id") == "price-weighting-metrics")
                from IPython.display import Markdown
                scope = {"COMPARISON_MODEL_TYPES": ["base"], "completed_runs": lambda kind: [weighted_run, regular_run],
                         "json": json, "pd": pd, "np": np, "Markdown": Markdown, "display": lambda item: None,
                         "compatible_run_metadata": price.compatible_run_metadata}
                exec(compile("".join(cell["source"]), "weighting-notebook", "exec"), scope)
                self.assertEqual(len(scope["weighting_overview"]), 6)
                self.assertEqual(len(scope["weighting_segments"]), 2)


    def test_staged_models_and_saved_postprocessing(self):
        from EDA import DATA_CLEANING
        n = 120
        rng = np.random.default_rng(41)
        value = rng.uniform(0, 1, n)
        frame = pl.DataFrame({
            "vin": [f"vin{i}" for i in range(n)], "value": value,
            "price": np.exp(9 + value), "mileage": [1000.] * n,
            "make_name": ["Toyota"] * n, "is_new": [False] * n,
            "ModelYear": [2020] * n, "Make": ["TOYOTA"] * n,
            "Model": [f"MODEL{i // 30}" for i in range(n)], "Trim": ["BASE"] * n,
            "_cohort_group": [f"family{i // 30}" for i in range(n)],
            "cohort_complete": [True] * n, "split_identity_source": ["vpic"] * n,
            "mileage_review_flag": [False] * n, "price_review_flag": [False] * n,
        })
        with tempfile.TemporaryDirectory(dir=price.PROJECT_ROOT) as temporary:
            workspace = Path(temporary)
            source = workspace / "source"
            source.mkdir()
            for name in ("used_cars", "NHTSA_vPIC", "NHTSA_Complaints", "NHTSA_Recalls", "NHTSA_Safety"):
                (source / f"{name}.parquet").write_bytes(b"synthetic")
            with patch.object(DATA_CLEANING, "FINAL_DATASET_PATH", source), \
                 patch.object(price, "MODEL_OUTPUT_DIR", workspace / "runs"), \
                 patch.object(price, "load_model_frame", return_value=(frame, {"feature_columns": ["value", "make_name"]})), \
                 patch.dict(price.TREE_PARAMETER_CANDIDATES, {k: v[:1] for k, v in price.TREE_PARAMETER_CANDIDATES.items()}), \
                 patch.dict(price.TREE_REFINEMENT_OVERRIDES, {k: [] for k in price.TREE_MODEL_NAMES}), \
                 patch.dict(price.CLASSICAL_PARAMETER_GRIDS, {k: ({"model__n_estimators": [5]} if k == "random_forest" else {}) for k in price.CLASSICAL_MODEL_NAMES}), \
                 contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                run = price.run_price_model_pilot(device="cpu", sample_size=n, model_names=list(price.PILOT_MODEL_NAMES), max_trees=12, tuning_rows=30, full_validation=True)
                metadata = json.loads((run / "run_metadata.json").read_text())
                self.assertEqual(metadata["status"], "completed")
                self.assertNotIn("selected_by_validation_mae", metadata)
                self.assertEqual(set(metadata["models"]), set(price.PILOT_MODEL_NAMES))
                timings = pd.read_csv(run / "timings.csv")
                self.assertEqual(len(timings), 32)
                for name in price.PILOT_MODEL_NAMES:
                    self.assertTrue((run / name / "selection.json").exists())
                    self.assertTrue((run / name / "predictor.joblib").exists())
                    self.assertTrue((run / f"{name}_validation_predictions.parquet").exists())
                    self.assertIsNone(metadata["models"][name]["shap_files"])
                # Postprocessing must work after the original sources disappear.
                with patch.object(price, "load_model_frame", side_effect=AssertionError("No refitting")):
                    results = price.explain_saved_run(run, rows=4)
                    self.assertEqual({k: v["status"] for k, v in results.items()}, {k: "completed" for k in price.PILOT_MODEL_NAMES})
                    comparison = price.compare_saved_runs([run])
                    self.assertEqual(len(pd.read_csv(comparison / "validation_ranking.csv")), 8)
                # A later estimator failure must preserve the completed predecessor.
                with patch.object(price, "_fit_classical_model", side_effect=RuntimeError("injected failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected failure"):
                        price.run_price_model_pilot(device="cpu", sample_size=n, model_names=["catboost", "ridge"], max_trees=12, tuning_rows=n)
                receipts = list((workspace / "runs/base").glob("*/run_metadata.json"))
                failed = next(path for path in receipts if json.loads(path.read_text())["status"] == "failed")
                self.assertTrue((failed.parent / "catboost/predictor.joblib").exists())
                self.assertEqual(list(json.loads(failed.read_text())["models"]), ["catboost"])
                changed = json.loads((run / "run_metadata.json").read_text())
                changed["split_fingerprint"] = "different"
                other = workspace / "other"
                other.mkdir()
                (other / "run_metadata.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "split_fingerprint"):
                    price.compare_saved_runs([run, other])


class DougPriceIntegrationTests(unittest.TestCase):
    def test_full_schema_and_price_informed_exclusions(self):
        self.assertEqual(set(price.MODEL_FEATURE_SETS['full']),
            set(price.MODEL_FEATURE_SETS['full_youtube']) | set(price.EXPERT_FEATURE_COLUMNS))
        for variant in price.EXPERT_MODEL_TYPES:
            self.assertFalse({'doug_value','doug_daily_total','doug_dougscore'} & set(price.MODEL_FEATURE_SETS[variant]))
        self.assertFalse(set(price.EXPERT_FEATURE_COLUMNS) & set(price.MODEL_FEATURE_SETS['full_youtube']))
        self.assertTrue(set(price.EXPERT_COVERAGE_COLUMNS).issubset(price.MODEL_FEATURE_SETS['full_youtube_expert_coverage']))

    def test_paired_changes_empty_segments_and_identity_checks(self):
        left = pd.DataFrame({'vin':list('abcd'), 'actual_price':[100.,200.,300.,400.],
            'prediction':[90.,180.,270.,360.], 'family_block':['a','a','b','b']})
        right = left.assign(prediction=[95.,190.,285.,380.], doug_source_level=['cohort','cohort','make_average','make_average'])
        result = price.paired_expert_validation(left,right)
        overall = result[0]
        self.assertAlmostEqual(overall['mae_delta'],-12.5)
        self.assertLess(overall['mae_delta_high'],0)
        self.assertIsNone(next(r for r in result if r['doug_source_level']=='missing')['mae_delta'])
        self.assertIsNone(next(r for r in result if r['doug_source_level']=='cohort')['mae_delta_low'])
        with self.assertRaisesRegex(ValueError,'different VINs'):
            price.paired_expert_validation(left,right.iloc[:-1])
        with self.assertRaisesRegex(ValueError,'targets'):
            price.paired_expert_validation(left,right.assign(actual_price=[101,200,300,400]))
        with self.assertRaisesRegex(ValueError,'family'):
            price.paired_expert_validation(left,right.assign(family_block=['z']*4))

    def test_training_and_inference_share_expert_selection(self):
        from EXPERT_REVIEWS import price_features as expert
        from polars.testing import assert_frame_equal
        listings = pl.DataFrame({'vin':['a','b','c'], 'price':[10000,20000,30000],
            'mileage':[100.,200.,300.], 'year':[2020]*3, 'make_name':['Toyota','Toyota','Honda'],
            'model_name':['A','A','X'], 'is_new':[False]*3,
            'major_options':["[]"]*3, 'maximum_seating':['5 seats']*3})
        vpic = pl.DataFrame({'vin':['a','b','c'], 'ModelYear':[2020]*3,
            'Make':['Toyota','Toyota','Honda'], 'Model':['A','A','X'], 'Trim':['Sport','Base',None]})
        for column in price.SUBTYPE_SUPPORT_COLUMNS:
            if column not in vpic.columns:
                vpic = vpic.with_columns(pl.lit(None,dtype=pl.String).alias(column))
        families = vpic.select('ModelYear','Make','Model').unique().with_columns(
            pl.col('Make').str.to_uppercase(),pl.col('Model').str.to_uppercase(),
            pl.lit(0).alias('complaint_source_records_present'),pl.lit(0).alias('recall_source_records_present'))
        row = {'review_record_id':'a','ModelYear':2020,'make_raw':'Toyota','model_raw':'A',
            **{c:6 for c in (*expert.WEEKEND,*expert.DAILY)}, 'weekend_total':30,'daily_total':30,'dougscore':60}
        cohorts,makes = expert.feature_tables(expert.resolve_reviews([row],vpic.to_dicts()))
        selected = ('mileage','year','make_name','model_name','is_new','yt_feature_row_present',*expert.FEATURE_COLUMNS)
        youtube = pl.DataFrame({'Make':['TOYOTA'],'yt_feature_row_present':[1]})
        def scan(name):
            return (listings if name=='used_cars' else vpic).lazy()
        with patch.dict(price.MODEL_FEATURE_SETS,{'full':selected}), \
             patch.object(price,'YOUTUBE_FEATURE_COLUMNS',('yt_feature_row_present',)), \
             patch.object(price,'scan_final_table',side_effect=scan), \
             patch.object(price,'_load_cohort_exclusion_vins',return_value=set()), \
             patch.object(price,'file_sha256',return_value='fixed'), \
             patch.object(price,'build_complaint_recall_features',return_value=(families,{})), \
             patch.object(price,'load_expert_features',return_value=(cohorts,makes,{'directory':'fixture'})), \
             patch.object(text,'load_youtube_features',return_value=(youtube,{})):
            training,info = price.load_model_frame('full',None)
            inference = price.prepare_prediction_frame('full',listings.drop('price'))
            assert_frame_equal(training.select('vin',*selected),inference.select('vin',*selected))
            self.assertEqual(training['doug_source_level'].to_list(),['cohort','cohort','missing'])
            self.assertEqual(training['Trim'].to_list(),['Sport','Base',None])

    def test_inference_requires_fitted_expert_snapshot(self):
        from unittest.mock import Mock
        predictor = Mock()
        predictor.raw_feature_columns_ = list(price.MODEL_FEATURE_SETS['full'])
        predictor.expert_feature_info_ = {'directory':'saved-expert', 'manifest_sha256':'original'}
        with patch.object(joblib,'load',return_value=predictor), \
             patch.object(price,'load_expert_features',return_value=(None,None,{'manifest_sha256':'changed'})), \
             patch.object(price,'prepare_prediction_frame') as prepare:
            with self.assertRaisesRegex(ValueError,'snapshot differs'):
                price.predict_listings('full',Path('predictor.joblib'),pl.DataFrame({'vin':['a']}))
            prepare.assert_not_called()
        predictor.raw_feature_columns_ = list(price.MODEL_FEATURE_SETS['full_youtube'])
        with patch.object(joblib,'load',return_value=predictor):
            with self.assertRaisesRegex(ValueError,'full_youtube'):
                price.predict_listings('full',Path('predictor.joblib'),pl.DataFrame({'vin':['a']}))


if __name__ == "__main__":
    unittest.main()
