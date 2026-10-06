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
                    regular_run = price.run_price_model_pilot(sample_size=n, model_names=["catboost", "ridge"],
                        shap_rows=0, tuning_rows=1000, max_trees=12, price_weighting="none")
                    weighted_run = price.run_price_model_pilot(sample_size=n, model_names=["catboost", "ridge"],
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


if __name__ == "__main__":
    unittest.main()
