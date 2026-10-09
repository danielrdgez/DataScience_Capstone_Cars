"""Focused checks for VIN linkage, evidence grain, errors, and independent UI scope."""
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
from streamlit.testing.v1 import AppTest

from car_app.data import COHORT, LISTING_COLUMNS, ROOT, TABLE_NAMES, Filters, Snapshot, evidence_bundle
from car_app.artifacts import errors
from car_app.state import RecommendationPanel, SimilarityPanel, update_dashboard_scope


def fixture_snapshot(directory):
    numeric = {"price": pl.Int64, "mileage": pl.Float64, "year": pl.Int64,
               "horsepower": pl.Float64, "city_fuel_economy": pl.Float64,
               "highway_fuel_economy": pl.Float64, "daysonmarket": pl.Int64,
               "listed_date": pl.Date}
    listing_schema = {c: numeric.get(c, pl.String) for c in LISTING_COLUMNS}
    rows = []
    for index, (vin, price, mileage) in enumerate([("a", 10000, 20000.), ("b", 20000, 50000.), ("c", 30000, 80000.)]):
        row = {c: None for c in LISTING_COLUMNS}
        row.update(vin=vin, listing_id=str(index), price=price, mileage=mileage,
                   is_new="False", body_type="Sedan", fuel_type="Gasoline", wheel_system="FWD",
                   maximum_seating="5 seats", year=2020, listed_date=date(2020, 1, 1))
        rows.append(row)
    tables = {"used_cars": pl.DataFrame(rows, schema=listing_schema),
              "NHTSA_vPIC": pl.DataFrame({
                  "vin": ["a", "b"], "VIN": ["WRONG_A", "WRONG_B"],
                  "ModelYear": [2020, 2020], "Make": ["TOYOTA", "TOYOTA"],
                  "Model": ["Corolla", "Corolla"], "Trim": ["LE", None]})}
    complaint = {
        "query_id": 1, "record_key": 1, "model_year": 2020, "make": "TOYOTA", "model": "Corolla",
        "odiNumber": 123, "dateComplaintFiled": date(2020, 1, 1), "dateOfIncident": date(2019, 12, 1),
        "summary": "Reported brake issue", "components": "BRAKES",
        "products": "[]", "crash": False, "fire": False, "numberOfInjuries": 0, "numberOfDeaths": 0}
    tables["NHTSA_Complaints"] = pl.DataFrame([
        complaint, {**complaint, "query_id": 2},
        {**complaint, "odiNumber": 456, "dateComplaintFiled": date(2021, 1, 1)},
        {**complaint, "odiNumber": 789, "products": json.dumps([{
            "type": "VEHICLE", "productYear": "2020", "productMake": "HONDA", "productModel": "Civic"}])},
    ])
    tables["NHTSA_Recalls"] = pl.DataFrame([{
        "query_id": 1, "record_key": 1, "model_year": 2020, "make": "TOYOTA", "model": "Corolla",
        "ModelYear": 2020, "Make": "TOYOTA", "Model": "Corolla", "NHTSACampaignNumber": "20A",
        "ReportReceivedDate": date(2020, 1, 1), "Component": "BRAKES", "Summary": "Recall",
        "Consequence": "Risk", "Remedy": "Inspect"},
        {"query_id": 2, "record_key": 1, "model_year": 2020, "make": "TOYOTA", "model": "Corolla",
        "ModelYear": 2020, "Make": "TOYOTA", "Model": "Corolla", "NHTSACampaignNumber": "20A",
        "ReportReceivedDate": date(2020, 1, 1), "Component": "BRAKES", "Summary": "Recall",
        "Consequence": "Risk", "Remedy": "Inspect"}])
    tables["NHTSA_Safety"] = pl.DataFrame([{
        "query_id": 1, "ModelYear": 2020, "Make": "TOYOTA", "Model": "Corolla",
        "VehicleId": 1, "VehicleDescription": "Variant", "OverallRating": 0,
        "RolloverRating": 0, "RolloverRating2": 0,
        "RolloverPossibility": 0., "RolloverPossibility2": 0.}])
    tables["Youtube_Comments"] = pl.DataFrame({"comment_id": ["comment1"], "text": ["Raw unscored comment"]})
    manifest = {"tables": {}}
    for table, frame in tables.items():
        path = directory / f"{table}.parquet"
        frame.write_parquet(path)
        manifest["tables"][TABLE_NAMES[table]] = {
            "file": path.name, "rows": frame.height, "bytes": path.stat().st_size,
            "schema": {c: str(t) for c, t in frame.schema.items()}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return Snapshot(directory)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = fixture_snapshot(Path(self.temp.name))

    def test_lowercase_vin_and_inclusive_filters_without_identity_fill(self):
        source = self.source
        summary = source.summary(Filters(makes=("TOYOTA",), max_price=20000, max_mileage=50000))
        self.assertEqual(summary["listings"], 2)
        self.assertEqual(summary["cohorts"], 1)
        self.assertEqual(summary["median_price"], 15000)
        catalog = source.catalog().collect()
        self.assertEqual(catalog.height, 3)
        self.assertEqual(catalog.filter(pl.col("vin") == "b")["Trim"].item(), None)
        self.assertEqual(catalog.filter(pl.col("vin") == "c")["Make"].item(), None)
        self.assertEqual(source.summary(Filters(max_price=1))["listings"], 0)

    def test_duplicate_decode_refuses_row_multiplication(self):
        path = self.source.directory / "NHTSA_vPIC.parquet"
        frame = pl.read_parquet(path)
        pl.concat([frame, frame.head(1)]).write_parquet(path)
        with self.assertRaises(pl.exceptions.ComputeError):
            self.source.catalog().collect()

    def test_evidence_deduplicates_ids_checks_dates_and_applicability(self):
        bundle = evidence_bundle(self.source, Filters(makes=("TOYOTA",)))
        self.assertEqual(bundle["NHTSA_Complaints"]["count"], 1)
        self.assertEqual(bundle["NHTSA_Complaints"]["components"]["source_ids"].sum(), 1)
        self.assertEqual(bundle["NHTSA_Recalls"]["count"], 1)
        self.assertEqual(bundle["NHTSA_Safety"]["records"]["OverallRating"].item(), None)
        self.assertEqual(bundle["NHTSA_Safety"]["records"]["RolloverPossibility"].item(), None)
        path = self.source.directory / "NHTSA_Complaints.parquet"
        frame = pl.read_parquet(path)
        conflict = frame.head(1).with_columns(pl.lit("DIFFERENT").alias("summary"))
        pl.concat([frame, conflict]).write_parquet(path)
        changed = evidence_bundle(self.source, Filters(makes=("TOYOTA",)))
        self.assertEqual(changed["NHTSA_Complaints"]["count"], 1)
        self.assertEqual(changed["NHTSA_Complaints"]["conflicts"], 1)
        self.assertTrue(changed["NHTSA_Complaints"]["components"].is_empty())

    def test_errors_and_empty_or_constant_segments(self):
        frame = pd.DataFrame({"actual_price": [100., 200., 300.], "prediction": [100., 150., 250.],
                              "family_block": ["a", "b", "c"]})
        result = errors(frame)
        self.assertAlmostEqual(result["mae"], 100 / 3)
        self.assertAlmostEqual(result["r2"], .75)
        self.assertAlmostEqual(result["mean_signed_error"], -100 / 3)
        self.assertIsNone(errors(frame.iloc[:1])["r2"])
        self.assertIsNone(errors(frame.iloc[:0])["mae"])
        with self.assertRaises(ValueError):
            errors(frame.assign(prediction=np.inf))


    def test_filtered_evaluation_refuses_changed_source_fingerprint(self):
        from car_app.artifacts import validate_sources
        with patch("car_app.artifacts.snapshot", return_value=self.source):
            with self.assertRaisesRegex(ValueError, "No snapshot hashes"):
                validate_sources({})
            with self.assertRaisesRegex(ValueError, "Source fingerprint changed"):
                validate_sources({"source_file_sha256": {"used_cars.parquet": "wrong"},
                                  "dataset_fingerprint": "wrong"})

    def test_pairing_rejects_incompatible_metadata_and_evaluation_ids(self):
        from types import SimpleNamespace
        from car_app.artifacts import paired_comparison
        metadata = {key: "same" for key in ["dataset_fingerprint", "split_fingerprint",
                    "code_fingerprint", "preprocessing_version", "package_versions",
                    "tuning_rows", "max_trees", "split_strategy"]}
        metadata["model_type"] = "base"
        def run(meta):
            return SimpleNamespace(metadata=meta,
                table=lambda _: pd.DataFrame({"model": ["estimator"]}),
                prediction_path=lambda _: self.source.directory / "used_cars.parquet")
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            paired_comparison(run(metadata), run({**metadata, "split_fingerprint": "other"}),
                              Filters(), self.source.identity)
        a = pd.DataFrame({"vin": ["a"], "actual_price": [100.]})
        b = pd.DataFrame({"vin": ["b"], "actual_price": [100.]})
        with patch("car_app.artifacts.evaluation", side_effect=[a, b]):
            with self.assertRaisesRegex(ValueError, "populations differ"):
                paired_comparison(run(metadata), run(metadata), Filters(), self.source.identity)

    def test_filter_change_invalidates_similarity_only(self):
        independent = RecommendationPanel(messages=[{"role": "user", "content": "Independent request"}],
                                          requirements={"budget": 10000}, result={"id": "keep"})
        similar = SimilarityPanel(result={"id": "stale"})
        state = {"recommendation_session": independent, "similarity_session": similar}
        update_dashboard_scope(state, Filters(makes=("TOYOTA",)), "snapshot")
        self.assertIs(state["recommendation_session"], independent)
        self.assertEqual(independent.result, {"id": "keep"})
        self.assertIsNone(similar.result)
        revision = similar.revision
        update_dashboard_scope(state, Filters(makes=("TOYOTA",)), "snapshot")
        self.assertEqual(similar.revision, revision)
        update_dashboard_scope(state, Filters(makes=("HONDA",)), "snapshot")
        self.assertEqual(similar.revision, revision + 1)

    def test_app_filters_navigation_and_inactive_recommendation_panels(self):
        from car_app.data import filter_options, query_snapshot
        filter_options.clear(); query_snapshot.clear()
        with patch("car_app.data.snapshot", return_value=self.source):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            app.multiselect(key="filter_makes").select("TOYOTA").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["dashboard_filters"].makes, ("TOYOTA",))
            app.switch_page("app_pages/recommendation.py").run()
            self.assertFalse(app.exception)
            self.assertTrue(any(h.value == "Similar vehicles to the filter in the dashboard" for h in app.header))
            self.assertTrue(app.button(key="recommendation_unconnected").disabled)
            self.assertTrue(app.button(key="similarity_unconnected").disabled)
            self.assertEqual(app.session_state["recommendation_session"].revision, 0)
        filter_options.clear(); query_snapshot.clear()


if __name__ == "__main__":
    unittest.main()
