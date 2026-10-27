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
from car_app.requirements import QuestionDraft
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
    def test_epa_hierarchy_defaults_filters_and_recorded_only_switch(self):
        catalog = self.filtered_category_fixture()
        catalog.manifest.update(schema_version="5.0.0", grain="cohort")
        catalog.profiles["market_family"] = catalog.profiles.Model
        catalog.labels["status"] = "source_supported"
        additions = pd.DataFrame([
            {"profile_id": "p0", "dimension": "epa_group", "label": "Compact Cars", "status": "source_supported"},
            {"profile_id": "p0", "dimension": "size_segment", "label": "compact", "status": "source_supported"},
            {"profile_id": "p1", "dimension": "epa_group", "label": "Small Sport Utility Vehicle", "status": "source_supported"}])
        catalog.labels = pd.concat([catalog.labels, additions], ignore_index=True)
        for dimension in ("epa_group", "common_segment", "drivetrain", "epa_class", "nhtsa_class"):
            catalog.profiles[dimension + "_labels"] = ""
        catalog.predictions = pd.DataFrame([{"profile_id": "p2", "dimension": "epa_group", "label": "Midsize Cars",
            "policy": "source_trained", "probability": .85, "model_uri": "model://fixture", "outer_fold": 0}])
        metrics = Path(self.temp.name) / "metrics.csv"
        pd.DataFrame([{"dimension": "epa_group", "source_agreement": .8}]).to_csv(metrics, index=False)
        catalog.hierarchy_receipt = {"metrics": str(metrics), "directory": self.temp.name}
        code = '''from car_app.recommendation import load_similarity, category_explorer
c = load_similarity("fixture", (), "fixture")
category_explorer(c, "hierarchy", profiles=c.profiles.assign(matching_listings=c.profiles.vin_count))
'''
        with patch("car_app.recommendation.load_similarity", return_value=catalog):
            app = AppTest.from_string(code, default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="hierarchy_dimension").value, "epa_group")
            self.assertEqual(app.selectbox(key="hierarchy_policy").value, "source_trained")
            app.multiselect(key="hierarchy_labels").set_value(["Midsize Cars"]).run()
            self.assertFalse(app.exception)
            app.selectbox(key="hierarchy_policy").select("verified").run()
            self.assertFalse(app.exception)
            self.assertNotIn("Midsize Cars", app.multiselect(key="hierarchy_labels").options)

    def category_catalog(self):
        from car_app.recommendation import SimilarityCatalog, CATEGORY_DIMENSIONS
        directory = Path(self.temp.name) / "categories"
        directory.mkdir(exist_ok=True)
        profiles = pd.DataFrame([{"profile_id": pid, "fold": 0, "family_block": "TEST|" + pid,
            "ModelYear": 2020, "Make": "TOYOTA", "Model": "Corolla", "Trim": "LE", "body_style": "sedan",
            "body_family": "passenger_car", "powertrain": "unknown", "vin_count": 1, "wheelbase": 100.,
            "category_summary": label or "Unresolved", **{d + "_labels": label if d == "market_positioning" else "" for d in CATEGORY_DIMENSIONS}}
            for pid, label in [("p0", "economy"), ("p1", "luxury"), ("p2", "")]])
        pl.DataFrame({"vin": ["a", "b", "c"], "profile_id": ["p0", "p1", "p2"]}).write_parquet(directory / "vin_profile_membership.parquet")
        labels = pd.DataFrame([{"profile_id": pid, "dimension": "market_positioning", "label": label,
            "status": "source_supported_provisional", "source_url": "https://example.test/" + pid,
            "evidence": label, "scope": "Test scope"} for pid, label in [("p0", "economy"), ("p1", "luxury")]])
        return SimilarityCatalog(directory, {"schema_version": "4.0.0"}, profiles,
            pd.DataFrame(columns=["query_profile_id", "peer_profile_id", "approach", "rank"]), pd.DataFrame(),
            pd.DataFrame(columns=["profile_id", "cluster", "training_fold", "view", "method"]),
            pd.DataFrame(columns=["fold", "view", "method", "cluster", "description"]),
            pd.DataFrame([{"fold": 0, "view": "passenger_car", "status": "insufficient_support"}]), {0: ["wheelbase"]},
            labels, pd.DataFrame([{"dimension": "market_positioning", "reason": "unknown", "profiles": 1}]))

    def test_v5_grain_selector_and_strict_prediction_exclusion(self):
        catalog=self.filtered_category_fixture()
        catalog.manifest.update(schema_version="5.0.0",grain="model")
        catalog.labels["status"]="source_supported"
        catalog.predictions=pd.DataFrame([{"profile_id":"p2","dimension":"market_positioning","label":"luxury","policy":"conservative","probability":.99,"model_uri":"model://fixture","outer_fold":0}])
        for field in ("drivetrain_labels","epa_class_labels","nhtsa_class_labels"):
            catalog.profiles[field]=""
        liberal=catalog.listing_categories().collect()
        strict=catalog.listing_categories(strict=True).collect()
        self.assertEqual(liberal.filter(pl.col("vin").eq("c"))["market_positioning_labels"].item(),"luxury")
        self.assertEqual(strict.filter(pl.col("vin").eq("c"))["market_positioning_labels"].item(),"")
        (catalog.path/"manifest.json").write_text(json.dumps({"schema_version":"5.0.0"}))
        with patch("car_app.data.snapshot",return_value=self.source), \
             patch("car_app.model_registry.registry",return_value={"similarity_run":str(catalog.path)}), \
             patch("car_app.artifacts.stat_key",return_value=()), \
             patch("car_app.artifacts.diagnostic_inventory",return_value=(None,pd.DataFrame(),None,None)), \
             patch("car_app.recommendation.load_similarity",return_value=catalog):
            app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
            app.segmented_control(key="dashboard_view").set_value("Vehicle categories").run()
            self.assertFalse(app.exception)
            app.selectbox(key="dashboard_discovery_grain").select("model").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Saved v5 clustering" in row.value and "Model-level discovery" in row.value for row in app.caption))
            self.assertEqual(app.selectbox(key="dashboard_categories_clusters_color").value,"Saved subgroup")
            app.selectbox(key="dashboard_categories_clusters_color").select("Vehicle category").run()
            catalog.manifest["completed_at"]="fresh-fixture"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="dashboard_categories_clusters_color").value,"Saved subgroup")
            catalog.predictions=catalog.predictions.iloc[:0]
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="dashboard_categories_policy").value,"verified")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = fixture_snapshot(Path(self.temp.name))
        # Tiny fixtures exercise navigation independently of the production support cutoff.
        support = patch("car_app.data.MIN_MAKE_ROWS", 1)
        support.start()
        self.addCleanup(support.stop)

    def filtered_category_fixture(self):
        from car_app.data import filter_options, query_snapshot
        from car_app.recommendation import matching_profile_statistics
        listings = pl.read_parquet(self.source.directory / "used_cars.parquet").with_columns(
            pl.Series("body_type", ["Sedan", "SUV / Crossover", "Sedan"]),
            pl.Series("fuel_type", ["Gasoline", "Electric", "Gasoline"]),
            pl.Series("wheel_system", ["FWD", "AWD", "FWD"]),
            pl.Series("is_new", ["False", "True", "False"]))
        listings.write_parquet(self.source.directory / "used_cars.parquet")
        decoded = pl.read_parquet(self.source.directory / "NHTSA_vPIC.parquet")
        decoded = pl.concat([decoded.head(1).with_columns(pl.lit(vin).alias("vin"), pl.lit("LE").alias("Trim")) for vin in ["a", "b", "c"]])
        decoded.write_parquet(self.source.directory / "NHTSA_vPIC.parquet")
        self.source = Snapshot(self.source.directory)
        filter_options.clear(); query_snapshot.clear(); matching_profile_statistics.clear()
        catalog = self.category_catalog()
        catalog.profiles["identity_complete"] = True
        catalog.profiles["wheelbase"] = [100., 120., 110.]
        catalog.profiles["height"] = [55., 70., 60.]
        catalog.profiles.loc[catalog.profiles.profile_id.eq("p1"), ["body_family", "body_style"]] = ["suv_crossover", "suv_crossover"]
        catalog.fields = {0: ["wheelbase", "height"]}
        memberships, compositions = [], []
        for view in ["global", "passenger_car", "suv_crossover"]:
            population = catalog.profiles if view == "global" else catalog.profiles[catalog.profiles.body_family.eq(view)]
            for pid in population.profile_id:
                memberships.append({"profile_id": pid, "cluster": 1 if pid == "p1" else 0, "training_fold": 0, "view": view, "method": "kmeans"})
            for cluster in sorted({row["cluster"] for row in memberships if row["view"] == view}):
                compositions.append({"fold": 0, "view": view, "method": "kmeans", "cluster": cluster,
                    "description": "Longer wheelbase" if cluster else "Shorter wheelbase"})
        catalog.members = pd.DataFrame(memberships)
        catalog.compositions = pd.DataFrame(compositions)
        catalog.experiments = pd.DataFrame([{"fold": 0, "view": view, "status": "accepted_partition"} for view in ["global", "passenger_car", "suv_crossover"]])
        peers, statuses = [], []
        for query in catalog.profiles.profile_id:
            for approach in ["direct_specs", "category_specs", "category_cluster"]:
                others = sorted(set(catalog.profiles.profile_id) - {query}, reverse=approach == "direct_specs")
                for rank, peer in enumerate(others, 1):
                    peers.append({"query_profile_id": query, "peer_profile_id": peer, "approach": approach,
                        "rank": rank, "distance": .1 * rank, "category_match": 1., "cluster_penalty": 0., "year_window": 5})
                statuses.append({"query_profile_id": query, "approach": approach, "slate_size": len(others)})
        catalog.peers, catalog.statuses = pd.DataFrame(peers), pd.DataFrame(statuses)
        return catalog

    def test_profile_scope_uses_all_filters_and_actual_vin_variants(self):
        catalog = self.filtered_category_fixture()
        combined = Filters(makes=("TOYOTA",), models=("Corolla",), years=(2020,), trims=("LE",),
            body_types=("SUV / Crossover",), fuels=("Electric",), drivetrains=("AWD",), condition="New",
            min_price=20000, max_price=20000, max_mileage=50000)
        for filters in [combined, Filters(body_types=("SUV / Crossover",)), Filters(fuels=("Electric",)),
                        Filters(drivetrains=("AWD",)), Filters(condition="New"), Filters(min_price=20000, max_price=20000)]:
            rows = catalog.filtered_profiles(self.source, filters)
            self.assertEqual(rows.profile_id.tolist(), ["p1"])
            self.assertEqual(rows.matching_listings.tolist(), [1])
            self.assertEqual(rows.filtered_median_price.tolist(), [20000.])
        self.assertEqual(catalog.filtered_profiles(self.source, Filters(max_mileage=20000)).profile_id.tolist(), ["p0"])
        self.assertTrue(catalog.filtered_profiles(self.source, Filters(models=("Missing",))).empty)
        from car_app.recommendation import cluster_population
        scope = catalog.filtered_profiles(self.source, combined)
        plotted = cluster_population(catalog, scope, 0, "global", "kmeans")
        self.assertEqual(plotted.profile_id.tolist(), ["p1"])
        self.assertEqual(plotted.cluster_label.tolist(), ["Longer wheelbase"])

    def test_profile_scope_does_not_truncate_reference_catalog(self):
        catalog = self.category_catalog()
        catalog.profiles = pd.concat([catalog.profiles.iloc[[0]].assign(profile_id=f"p{i}") for i in range(205)], ignore_index=True)
        pl.DataFrame({"vin": [f"v{i}" for i in range(205)], "profile_id": catalog.profiles.profile_id.tolist()}).write_parquet(catalog.membership_path())
        class Source:
            identity = "many-references"
            def catalog(self, filters):
                return pl.DataFrame({"vin": [f"v{i}" for i in range(205)], "price": [10000.] * 205,
                    "mileage": [20000.] * 205, "cohort_complete": [True] * 205}).lazy()
        self.assertEqual(len(catalog.filtered_profiles(Source(), Filters())), 205)

    def test_filtered_diagrams_and_v4_similarity_refresh_without_button(self):
        catalog = self.filtered_category_fixture()
        with patch("car_app.data.snapshot", return_value=self.source), \
             patch("car_app.model_registry.registry", return_value={"similarity_run": "synthetic"}), \
             patch("car_app.artifacts.stat_key", return_value=()), \
             patch("car_app.artifacts.diagnostic_inventory", return_value=(None, pd.DataFrame(), None, None)), \
             patch("car_app.recommendation.load_similarity", return_value=catalog), \
             patch("car_app.model_registry.price_estimates", side_effect=lambda rows, config: (rows, None)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.segmented_control(key="dashboard_view").set_value("Vehicle categories").run()
            self.assertFalse(app.exception)
            self.assertGreaterEqual(len(app.get("vega_lite_chart")), 3)
            app.selectbox(key="filter_body_types").select("SUV / Crossover").run()
            self.assertFalse(app.exception)
            tables = [table.value for table in app.dataframe if "profile_id" in table.value]
            self.assertTrue(tables)
            self.assertTrue(all(set(table.profile_id) == {"p1"} for table in tables))
            app.switch_page("app_pages/recommendation.py").run()
            self.assertFalse(app.exception)
            result = app.session_state["similarity_session"].result
            self.assertEqual(result["profile_id"], "p1")
            self.assertEqual(result["approach"], "category_cluster")
            self.assertEqual(result["alternatives"].peer_profile_id.tolist(), ["p0", "p2"])
            app.checkbox(key="rec_budget_known").check()
            app.button(key="rec_review").click().run()
            draft = app.session_state["recommendation_session"].requirements.copy()
            app.number_input(key="filter_max_price").set_value(15000.).run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["similarity_session"].result)
            self.assertEqual(app.session_state["recommendation_session"].requirements, draft)

    def test_saved_v4_approach_uses_its_own_peer_order_and_status(self):
        catalog = self.filtered_category_fixture()
        direct = catalog.similar("p0", self.source.identity, 1)
        clustered = catalog.similar("p0", self.source.identity, 1, approach="category_cluster")
        self.assertEqual(direct["alternatives"].peer_profile_id.tolist(), ["p2", "p1"])
        self.assertEqual(clustered["alternatives"].peer_profile_id.tolist(), ["p1", "p2"])
        self.assertEqual(clustered["status"]["approach"], "category_cluster")

    def test_category_preferences_strict_unknowns_and_sort_preservation(self):
        from car_app.recommendation import search_candidates
        catalog = self.category_catalog()
        draft = QuestionDraft("Daily commute", max_price_usd=40000,
            category_preferences={"market_positioning": ("luxury",)})
        draft.validate({"body_types": [], "makes": [], "fuels": [], "drivetrains": [], "transmissions": [],
                        "vehicle_categories": catalog.category_choices()})
        requirements = draft.contract(self.source.identity, 1)
        result = search_candidates(self.source, requirements, 1, catalog=catalog)
        self.assertEqual(result["listings"]["vin"].to_list(), ["b", "a", "c"])
        self.assertEqual(result["listings"]["category_match"].to_list(), [1., 0., 0.])
        requirements["category_mode"] = "strict"
        self.assertEqual(search_candidates(self.source, requirements, 1, catalog=catalog)["listings"]["vin"].to_list(), ["b"])
        requirements["hard_requirements"]["max_price_usd"] = 15000
        self.assertEqual(search_candidates(self.source, requirements, 1, catalog=catalog)["eligible_listings"], 0)
        requirements["category_preferences"] = {}
        requirements["hard_requirements"]["max_price_usd"] = 40000
        baseline = search_candidates(self.source, requirements, 1)
        self.assertEqual(search_candidates(self.source, requirements, 1, catalog=catalog)["listings"]["vin"].to_list(), baseline["listings"]["vin"].to_list())
        catalog.profiles.loc[catalog.profiles.profile_id.eq("p0"), "performance_character_labels"] = "sports_car"
        catalog.labels = pd.concat([catalog.labels, pd.DataFrame([{"profile_id": "p0", "dimension": "performance_character", "label": "sports_car"}])], ignore_index=True)
        requirements["category_preferences"] = {"market_positioning": ["economy", "luxury"], "performance_character": ["sports_car"]}
        self.assertEqual(search_candidates(self.source, requirements, 1, catalog=catalog)["listings"]["vin"].to_list(), ["a"])

    def test_category_form_provisional_display_and_artifact_change(self):
        from car_app.data import filter_options, query_snapshot
        filter_options.clear(); query_snapshot.clear()
        catalog = self.category_catalog()
        config = {"similarity_run": "synthetic"}
        with patch("car_app.data.snapshot", return_value=self.source), \
             patch("car_app.model_registry.registry", side_effect=lambda: dict(config)), \
             patch("car_app.artifacts.stat_key", return_value=()), \
             patch("car_app.recommendation.load_similarity", return_value=catalog), \
             patch("car_app.model_registry.price_estimates", side_effect=lambda rows, config: (rows, None)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.switch_page("app_pages/recommendation.py").run()
            app.checkbox(key="rec_budget_known").check()
            app.number_input(key="rec_budget").set_value(30000)
            app.multiselect(key="rec_category_market_positioning").set_value(["luxury"])
            app.selectbox(key="rec_categories_mode").select("strict")
            app.button(key="rec_review").click().run()
            app.button(key="recommendation_search").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["recommendation_session"].result["listings"]["vin"].to_list(), ["b"])
            self.assertTrue(any("provisional" in element.value.lower() for element in app.caption))
            config["similarity_run"] = "replacement"
            app.run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["recommendation_session"].result)
            self.assertEqual(app.multiselect(key="rec_category_market_positioning").value, [])
            self.assertFalse(app.session_state["recommendation_session"].requirements)

    def test_distribution_extremes_follow_filters_and_preserve_listing_identity(self):
        from car_app.data import query_snapshot
        query_snapshot.clear()
        self.addCleanup(query_snapshot.clear)
        with patch("car_app.data.snapshot", return_value=self.source):
            rows = query_snapshot(self.source.identity, Filters(), "distribution_extremes", "price")
            self.assertEqual(rows["vin"].to_list(), ["c", "b", "a"])
            self.assertEqual(rows["price"].to_list(), [30000, 20000, 10000])
            rows = query_snapshot(self.source.identity, Filters(max_mileage=50000),
                                  "distribution_extremes", "mileage")
            self.assertEqual(rows["vin"].to_list(), ["b", "a"])
            self.assertEqual(rows["Make"].to_list(), ["TOYOTA", "TOYOTA"])

    def test_reviewed_vins_are_excluded_from_dashboard_and_unfiltered_evaluation(self):
        from car_app.data import EXCLUDED_LISTING_VINS
        from car_app.artifacts import evaluation
        path = self.source.directory / "used_cars.parquet"
        original = pl.read_parquet(path)
        additions = [original.head(1).with_columns(pl.lit(vin).alias("vin"))
                     for vin in sorted(EXCLUDED_LISTING_VINS)]
        pl.concat([original, *additions]).write_parquet(path)
        self.assertEqual(self.source.catalog().select(pl.len()).collect().item(), 3)
        predictions = self.source.directory / "excluded_predictions.parquet"
        pl.DataFrame({"vin": ["a", *sorted(EXCLUDED_LISTING_VINS)],
                      "actual_price": [100.] * 7, "prediction": [100.] * 7}).write_parquet(predictions)
        evaluation.clear()
        self.addCleanup(evaluation.clear)
        result = evaluation(str(predictions), (), self.source.identity, Filters())
        self.assertEqual(result.vin.tolist(), ["a"])

    def test_evaluation_fallback_broadens_identity_and_preserves_sidebar_filters(self):
        from car_app.artifacts import evaluation_with_fallback
        requested = Filters(makes=("TOYOTA",), models=("Corolla",), years=(2020,), max_price=20000)
        with patch("car_app.artifacts.evaluation", side_effect=[pd.DataFrame(index=range(0)), pd.DataFrame(index=range(30)), pd.DataFrame(index=range(120))]) as read:
            frame, scope, original_rows = evaluation_with_fallback("receipt", (), "source", requested)
        self.assertEqual((len(frame), scope, original_rows), (120, "Make", 0))
        self.assertEqual(read.call_args_list[1].args[-1], Filters(makes=("TOYOTA",), models=("Corolla",)))
        self.assertEqual(read.call_args_list[2].args[-1], Filters(makes=("TOYOTA",)))
        self.assertEqual(requested.years, (2020,))
        with patch("car_app.artifacts.evaluation", return_value=pd.DataFrame(index=range(100))) as read:
            _, scope, _ = evaluation_with_fallback("receipt", (), "source", requested)
        self.assertEqual(scope, "Selected filters")
        self.assertEqual(read.call_count, 1)

    def test_make_choices_require_ten_listing_rows_before_cohort_deduplication(self):
        from car_app.data import supported_make_cohorts
        rows = pl.DataFrame({"Make": ["SMALL"] * 9 + ["SUPPORTED"] * 10, "Model": ["same"] * 19})
        with patch("car_app.data.MIN_MAKE_ROWS", 10):
            selected = supported_make_cohorts(rows)
        self.assertEqual(selected["Make"].unique().to_list(), ["SUPPORTED"])
        self.assertEqual(selected.height, 10)

    def test_safety_display_hides_secondary_rollover_and_explains_all_ratings(self):
        from car_app.views import evidence_display, safety_rating_definitions

        records = self.source.scan("NHTSA_Safety").collect()
        displayed = evidence_display(records, "Safety")
        self.assertNotIn("RolloverRating2", displayed.columns)
        self.assertNotIn("RolloverPossibility2", displayed.columns)
        guide = safety_rating_definitions(displayed.columns).set_index("Column")
        ratings = [c for c in displayed.columns if "rating" in c.casefold()]
        self.assertTrue(set(ratings).issubset(guide.index))
        self.assertIn("5 best", guide.loc["OverallRating", "Range and interpretation"])
        self.assertIn("lower is better", guide.loc["RolloverPossibility", "Range and interpretation"])
        self.assertIn("single-vehicle", guide.loc["RolloverRating", "Meaning"])
        self.assertTrue(safety_rating_definitions(["OverallCrashRating"])["Meaning"].str.contains("Not defined").all())

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

    def test_multiple_make_model_year_scope_reaches_all_sources(self):
        directory = Path(self.temp.name) / "multi" / "DATA" / "FINAL"
        directory.mkdir(parents=True)
        source = fixture_snapshot(directory)
        path = directory / "NHTSA_vPIC.parquet"
        decoded = pl.read_parquet(path).with_columns(
            pl.Series("Make", ["TOYOTA", "HONDA"]),
            pl.Series("Model", ["Corolla", "Civic"]),
            pl.Series("ModelYear", [2020, 2021]), pl.Series("Trim", ["LE", "LX"]))
        decoded.write_parquet(path)
        for name in ["NHTSA_Complaints", "NHTSA_Recalls", "NHTSA_Safety"]:
            path = directory / f"{name}.parquet"
            original = pl.read_parquet(path)
            extra = original.head(1)
            replacements = {"model_year": 2021, "make": "HONDA", "model": "Civic",
                            "ModelYear": 2021, "Make": "HONDA", "Model": "Civic",
                            "odiNumber": 999, "NHTSACampaignNumber": "20B", "VehicleId": 2}
            extra = extra.with_columns(pl.lit(value).cast(original.schema[column]).alias(column)
                                       for column, value in replacements.items() if column in original.columns)
            pl.concat([original, extra]).write_parquet(path)
        feature_dir = directory.parents[1] / "YOUTUBE/outputs/price_features_v1"
        feature_dir.mkdir(parents=True)
        pl.DataFrame({"Make": ["TOYOTA", "HONDA", "FORD"], "yt_comment_count": [20, 30, 40],
                      "yt_support_suppressed": [False, False, True]}).write_parquet(feature_dir / "make_features.parquet")
        filters = Filters(makes=("TOYOTA", "HONDA"), models=("Corolla", "Civic"), years=(2020, 2021))
        self.assertEqual(source.summary(filters)["listings"], 2)
        self.assertEqual(source.catalog(filters).collect()["Make"].n_unique(), 2)
        self.assertEqual(source.cohort_summary(filters)["Make"].n_unique(), 2)
        self.assertEqual(source.families(filters).collect().height, 2)
        for item in evidence_bundle(source, filters).values():
            self.assertEqual(item["families"], 2)
        self.assertEqual(set(source.youtube(filters)["Make"]), {"TOYOTA", "HONDA"})
        # Retaining only Corolla intentionally excludes Honda even with both makes selected.
        self.assertEqual(source.summary(Filters(makes=filters.makes, models=("Corolla",)))["listings"], 1)
        from car_app.views import youtube_display
        shown = youtube_display(source.youtube(filters))
        self.assertEqual(shown.columns, ["Make", "support_suppressed", "Metric", "Value"])
        self.assertEqual(set(shown["Metric"]), {"comment_count"})
        self.assertEqual(shown.height, 2)

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

    def test_filtered_evaluation_checks_external_sources(self):
        import hashlib
        from car_app.artifacts import validate_sources

        external = self.source.directory / "external"
        external.mkdir()
        for name in ["make_features.parquet", "cohort_features.parquet"]:
            (external / name).write_bytes(name.encode())
        paths = {"used_cars.parquet": self.source.directory / "used_cars.parquet",
                 "make_features.parquet": external / "make_features.parquet",
                 "expert/cohort_features.parquet": external / "cohort_features.parquet"}
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
        metadata = {"source_file_sha256": hashes,
                    "dataset_fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
        with patch("car_app.artifacts.snapshot", return_value=self.source), \
             patch("car_app.model_registry.price_sources", return_value=paths):
            validate_sources(metadata)
            (external / "cohort_features.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Source fingerprint changed: expert/cohort_features"):
                validate_sources(metadata)

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

    def test_question_draft_validates_categories_boundaries_and_ambiguity(self):
        categories = {"body_types": ["Sedan"], "makes": ["TOYOTA"], "fuels": ["Gasoline"],
                      "drivetrains": ["FWD"], "transmissions": ["M"]}
        draft = QuestionDraft(use_case="Enthusiast / weekend driving", max_price_usd=25000,
                              max_mileage=0, transmissions=("M",), min_horsepower=200,
                              required_equipment=("Leather Seats",), priorities=("Safety", "Reliability"))
        draft.validate(categories)
        result = draft.contract("fixture", 1)
        self.assertEqual(result["hard_requirements"]["max_mileage"], 0)
        self.assertNotIn("require_reported_clean_history", result["hard_requirements"])
        self.assertFalse(result["llm"]["api_calls_enabled"])
        self.assertEqual(result["llm"]["model"], "gpt-6-luna")
        self.assertIsNone(result["priority_weights"])
        self.assertEqual(len(result["unresolved_questions"]), 2)
        for bad in [
            QuestionDraft(use_case="Daily commute", min_year=2020, max_year=2010),
            QuestionDraft(use_case="Daily commute", max_price_usd=float("inf")),
            QuestionDraft(use_case="Daily commute", fuels=("invented",)),
            QuestionDraft(use_case="Daily commute", required_equipment=("invented",)),
        ]:
            with self.assertRaises(ValueError):
                bad.validate(categories)
        unresolved = QuestionDraft(use_case="Not decided").contract("fixture", 2)
        self.assertEqual(unresolved["hard_requirements"], {})
        self.assertEqual(len(unresolved["unresolved_questions"]), 2)

    def test_guided_review_enables_search_without_using_dashboard_scope(self):
        from car_app.data import filter_options, query_snapshot
        filter_options.clear(); query_snapshot.clear()
        self.addCleanup(filter_options.clear)
        self.addCleanup(query_snapshot.clear)
        with patch("car_app.data.snapshot", return_value=self.source):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.switch_page("app_pages/recommendation.py").run()
            app.selectbox(key="rec_use_case").select("Enthusiast / weekend driving")
            app.checkbox(key="rec_budget_known").check()
            app.number_input(key="rec_budget").set_value(18000)
            app.multiselect(key="rec_priorities").set_value(["Safety", "Performance"])
            app.checkbox(key="rec_hp_known").check()
            app.number_input(key="rec_hp").set_value(250)
            app.button(key="rec_review").click().run()
            self.assertFalse(app.exception)
            draft = app.session_state["recommendation_session"].requirements
            self.assertEqual(draft["hard_requirements"]["max_price_usd"], 18000)
            self.assertEqual(draft["hard_requirements"]["min_horsepower"], 250)
            self.assertNotIn("max_mileage", draft["hard_requirements"])
            self.assertNotIn("makes", draft["hard_requirements"])
            self.assertEqual(len(draft["unresolved_questions"]), 1)
            self.assertFalse(app.button(key="recommendation_search").disabled)
            self.assertIsNone(app.session_state["recommendation_session"].result)
            self.assertEqual(app.session_state["recommendation_session"].revision, 1)
            app.selectbox(key="filter_makes").select("TOYOTA").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["recommendation_session"].requirements, draft)
            self.assertEqual(app.session_state["recommendation_session"].revision, 1)
            app.number_input(key="rec_budget").set_value(17000).run()
            self.assertEqual(app.session_state["recommendation_session"].requirements, draft)
            app.number_input(key="rec_budget").set_value(17000)
            app.button(key="rec_review").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["recommendation_session"].revision, 2)
            self.assertEqual(app.session_state["recommendation_session"].requirements["hard_requirements"]["max_price_usd"], 17000)

    def test_app_filters_navigation_and_reference_requirements(self):
        from car_app.data import filter_options, query_snapshot
        filter_options.clear(); query_snapshot.clear()
        with patch("car_app.data.snapshot", return_value=self.source):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            app.selectbox(key="filter_makes").select("TOYOTA").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["dashboard_filters"].makes, ("TOYOTA",))
            app.switch_page("app_pages/recommendation.py").run()
            self.assertFalse(app.exception)
            self.assertTrue(any(h.value == "Similar vehicles to the filter in the dashboard" for h in app.header))
            self.assertTrue(app.button(key="recommendation_search").disabled)
            self.assertTrue(app.button(key="similarity_search").disabled)
            self.assertEqual(app.session_state["recommendation_session"].revision, 0)
        filter_options.clear(); query_snapshot.clear()

    def test_requirement_search_inclusive_filters_and_qualifying_cohorts(self):
        from car_app.recommendation import search_candidates
        requirements = QuestionDraft(use_case="Daily commute", max_price_usd=20000, max_mileage=50000).contract(self.source.identity, 1)
        result = search_candidates(self.source, requirements, 1)
        self.assertEqual(result["eligible_listings"], 2)
        self.assertEqual(result["listings"]["vin"].to_list(), ["a", "b"])
        self.assertEqual(result["cohorts"]["qualifying_listings"].to_list(), [1])
        self.assertEqual(result["cohorts"]["median_price"].item(), 10000)
        self.assertEqual(result["cohorts"]["Trim"].item(), "LE")
        self.assertTrue(result["cohorts"]["low_support"].item())
        requirements["hard_requirements"]["max_price_usd"] = 1
        self.assertTrue(search_candidates(self.source, requirements, 1)["listings"].is_empty())
        with self.assertRaisesRegex(ValueError, "stale"):
            search_candidates(self.source, requirements, 2)

    def test_unknown_equipment_history_and_horsepower_cannot_meet_requirements(self):
        from car_app.recommendation import search_candidates
        path = self.source.directory / "used_cars.parquet"
        data = pl.read_parquet(path).with_columns(
            pl.Series("horsepower", [150., 200., None]),
            pl.Series("major_options", ["['Leather Seats']", "['Leather Seats package']", None]),
            *[pl.Series(field, ["False", None, "False"]) for field in ["has_accidents", "frame_damaged", "salvage", "theft_title"]])
        data.write_parquet(path)
        request = QuestionDraft(use_case="Daily commute", max_price_usd=30000, min_horsepower=150,
                                required_equipment=("Leather Seats",), require_reported_clean_history=True).contract(self.source.identity, 1)
        self.assertEqual(search_candidates(self.source, request, 1)["listings"]["vin"].to_list(), ["a"])
        request["hard_requirements"] = {"max_price_usd": 30000, "min_horsepower": 200}
        self.assertEqual(search_candidates(self.source, request, 1, "Advertised horsepower")["listings"]["vin"].to_list(), ["b"])

    def test_price_failure_preserves_order_and_model_context_invalidates_results(self):
        from car_app.model_registry import price_estimates
        from car_app.state import update_result_context
        rows = self.source.catalog().head(2).collect()
        config = {"active_price": {"run": "fixture", "model": "fixture", "metadata_sha256": "x", "predictor_sha256": "y"}}
        with patch("car_app.model_registry.read_json", return_value={"source_file_sha256": {}}), \
             patch("car_app.model_registry.load_predictor", side_effect=ValueError("changed")):
            result, warning = price_estimates(rows, config)
        self.assertTrue(result.equals(rows))
        self.assertIn("changed", warning)
        panel = RecommendationPanel(artifact_identity="old", result={"stale": True})
        update_result_context(panel, "new")
        self.assertIsNone(panel.result)

    def test_cluster_comparison_uses_reference_fit_not_peer_own_fold(self):
        from car_app.recommendation import SimilarityCatalog
        profiles = pd.DataFrame([
            {"profile_id": "query", "fold": 0, "body_family": "passenger_car", "ModelYear": 2020,
             "Make": "TOYOTA", "Model": "Corolla", "Trim": "LE", "wheelbase": 100.},
            {"profile_id": "peer", "fold": 1, "body_family": "passenger_car", "ModelYear": 2020,
             "Make": "HONDA", "Model": "Civic", "Trim": "LX", "wheelbase": 102.}])
        members = pd.DataFrame([
            {"profile_id": pid, "cluster": cluster, "training_fold": fold, "view": "passenger_car", "method": "kmeans"}
            for pid, cluster, fold in [("query", 0, 0), ("peer", 0, 0), ("peer", 1, 1)]])
        compositions = pd.DataFrame([{"fold": fold, "view": "passenger_car", "method": "kmeans", "cluster": cluster, "description": "example"}
                                     for fold, cluster in [(0, 0), (1, 1)]])
        peers = pd.DataFrame([{"query_profile_id": "query", "peer_profile_id": "peer", "approach": "direct_specs", "rank": 1}])
        status = pd.DataFrame([{"query_profile_id": "query", "approach": "direct_specs", "slate_size": 1}])
        catalog = SimilarityCatalog(Path("."), {}, profiles, peers, status, members, compositions, pd.DataFrame(), {0: ["wheelbase"]})
        result = catalog.similar("query", "fixture", 1)
        self.assertTrue(result["alternatives"].shared_cluster.item())
        self.assertEqual(result["cluster_context"], (0, "passenger_car", "kmeans"))
        catalog.members.loc[catalog.members.profile_id.eq("peer") & catalog.members.training_fold.eq(0), "cluster"] = -1
        self.assertIsNone(catalog.similar("query", "fixture", 1)["alternatives"].shared_cluster.item())

    def test_similarity_reference_matches_case_and_preserves_variants(self):
        from car_app.recommendation import matching_similarity_profiles
        cohorts = pd.DataFrame([
            [2015, "Toyota", "Camry", " Hybrid "],
            [2015, "TOYOTA", "CAMRY", "HYBRID"],
            [2015, "TOYOTA", "Camry", None]], columns=COHORT)
        profiles = pd.DataFrame([
            [2015, "TOYOTA", "CAMRY", "HYBRID", "hybrid-a"],
            [2015, "toyota", "camry", "hybrid", "hybrid-b"],
            [2016, "TOYOTA", "CAMRY", "HYBRID", "wrong-year"],
            [2015, "TOYOTA", "CAMRY", "LE", "wrong-trim"],
            [2015, "TOYOTA", "CAMRY", None, "missing-trim"]], columns=[*COHORT, "profile_id"])
        original = profiles.copy(deep=True)
        matched = matching_similarity_profiles(cohorts, profiles)
        self.assertEqual(matched.profile_id.tolist(), ["hybrid-a", "hybrid-b"])
        pd.testing.assert_frame_equal(profiles, original)
        self.assertEqual(matched.iloc[0].Model, "CAMRY")
        self.assertTrue(matching_similarity_profiles(cohorts.iloc[:0], profiles).empty)

    def test_app_checkpoint_discovery_advances_each_model_independently(self):
        from car_app.artifacts import diagnostic_inventory
        root = Path(self.temp.name) / "dynamic_models"
        for name, models in [("20260101-full", {"catboost": "completed", "ridge": "completed"}),
                             ("20260102-full", {"catboost": "completed", "ridge": "running"})]:
            path = root / "MODEL_OUTPUT" / "base" / name
            path.mkdir(parents=True)
            (path / "run_metadata.json").write_text(json.dumps({"model_type": "base", "status": "running",
                "requested_sample_size": None, "models": {m: {"status": state} for m, state in models.items()}}))
            pd.DataFrame({"model": list(models), "mae": [1., 2.]}).to_csv(path / "metrics.csv", index=False)
            for model in models:
                (path / model).mkdir()
                (path / model / "predictor.joblib").touch()
                for partition in ("test", "validation"):
                    (path / f"{model}_{partition}_predictions.parquet").touch()
        with patch("car_app.artifacts.ROOT", root):
            _, choices, _, _ = diagnostic_inventory()
            selected = choices.set_index("label").run.to_dict()
        self.assertEqual(selected, {"base-catboost": "base/20260102-full", "base-ridge": "base/20260101-full"})

    def test_registry_reports_newer_running_and_pilot_without_admitting_them(self):
        from car_app.model_registry import COMMON_SOURCES, COMPARISON_KEYS, compatibility_group, latest_inventory
        root = Path(self.temp.name) / "models"
        metadata = {key: "same" for key in COMPARISON_KEYS}
        metadata.update(status="completed", model_type="base", requested_sample_size=None,
                        requested_models=["ok", "unfinished"], source_file_sha256={name: "same" for name in COMMON_SOURCES},
                        raw_feature_columns=["example"], models={"ok": {"status": "completed"}, "unfinished": {"status": "running"}})
        for variant, name, status in [("base", "20260101-full", "completed"), ("base", "20260102-full", "running"),
                                      ("full", "20260103-pilot-5", "completed")]:
            path = root / variant / name
            path.mkdir(parents=True)
            (path / "run_metadata.json").write_text(json.dumps({**metadata, "model_type": variant, "status": status}))
            pd.DataFrame({"model": ["ok", "unfinished"], "validation_mae": [10., 1.]}).to_csv(path / "validation_metrics.csv", index=False)
            for model in ("ok", "unfinished"):
                (path / model).mkdir()
                (path / model / "predictor.joblib").write_bytes(b"fixture-not-executed")
        inventory, candidates = latest_inventory(root)
        self.assertEqual(candidates.model.tolist(), ["ok"])
        self.assertEqual(candidates.run.tolist(), ["base/20260101-full"])
        self.assertEqual(inventory.loc[inventory.variant.eq("base"), "latest_status"].item(), "running")
        self.assertEqual(inventory.loc[inventory.variant.eq("full"), "status"].item(), "pilot_only")
        self.assertNotEqual(compatibility_group(metadata), compatibility_group({**metadata, "split_fingerprint": "different"}))

    def test_generate_renders_results_and_keeps_independent_budget(self):
        from car_app.data import filter_options, query_snapshot
        filter_options.clear(); query_snapshot.clear()
        self.addCleanup(filter_options.clear); self.addCleanup(query_snapshot.clear)
        with patch("car_app.data.snapshot", return_value=self.source), \
             patch("car_app.recommendation.load_similarity", return_value=None), \
             patch("car_app.model_registry.price_estimates", side_effect=lambda rows, config: (rows, None)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.switch_page("app_pages/recommendation.py").run()
            app.checkbox(key="rec_budget_known").check()
            app.number_input(key="rec_budget").set_value(20000)
            app.button(key="rec_review").click().run()
            app.button(key="recommendation_search").click().run()
            self.assertFalse(app.exception)
            result = app.session_state["recommendation_session"].result
            self.assertEqual(result["listings"]["vin"].to_list(), ["a", "b"])
            self.assertEqual(result["eligible_listings"], 2)
            app.selectbox(key="filter_makes").select("TOYOTA").run()
            self.assertEqual(app.session_state["recommendation_session"].result["eligible_listings"], 2)


if __name__ == "__main__":
    unittest.main()
