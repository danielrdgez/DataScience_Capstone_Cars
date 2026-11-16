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
        code = '''from car_app.recommendation import load_similarity, _research_category_explorer as category_explorer
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
            self.assertTrue(any(row.label == "Profiles" for row in app.metric))
            self.assertEqual(app.selectbox(key="dashboard_categories_compact_clusters_color").value,"Saved subgroup")
            app.selectbox(key="dashboard_categories_compact_clusters_color").select("Vehicle category").run()
            catalog.manifest["completed_at"]="fresh-fixture"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="dashboard_categories_compact_clusters_color").value,"Saved subgroup")
            catalog.predictions=catalog.predictions.iloc[:0]
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.checkbox(key="dashboard_categories_compact_clusters_details").value)
            profile_tables = [table.value for table in app.dataframe if "profile_id" in table.value]
            self.assertFalse(profile_tables)
            app.checkbox(key="dashboard_categories_compact_clusters_details").check().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="dashboard_categories_compact_clusters_detail_kind").value, "Profiles")
            profile_tables = [table.value for table in app.dataframe if "profile_id" in table.value]
            self.assertEqual(len(profile_tables), 1)

    def setUp(self):
        from car_app.agent import Settings
        credentials = patch("car_app.agent.settings", return_value=Settings(""))
        credentials.start()
        self.addCleanup(credentials.stop)
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
            self.assertGreaterEqual(len(app.get("vega_lite_chart")), 2)
            app.selectbox(key="filter_body_types").select("SUV / Crossover").run()
            self.assertFalse(app.exception)
            app.checkbox(key="dashboard_categories_compact_clusters_details").check().run()
            self.assertFalse(app.exception)
            tables = [table.value for table in app.dataframe if "profile_id" in table.value]
            self.assertTrue(tables)
            self.assertTrue(all(set(table.profile_id) == {"p1"} for table in tables))
            app.switch_page("app_pages/recommendation.py").run()
            self.assertFalse(app.exception)
            app.checkbox(key="rec_show_research").check().run()
            result = app.session_state["similarity_session"].result
            self.assertEqual(result["profile_id"], "p1")
            self.assertEqual(result["approach"], "category_cluster")
            self.assertEqual(result["alternatives"].peer_profile_id.tolist(), ["p0", "p2"])
            app.multiselect(key="rec_priorities").set_value(["Lower purchase price"])
            app.button(key="recommendation_search").click().run()
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
             patch("car_app.model_registry.registry", side_effect=lambda **kwargs: dict(config)), \
             patch("car_app.artifacts.stat_key", return_value=()), \
             patch("car_app.recommendation.load_similarity", return_value=catalog), \
             patch("car_app.model_registry.price_estimates", side_effect=lambda rows, config: (rows, None)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.switch_page("app_pages/recommendation.py").run()
            app.selectbox(key="rec_budget").select("$30,000")
            app.multiselect(key="rec_priorities").set_value(["Lower purchase price"])
            app.multiselect(key="rec_category_market_positioning").set_value(["luxury"])
            app.selectbox(key="rec_categories_mode").select("strict")
            app.button(key="recommendation_search").click().run()
            self.assertFalse(app.exception)
            self.assertEqual([r["candidate_id"] for r in app.session_state["recommendation_session"].agent_result["answer"]["listings"]], ["listing:b"])
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
        from YOUTUBE.text.sentiment import DEFAULT_PRICE_FEATURE_DIRECTORY
        from car_app.data import ROOT
        from MODELS.Price_ML_Models import file_sha256
        feature_dir = directory.parents[1] / DEFAULT_PRICE_FEATURE_DIRECTORY.relative_to(ROOT)
        feature_dir.mkdir(parents=True)
        pl.DataFrame({"Make": ["TOYOTA", "HONDA", "FORD"], "yt_comment_count": [20, 30, 40],
                      "yt_support_suppressed": [False, False, True]}).write_parquet(feature_dir / "make_features.parquet")
        (feature_dir / "price_feature_manifest.json").write_text(json.dumps({
            "state": "complete", "sentiment_schema_version": 2,
            "attribution_scope": "explicit_comment", "window": "cumulative",
            "features_sha256": file_sha256(feature_dir / "make_features.parquet"),
            "feature_blocks": {"coverage": ["yt_comment_count", "yt_support_suppressed"]}}))
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
        self.assertEqual(set(shown["Feature"]), {"yt_comment_count", "yt_support_suppressed"})
        self.assertEqual(shown.unique(subset=["Make", "Feature"]).height, 4)
        from car_app.model_registry import price_sources
        metadata = {"source_file_sha256": {"make_features.parquet": "fixture"},
                    "youtube_features": {"directory": str(feature_dir)}}
        self.assertEqual(price_sources(metadata)["make_features.parquet"], feature_dir / "make_features.parquet")
        with self.assertRaisesRegex(ValueError, "YouTube source directory missing"):
            price_sources({"source_file_sha256": metadata["source_file_sha256"]})

    @staticmethod
    def youtube_feature_fixture():
        from MODELS.Price_ML_Models import YOUTUBE_EXPORT_FEATURE_COLUMNS

        fields = {field: ([False, True] if field.endswith("_missing") or field == "yt_support_suppressed" else
                          [1, 1] if field == "yt_feature_row_present" else
                          [40, 0] if "count" in field else [.2, None])
                  for field in YOUTUBE_EXPORT_FEATURE_COLUMNS}
        return pl.DataFrame({"Make": ["TOYOTA", "HONDA"], **fields})

    def test_youtube_input_tables_match_model_transforms_without_duplicate_features(self):
        from MODELS.Price_ML_Models import YOUTUBE_FEATURE_COLUMNS
        from car_app.views import youtube_feature_tables

        inputs, additional = youtube_feature_tables(self.youtube_feature_fixture(), YOUTUBE_FEATURE_COLUMNS)
        self.assertEqual(inputs.height, 80)
        self.assertEqual(additional.height, 70)
        self.assertEqual(set(inputs["Feature"]), set(YOUTUBE_FEATURE_COLUMNS))
        self.assertFalse(set(inputs["Feature"]) & set(additional["Feature"]))
        for frame in (inputs, additional):
            self.assertEqual(frame.height, frame.unique(subset=["Make", "Feature"]).height)
        self.assertFalse({"yt_overall_opinion_count", "yt_overall_opinion_video_count"} & set(additional["Feature"]))
        counts = inputs.filter(pl.col("Feature") == "yt_overall_log1p_opinion_count")
        self.assertAlmostEqual(float(counts.filter(pl.col("Make") == "TOYOTA")["Value"][0]), np.log1p(40))
        self.assertEqual(counts.filter(pl.col("Make") == "TOYOTA")["Original count"][0], "40")
        self.assertEqual(float(counts.filter(pl.col("Make") == "HONDA")["Value"][0]), 0.)
        missing = inputs.filter((pl.col("Make") == "HONDA") & (pl.col("Feature") == "yt_reliability_video_balanced_net"))
        self.assertIsNone(missing["Value"][0])
        self.assertIn("insufficient", missing["Evidence"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            youtube_feature_tables(self.youtube_feature_fixture(), [*YOUTUBE_FEATURE_COLUMNS, YOUTUBE_FEATURE_COLUMNS[0]])

    def test_youtube_panel_charts_supported_evidence_and_preserves_comments(self):
        from MODELS.Price_ML_Models import YOUTUBE_FEATURE_COLUMNS

        data = self.youtube_feature_fixture()
        contract = {"directory": "fixture", "source_signature": (), "as_of": "2026-10-01",
                    "columns": list(YOUTUBE_FEATURE_COLUMNS), "runs": ["full/fixture-full"]}
        def query(identity, filters, operation, *args):
            if operation == "youtube_trace":
                make = filters.makes[0]
                return pl.DataFrame({"Make": [make, make], "comment_id": ["saved-1"] * 2,
                    "video_title": ["Review"] * 2, "published_date": ["2026-09-01"] * 2,
                    "text": [f"Example for {make}"] * 2,
                    "evidence_start": [0, 15], "evidence_end": [14, 30],
                    "evidence_text": ["Comfort is good", "Reliability is poor"],
                    "topics": [["comfort"], ["reliability"]],
                    "content_label": ["vehicle_opinion"] * 2, "overall_polarity": ["positive", "negative"],
                    "feature_eligible": [True] * 2, "development_excluded": [False] * 2,
                    "direct_price_mention": [False] * 2, "model_truncated": [False] * 2,
                    "windowed": [False] * 2})
            return data.filter(pl.col("Make").is_in(list(filters.makes))) if filters.makes else data
        code = '''from types import SimpleNamespace
from car_app.views import youtube
from car_app.data import Filters
import streamlit as st
st.session_state["youtube_walkthrough"] = True
youtube(SimpleNamespace(identity="fixture"), Filters(makes=("TOYOTA",)))
'''
        with patch("car_app.views.query_snapshot", side_effect=query), \
             patch("car_app.views.youtube_price_contract", return_value=contract), \
             patch("car_app.views.filter_options", return_value=(data.select("Make"), None)):
            app = AppTest.from_string(code, default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="youtube_base_make").value, "HONDA")
            self.assertEqual([text.value for text in app.text], ["Example for HONDA", "Comfort is good", "Reliability is poor"])
            self.assertEqual(len(app.get("vega_lite_chart")), 0)
            app.selectbox(key="youtube_base_make").select("TOYOTA").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 0)
            self.assertEqual(len(app.get("vega_lite_chart")), 2)
            self.assertEqual([heading.value for heading in app.subheader],
                             ["Discussion topics", "Overall sentiment"])
            self.assertEqual([item.label for item in app.expander], ["How the YouTube pipeline works"])
            self.assertIn("Combined overall opinion: mixed.", [item.value for item in app.markdown])
            self.assertEqual([text.value for text in app.text], ["Example for TOYOTA", "Comfort is good", "Reliability is poor"])
            self.assertNotIn("Additional YouTube evidence", [item.label for item in app.expander])
            app.selectbox(key="youtube_compare_make").select("HONDA").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe), 0)
            self.assertEqual(len(app.get("vega_lite_chart")), 2)
            self.assertEqual([text.value for text in app.text], ["Example for TOYOTA", "Comfort is good", "Reliability is poor"])
            self.assertIn("Overall sentiment unavailable: HONDA", [item.value for item in app.caption])
            data = data.filter(pl.col("Make") == "HONDA")
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("vega_lite_chart")), 0)
            self.assertIn("No supported overall sentiment scores are available for these makes.",
                          [item.value for item in app.info])
            self.assertEqual(app.selectbox(key="youtube_base_make").value, "HONDA")
            self.assertIsNone(app.selectbox(key="youtube_compare_make").value)
            self.assertEqual([text.value for text in app.text], ["Example for HONDA", "Comfort is good", "Reliability is poor"])
            data = data.head(0)
            app.run()
            self.assertFalse(app.exception)
            self.assertIn("No make has a saved feature row. This does not establish absence of discussion.",
                          [item.value for item in app.info])

    def test_snapshot_cache_refreshes_old_instances_after_code_change(self):
        from types import SimpleNamespace
        from car_app.data import snapshot_at

        old_source = SimpleNamespace(identity="unchanged-data")
        signature = (("fixture.parquet", 10, 20),)
        snapshot_at.clear()
        try:
            with patch("car_app.data.Snapshot", return_value=old_source):
                self.assertIs(snapshot_at(signature, "old-code"), old_source)
            with patch("car_app.data.Snapshot", return_value=self.source) as constructor:
                self.assertIs(snapshot_at(signature, "old-code"), old_source)
                refreshed = snapshot_at(signature, "new-code")
                self.assertIs(refreshed, self.source)
                self.assertTrue(callable(refreshed.youtube_trace))
                constructor.assert_called_once_with()
        finally:
            snapshot_at.clear()

    def test_youtube_trace_retains_all_clauses_for_one_make_and_comment(self):
        directory = Path(self.temp.name) / "trace/price"
        shards = directory.parent / "scored/shards"
        directory.mkdir(parents=True)
        shards.mkdir(parents=True)
        frame = pl.DataFrame({
            "Make": ["TOYOTA", "TOYOTA", "HONDA", "TOYOTA"],
            "comment_id": ["first", "first", "first", "other"],
            "video_id": ["v1"] * 4, "video_title": ["Review"] * 4,
            "text": ["Original"] * 4, "published_date": ["2026-09-01"] * 4,
            "updated_date": ["2026-09-01"] * 4,
            "attribution_scope": ["explicit_comment"] * 4,
            "evidence_start": [10, 0, 0, 0], "evidence_end": [19, 9, 9, 9],
            "evidence_text": ["Second", "First", "Wrong make", "Wrong comment"],
            "topics": [["comfort"]] * 4, "content_label": ["vehicle_opinion"] * 4,
            "overall_polarity": ["positive", "unresolved", "negative", "negative"],
            "feature_eligible": [True, False, True, True],
            "development_excluded": [False] * 4, "direct_price_mention": [False] * 4,
            "model_truncated": [False] * 4, "windowed": [False] * 4,
        })
        frame.write_parquet(shards / "scores_000000.parquet")
        source = object.__new__(Snapshot)
        with patch.object(Snapshot, "catalog", return_value=pl.DataFrame({"Make": ["TOYOTA"]}).lazy()):
            result = source.youtube_trace(Filters(makes=("TOYOTA",)), directory, "2026-10-01")
        self.assertEqual(result["evidence_text"].to_list(), ["First", "Second"])
        self.assertEqual(result["feature_eligible"].to_list(), [False, True])
        self.assertEqual(result["Make"].unique().to_list(), ["TOYOTA"])

    def test_youtube_contract_uses_recorded_source_and_rejects_source_changes(self):
        import hashlib
        from car_app.artifacts import youtube_price_contract

        root = Path(self.temp.name) / "youtube_contract"
        directory = root / "YOUTUBE/outputs/selected/price"
        directory.mkdir(parents=True)
        features = self.youtube_feature_fixture()
        features.write_parquet(directory / "make_features.parquet")
        manifest = {"state": "complete", "sentiment_schema_version": 2, "as_of": "2026-10-01",
                    "attribution_scope": "explicit_comment", "window": "cumulative",
                    "feature_blocks": {"all": [c for c in features.columns if c != "Make"]},
                    "features_sha256": hashlib.sha256((directory / "make_features.parquet").read_bytes()).hexdigest()}
        manifest_path = directory / "price_feature_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        run = root / "MODEL_OUTPUT/full/fixture-full"
        run.mkdir(parents=True)
        metadata = {"model_type": "full", "raw_feature_columns": ["price_feature", "yt_comment_count"],
                    "youtube_features": {"directory": str(directory), "as_of": manifest["as_of"],
                        "features_sha256": manifest["features_sha256"],
                        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}}
        (run / "run_metadata.json").write_text(json.dumps(metadata))
        (root / "car_app").mkdir()
        (root / "car_app/artifact_registry.json").write_text(json.dumps({
            "price_run_selection": {"runs": {"full": "full/fixture-full"}}}))
        contract = youtube_price_contract(root)
        self.assertEqual(contract["columns"], ["yt_comment_count"])
        self.assertEqual(Path(contract["directory"]), directory)
        manifest_path.write_text(json.dumps({**manifest, "as_of": "2026-10-02"}))
        with self.assertRaisesRegex(ValueError, "differs from the Full pricing receipt"):
            youtube_price_contract(root)

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

    def test_evaluation_filters_frozen_attributes_without_current_source(self):
        from car_app.artifacts import evaluation, stat_key

        file = self.source.directory / "frozen_predictions.parquet"
        pl.DataFrame({"vin": ["a", "b", "c"], "Make": ["TOYOTA"] * 3,
                      "Model": ["Corolla"] * 3, "ModelYear": [2020.] * 3,
                      "Trim": ["LE", "LE", "SE"], "is_new": ["False"] * 3,
                      "mileage": [20000., 50000., 80000.],
                      "actual_price": [9000., 19000., 29000.],
                      "prediction": [10000., 18000., 30000.]}).write_parquet(file)
        filters = Filters(makes=("TOYOTA",), models=("Corolla",), years=(2020,),
                          trims=("LE",), condition="Used", min_price=9000., max_price=19000.,
                          max_mileage=50000.)
        with patch("car_app.artifacts.validate_sources") as validate, \
             patch("car_app.artifacts.selected_vins") as current:
            result = evaluation(str(file), stat_key(file), "changed-source", filters)
        self.assertEqual(result.vin.tolist(), ["a", "b"])
        self.assertEqual(result.actual_price.tolist(), [9000., 19000.])
        self.assertNotIn("price", result)
        validate.assert_not_called()
        current.assert_not_called()
        (file.parent / "run_metadata.json").write_text("{}")
        with patch("car_app.artifacts.validate_sources", side_effect=ValueError("Source fingerprint changed")):
            with self.assertRaisesRegex(ValueError, "Source fingerprint changed"):
                evaluation(str(file), stat_key(file), "changed-source", Filters(body_types=("Sedan",)))

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

    def test_numeric_question_presets_and_custom_mileage(self):
        from car_app.requirements import amount_label, mileage_amount, range_presets
        for lower, upper in ((256, 3299995), (0, 400000), (1234.5, 1234.5)):
            values = range_presets(lower, upper)
            self.assertEqual((values[0], values[-1]), (lower, upper))
            self.assertEqual(values, sorted(set(values)))
            self.assertLess(len(values), 60)
        self.assertIn(25000, range_presets(256, 3299995))
        self.assertEqual(range_presets(None, None), [])
        self.assertEqual(range_presets(0, float("inf")), [])
        self.assertEqual(mileage_amount("75,000"), 75000)
        self.assertEqual(mileage_amount(" 75.5k "), 75500)
        self.assertEqual(mileage_amount("0"), 0)
        self.assertIsNone(mileage_amount(None))
        self.assertEqual(mileage_amount(amount_label(1234.5)), 1234.5)
        for invalid in ("-1", "nan", "inf", "$75,000", "75,00", "75k miles"):
            with self.assertRaises(ValueError):
                mileage_amount(invalid)

    def test_question_dropdowns_use_full_ranges_and_preserve_custom_caps(self):
        from car_app.data import filter_options, query_snapshot
        listings = pl.read_parquet(self.source.directory / "used_cars.parquet").with_columns(
            pl.Series("price", [256, 20000, 3299995]), pl.Series("mileage", [0., 50000., 400000.]))
        listings.write_parquet(self.source.directory / "used_cars.parquet")
        self.source = Snapshot(self.source.directory)
        filter_options.clear(); query_snapshot.clear()
        self.addCleanup(filter_options.clear); self.addCleanup(query_snapshot.clear)
        with patch("car_app.data.snapshot", return_value=self.source), \
             patch("car_app.data.MIN_MAKE_ROWS", 1), \
             patch("car_app.model_registry.registry", return_value={}), \
             patch("car_app.discovery.active_catalog", return_value=None), \
             patch("car_app.agent_ui.submit_request") as request:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.selectbox(key="filter_makes").select("TOYOTA").run()
            app.session_state["rec_form_version"] = 2
            app.session_state["rec_mileage"] = 50000.
            app.switch_page("app_pages/recommendation.py").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="rec_budget").options[0], "$256")
            self.assertEqual(app.selectbox(key="rec_budget").options[-1], "$3,299,995")
            self.assertEqual(app.selectbox(key="rec_mileage").options[0], "0")
            self.assertEqual(app.selectbox(key="rec_mileage").options[-1], "400,000")
            self.assertEqual(app.selectbox(key="rec_mileage").value, "50,000")
            self.assertTrue(app.selectbox(key="rec_budget").proto.accept_new_options)
            self.assertTrue(app.selectbox(key="rec_mileage").proto.accept_new_options)
            # AppTest requires custom choices in its local list to serialize typed values.
            app.selectbox(key="rec_budget").options.append("3500k")
            app.selectbox(key="rec_mileage").options.append("75.5k")
            app.selectbox(key="rec_budget").set_value("3500k")
            app.selectbox(key="rec_mileage").set_value("75.5k")
            app.multiselect(key="rec_priorities").set_value(["Lower purchase price"])
            app.button(key="recommendation_search").click().run()
            self.assertFalse(app.exception)
            hard = app.session_state["recommendation_session"].requirements["hard_requirements"]
            self.assertEqual(hard["max_price_usd"], 3500000)
            self.assertEqual(hard["max_mileage"], 75500)
            app.run()
            self.assertEqual(app.selectbox(key="rec_budget").value, "3500k")
            self.assertEqual(app.selectbox(key="rec_mileage").value, "75.5k")
            app.selectbox(key="rec_mileage").set_value("0")
            app.button(key="recommendation_search").click().run()
            self.assertEqual(app.session_state["recommendation_session"].requirements["hard_requirements"]["max_mileage"], 0)
            app.selectbox(key="rec_mileage").set_value(None)
            app.button(key="recommendation_search").click().run()
            self.assertNotIn("max_mileage", app.session_state["recommendation_session"].requirements["hard_requirements"])
            app.selectbox(key="rec_mileage").options.append("unknown")
            app.selectbox(key="rec_mileage").set_value("unknown")
            app.button(key="recommendation_search").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(request.call_count, 3)

    def test_guided_review_enables_search_without_using_dashboard_scope(self):
        from car_app.data import filter_options, query_snapshot
        filter_options.clear(); query_snapshot.clear()
        self.addCleanup(filter_options.clear)
        self.addCleanup(query_snapshot.clear)
        with patch("car_app.data.snapshot", return_value=self.source):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.switch_page("app_pages/recommendation.py").run()
            app.selectbox(key="rec_use_case").select("Enthusiast / weekend driving")
            app.selectbox(key="rec_budget").set_value("$20,000")
            app.multiselect(key="rec_priorities").set_value(["Safety", "Performance"])
            app.number_input(key="rec_hp").set_value(250)
            app.button(key="recommendation_search").click().run()
            self.assertFalse(app.exception)
            draft = app.session_state["recommendation_session"].requirements
            self.assertEqual(draft["hard_requirements"]["max_price_usd"], 20000)
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
            app.selectbox(key="rec_budget").set_value("$15,000").run()
            self.assertEqual(app.session_state["recommendation_session"].requirements, draft)
            app.selectbox(key="rec_budget").set_value("$15,000")
            app.button(key="recommendation_search").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["recommendation_session"].revision, 2)
            self.assertEqual(app.session_state["recommendation_session"].requirements["hard_requirements"]["max_price_usd"], 15000)

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
            self.assertFalse(any(h.value == "Similar vehicles to the filter in the dashboard" for h in app.header))
            self.assertFalse(app.button(key="recommendation_search").disabled)
            app.checkbox(key="rec_show_research").check().run()
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

    def test_ai_pending_oof_never_falls_back_to_dashboard_champion(self):
        from car_app.agent_context import selected_price_models
        config = {"recommendation_price": {"kind": "catboost_full_oof_v1", "run": None},
                  "active_price": {"run": "enhanced/legacy", "model": "ridge"}}
        with patch("car_app.artifacts.diagnostic_inventory") as inventory:
            selected = selected_price_models(config)
        self.assertIsNone(selected["primary"])
        self.assertNotIn("linear", selected)
        self.assertTrue(selected["unavailable"])
        inventory.assert_not_called()

    def test_source_equivalence_checks_all_retained_values_and_model_population(self):
        from car_app.model_registry import digest, verify_retained_source_equivalence

        root = Path(self.temp.name) / "equivalent_sources"
        originals = root / "outputs/slide13_update/original_snapshot"
        current = root / "DATA/CAR_DATA_FINAL"
        backup = current / ".vpic-alignment-backup-test"
        run = root / "MODEL_OUTPUT/full/example-full"
        for directory in (originals, backup, run):
            directory.mkdir(parents=True)
        old_listings = pl.DataFrame({"vin": ["a", "b", "removed"], "price": [1, 2, 3]})
        old_listings.write_parquet(originals / "used_cars.parquet")
        old_listings.filter(pl.col("vin") != "removed").write_parquet(current / "used_cars.parquet")
        old_vpic = pl.DataFrame({"vin": ["a", "removed"], "Make": ["A", "B"]})
        old_vpic.write_parquet(backup / "NHTSA_vPIC.parquet")
        pl.DataFrame({"vin": ["a", "b"], "Make": ["A", None]}).write_parquet(current / "NHTSA_vPIC.parquet")
        pl.DataFrame({"vin": ["a", "b"]}).write_parquet(run / "split_manifest.parquet")
        metadata = {"source_file_sha256": {"used_cars.parquet": digest(originals / "used_cars.parquet"),
                                             "NHTSA_vPIC.parquet": digest(backup / "NHTSA_vPIC.parquet")}}
        with patch("car_app.model_registry.ROOT", root):
            receipt = verify_retained_source_equivalence(run, metadata)
            self.assertEqual((receipt["removed_listings"], receipt["missing_vpic_placeholders"]), (1, 1))
            pl.DataFrame({"vin": ["a", "b"], "Make": ["CHANGED", None]}).write_parquet(current / "NHTSA_vPIC.parquet")
            with self.assertRaisesRegex(ValueError, "Retained source values changed"):
                verify_retained_source_equivalence(run, metadata)
            pl.DataFrame({"vin": ["a", "b"], "Make": ["A", None]}).write_parquet(current / "NHTSA_vPIC.parquet")
            pl.DataFrame({"vin": ["a", "removed"]}).write_parquet(run / "split_manifest.parquet")
            with self.assertRaisesRegex(ValueError, "Removed listings appear"):
                verify_retained_source_equivalence(run, metadata)

    def test_changed_inference_code_requires_frozen_input_and_vin_aligned_prediction_replay(self):
        import hashlib
        from types import SimpleNamespace
        from unittest.mock import Mock
        from car_app.model_registry import verify_inference_replay
        root = Path(self.temp.name) / "replay"
        run = root / "MODEL_OUTPUT/base/20260101-full"
        (run / "ridge").mkdir(parents=True)
        pd.DataFrame({"vin": ["a", "b"]}).to_parquet(run / "shap_receipt.parquet")
        pd.DataFrame({"x": [1., 2.]}).to_parquet(run / "shap_inputs.parquet")
        pd.DataFrame({"x": [1.]}).to_parquet(run / "shap_background.parquet")
        pd.DataFrame({"vin": ["b", "a"], "prediction": [200., 100.]}).to_parquet(run / "ridge_test_predictions.parquet")
        (run / "ridge/predictor.joblib").write_bytes(b"fixture")
        hashes = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                  for name in ("shap_inputs.parquet", "shap_receipt.parquet", "shap_background.parquet")}
        metadata = {"model_type": "base", "raw_feature_columns": ["x"], "feature_columns": ["x"],
            "shap_inputs": "shap_inputs.parquet", "shap_receipt": "shap_receipt.parquet",
            "shap_background": "shap_background.parquet", "shap_input_sha256": hashes,
            "source_file_sha256": {}, "code_fingerprint": "training"}
        predictor = SimpleNamespace(named_steps={"input": SimpleNamespace(transform=lambda data: data)},
                                    predict=Mock(return_value=np.array([100., 200.])))
        with patch("car_app.model_registry.ROOT", root), \
             patch("car_app.model_registry.prediction_inputs", return_value=pl.DataFrame({"vin": ["a", "b"], "x": [1., 2.]})), \
             patch("car_app.model_registry.inference_code_fingerprint", return_value="current"), \
             patch("joblib.load", return_value=predictor):
            receipt = verify_inference_replay(run, metadata, "ridge", "current")
            self.assertEqual(receipt["maximum_absolute_usd_difference"], 0)
            verify_inference_replay(run, metadata, "ridge", "current")
            self.assertEqual(predictor.predict.call_count, 1)
            predictor.predict.return_value = np.array([110., 200.])
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                verify_inference_replay(run, metadata, "ridge", "different-code")
            pd.DataFrame({"x": [9., 2.]}).to_parquet(run / "shap_inputs.parquet")
            with self.assertRaisesRegex(ValueError, "Frozen inference reference changed"):
                verify_inference_replay(run, metadata, "ridge", "current")

    def test_selected_suite_never_fills_pending_models_from_other_runs(self):
        from car_app.artifacts import diagnostic_inventory
        root = Path(self.temp.name) / "selected_models"
        selection = {"runs": {"base": "base/20260102-full", "full": "full/20260102-full"},
                     "models": ["catboost", "ridge", "pcr"]}

        def checkpoint(variant, name, states):
            path = root / "MODEL_OUTPUT" / variant / name
            path.mkdir(parents=True, exist_ok=True)
            metadata = {"model_type": variant, "status": "running", "requested_sample_size": None,
                        "models": {model: {"status": state} for model, state in states.items()}}
            (path / "run_metadata.json").write_text(json.dumps(metadata))
            pd.DataFrame([{"model": model, "mae": 10., "rmse": 20., "validation_mae": 11.}
                          for model in states]).to_csv(path / "metrics.csv", index=False)
            pd.DataFrame([{"model": model, "validation_mae": 11.}
                          for model in states]).to_csv(path / "validation_metrics.csv", index=False)
            for model in states:
                (path / model).mkdir(exist_ok=True)
                (path / model / "predictor.joblib").write_bytes(b"fixture-not-executed")
                for partition in ("test", "validation"):
                    (path / f"{model}_{partition}_predictions.parquet").write_bytes(b"fixture-not-read")
            return path

        checkpoint("base", "20260101-full", {"catboost": "completed", "ridge": "completed"})
        checkpoint("base", "20260103-full", {"catboost": "completed", "ridge": "completed"})
        checkpoint("base", "20260102-full", {"catboost": "completed", "ridge": "running", "random_forest": "completed"})
        full = checkpoint("full", "20260102-full", {"catboost": "completed", "pcr": "running"})
        with patch("car_app.artifacts.ROOT", root):
            _, choices, statuses, _ = diagnostic_inventory(selection)
            self.assertEqual(set(choices.label), {"base-catboost", "full-catboost"})
            self.assertEqual(set(choices.run), set(selection["runs"].values()))
            self.assertEqual(len(statuses), 5)
            self.assertFalse(statuses.loc[statuses.model.eq("pcr"), "status"].eq("completed").any())
            checkpoint("full", "20260102-full", {"catboost": "completed", "pcr": "completed"})
            self.assertIn("full-pcr", diagnostic_inventory(selection)[1].label.tolist())
            # A metadata receipt alone cannot publish an incomplete prediction file.
            (full / "pcr_validation_predictions.parquet").write_bytes(b"")
            self.assertNotIn("full-pcr", diagnostic_inventory(selection)[1].label.tolist())
            (full / "run_metadata.json").unlink()
            _, choices, statuses, _ = diagnostic_inventory(selection)
            self.assertEqual(choices.label.tolist(), ["base-catboost"])
            self.assertEqual(set(statuses.loc[statuses.feature_set.eq("full"), "status"]), {"pending"})

    def test_full_continuation_keeps_catboost_and_rejects_other_training_commands(self):
        from car_app.artifacts import diagnostic_inventory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = "full/20261029-full"
            models = ["ridge", "pcr"]
            selection = {"runs": {"full": reference}, "models": ["catboost", *models],
                         "full_continuation": {"after": "20261104", "models": models}}
            contract = {key: "same" for key in ("raw_feature_columns", "source_file_sha256",
                "split_fingerprint", "preprocessing_version", "package_versions", "tuning_rows",
                "max_trees", "price_weighting", "split_strategy", "device_request", "full_validation")}

            def checkpoint(name, requested):
                path = root / "MODEL_OUTPUT" / name
                path.mkdir(parents=True)
                (path / "run_metadata.json").write_text(json.dumps({**contract, "model_type": "full",
                    "requested_sample_size": None, "requested_models": requested, "status": "running",
                    "models": {model: {"status": "completed"} for model in requested}}))
                for table in ("metrics.csv", "validation_metrics.csv"):
                    pd.DataFrame([{"model": model, "mae": 1., "rmse": 2., "validation_mae": 3.}
                                  for model in requested]).to_csv(path / table, index=False)
                for model in requested:
                    (path / model).mkdir()
                    (path / model / "predictor.joblib").write_bytes(b"fixture")
                    for partition in ("test", "validation"):
                        (path / f"{model}_{partition}_predictions.parquet").write_bytes(b"fixture")

            checkpoint(reference, ["catboost", "ridge"])
            with patch("car_app.artifacts.ROOT", root):
                _, choices, statuses, _ = diagnostic_inventory(selection)
                self.assertEqual(choices.model.tolist(), ["catboost"])
                self.assertEqual(statuses.status.tolist(), ["completed", "pending", "pending"])
                checkpoint("full/20261104-full", models)
                checkpoint("full/20261105-full", ["random_forest", *models])
                _, choices, _, _ = diagnostic_inventory(selection)
                self.assertEqual(set(choices.model), {"catboost", *models})
                self.assertEqual(choices.loc[choices.model.eq("catboost"), "run"].item(), reference)
                self.assertEqual(set(choices.loc[choices.model.ne("catboost"), "run"]), {"full/20261104-full"})
                selection["full_continuation"]["per_estimator"] = True
                checkpoint("full/20261106-full", ["pcr"])
                checkpoint("full/20261107-full", ["pcr"])
                (root / "MODEL_OUTPUT/full/20261107-full/pcr/predictor.joblib").unlink()
                _, choices, _, _ = diagnostic_inventory(selection)
                self.assertEqual(choices.loc[choices.model.eq("pcr"), "run"].item(), "full/20261106-full")
                self.assertEqual(choices.loc[choices.model.eq("ridge"), "run"].item(), "full/20261104-full")
                models.append("lightgbm")
                selection["models"].append("lightgbm")
                checkpoint("full/20261108-full", ["lightgbm"])
                receipt = root / "MODEL_OUTPUT/full/20261108-full/run_metadata.json"
                metadata = json.loads(receipt.read_text())
                metadata["device_request"] = "cpu"
                receipt.write_text(json.dumps(metadata))
                _, choices, _, _ = diagnostic_inventory(selection)
                self.assertIn("lightgbm", choices.model.tolist())
                selection["full_continuation"]["lightgbm_serialization"] = "bounded_tree_shards_v1"
                self.assertNotIn("lightgbm", diagnostic_inventory(selection)[1].model.tolist())
                metadata["lightgbm_serialization"] = "bounded_tree_shards_v1"
                metadata["models"]["lightgbm"]["status"] = "running"
                receipt.write_text(json.dumps(metadata))
                self.assertNotIn("lightgbm", diagnostic_inventory(selection)[1].model.tolist())
                metadata["models"]["lightgbm"]["status"] = "completed"
                receipt.write_text(json.dumps(metadata))
                self.assertIn("lightgbm", diagnostic_inventory(selection)[1].model.tolist())
                metadata["split_fingerprint"] = "different"
                receipt.write_text(json.dumps(metadata))
                self.assertNotIn("lightgbm", diagnostic_inventory(selection)[1].model.tolist())

    def test_selected_suite_default_uses_validation_and_refreshes_metadata_pin(self):
        from car_app.model_registry import registry
        candidates = pd.DataFrame([
            {"feature_set": "base", "model": "catboost", "run": "base/selected-full", "validation_mae": 20., "test_mae": 1.},
            {"feature_set": "full", "model": "ridge", "run": "full/selected-full", "validation_mae": 10., "test_mae": 50.}])
        config = {"price_run_selection": {"runs": {"full": "full/selected-full"}, "models": ["ridge"]},
                  "active_price": {"run": "old/run", "model": "random_forest"}}
        with patch("car_app.model_registry.read_json", return_value=config), \
             patch("car_app.artifacts.diagnostic_inventory", return_value=({}, candidates, pd.DataFrame(), pd.DataFrame())), \
             patch("car_app.model_registry.digest", side_effect=["metadata-1", "predictor", "metadata-2", "predictor"]):
            first = registry()["active_price"]
            second = registry()["active_price"]
        self.assertEqual(first["run"], "full/selected-full")
        self.assertEqual(first["model"], "ridge")
        self.assertNotEqual(first["metadata_sha256"], second["metadata_sha256"])
        with patch("car_app.model_registry.read_json", return_value=config), \
             patch("car_app.artifacts.diagnostic_inventory", return_value=({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())):
            self.assertIsNone(registry()["active_price"])

    def test_performance_tab_scopes_both_partitions_and_shows_pending_models(self):
        from car_app.artifacts import Run
        run = Run(ROOT / "MODEL_OUTPUT/base/selected-full", {"model_type": "base", "split_strategy": "cohort_stratified"})
        candidates = pd.DataFrame([{"feature_set": "base", "model": "catboost", "run": run.name}])
        statuses = pd.DataFrame([{"feature_set": "base", "model": "catboost", "status": "completed"},
                                 {"feature_set": "full", "model": "pcr", "status": "pending"}])
        metrics = pd.DataFrame([{"model": model, "mae": 10., "validation_mae": 11.}
                                for model in ("catboost", "random_forest", "training_median_baseline")])

        def script():
            from types import SimpleNamespace
            from car_app.data import Filters
            from car_app.views import models
            models(SimpleNamespace(identity="fixture"), Filters())

        with patch("car_app.views.diagnostic_inventory", return_value=({run.name: run}, candidates, statuses, pd.DataFrame())), \
             patch.object(Run, "table", side_effect=lambda name: metrics if name == "metrics.csv" else pd.DataFrame()), \
             patch.object(Run, "prediction_path", return_value=None), \
             patch("car_app.views.saved_shap", return_value=pd.DataFrame()):
            app = AppTest.from_function(script, default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[0].value.status.tolist(), ["completed", "pending"])
            self.assertEqual(app.dataframe[1].value.model.tolist(), ["catboost"])
            self.assertIn("same groups appear", " ".join(c.value for c in app.caption).lower())
            app.segmented_control(key="evaluation_mode").set_value("Validation").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[1].value.model.tolist(), ["catboost"])
            self.assertEqual(app.dataframe[1].value.mae.tolist(), [11.])

    def test_generate_renders_results_and_keeps_independent_budget(self):
        from car_app.agent import Settings
        from car_app.data import filter_options, query_snapshot
        filter_options.clear(); query_snapshot.clear()
        self.addCleanup(filter_options.clear); self.addCleanup(query_snapshot.clear)
        with patch("car_app.data.snapshot", return_value=self.source), \
             patch("car_app.agent.settings", return_value=Settings("")), \
             patch("car_app.model_registry.registry", return_value={}), \
             patch("car_app.recommendation.load_similarity", return_value=None), \
             patch("car_app.model_registry.price_estimates", side_effect=lambda rows, config: (rows, None)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.switch_page("app_pages/recommendation.py").run()
            app.selectbox(key="rec_budget").select("$20,000")
            app.multiselect(key="rec_priorities").set_value(["Lower purchase price"])
            app.button(key="recommendation_search").click().run()
            self.assertFalse(app.exception)
            result = app.session_state["recommendation_session"].agent_result
            self.assertEqual([r["candidate_id"] for r in result["answer"]["listings"]], ["listing:a"])
            self.assertEqual(result["telemetry"]["failure_category"], "missing_credentials")
            self.assertFalse(any(h.value.startswith("Best AI match") for h in app.subheader))
            self.assertEqual(result["eligible_listings"], 2)
            app.selectbox(key="filter_makes").select("TOYOTA").run()
            self.assertEqual(app.session_state["recommendation_session"].agent_result["eligible_listings"], 2)


class DiscoveryRankingTests(unittest.TestCase):
    def test_v6_peers_use_query_fit_and_exclude_family_body_and_year_mismatches(self):
        import joblib
        from car_app.discovery import Catalog
        from MODELS.vehicle_similarity_v3 import fit_metric

        profiles = pd.DataFrame([
            ("query", "TOYOTA|COROLLA", 2020, 0, "Sedan", 100., 180.),
            ("same_family", "TOYOTA|COROLLA", 2019, 0, "Sedan", 100., 180.),
            ("peer", "HONDA|CIVIC", 2020, 1, "Sedan", 101., 181.),
            ("peer_older", "HONDA|CIVIC", 2019, 1, "Sedan", 102., 182.),
            ("second", "MAZDA|3", 2020, 2, "Sedan", 104., 186.),
            ("wrong_body", "HONDA|CRV", 2020, 1, "SUV / crossover", 100., 180.),
            ("too_old", "TEST|OLD", 1990, 1, "Sedan", 100., 180.),
            ("unsupported", "TEST|SPARSE", 2020, 1, "Sedan", 100., 180.),
        ], columns=["profile_id", "family_block", "ModelYear", "fold", "category", "wheelbase", "length"])
        profiles["powertrain"] = "gasoline"
        profiles["fuel_type"] = "GASOLINE"
        catalog = Catalog.__new__(Catalog)
        catalog.profiles, catalog.grain = profiles, "model"
        context = catalog.context(profiles.iloc[0])
        catalog.assignments = pd.DataFrame({"profile_id": profiles.profile_id, "context": context,
            "eligible": profiles.profile_id.ne("unsupported"), "cluster": 0, "description": "Query fit", "status": "assigned"})
        catalog.assignments = pd.concat([catalog.assignments, pd.DataFrame([{
            "profile_id": "peer", "context": catalog.context(profiles.iloc[2]), "eligible": True,
            "cluster": 1, "description": "Peer's own fit", "status": "assigned"}])], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            catalog.specs = Path(directory)
            path = catalog.specs / "model/contexts" / context
            path.mkdir(parents=True)
            metric = fit_metric(profiles.loc[profiles.fold.ne(0)], ["wheelbase", "length", "fuel_type"])
            joblib.dump({"metric": metric}, path / "retrieval.joblib")
            peers, actual_context = catalog.similar("query")
            self.assertEqual(peers.profile_id.tolist(), ["peer", "second"])
            self.assertEqual(actual_context, context)
            self.assertTrue(peers.shared_cluster.all())
            self.assertEqual(set(peers.description), {"Query fit"})
            scoped, _ = catalog.similar("query", candidate_ids={"peer_older", "second"})
            self.assertEqual(scoped.profile_id.tolist(), ["peer_older", "second"])
            catalog.assignments.loc[catalog.assignments.profile_id.eq("query"), "cluster"] = -1
            self.assertTrue(catalog.similar("query")[0].shared_cluster.isna().all())

    def test_v6_listing_labels_use_own_fold_without_changing_cohort_identity(self):
        from types import SimpleNamespace
        from car_app.discovery import with_listing_clusters

        catalog = SimpleNamespace(
            members=pd.DataFrame({"vin": ["a", "b"], "profile_id": ["p", "unresolved"]}),
            profiles=pd.DataFrame({"profile_id": ["p", "unresolved"], "fold": [1, 2]}),
            assignments=pd.DataFrame({"profile_id": ["p", "p"], "fold": [0, 1],
                "description": ["Other fit", "Own fit"], "status": ["assigned", "direct_fallback"], "context": ["c0", "c1"]}))
        listings = pl.DataFrame({"vin": ["a", "b"], "Model": ["Original case", "Missing body"], "Trim": [None, "BASE"]})
        result = with_listing_clusters(listings, catalog)
        self.assertEqual(result["Model"].to_list(), listings["Model"].to_list())
        self.assertEqual(result["Trim"].to_list(), listings["Trim"].to_list())
        self.assertEqual(result["cluster_description"].to_list(), ["Own fit", None])
        self.assertEqual(result["cluster_status"].to_list(), ["direct_fallback", None])

    def test_v6_loader_requires_matching_full_validation_and_unchanged_report(self):
        from car_app.discovery import load_catalog
        from MODELS import discovery_artifacts as artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, validation = root / "MODEL_OUTPUT/run/bundle", root / "validation"
            bundle.mkdir(parents=True)
            validation.mkdir()
            receipt = {"schema_version": artifacts.VERSION, "status": "completed", "stage": "attach-evidence",
                       "artifacts": {}, "dependencies": [], "source_hashes": {}, "reduced_budget": False}
            artifacts.write_json(bundle / "manifest.json", receipt)
            bundle_hash = artifacts.file_hash(bundle / "manifest.json")
            run = {"directory": str(bundle.resolve()), "sha256": bundle_hash}
            report = {"passed": True, "run": run, "checks": []}
            artifacts.write_json(validation / "validation.json", report)
            validated = {**receipt, "stage": "validate", "passed": True, "run": run,
                         "artifacts": {"validation.json": artifacts.file_hash(validation / "validation.json")}}
            artifacts.write_json(validation / "manifest.json", validated)
            arguments = ("run/bundle", (), bundle_hash, "fixture", directory, (), str(validation), (), "model")
            load_catalog.clear()
            self.addCleanup(load_catalog.clear)
            with patch.object(artifacts, "ROOT", root), patch.object(artifacts, "source_hashes", return_value={}), \
                 patch("car_app.discovery.Catalog") as catalog:
                self.assertIs(load_catalog(*arguments), catalog.return_value)
                load_catalog.clear()
                artifacts.write_json(validation / "manifest.json", {**validated, "run": {**run, "sha256": "other"}})
                with self.assertRaisesRegex(ValueError, "validation does not match"):
                    load_catalog(*arguments)
                load_catalog.clear()
                artifacts.write_json(validation / "manifest.json", validated)
                artifacts.write_json(validation / "validation.json", {**report, "passed": False})
                from car_app.artifacts import stat_key
                changed_arguments = (*arguments[:7], stat_key(validation / "validation.json"), *arguments[8:])
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    load_catalog(*changed_arguments)

    def test_full_population_is_ranked_before_five_result_limit(self):
        from car_app.discovery import diverse_top
        rows = pd.DataFrame({"vin": [str(i) for i in range(12)], "family_id": [str(i) for i in range(12)],
            "ModelYear": [2015] * 12, "Make": ["TEST"] * 12, "Model": [str(i) for i in range(12)],
            "Trim": ["BASE"] * 12, "price": [10000] * 12, "score": list(range(12))})
        result = diverse_top(rows, ["score"], [False])
        self.assertEqual(result.score.tolist(), [11, 10, 9, 8, 7])

    def test_question_draft_keeps_evidence_preferences_out_of_hard_filters(self):
        from car_app.requirements import QuestionDraft
        draft = QuestionDraft(use_case="Daily commute", max_price_usd=20000, evidence_period="latest_saved",
                              reference_profile_id="reference", avoided_components=("brakes",))
        contract = draft.contract("snapshot", 1)
        self.assertEqual(contract["schema_version"], 3)
        self.assertEqual(contract["avoided_components"], ["brakes"])
        self.assertNotIn("evidence_period", contract["hard_requirements"])
        self.assertNotIn("avoided_components", contract["hard_requirements"])
        self.assertFalse(contract["llm"]["api_calls_enabled"])

    def test_safety_ranking_requires_reference_weight_coverage(self):
        from car_app.discovery import applicable_safety
        reference = pd.DataFrame({"curb_weight_lb": [3200.], "curb_weight_lb_n": [5], "vin_count": [100]})
        result, reason = applicable_safety(pd.DataFrame(), None, reference)
        self.assertTrue(result.empty)
        self.assertIn("curb weight", reason)


class DiscoveryVisualizationTests(unittest.TestCase):
    def test_feature_scatter_preserves_observed_values_groups_and_distinct_axes(self):
        code = '''
import pandas as pd
from car_app.discovery import feature_scatter
profiles = pd.DataFrame({"profile_id": ["a", "b", "c", "d"],
    "wheelbase": [100., 110., None, 120.], "horsepower": [200., 300., 400., float("inf")],
    "length": [180., 190., 200., 210.], "subgroup": ["Group 1", "Group 2", "Partial", "Group 2"]})
feature_scatter(profiles, ["wheelbase", "horsepower", "length"], ["b"], "features")
'''
        with patch("car_app.discovery.map_chart") as draw:
            app = AppTest.from_string(code).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="features_x").value, "wheelbase")
            self.assertEqual(app.selectbox(key="features_y").value, "horsepower")
            points = draw.call_args.args[0]
            self.assertEqual(points.entity_id.tolist(), ["a", "b"])
            self.assertEqual(points.x.tolist(), [100., 110.])
            self.assertEqual(points.y.tolist(), [200., 300.])
            self.assertEqual(points.cluster_label.tolist(), ["Group 1", "Group 2"])
            self.assertEqual(draw.call_args.args[2], ["b"])
            self.assertEqual(draw.call_args.kwargs["axis_titles"], ("Wheelbase (in)", "Horsepower (hp)"))
            self.assertIn("2 profiles lack", " ".join(c.value for c in app.caption))
            app.selectbox(key="features_x").set_value("horsepower").run()
            self.assertFalse(app.exception)
            self.assertNotEqual(app.selectbox(key="features_y").value, "horsepower")
            app.selectbox(key="features_y").set_value("length").run()
            self.assertFalse(app.exception)
            self.assertEqual(draw.call_args.args[0].entity_id.tolist(), ["a", "b", "c"])

    def test_cells_preserve_unknown_zero_raw_units_and_observation_counts(self):
        from car_app.discovery import measure_cells, specification_summary

        profiles = pd.DataFrame({"cluster": [0, 0, -1, -1], "description": ["Compact"] * 4,
            "status": ["assigned", "assigned", "direct_fallback", "insufficient_observations"],
            "horsepower": [100., 300., 0., np.nan], "length": [180., np.nan, 150., np.nan]})
        original = profiles.copy(deep=True)
        _, summary = specification_summary(profiles, ["horsepower", "length"])
        self.assertEqual(len(summary), 3)
        self.assertEqual(set(summary.index), {"Specification group A", "Ungrouped — sufficient data",
                                             "Ungrouped — limited or conflicting data"})
        group = summary.loc[summary.index.str.startswith("Specification group")]
        self.assertEqual(group.horsepower.iloc[0], 200.)
        self.assertEqual(group.length_observed.iloc[0], 1)
        cells = measure_cells(summary, ["horsepower", "length"], {}, bounds=([100., 160.], [150., 200.]))
        self.assertTrue(cells.loc[cells.value.eq(200), "shade"].eq(1).all())
        self.assertIn("Unknown", cells.display.tolist())
        self.assertIn("0", cells.display.tolist())
        self.assertTrue(cells.loc[cells.measure.eq("length") & cells.value.eq(180), "observed_profiles"].eq(1).all())
        pd.testing.assert_frame_equal(original, profiles)

    def test_text_similarity_uses_saved_vectors_and_supported_distinct_families(self):
        from car_app.discovery import text_neighbors

        profiles = pd.DataFrame({"family_id": ["q", "same", "limited", "near", "far", "zero"],
            "family_block": ["QUERY", "QUERY", "LIMITED", "NEAR", "FAR", "ZERO"], "identity_complete": True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({"family_id": profiles.family_id, "supported": [True, True, False, True, True, True]}).to_parquet(path / "family_features.parquet")
            pd.DataFrame({"family_id": profiles.family_id, "vector_row": range(6)}).to_parquet(path / "family_vector_index.parquet")
            np.save(path / "family_vectors.npy", [[1., 0.], [1., 0.], [1., 0.], [.8, .6], [0., 1.], [0., 0.]])
            result = text_neighbors(path, profiles, "q")
            self.assertEqual(result.family_id.tolist(), ["near", "far"])
            self.assertAlmostEqual(result.cosine_similarity.iloc[0], .8)
            self.assertEqual(text_neighbors(path, profiles.loc[profiles.family_id.ne("near")], "q").family_id.tolist(), ["far"])
            self.assertTrue(text_neighbors(path, profiles, "limited").empty)

    def test_component_and_safety_views_render_without_projection_or_text_vectors(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = root / "text"
            text.mkdir()
            (text / "manifest.json").write_text(json.dumps({"audits": {"latest_saved": {
                "cutoff": "2026-10-28", "maximum_evidence_date": "2026-09-23"}}}))
            profiles = pd.DataFrame({"profile_id": ["p", "r"], "family_id": ["f", "g"], "fold": [0, 1],
                "ModelYear": [2020, 2020], "Make": ["TEST", "OTHER"], "Model": ["One", "Two"],
                "Trim": [None, None], "category": "Sedan", "identity_complete": True, "family_block": ["TEST|ONE", "OTHER|TWO"]})
            components = pd.DataFrame({"family_id": ["f", "f"], "source": ["complaint", "recall"], "record_count": [2, 2],
                "usable_text_count": [2, 2], "supported": [False, True], "brakes_share": [0., 0.], "brakes_count": [0, 0],
                "powertrain_share": [.5, .5], "powertrain_count": [1, 1]})
            safety = profiles.iloc[:1].assign(vehicle_id="v", variant_description="2020 TEST One", rating_era="post_2011",
                OverallSideCrashRating=5., RolloverRating=np.nan, SidePoleCrashRating=4.)
            catalog = SimpleNamespace(path=root / "bundle", text=text, profiles=profiles, safety=safety, grain="model",
                receipt={"completed_at": "2026-10-29", "periods": ["latest_saved"]},
                scope=lambda source, filters: profiles.copy(), components=lambda period: components.copy())
            for view in ("Complaints", "Recalls", "Safety"):
                with self.subTest(view=view), patch("car_app.discovery.active_catalog", return_value=catalog), \
                     patch("car_app.discovery.stat_key", return_value=()), \
                     patch("car_app.discovery.family_records", return_value=pd.DataFrame(columns=["family_id", "source"])):
                    code = "from car_app.discovery import nhtsa_dashboard\nfrom car_app.data import Filters\nnhtsa_dashboard(None, Filters(), view=" + repr(view) + ")"
                    app = AppTest.from_string(code, default_timeout=20).run()
                    self.assertFalse(app.exception)
                    self.assertEqual(len(app.get("vega_lite_chart")), 1)
                    display_key = "v6_nhtsa_safety_display" if view == "Safety" else "v6_nhtsa_display"
                    self.assertEqual(app.segmented_control(key=display_key).value, "Overview")
                    app.segmented_control(key=display_key).set_value("Compare vehicles").run()
                    self.assertFalse(app.exception)
                    self.assertTrue(app.info)
                    app.multiselect[0].set_value(["v"] if view == "Safety" else ["f", "g"]).run()
                    app.segmented_control(key=display_key).set_value("Compare vehicles").run()
                    self.assertFalse(app.exception)
                    self.assertGreater(len(app.get("vega_lite_chart")), 0)
                    self.assertNotIn("Advanced exploration", app.segmented_control(key=display_key).options)
                    self.assertFalse(any(widget.label in ("Pattern", "Plot axes") for widget in app.segmented_control))



class AgentTests(unittest.TestCase):
    @staticmethod
    def review(conversation):
        ids = list(conversation.context.listings)[:6]
        conversation.tool_result(lambda: conversation.context.recommendation_evidence(ids))

    def setUp(self):
        from car_app.agent import Settings
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = fixture_snapshot(Path(self.temp.name))
        self.categories = {"makes": ["TOYOTA"], "body_types": ["Sedan"], "fuels": ["Gasoline"],
                           "drivetrains": ["FWD"], "transmissions": [], "vehicle_categories": {}}
        self.requirements = QuestionDraft("Daily commute", max_price_usd=20000).contract(self.source.identity, 1)
        stub = patch("car_app.agent.settings", return_value=Settings("fixture-only"))
        stub.start()
        self.addCleanup(stub.stop)

    @staticmethod
    def provider_result(answer):
        from types import SimpleNamespace
        return SimpleNamespace(final_output=answer, context_wrapper=SimpleNamespace(usage=SimpleNamespace(
            requests=1, input_tokens=100, output_tokens=50, total_tokens=150)))

    def test_conversation_two_turn_revision_and_qualifying_cohort_receipt(self):
        from car_app.agent import AgentAnswer, PreferenceChange, RankedCandidate, run_turn, markdown_receipt
        async def provider(c, inputs, config):
            offered = json.loads(inputs[0]["content"].split("Current application context: ", 1)[1])
            self.assertIn("Lower advertised mileage", offered["preference_options"]["priorities"])
            self.assertIn("Not decided", offered["preference_options"]["use_case"])
            self.assertEqual(offered["data_dictionary"]["source"], "DATA_DICTIONARY.md")
            price_definition = next(row["definition"] for row in offered["data_dictionary"]["definitions"]
                                    if row["definition"].startswith("`price` |"))
            self.assertIn("Advertised price in USD", price_definition)
            if "$15,000" in c.user_message:
                c.update([PreferenceChange(field="max_price_usd", operation="set", value=15000,
                                           statement="maximum $15,000")])
            c.search()
            self.review(c)
            rows = list(c.context.listings.values())
            return self.provider_result(AgentAnswer(message="Historical results", questions=[],
                listings=[RankedCandidate(candidate_id=r["candidate_id"], reason="Observed catalog facts", tradeoff="Condition is unverified.", comparison="Fits the stated price priority.",
                                          source_refs=[r["source_refs"][0]]) for r in rows], cohorts=[]))
        first = run_turn(self.source, self.requirements, 1, self.categories, "Show results", [], {}, provider=provider)
        self.assertIsNone(first["telemetry"]["failure_category"])
        self.assertEqual(set(first["candidates"]), {"listing:a", next(k for k in first["candidates"] if k.startswith("cohort:"))})
        cohort = next(r for r in first["candidates"].values() if "qualifying_listings" in r)
        self.assertEqual((cohort["qualifying_listings"], cohort["median_price"]), (1, 10000))
        second = run_turn(self.source, first["requirements"], first["revision"], self.categories,
                          "Use maximum $15,000", [], {}, provider=provider)
        self.assertEqual(second["revision"], 2)
        self.assertEqual([r["candidate_id"] for r in second["answer"]["listings"]], ["listing:a"])
        json.dumps(second, allow_nan=False)
        self.assertIn(second["snapshot_identity"], markdown_receipt(second))
        self.assertIn("listing:a:facts", markdown_receipt(second))
        self.assertEqual(second["evidence_reviewed_ids"], ["listing:a"])
        self.assertEqual(first["data_dictionary"]["sha256"], second["data_dictionary"]["sha256"])

    def test_performance_retrieval_does_not_hide_stronger_eligible_vehicles_behind_cheap_models(self):
        from car_app.agent_context import RecommendationContext
        listings = pl.read_parquet(self.source.directory / "used_cars.parquet").with_columns(
            pl.when(pl.col("vin") == "b").then(600.).otherwise(100.).alias("horsepower"),
            pl.when(pl.col("vin") == "b").then(49000).otherwise(pl.col("price")).alias("price"))
        decoded = pl.read_parquet(self.source.directory / "NHTSA_vPIC.parquet").with_columns(
            pl.when(pl.col("vin") == "b").then(pl.lit("Performance model")).otherwise(pl.col("Model")).alias("Model"))
        for index in range(15):
            vin = f"cheap{index}"
            listings = pl.concat([listings, listings.head(1).with_columns(
                pl.lit(vin).alias("vin"), pl.lit(vin).alias("listing_id"))])
            decoded = pl.concat([decoded, decoded.head(1).with_columns(
                pl.lit(vin).alias("vin"), pl.lit(f"Cheap model {index}").alias("Model"))])
        listings.write_parquet(self.source.directory / "used_cars.parquet")
        decoded.write_parquet(self.source.directory / "NHTSA_vPIC.parquet")
        source = Snapshot(self.source.directory)
        requirements = QuestionDraft("Enthusiast / weekend driving", max_price_usd=50000,
            max_mileage=50000, min_seats=4, priorities=("Performance", "Safety")).contract(source.identity, 1)
        context = RecommendationContext(source, requirements, {})
        result = context.search()
        self.assertEqual(result["retrieval"]["primary_sort"], "Advertised horsepower")
        self.assertEqual(result["listings"][0]["candidate_id"], "listing:b")
        self.assertEqual(result["listings"][0]["price"], 49000)
        self.assertTrue(all(row["price"] <= 50000 for row in result["listings"]))

    def test_unreviewed_or_undelivered_evidence_cannot_produce_ai_recommendations(self):
        from car_app.agent import Conversation, AgentAnswer, RankedCandidate, run_turn
        async def provider(c, inputs, config):
            c.search()
            return self.provider_result(AgentAnswer(message="Unreviewed winner", questions=[], cohorts=[], listings=[
                RankedCandidate(candidate_id="listing:a", reason="Fits", tradeoff="Unknown", comparison="Leads", source_refs=["listing:a:facts"])]))
        result = run_turn(self.source, self.requirements, 1, self.categories, "Recommend", [], {}, provider=provider)
        self.assertEqual(result["telemetry"]["failure_category"], "invalid_response")
        self.assertNotIn("Unreviewed winner", json.dumps(result))
        c = Conversation(self.source, self.requirements, 1, self.categories, {})
        c.search()
        with patch("car_app.agent.MAX_CONTEXT_CHARS", 1):
            with self.assertRaisesRegex(ValueError, "Context budget"):
                self.review(c)
        self.assertFalse(c.context.reviewed_ids)

    def test_rejected_final_response_gets_one_validated_correction(self):
        from car_app.agent import AgentAnswer, RankedCandidate, run_turn

        calls = []
        async def provider(c, inputs, config):
            calls.append(inputs)
            if len(calls) == 1:
                c.search()
            else:
                self.assertIn("Retrieve recommendation evidence", inputs[-1]["content"])
                self.review(c)
            answer = AgentAnswer(message="Historical result", questions=[], cohorts=[], listings=[
                RankedCandidate(candidate_id="listing:a", reason="Fits the budget", tradeoff="Condition is unverified",
                                comparison="Meets the stated requirements", source_refs=["listing:a:facts"])])
            result = self.provider_result(answer)
            result.to_input_list = lambda: [{"role": "user", "content": "Prior context"}]
            return result

        result = run_turn(self.source, self.requirements, 1, self.categories, "Recommend", [], {}, provider=provider)
        self.assertIsNone(result["telemetry"]["failure_category"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["telemetry"]["requests"], 2)
        self.assertEqual(result["evidence_reviewed_ids"], ["listing:a"])
        self.assertIn("Retrieve recommendation evidence for every final listing before answering",
                      result["telemetry"]["validation_errors"])

    def test_response_correction_stops_after_second_invalid_response(self):
        from car_app.agent import AgentAnswer, RankedCandidate, run_turn

        calls = []
        async def provider(c, inputs, config):
            calls.append(1)
            if len(calls) == 1:
                c.search()
            answer = AgentAnswer(message="Unreviewed", questions=[], cohorts=[], listings=[
                RankedCandidate(candidate_id="listing:a", reason="Fits", tradeoff="Unknown", comparison="Leads",
                                source_refs=["listing:a:facts"])])
            result = self.provider_result(answer)
            result.to_input_list = lambda: [{"role": "user", "content": "Prior context"}]
            return result

        result = run_turn(self.source, self.requirements, 1, self.categories, "Recommend", [], {}, provider=provider)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["telemetry"]["failure_category"], "invalid_response")
        self.assertNotIn("Unreviewed", json.dumps(result))

    def test_structured_safety_is_retrieved_without_text_bundle_and_unrated_stays_unknown(self):
        from car_app.agent_context import evidence_for
        row = self.source.catalog().filter(pl.col("vin") == "a").collect().to_dicts()[0]
        evidence, _ = evidence_for(self.source, row, self.requirements, None)
        self.assertEqual(evidence["safety"]["total_variants"], 1)
        self.assertIsNone(evidence["safety"]["variants"][0]["OverallRating"])
        self.assertEqual(evidence["nhtsa"]["status"], "unavailable")

    def test_oof_does_not_borrow_old_champion_explanations(self):
        from car_app.agent_context import saved_price_explanations, selected_price_models
        config = {"price_run_selection": {"runs": {"full": "full/tree-full"}}}
        with patch("car_app.artifacts.saved_shap") as shap:
            self.assertIsNone(selected_price_models(config)["catboost"])
            self.assertEqual(saved_price_explanations(config), {})
            shap.assert_not_called()

    def test_invalid_ids_references_duplicates_and_provider_failure_use_valid_fallback(self):
        from car_app.agent import AgentAnswer, RankedCandidate, run_turn
        for failure in ("id", "reference", "duplicate", "provider", "units"):
            with self.subTest(failure=failure):
                async def provider(c, inputs, config):
                    c.search()
                    self.review(c)
                    if failure == "provider":
                        raise RuntimeError("private provider body must not appear")
                    row = RankedCandidate(candidate_id="listing:invented" if failure == "id" else "listing:a",
                        reason="20,000 miles" if failure == "units" else "Unvalidated model text must not appear",
                        tradeoff="Condition is unverified.", comparison="Lower observed price.",
                        source_refs=["invented"] if failure == "reference" else ["listing:a:facts"])
                    return self.provider_result(AgentAnswer(message="Unvalidated model text must not appear", questions=[],
                        listings=[row, row] if failure == "duplicate" else [row], cohorts=[]))
                result = run_turn(self.source, self.requirements, 1, self.categories, "Show results", [], {}, provider=provider)
                self.assertIsNotNone(result["telemetry"]["failure_category"])
                self.assertEqual({r["candidate_id"] for r in result["answer"]["listings"]}, {"listing:a"})
                self.assertNotIn("private provider body", json.dumps(result))
                self.assertNotIn("Unvalidated model text", json.dumps(result))
                self.assertNotIn("20,000 miles", json.dumps(result))

    def test_preference_patch_is_atomic_grounded_and_does_not_infer_unknown_facts(self):
        from car_app.agent import PreferenceChange, apply_changes
        invalid = PreferenceChange(field="max_price_usd", operation="set", value=15000, statement="under $10,000")
        with self.assertRaises(ValueError):
            apply_changes(self.requirements, [invalid], "under $10,000", self.categories, self.source.identity, 1)
        contradictory = [PreferenceChange(field="min_year", operation="set", value=2020, statement="2020 to 2010"),
                         PreferenceChange(field="max_year", operation="set", value=2010, statement="2020 to 2010")]
        with self.assertRaises(ValueError):
            apply_changes(self.requirements, contradictory, "2020 to 2010", self.categories, self.source.identity, 1)
        self.assertEqual(self.requirements["hard_requirements"], {"max_price_usd": 20000})
        removed = apply_changes(self.requirements, [PreferenceChange(field="max_price_usd", operation="remove", value=None,
            statement="remove budget")], "remove budget", self.categories, self.source.identity, 1)
        self.assertNotIn("max_price_usd", removed["hard_requirements"])

    def test_missing_key_deadline_and_changed_source_do_not_display_stale_ids(self):
        from car_app.agent import AgentAnswer, Settings, run_turn
        with patch("car_app.agent.settings", return_value=Settings("")), patch("car_app.agent._provider_run") as provider:
            result = run_turn(self.source, self.requirements, 1, self.categories, "Show results", [], {})
            provider.assert_not_called()
            self.assertEqual(result["telemetry"]["failure_category"], "missing_credentials")
            self.assertEqual(len(result["answer"]["listings"]), 1)
        async def expired(c, inputs, config):
            c.deadline = 0
            c.check()
        result = run_turn(self.source, self.requirements, 1, self.categories, "Show results", [], {}, provider=expired)
        self.assertEqual(result["telemetry"]["failure_category"], "timeout")
        async def changed(c, inputs, config):
            c.search()
            self.review(c)
            path = self.source.directory / "used_cars.parquet"
            frame = pl.read_parquet(path).with_columns(pl.lit(999999).alias("price"))
            frame.write_parquet(path)
            return self.provider_result(AgentAnswer(message="stale", questions=[], listings=[], cohorts=[]))
        result = run_turn(self.source, self.requirements, 1, self.categories, "Show results", [], {}, provider=changed)
        self.assertFalse(result["candidates"])
        self.assertFalse(result["answer"]["listings"])

    def test_price_selection_only_uses_pinned_full_catboost(self):
        from car_app.agent_context import selected_price_models
        from car_app.recommendation_price import oof
        selection = {"kind": oof.KIND, "run": "recommendation_price/current",
                     "manifest_sha256": "one", "validation_sha256": "two"}
        config = {"recommendation_price": selection, "active_price": {"model": "ridge"}}
        with patch("car_app.recommendation_price.load_bundle", return_value=(Path("fixture"), {"options": {"folds": 10}}, pd.DataFrame())), \
             patch("car_app.recommendation_price.oof.read_json", return_value={"pooled": {"mae": 1500}}):
            models = selected_price_models(config)
        self.assertEqual(models["primary"]["model"], "catboost")
        self.assertEqual(models["primary"]["feature_set"], "full")
        self.assertEqual(models["primary"]["run"], selection["run"])
        self.assertNotIn("linear", models)
        with patch("car_app.recommendation_price.load_bundle", side_effect=ValueError("Source changed")):
            self.assertIsNone(selected_price_models(config)["primary"])

    def test_sdk_tools_are_strict_and_validate_user_statement_before_mutation(self):
        import asyncio
        from car_app.agent import Conversation, agent_tools
        from agents.tool_context import ToolContext
        from agents import RunConfig
        c = Conversation(self.source, self.requirements, 1, self.categories, {})
        c.user_message = "Make my budget $15,000"
        tools = agent_tools(c)
        self.assertTrue(all(t.strict_json_schema for t in tools))
        tool = tools[0]
        args = {"changes": [{"field": "max_price_usd", "operation": "set", "value": 15000,
                             "statement": "budget $15,000"}]}
        ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id="fixture", tool_arguments=json.dumps(args),
                          run_config=RunConfig(tracing_disabled=True))
        result = json.loads(asyncio.run(tool.on_invoke_tool(ctx, json.dumps(args))))
        self.assertEqual(result["hard_requirements"]["max_price_usd"], 15000)
        result = json.loads(asyncio.run(tool.on_invoke_tool(ctx, json.dumps({"changes": [{**args["changes"][0], "value": 1}]}))))
        self.assertIn("error", result)
        self.assertEqual(c.requirements["hard_requirements"]["max_price_usd"], 15000)

    def test_native_chat_submits_once_preserves_other_session_and_invalidates_reference(self):
        from car_app.agent import AgentAnswer, run_turn
        from car_app.agent_ui import sync_reference
        async def provider(c, inputs, config):
            return self.provider_result(AgentAnswer(message="What maximum advertised USD budget should we use?",
                questions=["What is your budget?"], listings=[], cohorts=[]))
        result = run_turn(self.source, {}, 0, self.categories, "Help me", [], {}, provider=provider)
        script = '''import streamlit as st
from car_app.agent_ui import render_chat
from car_app.state import RecommendationPanel, SimilarityPanel
from types import SimpleNamespace
st.session_state.setdefault('independent', RecommendationPanel())
st.session_state.setdefault('similar', SimilarityPanel())
render_chat(st.session_state['independent'], SOURCE, {}, CATEGORIES, key='chat')
'''
        script = script.replace("SOURCE", f"SimpleNamespace(identity={self.source.identity!r})").replace("CATEGORIES", repr(self.categories))
        with patch("car_app.agent_ui.run_turn", return_value=result) as called:
            app = AppTest.from_string(script).run()
            self.assertFalse(app.exception)
            called.assert_not_called()
            app.chat_input[0].set_value("Help me").run()
            self.assertFalse(app.exception)
            self.assertEqual(called.call_count, 1)
            app.run()
            self.assertEqual(called.call_count, 1)
            self.assertFalse(app.session_state['similar'].messages)
            self.assertTrue(app.get("download_button"))
        panel = SimilarityPanel(messages=[{"role": "user", "content": "previous"}], chat_requirements={"previous": True}, agent_result=result)
        sync_reference(panel, "new-reference")
        self.assertFalse(panel.messages)
        self.assertFalse(panel.chat_requirements)
        self.assertIsNone(panel.agent_result)

    def test_empty_v6_reference_scope_clears_previous_chat_reference(self):
        from types import SimpleNamespace
        catalog = SimpleNamespace(scope=lambda source, filters: pd.DataFrame({"identity_complete": pd.Series([], dtype=bool)}))
        script = '''import streamlit as st
from car_app.discovery import similarity_dashboard
from car_app.data import Filters
st.session_state['v6_similarity_reference'] = 'old-reference'
similarity_dashboard(None, Filters())
'''
        with patch("car_app.discovery.active_catalog", return_value=catalog), patch("car_app.discovery.discovery_caption"):
            app = AppTest.from_string(script).run()
            self.assertFalse(app.exception)
            self.assertNotIn("v6_similarity_reference", app.session_state)

    def test_oof_prices_preserve_usd_without_final_model_substitution(self):
        from car_app.recommendation_price import price_evidence
        selection = {"run": "fixture", "model": "catboost", "feature_set": "full"}
        bank = pd.DataFrame({"vin": ["a"], "prediction": [15000.], "fold": [3], "predictor_sha256": ["fold-model"]})
        models = {"catboost": selection, "unavailable": []}
        with patch("car_app.recommendation_price.load_bundle", return_value=(Path("fixture"), {}, bank)), \
             patch("car_app.recommendation_price.oof.inference_rows") as inference:
            result, warnings = price_evidence(pl.DataFrame({"vin": ["a"]}), models)
            self.assertEqual(result["listing:a"]["catboost"]["predicted_price_usd"], 15000.)
            self.assertEqual(result["listing:a"]["catboost"]["prediction_kind"], "out_of_fold")
            self.assertFalse(warnings)
            inference.assert_not_called()
            with patch("car_app.recommendation_price.pl.read_parquet", return_value=pl.DataFrame({"vin": ["a", "b"]})):
                result, warnings = price_evidence(pl.DataFrame({"vin": ["b"]}), models)
            self.assertEqual(result["listing:b"], {})
            self.assertTrue(warnings)
            inference.assert_not_called()

    def test_oof_cache_is_bound_to_files_and_rejects_changed_pin(self):
        from car_app import recommendation_price as pricing
        selection = {"kind": pricing.oof.KIND, "run": "fixture", "manifest_sha256": "one", "validation_sha256": "one"}
        receipt = {"passed": True, "manifest_sha256": "one", "code": {}}
        pricing._load.cache_clear()
        self.addCleanup(pricing._load.cache_clear)
        signature = (("source", (1, 1)),)
        with patch.object(pricing.oof, "digest", return_value="one"), \
             patch.object(pricing.oof, "read_json", return_value=receipt), \
             patch.object(pricing.oof, "code_hashes", return_value={}), \
             patch.object(pricing.oof, "validate_bundle", return_value={"reduced_budget": False}) as validate, \
             patch.object(pricing.pd, "read_parquet", return_value=pd.DataFrame()), \
             patch.object(pricing, "bundle_signature", return_value=signature) as signatures:
            pricing._load(json.dumps(selection), signature)
            pricing._load(json.dumps(selection), signature)
            self.assertEqual(validate.call_count, 1)
            signature = (("source", (2, 2)),)
            signatures.return_value = signature
            pricing._load(json.dumps(selection), signature)
            self.assertEqual(validate.call_count, 2)
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                pricing._load(json.dumps({**selection, "manifest_sha256": "wrong"}), signature)

    def test_oof_activation_rejects_smoke_bundle_without_touching_registry(self):
        from car_app.recommendation_price import activate
        with patch("car_app.recommendation_price.oof.validate_bundle", return_value={"reduced_budget": True}), \
             patch("car_app.recommendation_price.oof.read_json", return_value={"passed": True}), \
             patch("car_app.recommendation_price.oof.write_json") as write:
            with self.assertRaisesRegex(ValueError, "full-budget"):
                activate(Path("MODEL_OUTPUT/fixture"))
            write.assert_not_called()

    def test_price_evidence_renders_without_extra_inference_or_provider_call(self):
        from car_app.agent import AgentAnswer, RankedCandidate, run_turn
        async def provider(c, inputs, config):
            c.search()
            self.review(c)
            return self.provider_result(AgentAnswer(message="Historical results", questions=[],
                listings=[RankedCandidate(candidate_id="listing:a", reason="Catalog facts", tradeoff="Condition is unverified.", comparison="Fits the price priority.", source_refs=["listing:a:facts"])], cohorts=[]))
        result = run_turn(self.source, self.requirements, 1, self.categories, "Show results", [], {}, provider=provider)
        model = {"model": "catboost", "run": "fixture", "feature_set": "full", "oof_mae_usd": 1000,
                 "prediction_kind": "out_of_fold", "fold": 2, "predicted_price_usd": 12000}
        result["source_refs"]["listing:a:price_evidence"] = {"models": {"catboost": model}}
        script = '''import streamlit as st
from car_app.agent_ui import render_result
from car_app.state import RecommendationPanel
st.session_state.setdefault('panel', RecommendationPanel())
render_result(RESULT, 'local', st.session_state['panel'], None, {})
'''.replace("RESULT", repr(result))
        with patch("car_app.recommendation_price.oof.inference_rows") as inference, patch("car_app.agent._provider_run") as remote:
            app = AppTest.from_string(script).run()
            self.assertFalse(app.exception)
            remote.assert_not_called()
            inference.assert_not_called()
            self.assertFalse(any(button.key == "local_prices" for button in app.button))

    def test_six_ai_cards_submit_once_and_sync_refined_preferences(self):
        from car_app.agent import AgentAnswer, RankedCandidate, run_turn, PreferenceChange
        from car_app.data import filter_options, query_snapshot
        listings = pl.read_parquet(self.source.directory / "used_cars.parquet")
        decoded = pl.read_parquet(self.source.directory / "NHTSA_vPIC.parquet")
        for index in range(1, 8):
            listings = pl.concat([listings, listings.head(1).with_columns(
                pl.lit(f"extra{index}").alias("vin"), pl.lit(f"extra{index}").alias("listing_id"),
                pl.lit(10000 + index * 1000).cast(pl.Int64).alias("price"))])
            decoded = pl.concat([decoded, decoded.head(1).with_columns(
                pl.lit(f"extra{index}").alias("vin"), pl.lit(f"Model {index}").alias("Model"))])
        listings.write_parquet(self.source.directory / "used_cars.parquet")
        decoded.write_parquet(self.source.directory / "NHTSA_vPIC.parquet")
        self.source = Snapshot(self.source.directory)
        self.requirements = QuestionDraft("Daily commute", max_price_usd=20000,
            priorities=("Lower purchase price",)).contract(self.source.identity, 1)
        async def provider(c, inputs, config):
            if c.user_message == "Lower my budget to $15,000":
                c.update([PreferenceChange(field="max_price_usd", operation="set", value=15000,
                    statement="budget to $15,000")])
            c.search()
            self.review(c)
            rows = list(c.context.listings.values())[:6]
            return self.provider_result(AgentAnswer(message="Here are the AI-selected matches.", questions=[],
                listings=[RankedCandidate(candidate_id=row["candidate_id"],
                    reason=f"Observed price ${row['price']:,.0f} meets your budget and lower-price priority.",
                    tradeoff="Vehicle condition is unverified.", comparison="Consider its observed price against your budget.",
                    source_refs=[row["source_refs"][0]]) for row in rows], cohorts=[]))
        result = run_turn(self.source, self.requirements, 1, self.categories, "Show recommendations", [], {}, provider=provider)
        self.assertIsNone(result["telemetry"]["failure_category"])
        self.assertEqual(len(result["answer"]["listings"]), 6)
        self.assertEqual(len({result["candidates"][r["candidate_id"]]["Model"] for r in result["answer"]["listings"]}), 6)
        filter_options.clear(); query_snapshot.clear()
        self.addCleanup(filter_options.clear); self.addCleanup(query_snapshot.clear)
        def request(*args, **kwargs):
            return run_turn(*args, **kwargs, provider=provider)
        with patch("car_app.data.snapshot", return_value=self.source), \
             patch("car_app.data.MIN_MAKE_ROWS", 1), \
             patch("car_app.model_registry.registry", return_value={}), \
             patch("car_app.discovery.active_catalog", return_value=None), \
             patch("car_app.artifacts.diagnostic_inventory", return_value=(None, pd.DataFrame(), None, None)), \
             patch("car_app.agent_ui.run_turn", side_effect=request) as called, \
             patch("car_app.agent._provider_run") as remote:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.session_state["rec_mileage_known"] = False
            app.session_state["rec_mileage"] = 100000.
            app.session_state["rec_seats_known"] = False
            app.session_state["rec_seats"] = 5
            app.switch_page("app_pages/recommendation.py").run()
            called.assert_not_called()
            self.assertIsNone(app.selectbox(key="rec_mileage").value)
            self.assertIsNone(app.number_input(key="rec_seats").value)
            app.selectbox(key="rec_budget").select("$20,000")
            app.multiselect(key="rec_priorities").set_value(["Lower purchase price"])
            app.multiselect(key="rec_bodies").set_value(["Sedan"])
            app.selectbox(key="rec_mileage").set_value("50,000")
            app.number_input(key="rec_seats").set_value(5)
            app.button(key="recommendation_search").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(called.call_count, 1)
            self.assertEqual(len(app.session_state['recommendation_session'].agent_result['answer']['listings']), 6)
            self.assertEqual(sum(h.value.startswith("Best AI match:") for h in app.subheader), 1)
            self.assertEqual(sum(h.value == "Other AI recommendations" for h in app.subheader), 1)
            self.assertEqual(sum(m.value.startswith("**Main tradeoff:**") for m in app.markdown), 6)
            app.run()
            self.assertEqual(called.call_count, 1)
            app.chat_input(key="recommendation_chat_input").set_value("Lower my budget to $15,000").run()
            self.assertFalse(app.exception)
            self.assertEqual(called.call_count, 2)
            self.assertEqual(app.selectbox(key="rec_budget").value, "$15,000")
            self.assertEqual(app.selectbox(key="rec_mileage").value, "50,000")
            hard = app.session_state['recommendation_session'].requirements['hard_requirements']
            self.assertEqual(hard['max_price_usd'], 15000)
            self.assertEqual(hard['body_types'], ('Sedan',))
            self.assertEqual(hard['min_seats'], 5)
            self.assertEqual(hard['max_mileage'], 50000)
            app.run()
            self.assertEqual(called.call_count, 2)
            remote.assert_not_called()

    def test_slate_rejects_seventh_vehicle_duplicate_families_and_unexplained_tradeoffs(self):
        from car_app.agent import Conversation, AgentAnswer, RankedCandidate
        from types import SimpleNamespace
        c = Conversation(self.source, self.requirements, 1, self.categories, {})
        def candidate(cid, model):
            return {"candidate_id": cid, "vin": cid, "Make": "TOYOTA", "Model": model, "source_refs": [cid + ":facts"]}
        rows = {f"c{i}": candidate(f"c{i}", f"Model {i}") for i in range(7)}
        c.context = SimpleNamespace(listings=rows, models=None)
        def selection(cid, **kwargs):
            return RankedCandidate(candidate_id=cid, reason="Meets the observed-price requirement.",
                tradeoff="Condition is unknown.", comparison="Fits the primary priority.", source_refs=[cid + ":facts"], **kwargs)
        with self.assertRaisesRegex(ValueError, "shortlist limits"):
            c.validate(AgentAnswer(message="", questions=[], listings=[selection(cid) for cid in rows], cohorts=[]))
        rows['c1']['Model'] = " model 0 "
        with self.assertRaisesRegex(ValueError, "distinct Make/Model"):
            c.validate(AgentAnswer(message="", questions=[], listings=[selection('c0'), selection('c1')], cohorts=[]))
        with self.assertRaises(ValueError):
            RankedCandidate(candidate_id='c0', reason='Fits', tradeoff='', comparison='Fits', source_refs=['c0:facts'])
        from car_app.requirements import budget_amount
        self.assertEqual(budget_amount("$18,000"), 18000)
        self.assertEqual(budget_amount("18k"), 18000)
        with self.assertRaises(ValueError):
            budget_amount("$18,000 per month")

    def test_cohort_evidence_keeps_family_scope_without_copying_member_cluster(self):
        from car_app.agent_context import RecommendationContext
        context = RecommendationContext(self.source, self.requirements, {})
        context.search()
        cid = next(iter(context.cohorts))
        with patch("car_app.agent_context.evidence_for", return_value=({"specifications": {"cluster": "one member"},
            "youtube": {"grain": "Make"}, "nhtsa": {"family_id": "2020/TOYOTA/COROLLA"}}, [])) as retrieve:
            evidence = context.evidence([cid])[0]
            self.assertEqual(retrieve.call_args.args[1]["vin"], "a")
            self.assertNotIn("specifications", evidence)
            self.assertIn("not trim-specific", evidence["applicability"])
            self.assertIn(evidence["source_ref"], context.cohorts[cid]["source_refs"])


if __name__ == "__main__":
    unittest.main()


class TreeLinearPricePolicyTests(unittest.TestCase):
    def test_selects_each_family_by_validation_not_test_error(self):
        from car_app.recommendation_price import select_model
        candidates = pd.DataFrame([
            {"feature_set": "full", "model": "catboost", "run": "full/a-full", "validation_mae": 10., "test_mae": 90.},
            {"feature_set": "base", "model": "xgboost", "run": "base/b-full", "validation_mae": 20., "test_mae": 1.},
            {"feature_set": "full", "model": "ridge", "run": "full/c-full", "validation_mae": 30., "test_mae": 1.},
            {"feature_set": "full", "model": "multiple_linear_regression", "run": "full/d-full", "validation_mae": 25., "test_mae": 80.}])
        with patch("car_app.artifacts.diagnostic_inventory", return_value=({}, candidates, None, None)), \
             patch("car_app.model_registry.digest", return_value="hash"), \
             patch("car_app.recommendation_price.check_checkpoint"):
            models = select_model({"recommendation_price": {"kind": "best_tree_linear_v1"}})
        self.assertEqual(models["tree"]["model"], "catboost")
        self.assertEqual(models["linear"]["model"], "multiple_linear_regression")
        self.assertEqual(models["primary"], models["tree"])

    def test_incompatible_checkpoints_are_unavailable_without_blocking_recommendations(self):
        from car_app.recommendation_price import select_model
        candidates = pd.DataFrame([
            {"feature_set": "full", "model": "catboost", "run": "full/a-full", "validation_mae": 10.},
            {"feature_set": "full", "model": "ridge", "run": "full/b-full", "validation_mae": 20.}])
        with patch("car_app.artifacts.diagnostic_inventory", return_value=({}, candidates, None, None)), \
             patch("car_app.model_registry.digest", return_value="hash"), \
             patch("car_app.recommendation_price.check_checkpoint", side_effect=ValueError("Source changed: used_cars.parquet")):
            models = select_model({"recommendation_price": {"kind": "best_tree_linear_v1"}})
        self.assertIsNone(models["tree"])
        self.assertIsNone(models["linear"])
        self.assertIsNone(models["primary"])
        self.assertEqual(len(models["unavailable"]), 2)
        self.assertTrue(all("Source changed" in warning for warning in models["unavailable"]))

    def test_checkpoint_bank_rejects_a_fitting_vin(self):
        from car_app.recommendation_price import checkpoint_prices
        rows = pd.DataFrame({"vin": ["training"], "prediction": [100.]})
        split = pd.DataFrame({"vin": ["training", "testing"], "partition": ["fit", "test"]})
        with patch("car_app.recommendation_price.check_checkpoint", return_value=Path("fixture")), \
             patch("car_app.recommendation_price.pd.read_parquet", side_effect=[rows, split]):
            with self.assertRaisesRegex(ValueError, "held-out"):
                checkpoint_prices({"model": "catboost"})

    def test_both_estimates_keep_prediction_kind_without_averaging(self):
        from car_app.recommendation_price import price_evidence
        models = {"tree": {"model": "catboost"}, "linear": {"model": "ridge"}, "unavailable": []}
        heldout = pd.DataFrame({"vin": ["a"], "predicted_price_usd": [15000.]})
        missing = pd.DataFrame({"vin": pd.Series(dtype=str), "predicted_price_usd": pd.Series(dtype=float)})
        with patch("car_app.recommendation_price.checkpoint_prices", side_effect=[heldout, missing]), \
             patch("car_app.model_registry.price_estimates", return_value=(pl.DataFrame({"vin": ["a"], "predicted_price_usd": [14000.]}), None)):
            result, warnings = price_evidence(pl.DataFrame({"vin": ["a"]}), models)
        self.assertFalse(warnings)
        self.assertEqual(set(result["listing:a"]), {"tree", "linear"})
        self.assertEqual(result["listing:a"]["tree"]["prediction_kind"], "held_out_test")
        self.assertEqual(result["listing:a"]["linear"]["prediction_kind"], "supplemental_inference")
