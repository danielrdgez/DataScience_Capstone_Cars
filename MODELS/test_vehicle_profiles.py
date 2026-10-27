"""Consequential research controls, using synthetic data only."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from MODELS import vehicle_profiles as research


def fixture(families=30):
    listings, decodes, complaints = [], [], []
    for family in range(families):
        kind = family % 2
        for trim in range(2):
            for vehicle in range(6):
                vin = f"synthetic-{family}-{trim}-{vehicle}"
                listings.append({"vin": vin, "maximum_seating": "5 seats", "wheelbase": 100. + kind * 30,
                    "length": 150. + kind * 60, "height": 55. + kind * 20, "horsepower": 100. + kind * 200,
                    "engine_displacement": 1500. + kind * 2000, "fuel_type": "Gasoline",
                    "city_fuel_economy": 25., "highway_fuel_economy": 30.,
                    "wheel_system": "FWD" if not kind else "AWD", "transmission": "A",
                    "major_options": "['Backup Camera']" if vehicle else "[]", "dealer_zip": 10000 + family,
                    "body_type": "Sedan" if not kind else "SUV / Crossover"})
                decodes.append({"vin": vin, "ModelYear": 2015, "Make": "TEST", "Model": f"MODEL{family}", "Trim": f"T{trim}",
                                "EngineHP": 100. + kind * 200, "DisplacementCC": 1500. + kind * 2000})
        for incident in range(25):
            complaints.append({"odiNumber": family * 100 + incident, "model_year": 2015, "make": "TEST", "model": f"MODEL{family}",
                "products": json.dumps([{"type": "Vehicle", "productYear": "2015", "productMake": "TEST", "productModel": f"MODEL{family}"}]),
                "summary": ("engine stalling power failure driving road " if not kind else "brakes stopping steering issue road ") + f"incident number {incident} family {family}",
                "components": "ENGINE" if not kind else "BRAKES", "dateComplaintFiled": "2018-03-01", "dateOfIncident": "2018-02-01"})
    return {"used_cars": pd.DataFrame(listings), "vpic": pd.DataFrame(decodes), "complaints": pd.DataFrame(complaints)}


class ResearchControls(unittest.TestCase):
    def test_options_require_valid_strings_and_preserve_absence_of_evidence(self):
        expected = [(None, "missing"), ("[broken]", "malformed"), ("[1]", "malformed"),
                    ("{}", "malformed"), ("[]", "valid_empty"), ("['Backup Camera']", "populated")]
        for value, state in expected:
            self.assertEqual(research.option_list(value)[0], state)

    def test_embedding_chunks_preserve_long_text_and_spherical_centers(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False, truncation=False):
                return list(text) + (["S", "E"] if add_special_tokens else [])

            def decode(self, tokens, skip_special_tokens=True):
                return "".join(tokens)

            def num_special_tokens_to_add(self, pair=False):
                return 2
        text = "abcdefghijklmnopqrstuvwxyz" * 10
        tokenizer = Tokenizer()
        chunks = research.token_chunks(text, tokenizer, 64, overlap=8)
        self.assertTrue(all(len(tokenizer.encode(chunk, add_special_tokens=True)) <= 64 for chunk in chunks))
        reconstructed = chunks[0] + "".join(chunk[8:] for chunk in chunks[1:])
        self.assertEqual(reconstructed, text)
        values = research.normalize(np.array([[1., 0.], [1., .1], [0., 1.], [.1, 1.]]))
        centers = research.spherical_fit(values, 2, 3, 42)
        np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.)

    def test_vin_conflicts_heterogeneity_and_option_denominator(self):
        data = fixture(4)
        decodes = pd.concat([data["vpic"], data["vpic"].iloc[[0]].assign(Trim="OTHER")], ignore_index=True)
        data["used_cars"].loc[6:11, "horsepower"] = [100, 100, 100, 300, 300, 300]
        data["used_cars"].loc[1, "major_options"] = None
        cohorts, representatives, audit = research.prepare_vehicles(data["used_cars"], decodes, research.Config())
        self.assertEqual(audit["conflicting_decode_vins"], 1)
        self.assertEqual(cohorts.vin_count.sum(), len(data["used_cars"]) - 1)
        self.assertEqual(cohorts.iloc[0].option_valid_count, 4)
        self.assertEqual(cohorts.iloc[0].advertised_3, 1.)
        self.assertTrue(pd.isna(cohorts.iloc[1].horsepower))
        self.assertTrue(representatives.vin.isin(data["used_cars"].vin).all())
        with self.assertRaises(ValueError):
            research.prepare_vehicles(pd.concat([data["used_cars"], data["used_cars"].iloc[[0]]]), decodes, research.Config())

    def test_missing_numeric_representatives_are_explicit_without_warnings(self):
        data = fixture(4)
        data["used_cars"].loc[:5, ["maximum_seating", *research.NUMERIC[1:]]] = None
        # Include a wholly missing VIN in a cohort that still has usable candidates.
        data["used_cars"].loc[6, ["maximum_seating", *research.NUMERIC[1:]]] = None
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _, representatives, audit = research.prepare_vehicles(data["used_cars"], data["vpic"], research.Config())
        self.assertEqual(audit["representative_unavailable_cohorts"], 1)
        self.assertTrue(pd.isna(representatives.iloc[0].vin))
        self.assertEqual(representatives.iloc[0].representative_status, "unavailable_no_numeric_specs")
        self.assertNotEqual(representatives.iloc[1].vin, data["used_cars"].iloc[6].vin)

    def test_automated_mapping_checks_and_development_text_cleanup(self):
        data = fixture(30)
        cohorts, _, _ = research.prepare_vehicles(data["used_cars"], data["vpic"], research.Config())
        records, _, _ = research.prepare_complaints(data["complaints"], cohorts, research.Config())
        self.assertTrue(research.automatic_mapping_checks(records)["passed"])
        corrupted = records.copy()
        corrupted.loc[corrupted.index[0], "products"] = "[]"
        self.assertFalse(research.automatic_mapping_checks(corrupted)["passed"])
        fitting = pd.DataFrame({"Make": ["TESTBRAND"], "Model": ["STEERING"]})
        words = research.topic_stop_words(fitting)
        self.assertIn("testbrand", words)
        self.assertIn("stated", words)
        self.assertIn("jb", words)
        self.assertNotIn("steering", words)
        self.assertNotIn("heldoutbrand", words)

    def test_applicability_windows_conflicts_and_cross_split_text(self):
        data = fixture(30)
        cohorts, _, _ = research.prepare_vehicles(data["used_cars"], data["vpic"], research.Config())
        raw = data["complaints"].copy()
        raw.loc[0, "products"] = "[]"
        raw.loc[1, "products"] = '[{"type":"Equipment"}]'
        conflicting = raw.iloc[[2]].assign(summary="different narrative for same incident")
        raw = pd.concat([raw, conflicting], ignore_index=True)
        raw.loc[[3, 28], "summary"] = "same copied narrative across different vehicle families"
        records, _, audit = research.prepare_complaints(raw, cohorts, research.Config())
        self.assertFalse(records.loc[records.odiNumber.isin([0, 1, 2]), "primary_eligible"].any())
        self.assertEqual(audit["conflicting_narratives"], 1)
        self.assertFalse(research.primary_window(2022, pd.Timestamp("2025-01-01")))
        self.assertTrue(research.primary_window(2021, pd.Timestamp("2025-12-31")))
        self.assertFalse(research.primary_window(2015, pd.Timestamp("2020-01-01")))
        self.assertFalse(research.primary_window(2015, pd.Timestamp("2014-12-31")))
        for _, group in records.groupby("text_hash"):
            if group.development.nunique() > 1:
                self.assertFalse(group.text_eligible.any())

    def test_development_only_preprocessing_and_family_denominators(self):
        data = fixture(30)
        cohorts, _, _ = research.prepare_vehicles(data["used_cars"], data["vpic"], research.Config())
        fields, eligible, _ = research.select_fields(cohorts)
        encoder = research.fit_encoding(cohorts[cohorts.development & eligible], fields)
        before = encoder.transform(cohorts[fields].replace({None: np.nan}))
        changed = cohorts.copy()
        changed.loc[~changed.development, "wheelbase"] = 1000000
        after = encoder.transform(changed[fields].replace({None: np.nan}))
        np.testing.assert_allclose(before[cohorts.development], after[cohorts.development])
        cohorts["eligible"] = True
        cohorts["cluster"] = np.tile([0, 1], len(cohorts) // 2)
        families = research.family_profiles(cohorts)
        np.testing.assert_allclose(families[["cluster_0", "cluster_1"]].sum(axis=1), 1)
        self.assertTrue((families.trim_count == 2).all())
        records, _, _ = research.prepare_complaints(data["complaints"], cohorts, research.Config())
        summary = research.family_complaints(pd.concat([records, records]))
        self.assertTrue((summary.complaint_count == 25).all())
        np.testing.assert_allclose(research.normalized_mixtures(np.array([[1., 3.], [0., 0.]])), [[.25, .75], [0., 0.]])

    def test_review_gate_and_bootstrap_support(self):
        template = pd.DataFrame({"review_id": ["a", "b"], "mapping_status": ["exact_supported", "unsupported"]})
        self.assertFalse(research.validate_mapping_review(template, None)[0])
        reviewed = template.assign(judgment=["incorrect", "correct"])
        self.assertEqual(research.validate_mapping_review(template, reviewed)[1], "mapping_rule_blocked")
        reviewed.judgment = ["correct", "incorrect"]
        self.assertTrue(research.validate_mapping_review(template, reviewed)[0])
        self.assertIsNone(research.paired_intervals(np.ones(10), np.arange(10), 20, 42))
        self.assertEqual(research.paired_intervals(np.ones(20), np.arange(20), 20, 42), [1., 1.])

    def test_incident_union_and_topic_denominators(self):
        records = pd.DataFrame({"ModelYear": [2015] * 4, "Make": ["TEST"] * 4,
            "Model": ["A", "A", "A", "B"], "odiNumber": [1, 1, 2, 3],
            "components": ["ENGINE", "BRAKES", "STEERING", None], "text_hash": ["a", "a", "b", "c"]})
        weights = pd.DataFrame({"topic_0": [.8, .3], "topic_1": [.2, .7]},
                               index=pd.Index(["a", "b"], dtype="str"))
        result = research.family_complaints(records, weights).set_index("Model")
        self.assertEqual(result.loc["A", "complaint_count"], 2)
        self.assertEqual(result.loc["A", "topic_complaint_count"], 2)
        self.assertEqual(result.loc["A", "component_powertrain"], .5)
        self.assertEqual(result.loc["A", "component_brakes"], .5)
        self.assertAlmostEqual(result.loc["A", "topic_0"], .55)
        self.assertEqual(result.loc["B", "topic_complaint_count"], 0)
        self.assertTrue(pd.isna(result.loc["B", "topic_0"]))

    def test_blocker_is_explicit_and_stages_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            with patch.object(research, "load_sources", side_effect=PermissionError("selected snapshot denied")):
                research.prepare(Path(directory) / "source", run)
            self.assertEqual(research.read_json(run / "prepare/manifest.json")["status"], "access_denied")
            research.vehicles(run)
            self.assertEqual(research.read_json(run / "vehicles/manifest.json")["status"], "skipped")
            with self.assertRaises(FileExistsError):
                research.prepare(Path(directory), run)

    def test_synthetic_stages_and_automatic_comparison(self):
        config = research.Config(starts=2, stability_refits=2, topic_refits=2, clusters=(2,), topics=(2,), bootstrap=20, smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch.object(research, "load_sources", return_value=(fixture(), {"synthetic": True})):
                research.prepare(directory, run, config)
            research.vehicles(run)
            self.assertEqual(research.read_json(run / "vehicles/manifest.json")["status"], "accepted_partition")
            research.complaints(run)
            research.compare(run)
            receipt = research.read_json(run / "compare/manifest.json")
            self.assertEqual(receipt["status"], "evaluated")
            self.assertEqual(receipt["mapping_status"], "automated_checks_passed")
            self.assertEqual(receipt["semantic_validation"], "not_established")
            self.assertTrue((run / "vehicles/cluster_cohorts.html").is_file())
            membership = pd.read_csv(run / "vehicles/cohort_membership.csv")
            self.assertFalse(membership.duplicated(research.COHORT).any())
            self.assertTrue(membership.loc[membership.cluster >= 0, ["profile_axis_1", "profile_axis_2"]].notna().all().all())
            reviewed = pd.read_csv(run / "prepare/mapping_review_template.csv")
            reviewed["judgment"] = "correct"
            reviewed.to_csv(run / "synthetic_mapping_review.csv", index=False)
            result = research.compare(run, run / "synthetic_mapping_review.csv")
            self.assertEqual(research.read_json(result / "manifest.json")["status"], "evaluated")
            predictions = pd.read_csv(result / "predictions.csv")
            self.assertEqual(set(predictions.variant), {"baseline", "clusters", "continuous"})
            self.assertTrue(np.isfinite(predictions.prediction).all())
            self.assertTrue((pd.read_csv(result / "paired_intervals.csv").interval_status == "fewer_than_20_blocks").all())

    def test_unknown_units_and_multi_family_complaints(self):
        data = fixture(4)
        data["vpic"]["DisplacementCC"] *= 1000
        cohorts, _, audit = research.prepare_vehicles(data["used_cars"], data["vpic"], research.Config())
        self.assertTrue(cohorts.engine_displacement.isna().all())
        self.assertFalse(audit["unit_checks"]["engine_displacement"]["verified"])
        raw = data["complaints"].copy()
        raw.loc[25, "odiNumber"] = 0
        raw.loc[25, "summary"] = raw.loc[0, "summary"]
        records, _, _ = research.prepare_complaints(raw, cohorts, research.Config())
        self.assertTrue(records.loc[records.odiNumber == 0, "multi_family"].all())
        self.assertFalse(records.loc[records.odiNumber == 0, "primary_eligible"].any())

    def test_snapshot_contract_and_topic_review_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(3)
            tables, definitions = {}, []
            for name, filename in research.FILES.items():
                path = root / filename
                data[name].to_parquet(path, index=False)
                schema = pl.scan_parquet(path).collect_schema()
                tables[name] = {"file": filename, "rows": len(data[name]), "schema": {k: str(v) for k, v in schema.items()}}
                definitions.append(f"## `{path.stem}`\n\nOne source row. **{len(data[name])} rows; {len(schema)} columns**.\n")
            research.write_json(root / "manifest.json", {"tables": tables})
            dictionary = root / "dictionary.md"
            dictionary.write_text("\n".join(definitions), encoding="utf-8")
            sources, receipt = research.load_sources(root, dictionary)
            self.assertEqual(len(sources["used_cars"]), 36)
            self.assertEqual(receipt["used_cars.parquet"]["sha256"], research.file_hash(root / "used_cars.parquet"))
            dictionary.write_text(dictionary.read_text().replace("36 rows", "35 rows"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                research.load_sources(root, dictionary)
            rows = [{"count": 2, "topic": topic, "kind": kind, "text_hash": f"{topic}-{kind}-{i}", "judgment": "coherent"}
                    for topic in range(2) for kind in ["representative", "random"] for i in range(5)]
            review = pd.DataFrame(rows)
            review.drop(columns="judgment").to_csv(root / "topic_review_template.csv", index=False)
            pd.DataFrame([{"count": 2, "median_similarity": .8}]).to_csv(root / "topic_candidates.csv", index=False)
            self.assertEqual(research.choose_topics(root, review), 2)
            review.loc[review.topic == 0, "judgment"] = "mixed"
            self.assertIsNone(research.choose_topics(root, review))


class SegmentationControls(unittest.TestCase):
    def test_verified_market_representation_compares_common_fitting_population(self):
        from MODELS import vehicle_segmentation as segmentation
        data = fixture(20)
        kind = data["used_cars"].horsepower > 150
        data["used_cars"]["price"] = np.where(kind, 30000., 10000.)
        data["used_cars"]["mileage"] = np.where(kind, 200000., 100000.)
        config = research.Config(starts=2, stability_refits=2, clusters=(2,), smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            research.write_json(source / "manifest.json", {"synthetic": True, "units": {"mileage": "miles"}})
            run = Path(directory) / "run"
            with patch.object(research, "load_sources", return_value=(data, {"synthetic": True})):
                research.prepare(source, run, config)
            segmentation.segments(run)
            self.assertEqual(research.read_json(run / "segments/market/manifest.json")["status"], "evaluated")
            comparisons = pd.read_csv(run / "segments/representation_agreement.csv")
            self.assertGreater(len(comparisons), 0)
            self.assertTrue(comparisons.common_population_refit_ari.notna().all())
            research.verified_stage(run, "segments")

    def test_missing_complaint_evidence_is_not_a_zero_profile(self):
        from MODELS import vehicle_segmentation as segmentation
        data = fixture(4)
        data["complaints"] = data["complaints"][data["complaints"].model != "MODEL0"]
        config = research.Config(clusters=(2,), starts=2, stability_refits=2, smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(research, "load_sources", return_value=(data, {"synthetic": True})):
                research.prepare(directory, directory, config)
            segmentation.complaint_profiles(directory)
            families = pd.read_parquet(Path(directory) / "complaint_profiles/family_features.parquet")
            missing = families[families.Model == "MODEL0"]
            self.assertEqual(missing.complaint_count.iloc[0], 0)
            self.assertTrue(missing[[c for c in missing if c.startswith("component_")]].isna().all().all())
            receipt = research.read_json(Path(directory) / "complaint_profiles/components_20/manifest.json")
            self.assertEqual(receipt["status"], "no_supported_partition")

    def test_mileage_unit_gate_uses_source_metadata(self):
        from MODELS import vehicle_segmentation as segmentation
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            research.write_json(source / "manifest.json", {"tables": {"used_cars": {"file": "used_cars.parquet"}}})
            self.assertFalse(segmentation.mileage_unit_receipt({"source_dir": directory})["verified"])
            research.write_json(source / "manifest.json", {"tables": {"used_cars": {"file": "used_cars.parquet", "units": {"mileage": "miles"}}}})
            receipt = segmentation.mileage_unit_receipt({"source_dir": directory})
            self.assertTrue(receipt["verified"])
            self.assertEqual(receipt["unit"], "miles")

    def test_market_support_and_structural_missingness(self):
        from MODELS import vehicle_segmentation as segmentation
        data = fixture(4)
        data["used_cars"]["price"] = [10000., 20000., -1., 0., np.nan, 30000.] * 8
        data["used_cars"]["mileage"] = [0., 100., -1., np.nan, 200., 300.] * 8
        data["used_cars"].loc[:5, "fuel_type"] = "Electric"
        data["used_cars"].loc[:5, "engine_displacement"] = None
        cohorts, _, _ = research.prepare_vehicles(data["used_cars"], data["vpic"], research.Config())
        self.assertEqual(cohorts.iloc[0].price_n, 3)
        self.assertEqual(cohorts.iloc[0].price_median, 20000.)
        self.assertEqual(cohorts.iloc[0].mileage_n, 4)
        self.assertEqual(cohorts.iloc[0].engine_displacement_state, "not_applicable")
        # Even contradictory numeric source evidence cannot become applicable electric displacement.
        cohorts.loc[0, "engine_displacement"] = 2500.
        cohorts.loc[0, "engine_displacement_n"] = 6
        prepared, _, eligible, _ = segmentation.research_fields(cohorts)
        self.assertTrue(eligible.iloc[0])
        self.assertTrue(pd.isna(prepared.iloc[0].engine_displacement))
        _, _, market_eligible, _ = segmentation.research_fields(cohorts, market=True)
        self.assertFalse(market_eligible.any())

    def test_multi_component_incidents_and_family_denominators(self):
        from MODELS import vehicle_segmentation as segmentation
        raw = pd.DataFrame([{**dict(zip(research.FAMILY, [2015, "A", "B"])), "odiNumber": 1, "components": text}
                            for text in ["ENGINE", "BRAKES", "ENGINE"]])
        incidents = segmentation.incident_features(raw)
        self.assertEqual(len(incidents), 1)
        self.assertTrue(incidents.component_powertrain.iloc[0])
        self.assertTrue(incidents.component_brakes.iloc[0])
        data = fixture(4)
        cohorts, _, _ = research.prepare_vehicles(data["used_cars"], data["vpic"], research.Config())
        cohorts["cluster"] = [0, -1] * 4
        cohorts["eligible"] = cohorts.cluster >= 0
        profiles = segmentation.family_mixtures(cohorts, [])
        self.assertTrue(np.allclose(profiles.cluster_0 + profiles.unassigned_share, 1))

    def test_selection_is_method_local_and_prefers_smaller_near_tie(self):
        from MODELS import vehicle_segmentation as segmentation
        table = pd.DataFrame([{"k": 2, "silhouette": .30, "accepted_broad": True},
                              {"k": 3, "silhouette": .31, "accepted_broad": True},
                              {"k": 4, "silhouette": .6, "accepted_broad": False}])
        self.assertEqual(segmentation.choose_candidate(table, "broad").k, 2)
        self.assertIsNone(segmentation.choose_candidate(table.assign(accepted_broad=False), "broad"))

    def test_fold_preprocessing_ignores_validation_values(self):
        from MODELS import vehicle_segmentation as segmentation
        data = fixture(12)
        cohorts, _, _ = research.prepare_vehicles(data["used_cars"], data["vpic"], research.Config())
        prepared, fields, eligible, _ = segmentation.research_fields(cohorts)
        config = research.Config(clusters=(2,), starts=2, smoke=True)
        model = research.fit_partition(prepared[eligible], fields, "kmeans", 2, config, 42)
        fit_blocks = {"TEST|MODEL0", "TEST|MODEL1", "TEST|MODEL2", "TEST|MODEL3"}
        before, receipt = segmentation.fold_profiles(cohorts, model, fields, config, fit_blocks)
        changed = cohorts.copy()
        outside = ~(changed.Make + "|" + changed.Model).isin(fit_blocks)
        changed.loc[outside, "horsepower"] = 999999.
        after, changed_receipt = segmentation.fold_profiles(changed, model, fields, config, fit_blocks)
        self.assertEqual(receipt, changed_receipt)
        inside = (before.Make + "|" + before.Model).isin(fit_blocks)
        pd.testing.assert_frame_equal(before.loc[inside], after.loc[inside])

    def test_new_stages_preserve_legacy_and_record_market_blocker(self):
        from MODELS import vehicle_segmentation as segmentation
        config = research.Config(starts=2, stability_refits=2, topic_refits=2, clusters=(2,), topics=(2,), bootstrap=20, smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch.object(research, "load_sources", return_value=(fixture(60), {"synthetic": True})):
                research.prepare(directory, run, config)
            segmentation.segments(run)
            receipt = research.read_json(run / "segments/market/manifest.json")
            self.assertEqual(receipt["status"], "blocked_unverified_mileage_unit")
            segmentation.complaint_profiles(run)
            membership = pd.read_parquet(run / "complaint_profiles/components_20/membership.parquet")
            self.assertTrue((membership.cluster >= 0).any())
            uncertainty = pd.read_csv(run / "complaint_profiles/components_20/assignment_uncertainty.csv")
            self.assertTrue(np.allclose(uncertainty.groupby(research.FAMILY).assignment_frequency.sum(), 1))
            segmentation.relationships(run)
            for path in (run / "relationships").glob("*/fold_receipts.json"):
                for fold in research.read_json(path):
                    self.assertFalse(set(fold["training_blocks"]) & set(fold["validation_blocks"]))
            research.verified_stage(run, "relationships")
            self.assertFalse((run / "vehicles").exists())


class TaxonomyControls(unittest.TestCase):
    def test_notebook_bootstrap_refreshes_stale_pipeline_modules(self):
        from MODELS import vehicle_segmentation as segmentation
        from MODELS import vehicle_taxonomy as taxonomy
        notebook = json.loads(Path(__file__).with_name("Vehicle_Profiles.ipynb").read_text(encoding="utf-8"))
        bootstrap, = [cell for cell in notebook["cells"]
                      if "pipeline-bootstrap" in cell.get("metadata", {}).get("tags", [])]
        # Reproduce a kernel that imported vehicle_profiles before these fields existed.
        del research.RESEARCH_NUMERIC
        try:
            with self.assertRaises(AttributeError):
                segmentation.research_fields(pd.DataFrame())
            namespace = {}
            exec("".join(bootstrap["source"]), namespace)
            self.assertEqual(research.RESEARCH_NUMERIC, research.NUMERIC + research.EXTRA_NUMERIC)
            self.assertIs(segmentation.r, research)
            self.assertIs(taxonomy.r, research)
            self.assertIs(namespace["segmentation"], segmentation)
        finally:
            import importlib
            importlib.reload(research)

    def test_label_scope_conflicts_and_overrides(self):
        from MODELS import vehicle_taxonomy as t
        definitions, registry = t.load_taxonomy()
        cohorts = pd.DataFrame([
            {"ModelYear": y, "Make": "TEST", "Model": "CAR", "Trim": trim, "body_type": "COUPE", "horsepower": 1000, "price_median": 500, "vin_count": 1}
            for y, trim in [(2017, "BASE"), (2018, "BASE"), (2017, "SPORT")]])
        source = dict(registry["sources"][0], id="test", make="TEST", model="CAR", year_start=2017, year_end=2017,
                      all_trims=False, trims=["BASE"], claims=[{"dimension": "performance_character", "label": "sports_car"}])
        mappings = t.draft_mappings({"sources": [source]})
        summary, labels, queue = t.assign_labels(cohorts, definitions, mappings)
        self.assertEqual(list(summary.performance_character_labels), ["sports_car", "", ""])
        self.assertTrue(summary.market_positioning_labels.eq("").all())
        self.assertEqual(len(summary), 3)
        self.assertEqual(summary.iloc[0].vin_count, 1)
        luxury = dict(source, id="luxury", all_trims=True, claims=[{"dimension": "market_positioning", "label": "luxury"}])
        economy = dict(luxury, id="economy", claims=[{"dimension": "market_positioning", "label": "economy"}])
        summary, labels, queue = t.assign_labels(cohorts, definitions, t.draft_mappings({"sources": [luxury, economy]}))
        self.assertTrue(summary.market_positioning_labels.eq("").all())
        self.assertIn("conflicting_sources", queue.reason.to_list())
        reviewed = dict(luxury, id="reviewed", all_trims=False, trims=["BASE"], status="reviewed")
        summary, labels, queue = t.assign_labels(cohorts, definitions, t.draft_mappings({"sources": [luxury, economy, reviewed]}))
        self.assertEqual(summary.iloc[0].market_positioning_labels, "luxury")
        self.assertEqual(summary.iloc[2].market_positioning_labels, "")

    def test_powertrain_body_and_torque_evidence(self):
        from MODELS import vehicle_taxonomy as t
        definitions, registry = t.load_taxonomy()
        mappings = t.draft_mappings(registry)
        cohorts = pd.DataFrame([
            {"ModelYear": 2020, "Make": "TEST", "Model": "TRUCK", "Trim": "LUXURY", "body_type": "Pickup Truck", "fuel_type": "Gasoline", "electrification": None},
            {"ModelYear": 2020, "Make": "TEST", "Model": "EV", "Trim": "BASE", "body_type": "SUV / Crossover", "fuel_type": "Electric", "electrification": "BEV (Battery Electric Vehicle)"},
            {"ModelYear": 2020, "Make": "TEST", "Model": "PHEV", "Trim": "BASE", "body_type": "Sedan", "fuel_type": "Gasoline", "electrification": "PHEV (Plug-in Hybrid Electric Vehicle)"}])
        summary, _, _ = t.assign_labels(cohorts, definitions, mappings)
        self.assertEqual(summary.iloc[0].body_family, "pickup")
        self.assertTrue(summary.market_positioning_labels.eq("").all())
        self.assertEqual(summary.powertrain_labels.to_list(), ["", "battery_electric", "plug_in_hybrid"])
        self.assertAlmostEqual(t.parse_torque("200 lb-ft @ 1,750 RPM"), 271.16358966)
        self.assertEqual(t.parse_torque("300 Nm"), 300)
        self.assertTrue(np.isnan(t.parse_torque("300")))
        self.assertTrue(np.isnan(t.parse_torque("300 nXm")))
        self.assertTrue(np.isnan(t.parse_torque("12,34 lb-ft")))
        self.assertEqual(t.parse_torque("300 N.m"), 300)

    def test_audit_cannot_validate_itself_or_change_frozen_identity(self):
        from MODELS import vehicle_taxonomy as t
        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp) / "frozen.csv"
            reviewed = Path(tmp) / "reviewed.csv"
            row = {"ModelYear": 2017, "Make": "TEST", "Model": "CAR", "Trim": "BASE", "dimension": "market_positioning", "label": "luxury",
                   "mapping_id": "test", "source_url": "https://example.com", "evidence": "source", "correct": None, "reviewer": "", "review_evidence": ""}
            pd.DataFrame([row]).to_csv(frozen, index=False)
            pd.DataFrame([row]).to_csv(reviewed, index=False)
            self.assertEqual(t.evaluate_audit(frozen, reviewed).iloc[0].status, "pending_independent_review")
            row.update(correct=True, reviewer="human", review_evidence="independent evidence")
            pd.DataFrame([row]).to_csv(reviewed, index=False)
            self.assertEqual(t.evaluate_audit(frozen, reviewed).iloc[0].status, "provisional_small_sample")
            row["ModelYear"] = 2018
            pd.DataFrame([row]).to_csv(reviewed, index=False)
            with self.assertRaises(ValueError):
                t.evaluate_audit(frozen, reviewed)

    def test_peers_exclude_self_and_preserve_body_year_and_observed_support(self):
        from MODELS import vehicle_taxonomy as t
        from MODELS import vehicle_segmentation as segmentation
        sources = fixture(12)
        cohorts, _, _ = research.prepare_vehicles(sources["used_cars"], sources["vpic"], research.Config())
        definitions, registry = t.load_taxonomy()
        summary, labels, _ = t.assign_labels(cohorts, definitions, t.draft_mappings(registry))
        data, fields, eligible, _ = segmentation.research_fields(summary)
        data["eligible"] = eligible
        data["body_family"] = summary.body_family
        members = pd.DataFrame(columns=research.COHORT + ["experiment_id", "cluster"])
        peers = t.peer_table(summary, labels, data, fields, summary.iloc[:1][research.COHORT], members)
        self.assertFalse(peers.empty)
        self.assertTrue((peers.peer_id != peers.query_id).all())
        self.assertTrue(peers.year_window.le(10).all())
        self.assertTrue(peers.groupby(["query_id", "approach", "view"]).size().le(5).all())
        self.assertTrue(summary.loc[peers.peer_id].body_family.eq(summary.iloc[0].body_family).all())
        self.assertTrue(peers.shared_fields.ge(2).all())
        self.assertTrue(peers.cross_model.all())
        for _, group in peers.groupby(["query_id", "approach", "view"]):
            identities = summary.loc[group.peer_id, ["Make", "Model"]]
            self.assertFalse(identities.duplicated().any())
        diagnostics = t.paired_peer_diagnostics(peers, [0])
        self.assertTrue(diagnostics.paired_queries.eq(1).all())

    def test_taxonomy_subdivision_enforces_ten_cohort_floor(self):
        from MODELS import vehicle_taxonomy as t
        sources = fixture(40)
        cohorts, _, _ = research.prepare_vehicles(sources["used_cars"], sources["vpic"], research.Config())
        definitions, registry = t.load_taxonomy()
        summary, labels, _ = t.assign_labels(cohorts, definitions, t.draft_mappings(registry))
        scores = pd.DataFrame([{"method": method, "k": 2, "median_ari": .9, "silhouette": .3,
                                "minimum_count": 5, "accepted_broad": True, "accepted_small": True}
                               for method in ["kmeans", "gower"]])
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            with patch.object(t.s, "candidates", return_value=({}, scores)):
                _, _, membership = t.cluster_experiments(summary, labels, research.Config(), target, target / "absent.parquet")
            self.assertTrue(membership.empty)
            report = pd.read_csv(target / "experiments.csv")
            self.assertTrue(report.status.eq("no_supported_subdivision").all())
            comparison = pd.read_csv(target / "global/model_comparison.csv")
            self.assertFalse(comparison.accepted_broad.any())
            self.assertTrue(comparison.required_minimum_count.ge(10).all())

    def test_peer_year_window_relaxes_only_for_distinct_model_support(self):
        from MODELS import vehicle_taxonomy as t, vehicle_segmentation as segmentation
        definitions, registry = t.load_taxonomy()
        for divisor, expected_window in [(1, 10), (2, 5)]:
            with self.subTest(year_divisor=divisor):
                sources = fixture(14)
                family_number = sources["vpic"].Model.str.extract(r"(\d+)", expand=False).astype(int)
                sources["vpic"]["ModelYear"] = 2015 + family_number // divisor
                cohorts, _, _ = research.prepare_vehicles(sources["used_cars"], sources["vpic"], research.Config())
                summary, labels, _ = t.assign_labels(cohorts, definitions, t.draft_mappings(registry))
                data, fields, eligible, _ = segmentation.research_fields(summary)
                data["eligible"] = eligible
                data["body_family"] = summary.body_family
                memberships = pd.DataFrame(columns=research.COHORT + ["experiment_id", "cluster"])
                peers = t.peer_table(summary, labels, data, fields, summary.iloc[:1][research.COHORT], memberships)
                self.assertTrue(peers.year_window.eq(expected_window).all())
                self.assertTrue(peers.groupby("approach").size().eq(5).all())
                self.assertTrue((summary.loc[peers.peer_id].ModelYear - 2015).abs().le(expected_window).all())

    def test_size_claim_requires_compatible_observed_body_evidence(self):
        from MODELS import vehicle_taxonomy as t
        definitions, registry = t.load_taxonomy()
        cohorts = pd.DataFrame([{"ModelYear": 2020, "Make": "TEST", "Model": "FAMILY", "Trim": trim,
                                "body_type": body} for trim, body in [("SUV", "SUV / Crossover"), ("TRUCK", "Pickup Truck"), ("UNKNOWN", None)]])
        source = dict(registry["sources"][0], id="size", make="TEST", model="FAMILY", year_start=2020, year_end=2020,
                      all_trims=True, claims=[{"dimension": "size_segment", "label": "large_suv"}, {"dimension": "market_positioning", "label": "luxury"}])
        summary, labels, queue = t.assign_labels(cohorts, definitions, t.draft_mappings({"sources": [source]}))
        self.assertEqual(summary.size_segment_labels.to_list(), ["large_suv", "", ""])
        self.assertTrue(summary.market_positioning_labels.eq("luxury").all())
        self.assertEqual(len(queue[queue.dimension.eq("size_segment")]), 2)

    def test_clustering_reuse_rejects_changed_specifications_and_categories(self):
        from dataclasses import asdict
        from MODELS import vehicle_taxonomy as t
        sources = fixture(40)
        cohorts, _, _ = research.prepare_vehicles(sources["used_cars"], sources["vpic"], research.Config())
        definitions, registry = t.load_taxonomy()
        summary, labels, _ = t.assign_labels(cohorts, definitions, t.draft_mappings(registry))
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path(tmp) / "previous"
            cluster = previous / "clusters"
            cluster.mkdir(parents=True)
            summary.to_parquet(previous / "cohorts.parquet", index=False)
            labels.to_parquet(previous / "labels.parquet", index=False)
            (previous / "execution_vehicle_taxonomy.py").write_bytes(Path(t.__file__).read_bytes())
            research.finish_stage(cluster, "completed", reduced_budget=False, configuration=asdict(research.Config()),
                input_cohorts_sha256=research.file_hash(previous / "cohorts.parquet"),
                input_labels_sha256=research.file_hash(previous / "labels.parquet"),
                execution_source_sha256=research.file_hash(previous / "execution_vehicle_taxonomy.py"))
            changed = summary.copy()
            changed.loc[0, "horsepower"] += 50
            with self.assertRaisesRegex(ValueError, "specification inputs changed"):
                t.reuse_clustering(previous, changed, labels, research.Config(), Path(tmp) / "destination")
            claim = dict(registry["sources"][0], id="category", make="TEST", model="MODEL0", year_start=2015, year_end=2015,
                         all_trims=True, claims=[{"dimension":"market_positioning", "label":"luxury"}])
            changed, changed_labels, _ = t.assign_labels(cohorts, definitions, t.draft_mappings({"sources":[claim]}))
            with self.assertRaisesRegex(ValueError, "view memberships changed"):
                t.reuse_clustering(previous, changed, changed_labels, research.Config(), Path(tmp) / "destination")


class VariantTaxonomyControls(unittest.TestCase):
    def test_powertrain_conflicts_are_resolved_per_vin(self):
        from MODELS import vehicle_taxonomy_v2 as v
        frame = pl.DataFrame({"fuel_type":["Gasoline","Electric","Gasoline","Electric","Gasoline"],
            "FuelTypePrimary":["Gasoline","Electric","Gasoline","Electric","Gasoline"],
            "FuelTypeSecondary":[None,None,"Electric",None,None],
            "ElectrificationLevel":["BEV (Battery Electric Vehicle)","BEV (Battery Electric Vehicle)","PHEV",None,None],
            "engine_displacement":[2500.,None,2000.,1200.,2500.],"DisplacementCC":[2500.,None,2000.,None,2500.]})
        self.assertEqual(v.powertrain_expressions(frame)["powertrain"].to_list(),
            ["conflicting","battery_electric","plug_in_hybrid","conflicting","unknown"])

    def test_parent_requires_population_support_and_no_competing_variant(self):
        from MODELS import vehicle_taxonomy_v2 as v
        for powers, counts, expected in [(["battery_electric","unknown"],[3,212],"insufficient_evidence"),
            (["battery_electric","combustion"],[99,1],"mixed"),(["battery_electric","conflicting"],[99,1],"conflicting"),
            (["battery_electric","unknown"],[90,10],"assigned")]:
            self.assertEqual(v.parent_state(pd.DataFrame({"powertrain":powers,"vin_count":counts}),"powertrain")[1],expected)

    def test_crosswalk_scope_and_alias_split_blocks(self):
        from MODELS import vehicle_taxonomy_v2 as v
        p=pd.DataFrame({"Make":["BMW"]*4,"Model":["328I","328I","335I","328I"],"ModelYear":[2011,2012,2012,2012],
            "Trim":["BASE"]*4,"body_style":["sedan","sedan","sedan","coupe"]})
        x=[dict(id="test",make="BMW",models=["328I","335I"],market_family="3 SERIES",year_start=2012,year_end=2012,body_styles=["sedan"])]
        out=v.resolve_families(p,pd.DataFrame(),x)
        self.assertEqual(out.family_resolved.to_list(),[False,True,True,False])
        self.assertEqual(out.family_block.nunique(),1)
        self.assertEqual(out.development.nunique(),1)

    def test_variant_peers_obey_powertrain_and_family_support(self):
        from MODELS import vehicle_taxonomy_v2 as v
        n=9
        d=pd.DataFrame({"profile_id":[f"p{i}" for i in range(n)],"parent_id":[f"c{i}" for i in range(n)],
            "Make":["TEST"]*n,"Model":[str(i) for i in range(n)],"ModelYear":[2015]*8+[2022],"Trim":["BASE"]*n,
            "family_block":["A","A","B","C","D","E","F","G","H"],"development":[True]*n,"eligible":[True]*n,
            "body_family":["passenger_car"]*n,"powertrain":["battery_electric"]*6+["hybrid","conflicting","battery_electric"],
            "wheelbase":np.arange(n)+100.,"height":np.arange(n)+50.,"horsepower":np.arange(n)+100.,"fuel_type":["ELECTRIC"]*n})
        labels=pd.DataFrame(columns=["profile_id","dimension","label"])
        memberships=pd.DataFrame(columns=["profile_id","experiment_id","cluster"])
        p=v.peers(d,["wheelbase","height","horsepower","fuel_type"],labels,memberships)
        q=p[(p.query_profile_id=="p0")&(p.approach=="direct_specs")]
        self.assertNotIn("p1",q.peer_profile_id.to_list())
        self.assertNotIn("p6",q.peer_profile_id.to_list())
        self.assertNotIn("p7",q.peer_profile_id.to_list())
        self.assertEqual(q.year_window.unique().tolist(),[10])
        self.assertEqual(len(q),5)

    def test_sparse_profiles_retain_semantic_evidence(self):
        from MODELS import vehicle_taxonomy_v2 as v
        definitions,registry=__import__('MODELS.vehicle_taxonomy',fromlist=['load_taxonomy']).load_taxonomy()
        p=pd.DataFrame([dict(profile_id="sparse",identity_complete=True,Make="MAZDA",Model="MX-5",market_family="MX-5",
            ModelYear=2015,Trim="BASE",body_style="convertible",body_family="passenger_car",vin_count=1)])
        _,labels,_=v.profile_labels(p,__import__('MODELS.vehicle_taxonomy',fromlist=['draft_mappings']).draft_mappings(registry),definitions)
        self.assertIn("sports_car",labels.label.to_list())

    def test_unverified_horsepower_can_fall_back_to_supported_torque(self):
        from MODELS import vehicle_taxonomy_v2 as v
        n=12
        profiles=pd.DataFrame({"development":[True]*n,"identity_complete":[True]*n,"vin_count":[6]*n,
            "powertrain":["unknown"]*n,"body_style":["sedan"]*n})
        for i,field in enumerate(v.NUMERIC):
            profiles[field]=np.arange(n,dtype=float)*(i+1)+100
            profiles[field+"_n"]=6
        for field in v.CATEGORICAL:
            profiles[field]=["A","B"]*(n//2)
            profiles[field+"_n"]=6
        data,fields,audit=v.select_fields(profiles,excluded_fields=("horsepower",))
        self.assertNotIn("horsepower",fields)
        self.assertIn("torque_nm",fields)
        self.assertEqual(audit.set_index("field").loc["horsepower","reason"],"unverified_source_scale")


class SimilarityV3Controls(unittest.TestCase):
    @staticmethod
    def profiles(families=40):
        from MODELS import vehicle_taxonomy_v2 as v2
        rng = np.random.default_rng(42)
        n = families * 2
        frame = pd.DataFrame({"profile_id": [f"p{i}" for i in range(n)], "parent_id": [f"c{i}" for i in range(n)],
            "ModelYear": [2015] * n, "Make": ["TEST"] * n, "Model": [f"M{i//2}" for i in range(n)],
            "Trim": [f"T{i%2}" for i in range(n)], "family_block": [f"TEST|M{i//2}" for i in range(n)],
            "identity_complete": True, "vin_count": 6, "powertrain": "unknown", "body_style": "sedan",
            "body_family": "passenger_car", "development": True})
        for field in v2.NUMERIC:
            frame[field] = 100 + rng.uniform(0, 100, n)
            frame[field + "_n"] = 6
        for field in v2.CATEGORICAL:
            frame[field] = ["A", "B"] * families
            frame[field + "_n"] = 6
        frame["fold"] = frame.family_block.map(__import__('MODELS.vehicle_similarity_v3', fromlist=['family_fold']).family_fold)
        return frame

    def test_observed_block_averages_ranges_and_support(self):
        from MODELS import vehicle_similarity_v3 as v
        fields = ["wheelbase", "height", "seating", "horsepower", "fuel_type"]
        training = pd.DataFrame({"wheelbase": [100., 200.], "height": [50., 100.], "seating": [2., 8.],
                                 "horsepower": [100., 400.], "fuel_type": ["A", "B"]})
        metric = v.fit_metric(training, fields)
        left = training.iloc[[0]].copy()
        right = left.copy()
        right["seating"] = np.nan
        right["horsepower"] = np.nan
        right["fuel_type"] = "B"
        self.assertAlmostEqual(v.distances(left, right, metric).item(), .5)
        right["height"] = np.nan
        self.assertTrue(np.isnan(v.distances(left, right, metric).item()))
        changed = training.copy()
        changed.loc[1, "wheelbase"] = 100000
        self.assertEqual(metric["bounds"]["wheelbase"], (101., 199.))
        self.assertAlmostEqual(v.distances(training.iloc[[0]], training.iloc[[0]], metric).item(), 0.)

    def test_balancing_retains_variants_and_bootstrap_multiplicity(self):
        from MODELS import vehicle_similarity_v3 as v
        frame = self.profiles()
        balanced = v.balanced_variants(frame, len(frame), 42)
        self.assertEqual(balanced.groupby("family_block").size().min(), 2)
        bootstrap = v.bootstrap_families(frame, 42)
        bootstrap["weight"] = v.family_weights(bootstrap)
        self.assertEqual(bootstrap._bootstrap_block.nunique(), frame.family_block.nunique())
        self.assertLess(bootstrap.family_block.nunique(), frame.family_block.nunique())
        self.assertTrue(np.allclose(bootstrap.groupby("_bootstrap_block").weight.sum(), 2.))

    def test_family_queries_are_unique_and_frozen_before_results(self):
        from dataclasses import replace
        from MODELS import vehicle_similarity_v3 as v
        frame = self.profiles()
        query = v.freeze_queries(frame, replace(v.Config(), queries=30))
        self.assertEqual(query.family_block.nunique(), 30)
        self.assertTrue(query.equals(v.freeze_queries(frame.sample(frac=1, random_state=9).sort_index(), replace(v.Config(), queries=30))))
        self.assertTrue((frame.groupby("family_block").fold.nunique() == 1).all())

    def test_cluster_penalty_is_soft_and_empty_queries_stay_in_denominator(self):
        from MODELS import vehicle_similarity_v3 as v
        frame = self.profiles(6)
        fields = ["wheelbase", "height", "horsepower", "fuel_type"]
        frame["eligible"] = True
        frame.loc[0, "eligible"] = False
        metric = v.fit_metric(frame.iloc[2:], fields)
        peers, status = v.retrieve(frame, ["p0", "p2"], fields, metric, {"p2": 0, "p4": 1}, v.Config())
        v.validate_peers(frame, peers, status)
        summary = v.diagnostics(status, peers).set_index("approach")
        self.assertEqual(summary.loc["direct_specs", "queries"], 2)
        self.assertEqual(summary.loc["direct_specs", "empty_list_share"], .5)
        self.assertEqual(set(peers[peers.approach.eq("direct_specs")].peer_profile_id), set(peers[peers.approach.eq("cluster_aware")].peer_profile_id))
        corrupt = peers.copy()
        corrupt.loc[0, "peer_profile_id"] = "p2"
        with self.assertRaisesRegex(ValueError, "compatibility"):
            v.validate_peers(frame, corrupt, status)

    def test_hierarchy_abstains_on_unsupported_and_outlying_profiles(self):
        from dataclasses import replace
        from MODELS import vehicle_similarity_v3 as v
        frame = self.profiles()
        fields = ["wheelbase", "height", "horsepower", "fuel_type"]
        config = replace(v.config_for(True), clusters=(2,))
        model = v.fit_candidates(frame, fields, "hierarchical", config, 42)[2]
        query = frame.iloc[[0]].copy()
        query["profile_id"] = "unseen"
        query["wheelbase"] = np.nan
        self.assertEqual(v.assign(model, query).item(), -1)
        query = frame.iloc[[0]].copy()
        query["profile_id"] = "outside"
        model["radius"][:] = 0.
        query["horsepower"] += 1000
        self.assertEqual(v.assign(model, query).item(), -1)

    def test_cached_retrieval_matches_dense_pair_distances(self):
        from MODELS import vehicle_similarity_v3 as v
        frame = self.profiles(12)
        fields = ["wheelbase", "height", "seating", "horsepower", "fuel_type"]
        frame.loc[::3, "seating"] = np.nan
        frame["eligible"] = v.eligible(frame, fields)
        metric = v.fit_metric(frame.iloc[4:], fields)
        peers, _ = v.retrieve(frame, ["p0"], fields, metric, {}, v.Config())
        candidate = frame[frame.family_block.ne(frame.iloc[0].family_block)]
        distance = v.distances(frame.iloc[[0]], candidate, metric).ravel()
        expected = candidate.assign(distance=distance).sort_values(["distance", "profile_id"]).drop_duplicates("family_block").head(5)
        direct = peers[peers.approach.eq("direct_specs")]
        self.assertEqual(direct.peer_profile_id.tolist(), expected.profile_id.tolist())
        np.testing.assert_allclose(direct.distance, expected.distance)

    def test_validation_recomputes_shared_evidence_and_distances(self):
        from MODELS import vehicle_similarity_v3 as v
        frame = self.profiles(6)
        fields = ["wheelbase", "height", "horsepower", "fuel_type"]
        frame["eligible"] = True
        metric = v.fit_metric(frame, fields)
        peers, status = v.retrieve(frame, ["p0"], fields, metric, {}, v.Config())
        v.validate_peers(frame, peers, status, fields, metric)
        corrupt = peers.copy()
        corrupt.loc[0, "shared_size"] = 0
        with self.assertRaisesRegex(ValueError, "Shared evidence metadata"):
            v.validate_peers(frame, corrupt, status, fields, metric)
        corrupt = peers.copy()
        corrupt.loc[0, "size_distance"] = .99
        corrupt.loc[0, "distance"] = (corrupt.loc[0, "size_distance"] + corrupt.loc[0, "powertrain_distance"]) / 2
        corrupt.loc[0, "ranking_distance"] = corrupt.loc[0, "distance"]
        with self.assertRaisesRegex(ValueError, "Saved distance differs"):
            v.validate_peers(frame, corrupt, status, fields, metric)

    def test_fold_pipeline_and_blinded_review_round_trip(self):
        from MODELS import vehicle_similarity_v3 as v
        frame = self.profiles()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            prepared = directory / "prepared"
            prepared.mkdir()
            frame.to_parquet(prepared / "variant_profiles.parquet", index=False)
            frame[["Make", "Model"]].to_parquet(prepared / "identity_names.parquet", index=False)
            records = []
            for row in frame.itertuples():
                for number in range(6):
                    records.append(dict(vin=f"{row.profile_id}-{number}", profile_id=row.profile_id,
                        horsepower=100., EngineHP=100., fuel_type="Gasoline", FuelTypePrimary="Gasoline",
                        FuelTypeSecondary=None, ElectrificationLevel=None, powertrain="unknown"))
            pd.DataFrame(records).to_parquet(prepared / "vin_profile_membership.parquet", index=False)
            research.write_json(prepared / "manifest.json", {"status": "prepared", "schema_version": "2.0.0",
                "source_hashes": {"synthetic": "fixture"}, "artifacts": {p.name: research.file_hash(p) for p in prepared.iterdir()}})
            # This integration check exercises fold preprocessing and review integrity;
            # clustering geometry is checked separately, without expensive refit budgets.
            with patch.object(v, "discover", return_value=({}, [], [], pd.DataFrame())):
                run = v.analyze(prepared, directory / "run", smoke=True)
            for fold in range(5):
                import joblib
                receipt = joblib.load(run / f"fold_{fold}_retrieval.joblib")
                heldout = frame.loc[frame.fold.eq(fold), "family_block"]
                self.assertFalse(set(heldout) & set(receipt["training_families"]))
            review = v.export_review(run, directory / "review")
            browser = (review / "review_cards.html").read_text(encoding="utf-8")
            self.assertNotIn('direct_specs', browser)
            self.assertNotIn('cluster_aware', browser)
            self.assertIn('drivetrain', browser)
            self.assertIn('transmission', browser)
            ratings = pd.read_csv(review / "ratings_template.csv", dtype=str, keep_default_na=False)
            ratings["winner"], ratings["reviewer"] = "tie", "synthetic_test_reviewer"
            for side in ("A", "B"):
                for rank in range(1, 6):
                    ratings[f"{side}{rank}_rating"] = np.where(ratings[f"{side}{rank}_peer_id"].ne(""), "1", "")
            submitted = directory / "submitted.csv"
            ratings.to_csv(submitted, index=False)
            result = v.evaluate_review(review, submitted, directory / "evaluated")
            self.assertEqual(result["selected_approach"], "direct_specs")
            self.assertEqual(result["mean_useful_at_5_change"], 0.)
            self.assertEqual(result["preference_counts"]["tie"], 10)
            partial = ratings.copy()
            partial.loc[0, "A1_rating"] = ""
            partial.loc[1, "winner"] = ""
            partial.to_csv(submitted, index=False)
            with self.assertRaisesRegex(ValueError, "preference and reviewer"):
                v.evaluate_review(review, submitted, directory / "strict-incomplete")
            result = v.evaluate_review(review, submitted, directory / "partial", allow_incomplete=True)
            self.assertEqual(result["status"], "partially_reviewed")
            self.assertEqual(result["fully_graded_pairs"], 9)
            self.assertEqual(result["preference_cards"], 9)
            self.assertEqual(result["complete_cards"], 8)
            self.assertEqual(result["selected_approach"], "direct_specs")
            scored = pd.read_csv(directory / "partial/scored_reviews.csv")
            self.assertTrue(scored.loc[scored.card_id.eq(partial.loc[0, "card_id"]), "useful_at_5"].isna().all())
            self.assertTrue(scored.loc[scored.card_id.eq(partial.loc[1, "card_id"]), "preferred"].isna().all())
            self.assertTrue(scored.useful_at_5.dropna().eq(1).all())
            self.assertTrue(pd.read_csv(directory / "partial/coverage_summary.csv").queries.eq(10).all())
            ratings.loc[0, "A1_peer_id"] = "changed"
            ratings.to_csv(submitted, index=False)
            with self.assertRaisesRegex(ValueError, "identities changed"):
                v.evaluate_review(review, submitted, directory / "corrupt")


class CategoryWorkflowControls(unittest.TestCase):
    @staticmethod
    def snapshot(directory, families=12):
        raw = fixture(families)
        listings, decodes = raw["used_cars"], raw["vpic"]
        listings["height"] = [55. + int(v.split("-")[1]) % 3 * 6 for v in listings.vin]
        for field, value in {"make_name": "Test", "model_name": "Model", "trim_name": "Base", "year": 2015,
            "width": 70., "front_legroom": 40., "back_legroom": 30., "torque": "200 lb-ft", "price": 10000.,
            "mileage": 20000., "listed_date": "2020-01-01"}.items():
            listings[field] = value
        decodes["VIN"] = "WRONG-CAPITALIZED-KEY"
        decodes["BodyClass"] = listings.body_type.to_numpy()
        for field, value in {"FuelTypePrimary": "Gasoline", "FuelTypeSecondary": None,
            "ElectrificationLevel": "Not Electrified", "CurbWeightLB": "3000", "Doors": "4"}.items():
            decodes[field] = value
        from EDA.DATA_CLEANING import EXCLUDED_LISTING_VINS
        excluded = next(iter(EXCLUDED_LISTING_VINS))
        listings = pd.concat([listings, listings.iloc[[0]].assign(vin=excluded)], ignore_index=True)
        decodes = pd.concat([decodes, decodes.iloc[[0]].assign(vin=excluded)], ignore_index=True)
        directory.mkdir()
        listings.to_parquet(directory / "used_cars.parquet", index=False)
        decodes.to_parquet(directory / "NHTSA_vPIC.parquet", index=False)

    @staticmethod
    def outcome(task, cache):
        model = task["model"]
        return {"make": task["make"], "model": model, "status": "source_supported", "attempts": [], "entries": [{
            "id": "test-" + model, "make": task["make"], "model": model, "year_start": 2015, "year_end": 2015,
            "url": "https://example.test/" + model, "excerpt": "Explicit economy sports car description",
            "scope_note": "Exact TEST model and 2015 model year", "all_trims": True, "retrieved_at": "2026-10-25",
            "claims": [{"dimension": "market_positioning", "label": "economy"},
                       {"dimension": "performance_character", "label": "sports_car"}]}]}

    def test_structured_sources_need_identity_and_model_year_scope(self):
        from MODELS import vehicle_similarity_v4 as v
        task = {"make": "TEST", "model": "CAR", "year_start": 2015, "year_end": 2020}
        document = {"url": "https://example.test/car", "retrieved_at": "2026-10-25", "content_sha256": "a" * 64,
            "html": '<h1>TEST CAR</h1><table class="infobox"><tr><th>Class</th><td>Economy car</td></tr><tr><th>Model years</th><td>2014–2018</td></tr></table>'}
        entries, status = v.parse_document(document, task)
        self.assertEqual(status, "source_supported")
        self.assertEqual((entries[0]["year_start"], entries[0]["year_end"]), (2015, 2018))
        self.assertEqual(entries[0]["claims"], [{"dimension": "market_positioning", "label": "economy"}])
        self.assertEqual(v.parse_document({**document, "html": document["html"].replace("Model years", "Production")}, task)[1], "ambiguous_scope")
        self.assertEqual(v.parse_document({**document, "html": document["html"].replace("TEST CAR", "TEST OTHER")}, task)[1], "unmatched_identity")
        self.assertEqual(v.parse_document({**document, "html": document["html"].replace("Economy car", "Economy car (Sport trim)")}, task)[1], "ambiguous_scope")

    def test_search_discovers_exact_titles_without_assigning_from_snippets(self):
        from MODELS import vehicle_similarity_v4 as v
        task = {"make": "TEST", "model": "CAR", "year_start": 2015, "year_end": 2020, "seed_sources": []}
        def document(url, cache):
            if "api.php" in url:
                return {"html": json.dumps({"query": {"search": [{"title": "Test Other", "snippet": "luxury supercar"}, {"title": "Test Car"}]}})}
            return {"url": url, "retrieved_at": "2026-10-25", "content_sha256": "a" * 64,
                    "html": '<h1>Test Car</h1><table class="infobox"><tr><th>Class</th><td>Economy car</td></tr><tr><th>Model years</th><td>2015–2020</td></tr></table>' if url.endswith("Test_Car") else '<h1>Unknown</h1>'}
        with patch.object(v, "fetch_document", side_effect=document):
            result = v.acquire_task(task, Path("unused"))
        self.assertEqual(result["status"], "source_supported")
        self.assertEqual(result["entries"][0]["claims"], [{"dimension": "market_positioning", "label": "economy"}])

    def test_incomplete_identities_remain_audit_only(self):
        from MODELS import vehicle_similarity_v4 as v
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.snapshot(root / "data", 2)
            decoded = pd.read_parquet(root / "data/NHTSA_vPIC.parquet")
            decoded["Trim"] = None
            decoded.to_parquet(root / "data/NHTSA_vPIC.parquet", index=False)
            v.prepare(root / "data", root / "prepared")
            self.assertFalse(pd.read_parquet(root / "prepared/variant_profiles.parquet").identity_complete.any())
            self.assertEqual(len(pd.read_parquet(root / "prepared/vin_profile_membership.parquet")), 24)

    def test_whole_catalog_resume_skips_committed_families(self):
        from MODELS import vehicle_similarity_v4 as v
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.snapshot(root / "data", 3)
            v.prepare(root / "data", root / "prepared")
            calls = []
            def acquire(task, cache):
                calls.append(task["model"])
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return self.outcome(task, cache)
            with patch.object(v, "acquire_task", side_effect=acquire):
                with self.assertRaises(KeyboardInterrupt):
                    v.collect_sources(root / "prepared", root / "sources", batch_size=1, workers=1)
            with patch.object(v, "acquire_task", side_effect=self.outcome) as resumed:
                v.collect_sources(root / "prepared", root / "sources", batch_size=1, workers=1, resume=True)
                self.assertEqual(resumed.call_count, 2)
            result = research.read_json(root / "sources/manifest.json")
            self.assertEqual(result["outcomes"], 3)
            with self.assertRaisesRegex(ValueError, "immutable"):
                v.collect_sources(root / "prepared", root / "sources", batch_size=1, workers=1, resume=True)

    def test_checkpoint_reuse_checks_contract_and_preserves_names(self):
        from MODELS import vehicle_similarity_v4 as v
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "view"
            def discover(data, fields, output, config):
                output.mkdir(parents=True)
                return {}, [{"status": "insufficient_support"}], [], pd.DataFrame()
            with patch.object(v.v3, "discover", side_effect=discover) as fit:
                for repeat in range(2):
                    v.discover_checkpoint(pd.DataFrame(), [], destination, v.v3.Config(), {"fold": 0})
                self.assertEqual(fit.call_count, 1)
                with self.assertRaisesRegex(ValueError, "contract changed"):
                    v.discover_checkpoint(pd.DataFrame(), [], destination, v.v3.Config(), {"fold": 1})
            self.assertIn("Luxury", v.view_title("market_positioning__luxury__passenger_car"))
            parent = pd.DataFrame({"height": [50., 100., 50., 100.], "seating": [2., 8., 2., 8.]})
            self.assertEqual(v.subgroup_description(parent.iloc[[1, 3]], parent, ["height", "seating"]), "taller height, more seats")

    def test_synthetic_bundle_validates_and_activation_preserves_prices(self):
        from MODELS import vehicle_similarity_v4 as v
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.snapshot(root / "data")
            v.prepare(root / "data", root / "prepared")
            membership = pd.read_parquet(root / "prepared/vin_profile_membership.parquet")
            self.assertEqual(len(membership), 144)
            self.assertTrue(membership.identity_complete.all())
            def outcome(task, cache):
                result = self.outcome(task, cache)
                if task["model"] == "MODEL0":
                    result["entries"][0]["claims"].append({"dimension": "market_positioning", "label": "luxury"})
                return result
            with patch.object(v, "acquire_task", side_effect=outcome):
                v.collect_sources(root / "prepared", root / "sources")
            v.build_categories(root / "prepared", root / "sources", root / "categories")
            labels = pd.read_parquet(root / "categories/profile_labels.parquet")
            self.assertEqual(labels[labels.label.eq("economy")].profile_id.nunique(), 22)
            coverage = pd.read_parquet(root / "categories/category_coverage.parquet")
            self.assertEqual(coverage.reason.eq("conflicting_evidence").sum(), 2)
            self.assertTrue(labels[labels.dimension.isin(v.SEMANTIC)].status.eq("source_supported_provisional").all())
            # Twelve families cannot satisfy the 20-family minimum: no statistical fit runs.
            original_discover = v.v3.discover
            completed_views = []
            def interrupt(data, fields, destination, config):
                if completed_views:
                    raise KeyboardInterrupt()
                result = original_discover(data, fields, destination, config)
                completed_views.append(destination)
                return result
            with patch.object(v.v3, "discover", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    v.analyze(root / "prepared", root / "categories", root / "run")
            with patch.object(v.v3, "fit_candidates", side_effect=AssertionError("Sparse categories must not fit")):
                v.analyze(root / "prepared", root / "categories", root / "run", resume=True)
                v.validate(root / "run", root / "validation")
            from car_app.recommendation import load_similarity
            from car_app.artifacts import stat_key
            catalog = load_similarity(str(root / "run"), stat_key(root / "run/manifest.json"), str(root / "data"))
            self.assertIn("economy", catalog.category_choices()["market_positioning"])
            self.assertEqual(len(catalog.category_details), 24 * len(v.DIMENSIONS))
            config = root / "registry.json"
            original = {"similarity_run": "old", "active_price": {"run": "do-not-change"}, "other": 7}
            research.write_json(config, original)
            backup = v.activate(root / "run", root / "validation", config)
            self.assertEqual(research.read_json(config)["active_price"], original["active_price"])
            self.assertEqual(research.read_json(config)["other"], 7)
            self.assertEqual(research.read_json(backup / "previous_similarity.json")["similarity_run"], "old")
            atomic = v.atomic_json
            def concurrent(path, value):
                atomic(path, value)
                if Path(path).name == "previous_similarity.json":
                    research.write_json(config, {**original, "active_price": {"run": "concurrent-price-update"}})
            with patch.object(v, "atomic_json", side_effect=concurrent):
                with self.assertRaisesRegex(ValueError, "concurrently"):
                    v.activate(root / "run", root / "validation", config)
            self.assertEqual(research.read_json(config)["active_price"]["run"], "concurrent-price-update")
            receipt = research.read_json(root / "run/manifest.json")
            receipt["reduced_budget"] = True
            research.write_json(root / "run/manifest.json", receipt)
            with self.assertRaisesRegex(ValueError, "full-budget"):
                v.activate(root / "run", root / "validation", config)

    def test_category_peer_comparisons_keep_direct_order_and_unknowns(self):
        from MODELS import vehicle_similarity_v4 as v
        frame = SimilarityV3Controls.profiles(8)
        for dimension in v.DIMENSIONS:
            frame[dimension + "_labels"] = ""
        frame.loc[0, "market_positioning_labels"] = "luxury"
        frame.loc[4, "market_positioning_labels"] = "luxury"
        frame["eligible"] = True
        fields = ["wheelbase", "height", "horsepower", "fuel_type"]
        metric = v3_metric = v.v3.fit_metric(frame.iloc[2:], fields)
        base, _ = v.v3.retrieve(frame, ["p0"], fields, metric, {}, v.v3.Config())
        empty = pd.DataFrame(columns=["view", "profile_id", "cluster"])
        peers, statuses = v.v3.retrieve(frame, ["p0"], fields, v3_metric, {}, v.v3.Config(), v.ranking_policy(frame, empty, v.v3.Config()), v.APPROACHES)
        self.assertEqual(base[base.approach.eq("direct_specs")].peer_profile_id.tolist(), peers[peers.approach.eq("direct_specs")].peer_profile_id.tolist())
        self.assertEqual(peers[peers.approach.eq("category_specs")].iloc[0].peer_profile_id, "p4")
        self.assertEqual(len(statuses), 3)

    def test_category_ranking_empty_candidates_have_numeric_scores(self):
        from MODELS import vehicle_similarity_v4 as v
        frame = SimilarityV3Controls.profiles(1)
        for dimension in v.DIMENSIONS:
            frame[dimension + "_labels"] = pd.Series(["", ""], dtype="string")
        frame.loc[0, "market_positioning_labels"] = "luxury"
        frame["eligible"] = True
        fields = ["wheelbase", "height", "horsepower", "fuel_type"]
        metric = v.v3.fit_metric(frame, fields)
        memberships = pd.DataFrame(columns=["view", "profile_id", "cluster"])
        policy = v.ranking_policy(frame, memberships, v.v3.Config())
        for score, penalty in policy("p0", [], np.zeros(0)).values():
            self.assertEqual(score.dtype, np.dtype("float64"))
            self.assertEqual(score.shape, (0,))
            self.assertEqual(penalty.dtype, np.dtype("float64"))
        peers, statuses = v.v3.retrieve(frame, ["p0"], fields, metric, {}, v.v3.Config(), policy, v.APPROACHES)
        self.assertTrue(peers.empty)
        self.assertEqual(len(statuses), 3)
        self.assertTrue(statuses.slate_size.eq(0).all())

    def test_cluster_validation_accepts_integer_width_but_rejects_changed_memberships(self):
        from MODELS import vehicle_similarity_v4 as v
        expected = pd.Series([0, 1, -1], index=["a", "b", "c"], dtype="int32")
        saved = expected.astype("int64")
        v.validate_cluster_memberships(expected, saved)
        for changed in [saved.replace({1: 0}), saved.rename(index={"a": "wrong"}), saved.iloc[:-1],
                        saved.astype(float), pd.concat([saved, saved.iloc[[0]]])]:
            with self.assertRaisesRegex(ValueError, "memberships differ"):
                v.validate_cluster_memberships(expected, changed)

    def test_validation_only_repair_checks_full_analysis_provenance(self):
        from copy import deepcopy
        from MODELS import vehicle_similarity_v4 as v
        recorded = deepcopy(v.execution())
        self.assertIsNone(v.validation_execution_compatibility(recorded))
        recorded["code_hashes"]["vehicle_similarity_v4.py"] = v.VALIDATION_REPAIR_BASE
        compatible = v.validation_execution_compatibility(recorded)
        self.assertEqual(compatible["analysis_code_sha256"], v.VALIDATION_REPAIR_BASE)
        for part, key in [("code_hashes", "vehicle_similarity_v3.py"), ("versions", "numpy"),
                          ("taxonomy_hashes", "evidence.v2.json")]:
            changed = deepcopy(recorded)
            changed[part][key] = "changed"
            with self.assertRaisesRegex(ValueError, "dependencies changed"):
                v.validation_execution_compatibility(changed)
        with patch.object(v, "analysis_output_fingerprint", return_value="changed-analysis"):
            with self.assertRaisesRegex(ValueError, "dependencies changed"):
                v.validation_execution_compatibility(recorded)

    def test_ranking_repair_preserves_original_checkpoint_contract(self):
        from copy import deepcopy
        from MODELS import vehicle_similarity_v4 as v
        current = {"stage": "analyze", "prepared": "same-preparation", "configuration": {"smoke": True}, **v.execution()}
        saved = deepcopy(current)
        saved["code_hashes"]["vehicle_similarity_v4.py"] = v.RANKING_REPAIR_BASE
        self.assertTrue(v.compatible_ranking_repair(saved, current))
        changed = deepcopy(current)
        changed["configuration"]["smoke"] = False
        self.assertFalse(v.compatible_ranking_repair(saved, changed))
        changed = deepcopy(current)
        changed["code_hashes"]["vehicle_similarity_v3.py"] = "changed-fit-code"
        self.assertFalse(v.compatible_ranking_repair(saved, changed))
        with patch.object(v, "analysis_fit_fingerprint", return_value="changed-v4-fit"):
            self.assertFalse(v.compatible_ranking_repair(saved, current))
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "smoke"
            v.stage(target, saved)
            original = (target / "stage_state.json").read_bytes()
            research.finish_stage(target, "blocked", schema_version=v.VERSION)
            resumed = deepcopy(current)
            v.stage(target, resumed, resume=True)
            self.assertEqual(resumed, saved)
            self.assertEqual((target / "stage_state.json").read_bytes(), original)
            self.assertTrue((target / "resume_compatibility.json").is_file())


class RepairV5Controls(unittest.TestCase):
    def test_displacement_fills_decoder_evidence_and_preserves_conflicts(self):
        from MODELS.vehicle_profiles_v5 import reconcile_displacement
        frame=pl.DataFrame({"engine_displacement":[None,2000.,2000.,0.,None,1000.],
            "DisplacementCC":[1998.,1998.,3500.,None,1500.,1050.],
            "powertrain":["combustion_unverified"]*4+["battery_electric","combustion_unverified"]})
        result=reconcile_displacement(frame)
        self.assertEqual(result["engine_displacement"].to_list(),[1998.,2000.,None,None,None,1000.])
        self.assertEqual(result["displacement_source"].to_list(),["vpic","compatible_observed","conflicting","unknown","not_applicable","compatible_observed"])
        self.assertEqual(result["listing_engine_displacement"].to_list(),frame["engine_displacement"].to_list())

    def test_new_displacement_coordinate_uses_only_development_evidence(self):
        from MODELS import vehicle_similarity_v5 as v
        data=pd.DataFrame({"profile_id":[f"p{i}" for i in range(60)],"development":[True]*40+[False]*20,
            "identity_complete":True,"vin_count":10,"powertrain":"combustion_unverified","body_style":"sedan"})
        for field in [*v.v2.NUMERIC,*v.v2.CATEGORICAL,"width","engine_displacement"]:
            data[field]="A" if field in v.v2.CATEGORICAL else 100.
            data[field+"_n"]=10
        data["wheelbase"]=np.arange(60)+90.
        data["height"]=np.tile([50.,60.,70.],20)
        data["transmission"]=np.tile(["A","M"],30)
        data["engine_displacement"]=2000.
        data.loc[~data.development,"engine_displacement"]=np.arange(20)*100.+1000
        pairs=pd.DataFrame(columns=["profile_id","ratio"])
        _,fields,_=v.select_fields(data,pairs)
        self.assertNotIn("engine_displacement",fields)
        data.loc[data.development,"engine_displacement"]=np.arange(40)*50.+1000
        _,fields,_=v.select_fields(data,pairs)
        self.assertIn("engine_displacement",fields)

    def test_drivetrain_specificity_and_propulsion_unknowns(self):
        from MODELS import vehicle_profiles_v5 as p
        self.assertEqual(p.reconcile_drive("FWD", "4x2"), ("FWD", "TWO_WHEEL", "compatible_broad"))
        self.assertEqual(p.reconcile_drive("AWD", "4WD/4-Wheel Drive/4x4"), ("ALL_WHEEL", "ALL_WHEEL", "detail_unresolved"))
        self.assertEqual(p.reconcile_drive("FWD", "RWD/Rear-Wheel Drive")[2], "detail_conflicting")
        self.assertEqual(p.propulsion("Gasoline", "Gasoline", None, None, 2000), ("combustion_unverified", "rule_derived", True))
        self.assertEqual(p.propulsion("Gasoline", "Electric", "Gasoline", "PHEV (Plug-in Hybrid Electric Vehicle)", 1500)[0], "plug_in_hybrid")
        self.assertEqual(p.propulsion("Gasoline", "Gasoline", None, "BEV (Battery Electric Vehicle)", 2000)[0], "conflicting")
        self.assertEqual(p.propulsion(None, "Fuel Cell", None, "FCEV", None)[0], "fuel_cell")
        self.assertEqual(p.propulsion("Hybrid","Electric",None,"BEV (Battery Electric Vehicle)",None)[0],"conflicting")
        self.assertEqual(p.propulsion(None,"Electric",None,"Strong HEV (Hybrid Electric Vehicle)",1500)[0],"hybrid")
        self.assertEqual(p.propulsion("Hybrid","Gasoline",None,None,1500)[0],"hybrid_unspecified")

    def test_reference_retry_after_and_success_cache(self):
        import requests
        from MODELS import vehicle_sources_v5 as s
        with tempfile.TemporaryDirectory() as temp:
            client = s.ReferenceClient(temp, interval=0)
            blocked = requests.Response()
            blocked.status_code = 429
            blocked.headers["Retry-After"] = "120"
            blocked._content = b"rate limited"
            with patch.object(client.session, "get", return_value=blocked):
                with self.assertRaises(s.DeferredReference):
                    client.get("https://example.test/deferred")
            state = research.read_json(Path(temp) / (research.digest("https://example.test/deferred") + ".json"))
            self.assertEqual(state["status"], "deferred")
            good = requests.Response()
            good.status_code, good._content, good.url = 200, b"ok", "https://example.test/canonical"
            with patch.object(client.session, "get", return_value=good) as call:
                client.get("https://example.test/good")
                client.get("https://example.test/good")
                self.assertEqual(call.call_count, 1)

    def test_epa_ambiguity_is_not_a_classification(self):
        from MODELS.vehicle_sources_v5 import epa_claims
        data = pd.DataFrame({"id": [1,2], "year": [2020,2020], "make": ["Test","Test"], "model": ["Car","Car"], "VClass": ["Compact Cars", "Mid-Size Cars"]})
        profile = {"ModelYear": 2020, "Make": "TEST", "Model": "CAR"}
        self.assertNotIn("epa_class", epa_claims(profile, data))
        data["VClass"] = "Compact Cars"
        self.assertEqual(epa_claims(profile, data)["epa_class"], "Compact Cars")

    def test_nhtsa_weight_boundaries_and_epa_base_model(self):
        from MODELS.vehicle_sources_v5 import nhtsa_class,epa_claims
        profile={"body_family":"passenger_car","vin_count":10,"curb_weight_lb_n":10,"curb_weight_lb_min":2500.,"curb_weight_lb_max":2999.}
        self.assertEqual(nhtsa_class(profile),"PC/C")
        profile["curb_weight_lb_max"]=3000.
        self.assertIsNone(nhtsa_class(profile))
        profile["curb_weight_lb_n"]=1
        self.assertIsNone(nhtsa_class(profile))
        data=pd.DataFrame({"id":[1],"year":[2020],"make":["Ford"],"model":["F150 Pickup 2WD"],"baseModel":["F150"],"VClass":["Standard Pickup Trucks 2WD"],"drive":["Rear-Wheel Drive"]})
        self.assertEqual(epa_claims({"ModelYear":2020,"Make":"FORD","Model":"F-150","body_family":"pickup","wheel_system":"RWD"},data)["epa_class"],"Standard Pickup Trucks 2WD")

    def test_category_bonus_cannot_dominate_specification_distance(self):
        from MODELS import vehicle_similarity_v5 as v
        data=pd.DataFrame({"profile_id":["q","near","far"],"family_block":["Q","N","F"],"ModelYear":[2020]*3,
            "body_family":["passenger_car"]*3,"powertrain":["combustion_unverified"]*3,"body_style":["sedan"]*3,
            "identity_complete":[True]*3,"vin_count":[10]*3,"eligible":[True]*3,"development":[False,True,True],"fold":[0,1,2],
            "wheelbase":[100.,100.,150.],"height":[50.,50.,80.],"horsepower":[100.,100.,300.],"fuel_type":["GASOLINE"]*3})
        fields=["wheelbase","height","horsepower","fuel_type"]
        metric=v.v3.fit_metric(data[data.development],fields)
        labels=pd.DataFrame({"profile_id":["q","far"],"dimension":["market_positioning"]*2,"label":["luxury"]*2})
        empty=pd.DataFrame(columns=["profile_id","policy","dimension","label","probability"])
        peers,status=v.ranked_peers(data,fields,metric,pd.DataFrame(columns=["view","profile_id","cluster"]),labels,empty)
        self.assertEqual(peers.loc[peers.approach.eq("category_specs")].iloc[0].peer_profile_id,"near")
        self.assertTrue(peers.category_bonus.between(0,.1).all())
        self.assertEqual(len(status),4)

    def test_review_gates_and_family_partitions(self):
        from MODELS import vehicle_classifiers_v5 as c
        self.assertFalse(c.gate(29, 29, "conservative"))
        self.assertTrue(c.gate(30, 30, "conservative"))
        self.assertFalse(c.gate(27, 30, "conservative"))
        self.assertEqual(c.partition("TEST|CAR"), c.partition("TEST|CAR"))

    def test_validation_compatibility_rejects_fitting_changes(self):
        from copy import deepcopy
        from MODELS import vehicle_similarity_v5 as v
        source=Path(v.__file__).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"execution").mkdir()
            archived=root/"execution"/Path(v.__file__).name
            archived.write_text(source+"\n# Prior validator revision\n",encoding="utf-8")
            receipt={"execution":deepcopy(v.execution())}
            receipt["execution"]["code_hashes"][archived.name]=research.file_hash(archived)
            self.assertIsNotNone(v.validation_compatibility(root,receipt))
            changed=source.replace("ranking = distance-bonus+applied","ranking = distance+bonus+applied")
            self.assertNotEqual(changed,source)
            archived.write_text(changed,encoding="utf-8")
            receipt["execution"]["code_hashes"][archived.name]=research.file_hash(archived)
            with self.assertRaisesRegex(ValueError,"Fitting or ranking"):
                v.validation_compatibility(root,receipt)

    def test_classifier_never_overwrites_reviewed_performance_negatives(self):
        from types import SimpleNamespace
        from MODELS import vehicle_classifiers_v5 as c
        from MODELS.vehicle_similarity_v3 import family_fold
        rows=[{"profile_id":f"p{i}","grain":"cohort","family_block":f"TEST|F{i}","identity_complete":True,"vin_count":10,"ModelYear":2020,"Make":"TEST"} for i in range(1000)]
        data=pd.DataFrame(rows)
        negatives=set()
        for fold in range(5):
            negatives.add(next(row["profile_id"] for row in rows[:900] if c.partition(row["family_block"])=="calibration" and family_fold(row["family_block"])==fold))
        negatives.add(next(row["profile_id"] for row in rows[:900] if c.partition(row["family_block"])=="train"))
        source=pd.DataFrame([{"profile_id":r["profile_id"],"dimension":"performance_character","label":"sports_car","status":"source_supported"} for r in rows[:900] if r["profile_id"] not in negatives])
        reviewed=pd.DataFrame([{"profile_id":r["profile_id"],"grain":"cohort","dimension":"performance_character","labels":"" if r["profile_id"] in negatives else "sports_car","reviewed_all":True} for r in rows[:900]])
        fake=SimpleNamespace(classes_=np.asarray(["no","yes"]),predict_proba=lambda x:np.tile([.01,.99],(len(x),1)))
        with tempfile.TemporaryDirectory() as temp, patch.object(c,"choose_model",return_value=("logistic",False)),patch.object(c,"fit_model",return_value=fake),patch.object(c.joblib,"dump"):
            predictions,_=c.train(data,source,reviewed,{"dimensions":{"size_segment":{},"market_positioning":{},"performance_character":{"sports_car":"Sports car"}}},Path(temp))
        self.assertFalse(predictions.empty)
        self.assertFalse(set(predictions.profile_id)&negatives)

    def test_missing_trim_has_independent_model_membership(self):
        from MODELS import vehicle_similarity_v5 as v
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            CategoryWorkflowControls.snapshot(root / "data", 3)
            path = root / "data/NHTSA_vPIC.parquet"
            decodes = pd.read_parquet(path)
            for field, value in {"DriveType": "4x2", "Seats": 5, "EngineCylinders": 4, "EngineHP_to": None, "WheelBaseShort": 100., "WheelBaseLong": 110., "VehicleType": "PASSENGER CAR"}.items():
                decodes[field] = value
            decodes["Trim"] = None
            decodes.loc[0,"ModelYear"] = None
            decodes.to_parquet(path, index=False)
            v.prepare(root / "data", root / "prepared")
            cohort = pd.read_parquet(root / "prepared/cohort/variant_profiles.parquet")
            model = pd.read_parquet(root / "prepared/model/variant_profiles.parquet")
            self.assertFalse(cohort.identity_complete.any())
            self.assertTrue(model.identity_complete.any())
            self.assertFalse(model.identity_complete.all())
            self.assertFalse(set(cohort.profile_id) & set(model.profile_id))
            for grain in ("cohort", "model"):
                membership = pd.read_parquet(root / f"prepared/{grain}/vin_profile_membership.parquet")
                self.assertEqual(membership.vin.nunique(), len(membership))
            v.collect_sources(root / "prepared", root / "sources", offline=True)
            v.build_categories(root / "prepared", root / "sources", root / "categories")
            data = pd.read_parquet(root / "categories/variant_profiles.parquet")
            labels = pd.read_parquet(root / "categories/profile_labels.parquet")
            review = root / "review"
            review.mkdir()
            from MODELS.vehicle_classifiers_v5 import export_review, train
            inventory = export_review(data, labels, review)
            self.assertTrue(inventory.grain.eq("model").all())
            definitions = research.read_json(root / "categories/definitions.json")
            predictions, _ = train(data, labels, pd.DataFrame(), definitions, review)
            self.assertTrue(predictions.empty)

    def test_review_rejects_unadjudicated_conflicts_and_unknown_negatives(self):
        from MODELS.vehicle_classifiers_v5 import import_review
        base={"review_id":"r1","profile_id":"p1","grain":"cohort","family_block":"A|B","dimension":"performance_character","proposed_labels":"sports_car"}
        inventory=pd.DataFrame([base])
        ratings=pd.DataFrame([{**base,"reviewer":"one","verdict":"accept","labels":"","reviewed_all":"true"},
                              {**base,"reviewer":"two","verdict":"unknown","labels":"","reviewed_all":"true"}])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"ratings.csv"
            ratings.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"adjudicat"):
                import_review(inventory,path,{"dimensions":{"performance_character":{"sports_car":"sports"}}})
            ratings.loc[1,"adjudicator"]="review lead"
            ratings.to_csv(path,index=False)
            result=import_review(inventory,path,{"dimensions":{"performance_character":{"sports_car":"sports"}}})
            self.assertEqual(result.iloc[0].labels,"")
            self.assertFalse(result.iloc[0].reviewed_all)

    def test_common_evidence_and_all_method_outlier_rejection(self):
        from dataclasses import replace
        from MODELS import vehicle_similarity_v5 as v
        data=pd.DataFrame({"profile_id":[f"p{i}" for i in range(60)],"family_block":[f"f{i//2}" for i in range(60)],
            "identity_complete":True,"vin_count":10,"body_style":"sedan","powertrain":"combustion_unverified",
            "wheelbase":[100.+(i%2)*20 for i in range(60)],"height":[50.+(i%2)*20 for i in range(60)],
            "horsepower":[100.+(i%2)*100 for i in range(60)],"fuel_type":"GASOLINE"})
        fields=["wheelbase","height","horsepower","fuel_type"]
        config=replace(v.v3.config_for(True),minimum_profiles=20,minimum_families=5)
        model=v.fit_candidates(data,fields,"kmeans",config,42,[2])[2]
        outlier=data.iloc[[0]].copy()
        outlier["wheelbase"],outlier["height"],outlier["horsepower"]=120.,50.,200.
        self.assertEqual(v.assign(model,outlier).tolist(),[-1])
        data.loc[:49,"horsepower"]=np.nan
        self.assertEqual(v.fit_candidates(data,fields,"gower",config,42,[2]),{})


class CategoryHierarchyTests(unittest.TestCase):
    def test_epa_parent_collapses_drive_only_and_keeps_market_terms_separate(self):
        from MODELS.vehicle_category_hierarchy import epa_group, common_segment
        self.assertEqual(epa_group("Small Sport Utility Vehicle 4WD"), "Small Sport Utility Vehicle")
        self.assertEqual(epa_group("Small Sport Utility Vehicle 2WD"), "Small Sport Utility Vehicle")
        self.assertNotEqual(epa_group("Small Station Wagons"), epa_group("Compact Cars"))
        self.assertEqual(common_segment("midsize", "luxury"), "Compact executive / entry-level luxury")
        self.assertEqual(common_segment("", "economy"), "")

    def test_hierarchy_keeps_recorded_epa_class_and_marks_inferred_subsegments(self):
        from MODELS.vehicle_category_hierarchy import hierarchy_rows
        data = pd.DataFrame({"profile_id": ["a", "b"], "Make": ["Synthetic"]*2, "market_family": ["A", "B"]})
        labels = pd.DataFrame({"profile_id": ["a", "a"], "dimension": ["epa_group", "size_segment"],
            "label": ["Compact Cars", "compact"]})
        predictions = pd.DataFrame({"profile_id": ["a", "b", "b"], "dimension": ["epa_group", "epa_group", "size_segment"],
            "label": ["Large Cars", "Midsize Cars", "midsize"]})
        result = hierarchy_rows(data, labels, predictions)
        self.assertEqual(result.epa_group.tolist(), ["Compact Cars", "Midsize Cars"])
        self.assertEqual(result.epa_group_state.tolist(), ["recorded", "predicted"])
        self.assertEqual(result.common_segment_state.tolist(), ["rule_derived", "predicted"])
        recorded = hierarchy_rows(data, labels, predictions, include_predictions=False)
        self.assertEqual(recorded.iloc[1].epa_group, "")
        self.assertEqual(len(result), len(data))

    def test_source_training_excludes_all_query_families_and_unknown_labels(self):
        from MODELS import vehicle_category_hierarchy as h
        data = pd.DataFrame({"profile_id": [f"p{i}" for i in range(36)], "family_block": [f"f{i//2}" for i in range(36)],
            "fold": [(i//2)%3 for i in range(36)], "identity_complete": True, "grain": "model", "vin_count": 10,
            "ModelYear": 2020, "body_family": "passenger_car", "body_style": "sedan",
            "wheelbase": [90.+(i//6)%2*30 for i in range(36)]})
        labels = pd.DataFrame({"profile_id": data.profile_id[:-2], "dimension": "size_segment", "status": "source_supported",
            "label": ["compact" if (i//6)%2 == 0 else "large" for i in range(34)]})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            predictions, metrics = h.source_predictions(data, labels, target)
            self.assertFalse(metrics.empty)
            import joblib
            for file in target.glob("*.joblib"):
                model = joblib.load(file)
                excluded = set(data.loc[data.fold.eq(model["excluded_fold"]), "family_block"])
                self.assertFalse(set(model["training_families"]) & excluded)
                for label in model["active_labels"]:
                    support = model["development_label_support"][label]
                    self.assertGreaterEqual(support["families"], 3)
                    self.assertGreaterEqual(support["agreement"], .8)
            self.assertTrue(set(predictions.profile_id).issubset(set(data.profile_id[-2:])))


if __name__ == "__main__":
    unittest.main()
