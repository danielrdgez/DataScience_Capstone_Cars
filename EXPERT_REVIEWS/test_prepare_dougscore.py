"""Integrity and linkage boundaries for the expert-opinion import."""
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from EXPERT_REVIEWS.prepare_dougscore import import_snapshot, parse_snapshot, coverage
from EXPERT_REVIEWS import price_features as features


class DougScoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv"
        header = [""] * 20
        header[3:8] = ["Styling", "Acceleration", "Handling", "Fun Factor", "Cool Factor"]
        header[9:14] = ["Features", "Comfort", "Quality", "Practicality", "Value"]
        self.rows = [["Welcome to the DougScore!"] + [""] * 19, header, [""] * 20,
                     ["2020", " McLaren ", "Speedtail", *(["6"] * 5), "30", *(["8"] * 5),
                      "40", "70", "12:30", "City", "Region", "Country"]]
        self.save()

    def save(self):
        with self.source.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(self.rows)
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_import_preserves_identity_scores_and_unknowns(self):
        output = self.root / "import"
        result = import_snapshot(self.source, output, self.sha)
        r = pq.read_table(output / "expert_review_scores.parquet").to_pylist()[0]
        self.assertEqual((r["ModelYear"], r["make_raw"], r["model_raw"]), (2020, " McLaren ", "Speedtail"))
        self.assertEqual(r["source_row_number"], 4)
        self.assertEqual(r["daily_total"], 40)
        self.assertEqual(r["daily_total_raw"], "40")
        self.assertIsNone(r["video_id"])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.sha)
        aspects = pq.read_table(output / "expert_aspect_scores.parquet").to_pylist()
        self.assertEqual({r["aspect"] for r in aspects}, {"design", "performance", "comfort", "technology", "price_value"})
        self.assertEqual(next(r for r in aspects if r["aspect"] == "performance")["expert_score"], 6)
        self.assertEqual(result["approved_vehicle_matches"], 0)
        with self.assertRaises(FileExistsError):
            import_snapshot(self.source, output, self.sha)

    def test_hash_and_score_failures(self):
        with self.assertRaises(ValueError):
            parse_snapshot(self.source, "0" * 64)
        self.rows[3][8] = "29"
        self.save()
        with self.assertRaises(ValueError):
            parse_snapshot(self.source, self.sha)
        self.rows[3][8] = "30"
        self.rows[3][3] = "11"
        self.save()
        with self.assertRaises(ValueError):
            parse_snapshot(self.source, self.sha)

    def test_duplicate_identity_rejected(self):
        self.rows.append(self.rows[3].copy())
        self.rows[4][1] = "mclaren"
        self.save()
        with self.assertRaises(ValueError):
            parse_snapshot(self.source, self.sha)

    def test_candidates_are_distinct_cohorts_and_unapproved(self):
        snapshot = self.root / "import"
        import_snapshot(self.source, snapshot, self.sha)
        vpic = self.root / "vpic.parquet"
        rows = [{"ModelYear": "2020", "Make": "MCLAREN", "Model": "Speedtail", "Trim": None}] * 3
        rows += [{"ModelYear": "2021", "Make": "MCLAREN", "Model": "Speedtail", "Trim": "Other"},
                 {"ModelYear": "2020", "Make": "MCLAREN", "Model": "Speed", "Trim": None}]
        pq.write_table(pa.Table.from_pylist(rows), vpic)
        report = coverage(snapshot, vpic, self.root / "coverage")
        self.assertEqual(report["candidate_rows"], 1)
        self.assertEqual(report["reviews_with_text_candidates"], 1)
        self.assertEqual(report["approved_vehicle_matches"], 0)
        pq.write_table(pa.Table.from_pylist([{"changed": 1}]), snapshot / "expert_review_scores.parquet")
        with self.assertRaises(ValueError):
            coverage(snapshot, vpic, self.root / "changed")


class DougPriceFeatureTests(unittest.TestCase):
    @staticmethod
    def review(model, score=6, identifier=None, year=2020, make="Toyota"):
        return {"review_record_id": identifier or model, "ModelYear": year, "make_raw": make,
                "model_raw": model, **{c: score for c in (*features.WEEKEND, *features.DAILY)},
                "weekend_total": score * 5, "daily_total": score * 5, "dougscore": score * 10}

    def test_year_make_model_matching_ignores_trim(self):
        cohorts = [{"ModelYear": 2020, "Make": "TOYOTA", "Model": "Alpha", "Trim": "Sport"},
                   {"ModelYear": 2020, "Make": "TOYOTA", "Model": "Beta", "Trim": None},
                   {"ModelYear": 2020, "Make": "TOYOTA", "Model": "A", "Trim": "B Sport"},
                   {"ModelYear": 2020, "Make": "TOYOTA", "Model": "A B", "Trim": "Sport"}]
        reviews = [self.review(m) for m in (" Alpha   SPORT ", "Alpha", "Beta", "Unknown", "A B Sport")]
        result = features.resolve_reviews(reviews, cohorts)
        self.assertEqual([r['match_status'] for r in result],
                         ['automatic_model_prefix', 'automatic_exact', 'automatic_exact', 'unmatched', 'automatic_model_prefix'])
        exact, make = features.feature_tables(result)
        self.assertEqual(exact.height, 3)
        self.assertEqual(make.height, 0)
        self.assertNotIn('Trim', exact.columns)
        self.assertEqual(set(exact['Model']), {'alpha', 'beta', 'a b'})

    def test_five_model_threshold_equal_weighting_and_fallback(self):
        cohorts = [{"ModelYear": 2020, "Make": "Toyota", "Model": m, "Trim": "Sport"}
                   for m in ('A', 'B', 'C', 'D', 'E')]
        reviews = [self.review(m + ' Sport', 2 if m == 'A' else 8) for m in ('A', 'B', 'C', 'D', 'E')]
        four = features.resolve_reviews(reviews[:4], cohorts)
        self.assertEqual(features.feature_tables(four)[1].height, 0)
        # A second variant increases review support, but not A's model weight.
        reviews.append(self.review('A Other', 4))
        exact, make = features.feature_tables(features.resolve_reviews(reviews, cohorts))
        self.assertEqual(make['doug_model_count'].to_list(), [5])
        self.assertEqual(make['doug_review_count'].to_list(), [6])
        self.assertAlmostEqual(make['doug_styling'][0], 7.0)
        listings = pl.DataFrame({'vin': ['a','b','c','d'], 'ModelYear': [2020]*4,
            'Make': [' TOYOTA ', 'Toyota', 'Honda', None], 'Model': ['A','Z','Z','Z'],
            'Trim': ['Sport', None, 'Sport', 'Sport']})
        joined = features.join_features(listings, exact, make)
        self.assertEqual(joined['vin'].to_list(), listings['vin'].to_list())
        self.assertEqual(joined['Make'].to_list(), listings['Make'].to_list())
        self.assertEqual(joined['doug_source_level'].to_list(), ['cohort','make_average','missing','missing'])
        self.assertEqual(joined['doug_styling'].to_list(), [3,7.0,None,None])
        self.assertEqual(joined['doug_review_count'].to_list(), [2,6,None,None])
        with self.assertRaisesRegex(ValueError, 'uniqueness'):
            features.join_features(listings, pl.concat([exact,exact]), make)

    def test_conflicts_suppress_both_cohort_and_make_evidence(self):
        cohorts = [{'ModelYear':2020, 'Make':'Toyota', 'Model':'A', 'Trim':'Sport'}]
        reviews = [self.review('A Sport', 3, 'one'), self.review('A Sport', 9, 'two')]
        resolved = features.resolve_reviews(reviews, cohorts)
        self.assertTrue(all(r['conflict_excluded'] for r in resolved))
        self.assertEqual(features.feature_tables(resolved)[0].height, 0)
        reviews[0]['styling'] = 11
        with self.assertRaisesRegex(ValueError, 'category'):
            features.resolve_reviews(reviews, cohorts)

    def test_artifact_hash_snapshot_validation_and_empty_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            review = self.review('A Sport')
            review.update(source_snapshot_sha256='snapshot')
            pl.DataFrame([review]).write_parquet(snapshot / 'expert_review_scores.parquet')
            digest = hashlib.sha256((snapshot / 'expert_review_scores.parquet').read_bytes()).hexdigest()
            (snapshot / 'import_manifest.json').write_text(__import__('json').dumps(
                {'files_sha256':{'expert_review_scores.parquet':digest}, 'source_snapshot_sha256':'snapshot'}))
            vpic = root / 'vpic.parquet'
            pl.DataFrame({'ModelYear':[2020], 'Make':['Toyota'], 'Model':['A'], 'Trim':['Sport']}).write_parquet(vpic)
            output = root / 'features'
            built = features.build_features(snapshot,vpic,output)
            exact, make, info = features.load_features(output, vpic_sha256=features.digest(vpic))
            self.assertEqual((exact.height,make.height),(1,0))
            self.assertEqual(info['coverage']['cohort_rows'],1)
            with self.assertRaisesRegex(ValueError, 'different vPIC'):
                features.load_features(output,vpic_sha256='other')
            with self.assertRaises(FileExistsError):
                features.build_features(snapshot,vpic,output)
            with (output / 'make_features.parquet').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                features.load_features(output)
            with self.assertRaisesRegex(ValueError,'Build a fresh'):
                features.load_features(root / 'absent')


if __name__ == "__main__":
    unittest.main()
