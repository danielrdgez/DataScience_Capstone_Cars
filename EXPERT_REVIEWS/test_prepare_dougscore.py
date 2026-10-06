"""Integrity and linkage boundaries for the expert-opinion import."""
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from EXPERT_REVIEWS.prepare_dougscore import import_snapshot, parse_snapshot, coverage


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


if __name__ == "__main__":
    unittest.main()
