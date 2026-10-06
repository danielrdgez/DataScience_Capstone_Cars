"""Focused checks for make title coverage and identical Parquet/SQLite input."""

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

import polars as pl

from YOUTUBE.relevance.prepare_make_sentiment_pilot import audit, prefilter_reason, sample


class MakePilotTest(unittest.TestCase):
    def test_comment_sampling_uses_all_make_title_matches(self):
        rows = [
            ("c1", "v1", "2021 Toyota Camry review", "p1", "CVT sucks", "01-01-2021"),
            ("c2", "v1", "2021 Toyota Camry review", "p1", "Great video!", "01-02-2021"),
            ("c3", "v1", "2021 Toyota Camry review", "p1", "My Camry has been reliable", "01-03-2021"),
            ("c4", "v2", "Toyota versus Honda", "p1", "Honda is better", "01-04-2021"),
            ("c5", "v3", "2021 Honda Civic review", "p1", "Honda is reliable", "01-05-2021"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "comments.db"
            parquet = root / "comments.parquet"
            conn = sqlite3.connect(db)
            try:
                conn.execute("""CREATE TABLE youtube_comments_sentiment
                    (comment_id TEXT, video_id TEXT, video_title TEXT,
                     playlist_id TEXT, text TEXT, published_at TEXT)""")
                conn.executemany("INSERT INTO youtube_comments_sentiment VALUES (?,?,?,?,?,?)", rows)
                conn.commit()
            finally:
                conn.close()
            pl.DataFrame(rows, schema=("comment_id", "video_id", "video_title",
                                       "playlist_id", "text", "published_at"),
                         orient="row").write_parquet(parquet)

            source_summaries = []
            for name, source in (("sqlite", db), ("parquet", parquet)):
                output_dir = root / name
                source_summaries.append(audit(source, "TOYOTA", [], output_dir))
                coverage_path = output_dir / "video_make_coverage.csv"
                with coverage_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    coverage_rows = list(csv.DictReader(handle))
                self.assertEqual({row["video_id"] for row in coverage_rows}, {"v1", "v2"})
                result = sample(source, "TOYOTA", [], output_dir / "labels.csv",
                                2, 2, 1, 42)
                self.assertEqual(result["candidate_videos"], 2)
                with (output_dir / "labels.csv").open("r", encoding="utf-8-sig",
                                                       newline="") as handle:
                    labels = list(csv.DictReader(handle))
                self.assertTrue(labels)
                self.assertEqual({row["video_id"] for row in labels}, {"v1", "v2"})
            self.assertEqual(
                [(s["source_videos"], s["source_comments"], s["title_candidate_videos"])
                 for s in source_summaries],
                [(3, 5, 2), (3, 5, 2)],
            )
        self.assertEqual(prefilter_reason("Great video!"), "exact_creator_filler")
        self.assertEqual(prefilter_reason("CVT sucks"), "")


if __name__ == "__main__":
    unittest.main()
