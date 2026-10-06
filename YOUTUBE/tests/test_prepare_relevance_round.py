"""Protect the blind split, label integrity, and reproducible sampling."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from YOUTUBE.relevance.prepare_relevance_round import FIELDS, check_labels, prepare, read_csv, write_csv


class RelevanceRoundTest(unittest.TestCase):
    def replace_labels(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def fixtures(self, root):
        rows = [{"comment_id": f"c{v}-{i}", "video_id": f"v{v}",
                 "video_title": f"2020 {'Toyota' if v % 2 else 'Honda'} review",
                 "published_at": "01-02-2020", "text": f"My car has issue number {v} item {i}"}
                for v in range(70) for i in range(4)]
        rows += [{"comment_id": f"empty{v}", "video_id": f"v{v}",
                  "video_title": "Toyota review", "published_at": "invalid", "text": ""}
                 for v in range(70)]
        source, vpic = root / "comments.parquet", root / "vpic.parquet"
        pq.write_table(pa.Table.from_pylist(rows), source)
        pq.write_table(pa.Table.from_pylist([{"Make": "TOYOTA"}, {"Make": "HONDA"}]), vpic)
        prior, predictions = root / "prior.csv", root / "predictions.csv"
        write_csv(prior, ("comment_id", "video_id", "text"), [rows[0]])
        write_csv(predictions, ("comment_id", "vehicle_relevant", "predicted_vehicle_relevant"),
                  [{"comment_id": "old_error", "vehicle_relevant": "yes", "predicted_vehicle_relevant": "no"}])
        return source, vpic, prior, predictions

    def test_reproducibility_split_and_label_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.fixtures(root)
            a, b = root / "round_a", root / "round_b"
            prepare(*inputs, a, blind_n=20, train_n=30)
            prepare(*inputs, b, blind_n=20, train_n=30)
            train, blind = read_csv(a / "train_labels.csv"), read_csv(a / "blind_labels.csv")
            self.assertEqual((a / "train_sample.csv").read_bytes(), (b / "train_sample.csv").read_bytes())
            self.assertEqual((a / "blind_sample.csv").read_bytes(), (b / "blind_sample.csv").read_bytes())
            self.assertFalse({r["video_id"] for r in train} & {r["video_id"] for r in blind})
            self.assertNotIn("v0", {r["video_id"] for r in blind})
            self.assertFalse({r["normalized_text_sha256"] for r in train} &
                             {r["normalized_text_sha256"] for r in blind})
            self.assertTrue(all(not r["prefilter_reason"] for r in train))
            self.assertEqual(len(read_csv(a / "toyota_error_review.csv")), 1)
            audit = json.loads((a / "source_audit.json").read_text())
            self.assertEqual(audit["empty_text"], 70)
            self.assertEqual(audit["publication_date_min"], "2020-01-02")
            result = check_labels(a, a / "partial.json")
            self.assertEqual(result["blind_completion"]["unlabeled"], 20)
            self.assertFalse(result["ready_for_model_comparison"])
            self.assertNotIn("blind_label_counts", result)
            # A copied ID, lost row, or edited text cannot silently enter training.
            train[0]["text"] = "accidental edit"
            with (a / "train_labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(train)
            with self.assertRaisesRegex(ValueError, "changed source"):
                check_labels(a, a / "bad.json")
            with self.assertRaises(FileExistsError):
                prepare(*inputs, a, blind_n=20, train_n=30)

    def test_duplicate_source_ids_fail_without_publishing_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.fixtures(root)
            source = inputs[0]
            rows = pq.read_table(source).to_pylist()
            rows.append(rows[0])
            pq.write_table(pa.Table.from_pylist(rows), source)
            output = root / "bad_round"
            with self.assertRaisesRegex(ValueError, "repeated comment IDs"):
                prepare(*inputs, output, blind_n=20, train_n=30)
            self.assertFalse(output.exists())

    def test_label_counts_and_second_reviewer_cannot_be_inflated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "round"
            prepare(*self.fixtures(root), output, blind_n=20, train_n=30)
            train = read_csv(output / "train_labels.csv")
            blind = read_csv(output / "blind_labels.csv")
            for i, row in enumerate(train):
                row["vehicle_relevant"] = "no" if i < 12 else "yes"
                row["negative_subtype"] = "creator_video" if i < 12 else ""
            for row in blind:
                row["vehicle_relevant"] = "unclear"
            self.replace_labels(output / "train_labels.csv", train)
            self.replace_labels(output / "blind_labels.csv", blind)
            by_id = {r["comment_id"]: r for r in train + blind}
            second = read_csv(output / "second_reviewer_labels.csv")
            for row in second:
                row.update({key: by_id[row["comment_id"]][key] for key in FIELDS[-3:]})
            self.replace_labels(output / "second_reviewer_labels.csv", second)
            result = check_labels(output, output / "counts.json")
            self.assertEqual(result["additional_nonfiller_negatives"], 12)
            self.assertEqual(result["training_disagreement_rate"], 0)
            self.assertEqual(result["blind_completion"]["unlabeled"], 0)
            self.assertFalse(result["ready_for_model_comparison"])
            self.replace_labels(output / "second_reviewer_labels.csv", second + [second[0]])
            with self.assertRaisesRegex(ValueError, "duplicate second-reviewer"):
                check_labels(output, output / "bad_second.json")
            self.replace_labels(output / "second_reviewer_labels.csv", second)
            train[0]["negative_subtype"] = ""
            self.replace_labels(output / "train_labels.csv", train)
            with self.assertRaisesRegex(ValueError, "Add negative_subtype"):
                check_labels(output, output / "bad_negative.json")


if __name__ == "__main__":
    unittest.main()
