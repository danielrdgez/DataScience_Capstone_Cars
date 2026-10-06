"""Check annotation recovery, reviewed-range bounds, and blind isolation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from YOUTUBE.relevance.partial_relevance import evaluate_partial, snapshot
from YOUTUBE.relevance.prepare_relevance_round import FIELDS, file_hash, read_csv, write_csv, write_json


def replace_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class PartialRelevanceTest(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        samples = {}
        for split, size in (("train", 24), ("blind", 6)):
            rows = []
            for i in range(size):
                row = dict.fromkeys(FIELDS, "")
                row.update(comment_id=f"{split}{i}", video_id=f"{split}_v{i}",
                           video_title="Car review", published_at="01-01-2020",
                           text=(f"My engine failed\nreport {i}" if i % 2 else f"Great camera work\nmessage {i}"),
                           split=split, sample_stratum="general_nonfiller", normalized_text_sha256=f"{split}_hash{i}")
                rows.append(row)
            samples[split] = rows
            write_csv(root / f"{split}_sample.csv", FIELDS, rows)
            write_csv(root / f"{split}_labels.csv", FIELDS, rows)
        write_csv(root / "second_reviewer_labels.csv", FIELDS, [])
        write_json(root / "sample_manifest.json", {
            "sample_sha256": {f"{split}_sample.csv": file_hash(root / f"{split}_sample.csv") for split in samples},
            "source": {"path": "fixture"}, "dependencies": {},
        })
        train = samples["train"]
        for i, row in enumerate(train[:20]):
            row["vehicle_relevant"] = "yes" if i % 2 else "no"
            row["negative_subtype"] = "generic_reaction" if not i % 2 else ""
            row["text"] = row["text"].replace("\n", "\r\n")
        replace_csv(root / "train_labels.csv", train)
        blind = samples["blind"]
        blind[0]["vehicle_relevant"] = "no"
        replace_csv(root / "blind_sample.csv", blind)

    def test_recovery_is_bounded_preserves_inputs_and_excludes_blind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, working, evaluated = root / "source", root / "working", root / "evaluated"
            self.fixture(source)
            before = {p.name: p.read_bytes() for p in source.glob("*.csv")}
            recovery = snapshot(source, working, reviewed_train=22, reviewed_blind=4, reviewed_blanks="no")
            self.assertEqual(recovery["training_labels"], {"no": 11, "yes": 10, "unlabeled": 3})
            blind = read_csv(working / "blind_labels.csv")
            self.assertEqual([r["vehicle_relevant"] for r in blind], ["no", "no", "no", "", "", ""])
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.glob("*.csv")})
            self.assertTrue(all("\r" not in r["text"] for r in read_csv(working / "train_labels.csv")))
            # Evaluation succeeds even when blind annotation file is unreadable/invalid.
            (working / "blind_labels.csv").write_text("not a valid annotation CSV", encoding="utf-8")
            metrics = evaluate_partial(working, evaluated, folds=3)
            self.assertEqual(metrics["training_label_counts"], {"no": 11, "yes": 10})
            self.assertEqual(metrics["pending_training_rows"], 3)
            self.assertFalse(metrics["production_ready"])
            self.assertTrue(all(r["comment_id"].startswith("train") for r in
                                read_csv(evaluated / "relevance_cv_predictions.csv")))
            with self.assertRaises(FileExistsError):
                evaluate_partial(working, evaluated, folds=3)

    def test_substantive_text_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            self.fixture(source)
            rows = read_csv(source / "train_labels.csv")
            rows[0]["text"] = "A different opinion"
            replace_csv(source / "train_labels.csv", rows)
            with self.assertRaisesRegex(ValueError, "Substantive source edit"):
                snapshot(source, root / "working")
            self.assertFalse((root / "working").exists())


if __name__ == "__main__":
    unittest.main()
