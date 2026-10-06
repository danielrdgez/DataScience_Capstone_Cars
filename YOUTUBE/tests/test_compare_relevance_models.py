"""Test review import, identical folds, blind isolation, and safe restart."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from YOUTUBE.relevance.compare_relevance_models import compare, metrics_for
from YOUTUBE.relevance.partial_relevance import apply_reviews, evaluate_partial, snapshot
from YOUTUBE.relevance.prepare_relevance_round import read_csv
from YOUTUBE.tests.test_partial_relevance import PartialRelevanceTest, replace_csv


class StubRunner:
    calls = []

    def __init__(self, *args):
        pass

    def prepare_embeddings(self, texts):
        pass

    def predict(self, method, fold, rows, train, test):
        StubRunner.calls.append((method, fold, train.tolist(), test.tolist()))
        return np.array([.75 if rows[i]["vehicle_relevant"] == "yes" else .25 for i in test])


class CompareRelevanceTest(unittest.TestCase):
    def fixture(self, root):
        source, working = root / "source", root / "working"
        PartialRelevanceTest().fixture(source)
        snapshot(source, working)
        return working

    def test_review_import_does_not_let_stale_subtype_export_undo_relevance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working = self.fixture(root)
            baseline = root / "baseline"
            evaluate_partial(working, baseline, folds=3)
            errors = read_csv(baseline / "cv_error_review.csv")
            # Replace the error file with a user-confirmed no->yes correction.
            original = read_csv(baseline / "relevance_cv_predictions.csv")[0]
            original["vehicle_relevant"] = "no"
            original["negative_subtype"] = "generic_reaction"
            # Fixture's first row is truly the existing no label.
            original["review_notes"] = "yes"
            import csv
            with (baseline / "cv_error_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(original))
                writer.writeheader()
                writer.writerow(original)
            # Same row in a subtype export still has old binary no. This isn't a new correction.
            issue = {**original, "negative_subtype": "other"}
            issue.pop("review_notes")
            with (baseline / "annotation_issues.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(issue))
                writer.writeheader()
                writer.writerow(issue)
            before = (working / "train_labels.csv").read_bytes()
            result = apply_reviews(working, baseline, root / "reviewed")
            reviewed = read_csv(root / "reviewed/train_labels.csv")[0]
            self.assertEqual(reviewed["vehicle_relevant"], "yes")
            self.assertEqual(reviewed["negative_subtype"], "")
            self.assertEqual((working / "train_labels.csv").read_bytes(), before)
            self.assertEqual(result["changes_by_field"]["vehicle_relevant"], 1)
            self.assertEqual((root / "reviewed/blind_labels.csv").read_bytes(),
                             (working / "blind_labels.csv").read_bytes())

    def test_all_methods_share_splits_and_resume_does_not_retrain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working = self.fixture(root)
            # This file must not be opened by model comparison.
            (working / "blind_labels.csv").write_text("not a label CSV", encoding="utf-8")
            StubRunner.calls = []
            result = compare(working, root / "compare", folds=3, runner_factory=StubRunner)
            self.assertEqual(len(StubRunner.calls), 9)
            for fold in range(1, 4):
                seen = [(train, test) for _, n, train, test in StubRunner.calls if n == fold]
                self.assertTrue(all(split == seen[0] for split in seen))
                self.assertFalse(set(seen[0][0]) & set(seen[0][1]))
            self.assertFalse(result["production_ready"])
            compare(working, root / "compare", folds=3, resume=True, runner_factory=StubRunner)
            self.assertEqual(len(StubRunner.calls), 9)
            labels = read_csv(working / "train_labels.csv")
            labels[0]["vehicle_relevant"] = "yes"
            replace_csv(working / "train_labels.csv", labels)
            with self.assertRaisesRegex(ValueError, "Resume refused"):
                compare(working, root / "compare", folds=3, resume=True, runner_factory=StubRunner)

    def test_zero_rejections_have_no_rejection_precision(self):
        rows = [{"vehicle_relevant": "yes"}, {"vehicle_relevant": "no"}]
        result = metrics_for(rows, [.9, .8])
        self.assertIsNone(result["rejection_precision"])
        self.assertEqual(result["relevant_recall"], 1)


if __name__ == "__main__":
    unittest.main()
