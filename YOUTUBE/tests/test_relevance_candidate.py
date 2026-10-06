"""Test frozen candidates, preserving pilots, restart, and one-use blind tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from YOUTUBE.relevance.partial_relevance import snapshot
from YOUTUBE.relevance.prepare_relevance_round import FIELDS, SOURCE_FIELDS, file_hash, read_csv, write_json
from YOUTUBE.relevance.relevance_candidate import (candidate_metadata, evaluate_blind, fit_candidate,
                                         read_json, reserved_video, score_pilot, validation_metrics, POLICY)
from YOUTUBE.tests.test_partial_relevance import PartialRelevanceTest, replace_csv


class FakeModel:
    calls = 0
    max_seq_length = 256

    @classmethod
    def train(cls, rows, encoder_dir, output, config):
        output.mkdir()
        write_json(output / "fake.json", {"training_ids": [r["comment_id"] for r in rows]})
        return cls()

    @classmethod
    def load(cls, directory, device):
        return cls()

    def score(self, texts, batch_size=32):
        FakeModel.calls += 1
        return np.array([.8 if "engine" in t else .1 for t in texts])

    def lengths(self, texts):
        return [len(t.split()) for t in texts]


def replace_any_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def fixture(root):
    raw, round_dir = root / "raw", root / "round"
    PartialRelevanceTest().fixture(raw)
    snapshot(raw, round_dir)
    original = read_csv(round_dir / "train_sample.csv") + read_csv(round_dir / "blind_sample.csv")
    source_rows = [{k: r[k] for k in SOURCE_FIELDS} for r in original]
    source_rows += [{"comment_id": f"fresh{i}", "video_id": f"fresh_video{i}",
                     "video_title": "TOYOTA car review" if i % 2 else "HONDA car review",
                     "published_at": "01-01-2020" if i % 3 else "01-01-2021",
                     "text": f"The engine works {i}" if i % 2 else f"Great camera work {i}"}
                    for i in range(90)]
    # A fresh ID/video with protected exact text must also be excluded.
    source_rows.append({**source_rows[0], "comment_id": "duplicate_text", "video_id": "new_duplicate_video"})
    source = root / "source.parquet"
    pq.write_table(pa.Table.from_pylist(source_rows), source)
    manifest = read_json(round_dir / "sample_manifest.json")
    manifest.update(source={"path": str(source), "sha256": file_hash(source)},
                    vpic={"makes": ["HONDA", "TOYOTA"]}, seed=42, reserve_fraction=.2)
    (round_dir / "sample_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    comparison = round_dir / "model_comparison_v1"
    comparison.mkdir()
    write_json(comparison / "run_manifest.json", {
        "config": {"train_labels_sha256": file_hash(round_dir / "train_labels.csv"),
                   "setfit_steps": 2, "setfit_iterations": 1}, "encoder_revision": "fixture_revision"})
    write_json(comparison / "comparison_metrics.json", {"models": {"setfit": {}}})
    return round_dir, source


class CandidateTest(unittest.TestCase):
    def test_fit_never_opens_blind_annotations_and_detects_artifact_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); round_dir, _ = fixture(root)
            (round_dir / "blind_labels.csv").write_text("not a label file", encoding="utf-8")
            before = {p.name: p.read_bytes() for p in round_dir.glob("*.csv")}
            candidate = root / "candidate"
            result = fit_candidate(round_dir, candidate, backend_class=FakeModel)
            self.assertEqual(result["training_counts"], {"yes": 10, "no": 10})
            self.assertFalse(result["automatic_rejection_enabled"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in round_dir.glob("*.csv")})
            self.assertEqual(candidate_metadata(candidate)["reload_max_absolute_difference"], 0)
            (candidate / "model/fake.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                candidate_metadata(candidate)

    def test_pilot_preserves_all_rows_protects_sources_and_resumes_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); round_dir, source = fixture(root)
            candidate, pilot = root / "candidate", root / "pilot"
            fit_candidate(round_dir, candidate, backend_class=FakeModel)
            FakeModel.calls = 0
            result = score_pilot(candidate, source, pilot, size=25, batch_size=7, backend_class=FakeModel)
            rows = read_csv(pilot / "pilot_scores.csv")
            self.assertEqual(result["rows"], 25)
            self.assertEqual(result["actual_exclusions"], 0)
            self.assertTrue(all(r["relevance_decision"] == "review_or_pass" for r in rows))
            self.assertTrue(all(r["comment_id"].startswith("fresh") for r in rows))
            self.assertTrue(all(not reserved_video(r["video_id"], {"seed": 42, "fraction": .2}) for r in rows))
            self.assertEqual(len({r["comment_id"] for r in rows}), 25)
            predictions = (pilot / "pilot_scores.csv").read_bytes()
            (pilot / "pilot_audit_labels.csv").write_text("human edits to preserve", encoding="utf-8")
            calls = FakeModel.calls
            score_pilot(candidate, source, pilot, size=25, batch_size=7, resume=True, backend_class=FakeModel)
            self.assertEqual(FakeModel.calls, calls)
            self.assertEqual((pilot / "pilot_scores.csv").read_bytes(), predictions)
            self.assertEqual((pilot / "pilot_audit_labels.csv").read_text(), "human edits to preserve")
            with self.assertRaisesRegex(ValueError, "Resume refused"):
                score_pilot(candidate, source, pilot, size=26, batch_size=7, resume=True, backend_class=FakeModel)
            checkpoint = pilot / "chunks/batch_00001.json"
            broken = read_json(checkpoint)
            broken["probabilities"][0] = float("nan")
            checkpoint.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid probabilities"):
                score_pilot(candidate, source, pilot, size=25, batch_size=7, resume=True, backend_class=FakeModel)

    def test_partial_blind_does_not_claim_test_and_completed_test_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); round_dir, _ = fixture(root)
            candidate = root / "candidate"
            fit_candidate(round_dir, candidate, backend_class=FakeModel)
            with self.assertRaisesRegex(ValueError, "Finish all blind"):
                evaluate_blind(candidate, round_dir, root / "evaluation", reviewer_unavailable=True, backend_class=FakeModel)
            self.assertFalse((round_dir / "blind_evaluation_receipt.json").exists())
            self.assertFalse((root / "evaluation").exists())
            blind = read_csv(round_dir / "blind_labels.csv")
            for r in blind:
                r["vehicle_relevant"] = "yes" if "engine" in r["text"] else "no"
                r["negative_subtype"] = "other" if r["vehicle_relevant"] == "no" else ""
            replace_csv(round_dir / "blind_labels.csv", blind)
            report = evaluate_blind(candidate, round_dir, root / "evaluation", reviewer_unavailable=True, backend_class=FakeModel)
            self.assertEqual(report["relevant_recall"], 1)
            self.assertFalse(report["statistical_screen_passed"])
            self.assertFalse(report["release_recommended"])
            self.assertEqual(read_json(round_dir / "blind_evaluation_receipt.json")["state"], "complete")
            with self.assertRaisesRegex(ValueError, "already claimed"):
                evaluate_blind(candidate, round_dir, root / "another_eval", reviewer_unavailable=True, backend_class=FakeModel)

    def test_uncertainty_guards_against_perfect_small_samples(self):
        rows = [{"vehicle_relevant": "yes" if i % 2 else "no", "video_id": str(i)} for i in range(20)]
        scores = [.9 if i % 2 else .1 for i in range(20)]
        result = validation_metrics(rows, scores, .2, {**POLICY, "bootstrap_replicates": 20}, 42)
        self.assertTrue(result["point_targets_met"])
        self.assertFalse(result["interval_targets_met"])
        self.assertLess(result["wilson_95_ci"]["relevant_recall"][0], .98)

    def test_changed_source_and_fitted_labels_are_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); round_dir, source = fixture(root)
            candidate = root / "candidate"
            fit_candidate(round_dir, candidate, backend_class=FakeModel)
            with source.open("ab") as f:
                f.write(b"changed snapshot")
            with self.assertRaisesRegex(ValueError, "source differs"):
                score_pilot(candidate, source, root / "bad_pilot", size=5, backend_class=FakeModel)
            self.assertFalse((root / "bad_pilot").exists())
            train = read_csv(round_dir / "train_labels.csv")
            train[0]["vehicle_relevant"] = "yes"
            train[0]["negative_subtype"] = ""
            replace_csv(round_dir / "train_labels.csv", train)
            with self.assertRaisesRegex(ValueError, "binary label changed"):
                evaluate_blind(candidate, round_dir, root / "bad_eval", reviewer_unavailable=True, backend_class=FakeModel)
            self.assertFalse((round_dir / "blind_evaluation_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
