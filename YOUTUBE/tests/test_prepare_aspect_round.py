"""Test source-preserving multi-make annotations and explicit aspect absence."""

import tempfile
import unittest
from pathlib import Path

from YOUTUBE.relevance.prepare_aspect_round import ASPECTS, prepare, check_labels
from YOUTUBE.relevance.prepare_relevance_round import read_csv
from YOUTUBE.tests.test_relevance_candidate import fixture, replace_any_csv


class AspectRoundTest(unittest.TestCase):
    def test_prepare_uses_only_confirmed_positives_and_never_blind_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); round_dir, _ = fixture(root)
            (round_dir / "blind_labels.csv").write_text("invalid blind file", encoding="utf-8")
            before = (round_dir / "train_labels.csv").read_bytes()
            aspect = root / "aspect"
            result = prepare(round_dir, aspect, size=6)
            self.assertEqual(result["sampled_comments"], 6)
            self.assertEqual(len(read_csv(aspect / "aspect_labels.csv")), 6 * len(ASPECTS))
            self.assertTrue(all("engine" in r["text"] for r in read_csv(aspect / "comment_sample.csv")))
            self.assertEqual((round_dir / "train_labels.csv").read_bytes(), before)
            progress = check_labels(aspect, root / "progress")
            self.assertEqual(progress["unfinished_comments"], 6)
            self.assertEqual(progress["aspect_opinion_records"], 0)

    def test_multiple_makes_and_evidence_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); round_dir, _ = fixture(root)
            aspect = root / "aspect"
            prepare(round_dir, aspect, size=1)
            targets = read_csv(aspect / "target_labels.csv")
            first = targets[0]
            first.update(target_resolution_status="resolved", Make="TOYOTA", opinion_expressed="yes")
            targets.append({**first, "target_index": "2", "Make": "HONDA"})
            replace_any_csv(aspect / "target_labels.csv", targets)
            rows = read_csv(aspect / "aspect_labels.csv")
            second = [{**r, "target_index": "2"} for r in rows]
            rows += second
            for r in rows:
                r["aspect_present"] = "no"
                if r["aspect"] == "reliability":
                    r.update(aspect_present="yes", polarity="negative", evidence_quote="My engine failed",
                             reliability_claim_basis="firsthand_owner")
            replace_any_csv(aspect / "aspect_labels.csv", rows)
            result = check_labels(aspect, root / "checked")
            self.assertEqual(result["annotation_complete_comments"], 1)
            self.assertEqual(result["aspect_opinion_records"], 2)
            self.assertEqual({r["Make"] for r in read_csv(root / "checked/labeled_aspects.csv")}, {"TOYOTA", "HONDA"})
            rows[0]["evidence_quote"] = "Invented evidence"
            replace_any_csv(aspect / "aspect_labels.csv", rows)
            with self.assertRaisesRegex(ValueError, "verbatim source evidence"):
                check_labels(aspect, root / "bad_evidence")

    def test_no_opinion_is_complete_and_produces_no_neutral_sentiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); round_dir, _ = fixture(root)
            aspect = root / "aspect"
            prepare(round_dir, aspect, size=1)
            targets = read_csv(aspect / "target_labels.csv")
            targets[0].update(target_resolution_status="resolved", Make="TOYOTA", opinion_expressed="no")
            replace_any_csv(aspect / "target_labels.csv", targets)
            result = check_labels(aspect, root / "checked")
            self.assertEqual(result["annotation_complete_comments"], 1)
            self.assertEqual(result["aspect_opinion_records"], 0)
            targets[0]["text"] = "Substantive source edit"
            replace_any_csv(aspect / "target_labels.csv", targets)
            with self.assertRaisesRegex(ValueError, "Changed frozen source"):
                check_labels(aspect, root / "bad_source")


if __name__ == "__main__":
    unittest.main()
