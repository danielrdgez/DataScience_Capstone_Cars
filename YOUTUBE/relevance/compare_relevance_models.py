"""Compare local relevance candidates on identical training-only video folds.

Neural candidates download a pinned public encoder and train locally. No blind
annotations, source-wide scoring, model release, or external logging is used.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

try:
    from .partial_relevance import EDITABLE, index_rows, verify_source
    from .prepare_relevance_round import FIELDS, file_hash, read_csv, write_csv, write_json
except ImportError:
    from partial_relevance import EDITABLE, index_rows, verify_source
    from prepare_relevance_round import FIELDS, file_hash, read_csv, write_csv, write_json

METHODS = ("tfidf", "embeddings", "setfit")
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def load_training(directory: Path) -> list[dict]:
    manifest = json.loads((directory / "sample_manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["sample_sha256"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError(f"Frozen sample changed: {name}")
    original = read_csv(directory / "train_sample.csv")
    supplied = read_csv(directory / "train_labels.csv")
    verify_source(original, supplied)
    canonical = index_rows(original)
    # Blind template supplies source keys only. Never open blind_labels.csv.
    blind = read_csv(directory / "blind_sample.csv")
    blind_videos = {r["video_id"] for r in blind}
    blind_hashes = {r["normalized_text_sha256"] for r in blind}
    if any(r["video_id"] in blind_videos or r["normalized_text_sha256"] in blind_hashes for r in supplied):
        raise ValueError("Training overlaps blind source identities")
    if len({r["normalized_text_sha256"] for r in supplied}) != len(supplied):
        raise ValueError("Duplicate normalized training text must be resolved before CV")
    rows = []
    for number, row in enumerate(supplied, 2):
        if row["vehicle_relevant"] not in {"", "yes", "no", "unclear"}:
            raise ValueError(f"Invalid label at training CSV row {number}")
        if row["vehicle_relevant"] in {"yes", "no"}:
            rows.append({**canonical[row["comment_id"]], **{k: row[k] for k in EDITABLE}, "csv_row": number})
    if len(rows) < 10 or len({r["vehicle_relevant"] for r in rows}) != 2:
        raise ValueError("Need at least ten yes/no training labels with both classes")
    return rows


def splits_for(rows: list[dict], folds: int, seed: int) -> list:
    if folds < 2 or len({r["video_id"] for r in rows}) < folds:
        raise ValueError("Need at least two folds and enough distinct training videos")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = [r["vehicle_relevant"] == "yes" for r in rows]
    groups = [r["video_id"] for r in rows]
    splits = list(splitter.split(np.zeros(len(rows)), labels, groups))
    for train, test in splits:
        if len({labels[i] for i in train}) != 2:
            raise ValueError("One CV training fold lacks a class; reduce --folds")
        if {groups[i] for i in train} & {groups[i] for i in test}:
            raise AssertionError("Video leakage in CV")
    return splits


def metrics_for(rows: list[dict], probabilities, cutoff: float = .5) -> dict:
    labels = np.array([r["vehicle_relevant"] == "yes" for r in rows])
    scores = np.asarray(probabilities, dtype=float)
    if scores.shape != labels.shape or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid probability shape/range")
    prediction = scores >= cutoff
    tp, fn = int((labels & prediction).sum()), int((labels & ~prediction).sum())
    tn, fp = int((~labels & ~prediction).sum()), int((~labels & prediction).sum())
    return {"n": len(rows), "cutoff": cutoff, "true_positive": tp, "false_negative": fn,
            "true_negative": tn, "false_positive": fp,
            "relevant_recall": tp / (tp + fn) if tp + fn else None,
            "irrelevant_recall": tn / (tn + fp) if tn + fp else None,
            "rejection_precision": tn / (tn + fn) if tn + fn else None,
            "brier_score": float(brier_score_loss(labels, scores))}


def cluster_intervals(rows: list[dict], scores, seed: int) -> dict:
    """Approximate 95% intervals resampling whole video groups, not individual rows."""
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["video_id"], []).append(index)
    blocks = list(groups.values())
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("relevant_recall", "rejection_precision")}
    scores = np.asarray(scores)
    for _ in range(500):
        indices = [index for block in rng.integers(0, len(blocks), len(blocks)) for index in blocks[block]]
        result = metrics_for([rows[i] for i in indices], scores[indices])
        for key in values:
            if result[key] is not None:
                values[key].append(result[key])
    return {key: [float(v) for v in np.quantile(value, [.025, .975])] if value else None
            for key, value in values.items()}


def dependency_versions() -> dict:
    result = {}
    for name in ("scikit-learn", "numpy", "torch", "sentence-transformers", "setfit", "datasets",
                 "transformers", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def preflight(methods):
    needed = []
    if set(methods) & {"embeddings", "setfit"}:
        needed.extend(("torch", "sentence_transformers", "huggingface_hub"))
    if "setfit" in methods:
        needed.extend(("setfit", "datasets"))
    missing = [module for module in needed if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError("Missing neural dependencies: " + ", ".join(missing) +
                           ". Install with .\\.venv\\Scripts\\python.exe -m pip install -r "
                           "YOUTUBE\\requirements_relevance.txt")


class CandidateRunner:
    def __init__(self, model_dir, device, output, steps, iterations, seed):
        self.model_dir, self.device, self.output = model_dir, device, output
        self.steps, self.iterations, self.seed = steps, iterations, seed
        self.embeddings = None

    def prepare_embeddings(self, texts):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(str(self.model_dir), device=self.device, trust_remote_code=False)
        token_lengths = [len(encoder.tokenizer(text, truncation=False)["input_ids"]) for text in texts]
        atomic_json(self.output / "encoder_input_audit.json", {
            "max_seq_length": encoder.max_seq_length,
            "rows_exceeding_token_limit": sum(n > encoder.max_seq_length for n in token_lengths),
            "note": "Neural input is truncated at the encoder limit; source text remains unchanged. "
                    "Fixed pretrained embeddings may be computed once before CV because no supervised fitting occurs.",
        })
        self.embeddings = encoder.encode(texts, batch_size=32, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=True)
        del encoder
        gc.collect()

    def predict(self, method, fold, rows, train, test):
        texts = [r["text"] for r in rows]
        labels = np.array([r["vehicle_relevant"] == "yes" for r in rows], dtype=int)
        head = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.seed)
        if method == "tfidf":
            model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                                 max_features=100_000, sublinear_tf=True), head)
            model.fit([texts[i] for i in train], labels[train])
            return model.predict_proba([texts[i] for i in test])[:, 1]
        if method == "embeddings":
            head.fit(self.embeddings[train], labels[train])
            return head.predict_proba(self.embeddings[test])[:, list(head.classes_).index(1)]
        import torch
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer
        from setfit import SetFitModel, Trainer, TrainingArguments
        torch.manual_seed(self.seed + fold)
        # Reload the pristine encoder for every fold; never reuse a fine-tuned body.
        body = SentenceTransformer(str(self.model_dir), device=self.device, trust_remote_code=False)
        model = SetFitModel(model_body=body, model_head=head, normalize_embeddings=True)
        arguments = TrainingArguments(
            output_dir=str(self.output / f"setfit_fold_{fold}"), batch_size=16,
            num_epochs=1, num_iterations=self.iterations, max_steps=self.steps,
            seed=self.seed + fold, report_to="none", save_strategy="no", eval_strategy="no",
        )
        dataset = Dataset.from_dict({"text": [texts[i] for i in train], "label": labels[train].tolist()})
        trainer = Trainer(model=model, args=arguments, train_dataset=dataset)
        trainer.train()
        encoded = model.encode([texts[i] for i in test], batch_size=32, show_progress_bar=False)
        scores = model.model_head.predict_proba(encoded)[:, list(model.model_head.classes_).index(1)]
        del trainer, model, body
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return scores


def atomic_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def compare(directory: Path, output: Path, methods=METHODS, folds=5, seed=42,
            encoder=ENCODER, revision=None, device="cpu", steps=100, iterations=5,
            resume=False, cache_dir=None, runner_factory=None) -> dict:
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Choose unique supported methods: tfidf embeddings setfit")
    if steps < 1 or iterations < 1:
        raise ValueError("Positive SetFit steps/iterations required")
    if output.exists() and not resume:
        raise FileExistsError(f"Choose a new output directory or use --resume: {output}")
    rows = load_training(directory)
    splits = splits_for(rows, folds, seed)
    if runner_factory is None:
        preflight(methods)
    config = {"train_labels_sha256": file_hash(directory / "train_labels.csv"), "methods": list(methods),
              "folds": folds, "seed": seed, "encoder": encoder, "requested_revision": revision,
              "device": device, "setfit_steps": steps, "setfit_iterations": iterations,
              "script_sha256": file_hash(Path(__file__)), "dependencies": dependency_versions()}
    manifest_path = output / "run_manifest.json"
    if resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("Resume refused: labels, code, dependencies, or run configuration changed")
    else:
        output.mkdir(parents=True)
        previous = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                    "training_counts": dict(Counter(r["vehicle_relevant"] for r in rows)),
                    "training_videos": len({r["video_id"] for r in rows}),
                    "note": "Review-only comparison on enriched, partially labeled training data. Labels were "
                            "revised after model error inspection; these CV results are not an independent test. "
                            "Blind annotation files are never read. No final model or production filter is released."}
        atomic_json(manifest_path, previous)
        fold_rows = [{"comment_id": rows[i]["comment_id"], "video_id": rows[i]["video_id"], "cv_fold": fold}
                     for fold, (_, test) in enumerate(splits, 1) for i in test]
        write_csv(output / "fold_assignments.csv", ("comment_id", "video_id", "cv_fold"), fold_rows)
    model_dir = None
    if set(methods) & {"embeddings", "setfit"} and runner_factory is None:
        from huggingface_hub import HfApi, snapshot_download
        pinned = previous.get("encoder_revision") or HfApi().model_info(encoder, revision=revision, token=False).sha
        previous["encoder_revision"] = pinned
        atomic_json(manifest_path, previous)
        print(f"Loading public encoder {encoder} at {pinned}; local inference only", flush=True)
        model_dir = snapshot_download(encoder, revision=pinned, token=False,
                                      cache_dir=cache_dir or output.parent / "model_cache",
                                      allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
    runner = (runner_factory or CandidateRunner)(model_dir, device, output, steps, iterations, seed)
    need_embeddings = "embeddings" in methods and any(not (output / f"embeddings_fold_{i}.json").exists()
                                                      for i in range(1, folds + 1))
    if need_embeddings:
        runner.prepare_embeddings([r["text"] for r in rows])
    final = {"production_ready": False, "models": {}, "note": previous["note"]}
    summary_rows, threshold_rows, predictions, calibration_rows, segment_rows = [], [], [], [], []
    for method in methods:
        scores = np.full(len(rows), np.nan)
        timings = []
        fold_numbers = np.zeros(len(rows), dtype=int)
        for fold, (train, test) in enumerate(splits, 1):
            checkpoint_path = output / f"{method}_fold_{fold}.json"
            expected_ids = [rows[i]["comment_id"] for i in test]
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if checkpoint["comment_ids"] != expected_ids:
                    raise ValueError("Checkpoint no longer matches fold identities")
                print(f"Reusing {method} fold {fold}/{folds}", flush=True)
            else:
                print(f"Training {method} fold {fold}/{folds} ({len(train)} train, {len(test)} validation)", flush=True)
                started = time.perf_counter()
                values = np.asarray(runner.predict(method, fold, rows, train, test), dtype=float)
                metrics_for([rows[i] for i in test], values)
                checkpoint = {"comment_ids": expected_ids, "probabilities": values.tolist(),
                              "seconds": time.perf_counter() - started}
                atomic_json(checkpoint_path, checkpoint)
            scores[test] = checkpoint["probabilities"]
            fold_numbers[test] = fold
            timings.append(checkpoint["seconds"])
        result = metrics_for(rows, scores)
        result["video_bootstrap_95_ci"] = cluster_intervals(rows, scores, seed)
        result["fit_and_validation_seconds"] = sum(timings)
        final["models"][method] = result
        summary_rows.append({"model": method, **{k: v for k, v in result.items() if k != "video_bootstrap_95_ci"}})
        for cutoff in (.01, .02, .05, .10, .15, .20, .25, .30, .40, .50):
            threshold_rows.append({"model": method, **metrics_for(rows, scores, cutoff)})
        for row, score, fold in zip(rows, scores, fold_numbers):
            predictions.append({**row, "model": method, "cv_fold": int(fold),
                                "prob_vehicle_relevant": float(score),
                                "predicted_vehicle_relevant": "yes" if score >= .5 else "no"})
        for bin_index in range(10):
            selected = [i for i, score in enumerate(scores) if min(int(score * 10), 9) == bin_index]
            calibration_rows.append({"model": method, "bin_lower": bin_index / 10, "n": len(selected),
                                     "mean_probability": float(np.mean(scores[selected])) if selected else None,
                                     "observed_relevant_fraction": float(np.mean([rows[i]["vehicle_relevant"] == "yes"
                                                                                for i in selected])) if selected else None})
        for field in ("sample_stratum", "candidate_makes"):
            for value in sorted({r[field] for r in rows}):
                indices = [i for i, row in enumerate(rows) if row[field] == value]
                segment_rows.append({"model": method, "segment_field": field, "segment_value": value or "unmatched",
                                     **metrics_for([rows[i] for i in indices], scores[indices])})
        atomic_json(output / "comparison_metrics.json", final)
    # Atomic replacement allows a matching resumed run to finish its derived summaries.
    def publish_csv(name, fields, data):
        temporary = output / (name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        write_csv(temporary, fields, data)
        os.replace(temporary, output / name)
    metric_fields = tuple(summary_rows[0])
    publish_csv("model_comparison.csv", metric_fields, summary_rows)
    publish_csv("threshold_diagnostics.csv", tuple(threshold_rows[0]), threshold_rows)
    prediction_fields = (*FIELDS, "csv_row", "model", "cv_fold", "prob_vehicle_relevant", "predicted_vehicle_relevant")
    publish_csv("cv_predictions.csv", prediction_fields, predictions)
    publish_csv("cv_errors.csv", (*prediction_fields, "review_notes"),
                [r for r in predictions if r["vehicle_relevant"] != r["predicted_vehicle_relevant"]])
    publish_csv("calibration_bins.csv", tuple(calibration_rows[0]), calibration_rows)
    publish_csv("segment_metrics.csv", tuple(segment_rows[0]), segment_rows)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--setfit-steps", type=int, default=100)
    parser.add_argument("--setfit-iterations", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = compare(args.round_dir, args.output_dir, args.methods, args.folds, args.seed, args.encoder,
                     args.revision, args.device, args.setfit_steps, args.setfit_iterations, args.resume, args.cache_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
