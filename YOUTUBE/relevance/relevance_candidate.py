"""Fit a saved SetFit candidate, score a preserving pilot, or evaluate blind labels.

Run fit/pilot/evaluate --help. Pilot rows are never excluded automatically.
Blind annotations are opened only by evaluate after the candidate is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from .compare_relevance_models import ENCODER, dependency_versions as model_versions, load_training, metrics_for, preflight
    from .partial_relevance import index_rows, verify_source
    from .prepare_make_sentiment_pilot import ROOT, _normalized_words, prefilter_reason
    from .prepare_relevance_round import (FIELDS, NEGATIVE_TYPES, SOURCE_FIELDS, date_value, digest,
                                          file_hash, iter_rows, offer, ordered, read_csv, text_hash,
                                          write_csv, write_json)
except ImportError:
    from compare_relevance_models import ENCODER, dependency_versions as model_versions, load_training, metrics_for, preflight
    from partial_relevance import index_rows, verify_source
    from prepare_make_sentiment_pilot import ROOT, _normalized_words, prefilter_reason
    from prepare_relevance_round import (FIELDS, NEGATIVE_TYPES, SOURCE_FIELDS, date_value, digest,
                                        file_hash, iter_rows, offer, ordered, read_csv, text_hash,
                                        write_csv, write_json)

DEFAULT_ROUND = ROOT / "YOUTUBE/outputs/relevance_round_01_reviewed"
PILOT_FIELDS = (*SOURCE_FIELDS, "candidate_makes", "published_month", "length_band",
                "normalized_text_sha256", "prefilter_reason", "prob_vehicle_relevant",
                "proposed_rejection", "relevance_decision", "input_tokens", "input_truncated",
                "model_version", "candidate_manifest_sha256")
POLICY = {"relevant_recall_target": .98, "rejection_precision_target": .95,
          "minimum_relevant_examples": 100, "minimum_rejections": 50,
          "confidence": .95, "bootstrap_replicates": 1000,
          "unclear_handling": "exclude from binary metrics; retain and audit separately",
          "release_requires": ["independent reviewer reconciliation", "language audit",
                               "semantic duplicate audit", "pilot audit", "unseen-make test"]}


def now():
    return datetime.now(timezone.utc).isoformat()


def dependency_versions():
    result = model_versions()
    for package in ("pyarrow", "psutil"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def supporting_scripts():
    return {name: file_hash(Path(__file__).parent / name) for name in
            ("compare_relevance_models.py", "partial_relevance.py", "prepare_relevance_round.py",
             "prepare_make_sentiment_pilot.py")}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def artifact_hashes(directory):
    return {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
            if p.is_file()}


def probabilities(values, n):
    values = np.asarray(values, dtype=float)
    if values.shape != (n,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Model produced invalid probabilities")
    return values


class LocalSetFit:
    """All supervised fitting and inference use local weights and disable logging."""

    def __init__(self, model):
        self.model = model
        self.max_seq_length = model.model_body.max_seq_length

    @classmethod
    def train(cls, rows, encoder_dir, output, config):
        import torch
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer
        from setfit import SetFitModel, Trainer, TrainingArguments
        from sklearn.linear_model import LogisticRegression
        torch.manual_seed(config["seed"])
        body = SentenceTransformer(str(encoder_dir), device=config["device"], trust_remote_code=False,
                                   local_files_only=True)
        body.max_seq_length = config["max_seq_length"]
        model = SetFitModel(model_body=body, normalize_embeddings=True,
                            model_head=LogisticRegression(class_weight="balanced", max_iter=1000,
                                                          random_state=config["seed"]))
        args = TrainingArguments(output_dir=str(output.parent / "training_runtime"), batch_size=16,
                                 num_epochs=1, num_iterations=config["iterations"], max_steps=config["steps"],
                                 seed=config["seed"], report_to="none", save_strategy="no", eval_strategy="no")
        data = Dataset.from_dict({"text": [r["text"] for r in rows],
                                  "label": [int(r["vehicle_relevant"] == "yes") for r in rows]})
        Trainer(model=model, args=args, train_dataset=data).train()
        model.save_pretrained(str(output))
        return cls(model)

    @classmethod
    def load(cls, directory, device):
        from setfit import SetFitModel
        model = SetFitModel.from_pretrained(str(directory), local_files_only=True,
                                           token=False, trust_remote_code=False)
        model.model_body.to(device)
        return cls(model)

    def score(self, texts, batch_size=32):
        encoded = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        if hasattr(encoded, "detach"):
            encoded = encoded.detach().cpu().numpy()
        classes = list(self.model.model_head.classes_)
        return probabilities(self.model.model_head.predict_proba(encoded)[:, classes.index(1)], len(texts))

    def lengths(self, texts):
        return [len(self.model.model_body.tokenizer(t, truncation=False, verbose=False)["input_ids"])
                for t in texts]


def fit_candidate(directory, output, comparison=None, cutoff=.2, seed=42, device="cpu",
                  backend_class=LocalSetFit):
    if output.exists():
        raise FileExistsError(f"Preserve prior outputs; choose a new candidate directory: {output}")
    if not 0 < cutoff < 1:
        raise ValueError("The rejection cutoff must be strictly between zero and one")
    comparison = comparison or directory / "model_comparison_v1"
    comparison_manifest = read_json(comparison / "run_manifest.json")
    compared = comparison_manifest["config"]
    if compared["train_labels_sha256"] != file_hash(directory / "train_labels.csv"):
        raise ValueError("Training annotations changed since comparison; run a new comparison first")
    if "setfit" not in read_json(comparison / "comparison_metrics.json")["models"]:
        raise ValueError("The SetFit comparison has not completed")
    rows = load_training(directory)  # Never reads blind annotations.
    sample_manifest = read_json(directory / "sample_manifest.json")
    revision = comparison_manifest.get("encoder_revision")
    if not revision:
        raise ValueError("The completed comparison must contain a pinned encoder revision")
    encoder_dir = None
    if backend_class is LocalSetFit:
        preflight(["setfit"])
        from huggingface_hub import snapshot_download
        encoder_dir = snapshot_download(compared.get("encoder", ENCODER), revision=revision, token=False,
                                        cache_dir=directory / "model_cache", local_files_only=True,
                                        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
    config = {"encoder": compared.get("encoder", ENCODER), "encoder_revision": revision,
              "seed": seed, "device": device, "steps": compared["setfit_steps"],
              "iterations": compared["setfit_iterations"], "max_seq_length": 256,
              "normalize_embeddings": True, "class_weight": "balanced", "cutoff": cutoff,
              "automatic_rejection_enabled": False, "evaluation_policy": POLICY}
    output.mkdir(parents=True)
    write_json(output / "fit_status.json", {"state": "training", "created_at_utc": now()})
    write_csv(output / "training_snapshot.csv", (*FIELDS, "csv_row"), rows)
    (output / "protected_train_sample.csv").write_bytes((directory / "train_sample.csv").read_bytes())
    (output / "protected_blind_sample.csv").write_bytes((directory / "blind_sample.csv").read_bytes())
    guide = ROOT / "YOUTUBE/docs/MAKE_SENTIMENT_LABEL_GUIDE.md"
    (output / "label_guide_snapshot.md").write_bytes(guide.read_bytes())
    manifest = {"schema_version": 1, "state": "complete", "created_at_utc": now(),
                "model_version": output.name, "config": config,
                "training_counts": dict(Counter(r["vehicle_relevant"] for r in rows)),
                "training_videos": len({r["video_id"] for r in rows}),
                "round_dir": str(directory.resolve()), "source": sample_manifest.get("source", {}),
                "canonical_makes": sample_manifest.get("vpic", {}).get("makes", []),
                "reservation": {"seed": sample_manifest.get("seed", 42),
                                "fraction": sample_manifest.get("reserve_fraction", .2)},
                "input_hashes": {"train_labels.csv": file_hash(directory / "train_labels.csv"),
                                 "sample_manifest.json": file_hash(directory / "sample_manifest.json"),
                                 "comparison_manifest": file_hash(comparison / "run_manifest.json"),
                                 "label_guide": file_hash(guide)},
                "dependencies": dependency_versions(), "script_sha256": file_hash(Path(__file__)),
                "supporting_scripts_sha256": supporting_scripts(),
                "note": "Uncalibrated review candidate; no automatic exclusion. Blind outcomes were not read."}
    started = time.perf_counter()
    print(f"Fitting SetFit on {len(rows)} labels; encoder revision {revision}", flush=True)
    trained = backend_class.train(rows, encoder_dir, output / "model", config)
    texts = [r["text"] for r in rows[:12]]
    before = probabilities(trained.score(texts), len(texts))
    reloaded = backend_class.load(output / "model", device)
    after = probabilities(reloaded.score(texts), len(texts))
    if reloaded.max_seq_length != config["max_seq_length"]:
        raise ValueError("Reloaded encoder has a different input limit")
    if not np.allclose(before, after, rtol=1e-5, atol=1e-6):
        raise ValueError("Saved-model reload did not reproduce scores")
    lengths = trained.lengths([r["text"] for r in rows])
    manifest.update(fit_seconds=time.perf_counter() - started,
                    rows_exceeding_token_limit=sum(n > config["max_seq_length"] for n in lengths),
                    reload_max_absolute_difference=float(np.max(np.abs(before - after))),
                    model_artifacts_sha256=artifact_hashes(output / "model"),
                    training_snapshot_sha256=file_hash(output / "training_snapshot.csv"),
                    protected_train_sample_sha256=file_hash(output / "protected_train_sample.csv"),
                    protected_blind_sample_sha256=file_hash(output / "protected_blind_sample.csv"))
    write_json(output / "candidate_manifest.json", manifest)
    atomic_json(output / "fit_status.json", {"state": "complete", "finished_at_utc": now()})
    return {"candidate_dir": str(output), "training_counts": manifest["training_counts"],
            "fit_seconds": manifest["fit_seconds"], "cutoff": cutoff, "automatic_rejection_enabled": False}


def candidate_metadata(candidate):
    manifest = read_json(candidate / "candidate_manifest.json")
    if manifest.get("state") != "complete" or manifest["config"]["automatic_rejection_enabled"]:
        raise ValueError("Expected a completed review-only candidate")
    if artifact_hashes(candidate / "model") != manifest["model_artifacts_sha256"]:
        raise ValueError("Saved model artifacts changed")
    if file_hash(candidate / "label_guide_snapshot.md") != manifest["input_hashes"]["label_guide"]:
        raise ValueError("Frozen label guide changed")
    for filename, key in (("training_snapshot.csv", "training_snapshot_sha256"),
                          ("protected_train_sample.csv", "protected_train_sample_sha256"),
                          ("protected_blind_sample.csv", "protected_blind_sample_sha256")):
        if file_hash(candidate / filename) != manifest[key]:
            raise ValueError(f"Candidate source snapshot changed: {filename}")
    return manifest


def relocation_only_hashes():
    """Recognize only the two path edits needed when moving unchanged helpers.

    Reconstruct the recorded source bytes for identity comparison. Any other
    code edit still changes the hash and fails the frozen-candidate guard.
    """
    hashes = {}
    for name in supporting_scripts():
        source = (Path(__file__).parent / name).read_text(encoding="utf-8")
        if name == "prepare_make_sentiment_pilot.py":
            source = source.replace("ROOT = Path(__file__).resolve().parents[2]",
                                    "ROOT = Path(__file__).resolve().parents[1]")
        elif name == "prepare_relevance_round.py":
            source = source.replace("YOUTUBE/relevance/prepare_make_sentiment_pilot.py",
                                    "YOUTUBE/prepare_make_sentiment_pilot.py")
        hashes[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return hashes


def candidate_backend(candidate, manifest, backend_class, device):
    if dependency_versions() != manifest["dependencies"]:
        raise ValueError("Candidate runtime dependency versions changed; use its recorded environment")
    if (supporting_scripts() != manifest["supporting_scripts_sha256"]
            and relocation_only_hashes() != manifest["supporting_scripts_sha256"]):
        raise ValueError("Candidate supporting scripts changed; use its recorded implementation")
    model = backend_class.load(candidate / "model", device)
    if model.max_seq_length != manifest["config"]["max_seq_length"]:
        raise ValueError("Candidate input limit changed")
    return model


def reserved_video(vid, reservation):
    return int(digest(f"{reservation['seed']}|video|{vid}")[:16], 16) / 2**64 < reservation["fraction"]


def pilot_stratum(row):
    date = date_value(row["published_at"])
    length = len(row["text"] or "")
    band = "short" if length < 40 else "medium" if length < 200 else "long"
    return (row["candidate_makes"] or "unmatched", date[:4] if date else "unknown", band)


def sample_pilot(source, candidate, manifest, size, seed):
    blind = read_csv(candidate / "protected_blind_sample.csv")
    train = read_csv(candidate / "protected_train_sample.csv")
    protected_videos = {r["video_id"] for r in train + blind}
    protected_ids = {r["comment_id"] for r in train + blind}
    protected_hashes = {text_hash(r["text"]) for r in train + blind}
    terms = {m: _normalized_words(m) for m in manifest["canonical_makes"]}

    @lru_cache(maxsize=20_000)
    def makes(title):
        title_words = " " + _normalized_words(title or "") + " "
        return " | ".join(m for m, words in terms.items() if " " + words + " " in title_words)

    def eligible_rows():
        for i, row in enumerate(iter_rows(source), 1):
            if i % 500_000 == 0:
                print(f"Pilot sampling scan: {i:,} source rows", flush=True)
            if not row["comment_id"] or not row["video_id"]:
                raise ValueError(f"Missing source ID at row {i}")
            if row["video_id"] in protected_videos or row["comment_id"] in protected_ids or \
                    reserved_video(row["video_id"], manifest["reservation"]):
                continue
            if text_hash(row["text"]) in protected_hashes:
                continue
            yield {**{k: row[k] or "" for k in SOURCE_FIELDS}, "candidate_makes": makes(row["video_title"])}

    counts = Counter(pilot_stratum(row) for row in eligible_rows())
    if not counts:
        raise ValueError("No eligible pilot rows after protecting training and blind videos/text")
    # Allocate evenly across context/year/length strata; distribute remaining slots by capacity.
    quotas = dict.fromkeys(sorted(counts), 0)
    for _ in range(min(size, sum(counts.values()))):
        available = [k for k in quotas if quotas[k] < counts[k]]
        selected = min(available, key=lambda k: (quotas[k], k))
        quotas[selected] += 1
    heaps = {k: [] for k, n in quotas.items() if n}
    for row in eligible_rows():
        stratum = pilot_stratum(row)
        if stratum in heaps:
            offer(heaps[stratum], quotas[stratum], row, seed)
    sampled = [r for heap in heaps.values() for r in ordered(heap)]
    sampled.sort(key=lambda r: digest(f"{seed}|display|{r['comment_id']}"))
    index_rows(sampled)
    return sampled, {"eligible_rows": sum(counts.values()), "strata": len(counts),
                     "sampling": "Equal allocation across tentative title-make/year/length strata, capped by capacity. "
                                 "Enriched sample; not a population prevalence estimate.",
                     "training_and_blind_videos_excluded": len(protected_videos),
                     "all_seed_reserved_videos_excluded": True,
                     "quota_counts": [{"candidate_makes": k[0], "year": k[1], "length_band": k[2],
                                       "eligible": counts[k], "sampled": quotas[k]} for k in quotas]}


def score_pilot(candidate, source, output, size=10_000, seed=43, batch_size=128,
                resume=False, device="cpu", backend_class=LocalSetFit):
    if min(size, batch_size) < 1:
        raise ValueError("Positive sample and batch sizes required")
    if output.exists() and not resume:
        raise FileExistsError(f"Choose a new pilot output directory or use --resume: {output}")
    manifest = candidate_metadata(candidate)
    before = (source.stat().st_size, source.stat().st_mtime_ns)
    source_hash = file_hash(source)
    if source_hash != manifest["source"].get("sha256"):
        raise ValueError("Pilot source differs from the candidate's frozen source snapshot")
    config = {"candidate_manifest_sha256": file_hash(candidate / "candidate_manifest.json"),
              "source_sha256": source_hash, "size": size, "seed": seed, "batch_size": batch_size,
              "device": device, "dependencies": dependency_versions(), "script_sha256": file_hash(Path(__file__)),
              "supporting_scripts_sha256": supporting_scripts()}
    if resume:
        existing = read_json(output / "pilot_manifest.json")
        if config != existing["config"]:
            raise ValueError("Resume refused: candidate, source, code, dependencies, or configuration changed")
        if file_hash(output / "pilot_sample.csv") != existing["sample_sha256"]:
            raise ValueError("Frozen pilot sample changed")
        rows = read_csv(output / "pilot_sample.csv")
    else:
        print("Sampling pilot in two streaming passes; training and blind videos are protected", flush=True)
        rows, sampling = sample_pilot(source, candidate, manifest, size, seed)
        if before != (source.stat().st_size, source.stat().st_mtime_ns):
            raise ValueError("Source changed during sampling")
        output.mkdir(parents=True)
        (output / "chunks").mkdir()
        write_csv(output / "pilot_sample.csv", (*SOURCE_FIELDS, "candidate_makes"), rows)
        existing = {"created_at_utc": now(), "config": config, "sampling": sampling,
                    "rows": len(rows), "sample_sha256": file_hash(output / "pilot_sample.csv")}
        write_json(output / "pilot_manifest.json", existing)
    model = None
    scored, seconds = [], []
    started = time.perf_counter()
    for number, offset in enumerate(range(0, len(rows), batch_size), 1):
        batch = rows[offset:offset + batch_size]
        path = output / "chunks" / f"batch_{number:05d}.json"
        ids = [r["comment_id"] for r in batch]
        if path.exists():
            chunk = read_json(path)
            if chunk["comment_ids"] != ids or chunk.get("config") != config:
                raise ValueError("Pilot checkpoint identities changed")
            scores = probabilities(chunk["probabilities"], len(batch))
        else:
            if model is None:
                model = candidate_backend(candidate, manifest, backend_class, device)
            texts = [r["text"] for r in batch]
            t0 = time.perf_counter()
            scores = probabilities(model.score(texts), len(batch))
            lengths = model.lengths(texts)
            chunk = {"comment_ids": ids, "probabilities": scores.tolist(), "input_tokens": lengths,
                     "seconds": time.perf_counter() - t0, "config": config}
            atomic_json(path, chunk)
        if len(chunk["input_tokens"]) != len(batch) or any(not isinstance(n, int) or n < 0 for n in chunk["input_tokens"]):
            raise ValueError("Invalid checkpoint token lengths")
        if not isinstance(chunk["seconds"], (int, float)) or not math.isfinite(chunk["seconds"]) or chunk["seconds"] < 0:
            raise ValueError("Invalid checkpoint timing")
        seconds.append(chunk["seconds"])
        for row, score, tokens in zip(batch, scores, chunk["input_tokens"]):
            date = date_value(row["published_at"])
            scored.append({**row, "published_month": date[:7] if date else "unknown",
                           "length_band": pilot_stratum(row)[2], "normalized_text_sha256": text_hash(row["text"]),
                           "prefilter_reason": prefilter_reason(row["text"]), "prob_vehicle_relevant": float(score),
                           "proposed_rejection": "yes" if score < manifest["config"]["cutoff"] else "no",
                           "relevance_decision": "review_or_pass", "input_tokens": tokens,
                           "input_truncated": "yes" if tokens > manifest["config"]["max_seq_length"] else "no",
                           "model_version": manifest["model_version"],
                           "candidate_manifest_sha256": config["candidate_manifest_sha256"]})
        print(f"Pilot {min(offset + batch_size, len(rows)):,}/{len(rows):,} rows checkpointed", flush=True)
    if before != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError("Source changed during scoring")
    # Rebuild derived outputs from immutable scored chunks on resume.
    def publish(name, fields, data):
        temporary = output / (name + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            import csv
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(data)
        os.replace(temporary, output / name)
    publish("pilot_scores.csv", PILOT_FIELDS, scored)
    audit = []
    for kind in ("proposed_rejection", "boundary", "retained"):
        group = [r for r in scored if ("proposed_rejection" if r["proposed_rejection"] == "yes" else
                 "boundary" if float(r["prob_vehicle_relevant"]) < .5 else "retained") == kind]
        chosen = sorted(group, key=lambda r: digest(f"{seed}|audit|{r['comment_id']}"))[:100]
        audit.extend({**r, "audit_stratum": kind, "audit_stratum_rows": len(group),
                      "audit_stratum_sampled": len(chosen)} for r in chosen)
    publish("pilot_audit_sample.csv", (*PILOT_FIELDS, "audit_stratum", "audit_stratum_rows", "audit_stratum_sampled"), audit)
    labels = output / "pilot_audit_labels.csv"
    if not labels.exists():
        write_csv(labels, (*PILOT_FIELDS, "audit_stratum", "audit_stratum_rows", "audit_stratum_sampled",
                           "vehicle_relevant", "negative_subtype", "language", "labeler_notes"), audit)
    summary = {"rows": len(scored), "unique_comment_ids": len({r["comment_id"] for r in scored}),
               "proposed_rejections": sum(r["proposed_rejection"] == "yes" for r in scored),
               "actual_exclusions": 0, "automatic_rejection_enabled": False,
               "input_truncated_rows": sum(r["input_truncated"] == "yes" for r in scored),
               "inference_and_tokenization_seconds": sum(seconds),
               "rows_per_second": len(scored) / sum(seconds) if sum(seconds) else None,
               "this_invocation_seconds": time.perf_counter() - started,
               "video_counts": dict(Counter(r["video_id"] for r in scored)),
               "candidate_make_counts": dict(Counter(r["candidate_makes"] or "unmatched" for r in scored)),
               "month_counts": dict(Counter(r["published_month"] for r in scored)),
               "length_counts": dict(Counter(r["length_band"] for r in scored)),
               "note": "All comments retained. Audit samples are enriched; assess strata separately or weight by stratum size. "
                       "Language, semantic duplicates, unseen-make performance, and sentiment are not validated."}
    try:
        import psutil
        summary["process_rss_bytes_at_end"] = psutil.Process().memory_info().rss
    except ImportError:
        summary["process_rss_bytes_at_end"] = None
    atomic_json(output / "pilot_summary.json", summary)
    return {k: summary[k] for k in ("rows", "proposed_rejections", "actual_exclusions", "rows_per_second")}


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0., center - half), min(1., center + half)]


def validation_metrics(rows, scores, cutoff, policy, seed):
    result = metrics_for(rows, scores, cutoff)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["video_id"], []).append(i)
    blocks = list(groups.values())
    rng = np.random.default_rng(seed)
    values = {k: [] for k in ("relevant_recall", "rejection_precision")}
    scores = np.asarray(scores)
    for _ in range(policy["bootstrap_replicates"]):
        indices = [i for b in rng.integers(0, len(blocks), len(blocks)) for i in blocks[b]]
        measured = metrics_for([rows[i] for i in indices], scores[indices], cutoff)
        for key in values:
            if measured[key] is not None:
                values[key].append(measured[key])
    result["video_bootstrap_95_ci"] = {k: np.quantile(v, [.025, .975]).tolist() if v else None for k, v in values.items()}
    result["wilson_95_ci"] = {
        "relevant_recall": wilson(result["true_positive"], result["true_positive"] + result["false_negative"]),
        "rejection_precision": wilson(result["true_negative"], result["true_negative"] + result["false_negative"])}
    recall, precision = result["relevant_recall"], result["rejection_precision"]
    result["point_targets_met"] = (recall is not None and precision is not None and
                                   recall >= policy["relevant_recall_target"] and
                                   precision >= policy["rejection_precision_target"])
    result["support_targets_met"] = (result["true_positive"] + result["false_negative"] >= policy["minimum_relevant_examples"] and
                                     result["true_negative"] + result["false_negative"] >= policy["minimum_rejections"])
    result["interval_targets_met"] = all(
        result[method][key] is not None and result[method][key][0] >= policy[target]
        for method in ("video_bootstrap_95_ci", "wilson_95_ci")
        for key, target in (("relevant_recall", "relevant_recall_target"),
                            ("rejection_precision", "rejection_precision_target")))
    result["statistical_screen_passed"] = all(result[k] for k in ("point_targets_met", "support_targets_met", "interval_targets_met"))
    result["interval_note"] = "Video bootstrap is approximate. Wilson treats rows as independent and guards against " \
                              "zero-error bootstrap degeneracy. These screens are not proof of population guarantees."
    return result


def evaluate_blind(candidate, directory, output, reviewer_unavailable=False, resume=False,
                   device="cpu", backend_class=LocalSetFit):
    manifest = candidate_metadata(candidate)
    if str(directory.resolve()) != manifest["round_dir"]:
        raise ValueError("Evaluate using the candidate's original annotation round directory")
    if file_hash(directory / "sample_manifest.json") != manifest["input_hashes"]["sample_manifest.json"]:
        raise ValueError("Evaluation round differs from the candidate's frozen round")
    load_training(directory)  # Source integrity and separation; still no blind outcomes.
    training = read_csv(directory / "train_labels.csv")
    for r in training:
        if r["vehicle_relevant"] == "no" and r["negative_subtype"] not in NEGATIVE_TYPES:
            raise ValueError("Resolve the missing/unsupported training negative subtypes before blind evaluation")
    current = index_rows(training)
    for old in read_csv(candidate / "training_snapshot.csv"):
        if current[old["comment_id"]]["vehicle_relevant"] != old["vehicle_relevant"]:
            raise ValueError("A fitted binary label changed; freeze a new candidate before evaluating")
    blind_original = read_csv(directory / "blind_sample.csv")
    if file_hash(directory / "blind_sample.csv") != manifest["protected_blind_sample_sha256"]:
        raise ValueError("Blind source changed")
    labels = read_csv(directory / "blind_labels.csv")
    verify_source(blind_original, labels)
    if any(r["vehicle_relevant"] not in {"yes", "no", "unclear"} for r in labels):
        raise ValueError("Finish all blind annotations with yes/no/unclear before evaluation")
    if any((r["vehicle_relevant"] == "no" and r["negative_subtype"] not in NEGATIVE_TYPES) or
           (r["vehicle_relevant"] != "no" and r["negative_subtype"]) for r in labels):
        raise ValueError("Resolve blind negative subtypes before evaluation")
    originals = index_rows(read_csv(directory / "train_sample.csv") + blind_original)
    second = read_csv(directory / "second_reviewer_labels.csv")
    seen = set()
    compared = disagreement = 0
    first_labels = index_rows(training + labels)
    for r in second:
        cid = r["comment_id"]
        if cid in seen or cid not in originals:
            raise ValueError("Unknown/duplicate second-reviewer ID")
        seen.add(cid)
        verify_source([originals[cid]], [r])
        if r["vehicle_relevant"] not in {"", "yes", "no", "unclear"}:
            raise ValueError("Invalid second-reviewer label")
        if r["vehicle_relevant"] and first_labels[cid]["vehicle_relevant"]:
            compared += 1
            disagreement += r["vehicle_relevant"] != first_labels[cid]["vehicle_relevant"]
    reviewer_complete = bool(second) and all(r["vehicle_relevant"] for r in second)
    if not reviewer_complete and not reviewer_unavailable:
        raise ValueError("Complete independent reviewer labels or explicitly use --reviewer-unavailable")
    binary = [r for r in labels if r["vehicle_relevant"] in {"yes", "no"}]
    if not binary:
        raise ValueError("No binary blind labels to evaluate")
    config = {"candidate_manifest_sha256": file_hash(candidate / "candidate_manifest.json"),
              "blind_labels_sha256": file_hash(directory / "blind_labels.csv"),
              "second_reviewer_labels_sha256": file_hash(directory / "second_reviewer_labels.csv"),
              "reviewer_unavailable": reviewer_unavailable, "output_dir": str(output.resolve()),
              "script_sha256": file_hash(Path(__file__)), "dependencies": dependency_versions(), "device": device}
    receipt_path = directory / "blind_evaluation_receipt.json"
    if resume:
        receipt = read_json(receipt_path)
        if receipt["config"] != config or receipt["state"] == "complete":
            raise ValueError("Evaluation resume refused; use the existing completed report or a fresh holdout")
    else:
        if receipt_path.exists():
            raise ValueError("This blind round was already claimed for evaluation; do not tune/retest it")
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        receipt = {"created_at_utc": now(), "state": "evaluating", "config": config}
        write_json(receipt_path, receipt)  # Exclusive creation claims the round, including other model versions.
    model = candidate_backend(candidate, manifest, backend_class, device)
    canonical = index_rows(blind_original)
    rows = [{**canonical[r["comment_id"]], **{k: r[k] for k in FIELDS[-3:]}} for r in labels]
    scores = probabilities(model.score([r["text"] for r in rows]), len(rows))
    selected = [i for i, r in enumerate(rows) if r["vehicle_relevant"] in {"yes", "no"}]
    report = validation_metrics([rows[i] for i in selected], scores[selected], manifest["config"]["cutoff"],
                                manifest["config"]["evaluation_policy"], manifest["config"]["seed"])
    exact_scores = [0. if prefilter_reason(r["text"]) else 1. for r in binary]
    report.update(total_rows=len(rows), unclear_rows=len(rows) - len(selected),
                  unclear_proposed_rejections=sum(r["vehicle_relevant"] == "unclear" and s < manifest["config"]["cutoff"]
                                                  for r, s in zip(rows, scores)),
                  exact_prefilter_metrics=metrics_for(binary, exact_scores),
                  reviewer_complete=reviewer_complete, reviewer_unavailable=reviewer_unavailable,
                  reviewer_compared_rows=compared, independent_reviewer_disagreements=disagreement,
                  automatic_rejection_enabled=False, release_recommended=False,
                  outstanding_audits=manifest["config"]["evaluation_policy"]["release_requires"],
                  config=config, note="Single frozen-candidate test. No threshold sweep. Additional manual audits are required "
                                     "even when statistical screens pass. Prevalence describes eligible reserved-video comments.")
    report["relevant_fraction_among_binary_labels"] = sum(r["vehicle_relevant"] == "yes" for r in binary) / len(binary)
    fields = (*FIELDS, "prob_vehicle_relevant", "proposed_rejection", "relevance_decision")
    predictions = [{**r, "prob_vehicle_relevant": float(s),
                    "proposed_rejection": "yes" if s < manifest["config"]["cutoff"] else "no",
                    "relevance_decision": "review_or_pass"} for r, s in zip(rows, scores)]
    path = output / "blind_predictions.csv"
    if not path.exists():
        write_csv(path, fields, predictions)
    atomic_json(output / "blind_metrics.json", report)
    receipt.update(state="complete", completed_at_utc=now(), report_sha256=file_hash(output / "blind_metrics.json"))
    atomic_json(receipt_path, receipt)
    return {k: report[k] for k in ("n", "relevant_recall", "rejection_precision", "statistical_screen_passed",
                                  "automatic_rejection_enabled", "release_recommended")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="Fit and save the frozen review candidate using cached public weights")
    fit.add_argument("--round-dir", type=Path, default=DEFAULT_ROUND)
    fit.add_argument("--comparison-dir", type=Path)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--cutoff", type=float, default=.2)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", default="cpu")
    pilot = sub.add_parser("pilot", help="Score a protected, enriched pilot; preserve every comment")
    pilot.add_argument("--candidate-dir", type=Path, required=True)
    pilot.add_argument("--source", type=Path, default=ROOT / "DATA/CAR_DATA_FINAL/Youtube_Comments.parquet")
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument("--sample-size", type=int, default=10_000)
    pilot.add_argument("--seed", type=int, default=43)
    pilot.add_argument("--batch-size", type=int, default=128)
    pilot.add_argument("--device", default="cpu")
    pilot.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate", help="Test once on completed blind annotations; records round consumption")
    evaluate.add_argument("--candidate-dir", type=Path, required=True)
    evaluate.add_argument("--round-dir", type=Path, default=DEFAULT_ROUND)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--reviewer-unavailable", action="store_true")
    evaluate.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "fit":
        result = fit_candidate(args.round_dir, args.output_dir, args.comparison_dir, args.cutoff, args.seed, args.device)
    elif args.command == "pilot":
        result = score_pilot(args.candidate_dir, args.source, args.output_dir, args.sample_size, args.seed,
                             args.batch_size, args.resume, args.device)
    else:
        result = evaluate_blind(args.candidate_dir, args.round_dir, args.output_dir, args.reviewer_unavailable,
                                args.resume, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
