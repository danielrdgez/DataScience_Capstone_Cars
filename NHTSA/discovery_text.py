"""Offline TF-IDF and local embeddings for dated, family-held-out discovery."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from MODELS import discovery_artifacts as a
from NHTSA.text_features import CHANNELS, COMPONENTS, model_text

PERIODS = ("historical", "latest_saved")
COMPONENT_NAMES = list(COMPONENTS)
ENCODER_CACHE_NAME = "models--sentence-transformers--all-MiniLM-L6-v2"


def complete_encoder_snapshot(path):
    """Accept a local transformer export as well as a Sentence Transformers export."""
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    tokenizer = any((path / name).is_file() for name in ("tokenizer.json", "vocab.txt", "tokenizer.model", "spiece.model"))
    weights = any((path / name).is_file() and (path / name).stat().st_size > 0
                  for name in ("model.safetensors", "pytorch_model.bin"))
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / index_name).is_file():
            index = a.read_json(path / index_name)
            shards = set(index.get("weight_map", {}).values())
            weights |= bool(shards) and all((path / shard).is_file() for shard in shards)
    return tokenizer and weights


def encoder_candidates(directory):
    directory = Path(directory)
    yield directory
    reference = directory / "refs/main"
    if reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
        if revision and Path(revision).name == revision:
            yield directory / "snapshots" / revision
    # A stale main reference must not hide another complete cached snapshot.
    yield from sorted((directory / "snapshots").glob("*"), reverse=True)


def cached_encoder(path=None):
    if path:
        roots = [Path(path)]
    else:
        roots = []
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            if os.environ.get(name):
                roots.append(Path(os.environ[name]) / ENCODER_CACHE_NAME)
        if os.environ.get("HF_HOME"):
            roots.append(Path(os.environ["HF_HOME"]) / "hub" / ENCODER_CACHE_NAME)
        if os.environ.get("XDG_CACHE_HOME"):
            roots.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface/hub" / ENCODER_CACHE_NAME)
        roots.append(Path.home() / ".cache/huggingface/hub" / ENCODER_CACHE_NAME)
    for root in dict.fromkeys(roots):
        for candidate in encoder_candidates(root):
            if complete_encoder_snapshot(candidate):
                return candidate.resolve()
    inspected = ", ".join(str(root) for root in roots)
    raise ValueError("No complete local encoder found. Required: config, tokenizer and model weights. "
                     "Inspected: " + inspected + ". Supply --embedding-model with a complete local snapshot.")


def encoder_hashes(path):
    return {p.relative_to(path).as_posix(): a.file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()}


def collect_records(data_dir, families, cutoff):
    from MODELS.Price_ML_Models import _complaint_product_matches

    pieces, audit = [], {}
    family_keys = families[a.FAMILY + ["family_id", "family_block", "fold"]].drop_duplicates(a.FAMILY)
    for source, table, identifier, dated in [("complaint", "NHTSA_Complaints", "odiNumber", "dateComplaintFiled"),
                                              ("recall", "NHTSA_Recalls", "NHTSACampaignNumber", "ReportReceivedDate")]:
        raw = pl.read_parquet(Path(data_dir) / (table + ".parquet"))
        if source == "complaint":
            raw = raw.filter(pl.struct("products", "model_year", "make", "model")
                             .map_elements(_complaint_product_matches, return_dtype=pl.Boolean))
            frame = raw.to_pandas().rename(columns={"model_year": "ModelYear", "make": "Make", "model": "Model"})
        else:
            frame = raw.to_pandas()
            query = a.normalize_families(frame.rename(columns={"ModelYear": "returned_year", "Make": "returned_make", "Model": "returned_model",
                                                                "model_year": "ModelYear", "make": "Make", "model": "Model"}))
            returned = a.normalize_families(frame)
            frame = frame.loc[query.family_id.eq(returned.family_id)].copy()
            frame = frame.rename(columns={"Summary": "summary", "Consequence": "consequence", "Remedy": "remedy", "Component": "components"})
        frame = a.normalize_families(frame)
        dates = pd.to_datetime(frame[dated], errors="coerce")
        frame = frame.loc[dates.notna() & dates.le(pd.Timestamp(cutoff)) & frame[identifier].notna()].copy()
        frame["evidence_date"] = dates.loc[frame.index]
        frame["source_id"] = frame[identifier].astype("string").str.replace(r"\.0$", "", regex=True)
        frame = frame.loc[frame.source_id.str.strip().ne("")]
        if frame.empty:
            audit[source] = {"eligible_rows_before_conflicts": 0, "conflicting_ids": 0, "multi_family_ids": 0, "excluded_ids": 0}
            continue
        for field in ("summary", "consequence", "remedy", "components"):
            if field not in frame:
                frame[field] = ""
            frame[field] = frame[field].fillna("").astype(str)
        payload = ["summary", "consequence", "remedy", "components"]
        if source == "complaint":
            payload += [c for c in ("crash", "fire", "numberOfInjuries", "numberOfDeaths") if c in frame]
        canonical = frame[payload].astype("string").fillna("")
        for field in ("summary", "consequence", "remedy", "components"):
            canonical[field] = canonical[field].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        frame["version_hash"] = canonical.agg("|".join, axis=1).map(a.digest)
        conflicts = frame.groupby("source_id").version_hash.nunique()
        bad = set(conflicts[conflicts.gt(1)].index)
        multi = frame.groupby("source_id").family_id.nunique()
        if source == "complaint":
            bad |= set(multi[multi.gt(1)].index)
        audit[source] = {"eligible_rows_before_conflicts": len(frame), "conflicting_ids": int(conflicts.gt(1).sum()),
                         "multi_family_ids": int(multi.gt(1).sum()), "excluded_ids": len(bad)}
        frame = frame.loc[~frame.source_id.isin(bad)].drop_duplicates(["family_id", "source_id"]).copy()
        frame = frame.drop(columns=["family_block", "fold"]).merge(family_keys[["family_id", "family_block", "fold"]], on="family_id", validate="m:1")
        frame["source"] = source
        frame["record_id"] = source + ":" + frame.source_id
        frame["document_id"] = frame.record_id + ":" + frame.version_hash
        pieces.append(frame[a.FAMILY + ["family_id", "family_block", "fold", "source", "source_id", "record_id", "document_id", "evidence_date",
                                        "summary", "consequence", "remedy", "components"]])
    columns = a.FAMILY + ["family_id", "family_block", "fold", "source", "source_id", "record_id", "document_id", "evidence_date",
                         "summary", "consequence", "remedy", "components"]
    return (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)), audit


def family_components(records, families):
    rows = []
    for source in ("complaint", "recall"):
        for family, group in records.loc[records.source.eq(source)].groupby("family_id", sort=True):
            row = {"family_id": family, "source": source, "record_count": group.record_id.nunique(),
                   "usable_text_count": int(group.summary.str.strip().str.len().ge(20).sum()),
                   "collection_complete": False}
            row["supported"] = row["usable_text_count"] >= 20 if source == "complaint" else row["usable_text_count"] >= 1
            for name, pattern in COMPONENTS.items():
                count = int(group.components.str.contains(pattern, case=False, regex=True, na=False).sum())
                row[name + "_count"] = count
                row[name + "_share"] = count / row["record_count"]
            rows.append(row)
    columns = ["family_id", "source", "record_count", "usable_text_count", "collection_complete", "supported",
               *[n + suffix for n in COMPONENT_NAMES for suffix in ("_count", "_share")]]
    return pd.DataFrame(rows, columns=columns)


def encode_documents(model, texts, output, device, batch_size):
    from sentence_transformers import SentenceTransformer
    import torch

    encoder = SentenceTransformer(str(model), local_files_only=True, device=("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    # Local MiniLM transformer-only exports use the library's Transformer + mean-pooling loader.
    # Both layouts use the same token limit and explicit L2 normalization below.
    encoder.max_seq_length = 256
    limit = min(int(encoder.max_seq_length), 256) - 2
    if limit < 16:
        raise ValueError("Unsupported embedding token limit")
    shards = Path(output) / "embedding_shards"
    shards.mkdir(exist_ok=True)
    for begin in range(0, len(texts), 1000):
        target = shards / f"{begin:09d}.npy"
        if target.exists():
            existing = np.load(target, allow_pickle=False)
            if existing.shape != (min(1000, len(texts) - begin), encoder.get_sentence_embedding_dimension()) or not np.isfinite(existing).all():
                raise ValueError("Invalid completed embedding shard: " + str(target))
            continue
        vectors, all_chunks, spans = [], [], []
        for text in texts[begin:begin + 1000]:
            tokens = encoder.tokenizer.encode(text, add_special_tokens=False)
            chunks = [encoder.tokenizer.decode(tokens[i:i + limit], skip_special_tokens=True) for i in range(0, len(tokens), limit)] or [text]
            weights = np.array([max(1, min(limit, len(tokens) - i)) for i in range(0, len(tokens), limit)] or [1])
            spans.append((len(all_chunks), len(all_chunks) + len(chunks), weights))
            all_chunks.extend(chunks)
        values = encoder.encode(all_chunks, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        for first, last, weights in spans:
            # Non-overlapping chunks contribute once to a record, never as separate incidents.
            vector = np.average(values[first:last], axis=0, weights=weights)
            vectors.append(vector / max(np.linalg.norm(vector), 1e-12))
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
        temporary.replace(target)
        print("embeddings", begin + len(vectors), "/", len(texts), flush=True)
    result = np.lib.format.open_memmap(Path(output) / "embeddings.npy", mode="w+", dtype="float32",
                                        shape=(len(texts), encoder.get_sentence_embedding_dimension()))
    for begin in range(0, len(texts), 1000):
        values = np.load(shards / f"{begin:09d}.npy", allow_pickle=False)
        result[begin:begin + len(values)] = values
    result.flush()
    return result


def issue_clusters(vectors, documents, fit_mask, output, smoke=False):
    """Group bootstrap stability is measured on disjoint family anchors."""
    path = Path(output)
    blocks = documents.family_block.to_numpy()
    anchor = fit_mask & documents.family_block.map(lambda x: int(a.digest("anchor|" + x)[:16], 16) % 5 == 0).to_numpy()
    train = fit_mask & ~anchor
    labels = np.full(len(documents), -1, dtype=np.int32)
    rows, selected, best = [], None, -np.inf
    rng = np.random.default_rng(42)
    if train.sum() >= 50 and anchor.sum() >= 20 and len(set(blocks[train])) >= 20:
        weights = pd.Series(blocks[train]).map(pd.Series(blocks[train]).value_counts()).rdiv(1).to_numpy()
        for k in ((2, 3) if smoke else range(2, 9)):
            model = KMeans(k, n_init=5, random_state=42).fit(vectors[train], sample_weight=weights)
            reference = model.predict(vectors[anchor])
            if len(set(reference)) < 2:
                continue
            scores = []
            for repeat in range(2 if smoke else 30):
                unique = np.unique(blocks[train])
                counts = pd.Series(rng.choice(unique, len(unique), replace=True)).value_counts()
                bootstrap_weights = pd.Series(blocks[train]).map(counts).fillna(0).to_numpy() * weights
                changed = KMeans(k, n_init=5, random_state=1001 + repeat).fit(vectors[train], sample_weight=bootstrap_weights)
                scores.append(adjusted_rand_score(reference, changed.predict(vectors[anchor])))
            train_labels = model.predict(vectors[train])
            min_families = min(len(set(blocks[train][train_labels == c])) for c in range(k))
            score = silhouette_score(vectors[anchor], reference, metric="cosine", sample_size=min(1000, anchor.sum()), random_state=42)
            median, lower = float(np.median(scores)), float(np.quantile(scores, .1))
            accepted = median >= .7 and lower >= .6 and min_families >= 5
            rows.append({"k": k, "median_ari": median, "ari_q10": lower, "minimum_families": min_families,
                         "silhouette": float(score), "accepted": accepted})
            if accepted and score > best:
                selected, best = model, score
    pd.DataFrame(rows, columns=["k", "median_ari", "ari_q10", "minimum_families", "silhouette", "accepted"]).to_csv(path / "cluster_checks.csv", index=False)
    if selected is None:
        return labels, {}
    labels = selected.predict(vectors)
    centers = normalize(selected.cluster_centers_)
    distance = 1 - np.sum(vectors * centers[labels], axis=1)
    radii = [float(np.quantile(distance[train & (labels == c)], .95)) for c in range(selected.n_clusters)]
    labels[distance > np.asarray(radii)[labels] + 1e-7] = -1
    joblib.dump({"model": selected, "radius": radii, "training_blocks": sorted(set(blocks[train])),
                 "anchor_blocks": sorted(set(blocks[anchor]))}, path / "clusters.joblib")
    return labels, {"Theme " + str(c): centers[c] for c in range(selected.n_clusters)}


def channel_vectors(records, period_path, channel, encoder, args):
    source, field = CHANNELS[channel]
    assoc = records.loc[records.source.eq(source)].copy()
    assoc["model_text"] = [model_text(row, field) for row in assoc.to_dict("records")]
    assoc = assoc.loc[assoc.model_text.str.len().ge(20)].copy()
    assoc["text_hash"] = assoc.model_text.map(a.digest)
    docs = assoc.sort_values(["document_id", "family_id"]).drop_duplicates("document_id").reset_index(drop=True)
    path = period_path / channel
    path.mkdir(exist_ok=True)
    if docs.empty:
        a.write_json(path / "status.json", {"status": "no_usable_text"})
        return
    if args.smoke:
        docs = docs.head(500)
        assoc = assoc.loc[assoc.document_id.isin(docs.document_id)]
    docs.to_parquet(path / "documents.parquet", index=False)
    assoc[["family_id", "family_block", "fold", "document_id", "record_id", "text_hash"]].drop_duplicates().to_parquet(path / "associations.parquet", index=False)
    embedded = encode_documents(encoder, docs.model_text.tolist(), path, args.device, args.batch_size)
    all_folds = assoc.groupby("document_id").fold.agg(set)
    hash_folds = assoc.groupby("text_hash").fold.agg(set)
    document_index = pd.Series(np.arange(len(docs)), index=docs.document_id)
    for fold in range(5):
        out = path / f"fold_{fold}"
        if (out / "complete.json").exists():
            continue
        out.mkdir(exist_ok=True)
        fit_mask = np.array([fold not in all_folds[doc] and fold not in hash_folds[h]
                             for doc, h in docs[["document_id", "text_hash"]].itertuples(index=False, name=None)])
        fit = docs.loc[fit_mask].drop_duplicates("text_hash")
        fit = fit.assign(order=fit.document_id.map(lambda x: a.digest("42|" + x)))
        fit = fit.sort_values("order").assign(position=lambda f: f.groupby("family_block").cumcount()).sort_values(["position", "order"]).head(1000 if args.smoke else 100_000)
        if len(fit) < 3:
            a.write_json(out / "complete.json", {"status": "insufficient_training_documents", "training_documents": len(fit)})
            continue
        fitted = docs.index.isin(fit.index)
        for representation in ("tfidf", "embeddings"):
            target = out / representation
            target.mkdir(exist_ok=True)
            if representation == "tfidf":
                stopwords = sorted(set(ENGLISH_STOP_WORDS) - {"no", "not", "never", "without"})
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, min_df=2,
                                             sublinear_tf=True, stop_words=stopwords, dtype=np.float32)
                try:
                    training = vectorizer.fit_transform(fit.model_text)
                except ValueError:
                    a.write_json(target / "status.json", {"status": "empty_vocabulary"})
                    continue
                dimensions = min(64, training.shape[0] - 1, training.shape[1] - 1)
                if dimensions < 2:
                    a.write_json(target / "status.json", {"status": "insufficient_rank"})
                    continue
                reducer = TruncatedSVD(dimensions, random_state=42).fit(training)
                sparse = vectorizer.transform(docs.model_text)
                vectors = normalize(reducer.transform(sparse)).astype(np.float32)
                usable = np.asarray(sparse.getnnz(axis=1) > 0) & (np.linalg.norm(vectors, axis=1) > 1e-8)
                terms = vectorizer.get_feature_names_out()
                projection = TruncatedSVD(2, random_state=42).fit(training)
                xy = projection.transform(sparse)
                pd.DataFrame({"entity_id": docs.document_id, "x": xy[:, 0], "y": xy[:, 1]}).loc[usable].to_parquet(target / "points.parquet", index=False)
                basis = projection.components_.T
                order = np.argsort(-np.linalg.norm(basis, axis=1))[:8]
                pd.DataFrame([{"label": terms[i], "x": float(basis[i, 0]), "y": float(basis[i, 1]),
                               "definition": "TF-IDF term loading"} for i in order]).to_parquet(target / "arrows.parquet", index=False)
                a.write_json(target / "projection.json", {"status": "completed", "display_only": True,
                    "explained_variance_ratio": projection.explained_variance_ratio_.tolist()})
                joblib.dump({"vectorizer": vectorizer, "reducer": reducer, "projection": projection}, target / "encoder.joblib")
            else:
                vectors = np.asarray(embedded)
                usable = np.linalg.norm(vectors, axis=1) > 0
            if representation == "tfidf":
                np.save(target / "vectors.npy", vectors, allow_pickle=False)
            else:
                a.write_json(target / "vector_store.json", {"relative_path": "../../embeddings.npy", "normalization": "L2",
                                                           "dimensions": int(vectors.shape[1])})
            labels, prototypes = issue_clusters(vectors, docs, fitted & usable, target, args.smoke)
            labels[~usable] = -1
            if representation == "embeddings":
                a.project(target, vectors, docs.document_id.to_numpy(), fitted & usable, prototypes=prototypes)
            mapping = assoc.copy()
            mapping["row"] = mapping.document_id.map(document_index)
            mapping["usable"] = usable[mapping.row]
            mapping["theme"] = labels[mapping.row]
            from scipy.sparse import csr_matrix
            from sklearn.preprocessing import normalize as l2_normalize
            usable_mapping = mapping.loc[mapping.usable].drop_duplicates(["family_id", "record_id"])
            family_ids = sorted(usable_mapping.family_id.unique())
            family_positions = {value: i for i, value in enumerate(family_ids)}
            if family_ids:
                codes = usable_mapping.family_id.map(family_positions).to_numpy(int)
                aggregation = csr_matrix((np.ones(len(usable_mapping)), (codes, usable_mapping.row.to_numpy(int))), shape=(len(family_ids), len(docs)))
                family_vectors = l2_normalize(aggregation @ vectors).astype(np.float32)
            else:
                family_vectors = np.empty((0, vectors.shape[1]), dtype=np.float32)
            np.save(target / "family_vectors.npy", family_vectors, allow_pickle=False)
            pd.DataFrame({"family_id": family_ids, "vector_row": range(len(family_ids))}).to_parquet(target / "family_vector_index.parquet", index=False)
            summaries, points = [], []
            projected = pd.read_parquet(target / "points.parquet").set_index("entity_id")
            for family, group in mapping.groupby("family_id", sort=True):
                group = group.loc[group.usable].drop_duplicates("record_id")
                if group.empty:
                    continue
                indices = group.row.to_numpy(int)
                count = len(group)
                supported = count >= 20 if source == "complaint" else count >= 1
                row = {"family_id": family, "usable_documents": count, "supported": supported,
                       "rejected_theme_documents": int((group.theme < 0).sum())}
                for label in sorted(set(labels) - {-1}):
                    row["theme_" + str(label) + "_share"] = float(group.theme.eq(label).sum() / count)
                summaries.append(row)
                subset = projected.reindex(group.document_id).dropna()
                if not subset.empty:
                    points.append({"entity_id": family, "x": float(subset.x.mean()), "y": float(subset.y.mean()), "supported": supported})
            pd.DataFrame(summaries, columns=None if summaries else ["family_id", "usable_documents", "supported"]).to_parquet(target / "family_features.parquet", index=False)
            pd.DataFrame(points, columns=["entity_id", "x", "y", "supported"]).to_parquet(target / "family_points.parquet", index=False)
            mapping.rename(columns={"row": "document_row"})[["family_id", "document_id", "record_id", "document_row", "theme", "usable"]].to_parquet(target / "memberships.parquet", index=False)
        a.write_json(out / "complete.json", {"status": "completed", "held_out_fold": fold,
            "training_documents": fit.document_id.tolist(), "training_blocks": sorted(set(fit.family_block)),
            "purged_shared_documents": int((~fit_mask).sum())})


def build(args):
    prepared = a.verify(args.prepared)
    encoder = cached_encoder(args.embedding_model)
    hashes = a.source_hashes(args.data_dir)
    if hashes != prepared["source_hashes"]:
        raise ValueError("Text sources do not match the prepared snapshot")
    configuration = {"source_hashes": hashes, "period": args.period, "latest_cutoff": args.latest_cutoff,
                     "embedding_model": str(encoder), "encoder_hashes": encoder_hashes(encoder),
                     "encoder_layout": "sentence_transformers" if (encoder / "modules.json").is_file() else "transformer_mean_pooling",
                     "device": args.device, "batch_size": args.batch_size, "smoke": args.smoke}
    target = a.start(args.output, "text-vectors", configuration, [args.prepared], args.resume)
    families = pd.read_parquet(Path(args.prepared) / "model/profiles.parquet")
    families = a.normalize_families(families).dropna(subset=a.FAMILY).drop_duplicates("family_id")
    periods = PERIODS if args.period == "both" else (args.period,)
    audits = {}
    for period in periods:
        path = target / period
        path.mkdir(exist_ok=True)
        cutoff = date(2020, 12, 31) if period == "historical" else date.fromisoformat(args.latest_cutoff)
        records, audit = collect_records(args.data_dir, families, cutoff)
        records.sort_values(["family_id", "source", "source_id"]).to_parquet(path / "records.parquet", index=False)
        family_components(records, families).to_parquet(path / "family_components.parquet", index=False)
        components = pd.read_parquet(path / "family_components.parquet")
        scales = {}
        for source_name, group in components.groupby("source"):
            development_ids = set(families.loc[families.fold.ne(0), "family_id"])
            development = group.loc[group.family_id.isin(development_ids)]
            scales[source_name] = {name: max(float(development[name + "_count"].quantile(.99)), 1.) if len(development) else 1.
                                   for name in COMPONENT_NAMES}
        a.write_json(path / "component_scales.json", scales)
        audits[period] = {"cutoff": str(cutoff), "sources": audit, "collection_freshness": "unverified",
                          "maximum_evidence_date": str(records.evidence_date.max()) if len(records) else None}
        for channel in CHANNELS:
            print(period, channel, flush=True)
            channel_vectors(records, path, channel, encoder, args)
    if a.source_hashes(args.data_dir) != hashes or encoder_hashes(encoder) != configuration["encoder_hashes"]:
        raise ValueError("Sources or embedding model changed during vector generation")
    a.finish(target, source_hashes=hashes, source_directory=str(Path(args.data_dir).resolve()),
             periods=list(periods), audits=audits, reduced_budget=args.smoke)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--period", choices=[*PERIODS, "both"], default="both")
    parser.add_argument("--latest-cutoff", default=str(date.today()))
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or date.fromisoformat(args.latest_cutoff) > date.today():
        parser.error("Batch size must be positive and latest cutoff cannot be in the future")
    print(build(args))


if __name__ == "__main__":
    main()
