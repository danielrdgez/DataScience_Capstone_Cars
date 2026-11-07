"""Build specification discovery and attach separately dated NHTSA evidence.

Stages are explicit. Importing this module never trains, collects or activates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import os
import shutil

import joblib
import numpy as np
import pandas as pd
import polars as pl

from MODELS import discovery_artifacts as a
from MODELS import vehicle_similarity_v3 as v3
from MODELS import vehicle_similarity_v5 as v5
from MODELS import vehicle_profiles_v5 as features
from MODELS import vehicle_taxonomy_v2 as v2
from EDA.DATA_CLEANING import EXCLUDED_LISTING_VINS

NUMERIC = ["seating", "wheelbase", "length", "width", "height", "front_legroom", "back_legroom",
           "horsepower", "torque_nm", "engine_displacement"]
BODY_LABELS = {"sedan": "Sedan", "hatchback": "Hatchback", "suv": "SUV / crossover", "crossover": "SUV / crossover",
               "pickup": "Pickup", "van": "Van", "minivan": "Van", "coupe": "Coupe", "convertible": "Convertible",
               "wagon": "Wagon"}
RATINGS = ["OverallRating", "OverallFrontCrashRating", "OverallSideCrashRating", "RolloverRating", "SidePoleCrashRating"]


def body_category(value):
    text = str(value).lower()
    if text == "passenger_car":
        return "Passenger car (style unresolved)"
    return next((label for token, label in BODY_LABELS.items() if token in text), "Body unresolved")


def prepare(data_dir, output, resume=False):
    hashes = a.source_hashes(data_dir)
    target = a.start(output, "prepare", {"source_hashes": hashes, "excluded_vins": sorted(EXCLUDED_LISTING_VINS),
                    "body_policy": features.BODY_POLICY}, resume=resume)
    frame, conflicts = features.join_sources(Path(data_dir))
    frame = features.reconcile_body(frame)
    frame.select("vin", *features.r.COHORT, "body_type", "BodyClass", "listing_body", "decoder_body",
                 "body_style", "body_family", "body_partition", "body_resolution", "body_selected_source",
                 "body_detail_conflict", "body_family_conflict").write_parquet(target / "body_evidence.parquet")
    body_fields = ["body_type", "BodyClass", "listing_body", "decoder_body", "body_style", "body_family",
                   "body_partition", "body_resolution", "body_selected_source", "body_detail_conflict", "body_family_conflict"]
    frame.group_by(body_fields).agg(pl.len().alias("listings")).sort(
        ["listings", *body_fields], descending=[True, *([False] * len(body_fields))]).write_csv(target / "body_reconciliation.csv")
    audit = []
    for grain in ("cohort", "model"):
        path = target / grain
        path.mkdir(exist_ok=True)
        profiles, membership = features.aggregate(frame, grain, body_partitions=True)
        profiles = a.normalize_families(profiles)
        profiles["category"] = profiles.body_partition.map(body_category)
        profiles["fallback_level"] = "trim" if grain == "cohort" else "model_body_propulsion"
        profiles.to_parquet(path / "profiles.parquet", index=False)
        membership.write_parquet(path / "memberships.parquet")
        pairs = frame.select("vin", (pl.col("listing_horsepower").cast(pl.Float64, strict=False)
                            / pl.col("EngineHP").cast(pl.Float64, strict=False)).alias("ratio"))
        pairs = pairs.filter(pl.col("ratio").is_finite() & (pl.col("ratio") > 0)).join(membership, on="vin", validate="1:1")
        pairs.select("profile_id", "ratio").write_parquet(path / "horsepower_scale_pairs.parquet")
        for reason, subset, count_field in [
                ("all", profiles, "vin_count"),
                ("incomplete_identity", profiles.loc[~profiles.identity_complete], "vin_count"),
                ("body_unresolved", profiles.loc[profiles.category.eq("Body unresolved")], "vin_count"),
                ("body_broad_only", profiles.loc[profiles.body_partition.isin(["passenger_car", "van_minivan"])], "vin_count"),
                ("body_style_conflicting", profiles.loc[profiles.body_detail_conflict_n.gt(0)], "body_detail_conflict_n"),
                ("body_source_conflicting", profiles.loc[profiles.body_family_conflict_n.gt(0)], "body_family_conflict_n"),
                ("body_listing_preferred", profiles.loc[profiles.body_listing_preferred_n.gt(0)], "body_listing_preferred_n"),
                ("body_source_unknown", profiles.loc[profiles.body_partition.eq("unknown")], "vin_count"),
                ("fewer_than_five_vins", profiles.loc[profiles.vin_count.lt(5)], "vin_count")]:
            audit.append({"grain": grain, "reason": reason, "profiles": len(subset), "listings": int(subset[count_field].sum())})
    pd.DataFrame(audit).to_csv(target / "coverage.csv", index=False)
    if a.source_hashes(data_dir) != hashes:
        raise ValueError("Source snapshot changed during preparation")
    a.finish(target, source_directory=str(Path(data_dir).resolve()), source_hashes=hashes,
             conflicting_decode_vins=conflicts, reduced_budget=False)
    return target


def select_fields(data, pairs):
    selected, audit = [], []
    training = data.loc[data.development & data.identity_complete & data.vin_count.ge(5)]
    ratios = pairs.loc[pairs.profile_id.isin(training.profile_id), "ratio"]
    hp_verified = len(ratios) >= 30 and .9 <= ratios.median() <= 1.1
    for field in NUMERIC + v2.CATEGORICAL:
        if field not in data:
            continue
        values = training[field]
        reason = "selected"
        if values.notna().mean() < .7 or values.nunique() < 2:
            reason = "coverage_or_variation"
        if field == "horsepower" and not hp_verified:
            reason = "unverified_horsepower_scale"
        if field in NUMERIC and reason == "selected":
            for previous in selected:
                if previous in NUMERIC and v2.feature_block(field) == v2.feature_block(previous):
                    if abs(training[[previous, field]].corr(method="spearman").iloc[0, 1]) > .9:
                        reason = "redundant_with_" + previous
                        break
        if reason == "selected":
            selected.append(field)
        audit.append({"field": field, "development_coverage": float(values.notna().mean()), "reason": reason})
    return selected, audit


def fit_specs(prepared, output, smoke=False, resume=False):
    receipt = a.verify(prepared)
    config = v3.config_for(smoke)
    target = a.start(output, "fit-specs", {"config": asdict(config)}, [prepared], resume)
    for grain in ("cohort", "model"):
        source = Path(prepared) / grain
        path = target / grain
        path.mkdir(exist_ok=True)
        profiles = pd.read_parquet(source / "profiles.parquet")
        pairs = pd.read_parquet(source / "horsepower_scale_pairs.parquet")
        shutil.copyfile(source / "memberships.parquet", path / "memberships.parquet")
        profiles.to_parquet(path / "profiles.parquet", index=False)
        checks, audits, assignments = [], [], []
        for category in sorted(profiles.category.unique()):
            group = profiles.loc[profiles.category.eq(category)].reset_index(drop=True)
            if category == "Body unresolved":
                continue
            for fold in range(5):
                context = a.digest(category)[:12] + f"-fold-{fold}"
                context_path = path / "contexts" / context
                checkpoint = context_path / "manifest.json"
                if checkpoint.exists():
                    a.verify(context_path)
                    continue
                a.start(context_path, "spec-context", {"category": category, "fold": fold, "config": asdict(config)}, [prepared], resume=context_path.exists())
                data = group.copy()
                data["development"] = data.fold.ne(fold)
                fields, audit = select_fields(data, pairs)
                eligible = v3.eligible(data, fields)
                data["eligible"] = eligible
                training = data.loc[data.development & eligible]
                print(grain, category, "fold", fold, "supported development profiles", len(training), flush=True)
                accepted, experiments, sensitivities = v5.discover(training, fields, config)
                winner = next((m for m in ("gower", "hierarchical", "kmeans") if m in accepted), None)
                metric = v3.fit_metric(training, fields) if len(training) and fields else None
                labels = np.full(len(data), -1, dtype=int)
                if winner:
                    labels[eligible] = v5.assign(accepted[winner], data.loc[eligible])
                descriptions = {}
                for cluster in sorted(set(labels) - {-1}):
                    subset = data.loc[(labels == cluster) & data.development]
                    parts = [category]
                    for field, unit in (("seating", "seats"), ("length", "in long"), ("horsepower", "hp")):
                        if field in fields and subset[field].notna().any():
                            parts.append(f"{subset[field].median():.0f} {unit}")
                    descriptions[int(cluster)] = "; ".join(parts)
                rows = pd.DataFrame({"profile_id": data.profile_id, "eligible": eligible, "cluster": labels})
                rows["description"] = rows.cluster.map(descriptions).fillna("Direct specification peers")
                rows["status"] = np.where(~eligible, "insufficient_observations", np.where(labels < 0, "direct_fallback", "assigned"))
                rows.to_parquet(context_path / "assignments.parquet", index=False)
                joblib.dump({"fields": fields, "metric": metric, "model": accepted.get(winner), "method": winner,
                             "training_families": sorted(training.family_block.unique()), "category": category,
                             "held_out_fold": fold, "descriptions": descriptions}, context_path / "retrieval.joblib")
                pd.DataFrame(experiments).to_csv(context_path / "checks.csv", index=False)
                pd.DataFrame(sensitivities).to_csv(context_path / "sensitivities.csv", index=False)
                pd.DataFrame(audit).to_csv(context_path / "feature_support.csv", index=False)
                numeric = [f for f in fields if f in NUMERIC]
                a.project(context_path / "projection", data[numeric].to_numpy(float), data.profile_id.to_numpy(),
                          (data.development & eligible).to_numpy(), names=numeric, standardize=True)
                a.finish(context_path, category=category, fold=fold, fields=fields, selected_method=winner,
                         training_families=sorted(training.family_block.unique()), reduced_budget=smoke)
        # Context files are the resume unit. Rebuild these compact indexes after all contexts finish.
        for checkpoint in sorted((path / "contexts").glob("*/manifest.json")):
            context_path = checkpoint.parent
            meta = a.read_json(checkpoint)
            rows = pd.read_parquet(context_path / "assignments.parquet")
            rows["context"] = context_path.name
            rows["fold"] = meta["fold"]
            assignments.append(rows)
        pd.concat(assignments, ignore_index=True).to_parquet(path / "assignments.parquet", index=False) if assignments else pd.DataFrame(
            columns=["profile_id", "eligible", "cluster", "description", "status", "context", "fold"]).to_parquet(path / "assignments.parquet", index=False)
    a.finish(target, prepared=str(Path(prepared).resolve()), source_hashes=receipt["source_hashes"],
             source_directory=receipt["source_directory"], reduced_budget=smoke)
    return target


def safety_records(data_dir, profiles):
    raw = pd.read_parquet(Path(data_dir) / "NHTSA_Safety.parquet")
    raw = a.normalize_families(raw)
    id_column = "VehicleId" if "VehicleId" in raw else "vehicle_id"
    if id_column not in raw:
        raise ValueError("Safety snapshot has no vehicle-variant identity")
    raw["vehicle_id"] = raw[id_column].astype("string")
    raw = raw.loc[raw[a.FAMILY].notna().all(axis=1) & raw.vehicle_id.notna()].copy()
    for field in RATINGS:
        if field not in raw:
            raw[field] = np.nan
        raw[field] = pd.to_numeric(raw[field], errors="coerce").where(lambda x: x.between(1, 5))
    conflict = raw.groupby("vehicle_id")[a.FAMILY + RATINGS].nunique(dropna=False).gt(1).any(axis=1)
    raw = raw.loc[~raw.vehicle_id.isin(conflict[conflict].index)].drop_duplicates("vehicle_id")
    raw["rating_era"] = np.where(raw.ModelYear.ge(2011), "2011_onward", "pre_2011")
    description = raw.get("VehicleDescription", raw.get("Description", pd.Series("", index=raw.index))).fillna("").str.upper()
    raw["variant_description"] = description
    raw["variant_body"] = description.map(body_category)
    # Unknown variant descriptions cannot establish body or drivetrain installation.
    raw["variant_drive"] = description.str.extract(r"\b(FWD|RWD|AWD|4WD)\b", expand=False)
    families = profiles.drop_duplicates("family_id")[["family_id", "family_block", "fold"]]
    raw = raw.drop(columns=["family_block", "fold"]).merge(families, on="family_id", validate="m:1")
    return raw


def attach(specs, text, output, resume=False):
    spec_receipt, text_receipt = a.verify(specs), a.verify(text)
    if spec_receipt["source_hashes"] != text_receipt["source_hashes"]:
        raise ValueError("Specification and text snapshots differ")
    target = a.start(output, "attach-evidence", {}, [specs, text], resume)
    profiles = pd.read_parquet(Path(specs) / "model/profiles.parquet")
    safety = safety_records(spec_receipt["source_directory"], profiles)
    safety.to_parquet(target / "safety.parquet", index=False)
    for era, group in safety.groupby("rating_era"):
        # Side/pole/rollover support cross-class display; overall/frontal need a weight context.
        fields = [f for f in ("OverallSideCrashRating", "RolloverRating", "SidePoleCrashRating")
                  if group.loc[group.fold.ne(0), f].notna().mean() >= .7 and group[f].nunique() > 1]
        a.project(target / "safety_projections" / era, group[fields].to_numpy(float), group.vehicle_id.to_numpy(),
                  group.fold.ne(0).to_numpy(), names=fields, standardize=True)
    if a.source_hashes(spec_receipt["source_directory"]) != spec_receipt["source_hashes"]:
        raise ValueError("Source snapshot changed before evidence attachment")
    a.finish(target, specs=str(Path(specs).resolve()), text=str(Path(text).resolve()),
             source_hashes=spec_receipt["source_hashes"], source_directory=spec_receipt["source_directory"],
             periods=text_receipt["periods"], reduced_budget=spec_receipt["reduced_budget"] or text_receipt["reduced_budget"])
    return target


def validate(run, output, resume=False):
    receipt = a.verify(run)
    target = a.start(output, "validate", {}, [run], resume)
    checks = []
    specs, text = Path(receipt["specs"]), Path(receipt["text"])
    for grain in ("cohort", "model"):
        profiles = pd.read_parquet(specs / grain / "profiles.parquet")
        memberships = pd.read_parquet(specs / grain / "memberships.parquet")
        if profiles.profile_id.duplicated().any() or memberships.vin.duplicated().any() or memberships.profile_id.isna().any():
            raise ValueError("Membership identities are not unique")
        if set(memberships.vin) & set(EXCLUDED_LISTING_VINS):
            raise ValueError("Excluded VIN found in discovery population")
        if not set(memberships.profile_id).issubset(set(profiles.profile_id)) or len(memberships) != int(profiles.vin_count.sum()):
            raise ValueError("Membership counts do not reconcile")
        for file in (specs / grain / "contexts").glob("*/retrieval.joblib"):
            model = joblib.load(file)
            query = profiles.loc[profiles.fold.eq(model["held_out_fold"])]
            if set(query.family_block) & set(model["training_families"]):
                raise ValueError("Query family leaked into specification fitting")
        checks.append({"check": grain + " memberships and fold isolation", "passed": True})
    for period in receipt["periods"]:
        records = pd.read_parquet(text / period / "records.parquet")
        cutoff = a.read_json(text / "manifest.json")["audits"][period]["cutoff"]
        if (pd.to_datetime(records.evidence_date) > pd.Timestamp(cutoff)).any():
            raise ValueError("Post-cutoff evidence found")
        for file in (text / period).glob("*/fold_*/complete.json"):
            meta = a.read_json(file)
            docs_path = file.parent.parent / "documents.parquet"
            assoc = pd.read_parquet(file.parent.parent / "associations.parquet")
            held = assoc.loc[assoc.fold.eq(meta.get("held_out_fold", -1))]
            trained = set(meta.get("training_documents", []))
            train_hashes = set(assoc.loc[assoc.document_id.isin(trained), "text_hash"])
            if trained & set(held.document_id) or train_hashes & set(held.text_hash):
                raise ValueError("Shared campaign/narrative leaked into text fitting")
            for representation in file.parent.iterdir():
                if not representation.is_dir() or not (representation / "memberships.parquet").exists():
                    continue
                vector_file = a.vector_path(representation)
                vectors = np.load(vector_file, mmap_mode="r", allow_pickle=False)
                if not np.isfinite(vectors).all():
                    raise ValueError("Nonfinite text vectors")
                if representation.name == "embeddings" and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
                    raise ValueError("Embedding normalization failed")
        checks.append({"check": period + " dates, text isolation and vectors", "passed": True})
    if a.source_hashes(receipt["source_directory"]) != receipt["source_hashes"]:
        raise ValueError("Source snapshot no longer matches discovery bundle")
    a.write_json(target / "validation.json", {"passed": True, "run": a.dependency(run), "checks": checks,
                 "peer_quality": "unmeasured; direct specification retrieval remains default"})
    a.finish(target, passed=True, run=a.dependency(run), reduced_budget=receipt["reduced_budget"])
    return target


def compare_peers(run, output, resume=False):
    """Export controlled candidate lists for review; never promote a learned ranking."""
    receipt = a.verify(run)
    target = a.start(output, "compare-peers", {"query_cap_per_grain": 100}, [run], resume)
    specs, text = Path(receipt["specs"]), Path(receipt["text"])
    rows, coverage = [], []
    vector_cache = {}
    for grain in ("cohort", "model"):
        profiles = pd.read_parquet(specs / grain / "profiles.parquet")
        assignments = pd.read_parquet(specs / grain / "assignments.parquet")
        queries = profiles.loc[profiles.identity_complete & profiles.vin_count.ge(5)].copy()
        queries["query_order"] = queries.profile_id.map(lambda x: a.digest("review|" + x))
        for _, query in queries.sort_values("query_order").head(100).iterrows():
            context = a.digest(query.category)[:12] + f"-fold-{int(query.fold)}"
            path = specs / grain / "contexts" / context
            if not path.exists():
                continue
            model = joblib.load(path / "retrieval.joblib")
            membership = assignments.loc[assignments.context.eq(context)].set_index("profile_id")
            if query.profile_id not in membership.index or not membership.loc[query.profile_id, "eligible"] or not model["metric"]:
                continue
            candidates = profiles.loc[profiles.category.eq(query.category) & profiles.family_block.ne(query.family_block)]
            candidates = candidates.loc[candidates.profile_id.isin(membership.loc[membership.eligible].index)]
            candidates = candidates.loc[(pd.to_numeric(candidates.ModelYear) - int(query.ModelYear)).abs().le(10)].copy()
            if candidates.empty:
                continue
            candidates["distance"] = v3.distances(profiles.loc[profiles.profile_id.eq(query.profile_id)], candidates, model["metric"])[0]
            candidates = candidates.loc[np.isfinite(candidates.distance)]
            scores = {"direct_specs": candidates.distance.to_numpy()}
            query_cluster = membership.loc[query.profile_id, "cluster"]
            labels = membership.reindex(candidates.profile_id).cluster.to_numpy()
            scores["cluster_assisted"] = scores["direct_specs"] + .1 * ((query_cluster >= 0) & (labels >= 0) & (labels != query_cluster))
            for representation in ("tfidf", "embeddings"):
                distances = []
                for channel in ("complaint", "defect", "hazard", "remedy"):
                    vector_dir = text / "historical" / channel / f"fold_{int(query.fold)}" / representation
                    if not (vector_dir / "family_vector_index.parquet").exists():
                        continue
                    cache_key = str(vector_dir)
                    if cache_key not in vector_cache:
                        index = pd.read_parquet(vector_dir / "family_vector_index.parquet").set_index("family_id").vector_row
                        support = pd.read_parquet(vector_dir / "family_features.parquet").set_index("family_id").supported
                        vector_cache[cache_key] = (index, support, np.load(vector_dir / "family_vectors.npy", mmap_mode="r", allow_pickle=False))
                    index, support, vectors = vector_cache[cache_key]
                    if query.family_id not in index or not support.get(query.family_id, False):
                        continue
                    positions = candidates.family_id.map(index)
                    valid = positions.notna() & candidates.family_id.map(support).fillna(False)
                    delta = np.full(len(candidates), np.nan)
                    delta[valid] = np.clip(1 - vectors[positions.loc[valid].to_numpy(int)] @ vectors[int(index[query.family_id])], 0, 2)
                    distances.append(delta)
                if distances:
                    matrix = np.vstack(distances)
                    count = np.isfinite(matrix).sum(axis=0)
                    average = np.divide(np.nansum(matrix, axis=0), count, out=np.full(len(candidates), np.nan), where=count > 0)
                    scores[representation + "_assisted"] = np.where(np.isfinite(average), .8 * candidates.distance + .1 * average, np.nan)
                    scores["direct_specs_matched_" + representation] = np.where(np.isfinite(average), candidates.distance, np.nan)
            # Match populations for each experimental comparison; do not reward absent text.
            for approach, score in scores.items():
                valid = np.isfinite(score)
                local = candidates.loc[valid].copy()
                local["score"] = np.asarray(score)[valid]
                local = local.sort_values(["score", "profile_id"]).drop_duplicates("family_block").head(5)
                for rank, (_, candidate) in enumerate(local.iterrows(), 1):
                    rows.append({"grain": grain, "query_profile_id": query.profile_id, "query_family": query.family_block,
                                 "approach": approach, "rank": rank, "peer_profile_id": candidate.profile_id,
                                 "distance": float(candidate.score), "context": context, "rating": None,
                                 "reviewer": "", "interpretation": "experimental; no buyer usefulness established"})
                coverage.append({"grain": grain, "query_profile_id": query.profile_id, "approach": approach,
                                 "supported_candidates": int(valid.sum()), "returned": len(local)})
    pd.DataFrame(rows).to_json(target / "peer_review.jsonl", orient="records", lines=True)
    pd.DataFrame(coverage).to_csv(target / "coverage.csv", index=False)
    a.finish(target, run=a.dependency(run), reduced_budget=receipt["reduced_budget"], default_approach="direct_specs", quality="unmeasured")
    return target


def activate(run, validation):
    receipt, checked = a.verify(run), a.verify(validation)
    if receipt["reduced_budget"] or not checked.get("passed") or checked["run"] != a.dependency(run):
        raise ValueError("Activation requires a matching full validated bundle")
    config_path = a.ROOT / "car_app/artifact_registry.json"
    lock = config_path.with_suffix(".lock")
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        original = config_path.read_bytes()
        config = a.read_json(config_path)
        # Store rollback outside the immutable validation directory.
        backup = Path(validation).parent / ("registry-before-" + a.file_hash(Path(run) / "manifest.json")[:12] + ".json")
        if not backup.exists():
            backup.write_bytes(original)
        config.update(discovery_run=str(Path(run).resolve().relative_to(a.ROOT / "MODEL_OUTPUT")).replace("\\", "/"),
                      discovery_manifest_sha256=a.file_hash(Path(run) / "manifest.json"),
                      discovery_validation=str(Path(validation).resolve()), recommendation_policy_version=3)
        if config_path.read_bytes() != original:
            raise ValueError("Registry changed concurrently")
        a.write_json(config_path, config)
    finally:
        os.close(handle)
        lock.unlink()
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check-dependencies", "preflight", "prepare", "fit-specs", "attach-evidence", "compare-peers", "validate", "activate"])
    for name in ("data-dir", "prepared", "specs", "text", "run", "validation", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    required = {"check-dependencies": [], "preflight": ["data_dir"], "prepare": ["data_dir", "output"], "fit-specs": ["prepared", "output"], "compare-peers": ["run", "output"],
                "attach-evidence": ["specs", "text", "output"], "validate": ["run", "output"], "activate": ["run", "validation"]}
    for name in required[args.stage]:
        if getattr(args, name) is None:
            parser.error("Missing --" + name.replace("_", "-"))
    if args.stage == "check-dependencies":
        from importlib.util import find_spec
        import sys
        missing = [name for name in ("IPython", "shap") if find_spec(name) is None]
        if missing:
            parser.exit(1, "Missing regression dependencies: " + ", ".join(missing)
                        + "\nInstall them with: " + sys.executable + " -m pip install -r requirements-discovery.txt\n")
        print("Regression dependencies available: IPython, shap")
        return
    if args.stage == "preflight":
        for name in a.TABLES:
            pl.read_parquet_schema(args.data_dir / (name + ".parquet"))
        print({"source_hashes": a.source_hashes(args.data_dir), "source_directory": str(args.data_dir.resolve())})
        return
    if args.stage == "prepare":
        result = prepare(args.data_dir, args.output, args.resume)
    elif args.stage == "fit-specs":
        result = fit_specs(args.prepared, args.output, args.smoke, args.resume)
    elif args.stage == "attach-evidence":
        result = attach(args.specs, args.text, args.output, args.resume)
    elif args.stage == "validate":
        result = validate(args.run, args.output, args.resume)
    elif args.stage == "compare-peers":
        result = compare_peers(args.run, args.output, args.resume)
    else:
        result = activate(args.run, args.validation)
    print(result)


if __name__ == "__main__":
    main()
