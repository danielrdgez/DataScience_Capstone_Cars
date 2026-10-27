"""Evidence-backed, overlapping vehicle labels and specification peer experiments.

No network calls or model downloads occur during a run. Source excerpts and exact
scope live in the versioned evidence registry; an audit is not its own gold label.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import adjusted_rand_score, silhouette_score

from MODELS import vehicle_profiles as r
from MODELS import vehicle_segmentation as s

ASSETS = Path(__file__).with_name("taxonomy")
KEYS = r.COHORT
SEMANTIC = ["market_positioning", "performance_character"]
BODY = {"SEDAN": "sedan", "SEDAN/SALOON": "sedan", "HATCHBACK": "hatchback",
        "HATCHBACK/LIFTBACK/NOTCHBACK": "hatchback", "WAGON": "wagon", "COUPE": "coupe",
        "CONVERTIBLE": "convertible", "CONVERTIBLE/CABRIOLET": "convertible",
        "SUV / CROSSOVER": "suv_crossover", "SPORT UTILITY VEHICLE (SUV)/MULTI-PURPOSE VEHICLE (MPV)": "suv_crossover",
        "CROSSOVER UTILITY VEHICLE (CUV)": "suv_crossover", "PICKUP TRUCK": "pickup", "PICKUP": "pickup",
        "MINIVAN": "minivan", "VAN": "van"}
LABEL_COLUMNS = KEYS + ["dimension", "namespace", "label", "status", "mapping_id", "source_url",
    "evidence", "scope", "taxonomy_version", "definition"]
QUEUE_COLUMNS = KEYS + ["dimension", "reason", "mapping_ids"]


def verify_artifacts(directory):
    manifest = r.read_json(Path(directory) / "manifest.json")
    for relative, expected in manifest.get("artifacts", {}).items():
        if r.file_hash(Path(directory) / relative) != expected:
            raise ValueError(f"Changed frozen artifact: {relative}")
    return manifest


def load_taxonomy(definitions=ASSETS / "definitions.v1.json", evidence=ASSETS / "evidence.v1.json"):
    definitions, registry = r.read_json(definitions), r.read_json(evidence)
    if definitions["version"] != registry["taxonomy_version"]:
        raise ValueError("Definition/evidence version mismatch")
    seen = set()
    for entry in registry["sources"]:
        if entry["id"] in seen:
            raise ValueError("Duplicate source id")
        seen.add(entry["id"])
        if not entry["url"].startswith("https://") or not entry.get("retrieved_at"):
            raise ValueError("Evidence requires URL and retrieval date")
        if not entry.get("excerpt") or not entry.get("scope_note"):
            raise ValueError("Evidence requires excerpt and scope")
        if entry["year_start"] > entry["year_end"]:
            raise ValueError("Invalid year boundary")
        for claim in entry["claims"]:
            if claim["label"] not in definitions["dimensions"][claim["dimension"]]:
                raise ValueError("Unknown taxonomy label")
            if claim.get("namespace", "market_terminology") not in definitions["namespaces"]:
                raise ValueError("Unknown classification namespace")
        if not entry.get("all_trims") and not entry.get("trims"):
            raise ValueError("Source must explicitly cover all trims or name exact trims")
    return definitions, registry


def draft_mappings(registry):
    """Expand explicit source claims; never guess identity from free-form text."""
    rows = []
    for entry in registry["sources"]:
        for i, claim in enumerate(entry["claims"]):
            rows.append({**{k: v for k, v in entry.items() if k != "claims"}, **claim,
                "mapping_id": f"{entry['id']}:{i}", "status": entry.get("status", "source_supported"),
                "trims": entry.get("trims", []), "all_trims": entry.get("all_trims", False),
                "namespace": claim.get("namespace", "market_terminology")})
    return pd.DataFrame(rows)


def body_family(label):
    if label in {"sedan", "hatchback", "wagon", "coupe", "convertible"}:
        return "passenger_car"
    if label in {"van", "minivan"}:
        return "van_minivan"
    return label if label in {"suv_crossover", "pickup"} else "unknown"


def assign_labels(cohorts, definitions, mappings):
    if cohorts.duplicated(KEYS).any():
        raise ValueError("Cohorts must be unique")
    rows, queue = [], []
    version = definitions["version"]

    def emit(identity, dimension, label, evidence, url, scope, mapping_id, status="source_supported", namespace="observed"):
        rows.append({**identity, "dimension": dimension, "namespace": namespace, "label": label,
            "status": status, "mapping_id": mapping_id, "source_url": url, "evidence": evidence,
            "scope": scope, "taxonomy_version": version,
            "definition": definitions["dimensions"][dimension][label]})

    for record in cohorts.to_dict("records"):
        identity = {key: record[key] for key in KEYS}
        listing_body = BODY.get(r.canonical(record.get("body_type")))
        decoded_body = BODY.get(r.canonical(record.get("decoded_body")))
        if listing_body and decoded_body and listing_body != decoded_body:
            queue.append({**identity, "dimension": "body_style", "reason": "listing_decoder_disagreement", "mapping_ids": ""})
        elif decoded_body or listing_body:
            label = decoded_body or listing_body
            field = "decoded_body" if decoded_body else "body_type"
            origin = "NHTSA_vPIC.parquet#BodyClass" if decoded_body else "used_cars.parquet#body_type"
            evidence = f"{field} mode={label}; agreement={record.get(field + '_agreement')}; observed VINs={record.get(field + '_n')}"
            emit(identity, "body_style", label, evidence,
                 "snapshot://" + origin, "observed cohort; source hashes in join_receipt.json", "body_mode")
        electric = r.canonical(record.get("electrification")) or ""
        fuel = r.canonical(record.get("fuel_type")) or ""
        secondary = r.canonical(record.get("secondary_fuel")) or ""
        power = None
        if "PLUG-IN" in electric or "PHEV" in electric:
            power = "plug_in_hybrid"
        elif "BEV" in electric or "BATTERY ELECTRIC" in electric:
            power = "battery_electric"
        elif "HEV" in electric or "HYBRID" in electric or "HYBRID" in fuel:
            power = "hybrid"
        elif fuel == "ELECTRIC" and not secondary:
            power = "battery_electric"
        elif fuel in {"GASOLINE", "DIESEL", "FLEX FUEL VEHICLE", "BIODIESEL"} and not secondary:
            # A fuel label alone cannot distinguish an ICE from a hybrid.
            if electric.startswith("NOT ELECTRIFIED"):
                power = "combustion"
        if power:
            emit(identity, "powertrain", power, f"electrification={electric}; primary={fuel}; secondary={secondary}",
                 "snapshot://joined-cohort#powertrain", "observed cohort; source hashes in join_receipt.json", "powertrain_evidence")

        matches = mappings[(mappings.make == identity["Make"]) & (mappings.model == identity["Model"])
                           & (mappings.year_start <= int(identity["ModelYear"])) & (mappings.year_end >= int(identity["ModelYear"]))]
        supported = []
        for mapping in matches.to_dict("records"):
            if not mapping["all_trims"] and identity["Trim"] not in mapping["trims"]:
                continue
            if mapping["status"] not in {"source_supported", "reviewed"}:
                queue.append({**identity, "dimension": mapping["dimension"], "reason": "weak_or_disputed_evidence", "mapping_ids": mapping["mapping_id"]})
                continue
            supported.append(mapping)
        for dimension in {item["dimension"] for item in supported}:
            current = [item for item in supported if item["dimension"] == dimension]
            specific = [item for item in current if item["status"] == "reviewed" and not item["all_trims"]
                        and item["year_start"] == item["year_end"] == int(identity["ModelYear"])]
            current = specific or current
            # Performance descriptions overlap; exclusive market/size disagreements abstain.
            for namespace in {item["namespace"] for item in current}:
                block = [item for item in current if item["namespace"] == namespace]
                if dimension != "performance_character" and len({item["label"] for item in block}) > 1:
                    queue.append({**identity, "dimension": dimension, "reason": "conflicting_sources", "mapping_ids": ";".join(item["mapping_id"] for item in block)})
                    continue
                for item in block:
                    if dimension == "size_segment":
                        family = body_family(decoded_body or listing_body) if not (decoded_body and listing_body and decoded_body != listing_body) else "unknown"
                        expected = "suv_crossover" if item["label"].endswith("_suv") else "pickup" if item["label"].endswith("_pickup") else "passenger_car"
                        if family != expected:
                            queue.append({**identity, "dimension": dimension, "reason": "size_claim_requires_compatible_body", "mapping_ids": item["mapping_id"]})
                            continue
                    emit(identity, dimension, item["label"], item["excerpt"], item["url"], item["scope_note"],
                         item["mapping_id"], item["status"], namespace)
    labels = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    queue = pd.DataFrame(queue, columns=QUEUE_COLUMNS)
    summary = cohorts.copy()
    for dimension in definitions["dimensions"]:
        subset = labels[labels.dimension == dimension]
        grouped = subset.groupby(KEYS).label.agg(lambda x: ";".join(sorted(set(x)))).rename(dimension + "_labels")
        summary = summary.merge(grouped, on=KEYS, how="left", validate="one_to_one")
        summary[dimension + "_status"] = np.where(summary[dimension + "_labels"].notna(), "assigned", "unresolved")
        summary[dimension + "_labels"] = summary[dimension + "_labels"].fillna("")
    summary["body_family"] = summary.body_style_labels.map(body_family)
    summary["taxonomy_version"] = version
    return summary, labels, queue


def parse_torque(value):
    match = re.fullmatch(r"\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(lb[- ]?ft|n[ .\u00b7-]?m)\s*(?:@\s*([\d,]+)\s*RPM)?\s*", str(value), re.I)
    if not match:
        return np.nan
    number = float(match[1].replace(",", ""))
    return number * (1.3558179483 if match[2].lower().startswith("lb") else 1.) if np.isfinite(number) and number > 0 else np.nan


def joined_audit(data_dir, preparation, cohorts, target):
    """Read narrow columns, validate snapshot hashes, aggregate at the original grain."""
    data_dir = Path(data_dir)
    original = preparation["sources"]
    for name in ["used_cars.parquet", "NHTSA_vPIC.parquet"]:
        if r.file_hash(data_dir / name) != original[name]["sha256"]:
            raise ValueError(f"Snapshot differs from preparation: {name}")
    listing_fields = ["vin", "torque", "listed_date"]
    decoder_fields = ["vin", *KEYS, "BodyClass", "Doors", "CurbWeightLB", "ElectrificationLevel", "FuelTypeSecondary"]
    left = pl.scan_parquet(data_dir / "used_cars.parquet").select(listing_fields)
    right = pl.scan_parquet(data_dir / "NHTSA_vPIC.parquet").select(decoder_fields).unique()
    right = right.with_columns([pl.col(k).cast(pl.String).str.strip_chars().str.to_uppercase().str.replace_all(r"\s+", " ") for k in KEYS])
    right = right.with_columns([pl.when(pl.col(k).is_in(["", "UNKNOWN", "NOT APPLICABLE", "N/A", "--"]))
                                .then(None).otherwise(pl.col(k)).alias(k) for k in KEYS])
    conflicts = right.group_by("vin").len().filter(pl.col("len") > 1).select("vin")
    right = right.join(conflicts, on="vin", how="anti")
    joined = left.join(right, on="vin", how="left", validate="1:1").collect(engine="streaming")
    if joined.height != preparation["vehicles"]["retained_listings"]:
        raise ValueError("Lowercase-vin join changed listing count")
    audit = []
    for field in listing_fields[1:] + decoder_fields[5:]:
        audit.append({"field": field, "retained_rows": joined.height, "observed_rows": joined[field].len() - joined[field].null_count(),
            "coverage": 1 - joined[field].null_count() / joined.height})
    pd.DataFrame(audit).to_csv(target / "joined_field_audit.csv", index=False)
    print("Supplemental VIN join verified; aggregating complete cohorts", flush=True)
    frame = joined.drop_nulls(KEYS).to_pandas()
    frame["ModelYear"] = frame.ModelYear.astype(int)
    frame["torque_nm"] = frame.torque.map(parse_torque)
    records = []
    for key, group in frame.groupby(KEYS, sort=True):
        row = dict(zip(KEYS, key))
        for source, field in [("BodyClass", "decoded_body"), ("ElectrificationLevel", "electrification"), ("FuelTypeSecondary", "secondary_fuel")]:
            mode, agreement = r.mode_agreement(group[source].map(r.canonical))
            row[field] = mode if agreement >= .9 else None
            row[field + "_agreement"] = agreement
            row[field + "_n"] = int(group[source].notna().sum())
        for source, field in [("Doors", "doors"), ("CurbWeightLB", "curb_weight_lb"), ("torque_nm", "torque_nm")]:
            values = pd.to_numeric(group[source], errors="coerce")
            values = values[np.isfinite(values) & (values > 0)]
            row[field + "_n"] = len(values)
            row[field] = values.median() if len(values) >= 5 and values.quantile(.75) - values.quantile(.25) <= .2 * values.median() else np.nan
        row["listed_date_min"] = group.listed_date.min()
        row["listed_date_max"] = group.listed_date.max()
        records.append(row)
    extra = pd.DataFrame(records)
    result = cohorts.merge(extra, on=KEYS, how="left", validate="one_to_one")
    if len(result) != len(cohorts) or result.vin_count.sum() != len(frame):
        raise ValueError(f"Supplemental cohort population differs from preparation: {len(frame)} joined complete VINs versus {result.vin_count.sum()} prepared VINs")
    r.save_table(target / "cohorts.parquet", result)
    parsed = [{"field": field, "supported_cohorts": int(result[field].notna().sum()), "all_cohorts": len(result),
               "valid_vin_values": int(result[field + "_n"].sum()), "use": "supplemental_only"}
              for field in ["doors", "curb_weight_lb", "torque_nm"]]
    pd.DataFrame(parsed).to_csv(target / "supplemental_support.csv", index=False)
    r.write_json(target / "join_receipt.json", {"retained_listings": joined.height, "complete_listing_rows": len(frame),
        "cohorts": len(result), "key": "vin", "source_hashes": {k: original[k]["sha256"] for k in ["used_cars.parquet", "NHTSA_vPIC.parquet"]}})
    return result


def source_audit_fingerprint():
    dependencies = [inspect.getsource(joined_audit), inspect.getsource(parse_torque), r.file_hash(r.__file__),
                    np.__version__, pd.__version__, pl.__version__]
    return hashlib.sha256("\n".join(dependencies).encode()).hexdigest()


def stratified_rows(frame, count, strata, seed=42):
    """Round-robin seeded strata so common labels do not consume every slot."""
    if frame.empty:
        return frame.copy()
    shuffled = frame.sample(frac=1, random_state=seed).copy()
    shuffled["_position"] = shuffled.groupby(strata, dropna=False).cumcount()
    return shuffled.sort_values("_position", kind="stable").head(count).drop(columns="_position")


def freeze_review_sets(summary, labels, target):
    semantic = labels[labels.dimension.isin(SEMANTIC)].merge(summary[KEYS + ["body_family", "development"]], on=KEYS, validate="many_to_one")
    # Only evaluation Make/Model blocks enter acceptance auditing.
    automatic = semantic[(semantic.status == "source_supported") & ~semantic.development].drop_duplicates(KEYS + ["dimension", "label"])
    audit = stratified_rows(automatic, 200, ["dimension", "label", "body_family"])
    audit["correct"] = pd.Series(index=audit.index, dtype="boolean")
    audit["reviewer"] = ""
    audit["review_evidence"] = ""
    audit.to_csv(target / "acceptance_audit.csv", index=False)
    rare = semantic[semantic.label.isin(["supercar", "hypercar"])].copy()
    rare["correct"] = pd.Series(index=rare.index, dtype="boolean")
    rare["reviewer"] = ""
    rare.to_csv(target / "all_supercar_hypercar_review.csv", index=False)
    pilot = stratified_rows(summary[summary.development], 100, ["body_family", "market_positioning_labels", "performance_character_labels"])
    pilot.to_csv(target / "pilot_100.csv", index=False)
    queries = stratified_rows(summary[~summary.development], 100, ["body_family", "market_positioning_labels", "performance_character_labels"])
    queries[KEYS].to_csv(target / "peer_queries.csv", index=False)
    status = [{"label": label, "sampled": int((audit.label == label).sum()), "reviewed": 0,
               "precision": None, "status": "pending_independent_review"} for label in sorted(semantic.label.unique())]
    pd.DataFrame(status, columns=["label", "sampled", "reviewed", "precision", "status"]).to_csv(target / "audit_results.csv", index=False)
    return queries[KEYS]


def evaluate_audit(frozen, reviewed):
    frozen, reviewed = pd.read_csv(frozen), pd.read_csv(reviewed)
    identity = KEYS + ["dimension", "label"]
    if reviewed.duplicated(identity).any():
        raise ValueError("Duplicate reviewed audit assignment")
    evidence_columns = identity + ["mapping_id", "source_url", "evidence"]
    if r.digest(frozen[evidence_columns].fillna("").to_dict("records")) != r.digest(reviewed[evidence_columns].fillna("").to_dict("records")):
        raise ValueError("Audit identity, order or evidence changed after freeze")
    verdict = reviewed.correct.astype("string").str.lower().map({"true": True, "false": False, "1": True, "0": False})
    valid = verdict.notna() & reviewed.reviewer.fillna("").str.strip().ne("") & reviewed.review_evidence.fillna("").str.strip().ne("")
    rows = []
    for label, group in reviewed.groupby("label"):
        usable = group.index[valid.loc[group.index]]
        n = len(usable)
        precision = float(verdict.loc[usable].mean()) if n else None
        blocks = (reviewed.loc[usable, "Make"] + "|" + reviewed.loc[usable, "Model"]).to_numpy()
        interval = r.paired_intervals(verdict.loc[usable].astype(float).to_numpy(), blocks, 1000, 42) if n else None
        rows.append({"label": label, "sampled": len(group), "reviewed": n, "precision": precision,
            "make_model_blocks": len(set(blocks)), "lower": interval[0] if interval else None, "upper": interval[1] if interval else None,
            "interval_status": "available" if interval else "insufficient_blocks_for_interval",
            "status": "pending_independent_review" if n < len(group) else "provisional_small_sample" if n < 20 else "passed" if precision >= .9 else "failed"})
    return pd.DataFrame(rows)


def views(summary, labels):
    yield "global", pd.Series(True, index=summary.index)
    accepted = labels[labels.dimension.isin(SEMANTIC)].drop_duplicates(KEYS + ["dimension", "label"])
    for (dimension, label), group in accepted.groupby(["dimension", "label"]):
        members = pd.MultiIndex.from_frame(group[KEYS])
        included = pd.Series(pd.MultiIndex.from_frame(summary[KEYS]).isin(members), index=summary.index)
        for family in ["passenger_car", "suv_crossover", "pickup", "van_minivan"]:
            mask = included & summary.body_family.eq(family)
            if mask.any():
                yield f"{dimension}__{label}__{family}", mask


def subgroup_name(group, parent, fields):
    phrases = []
    for field, lower, upper in [("wheelbase", "shorter wheelbase", "longer wheelbase"), ("horsepower", "lower power", "higher power")]:
        if field not in fields:
            continue
        center = parent[field].median()
        value = group[field].median()
        if np.isfinite(center) and center > 0 and np.isfinite(value) and abs(value / center - 1) >= .1:
            phrases.append(upper if value > center else lower)
    return ", ".join(phrases) or "overlapping specifications"


def cluster_experiments(summary, labels, config, target, legacy):
    data, fields, eligible, audit = s.research_fields(summary)
    audit.to_csv(target / "feature_support.csv", index=False)
    data["eligible"] = eligible
    data["body_family"] = summary.body_family
    experiments, memberships = [], []
    for view, mask in views(summary, labels):
        population = data[mask & eligible].copy()
        development = population[population.development]
        blocks = development[r.FAMILY[1:]].drop_duplicates()
        destination = target / view
        destination.mkdir()
        if len(development) < 50 or len(blocks) < 10 or len(fields) < 2:
            experiments.append({"view": view, "method": "all", "status": "insufficient_support", "development_cohorts": len(development), "blocks": len(blocks)})
            continue
        selected, scores = s.candidates(population, fields, config, destination)
        floor = max(10, int(np.ceil(.05 * len(development))))
        scores["accepted_broad"] &= scores.minimum_count >= floor
        scores["required_minimum_count"] = floor
        scores.to_csv(destination / "model_comparison.csv", index=False)
        for method in ["kmeans", "gower"]:
            table = scores[scores.method == method].copy()
            row = s.choose_candidate(table, "broad")
            if row is None:
                experiments.append({"view": view, "method": method, "status": "no_supported_subdivision", "development_cohorts": len(development), "blocks": len(blocks)})
                continue
            # Existing candidate selection has the same broad policy, except its smaller floor.
            chosen = selected.get(method + "_broad")
            model = chosen[0] if chosen is not None and int(chosen[1]["k"]) == int(row.k) else r.fit_partition(
                r.stratified_sample(development, config.vehicle_cap, config.seed), fields, method, int(row.k), config, config.seed)
            experiment = view + "/" + method
            directory = destination / method
            directory.mkdir()
            membership = population[KEYS + ["development", "body_family"]].copy()
            membership["cluster"] = r.assign_partition(model, population)
            membership["experiment_id"] = experiment
            membership["subgroup_name"] = ""
            profiles, representatives = [], []
            for cluster, group in population.groupby(membership.cluster):
                name = subgroup_name(group, population, fields)
                membership.loc[membership.cluster == cluster, "subgroup_name"] = name
                for field in fields:
                    values = group[field].dropna()
                    profiles.append({"cluster": cluster, "field": field, "observed": len(values), "missing_share": 1-len(values)/len(group),
                        "median": values.median() if field not in r.CATEGORICAL else None,
                        "mode": r.mode_agreement(values)[0] if field in r.CATEGORICAL else None})
                if method == "gower":
                    distance = r.gower(group, model["centers"].iloc[[int(cluster)]], model["metric"]).ravel()
                else:
                    distance = np.linalg.norm(model["encoder"].transform(group[fields].replace({None: np.nan})) - model["model"].cluster_centers_[int(cluster)], axis=1)
                for index in np.argsort(distance)[:3]:
                    representatives.append({**group.iloc[index][KEYS].to_dict(), "cluster": int(cluster), "distance": float(distance[index])})
            pd.DataFrame(profiles).to_csv(directory / "profiles.csv", index=False)
            pd.DataFrame(representatives).to_csv(directory / "representative_cohorts.csv", index=False)
            membership.to_parquet(directory / "membership.parquet", index=False)
            joblib.dump(model, directory / "partition.joblib")
            # One observed trim per year/make/model family for a family-balanced refit.
            balanced = development.sample(frac=1, random_state=config.seed).drop_duplicates(r.FAMILY)
            anchor = development.sample(min(1000, len(development)), random_state=config.seed)
            sensitivity = r.fit_partition(balanced, fields, method, int(row.k), config, config.seed)
            ari = adjusted_rand_score(r.assign_partition(model, anchor), r.assign_partition(sensitivity, anchor))
            joblib.dump(sensitivity, directory / "equal_family_partition.joblib")
            experiments.append({"view": view, "method": method, "status": "accepted_partition", "development_cohorts": len(development),
                "blocks": len(blocks), "k": int(row.k), "median_ari": row.median_ari, "silhouette": row.silhouette,
                "minimum_count": row.minimum_count, "equal_family_ari": ari, "experiment_id": experiment})
            memberships.append(membership)
    results = pd.DataFrame(experiments)
    results.to_csv(target / "experiments.csv", index=False)
    combined = pd.concat(memberships, ignore_index=True) if memberships else pd.DataFrame(columns=KEYS + ["development", "body_family", "cluster", "experiment_id", "subgroup_name"])
    combined.to_parquet(target / "memberships.parquet", index=False)
    global_members = combined[combined.experiment_id.str.startswith("global/")]
    if len(global_members):
        composition = global_members.merge(labels.drop_duplicates(KEYS + ["dimension", "label"]), on=KEYS, validate="many_to_many")
        composition.groupby(["experiment_id", "cluster", "dimension", "label"]).size().rename("cohorts").to_csv(target / "global_category_composition.csv")
        if legacy.is_file():
            old = pd.read_parquet(legacy, columns=KEYS + ["cluster"])
            agreements = []
            for experiment, group in global_members.groupby("experiment_id"):
                paired = group.merge(old, on=KEYS, validate="one_to_one", suffixes=("_new", "_legacy"))
                paired = paired[paired.cluster_legacy >= 0]
                agreements.append({"experiment_id": experiment, "common_cohorts": len(paired), "ari": adjusted_rand_score(paired.cluster_new, paired.cluster_legacy) if len(paired) > 1 else None})
            pd.DataFrame(agreements).to_csv(target / "legacy_agreement.csv", index=False)
    return data, fields, combined


def fit_body_fingerprint(source):
    """Compare fitting statements separately from downstream label composition."""
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cluster_experiments")
    prefix = []
    for node in function.body:
        if isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id == "results" for item in node.targets):
            break
        prefix.append(node)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"subgroup_name", "views"}]
    return hashlib.sha256((ast.dump(ast.Module(body=prefix, type_ignores=[])) + "".join(ast.dump(node) for node in helpers)).encode()).hexdigest()


def reuse_clustering(previous, summary, labels, config, destination):
    """Reuse completed fits only after exact input, view, code and budget checks."""
    previous = Path(previous)
    receipt = verify_artifacts(previous / "clusters")
    if receipt["status"] != "completed" or receipt.get("reduced_budget") != config.smoke:
        raise ValueError("Clustering checkpoint is incomplete or uses another budget")
    for name, key in [("cohorts.parquet", "input_cohorts_sha256"), ("labels.parquet", "input_labels_sha256"), ("execution_vehicle_taxonomy.py", "execution_source_sha256")]:
        if r.file_hash(previous / name) != receipt[key]:
            raise ValueError("Clustering checkpoint input changed")
    if receipt["code_sha256"] != r.file_hash(r.__file__) or receipt["segmentation_code_sha256"] != r.file_hash(s.__file__):
        raise ValueError("Upstream clustering code changed")
    if receipt["configuration"] != json.loads(json.dumps(asdict(config))):
        raise ValueError("Clustering configuration changed")
    current_source = Path(__file__).read_text(encoding="utf-8")
    if fit_body_fingerprint(current_source) != fit_body_fingerprint((previous / "execution_vehicle_taxonomy.py").read_text(encoding="utf-8")):
        raise ValueError("Clustering fitting logic changed")
    old_summary = pd.read_parquet(previous / "cohorts.parquet")
    old_labels = pd.read_parquet(previous / "labels.parquet")
    data, fields, eligible, _ = s.research_fields(summary)
    old_data, old_fields, old_eligible, _ = s.research_fields(old_summary)
    compared = KEYS + ["development", "body_family"] + fields
    if fields != old_fields or not eligible.equals(old_eligible) or not data[compared].equals(old_data[compared]):
        raise ValueError("Clustering specification inputs changed")
    current_views, old_views = dict(views(summary, labels)), dict(views(old_summary, old_labels))
    if set(current_views) != set(old_views) or any(not mask.equals(old_views[name]) for name, mask in current_views.items()):
        raise ValueError("Clustering view memberships changed")
    if set(pd.read_csv(previous / "clusters/experiments.csv")["view"]) != set(current_views):
        raise ValueError("Clustering checkpoint has a different view inventory")
    shutil.copytree(previous / "clusters", destination, dirs_exist_ok=True)
    (destination / "manifest.json").unlink()
    membership = pd.read_parquet(destination / "memberships.parquet")
    global_members = membership[membership.experiment_id.str.startswith("global/")]
    if len(global_members):
        composition = global_members.merge(labels.drop_duplicates(KEYS + ["dimension", "label"]), on=KEYS, validate="many_to_many")
        composition.groupby(["experiment_id", "cluster", "dimension", "label"]).size().rename("cohorts").to_csv(destination / "global_category_composition.csv")
    r.write_json(destination / "reuse_receipt.json", {"source": str(previous.resolve()), "source_manifest_sha256": r.file_hash(previous / "clusters/manifest.json"),
        "input_equivalence": True, "category_view_equivalence": True, "fit_body_fingerprint": fit_body_fingerprint(current_source),
        "execution": "reused_verified_equivalent_clustering", "independent_replication": False,
        "regenerated": ["global_category_composition.csv"]})
    data["eligible"] = eligible
    data["body_family"] = summary.body_family
    return data, fields, membership


def category_profiles(summary, labels, data, fields):
    rows = []
    for view, mask in views(summary, labels):
        if view == "global":
            continue
        group = data[mask]
        for field in [f for f in fields if f not in r.CATEGORICAL]:
            values = group[field].dropna()
            rows.append({"view": view, "field": field, "unit": r.UNITS[field], "cohorts": len(group), "observed": len(values),
                "q25": values.quantile(.25), "median": values.median(), "q75": values.quantile(.75),
                "minimum": values.min(), "maximum": values.max()})
    return pd.DataFrame(rows)


def peer_table(summary, labels, data, fields, queries, memberships):
    development = data[data.development & data.eligible]
    columns = ["query_id", "peer_id", "approach", "view", "rank", "distance", "year_window", "cross_model", "shared_fields", "review_useful", "reviewer"]
    if development.empty or not fields:
        return pd.DataFrame(columns=columns)
    metric = r.fit_gower(development, fields)
    result = []
    keys = [tuple(row) for row in summary[KEYS].to_numpy()]
    index = {key: i for i, key in enumerate(keys)}
    query_indices = [index[tuple(row)] for row in queries.to_numpy()]
    view_masks = dict(views(summary, labels))
    family_codes = pd.factorize(summary.Make + "|" + summary.Model)[0]
    cluster_masks = {}
    for (experiment, cluster), group in memberships.groupby(["experiment_id", "cluster"]):
        members = pd.MultiIndex.from_frame(group[KEYS])
        cluster_masks[(experiment, cluster)] = pd.Series(pd.MultiIndex.from_frame(summary[KEYS]).isin(members), index=summary.index)
    for position, qi in enumerate(query_indices, 1):
        if (position - 1) % 1000 == 0:
            print(f"Peer queries visited: {position - 1}/{len(query_indices)}", flush=True)
        if not data.iloc[qi].eligible:
            continue
        q = data.iloc[[qi]]
        common = data.eligible & summary.body_family.eq(summary.iloc[qi].body_family)
        common &= ~(summary.Make.eq(summary.iloc[qi].Make) & summary.Model.eq(summary.iloc[qi].Model))
        if summary.iloc[qi].body_family == "unknown":
            continue
        # Compute each query/candidate distance once, then reuse it for all views.
        year_difference = (pd.to_numeric(data.ModelYear) - int(q.ModelYear.iloc[0])).abs()
        available = data[common & (year_difference <= 10)]
        if available.empty:
            continue
        distances = pd.Series(r.gower(q, available, metric).ravel(), index=available.index)
        shared_counts = (available[fields].notna() & q[fields].notna().iloc[0]).sum(axis=1)
        for approach in ["body_year_baseline", "global_specs", "category_specs", "global_cluster", "category_cluster"]:
            if approach == "category_specs":
                masks = [(v, common & m) for v, m in view_masks.items() if v != "global" and m.iloc[qi]]
            elif approach.endswith("_cluster"):
                masks = [(experiment, common & mask) for (experiment, cluster), mask in cluster_masks.items()
                         if mask.iloc[qi] and experiment.startswith("global/") == (approach == "global_cluster")]
            else:
                masks = [("global", common)]
            for view, mask in masks:
                candidates = data[mask & (year_difference <= 10)]
                if candidates.empty:
                    continue
                distance = distances.loc[candidates.index].to_numpy()
                shared = shared_counts.loc[candidates.index].to_numpy()
                # Pairwise missingness must not create spuriously close peers.
                usable = shared >= max(2, int(np.ceil(.5 * len(fields))))
                order = np.lexsort((candidates.index.to_numpy(), year_difference.loc[candidates.index].to_numpy(),
                                    distance if approach != "body_year_baseline" else np.zeros(len(candidates))))
                ordered = [i for i in order if usable[i]]
                ordered = np.asarray(ordered, dtype=int)
                candidate_years = year_difference.loc[candidates.index].to_numpy()
                near = ordered[candidate_years[ordered] <= 5]
                distinct_near = np.unique(family_codes[candidates.index.to_numpy()[near]])
                window = 5 if len(distinct_near) >= 5 else 10
                if window == 5:
                    ordered = near
                selected = ordered[~pd.Index(family_codes[candidates.index.to_numpy()[ordered]]).duplicated()][:5]
                for rank, offset in enumerate(selected, 1):
                    pi = int(candidates.index[offset])
                    result.append({"query_id": qi, "peer_id": pi, "approach": approach, "view": view,
                        "rank": rank, "distance": float(distance[offset]), "year_window": window,
                        "cross_model": True,
                        "shared_fields": int(shared[offset]), "review_useful": None, "reviewer": ""})
    return pd.DataFrame(result, columns=columns)


def paired_peer_diagnostics(peers, queries):
    """Match query populations before comparing constrained and global peers."""
    peers = peers[peers.query_id.isin(queries)]
    average = peers.groupby(["query_id", "approach", "view"]).agg(mean_distance=("distance", "mean"), peers=("peer_id", "size")).reset_index()
    rows = []
    for (approach, view), group in average.groupby(["approach", "view"]):
        if approach == "body_year_baseline":
            continue
        baseline = average[average.approach == "body_year_baseline"]
        paired = group.merge(baseline, on="query_id", suffixes=("", "_baseline"), validate="one_to_one")
        rows.append({"approach": approach, "view": view, "paired_queries": len(paired),
            "mean_distance": paired.mean_distance.mean(), "baseline_mean_distance": paired.mean_distance_baseline.mean(),
            "mean_distance_change": (paired.mean_distance - paired.mean_distance_baseline).mean(),
            "interpretation": "specification distance only; independent usefulness review pending"})
    return pd.DataFrame(rows)


def evaluate_peer_review(directory, reviewed_path):
    """Score independently judged peers on paired held-out Make/Model blocks."""
    directory = Path(directory)
    verify_artifacts(directory)
    frozen = pd.read_csv(directory / "peer_review.csv")
    reviewed = pd.read_csv(reviewed_path)
    identity = ["query_id", "peer_id", "approach", "view", "rank"]
    if r.digest(frozen[identity].to_dict("records")) != r.digest(reviewed[identity].to_dict("records")):
        raise ValueError("Peer review differs from frozen sample")
    judged = reviewed.review_useful.astype("string").str.lower().map({"true": 1., "false": 0., "1": 1., "0": 0.})
    reviewed["useful"] = judged.where(reviewed.reviewer.fillna("").str.strip().ne(""))
    # An incomplete five-peer slate must not be compared with a fully judged slate.
    means = reviewed.groupby(["query_id", "approach", "view"]).useful.agg(["mean", "count", "size"]).reset_index()
    means = means[(means["count"] == means["size"]) & (means["size"] == 5)]
    baseline = means[means.approach == "body_year_baseline"][["query_id", "mean"]]
    cohorts = pd.read_parquet(directory / "cohorts.parquet").set_index("cohort_id")
    rows = []
    for (approach, view), group in means.groupby(["approach", "view"]):
        if approach == "body_year_baseline":
            continue
        paired = group.merge(baseline, on="query_id", suffixes=("", "_baseline"), validate="one_to_one")
        if paired.empty:
            continue
        identities = cohorts.loc[paired.query_id]
        if identities.development.any():
            raise ValueError("Peer acceptance sample includes development queries")
        blocks = (identities.Make + "|" + identities.Model).to_numpy()
        change = (paired["mean"] - paired.mean_baseline).to_numpy()
        interval = r.paired_intervals(change, blocks, 1000, 42) if len(set(blocks)) >= 20 else None
        rows.append({"approach": approach, "view": view, "paired_queries": len(paired), "make_model_blocks": len(set(blocks)),
            "usefulness_change": float(change.mean()), "lower": interval[0] if interval else None, "upper": interval[1] if interval else None,
            "status": "evaluated" if interval else "insufficient_blocks_for_interval"})
    return pd.DataFrame(rows, columns=["approach", "view", "paired_queries", "make_model_blocks", "usefulness_change", "lower", "upper", "status"])


def lookup(directory, year, make, model, trim):
    directory = Path(directory)
    verify_artifacts(directory)
    summary = pd.read_parquet(directory / "cohorts.parquet")
    if (directory / "cohort_support.parquet").is_file():
        support = pd.read_parquet(directory / "cohort_support.parquet")
        summary = summary.merge(support, on=KEYS + ["cohort_id"], validate="one_to_one")
    match = summary.ModelYear.eq(int(year))
    for field, value in [("Make", make), ("Model", model), ("Trim", trim)]:
        match &= summary[field].eq(r.canonical(value))
    found = summary[match]
    if len(found) != 1:
        raise KeyError("Select an exact observed year/make/model/trim cohort")
    labels = pd.read_parquet(directory / "labels.parquet")
    key = pd.MultiIndex.from_frame(found[KEYS])
    evidence = labels[pd.MultiIndex.from_frame(labels[KEYS]).isin(key)]
    peers = pd.read_parquet(directory / "peers.parquet")
    peers = peers[peers.query_id == found.index[0]].merge(summary.reset_index(names="peer_id"), on="peer_id", validate="many_to_one")
    return found, evidence, peers


def run(research_run, output, data_dir=r.ROOT / "DATA/CAR_DATA_FINAL", smoke=False, skip_source_audit=False, audit_run=None, reuse_clusters=None):
    execution_code = r.file_hash(__file__)
    audit_code = source_audit_fingerprint()
    source = Path(research_run)
    preparation = verify_artifacts(source / "prepare")
    cohorts = pd.read_parquet(source / "prepare/cohorts.parquet")
    target = r.begin_stage(Path(output).parent, Path(output).name)
    (target / "execution_vehicle_taxonomy.py").write_bytes(Path(__file__).read_bytes())
    definitions, registry = load_taxonomy()
    r.write_json(target / "definitions.json", definitions)
    r.write_json(target / "evidence_registry.json", registry)
    mappings = draft_mappings(registry)
    mappings.to_json(target / "mappings.json", orient="records", indent=2)
    matching = []
    for mapping in mappings.to_dict("records"):
        mask = cohorts.Make.eq(mapping["make"]) & cohorts.Model.eq(mapping["model"]) & cohorts.ModelYear.between(mapping["year_start"], mapping["year_end"])
        if not mapping["all_trims"]:
            mask &= cohorts.Trim.isin(mapping["trims"])
        matching.append({"mapping_id": mapping["mapping_id"], "status": mapping["status"], "observed_matching_cohorts": int(mask.sum())})
    pd.DataFrame(matching).to_csv(target / "mapping_match_audit.csv", index=False)
    if audit_run:
        receipt = verify_artifacts(audit_run)
        if receipt.get("preparation_sha256") != r.file_hash(source / "prepare/manifest.json") or receipt.get("source_audit") != "verified":
            raise ValueError("Supplemental audit belongs to another preparation or was skipped")
        if receipt.get("audit_code_sha256") != audit_code:
            raise ValueError("Audit code changed; repeat source audit")
        supplemental = pd.read_parquet(Path(audit_run) / "cohorts.parquet")
        extra = [c for c in supplemental if c not in cohorts and c not in {"cohort_id", "body_family", "taxonomy_version"} and not c.endswith(("_labels", "_status"))]
        cohorts = cohorts.merge(supplemental[KEYS + extra], on=KEYS, validate="one_to_one", how="left")
        for name in ["join_receipt.json", "joined_field_audit.csv", "supplemental_support.csv"]:
            (target / name).write_bytes((Path(audit_run) / name).read_bytes())
        r.write_json(target / "audit_reuse_receipt.json", {"source": str(Path(audit_run).resolve()), "manifest_sha256": r.file_hash(Path(audit_run) / "manifest.json")})
    elif skip_source_audit:
        r.write_json(target / "join_receipt.json", {"status": "skipped_explicitly", "reason": "uses verified preparation only; supplemental fields unavailable"})
    else:
        audit_target = r.begin_stage(target, "source_audit")
        cohorts = joined_audit(data_dir, preparation, cohorts, audit_target)
        r.finish_stage(audit_target, "completed", source_audit="verified", preparation_sha256=r.file_hash(source / "prepare/manifest.json"),
                       audit_code_sha256=audit_code, taxonomy_code_sha256=execution_code)
        for name in ["join_receipt.json", "joined_field_audit.csv", "supplemental_support.csv"]:
            (target / name).write_bytes((audit_target / name).read_bytes())
    print("Assigning evidence-backed taxonomy labels", flush=True)
    summary, labels, queue = assign_labels(cohorts, definitions, mappings)
    summary["cohort_id"] = np.arange(len(summary))
    r.save_table(target / "cohorts.parquet", summary)
    r.save_table(target / "labels.parquet", labels)
    queue.to_csv(target / "review_queue.csv", index=False)
    queries = freeze_review_sets(summary, labels, target)
    config = replace(r.Config(), stability_refits=2, starts=2, vehicle_cap=200, medoid_cap=100, clusters=(2, 3), smoke=True) if smoke else r.Config()
    r.write_json(target / "execution_configuration.json", {"config": asdict(config), "effective_stability_refits": config.stability_refits if smoke else 30})
    clustering = target / "clusters"
    clustering.mkdir()
    if reuse_clusters:
        data, fields, memberships = reuse_clustering(reuse_clusters, summary, labels, config, clustering)
    else:
        data, fields, memberships = cluster_experiments(summary, labels, config, clustering, source / "vehicles/assignments.parquet")
    r.finish_stage(clustering, "completed", taxonomy_code_sha256=execution_code, configuration=asdict(config), reduced_budget=smoke,
        input_cohorts_sha256=r.file_hash(target / "cohorts.parquet"), input_labels_sha256=r.file_hash(target / "labels.parquet"),
        execution_source_sha256=r.file_hash(target / "execution_vehicle_taxonomy.py"))
    category_profiles(summary, labels, data, fields).to_csv(target / "category_profiles.csv", index=False)
    summary["specification_eligible"] = data.eligible
    summary["specification_status"] = np.where(data.eligible, "eligible", "insufficient_observed_specifications")
    # The checkpoint binds its input before these display-only support columns are added.
    summary[KEYS + ["cohort_id", "specification_eligible", "specification_status"]].to_parquet(target / "cohort_support.parquet", index=False)
    print("Building specification peers for all eligible cohorts", flush=True)
    peers = peer_table(summary, labels, data, fields, summary[KEYS], memberships)
    r.save_table(target / "peers.parquet", peers)
    query_ids = summary.merge(queries, on=KEYS, validate="one_to_one").cohort_id
    review = peers[peers.query_id.isin(query_ids)]
    review.to_csv(target / "peer_review.csv", index=False)
    query_coverage = summary[summary.cohort_id.isin(query_ids)][KEYS + ["cohort_id", "specification_eligible", "body_family"]].copy()
    counts = review.groupby("query_id").size().rename("saved_peer_pairs")
    query_coverage = query_coverage.merge(counts, left_on="cohort_id", right_index=True, how="left", validate="one_to_one")
    query_coverage["saved_peer_pairs"] = query_coverage.saved_peer_pairs.fillna(0).astype(int)
    query_coverage.to_csv(target / "peer_query_coverage.csv", index=False)
    paired_peer_diagnostics(peers, query_ids).to_csv(target / "peer_diagnostics.csv", index=False)
    r.write_json(target / "evaluation.json", {"semantic_accuracy": "pending_independent_review", "peer_usefulness": "pending_independent_review",
        "audit_requested": 200, "audit_sampled": len(pd.read_csv(target / "acceptance_audit.csv")),
        "query_population": "held-out Make/Model blocks", "distance_is_not_usefulness": True})
    for dimension in definitions["dimensions"]:
        labels[labels.dimension == dimension].drop_duplicates(KEYS + ["label"]).groupby("label").size().rename("cohorts").to_csv(target / (dimension + "_coverage.csv"))
    if execution_code != r.file_hash(__file__):
        raise ValueError("Taxonomy code changed during execution; outputs are not a completed run")
    r.finish_stage(target, "completed", taxonomy_version=definitions["version"], taxonomy_code_sha256=execution_code,
        definitions_sha256=r.file_hash(ASSETS / "definitions.v1.json"), evidence_sha256=r.file_hash(ASSETS / "evidence.v1.json"),
        research_run=str(source.resolve()), preparation_sha256=r.file_hash(source / "prepare/manifest.json"), reduced_budget=smoke,
        cohorts=len(summary), label_rows=len(labels), pending_exceptions=len(queue), numerical_fields=fields,
        source_audit="skipped" if skip_source_audit else "verified", review_status="pending_independent_review")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=r.ROOT / "DATA/CAR_DATA_FINAL")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-source-audit", action="store_true")
    parser.add_argument("--audit-run", type=Path, help="Reuse a hash-verified supplemental audit from the same code and preparation")
    parser.add_argument("--reuse-clusters", type=Path, help="Reuse a completed checkpoint after exact fitting-code, input, view and budget equivalence checks")
    args = parser.parse_args()
    if args.skip_source_audit and args.audit_run:
        parser.error("--skip-source-audit and --audit-run are mutually exclusive")
    if args.output.exists():
        parser.error("Output exists; use a fresh directory")
    try:
        print(run(args.research_run, args.output, args.data_dir, args.smoke, args.skip_source_audit, args.audit_run, args.reuse_clusters))
    except (PermissionError, FileNotFoundError, ValueError, KeyError) as error:
        if args.output.is_dir():
            r.finish_stage(args.output, "blocked", error=str(error), taxonomy_code_sha256=r.file_hash(__file__), reduced_budget=args.smoke)
        raise


if __name__ == "__main__":
    main()
