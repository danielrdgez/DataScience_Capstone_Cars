"""VIN-supported variant taxonomy. Legacy taxonomy runs and APIs remain readable."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from MODELS import vehicle_profiles as r
from MODELS import vehicle_segmentation as s
from MODELS import vehicle_taxonomy as legacy

VERSION = "2.0.0"
ASSETS = Path(__file__).with_name("taxonomy")
NUMERIC = ["seating", "wheelbase", "length", "height", "front_legroom", "back_legroom", "horsepower", "torque_nm"]
DESCRIPTIVE = ["width", "engine_displacement", "curb_weight_lb", "doors"]
CATEGORICAL = ["fuel_type", "transmission", "wheel_system"]


def stable_id(values):
    return hashlib.sha256(json.dumps(values, default=str, ensure_ascii=True).encode()).hexdigest()[:24]


def canonical_expr(name):
    value = pl.col(name).cast(pl.String).str.strip_chars().str.to_uppercase().str.replace_all(r"\s+", " ")
    return pl.when(value.is_in(["", "UNKNOWN", "NOT APPLICABLE", "N/A", "--"])).then(None).otherwise(value)


def powertrain_expressions(frame):
    """Resolve explicit VIN evidence; combustion fuel alone does not identify ICE."""
    frame = frame.with_columns([canonical_expr(f).alias(f) for f in
        ["fuel_type", "FuelTypePrimary", "FuelTypeSecondary", "ElectrificationLevel"]])
    listing = pl.col("fuel_type").fill_null("")
    primary = pl.col("FuelTypePrimary").fill_null("")
    secondary = pl.col("FuelTypeSecondary").fill_null("")
    electric = pl.col("ElectrificationLevel").fill_null("")
    phev = electric.str.contains("PHEV|PLUG.IN")
    bev = electric.str.contains("BEV|BATTERY ELECTRIC")
    hev = electric.str.contains("HEV|HYBRID") | listing.str.contains("HYBRID")
    listed_bev = listing.eq("ELECTRIC")
    decoded_bev = primary.eq("ELECTRIC") & secondary.eq("")
    combustion = listing.is_in(["GASOLINE", "DIESEL", "FLEX FUEL VEHICLE", "BIODIESEL"]) | primary.is_in(["GASOLINE", "DIESEL"])
    positive_displacement = (pl.col("engine_displacement").fill_null(0) > 0) | (pl.col("DisplacementCC").fill_null(0) > 0)
    conflict = ((bev | listed_bev | decoded_bev) & (combustion | positive_displacement | phev | hev))
    explicit_ice = electric.str.starts_with("NOT ELECTRIFIED") & combustion & ~secondary.str.contains("ELECTRIC")
    power = (pl.when(conflict).then(pl.lit("conflicting"))
        .when(phev).then(pl.lit("plug_in_hybrid"))
        .when(bev | listed_bev | decoded_bev).then(pl.lit("battery_electric"))
        .when(hev).then(pl.lit("hybrid"))
        .when(explicit_ice).then(pl.lit("combustion"))
        .otherwise(pl.lit("unknown")))
    return frame.with_columns(power.alias("powertrain"))


def body_expressions(frame):
    frame = frame.with_columns([canonical_expr(f).replace_strict(legacy.BODY, default=None).alias(alias)
        for f, alias in [("body_type", "listing_body"), ("BodyClass", "decoder_body")]])
    conflict = pl.col("listing_body").is_not_null() & pl.col("decoder_body").is_not_null() & (pl.col("listing_body") != pl.col("decoder_body"))
    return frame.with_columns(pl.when(conflict).then(pl.lit("conflicting"))
        .otherwise(pl.coalesce("decoder_body", "listing_body", pl.lit("unknown"))).alias("body_style"))


def parent_state(group, column):
    counts = group.groupby(column).vin_count.sum()
    total = counts.sum()
    known = counts.drop(index=["unknown", "conflicting"], errors="ignore")
    if counts.get("conflicting", 0):
        return "", "conflicting"
    if len(known) > 1:
        return "", "mixed"
    if len(known) == 1 and known.iloc[0] / total >= .9:
        return str(known.index[0]), "assigned"
    return "", "insufficient_evidence"


def prepare(data_dir, research_run, output, *, excluded_vins=(), schema_version=VERSION):
    """Audit all retained VINs and aggregate actual body/powertrain partitions."""
    data_dir, output = Path(data_dir), Path(output)
    source = Path(research_run) if research_run is not None else None
    receipt = legacy.verify_artifacts(source / "prepare") if source else None
    target = r.begin_stage(output.parent, output.name)
    (target / "execution_prepare_source.py").write_bytes(Path(__file__).read_bytes())
    hashes = {}
    for name in ["used_cars.parquet", "NHTSA_vPIC.parquet"]:
        hashes[name] = r.file_hash(data_dir / name)
        if receipt and hashes[name] != receipt["sources"][name]["sha256"]:
            raise ValueError("Raw snapshot differs from preserved preparation")
    listing_fields = ["vin", "make_name", "model_name", "trim_name", "year", "body_type", "fuel_type",
        "engine_displacement", "maximum_seating", "wheelbase", "length", "height", "width", "front_legroom",
        "back_legroom", "horsepower", "torque", "transmission", "wheel_system", "price", "mileage", "listed_date"]
    decoded_fields = ["vin", *r.COHORT, "BodyClass", "FuelTypePrimary", "FuelTypeSecondary", "ElectrificationLevel",
        "DisplacementCC", "EngineHP", "CurbWeightLB", "Doors"]
    left = pl.scan_parquet(data_dir / "used_cars.parquet").select(listing_fields)
    right = pl.scan_parquet(data_dir / "NHTSA_vPIC.parquet").select(decoded_fields)
    if left.select(pl.col("vin").n_unique()).collect().item() != left.select(pl.len()).collect().item():
        raise ValueError("Duplicate listing VIN")
    left = left.filter(~pl.col("vin").is_in(sorted(excluded_vins)))
    retained = left.select(pl.len()).collect().item()
    right = right.with_columns([canonical_expr(f).alias(f) for f in r.COHORT])
    right = right.with_columns(pl.col("ModelYear").cast(pl.Int64, strict=False))
    right = right.unique()
    conflicts = right.group_by("vin").len().filter(pl.col("len") > 1).select("vin")
    right = right.join(conflicts, on="vin", how="anti")
    frame = left.join(right, on="vin", how="left", validate="1:1").collect(engine="streaming")
    if frame.height != (receipt["vehicles"]["retained_listings"] if receipt and not excluded_vins else retained):
        raise ValueError("VIN join changed retained population")
    print("Joined retained population:", frame.height, flush=True)
    frame = frame.with_columns([pl.col(f).cast(pl.Float64, strict=False).alias(f) for f in
        ["engine_displacement", "DisplacementCC", "EngineHP", "CurbWeightLB", "Doors"]])
    frame = body_expressions(powertrain_expressions(frame))
    frame = frame.with_columns(pl.all_horizontal([pl.col(f).is_not_null() for f in r.COHORT]).alias("identity_complete"))
    identity_fields = ["year", "make_name", "model_name", "trim_name", *r.COHORT]
    listing_make = canonical_expr("make_name")
    listing_model = canonical_expr("model_name")
    frame = frame.with_columns([listing_make.alias("listing_make_normalized"), listing_model.alias("listing_model_normalized")])
    same_make = pl.col("listing_make_normalized").eq(pl.col("Make")).fill_null(False)
    same_model = pl.col("listing_model_normalized").eq(pl.col("Model")).fill_null(False)
    frame = frame.with_columns([
        pl.when(~pl.col("identity_complete")).then(pl.lit("incomplete_decoded_identity"))
          .when(same_make & same_model).then(pl.lit("exact_after_format_normalization"))
          .when(same_make).then(pl.lit("model_difference_requires_evidence"))
          .otherwise(pl.lit("make_difference_requires_evidence")).alias("identity_relation"),
        (pl.col("year").cast(pl.Int64,strict=False) != pl.col("ModelYear")).fill_null(False).alias("year_disagreement")])
    frame.group_by(identity_fields + ["identity_relation", "year_disagreement"]).agg(pl.len().alias("vin_count")).write_parquet(target / "identity_comparison.parquet")
    identity_summary = frame.group_by(["ModelYear", "Make", "Model", "listing_make_normalized", "listing_model_normalized", "identity_relation", "year_disagreement"]).agg(pl.len().alias("vin_count"))
    identity_summary.write_parquet(target / "identity_names.parquet")
    torque_values = frame.select("torque").unique().to_series().to_list()
    torque_lookup = {value: legacy.parse_torque(value) for value in torque_values if value is not None}
    frame = frame.with_columns([
        pl.col("torque").replace_strict(torque_lookup, default=None, return_dtype=pl.Float64).alias("torque_nm"),
        pl.col("maximum_seating").cast(pl.String).str.extract(r"^(\d+)\s+seats?$", 1).cast(pl.Float64, strict=False).alias("seating"),
        pl.col("CurbWeightLB").alias("curb_weight_lb"), pl.col("Doors").alias("doors")])
    # All profiles, including incomplete-identity residuals, retain one VIN membership.
    groups = [*r.COHORT, "identity_complete", "body_style", "powertrain"]
    aggregate = [pl.len().alias("vin_count")]
    for field in NUMERIC + DESCRIPTIVE:
        values = pl.col(field).cast(pl.Float64, strict=False)
        values = pl.when(values.is_finite() & (values > 0)).then(values).otherwise(None)
        aggregate += [values.median().alias(field), values.count().alias(field + "_n"),
            (values.quantile(.75, interpolation="linear") - values.quantile(.25, interpolation="linear")).alias(field + "_iqr")]
    for field in CATEGORICAL:
        value = canonical_expr(field)
        aggregate += [value.drop_nulls().mode().sort().first().alias(field), value.count().alias(field + "_n"),
            value.drop_nulls().value_counts(sort=True).struct.field("count").first().alias(field + "_mode_n")]
    aggregate += [pl.col("listed_date").min().alias("listed_date_min"), pl.col("listed_date").max().alias("listed_date_max"),
        pl.col("price").median().alias("price_median"), pl.col("mileage").median().alias("mileage_median")]
    profiles = frame.group_by(groups).agg(aggregate).sort(groups).to_pandas()
    profiles["ModelYear"] = pd.to_numeric(profiles.ModelYear).astype("Int64")
    profiles["profile_id"] = [stable_id([None if pd.isna(v) else v for v in row]) for row in profiles[groups].to_numpy()]
    profiles["parent_id"] = [stable_id([int(row[0]), *row[1:]]) if complete else None for row, complete in zip(profiles[r.COHORT].to_numpy(), profiles.identity_complete)]
    for field in NUMERIC + DESCRIPTIVE:
        valid = profiles[field + "_n"].ge(5) & profiles[field + "_iqr"].le(.2 * profiles[field])
        profiles[field + "_state"] = np.where(valid, "observed", np.where(profiles[field + "_n"].eq(0), "unknown", "insufficient_or_heterogeneous"))
        profiles.loc[~valid, field] = np.nan
    profiles.loc[profiles.powertrain.eq("battery_electric"), "engine_displacement"] = np.nan
    profiles.loc[profiles.powertrain.eq("battery_electric"), "engine_displacement_state"] = "not_applicable"
    for field in CATEGORICAL:
        profiles[field + "_agreement"] = profiles[field + "_mode_n"] / profiles[field + "_n"].replace(0, np.nan)
        profiles.loc[profiles[field + "_agreement"].lt(.9), field] = None
    profiles["body_family"] = profiles.body_style.map(legacy.body_family)
    profiles.to_parquet(target / "variant_profiles.parquet", index=False)
    keys = pl.from_pandas(profiles[groups + ["profile_id", "parent_id"]])
    membership = frame.join(keys, on=groups, how="left", nulls_equal=True, validate="m:1").select(
        ["vin", "profile_id", "parent_id", "identity_complete", "body_style", "powertrain", "fuel_type", "FuelTypePrimary", "FuelTypeSecondary", "ElectrificationLevel", "horsepower", "EngineHP", "engine_displacement", "DisplacementCC"])
    if membership.height != frame.height or membership["profile_id"].null_count() or membership["vin"].n_unique() != frame.height:
        raise ValueError("VIN/profile partition lost or duplicated population")
    membership.write_parquet(target / "vin_profile_membership.parquet")
    parents = (pd.read_parquet(source / "prepare/cohorts.parquet") if source else
               profiles[profiles.identity_complete].groupby(r.COHORT, dropna=False).vin_count.sum().reset_index())
    parents["parent_id"] = pd.Series([stable_id(list(row)) for row in parents[r.COHORT].to_numpy()], index=parents.index, dtype=object)
    if profiles.loc[profiles.identity_complete, "vin_count"].sum() != parents.vin_count.sum():
        raise ValueError("Complete profile population differs from legacy cohorts")
    states = []
    for parent, group in profiles[profiles.identity_complete].groupby("parent_id"):
        row = {"parent_id": parent}
        for field in ["body_style", "powertrain"]:
            row[field + "_labels"], row[field + "_status"] = parent_state(group, field)
        row["variant_count"] = len(group)
        states.append(row)
    state_columns = ["parent_id", "body_style_labels", "body_style_status", "powertrain_labels", "powertrain_status", "variant_count"]
    parents = parents.merge(pd.DataFrame(states, columns=state_columns), on="parent_id", validate="1:1")
    if parents.variant_count.isna().any():
        raise ValueError("Variant profiles do not link to every complete parent cohort")
    parents.to_parquet(target / "cohorts.parquet", index=False)
    frame.group_by(["identity_complete", "body_style", "powertrain"]).agg(pl.len().alias("vin_count")).write_csv(target / "vin_evidence_audit.csv")
    r.write_json(target / "join_receipt.json", {"retained_listings": frame.height, "complete_listing_rows": int(parents.vin_count.sum()),
        "complete_cohorts": len(parents), "profiles": len(profiles), "key": "vin", "source_hashes": hashes})
    r.finish_stage(target, "prepared", schema_version=schema_version, reduced_budget=False, source_hashes=hashes,
        source_directory=str(data_dir.resolve()), excluded_vins=sorted(excluded_vins),
        preparation_sha256=r.file_hash(source / "prepare/manifest.json") if source else None,
        conflicting_decode_vins=conflicts.select(pl.len()).collect().item(), v2_code_sha256=r.file_hash(__file__))
    return target


def resolve_families(profiles, names, crosswalk):
    result = profiles.copy()
    result["market_family"] = result.Model.fillna("UNRESOLVED")
    result["family_resolved"] = False
    result["family_source"] = "decoded_identity_fallback"
    for row in crosswalk:
        mask = result.Make.eq(row["make"]) & result.Model.isin(row["models"]) & pd.to_numeric(result.ModelYear).between(row["year_start"], row["year_end"])
        if row.get("trims"):
            mask &= result.Trim.isin(row["trims"])
        if row.get("body_styles"):
            mask &= result.body_style.isin(row["body_styles"])
        conflict = mask & result.family_resolved & result.market_family.ne(row["market_family"])
        if conflict.any():
            raise ValueError("Conflicting sourced identity crosswalk")
        result.loc[mask, ["market_family", "family_resolved", "family_source"]] = [row["market_family"], True, row["id"]]
    result["family_block"] = result.Make.fillna("UNRESOLVED") + "|" + result.market_family
    # Connect aliases across years, even when a year-specific mapping is unresolved.
    links = {}
    for row in crosswalk:
        for model in row["models"]:
            links.setdefault((row["make"], model), set()).add(row["market_family"])
    for (make, model), families in links.items():
        if len(families) == 1:
            result.loc[result.Make.eq(make) & result.Model.eq(model), "family_block"] = make + "|" + next(iter(families))
    result["development"] = result.family_block.map(lambda key: r.is_development(*key.split("|", 1), 42))
    return result


def feature_block(field):
    return "powertrain" if field in ["horsepower", "torque_nm", "engine_displacement", *CATEGORICAL] else "size"


def select_fields(profiles, excluded_fields=()):
    data = profiles.copy()
    development = data[data.development & data.identity_complete & data.vin_count.ge(5)]
    fields, records = [], []
    for field in NUMERIC + CATEGORICAL:
        data[field] = data[field].where(data[field + "_n"].ge(5))
        values = data.loc[development.index, field]
        coverage = values.notna().mean()
        reason = "selected" if coverage >= .7 and values.nunique() > 1 else "coverage_or_variation"
        if field in excluded_fields:
            reason = "unverified_source_scale"
        if reason == "selected" and field in NUMERIC:
            for prior in fields:
                if prior in NUMERIC and feature_block(field) == feature_block(prior):
                    if abs(data.loc[development.index, [prior, field]].corr(method="spearman").iloc[0, 1]) > .9:
                        reason = "redundant_with_" + prior
                        break
        if reason == "selected":
            fields.append(field)
        records.append({"field": field, "block": feature_block(field), "development_coverage": coverage, "reason": reason})
    eligible = data.identity_complete & data.vin_count.ge(5) & ~data.powertrain.eq("conflicting") & ~data.body_style.isin(["unknown", "conflicting"])
    for block, minimum in [("size", 2), ("powertrain", 1)]:
        selected = [f for f in fields if feature_block(f) == block]
        eligible &= data[selected].notna().sum(axis=1).ge(max(minimum, int(np.ceil(.5 * len(selected))))) if selected else False
    data["eligible"] = eligible
    return data, fields, pd.DataFrame(records)


def metric_for(frame, fields):
    metric = r.fit_gower(frame, fields, equal_blocks=False)
    metric["weights"] = np.array([1 / sum(feature_block(x) == feature_block(f) for x in fields) for f in fields])
    return metric


def encoder_for(frame, fields):
    encoder = r.fit_encoding(frame, fields, equal_blocks=False)
    encoder.set_params(transformer_weights={f: np.sqrt(1 / sum(feature_block(x) == feature_block(f) for x in fields) / (2 if f in CATEGORICAL else 1)) for f in fields})
    return encoder


def fit_partition(frame, fields, method, config, k, seed):
    if method == "kmeans":
        encoder = encoder_for(frame, fields)
        model = KMeans(k, n_init=config.starts, random_state=seed).fit(encoder.transform(frame[fields].replace({None: np.nan})))
        return {"method": method, "encoder": encoder, "model": model, "fields": fields}
    sample = frame.sample(min(config.medoid_cap, len(frame)), random_state=seed).reset_index(drop=True)
    metric = metric_for(frame, fields)
    distance = r.gower(sample, sample, metric)
    best = None
    for start in range(config.starts):
        medoids = np.random.default_rng(seed + start).choice(len(sample), k, replace=False)
        for _ in range(50):
            assignment = distance[:, medoids].argmin(axis=1)
            updated = medoids.copy()
            for cluster in range(k):
                members = np.flatnonzero(assignment == cluster)
                if len(members):
                    updated[cluster] = members[distance[np.ix_(members, members)].sum(axis=1).argmin()]
            if np.array_equal(updated, medoids):
                break
            medoids = updated
        loss = distance[:, medoids].min(axis=1).mean()
        if best is None or loss < best[0]:
            best = (loss, medoids.copy())
    return {"method": method, "metric": metric, "centers": sample.iloc[best[1]], "fields": fields}


def refit_blocks(frame, rng):
    groups = list(frame.groupby("family_block", sort=True).indices.values())
    return frame.iloc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])]


def cluster_profiles(data, fields, labels, target, smoke):
    config = replace(r.Config(), stability_refits=2 if smoke else 30, starts=2 if smoke else 5,
        vehicle_cap=200 if smoke else 10000, medoid_cap=100 if smoke else 2000, smoke=smoke,
        clusters=(2, 3) if smoke else tuple(range(2, 9)))
    results, members, profiles, representatives, sensitivity_rows = [], [], [], [], []
    for view, mask in profile_views(data, labels):
        population = data[mask]
        development = population[population.development]
        blocks = development.family_block.nunique()
        if len(development) < 50 or blocks < 10:
            results.append({"view": view, "method": "all", "status": "insufficient_support", "development_profiles": len(development), "family_blocks": blocks})
            continue
        destination = target / view
        destination.mkdir(parents=True)
        training = development if len(development) <= config.vehicle_cap else development.sample(config.vehicle_cap, random_state=42)
        anchor = development.sample(min(1000, len(development)), random_state=42)
        scores, fitted, repeats = [], {}, []
        floor = max(10, int(np.ceil(.05 * len(development))))
        for method in ["kmeans", "gower"]:
            for k in config.clusters:
                model = fit_partition(training, fields, method, config, k, 42)
                reference = r.assign_partition(model, anchor)
                sizes = np.bincount(r.assign_partition(model, development), minlength=k)
                aris = []
                rng = np.random.default_rng(42)
                for repeat in range(config.stability_refits):
                    changed = fit_partition(refit_blocks(training, rng), fields, method, config, k, 43 + repeat)
                    ari = adjusted_rand_score(reference, r.assign_partition(changed, anchor))
                    aris.append(ari)
                    repeats.append({"method": method, "k": k, "repeat": repeat, "ari": ari})
                silhouette = -1.
                if 1 < len(np.unique(reference)) < len(reference):
                    if method == "gower":
                        matrix = r.gower(anchor, anchor, model["metric"])
                        np.fill_diagonal(matrix, 0)
                        silhouette = silhouette_score(matrix, reference, metric="precomputed")
                    else:
                        silhouette = silhouette_score(model["encoder"].transform(anchor[fields].replace({None: np.nan})), reference)
                median = float(np.median(aris))
                scores.append({"method": method, "k": k, "median_ari": median, "ari_q10": float(np.quantile(aris,.1)),
                    "silhouette": silhouette, "minimum_count": int(sizes.min()), "required_minimum_count": floor,
                    "accepted_broad": median >= .7 and silhouette > 0 and sizes.min() >= floor})
                fitted[(method, k)] = model
                print(view, method, k, "ARI", round(median, 3), flush=True)
        scores = pd.DataFrame(scores)
        scores.to_csv(destination / "model_comparison.csv", index=False)
        pd.DataFrame(repeats).to_csv(destination / "stability.csv", index=False)
        for method in ["kmeans", "gower"]:
            row = s.choose_candidate(scores[scores.method.eq(method)], "broad")
            status = {"view": view, "method": method, "development_profiles": len(development), "family_blocks": blocks}
            if row is None:
                results.append({**status, "status": "no_supported_subdivision"})
                continue
            model = fitted[(method, int(row.k))]
            experiment = view + "/" + method
            directory = target / experiment
            directory.mkdir()
            joblib.dump(model, directory / "partition.joblib")
            membership = population[["profile_id", "parent_id", "development", "family_block"]].copy()
            membership["cluster"] = r.assign_partition(model, population)
            membership["experiment_id"] = experiment
            members.append(membership)
            for cluster, group in population.groupby(membership.cluster):
                name = legacy.subgroup_name(group, population, fields)
                for field in fields:
                    values = group[field].dropna()
                    profiles.append({"experiment_id": experiment, "cluster": cluster, "description": name, "field": field,
                        "observed": len(values), "missing_share": 1-len(values)/len(group), "median": values.median() if field not in CATEGORICAL else None,
                        "mode": r.mode_agreement(values)[0] if field in CATEGORICAL else None})
                distance = r.gower(group, model["centers"].iloc[[cluster]], model["metric"]).ravel() if method == "gower" else np.linalg.norm(model["encoder"].transform(group[fields].replace({None: np.nan}))-model["model"].cluster_centers_[cluster],axis=1)
                for offset in np.argsort(distance)[:3]:
                    representatives.append({"experiment_id": experiment, "cluster": cluster, **group.iloc[offset][["profile_id", *r.COHORT, "body_style", "powertrain"]].to_dict()})
            for name, alternative, subset in [
                ("equal_family", fields, development.sample(frac=1, random_state=42).drop_duplicates("family_block")),
                ("without_transmission", [f for f in fields if f != "transmission"], training)]:
                changed = fit_partition(subset, alternative, method, config, int(row.k), 42)
                sensitivity_rows.append({"experiment_id": experiment, "sensitivity": name,
                    "ari": adjusted_rand_score(r.assign_partition(model, anchor), r.assign_partition(changed, anchor))})
            results.append({**status, "status": "accepted_partition", **row.to_dict(), "experiment_id": experiment})
    combined = pd.concat(members, ignore_index=True) if members else pd.DataFrame(columns=["profile_id", "cluster", "experiment_id"])
    pd.DataFrame(results).to_csv(target / "experiments.csv", index=False)
    pd.DataFrame(profiles).to_csv(target / "profiles.csv", index=False)
    pd.DataFrame(representatives).to_csv(target / "representatives.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(target / "sensitivities.csv", index=False)
    combined.to_parquet(target / "memberships.parquet", index=False)
    return combined


def profile_labels(profiles, mappings, definitions):
    # Legacy exact-scope matching is reused through temporary profile identities.
    accepted, exceptions = [], []
    for row in mappings.to_dict("records"):
        mask = profiles.identity_complete & profiles.Make.eq(row["make"]) & profiles.market_family.eq(row["model"]) & pd.to_numeric(profiles.ModelYear).between(row["year_start"], row["year_end"])
        if not row["all_trims"]:
            mask &= profiles.Trim.isin(row["trims"])
        for profile in profiles[mask].to_dict("records"):
            reason = None
            if row["status"] != "source_supported":
                reason = "ambiguous_scope"
            if isinstance(row.get("body_styles"), list) and profile["body_style"] not in row["body_styles"]:
                reason = "ambiguous_scope"
            if row["dimension"] == "size_segment":
                expected = "suv_crossover" if row["label"].endswith("_suv") else "pickup" if row["label"].endswith("_pickup") else "passenger_car"
                if profile["body_family"] != expected:
                    reason = "ambiguous_scope"
            item = {"profile_id": profile["profile_id"], "dimension": row["dimension"], "label": row["label"], "mapping_id": row["mapping_id"],
                "source_url": row["url"], "evidence": row["excerpt"], "scope": row["scope_note"], "status": "source_supported", "taxonomy_version": VERSION}
            if reason:
                exceptions.append({**item, "reason": reason})
            else:
                accepted.append(item)
    labels = pd.DataFrame(accepted, columns=["profile_id", "dimension", "label", "mapping_id", "source_url", "evidence", "scope", "status", "taxonomy_version"])
    if len(labels):
        conflict = labels.groupby(["profile_id", "dimension"]).label.transform("nunique").gt(1) & ~labels.dimension.eq("performance_character")
        exceptions.extend(labels[conflict].assign(reason="conflicting_evidence").to_dict("records"))
        labels = labels[~conflict].drop_duplicates()
    labels["namespace"] = "market_terminology"
    labels["definition"] = [definitions["dimensions"][d][l] for d,l in zip(labels.dimension,labels.label)]
    for dimension in ["market_positioning", "performance_character", "size_segment"]:
        grouped = labels[labels.dimension.eq(dimension)].groupby("profile_id").label.agg(lambda x: ";".join(sorted(set(x))))
        profiles[dimension + "_labels"] = profiles.profile_id.map(grouped).fillna("")
    return profiles, labels, pd.DataFrame(exceptions)


def profile_views(data, labels):
    yield "global", data.eligible
    for (dimension, label), group in labels[labels.dimension.isin(legacy.SEMANTIC)].groupby(["dimension", "label"]):
        for body in ["passenger_car", "suv_crossover", "pickup", "van_minivan"]:
            mask = data.eligible & data.profile_id.isin(group.profile_id) & data.body_family.eq(body)
            if mask.any():
                yield f"{dimension}__{label}__{body}", mask


def peers(data, fields, labels, memberships, query_limit=None):
    training = data[data.development & data.eligible]
    metric = metric_for(training, fields)
    rows = []
    views = dict(profile_views(data, labels))
    clusters = {}
    for (experiment, cluster), group in memberships.groupby(["experiment_id", "cluster"]):
        clusters[(experiment, cluster)] = data.profile_id.isin(group.profile_id)
    observed = data[fields].notna().to_numpy()
    size = np.array([feature_block(f) == "size" for f in fields])
    power = ~size
    for position, qi in enumerate(data.index[data.eligible][:query_limit]):
        if position % 1000 == 0:
            print("Peer profiles:", position, flush=True)
        query = data.loc[qi]
        common = data.eligible & data.body_family.eq(query.body_family) & data.family_block.ne(query.family_block)
        # Unknown is explicitly exploratory; it never borrows known-powertrain peers.
        common &= data.powertrain.eq(query.powertrain)
        years = (pd.to_numeric(data.ModelYear) - float(query.ModelYear)).abs()
        indices = data.index[common & years.le(10)]
        if not len(indices):
            continue
        shared = observed[indices] & observed[qi]
        supported = (shared[:, size].sum(axis=1) >= max(2, int(np.ceil(size.sum() * .5)))) & (shared[:, power].sum(axis=1) >= max(1, int(np.ceil(power.sum() * .5))))
        indices = indices[supported]
        if not len(indices):
            continue
        distances = pd.Series(r.gower(data.loc[[qi]], data.loc[indices], metric).ravel(), index=indices)
        approaches = [("body_year_baseline", "global", common), ("direct_specs", "global", common)]
        approaches += [("category_specs", view, mask & common) for view, mask in views.items() if view != "global" and mask.loc[qi]]
        approaches += [("global_cluster" if experiment.startswith("global/") else "category_cluster", experiment, mask & common)
            for (experiment, cluster), mask in clusters.items() if mask.loc[qi]]
        for approach, view, mask in approaches:
            candidates = indices[mask.loc[indices]]
            near = candidates[years.loc[candidates].le(5)]
            window = 5 if data.loc[near, "family_block"].nunique() >= 5 else 10
            candidates = near if window == 5 else candidates
            order = pd.DataFrame({"index": candidates, "distance": distances.loc[candidates], "year": years.loc[candidates], "family": data.loc[candidates, "family_block"]})
            order = order.sort_values((["year", "index"] if approach == "body_year_baseline" else ["distance", "year", "index"])).drop_duplicates("family").head(5)
            for rank, (_, item) in enumerate(order.iterrows(), 1):
                pi = int(item["index"])
                rows.append({"query_profile_id": query.profile_id, "peer_profile_id": data.loc[pi, "profile_id"], "approach": approach, "view": view,
                    "rank": rank, "distance": item["distance"], "year_window": window, "slate_size": len(order),
                    "evidence_scope": "exploratory_unknown_powertrain" if query.powertrain == "unknown" else "supported_powertrain",
                    "shared_size": int((observed[qi] & observed[pi])[size].sum()), "shared_powertrain": int((observed[qi] & observed[pi])[power].sum())})
    return pd.DataFrame(rows, columns=["query_profile_id", "peer_profile_id", "approach", "view", "rank", "distance", "year_window", "slate_size", "evidence_scope", "shared_size", "shared_powertrain"])


def lookup_profiles(directory, year, make, model, trim):
    directory = Path(directory)
    legacy.verify_artifacts(directory)
    profiles = pd.read_parquet(directory / "variant_profiles.parquet")
    mask = pd.Series(True, index=profiles.index)
    for key, value in zip(r.COHORT, [year, make, model, trim]):
        mask &= profiles[key].astype(str).str.upper().eq(str(value).upper())
    selected = profiles[mask]
    labels = pd.read_parquet(directory / "profile_labels.parquet")
    peer_rows = pd.read_parquet(directory / "profile_peers.parquet")
    evidence = labels[labels.profile_id.isin(selected.profile_id)]
    similar = peer_rows[peer_rows.query_profile_id.isin(selected.profile_id)].merge(profiles, left_on="peer_profile_id", right_on="profile_id", validate="m:1")
    return selected, evidence, similar


def coverage_inventory(data, labels):
    records = []
    for (make, model, family), group in data[data.identity_complete].groupby(["Make", "Model", "market_family"]):
        for dimension in ["market_positioning", "performance_character", "size_segment"]:
            accepted = labels[labels.profile_id.isin(group.profile_id) & labels.dimension.eq(dimension)]
            records.append({"Make": make, "Model": model, "market_family": family, "dimension": dimension,
                "vin_count": int(group.vin_count.sum()), "profiles": len(group), "assigned_profiles": accepted.profile_id.nunique(),
                "reason": "source_supported" if len(accepted) else "missing_reference" if group.family_resolved.any() else "unmatched_identity"})
    return pd.DataFrame(records)


def evidence_batch(data):
    complete = data[data.identity_complete].copy()
    totals = complete.groupby(["Make", "market_family"]).vin_count.sum().sort_values(ascending=False)
    selected = list(totals.head(100).index)
    representatives = complete.sort_values("vin_count", ascending=False).drop_duplicates(["Make", "market_family"])
    representatives["decade"] = (pd.to_numeric(representatives.ModelYear) // 10 * 10).astype(int)
    representatives = representatives[~pd.MultiIndex.from_frame(representatives[["Make", "market_family"]]).isin(selected)]
    representatives["round"] = representatives.groupby(["body_family", "decade", "powertrain"]).cumcount()
    for row in representatives.sort_values(["round", "vin_count"], ascending=[True, False]).itertuples():
        if len(selected) >= 200:
            break
        selected.append((row.Make, row.market_family))
    return pd.DataFrame([{"Make": make, "market_family": model, "priority": i+1, "vin_count": int(totals.loc[(make, model)])} for i,(make,model) in enumerate(selected)])


def validate_outputs(data, labels, peer_rows, fields):
    findings = []
    def check(name, count):
        findings.append({"check": name, "violations": int(count), "status": "passed" if not count else "failed"})
    check("unique_profile_ids", data.profile_id.duplicated().sum())
    check("family_split_leakage", (data.groupby("family_block").development.nunique() > 1).sum())
    check("electric_displacement_coordinates", int("engine_displacement" in fields))
    check("width_coordinates", int("width" in fields))
    check("accepted_missing_source", labels.source_url.isna().sum() if len(labels) else 0)
    if len(peer_rows):
        q = data.set_index("profile_id").loc[peer_rows.query_profile_id].reset_index(drop=True)
        p = data.set_index("profile_id").loc[peer_rows.peer_profile_id].reset_index(drop=True)
        check("peer_family_leakage", q.family_block.eq(p.family_block).sum())
        check("peer_body_mismatch", q.body_family.ne(p.body_family).sum())
        check("peer_powertrain_mismatch", q.powertrain.ne(p.powertrain).sum())
        check("conflicting_peer_powertrain", q.powertrain.eq("conflicting").sum() + p.powertrain.eq("conflicting").sum())
        check("peer_year_window", ((pd.to_numeric(q.ModelYear)-pd.to_numeric(p.ModelYear)).abs().to_numpy() > peer_rows.year_window.to_numpy()).sum())
        check("shared_block_support", ((peer_rows.shared_size < max(2, int(np.ceil(.5 * sum(feature_block(f)=="size" for f in fields))))) |
            (peer_rows.shared_powertrain < max(1, int(np.ceil(.5 * sum(feature_block(f)=="powertrain" for f in fields)))))).sum())
        pairs = peer_rows.assign(peer_family=p.family_block.to_numpy())
        check("duplicate_peer_family", pairs.duplicated(["query_profile_id", "approach", "view", "peer_family"]).sum())
    result = pd.DataFrame(findings)
    if result.violations.any():
        raise ValueError("Automated validation failed: " + result[result.violations.gt(0)].to_json(orient="records"))
    return result


def analyze(prepared, output, smoke=False):
    prepared, output = Path(prepared), Path(output)
    legacy.verify_artifacts(prepared)
    target = r.begin_stage(output.parent, output.name)
    source_code = Path(__file__).read_bytes()
    (target / "execution_vehicle_taxonomy_v2.py").write_bytes(source_code)
    for filename in ["cohorts.parquet", "vin_profile_membership.parquet", "identity_comparison.parquet", "identity_names.parquet", "join_receipt.json", "vin_evidence_audit.csv"]:
        import shutil
        shutil.copyfile(prepared / filename, target / filename)
    profiles = pd.read_parquet(prepared / "variant_profiles.parquet")
    names = pd.read_parquet(prepared / "identity_names.parquet")
    definitions, registry = legacy.load_taxonomy(ASSETS / "definitions.v2.json", ASSETS / "evidence.v2.json")
    crosswalk = r.read_json(ASSETS / "identity_crosswalk.v2.json")["mappings"]
    profiles = resolve_families(profiles, names, crosswalk)
    membership_path = prepared / "vin_profile_membership.parquet"
    membership_schema = pl.scan_parquet(membership_path).collect_schema()
    if not {"horsepower", "EngineHP"}.issubset(membership_schema.names()):
        raise ValueError("Preparation predates VIN-level numerical scale evidence. Run the updated prepare stage into a fresh directory.")
    development_ids = pl.from_pandas(profiles.loc[profiles.identity_complete & profiles.development, ["profile_id"]])
    paired = pl.scan_parquet(membership_path).select(["profile_id", "horsepower", "EngineHP"]).join(development_ids.lazy(), on="profile_id", how="inner").collect()
    ratio = paired.filter(pl.col("horsepower").is_finite() & pl.col("EngineHP").is_finite() & (pl.col("horsepower") > 0) & (pl.col("EngineHP") > 0)).select((pl.col("horsepower") / pl.col("EngineHP")).alias("ratio"))["ratio"]
    median_ratio = ratio.median()
    verified_hp = len(ratio) >= 30 and median_ratio is not None and .9 <= median_ratio <= 1.1
    r.write_json(target / "numeric_scale_audit.json", {"horsepower": {"paired_development_vins":len(ratio), "median_source_reference_ratio":median_ratio,
        "verified":verified_hp,"reference":"NHTSA_vPIC.EngineHP","interpretation":"scale compatibility, not per-VIN accuracy"}})
    profiles, labels, exceptions = profile_labels(profiles, legacy.draft_mappings(registry), definitions)
    observed_labels = []
    for dimension, column in [("body_style", "body_style"), ("powertrain", "powertrain")]:
        for profile in profiles[profiles.identity_complete & ~profiles[column].isin(["unknown", "conflicting"])].itertuples():
            label = getattr(profile, column)
            observed_labels.append({"profile_id":profile.profile_id,"dimension":dimension,"label":label,"mapping_id":"vin_observed_"+dimension,
                "source_url":"snapshot://vin_profile_membership.parquet","evidence":f"{profile.vin_count} VINs in observed {column} partition",
                "scope":"Observed subordinate profile only; parent state is evaluated separately","status":"source_supported",
                "taxonomy_version":VERSION,"namespace":"observed","definition":definitions["dimensions"][dimension][label]})
    labels = pd.concat([labels,pd.DataFrame(observed_labels)],ignore_index=True)
    data, fields, audit = select_fields(profiles, excluded_fields=() if verified_hp else ("horsepower",))
    if not fields:
        raise ValueError("No supported specification fields")
    # Check original exporter unit handling and paired scale evidence before fitting.
    from EDA.DATA_CLEANING import USED_CARS_UNITS
    for field in [f for f in fields if f in r.UNITS and r.UNITS[f] == "in"]:
        if USED_CARS_UNITS.get(field) != "in":
            raise ValueError("Unverified dimension unit: " + field)
    data["taxonomy_version"] = VERSION
    data.to_parquet(target / "variant_profiles.parquet", index=False)
    labels.to_parquet(target / "profile_labels.parquet", index=False)
    label_support = labels.merge(profiles[["profile_id","parent_id","vin_count"]],on="profile_id",validate="m:1")
    label_support.drop_duplicates(["parent_id","profile_id","dimension","label"]).groupby(["parent_id","dimension","label"]).agg(
        supported_vins=("vin_count","sum"),supported_profiles=("profile_id","nunique")).to_csv(target / "parent_label_support.csv")
    exceptions.to_csv(target / "exceptions.csv", index=False)
    audit.to_csv(target / "feature_support.csv", index=False)
    inventory = coverage_inventory(data, labels)
    if len(exceptions):
        for (profile_id, dimension), group in exceptions.groupby(["profile_id", "dimension"]):
            identity = data.set_index("profile_id").loc[profile_id]
            mask = inventory.Make.eq(identity.Make) & inventory.Model.eq(identity.Model) & inventory.dimension.eq(dimension) & inventory.assigned_profiles.eq(0)
            inventory.loc[mask, "reason"] = "conflicting_evidence" if group.reason.eq("conflicting_evidence").any() else "ambiguous_scope"
    inventory.to_csv(target / "coverage_inventory.csv", index=False)
    batch = evidence_batch(data)
    batch.to_csv(target / "evidence_batch_001.csv", index=False)
    # Frozen selection records the actual attempted identities, not inferred label coverage.
    acquisition = registry.get("reference_batch")
    if acquisition:
        attempts = r.read_json(r.ROOT / acquisition["directory"] / "reference_results.json")
        pd.DataFrame([{"priority":i+1,"Make":row["make"],"market_family":row["model"],
            "observed_year_start":row["observed_year_start"],"observed_year_end":row["observed_year_end"],
            "acquisition_status":row["status"]} for i,row in enumerate(attempts)]).to_csv(target / "evidence_batch_001.csv",index=False)
        attempted = {(row["make"], row["model"]) for row in attempts}
        followup = data[~pd.MultiIndex.from_frame(data[["Make", "market_family"]]).isin(attempted)]
        next_batch = evidence_batch(followup)
        pd.concat([next_batch.iloc[:50], next_batch.iloc[100:150]], ignore_index=True).to_csv(target / "evidence_batch_next.csv", index=False)
        for row in attempts:
            mask = inventory.Make.eq(row["make"]) & inventory.Model.eq(row["model"]) & inventory.assigned_profiles.eq(0)
            if row["status"] == "ambiguous_scope":
                inventory.loc[mask, "reason"] = "ambiguous_scope"
            elif row["status"] == "unmatched_identity":
                inventory.loc[mask, "reason"] = "unmatched_identity"
            elif row["status"] in {"unavailable_classification", "source_supported"}:
                inventory.loc[mask, "reason"] = "unavailable_classification"
            else:
                inventory.loc[mask, "reason"] = "missing_reference"
        inventory.to_csv(target / "coverage_inventory.csv", index=False)
    data[data.identity_complete & data.development].sample(min(100, int((data.identity_complete & data.development).sum())), random_state=42).to_csv(target / "pilot_100.csv", index=False)
    matches = []
    for mapping in legacy.draft_mappings(registry).to_dict("records"):
        count = labels[labels.mapping_id.eq(mapping["mapping_id"])].profile_id.nunique()
        matching = data.identity_complete & data.Make.eq(mapping["make"]) & data.market_family.eq(mapping["model"]) & pd.to_numeric(data.ModelYear).between(mapping["year_start"],mapping["year_end"])
        matches.append({"mapping_id": mapping["mapping_id"], "assigned_profiles": count,
            "matching_profiles":int(matching.sum()),
            "zero_match_reason": "" if count else "ambiguous_scope_or_draft_evidence" if mapping["status"]!="source_supported" else "no_compatible_observed_profile_in_documented_scope"})
    pd.DataFrame(matches).to_csv(target / "mapping_match_audit.csv", index=False)
    for name, obj in [("definitions.json", definitions), ("evidence_registry.json", registry), ("identity_crosswalk.json", crosswalk)]:
        r.write_json(target / name, obj)
    acquisition = registry.get("reference_batch")
    if acquisition:
        directory = r.ROOT / acquisition["directory"]
        for filename in ["reference_results.json","receipt.json","selection.csv"]:
            if r.file_hash(directory/filename) != acquisition["hashes"][filename]:
                raise ValueError("Changed acquisition receipt: "+filename)
            import shutil
            shutil.copyfile(directory/filename,target/filename)
    scope_checks=[]
    index=profiles.set_index("profile_id")
    for mapping in legacy.draft_mappings(registry).to_dict("records"):
        assigned=labels[labels.mapping_id.eq(mapping["mapping_id"])]
        identities=index.loc[assigned.profile_id]
        valid=identities.Make.eq(mapping["make"]) & identities.market_family.eq(mapping["model"]) & pd.to_numeric(identities.ModelYear).between(mapping["year_start"],mapping["year_end"])
        if not mapping["all_trims"]:
            valid &= identities.Trim.isin(mapping["trims"])
        if isinstance(mapping.get("body_styles"),list):
            valid &= identities.body_style.isin(mapping["body_styles"])
        scope_checks.append({"mapping_id":mapping["mapping_id"],"accepted_assignments":len(assigned),"scope_violations":int((~valid).sum()),
            "source_present":bool(mapping["url"] and mapping["excerpt"] and mapping["scope_note"]),"accuracy":"not_measured"})
    scope_audit=pd.DataFrame(scope_checks)
    if scope_audit.scope_violations.any() or not scope_audit.source_present.all():
        raise ValueError("Invalid accepted evidence scope")
    scope_audit.to_csv(target/"evidence_support_audit.csv",index=False)
    for name, mask in [("all_retained", pd.Series(True,index=data.index)), ("complete_identity", data.identity_complete),
        ("specification_eligible", data.eligible), ("known_powertrain_eligible", data.eligible & ~data.powertrain.eq("unknown")),
        ("semantic_label", data.profile_id.isin(labels[labels.dimension.isin(legacy.SEMANTIC)].profile_id))]:
        record = {"stage": name, "listings": int(data.loc[mask,"vin_count"].sum()), "profiles": int(mask.sum()),
            "cohorts": data.loc[mask,"parent_id"].nunique(), "family_blocks": data.loc[mask,"family_block"].nunique()}
        if name == "all_retained":
            funnel = [record]
        else:
            funnel.append(record)
    pd.DataFrame(funnel).to_csv(target / "population_funnel.csv", index=False)
    # Freeze held-out query identities before fitting; no human-review files are generated.
    queries = legacy.stratified_rows(data[data.identity_complete & ~data.development], 200,
        ["body_family", "powertrain", "market_positioning_labels", "performance_character_labels"])
    queries[["profile_id", *r.COHORT, "family_block", "eligible", "powertrain"]].to_csv(target / "evaluation_queries.csv", index=False)
    clusters = target / "clusters"
    clusters.mkdir()
    memberships = cluster_profiles(data, fields, labels, clusters, smoke)
    peer_rows = peers(data, fields, labels, memberships, query_limit=100 if smoke else None)
    peer_rows.to_parquet(target / "profile_peers.parquet", index=False)
    validation = validate_outputs(data, labels, peer_rows, fields)
    validation.to_csv(target / "automated_validation.csv", index=False)
    evaluation = peer_rows[peer_rows.query_profile_id.isin(queries.profile_id)]
    slates = evaluation.groupby(["query_profile_id", "approach", "view", "evidence_scope"]).agg(mean_distance=("distance", "mean"),
        slate_size=("peer_profile_id", "size"), year_window=("year_window", "first")).reset_index()
    summary = slates.groupby(["approach", "view", "evidence_scope"]).agg(query_profiles=("query_profile_id", "nunique"), mean_distance=("mean_distance", "mean"),
        complete_list_share=("slate_size", lambda x: x.eq(5).mean()), relaxed_share=("year_window",lambda x:x.eq(10).mean())).reset_index()
    summary.to_csv(target / "peer_diagnostics.csv", index=False)
    average = evaluation.groupby(["query_profile_id","approach","view"]).distance.mean().reset_index()
    comparisons = []
    for approach, view in average[["approach","view"]].drop_duplicates().itertuples(index=False,name=None):
        if approach == "direct_specs":
            continue
        pair = average[average.approach.eq(approach)&average.view.eq(view)].merge(average[average.approach.eq("direct_specs")], on="query_profile_id",suffixes=("","_direct"),validate="1:1")
        comparisons.append({"approach":approach,"view":view,"paired_queries":len(pair),"mean_distance":pair.distance.mean(),
            "direct_mean_distance":pair.distance_direct.mean(),"distance_change":(pair.distance-pair.distance_direct).mean()})
    pd.DataFrame(comparisons).to_csv(target / "paired_peer_comparisons.csv",index=False)
    r.write_json(target / "evaluation.json", {"validation": "automated_only", "semantic_accuracy": "not_measured", "buyer_usefulness": "not_measured",
        "human_review": "not_performed_by_design", "query_population": "held_out_market_family_blocks", "query_profiles":len(queries),
        "eligible_queries":int(queries.eligible.sum()), "reduced_budget":smoke})
    # Combustion displacement is a sensitivity coordinate only, never an EV coordinate.
    ice = data[data.eligible & data.development & data.powertrain.eq("combustion") & data.engine_displacement.notna()]
    ice_supported = len(ice) >= 50 and ice.family_block.nunique() >= 10
    if ice_supported:
        sensitivity_target = target / "combustion_sensitivity"
        sensitivity_target.mkdir()
        ice_population = data[data.eligible & data.powertrain.eq("combustion") & data.engine_displacement.notna()].copy()
        cluster_profiles(ice_population, fields + ["engine_displacement"], labels, sensitivity_target, smoke)
    r.write_json(target / "combustion_sensitivity.json", {"status":"evaluated" if ice_supported else "insufficient_support",
        "development_profiles":len(ice),"family_blocks":ice.family_block.nunique(),"displacement_primary":False})
    r.write_json(target / "execution_configuration.json", {"schema_version":VERSION,"seed":42,"stability_refits":2 if smoke else 30,
        "starts":2 if smoke else 5,"vehicle_cap":200 if smoke else 10000,"medoid_cap":100 if smoke else 2000,
        "clusters":[2,3] if smoke else list(range(2,9)),"minimum_view_profiles":50,"minimum_family_blocks":10,
        "numeric_minimum_vins":5,"numeric_iqr_fraction":.2,"shared_block_observed_fraction":.5,
        "selected_fields":fields,"human_review":"not_performed_by_design","smoke_peer_query_limit":100 if smoke else None})
    if source_code != Path(__file__).read_bytes():
        raise ValueError("Execution source changed during analysis")
    r.finish_stage(target, "completed", schema_version=VERSION, reduced_budget=smoke, v2_code_sha256=r.file_hash(__file__),
        preparation_sha256=r.file_hash(prepared / "manifest.json"), source_run=str(prepared.resolve()),
        semantic_accuracy="not_measured", human_review="not_performed_by_design", selected_fields=fields)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "analyze"])
    parser.add_argument("--research-run", type=Path)
    parser.add_argument("--data-dir", type=Path, default=r.ROOT / "DATA/CAR_DATA_FINAL")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare" and args.research_run is None:
        parser.error("prepare requires --research-run")
    if args.stage == "analyze" and args.prepared is None:
        parser.error("analyze requires --prepared")
    if args.output.exists():
        parser.error("Output already exists. Use a fresh directory; historical runs are immutable.")
    try:
        if args.stage == "prepare":
            print(prepare(args.data_dir, args.research_run, args.output))
        else:
            print(analyze(args.prepared, args.output, args.smoke))
    except BaseException as error:
        if args.output.is_dir():
            r.finish_stage(args.output, "interrupted" if isinstance(error, KeyboardInterrupt) else "blocked",
                error=str(error) or type(error).__name__, schema_version=VERSION, reduced_budget=args.smoke,
                v2_code_sha256=r.file_hash(__file__))
        raise


if __name__ == "__main__":
    main()
