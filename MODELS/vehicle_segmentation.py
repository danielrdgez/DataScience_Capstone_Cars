"""Independent specification, market and family complaint experiments.

Legacy vehicle/complaint stages remain available through vehicle_profiles.
All new stages own immutable directories and use the preparation split receipt.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from MODELS import vehicle_profiles as r

MARKET = ["log_price", "log_mileage"]


def research_fields(cohorts, market=False):
    """Select supported fields on development cohorts; never on evaluation rows."""
    data = cohorts.copy()
    for field in r.RESEARCH_NUMERIC:
        data[field] = data[field].where((data[field + "_n"] >= 5) & (data[field + "_state"] != "not_applicable"))
    for field in r.CATEGORICAL:
        data[field] = data[field].where(data[field + "_n"] >= 5)
    for source, derived in [("price", "log_price"), ("mileage", "log_mileage")]:
        values = data[source + "_median"].where(data[source + "_n"] >= 5)
        data[derived] = np.log(values) if source == "price" else np.log1p(values)
    development = data[data.development & (data.vin_count >= 5)]
    fields, audit = [], []
    candidates = r.RESEARCH_NUMERIC + r.CATEGORICAL + (MARKET if market else [])
    for field in candidates:
        applicable = pd.Series(True, index=development.index)
        if field in r.RESEARCH_NUMERIC:
            applicable = development[field + "_state"] != "not_applicable"
        values = development.loc[applicable, field]
        coverage = float(values.notna().mean()) if len(values) else 0.
        reason = "selected" if coverage >= .7 and values.nunique() > 1 else "coverage_or_variation"
        if reason == "selected" and field not in r.CATEGORICAL and field not in MARKET:
            for prior in fields:
                if prior not in r.CATEGORICAL and r.feature_block(prior) == r.feature_block(field):
                    if abs(development[[prior, field]].corr(method="spearman").iloc[0, 1]) > .9:
                        reason = "redundant_with_" + prior
                        break
        if reason == "selected":
            fields.append(field)
        unit = r.UNITS.get(field, "in" if field in r.EXTRA_NUMERIC else "category")
        if field in MARKET:
            unit = "log USD" if field == "log_price" else "log1p source mileage"
        audit.append({"field": field, "development_coverage": coverage, "reason": reason,
                      "unit": unit, "block": r.feature_block(field), "applicable_cohorts": int(applicable.sum()),
                      "observed_cohorts": int(values.notna().sum()),
                      "heterogeneous_cohorts": int((development.get(field + "_state", pd.Series(dtype=str)) == "heterogeneous").sum())})
    observed = data[fields].notna().sum(axis=1)
    applicable_count = pd.Series(len(fields), index=data.index)
    for field in fields:
        if field in r.RESEARCH_NUMERIC:
            applicable_count -= (data[field + "_state"] == "not_applicable").astype(int)
    eligible = (data.vin_count >= 5) & (observed / applicable_count.clip(lower=1) >= .7)
    if market:
        eligible &= data[MARKET].notna().all(axis=1)
    return data, fields, eligible, pd.DataFrame(audit)


def matched_jaccard(reference, changed, k):
    table = pd.crosstab(pd.Series(reference, name="a"), pd.Series(changed, name="b")).reindex(index=range(k), columns=range(k), fill_value=0).to_numpy()
    union = table.sum(axis=1)[:, None] + table.sum(axis=0)[None, :] - table
    score = np.divide(table, union, out=np.zeros_like(table, dtype=float), where=union > 0)
    rows, columns = linear_sum_assignment(-score)
    return {int(a): float(score[a, b]) for a, b in zip(rows, columns)}


def choose_candidate(table, policy):
    passing = table[table["accepted_" + policy]].copy()
    if passing.empty:
        return None
    top = passing.silhouette.max()
    return passing[passing.silhouette >= top - .02].sort_values(["k", "silhouette"], ascending=[True, False]).iloc[0]


def candidates(data, fields, config, target, methods=("kmeans", "gower"), complaint=False):
    """Fixed fields, Make/Model resampling, method-specific selection."""
    development = data[data.development]
    training = r.stratified_sample(development, config.vehicle_cap, config.seed) if "body_type" in data else development
    anchor = development.sample(min(1000, len(development)), random_state=config.seed)
    rows, stability, fitted = [], [], {}
    repeats = config.stability_refits if config.smoke else 30
    if len(training) < 10 or not fields:
        pd.DataFrame(columns=["method", "k", "silhouette", "accepted_broad", "accepted_small"]).to_csv(target / "model_comparison.csv", index=False)
        return {}, pd.DataFrame()
    for method in methods:
        for k in config.clusters:
            if k >= len(training):
                continue
            model = r.fit_partition(training, fields, method, k, config, config.seed)
            labels = r.assign_partition(model, anchor)
            all_labels = r.assign_partition(model, development)
            sizes = np.bincount(all_labels, minlength=k)
            scores = []
            rng = np.random.default_rng(config.seed)
            for repeat in range(repeats):
                refit = r.fit_partition(r.resample_blocks(training, rng), fields, method, k, config, config.seed + repeat + 1)
                changed = r.assign_partition(refit, anchor)
                ari = adjusted_rand_score(labels, changed)
                scores.append(ari)
                for cluster, value in matched_jaccard(labels, changed, k).items():
                    stability.append({"method": method, "k": k, "repeat": repeat, "ari": ari, "cluster": cluster, "matched_jaccard": value})
            separation = -1.
            if 1 < len(np.unique(labels)) < len(labels):
                if method == "gower":
                    distance = r.gower(anchor, anchor, model["metric"])
                    np.fill_diagonal(distance, 0)
                    separation = silhouette_score(distance, labels, metric="precomputed")
                else:
                    separation = silhouette_score(model["encoder"].transform(anchor[fields].replace({None: np.nan})), labels)
            broad_floor = max(20, int(np.ceil(.05 * len(development)))) if complaint else int(np.ceil(.05 * len(development)))
            small_floor = broad_floor if complaint else max(30, int(np.ceil(.01 * len(development))))
            stable = np.median(scores) >= .7 and separation > 0
            row = {"method": method, "k": k, "median_ari": float(np.median(scores)),
                   "ari_q10": float(np.quantile(scores, .1)), "ari_q90": float(np.quantile(scores, .9)),
                   "silhouette": float(separation), "minimum_count": int(sizes.min()), "development_cohorts": len(development),
                   "accepted_broad": bool(stable and sizes.min() >= broad_floor),
                   "accepted_small": bool(stable and sizes.min() >= small_floor)}
            rows.append(row)
            fitted[(method, k)] = model
            print(target.name, method, k, "ARI", round(row["median_ari"], 3), "silhouette", round(separation, 3), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(target / "model_comparison.csv", index=False)
    pd.DataFrame(stability).to_csv(target / "stability.csv", index=False)
    selected = {}
    for method in methods:
        for policy in (["broad"] if complaint else ["broad", "small"]):
            row = choose_candidate(table[table.method == method], policy)
            if row is not None:
                selected[f"{method}_{policy}"] = (fitted[(method, int(row.k))], row.to_dict())
    return selected, table


def save_membership(data, eligible, model, experiment, target, keys=r.COHORT):
    membership = data.copy()
    membership["experiment_id"] = experiment
    membership["cluster"] = -1
    membership["eligible"] = eligible
    membership["assignment_status"] = np.where(eligible, "assigned", "insufficient_evidence")
    if "complaint_count" in membership:
        membership.loc[membership.complaint_count == 0, "assignment_status"] = "no_complaint_evidence"
    membership.loc[eligible, "cluster"] = r.assign_partition(model, membership.loc[eligible])
    preparation = next(p / "prepare/manifest.json" for p in target.parents if (p / "prepare/manifest.json").is_file())
    membership["source_prepare_sha256"] = r.file_hash(preparation)
    r.save_table(target / "membership.parquet", membership)
    membership[keys + ["experiment_id", "cluster", "assignment_status"]].to_csv(target / "membership.csv", index=False)
    joblib.dump(model, target / "partition.joblib")
    return membership


def field_audit(cohorts, prepare_manifest, target):
    _, _, _, audit = research_fields(cohorts)
    audit["median_observed_vins"] = [float(cohorts[f + "_n"].median()) if f + "_n" in cohorts else None for f in audit.field]
    audit["median_observed_agreement"] = [float(cohorts[f + "_agreement"].median()) if f + "_agreement" in cohorts else None for f in audit.field]
    audit["listing_vpic_comparison"] = ["paired_ratio_audit" if f in ["horsepower", "engine_displacement"] else "not_checked_no_shared_verified_measure" for f in audit.field]
    audit.to_csv(target / "field_audit.csv", index=False)
    checks = prepare_manifest.get("vehicles", {}).get("unit_checks", {})
    r.write_json(target / "source_field_checks.json", {"listing_vpic_checks": checks,
        "disagreement_interpretation": "paired median ratios check scale, not per-VIN agreement",
        "structural_missingness": "electric fuel type marks displacement not applicable; unknown remains unknown",
        "bed_length": "truck context only", "mileage_unit": "unverified unless source manifest declares mileage unit",
        "numeric_support": "at least five observed VIN values per modeled cohort field",
        "kmeans_missingness": "median imputation is a modeling coordinate, never observed equipment evidence"})


def mileage_unit_receipt(preparation):
    source = Path(preparation["source_dir"]) / "manifest.json"
    if not source.is_file():
        return {"verified": False, "reason": "source unit metadata absent"}
    manifest = r.read_json(source)
    units = manifest.get("units", {})
    for table in manifest.get("tables", {}).values():
        if table.get("file") == "used_cars.parquet":
            units = {**units, **table.get("units", {})}
    unit = units.get("mileage")
    return {"verified": unit in ["mi", "miles", "km", "kilometers"], "unit": unit,
            "evidence": str(source), "source_manifest_sha256": r.file_hash(source),
            "reason": "explicit source metadata" if unit else "source unit metadata absent"}


def segment_diagnostics(membership, model, fields, config, target):
    assigned = membership[membership.cluster >= 0]
    numeric = [f for f in r.RESEARCH_NUMERIC + ["price_median", "mileage_median", "new_listing_share"] if f in membership]
    assigned.groupby("cluster")[numeric].agg(["count", "median", "min", "max"]).to_csv(target / "profiles.csv")
    long = assigned.melt(id_vars=r.COHORT + ["cluster", "vin_count"], value_vars=numeric, var_name="field", value_name="value")
    r.save_table(target / "observed_distributions.parquet", long)
    for field in ["body_type", "ModelYear", *r.CATEGORICAL]:
        pd.crosstab(membership.cluster, membership[field].fillna("UNKNOWN")).to_csv(target / (field + "_composition.csv"))
    missing = membership.assign(missing_fraction=membership[fields].isna().mean(axis=1))
    missing.groupby("cluster")[["missing_fraction", "vin_count", "new_listing_share"]].agg(["count", "mean", "median"]).to_csv(target / "missingness_support.csv")
    references = []
    for name, labels in [("body", assigned.body_type.fillna("UNKNOWN")),
                         ("body_powertrain", assigned[["body_type", *r.CATEGORICAL]].fillna("UNKNOWN").astype(str).agg("|".join, axis=1))]:
        references.append({"reference": name, "ari": adjusted_rand_score(assigned.cluster, labels)})
    pd.DataFrame(references).to_csv(target / "reference_agreement.csv", index=False)
    development = assigned[assigned.development]
    anchor = development.sample(min(1000, len(development)), random_state=config.seed)
    complete = development.dropna(subset=fields)
    diagnostics = []
    k = model["model"].n_clusters if model["method"] == "kmeans" else len(model["centers"])
    if len(complete) > k:
        refit = r.fit_partition(r.stratified_sample(complete, config.vehicle_cap, config.seed), fields, model["method"], k, config, config.seed)
        for name, common in [("complete_case_overlap", anchor.dropna(subset=fields)), ("same_anchor_population_refit", anchor)]:
            if len(common) > 1:
                diagnostics.append({"sensitivity": name, "common_cohorts": len(common), "development_complete": len(complete),
                                    "development_total": len(development), "ari": adjusted_rand_score(r.assign_partition(model, common), r.assign_partition(refit, common)),
                                    "fields_frozen": True})
    # Compare missing-data treatments under the same Gower geometry and same anchor.
    metric = r.fit_gower(development, fields)
    observed = r.gower(anchor, anchor, metric)
    filled = anchor.copy()
    for field in fields:
        value = r.mode_agreement(development[field])[0] if field in r.CATEGORICAL else development[field].median()
        filled[field] = filled[field].fillna(value)
    imputed = r.gower(filled, filled, metric)
    upper = np.triu_indices(len(anchor), 1)
    diagnostics.append({"sensitivity": "observed_pair_vs_imputed_gower", "common_cohorts": len(anchor),
                        "mean_absolute_distance_change": float(np.abs(observed[upper] - imputed[upper]).mean()), "fields_frozen": True})
    # Equal year/make/model family sensitivity balances observed trims within each family.
    balanced = development.copy()
    groups = balanced.groupby(r.FAMILY, sort=True)
    count = max(1, int(groups.size().median()))
    balanced = groups.sample(n=count, replace=True, random_state=config.seed)
    refit = r.fit_partition(r.stratified_sample(balanced, config.vehicle_cap, config.seed), fields, model["method"], k, config, config.seed)
    diagnostics.append({"sensitivity": "equal_family_fitting", "common_cohorts": len(anchor),
                        "ari": adjusted_rand_score(r.assign_partition(model, anchor), r.assign_partition(refit, anchor)), "fields_frozen": True})
    pd.DataFrame(diagnostics).to_csv(target / "sensitivities.csv", index=False)
    weighted = []
    for field in numeric:
        usable = assigned[field].notna()
        for cluster, group in assigned[usable].groupby("cluster"):
            weighted.append({"cluster": cluster, "field": field, "listing_weighted_mean": np.average(group[field], weights=group.vin_count)})
    pd.DataFrame(weighted).to_csv(target / "listing_weighted_profiles.csv", index=False)


def segments(run_dir):
    target = r.begin_stage(run_dir, "segments")
    source, preparation = r.verified_stage(run_dir, "prepare")
    if preparation["status"] != "prepared":
        return r.finish_stage(target, "skipped", reason=preparation["status"])
    cohorts = pd.read_parquet(source / "cohorts.parquet")
    config = r.run_config(preparation)
    field_audit(cohorts, preparation, target)
    unit = mileage_unit_receipt(preparation)
    r.write_json(target / "mileage_units.json", unit)
    summary, memberships = [], {}
    for representation in ["specifications", "market"]:
        experiment = r.begin_stage(target, representation)
        if representation == "market" and not unit["verified"]:
            r.finish_stage(experiment, "blocked_unverified_mileage_unit", evidence=unit)
            continue
        data, fields, eligible, audit = research_fields(cohorts, representation == "market")
        audit.to_csv(experiment / "field_audit.csv", index=False)
        if not any(r.feature_block(f) == "size" for f in fields) or not any(r.feature_block(f) == "powertrain" for f in fields) or (representation == "market" and not set(MARKET) <= set(fields)):
            r.finish_stage(experiment, "insufficient_specifications", fields=fields)
            continue
        selected, _ = candidates(data[eligible], fields, config, experiment)
        for name, (model, row) in selected.items():
            path = r.begin_stage(experiment, name)
            identity = representation + "/" + name
            membership = save_membership(data, eligible, model, identity, path)
            representatives = pd.read_parquet(source / "representatives.parquet")
            r.save_table(path / "representative_vehicles.parquet", membership[r.COHORT + ["cluster", "experiment_id"]].merge(representatives, on=r.COHORT, validate="one_to_one"))
            segment_diagnostics(membership, model, fields, config, path)
            r.cluster_charts(membership, model, data[eligible & data.development], path)
            memberships[identity] = membership
            summary.append({"experiment_id": identity, "assigned": int(eligible.sum()), "unassigned": int((~eligible).sum()), **row})
            r.finish_stage(path, "accepted_partition", selected=row, fields=fields, preparation_sha256=r.file_hash(source / "manifest.json"))
        r.finish_stage(experiment, "evaluated" if selected else "no_supported_partition", fields=fields)
    pd.DataFrame(summary).to_csv(target / "experiments.csv", index=False)
    # Compare common members, never different eligibility populations.
    comparisons = []
    common_models = r.begin_stage(target, "common_population")
    for identity, left in memberships.items():
        if not identity.startswith("specifications/"):
            continue
        right = memberships.get(identity.replace("specifications/", "market/"))
        if right is not None:
            common = (left.cluster >= 0) & (right.cluster >= 0)
            common_labels = []
            for representation, frame in [("specifications", left), ("market", right)]:
                selected_path = target / representation / identity.split("/")[1]
                frozen = r.read_json(selected_path / "manifest.json")
                fields = frozen["fields"]
                subset = frame.loc[common]
                training = subset[subset.development]
                k = int(frozen["selected"]["k"])
                if len(training) <= k:
                    common_labels = []
                    break
                model = r.fit_partition(r.stratified_sample(training, config.vehicle_cap, config.seed), fields,
                                        frozen["selected"]["method"], k, config, config.seed)
                common_labels.append(r.assign_partition(model, subset))
                joblib.dump(model, common_models / (representation + "_" + identity.split("/")[1] + ".joblib"))
            comparisons.append({"experiment": identity.split("/")[1], "common_cohorts": int(common.sum()),
                                "common_development_cohorts": int((common & left.development).sum()),
                                "full_fit_common_member_ari": adjusted_rand_score(left.loc[common, "cluster"], right.loc[common, "cluster"]),
                                "common_population_refit_ari": adjusted_rand_score(*common_labels) if common_labels else None})
    pd.DataFrame(comparisons, columns=["experiment", "common_cohorts", "common_development_cohorts", "full_fit_common_member_ari", "common_population_refit_ari"]).to_csv(target / "representation_agreement.csv", index=False)
    r.finish_stage(common_models, "evaluated" if comparisons else "unavailable_no_common_representation", policy="fixed representation fields and selected k; identical fitting and evaluation cohort population")
    return r.finish_stage(target, "evaluated", experiments=len(summary), market_unit_status=unit,
                          preparation_sha256=r.file_hash(source / "manifest.json"), reduced_budget=config.smoke)


def incident_features(records, topic_weights=None):
    keys = r.FAMILY + ["odiNumber"]
    flags = records[keys].copy()
    for name, pattern in r.COMPONENTS.items():
        flags["component_" + name] = records.components.fillna("").str.contains(pattern, case=False, regex=True).to_numpy()
    incidents = flags.groupby(keys, observed=True, sort=True).max().reset_index()
    if topic_weights is not None:
        hashes = records[keys + ["text_hash"]].drop_duplicates(keys)
        incidents = incidents.merge(hashes, on=keys, validate="one_to_one").merge(topic_weights, on="text_hash", how="left", validate="many_to_one")
    return incidents


def complaint_uncertainty(incidents, membership, model, fields, config, target):
    k = model["model"].n_clusters
    draws = 10 if config.smoke else 200
    rows, halves = [], []
    rng = np.random.default_rng(config.seed)
    indexed = membership.set_index(r.FAMILY)
    for key, group in incidents.groupby(r.FAMILY, sort=True):
        if key not in indexed.index or indexed.loc[key, "cluster"] < 0:
            continue
        values = group[fields].astype(float).to_numpy()
        sampled = []
        for _ in range(draws):
            sample = values[rng.integers(0, len(values), len(values))]
            sampled.append(np.nanmean(sample, axis=0))
        labels = r.assign_partition(model, pd.DataFrame(sampled, columns=fields))
        frequencies = np.bincount(labels, minlength=k) / draws
        for cluster, probability in enumerate(frequencies):
            rows.append({**dict(zip(r.FAMILY, key)), "cluster": cluster, "assignment_frequency": probability,
                         "modal_assignment_support": float(frequencies.max()), "uncertain": bool(frequencies.max() < .8), "draws": draws})
        if len(values) >= 50:
            order = rng.permutation(len(values))
            a, b = order[:len(order)//2], order[len(order)//2:]
            pair = pd.DataFrame([np.nanmean(values[a], axis=0), np.nanmean(values[b], axis=0)], columns=fields)
            result = r.assign_partition(model, pair)
            halves.append({**dict(zip(r.FAMILY, key)), "half_a_cluster": int(result[0]), "half_b_cluster": int(result[1]), "agreement": bool(result[0] == result[1]), "half_a_n": len(a), "half_b_n": len(b)})
    pd.DataFrame(rows).to_csv(target / "assignment_uncertainty.csv", index=False)
    pd.DataFrame(halves, columns=r.FAMILY + ["half_a_cluster", "half_b_cluster", "agreement", "half_a_n", "half_b_n"]).to_csv(target / "independent_halves.csv", index=False)


def complaint_profiles(run_dir):
    target = r.begin_stage(run_dir, "complaint_profiles")
    source, preparation = r.verified_stage(run_dir, "prepare")
    if preparation["status"] != "prepared":
        return r.finish_stage(target, "skipped", reason=preparation["status"])
    config = r.run_config(preparation)
    records = pd.read_parquet(source / "complaints.parquet")
    checks = r.automatic_mapping_checks(records)
    r.write_json(target / "mapping_checks.json", checks)
    if not checks["passed"]:
        return r.finish_stage(target, "blocked_mapping_checks")
    records = records[records.primary_eligible].copy()
    mapping = records[["odiNumber", "components"]].drop_duplicates()
    for name, pattern in r.COMPONENTS.items():
        mapping[name] = mapping.components.fillna("").str.contains(pattern, case=False, regex=True)
    mapping["mapped_categories"] = mapping[list(r.COMPONENTS)].sum(axis=1)
    mapping.groupby("components", dropna=False).agg(records=("odiNumber", "nunique"), matched_categories=("mapped_categories", "max")).to_csv(target / "component_mapping_receipt.csv")
    union = mapping.groupby("odiNumber")[list(r.COMPONENTS)].max().sum(axis=1)
    r.write_json(target / "taxonomy_audit.json", {"distinct_incidents": len(union),
        "unmapped_incidents": int((union == 0).sum()),
        "overlapping_category_incidents": int((union > 1).sum()),
        "patterns": r.COMPONENTS, "shares_sum_to_one": False})
    weights = None
    complaint_dir = Path(run_dir) / "complaints"
    if (complaint_dir / "manifest.json").is_file():
        _, receipt = r.verified_stage(run_dir, "complaints")
        if receipt["status"] != "skipped":
            count = r.choose_topics(complaint_dir, None)
            if count is not None:
                saved = joblib.load(complaint_dir / "topics.joblib")
                docs = records[records.text_eligible].drop_duplicates("text_hash")
                values = r.normalized_mixtures(saved["models"][count].transform(saved["vectorizer"].transform(docs.normalized_text)))
                weights = pd.DataFrame(values, columns=[f"topic_{i}" for i in range(count)])
                weights.insert(0, "text_hash", docs.text_hash.to_numpy())
                weights = weights.loc[values.sum(axis=1) > 0]
    incidents = incident_features(records, weights)
    r.save_table(target / "incidents.parquet", incidents)
    components = ["component_" + name for name in r.COMPONENTS]
    topics = [] if weights is None else [f for f in weights if f.startswith("topic_")]
    families = incidents.groupby(r.FAMILY, sort=True)[components + topics].mean().reset_index()
    counts = incidents.groupby(r.FAMILY, sort=True).size().rename("complaint_count").reset_index()
    families = families.merge(counts, on=r.FAMILY, validate="one_to_one")
    if topics:
        usable = incidents[topics].notna().all(axis=1)
        support = incidents[usable].groupby(r.FAMILY).size().rename("topic_complaint_count").reset_index()
        families = families.merge(support, on=r.FAMILY, how="left", validate="one_to_one")
    population = pd.read_parquet(source / "cohorts.parquet", columns=r.FAMILY).drop_duplicates()
    families = population.merge(families, on=r.FAMILY, how="outer", validate="one_to_one")
    families["complaint_count"] = families.complaint_count.fillna(0).astype(int)
    families["development"] = [r.is_development(a, b, config.seed) for a, b in zip(families.Make, families.Model)]
    r.save_table(target / "family_features.parquet", families)
    summary = []
    for representation, fields in [("components", components), ("components_topics", components + topics)]:
        if representation == "components_topics" and not topics:
            path = target / "components_topics"
            path.mkdir()
            r.finish_stage(path, "skipped_no_accepted_topics")
            continue
        for minimum in [20, 50, 100]:
            identity = f"{representation}_{minimum}"
            path = r.begin_stage(target, identity)
            eligible = families.complaint_count >= minimum
            if representation == "components_topics":
                eligible &= families.topic_complaint_count.fillna(0) >= minimum
            selected, _ = candidates(families[eligible], fields, config, path, methods=("kmeans",), complaint=True)
            if not selected:
                r.finish_stage(path, "no_supported_partition", fields=fields, minimum_complaints=minimum)
                continue
            model, row = selected["kmeans_broad"]
            membership = save_membership(families, eligible, model, identity, path, keys=r.FAMILY)
            complaint_uncertainty(incidents, membership, model, fields, config, path)
            membership[membership.cluster >= 0].groupby("cluster")[fields + ["complaint_count"]].agg(["count", "mean", "median"]).to_csv(path / "profiles.csv")
            summary.append({"experiment_id": identity, "assigned": int(eligible.sum()), **row})
            r.finish_stage(path, "accepted_partition", selected=row, fields=fields, minimum_complaints=minimum,
                           semantic_validation="not_established", preparation_sha256=r.file_hash(source / "manifest.json"))
    pd.DataFrame(summary).to_csv(target / "experiments.csv", index=False)
    return r.finish_stage(target, "evaluated", experiments=len(summary), reduced_budget=config.smoke,
                          preparation_sha256=r.file_hash(source / "manifest.json"))


def family_mixtures(membership, numeric):
    result = r.family_profiles(membership)
    result["unassigned_share"] = 1 - result.assignment_coverage
    for field in numeric:
        if field not in result:
            values = membership.groupby(r.FAMILY)[field].median().rename(field).reset_index()
            result = result.merge(values, on=r.FAMILY, validate="one_to_one")
    return result


def ridge_design(training, test, fields, target, alpha):
    categorical = [f for f in fields if f in ["body_type", *r.CATEGORICAL]]
    numeric = [f for f in fields if f not in categorical]
    pipeline = make_pipeline(ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric),
        ("category", make_pipeline(SimpleImputer(strategy="constant", fill_value="__missing__"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical)]), Ridge(alpha=alpha))
    pipeline.fit(training[fields].replace({None: np.nan}), training[target])
    return pipeline.predict(test[fields].replace({None: np.nan}))


def fold_profiles(cohorts, model, fields, config, fit_blocks):
    """All learned vehicle inputs are refitted using only fold-training families."""
    local = cohorts.copy()
    local["development"] = (local.Make.astype(str) + "|" + local.Model.astype(str)).isin(fit_blocks)
    local, selected_fields, eligible, _ = research_fields(local, any(f in MARKET for f in fields))
    training = local[local.development & eligible]
    k = model["model"].n_clusters if model["method"] == "kmeans" else len(model["centers"])
    if len(training) <= k or not selected_fields:
        raise ValueError("Insufficient fold-specific vehicle support")
    fitted = r.fit_partition(r.stratified_sample(training, config.vehicle_cap, config.seed), selected_fields, model["method"], k, config, config.seed)
    local["eligible"] = eligible
    local["cluster"] = -1
    local.loc[eligible, "cluster"] = r.assign_partition(fitted, local[eligible])
    profile = family_mixtures(local, fields)
    receipt = {"training_blocks": sorted(fit_blocks), "fields": selected_fields, "training_cohorts": len(training)}
    return profile, receipt


def evaluate_relationship(cohorts, membership, outcomes, model, fields, config, target):
    numeric = [f for f in fields if f not in r.CATEGORICAL]
    profiles = family_mixtures(membership, numeric)
    r.save_table(target / "family_mixtures.parquet", profiles)
    data = profiles.merge(outcomes, on=r.FAMILY, validate="one_to_one")
    baseline = ["body_type", "ModelYear", *r.CATEGORICAL]
    cluster_fields = [f for f in profiles if f.startswith("cluster_")]
    variants = {"baseline": baseline, "clusters": baseline + cluster_fields,
                "continuous": baseline + numeric, "both": baseline + numeric + cluster_fields}
    reports, predictions, intervals, fold_receipts = [], [], [], []
    targets = [f for f in outcomes if f.startswith("component_")]
    # Topic effects remain in the legacy exploratory comparison; this comparison isolates structured evidence.
    for coverage_name, coverage in [("primary", .8), ("coverage_sensitivity", 0.)]:
        eligible = data[(data.complaint_count >= 20) & (data.assignment_coverage >= coverage) & (data.assignment_coverage > 0)].copy()
        training, heldout = eligible[eligible.development], eligible[~eligible.development]
        blocks = training.Make.astype(str) + "|" + training.Model.astype(str)
        if blocks.nunique() < 2 or len(heldout) < 2:
            reports.append({"coverage_policy": coverage_name, "status": "insufficient_overlap"})
            continue
        folds = []
        for fold, (fit_index, validation_index) in enumerate(GroupKFold(min(5, blocks.nunique())).split(training, groups=blocks)):
            fit_blocks = set(blocks.iloc[fit_index])
            fold_profile, receipt = fold_profiles(cohorts, model, fields, config, fit_blocks)
            receipt.update(fold=fold, coverage_policy=coverage_name, validation_blocks=sorted(set(blocks.iloc[validation_index])))
            fold_receipts.append(receipt)
            fold_data = fold_profile.merge(outcomes, on=r.FAMILY, validate="one_to_one")
            ids = fold_data.Make.astype(str) + "|" + fold_data.Model.astype(str)
            usable = (fold_data.complaint_count >= 20) & (fold_data.assignment_coverage >= coverage) & (fold_data.assignment_coverage > 0)
            fit = fold_data[usable & ids.isin(fit_blocks)]
            validation = fold_data[usable & ids.isin(set(blocks.iloc[validation_index]))]
            folds.append((fit, validation))
        for outcome in targets:
            errors = {}
            for variant, predictors in variants.items():
                losses = []
                for alpha in [.1, 1., 10., 100.]:
                    fold_errors = []
                    for fit, validation in folds:
                        if fit.empty or validation.empty:
                            continue
                        values = ridge_design(fit, validation, predictors, outcome, alpha)
                        fold_errors.extend(np.abs(validation[outcome].to_numpy() - values))
                    losses.append(np.mean(fold_errors) if fold_errors else np.inf)
                if not np.isfinite(losses).any():
                    raise ValueError("No valid fold for relationship validation")
                alpha = [.1, 1., 10., 100.][int(np.argmin(losses))]
                values = ridge_design(training, heldout, predictors, outcome, alpha)
                errors[variant] = np.abs(heldout[outcome].to_numpy() - values)
                reports.append({"coverage_policy": coverage_name, "status": "evaluated", "outcome": outcome, "variant": variant,
                                "mae": float(errors[variant].mean()), "alpha": alpha, "heldout_families": len(heldout)})
                for key, actual, prediction in zip(heldout[r.FAMILY].to_numpy(), heldout[outcome], values):
                    predictions.append({**dict(zip(r.FAMILY, key)), "coverage_policy": coverage_name, "outcome": outcome,
                                        "variant": variant, "actual": actual, "prediction": prediction})
            test_blocks = (heldout.Make.astype(str) + "|" + heldout.Model.astype(str)).to_numpy()
            for variant, reference in [("clusters", "baseline"), ("continuous", "baseline"), ("both", "continuous")]:
                difference = errors[variant] - errors[reference]
                bounds = r.paired_intervals(difference, test_blocks, config.bootstrap, config.seed)
                intervals.append({"coverage_policy": coverage_name, "outcome": outcome, "variant": variant, "reference": reference,
                                  "mean_mae_change": difference.mean(), "lower": bounds[0] if bounds else None,
                                  "upper": bounds[1] if bounds else None, "heldout_blocks": len(np.unique(test_blocks)),
                                  "interval_status": "available" if bounds else "fewer_than_20_blocks"})
    pd.DataFrame(reports).to_csv(target / "comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(target / "predictions.csv", index=False)
    pd.DataFrame(intervals).to_csv(target / "paired_intervals.csv", index=False)
    r.write_json(target / "fold_receipts.json", fold_receipts)


def cross_profile_bootstrap(profiles, complaint_membership, config, target):
    columns = [f for f in profiles if f.startswith("cluster_")] + ["unassigned_share"]
    linked = profiles.merge(complaint_membership[r.FAMILY + ["cluster"]], on=r.FAMILY, validate="one_to_one")
    linked = linked[linked.cluster >= 0]
    rows = []
    rng = np.random.default_rng(config.seed)
    for cluster, subset in linked.groupby("cluster"):
        groups = list(subset.groupby(["Make", "Model"]).indices.values())
        estimates = []
        if len(groups) >= 20:
            for _ in range(config.bootstrap):
                indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                estimates.append(subset.iloc[indices][columns].mean().to_numpy())
        for i, field in enumerate(columns):
            bounds = np.quantile(np.array(estimates)[:, i], [.025, .975]) if estimates else [None, None]
            rows.append({"complaint_cluster": cluster, "vehicle_share": field, "mean": subset[field].mean(),
                         "lower": bounds[0], "upper": bounds[1], "families": len(subset), "make_model_blocks": len(groups)})
    pd.DataFrame(rows).to_csv(target / "complaint_vehicle_cross_profiles.csv", index=False)


def relationships(run_dir):
    target = r.begin_stage(run_dir, "relationships")
    source, preparation = r.verified_stage(run_dir, "prepare")
    r.verified_stage(run_dir, "segments")
    complaint_dir, complaint_manifest = r.verified_stage(run_dir, "complaint_profiles")
    if complaint_manifest["status"] != "evaluated":
        return r.finish_stage(target, "skipped", reason=complaint_manifest["status"])
    config = r.run_config(preparation)
    cohorts = pd.read_parquet(source / "cohorts.parquet")
    outcomes = pd.read_parquet(complaint_dir / "family_features.parquet").drop(columns="development")
    completed = []
    for path in sorted((Path(run_dir) / "segments").glob("*/*/membership.parquet")):
        _, receipt = r.verified_stage(path.parents[2], str(path.parent.relative_to(path.parents[2])))
        membership = pd.read_parquet(path)
        model = joblib.load(path.parent / "partition.joblib")
        destination = r.begin_stage(target, path.parent.parent.name + "_" + path.parent.name)
        evaluate_relationship(cohorts, membership, outcomes, model, receipt["fields"], config, destination)
        primary = complaint_dir / "components_20/membership.parquet"
        if primary.is_file():
            profiles = pd.read_parquet(destination / "family_mixtures.parquet")
            cross_profile_bootstrap(profiles, pd.read_parquet(primary), config, destination)
        r.finish_stage(destination, "evaluated", upstream_manifest_sha256=r.file_hash(path.parent / "manifest.json"))
        completed.append(destination.name)
    return r.finish_stage(target, "evaluated", experiments=completed, reduced_budget=config.smoke,
                          interpretation="exploratory ecological associations; previous held-out results informed design; intervals are not multiplicity-adjusted",
                          preparation_sha256=r.file_hash(source / "manifest.json"))
