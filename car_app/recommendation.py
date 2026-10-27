"""Deterministic historical listing search and frozen specification peers."""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from pathlib import Path

import joblib
import altair as alt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from car_app.data import COHORT, ROOT, Snapshot, Filters, filter_expression
from car_app.model_registry import digest, read_json
from car_app.requirements import HISTORY_FIELDS, EQUIPMENT

SORTS = {"Observed price": ("price", False), "Mileage": ("mileage", False),
         "Seating": ("seats", True), "Advertised horsepower": ("horsepower", True)}
METHOD_ORDER = ("gower", "hierarchical", "kmeans")
CATEGORY_DIMENSIONS = ("market_positioning", "performance_character", "size_segment", "body_style", "powertrain")
SIMILARITY_LABELS = {"category_cluster": "Category and cluster similarity (experimental)",
                     "category_specs": "Category and specification similarity", "direct_specs": "Direct specification similarity",
                     "broader_category_cluster": "Broader predicted categories and clusters (experimental)"}


@st.cache_data(max_entries=32, ttl=900, show_spinner=False)
def matching_profile_statistics(source_identity, filters, membership_path, membership_signature, _source):
    if _source.identity != source_identity:
        raise ValueError("Snapshot changed; refresh the dashboard")
    members = pl.scan_parquet(membership_path).select("vin", "profile_id")
    return (_source.catalog(filters).join(members, on="vin", how="inner", validate="1:1")
        .group_by("profile_id").agg(pl.len().alias("matching_listings"),
            pl.col("cohort_complete").sum().alias("complete_matching_listings"),
            pl.col("price").median().alias("filtered_median_price"),
            pl.col("mileage").median().alias("filtered_median_mileage"))
        .collect().to_pandas())


def view_profiles(profiles, view):
    if view == "global":
        return profiles
    if "__" not in view:
        return profiles[profiles.body_family.eq(view)]
    dimension, label, body = view.split("__")
    values = profiles.get(dimension + "_labels", pd.Series("", index=profiles.index)).fillna("")
    return profiles[profiles.body_family.eq(body) & values.map(lambda value: label in str(value).split(";"))]


def cluster_population(catalog, profiles, fold, view, method):
    selected = view_profiles(profiles, view)
    rows = catalog.cluster_rows(selected.profile_id.tolist(), (fold, view, method))
    extra = [c for c in selected.columns if c not in rows.columns]
    rows = rows.merge(selected[["profile_id", *extra]], on="profile_id", how="inner", validate="1:1")
    if catalog.hierarchy_receipt:
        displayed = selected.set_index("profile_id")
        for column in [c for c in selected if c.endswith("_labels")]:
            rows[column] = rows.profile_id.map(displayed[column])
    assigned = rows.cluster.fillna(-1).ge(0)
    rows["cluster_label"] = rows.description.fillna("Unsupported subdivision")
    duplicate_names = rows.loc[assigned, ["cluster", "description"]].drop_duplicates().description.duplicated(keep=False)
    repeated = set(rows.loc[assigned, ["cluster", "description"]].drop_duplicates().loc[duplicate_names, "description"])
    rows.loc[assigned & rows.description.isin(repeated), "cluster_label"] = rows.loc[assigned & rows.description.isin(repeated)].apply(
        lambda row: str(row.description) + " · subgroup " + chr(65 + int(row.cluster)), axis=1)
    if "matching_listings" not in rows:
        rows["matching_listings"] = rows.vin_count
    return rows


def search_candidates(source: Snapshot, requirements, revision, sort_by="Observed price", catalog=None):
    if requirements.get("snapshot_identity") != source.identity or requirements.get("revision") != revision:
        raise ValueError("Requirements are stale; review and submit them again")
    if sort_by not in SORTS:
        raise ValueError("Unsupported sort")
    hard = requirements.get("hard_requirements", {})
    allowed = {"max_price_usd", "condition", "body_types", "min_seats", "max_mileage", "min_year", "max_year",
               "makes", "fuels", "drivetrains", "transmissions", "min_horsepower", "required_equipment",
               "require_reported_clean_history"}
    if set(hard) - allowed:
        raise ValueError("Unsupported hard requirement")
    if hard.get("max_price_usd") is None:
        raise ValueError("Review a maximum observed-price budget before searching")
    frame = source.catalog().with_columns(
        pl.col("horsepower").cast(pl.Float64, strict=False),
        pl.col("price").cast(pl.Float64, strict=False),
        pl.col("mileage").cast(pl.Float64, strict=False))
    predicates = [("positive observed price", pl.col("price").is_finite() & (pl.col("price") > 0))]
    for key, column, operator in [("max_price_usd", "price", "max"), ("max_mileage", "mileage", "max"),
                                  ("min_year", "ModelYear", "min"), ("max_year", "ModelYear", "max"),
                                  ("min_seats", "seats", "min"), ("min_horsepower", "horsepower", "min")]:
        if key in hard:
            value = hard[key]
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError("Invalid numeric requirement")
            expression = pl.col(column) <= value if operator == "max" else pl.col(column) >= value
            expression &= pl.col(column) >= 0 if column == "mileage" else pl.col(column) > 0
            predicates.append((key, expression & pl.col(column).is_finite()))
    for key, column in [("makes", "Make"), ("body_types", "body_type"), ("fuels", "fuel_type"),
                        ("drivetrains", "wheel_system"), ("transmissions", "transmission")]:
        if hard.get(key):
            predicates.append((key, pl.col(column).is_in(hard[key])))
    if hard.get("condition") in ("New", "Used"):
        predicates.append(("condition", pl.col("is_new").str.to_lowercase() == ("true" if hard["condition"] == "New" else "false")))
    if hard.get("require_reported_clean_history"):
        for field in HISTORY_FIELDS:
            predicates.append((field, pl.col(field).cast(pl.String).str.to_lowercase().eq("false")))
    for equipment in hard.get("required_equipment", []):
        if equipment not in EQUIPMENT:
            raise ValueError("Unsupported equipment requirement")
        options = pl.col("major_options").str.strip_chars()
        predicates.append((equipment, options.str.starts_with("[") & options.str.ends_with("]")
                           & options.str.contains("['\"]" + re.escape(equipment) + "['\"]")))
    preferences = {d: values for d, values in requirements.get("category_preferences", {}).items() if values}
    if preferences:
        if catalog is None or catalog.manifest.get("schema_version") not in ("4.0.0", "5.0.0"):
            raise ValueError("Category preferences require an activated v4 or v5 catalog")
        if requirements.get("category_mode", "prefer") not in ("prefer", "strict"):
            raise ValueError("Unsupported category matching mode")
        choices = catalog.category_choices()
        if any(d not in choices or set(values) - set(choices[d]) for d, values in preferences.items()):
            raise ValueError("Category selections are stale; review the requirements again")
        frame = frame.join(catalog.listing_categories(strict=requirements.get("category_mode") == "strict"), on="vin", how="left", validate="1:1")
        matches = [pl.col(d + "_labels").fill_null("").str.split(";").list.eval(pl.element().is_in(values)).list.any()
                   for d, values in preferences.items()]
        frame = frame.with_columns((pl.sum_horizontal([m.cast(pl.Float64) for m in matches]) / len(matches)).alias("category_match"))
        if requirements.get("category_mode") == "strict":
            predicates.append(("source-supported categories (strict)", pl.all_horizontal(matches)))
    expressions, counts = [pl.len().alias("all")], []
    cumulative = pl.lit(True)
    for index, (label, predicate) in enumerate(predicates):
        cumulative &= predicate.fill_null(False)
        expressions.append(cumulative.sum().alias(str(index)))
        counts.append(label)
    totals = frame.select(expressions).collect().row(0, named=True)
    exclusions, previous = [], totals["all"]
    for index, label in enumerate(counts):
        remaining = totals[str(index)]
        exclusions.append({"requirement": label, "excluded_at_step": previous - remaining, "remaining": remaining})
        previous = remaining
    eligible = frame.filter(cumulative)
    column, descending = SORTS[sort_by]
    order = [column, "price", "listing_id", "vin"] if column != "price" else ["price", "listing_id", "vin"]
    directions = [descending] + [False] * (len(order) - 1)
    if preferences:
        order, directions = ["category_match", *order], [True, *directions]
    ordered = eligible.sort(order, descending=directions, nulls_last=True)
    leaders = ordered.unique(subset=COHORT[:3], keep="first", maintain_order=True).head(10).collect()
    selected = leaders
    if selected.height < 10:
        rest = ordered.filter(~pl.col("vin").is_in(selected["vin"].to_list())).head(10-selected.height).collect()
        selected = pl.concat([selected, rest])
    if preferences:
        selected = selected.sort(order, descending=directions, nulls_last=True)
    summary = eligible.filter(pl.col("cohort_complete")).group_by(COHORT).agg(
        pl.len().alias("qualifying_listings"), pl.col("price").median().alias("median_price"),
        pl.col("price").quantile(.25).alias("price_q25"), pl.col("price").quantile(.75).alias("price_q75"),
        pl.col("mileage").median().alias("median_mileage"),
        pl.col("mileage").quantile(.25).alias("mileage_q25"), pl.col("mileage").quantile(.75).alias("mileage_q75"),
        pl.col(column).median().alias("sort_value"),
        *([pl.col("category_match").mean().alias("category_match")] if preferences else []))
    summary = summary.sort((["category_match"] if preferences else []) + ["sort_value", *COHORT],
        descending=([True] if preferences else []) + [descending, False, False, False, False], nulls_last=True)
    cohorts = summary.unique(subset=COHORT[:3], keep="first", maintain_order=True).head(5).collect()
    if cohorts.height < 5:
        remaining = summary.join(cohorts.select(COHORT).lazy(), on=COHORT, how="anti").head(5-cohorts.height).collect()
        cohorts = pl.concat([cohorts, remaining])
    cohorts = cohorts.with_columns((pl.col("qualifying_listings") < 5).alias("low_support"))
    if preferences:
        cohorts = cohorts.sort(["category_match", "sort_value", *COHORT], descending=[True, descending, False, False, False, False], nulls_last=True)
    if selected["vin"].n_unique() != selected.height:
        raise ValueError("Duplicate listing identities")
    return {"snapshot_identity": source.identity, "revision": revision, "sort_by": sort_by,
            "listings": selected, "cohorts": cohorts, "eligible_listings": previous,
            "exclusions": pd.DataFrame(exclusions), "source": str(source.directory),
            "category_preferences": preferences, "category_mode": requirements.get("category_mode", "prefer"),
            "unscored_priorities": requirements.get("soft_priorities", [])}


def matching_similarity_profiles(cohorts: pd.DataFrame, profiles: pd.DataFrame):
    """Match complete decoded cohorts without changing saved identities or labels."""
    def keys(frame):
        result = frame[COHORT].copy()
        for column in ("Make", "Model", "Trim"):
            result[column] = result[column].astype("string").str.strip().str.casefold()
        result["ModelYear"] = pd.to_numeric(result["ModelYear"], errors="coerce")
        return result

    reference = keys(cohorts)
    complete = reference.notna().all(axis=1) & reference[["Make", "Model", "Trim"]].ne("").all(axis=1)
    reference = reference.loc[complete].drop_duplicates()
    profile_keys = pd.MultiIndex.from_frame(keys(profiles))
    reference_keys = pd.MultiIndex.from_frame(reference)
    return profiles.loc[profile_keys.isin(reference_keys)].copy()


@dataclass
class SimilarityCatalog:
    path: Path
    manifest: dict
    profiles: pd.DataFrame
    peers: pd.DataFrame
    statuses: pd.DataFrame
    members: pd.DataFrame
    compositions: pd.DataFrame
    experiments: pd.DataFrame
    fields: dict
    labels: pd.DataFrame | None = None
    category_coverage: pd.DataFrame | None = None
    category_details: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    hierarchy_receipt: dict | None = None

    def filtered_profiles(self, source, filters=Filters()):
        from car_app.artifacts import stat_key
        path = self.membership_path()
        counts = matching_profile_statistics(source.identity, filters, str(path), stat_key(path), _source=source)
        result = self.profiles.merge(counts, on="profile_id", how="inner", validate="1:1")
        result = result.sort_values(["matching_listings", "profile_id"], ascending=[False, True]).reset_index(drop=True)
        result.attrs["scope_identity"] = source.identity + filters.identity()
        return result

    def approaches(self):
        available = set(self.peers.approach) if "approach" in self.peers else set()
        if "approach" in self.statuses:
            available |= set(self.statuses.approach)
        if self.manifest.get("schema_version") == "5.0.0":
            from car_app.model_registry import registry
            config = registry()
            if self.predictions is None or self.predictions.empty:
                available.discard("broader_category_cluster")
            selected = config.get("preferred_similarity_approaches", {}).get(self.manifest.get("grain"), "direct_specs")
            return [selected, *[a for a in SIMILARITY_LABELS if a != selected and a in available]]
        return [a for a in SIMILARITY_LABELS if a in available] or ["direct_specs"]

    def category_choices(self):
        if self.manifest.get("schema_version") not in ("4.0.0", "5.0.0") or self.labels is None:
            return {}
        dimensions=CATEGORY_DIMENSIONS
        return {d: sorted(self.labels.loc[self.labels.dimension.eq(d) & self.labels.profile_id.isin(self.profiles.profile_id), "label"].unique()) for d in dimensions}

    def listing_categories(self, strict=False):
        dimensions=tuple(self.category_choices()) or CATEGORY_DIMENSIONS
        columns = ["profile_id", "category_summary", *[d + "_labels" for d in dimensions]]
        memberships = pl.scan_parquet(self.membership_path()).select("vin", "profile_id")
        profiles = self.profiles[columns].copy()
        if self.manifest.get("schema_version") == "5.0.0":
            permitted = self.labels[self.labels.status.isin(["observed", "source_supported"])] if strict else self.labels
            for dimension in dimensions:
                values = permitted[permitted.dimension.eq(dimension)].groupby("profile_id").label.agg(lambda x: ";".join(sorted(set(x))))
                profiles[dimension + "_labels"] = profiles.profile_id.map(values).fillna("")
                if not strict and self.predictions is not None and not self.predictions.empty:
                    prediction = self.predictions[self.predictions.policy.eq("conservative") & self.predictions.dimension.eq(dimension)].groupby("profile_id").label.agg(lambda x: ";".join(sorted(set(x))))
                    profiles[dimension + "_labels"] = profiles[dimension + "_labels"].mask(profiles[dimension + "_labels"].eq(""), profiles.profile_id.map(prediction)).fillna("")
        return memberships.join(pl.from_pandas(profiles).lazy(), on="profile_id", how="left", validate="m:1")

    def context(self, profile_id, view=None, method=None, training_fold=None):
        query = self.profiles.set_index("profile_id").loc[profile_id]
        fold = int(query.fold) if training_fold is None else int(training_fold)
        if view is None and self.manifest.get("schema_version") == "4.0.0":
            available = self.members[self.members.training_fold.eq(fold) & self.members.profile_id.eq(profile_id) & self.members.cluster.ge(0)]
            candidates = [f"{d}__{label}__{query.body_family}" for d in CATEGORY_DIMENSIONS[:3]
                          for label in str(query.get(d + "_labels", "")).split(";") if label]
            view = next((candidate for candidate in candidates if candidate in set(available.view)), None)
        view = view or query.body_family
        available = self.members[self.members.training_fold.eq(fold) & self.members.view.eq(view)]
        methods = set(available.method)
        selected = method or next((m for m in METHOD_ORDER if m in methods), None)
        return fold, view, selected

    def cluster_rows(self, ids, context):
        fold, view, method = context
        result = self.profiles[self.profiles.profile_id.isin(ids)].copy()
        members = self.members[self.members.training_fold.eq(fold) & self.members.view.eq(view)
                               & self.members.method.eq(method)]
        member_columns = ["profile_id", "cluster"] + (["assignment_status"] if "assignment_status" in members else [])
        result = result.merge(members[member_columns], on="profile_id", how="left", validate="1:1")
        descriptions = self.compositions[self.compositions.fold.eq(fold) & self.compositions.view.eq(view)
                                         & self.compositions.method.eq(method)][["cluster", "description"]]
        result = result.merge(descriptions, on="cluster", how="left", validate="m:1")
        result["cluster_status"] = np.where(result.cluster.ge(0), "assigned", "unsupported")
        if self.manifest.get("schema_version") == "5.0.0":
            fallback = "no_supported_subdivision" if method is None else "insufficient_observations"
            if "assignment_status" in result:
                result["cluster_status"] = result.assignment_status.fillna(fallback)
            else:
                result["cluster_status"] = np.where(result.cluster.ge(0), "assigned", fallback)
        result["cluster_context"] = f"fold {fold} / {view} / {method or 'no accepted model'}"
        return result

    def similar(self, profile_id, source_identity, revision, approach="direct_specs"):
        if profile_id not in set(self.profiles.profile_id):
            raise ValueError("Unknown reference profile")
        if approach not in self.approaches():
            raise ValueError("Similarity approach unavailable in this saved run")
        context = self.context(profile_id)
        peers = self.peers[self.peers.query_profile_id.eq(profile_id) & self.peers.approach.eq(approach)].sort_values("rank")
        rows = self.cluster_rows([profile_id, *peers.peer_profile_id], context).set_index("profile_id")
        query = rows.loc[profile_id]
        results = peers.merge(rows.reset_index(), left_on="peer_profile_id", right_on="profile_id", validate="1:1")
        comparable = results.cluster.ge(0) & pd.notna(query.cluster) & (query.cluster >= 0)
        results["shared_cluster"] = np.where(comparable, results.cluster.eq(query.cluster), None)
        evidence = []
        for peer in results.to_dict("records"):
            for field in self.fields[int(query.fold)]:
                supported = bool(pd.notna(query[field]) and pd.notna(peer[field]))
                if self.manifest.get("schema_version")=="5.0.0":
                    supported &= query.get(field+"_n",0)>=5 and peer.get(field+"_n",0)>=5
                evidence.append({"peer_profile_id": peer["peer_profile_id"], "field": field,
                    "reference": str(query[field]) if pd.notna(query[field]) else None,
                    "alternative": str(peer[field]) if pd.notna(peer[field]) else None,
                    "shared_evidence": supported})
        saved_status = self.statuses[self.statuses.query_profile_id.eq(profile_id) & self.statuses.approach.eq(approach)] if {"query_profile_id", "approach"}.issubset(self.statuses.columns) else pd.DataFrame()
        status = saved_status.iloc[0].to_dict() if not saved_status.empty else {"status": "unavailable", "slate_size": len(results)}
        return {"snapshot_identity": source_identity, "revision": revision, "profile_id": profile_id,
                "reference": query.to_frame().T, "alternatives": results,
                "evidence": pd.DataFrame(evidence), "status": status, "cluster_context": context, "approach": approach}

    def membership_path(self):
        if self.manifest.get("schema_version") in ("4.0.0", "5.0.0"):
            return self.path / "vin_profile_membership.parquet"
        return Path(self.manifest["prepared_directory"]) / "vin_profile_membership.parquet"

    def market_rows(self, source, ids, reference_id=None, reference_filters=None):
        members = pl.scan_parquet(self.membership_path()).select("vin", "profile_id").filter(pl.col("profile_id").is_in(ids))
        listings = source.catalog().join(members, on="vin", how="inner", validate="1:1")
        if reference_id is not None and reference_filters is not None:
            listings = listings.filter(pl.col("profile_id").ne(reference_id) | filter_expression(reference_filters))
        summary = listings.group_by("profile_id").agg(pl.len().alias("observed_listings"),
            pl.col("price").median().alias("median_price"), pl.col("mileage").median().alias("median_mileage"),
            pl.col("price").quantile(.25).alias("price_q25"), pl.col("price").quantile(.75).alias("price_q75")).collect()
        labels = pl.from_pandas(self.profiles[self.profiles.profile_id.isin(ids)][["profile_id", *COHORT]])
        summary = labels.join(summary, on="profile_id", how="left", validate="1:1").select(
            *COHORT, "observed_listings", "median_price", "price_q25", "price_q75", "median_mileage", "profile_id")
        examples = listings.sort(["price", "vin"]).unique(subset="profile_id", keep="first", maintain_order=True).collect()
        return summary, examples


@st.cache_resource(max_entries=2, show_spinner=False)
def _load_similarity(run, manifest_signature, source_directory, source_identity=None, grain="cohort"):
    from MODELS.vehicle_taxonomy import verify_artifacts
    path = ROOT / "MODEL_OUTPUT" / run
    manifest = verify_artifacts(path)
    if manifest.get("status") != "completed":
        raise ValueError("Similarity requires a completed saved bundle")
    if manifest.get("schema_version") == "5.0.0" and "grains" in manifest:
        if grain not in ("cohort", "model") or manifest.get("reduced_budget", True):
            raise ValueError("A completed full-budget v5 discovery grain is required")
        path = path / grain
        manifest = verify_artifacts(path)
    if manifest.get("status") != "completed" or manifest.get("reduced_budget", True) or manifest.get("schema_version") not in ("3.0.0", "4.0.0", "5.0.0"):
        raise ValueError("Similarity requires a completed full-budget v3, v4 or v5 run")
    if manifest.get("schema_version") in ("4.0.0", "5.0.0"):
        from EDA.DATA_CLEANING import EXCLUDED_LISTING_VINS
        if manifest.get("excluded_vins") != sorted(EXCLUDED_LISTING_VINS):
            raise ValueError("Category catalog uses a different listing exclusion policy")
    for name, expected in manifest["source_hashes"].items():
        if digest(Path(source_directory) / name) != expected:
            raise ValueError(f"Similarity source changed: {name}")
    prepared = Path(manifest["prepared_directory"])
    if digest(prepared / "manifest.json") != manifest["prepared_manifest_sha256"]:
        raise ValueError("Prepared membership receipt changed")
    receipt = read_json(prepared / "manifest.json")
    member_name = "vin_profile_membership.parquet"
    if manifest.get("schema_version") == "5.0.0":
        member_name = manifest["grain"] + "/" + member_name
    if digest(prepared / member_name) != receipt["artifacts"][member_name]:
        raise ValueError("Prepared VIN memberships changed")
    profiles = pd.read_parquet(path / "variant_profiles.parquet")
    predictions = None
    if manifest.get("schema_version") == "5.0.0":
        predictions = pd.read_parquet(path / "predictions.parquet")
        if not predictions.empty:
            folds = profiles.set_index("profile_id").fold
            predictions = predictions[predictions.outer_fold.eq(predictions.profile_id.map(folds))]
    fields = {int(fold): joblib.load(path / f"fold_{fold}_retrieval.joblib")["fields"] for fold in profiles.fold.unique()}
    return SimilarityCatalog(path, manifest, profiles, pd.read_parquet(path / "profile_peers.parquet"),
        pd.read_parquet(path / "query_status.parquet"), pd.read_parquet(path / "cluster_memberships.parquet"),
        pd.read_csv(path / "cluster_profiles.csv"), pd.read_csv(path / "cluster_experiments.csv"), fields,
        pd.read_parquet(path / "profile_labels.parquet") if manifest.get("schema_version") in ("4.0.0", "5.0.0") else None,
        pd.read_csv(path / "category_coverage.csv") if manifest.get("schema_version") in ("4.0.0", "5.0.0") else None,
        pd.read_parquet(path / "category_coverage.parquet") if manifest.get("schema_version") in ("4.0.0", "5.0.0") else None,
        predictions)


@st.cache_resource(max_entries=4, show_spinner=False)
def load_category_hierarchy(run, signature, base_signature, grain, _catalog):
    from MODELS.vehicle_taxonomy import verify_artifacts
    from MODELS.vehicle_category_hierarchy import hierarchy_rows
    path = ROOT / "MODEL_OUTPUT" / run
    receipt = verify_artifacts(path)
    if receipt.get("status") != "completed" or receipt.get("schema_version") != "epa-hierarchy-1" or receipt.get("base_manifest_sha256") != base_signature:
        raise ValueError("Category hierarchy does not match the active discovery run")
    labels = pd.read_parquet(path / grain / "profile_labels.parquet")
    predictions = pd.read_parquet(path / grain / "predictions.parquet")
    profiles = _catalog.profiles.copy()
    hierarchy = hierarchy_rows(profiles, labels, predictions, include_predictions=False).set_index("profile_id")
    for dimension in ("epa_group", "common_segment"):
        profiles[dimension + "_labels"] = profiles.profile_id.map(hierarchy[dimension]).fillna("")
    profiles["category_summary"] = profiles.profile_id.map(hierarchy.hierarchy_path)
    return replace(_catalog, profiles=profiles, labels=labels, predictions=predictions,
        hierarchy_receipt={**receipt, "directory": str(path), "metrics": str(path / grain / "classifier_metrics.csv")})


def load_similarity(run, manifest_signature, source_directory, source_identity=None, grain="cohort"):
    from car_app.model_registry import registry
    config = registry()
    catalog = _load_similarity(run, manifest_signature, source_directory, source_identity, grain)
    overlay = config.get("category_hierarchy_run")
    if overlay and run == config.get("similarity_run"):
        path = ROOT / "MODEL_OUTPUT" / overlay / "manifest.json"
        signature = digest(path)
        if signature != config.get("category_hierarchy_manifest_sha256"):
            raise ValueError("Pinned category hierarchy receipt changed")
        catalog = load_category_hierarchy(overlay, signature, config["similarity_manifest_sha256"], grain, catalog)
    return catalog


def select_discovery_grain(run, key):
    path = ROOT / "MODEL_OUTPUT" / run / "manifest.json"
    if not path.exists():
        return "cohort"
    manifest = read_json(path)
    if manifest.get("schema_version") != "5.0.0":
        return "cohort"
    grain = st.selectbox("Discovery population", ["cohort", "model"], format_func=lambda v: "Trim-level cohorts" if v == "cohort" else "Model-level discovery (across trims)", key=key)
    st.caption("Each population has separate profiles, fits and listing counts. Model-level discovery includes vehicles with missing decoded trim.")
    return grain


def active_similarity(source):
    from car_app.model_registry import registry
    from car_app.artifacts import stat_key
    config = registry()
    run = config["similarity_run"]
    path = ROOT / "MODEL_OUTPUT" / run / "manifest.json"
    if config.get("similarity_manifest_sha256") and digest(path) != config["similarity_manifest_sha256"]:
        raise ValueError("Pinned similarity receipt changed")
    return load_similarity(run, stat_key(path), str(source.directory), source.identity, select_discovery_grain(run, "dashboard_discovery_grain"))


def _choice(label, options, key, default=None, format_func=str):
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    return st.selectbox(label, options, index=options.index(default) if default in options else 0, key=key, format_func=format_func)


def cluster_explorer(catalog, profiles, key="clusters", reference_id=None, preferred_view=None, color_dimension="market_positioning"):
    st.subheader("Filtered specification clusters")
    if profiles.empty:
        st.info("No saved profiles match these filters.")
        return
    scope = profiles.attrs.get("scope_identity", "") + str(pd.util.hash_pandas_object(
        profiles[[c for c in ("profile_id", "matching_listings") if c in profiles]], index=False).sum()) + str(reference_id)
    if st.session_state.get(key + "_scope") != scope:
        for suffix in ("_fold", "_view", "_method", "_group"):
            st.session_state.pop(key + suffix, None)
        st.session_state[key + "_scope"] = scope
    reference = reference_id or profiles.iloc[0].profile_id
    default_fold, default_view, _ = catalog.context(reference)
    if profiles.body_family.nunique() > 1:
        default_view = profiles.groupby("body_family").matching_listings.sum().idxmax() if catalog.manifest.get("schema_version")=="5.0.0" and "matching_listings" in profiles else "global"
    folds = sorted(catalog.experiments.fold.unique()) if "fold" in catalog.experiments else sorted(catalog.profiles.fold.unique())
    fold = _choice("Fitted model fold", folds, key + "_fold", default_fold)
    views = {"global", *profiles.body_family.unique()}
    for dimension in CATEGORY_DIMENSIONS[:3]:
        for body, group in profiles.groupby("body_family"):
            values = group.get(dimension + "_labels", pd.Series("", index=group.index)).fillna("")
            for label in {label for value in values for label in str(value).split(";") if label}:
                views.add(f"{dimension}__{label}__{body}")
    views = sorted(views, key=lambda value: (value != "global", value))
    pretty_view = lambda value: value.replace("market_positioning__", "").replace("performance_character__", "").replace("size_segment__", "").replace("__", " · ").replace("_", " ").capitalize()
    view = _choice("Cluster view", views, key + "_view", preferred_view or default_view, pretty_view)
    memberships = catalog.members[catalog.members.training_fold.eq(fold) & catalog.members.view.eq(view)]
    methods = [m for m in METHOD_ORDER if m in set(memberships.method)]
    method = _choice("Accepted method", methods, key + "_method") if methods else None
    if method is None:
        st.info("No supported subdivision for this category and fitted fold. Matching vehicles remain visible as unsupported.")
    rows = cluster_population(catalog, profiles, fold, view, method)
    if rows.empty:
        st.info("No matching profiles belong to this cluster view.")
        return
    options = ["All", *sorted(rows.cluster_label.unique())]
    selected = _choice("Inspect group", options, key + "_group", "All")
    if selected != "All":
        rows = rows[rows.cluster_label.eq(selected)]
    counts = rows.groupby("cluster_label", dropna=False).agg(profiles=("profile_id", "size"), matching_listings=("matching_listings", "sum")).reset_index()
    st.altair_chart(alt.Chart(counts).mark_bar().encode(
        x=alt.X("matching_listings:Q", title="Matching listings"),
        y=alt.Y("cluster_label:N", title="Saved subgroup", sort="-x"),
        color=alt.Color("cluster_label:N", legend=None), tooltip=["cluster_label", "profiles", "matching_listings"]),
        alt="Saved cluster composition for listings matching the dashboard and category filters")
    coordinates = [f for f in catalog.fields[int(fold)] if f in rows and f not in ("fuel_type", "transmission", "wheel_system") and pd.to_numeric(rows[f], errors="coerce").notna().any()]
    if len(coordinates) >= 2:
        x = _choice("Horizontal specification", coordinates, key + "_x", "wheelbase")
        y = _choice("Vertical specification", coordinates, key + "_y", "height" if "height" in coordinates else next(f for f in coordinates if f != x))
        plotted = rows.dropna(subset=[x, y]).copy()
        total = len(plotted)
        if total > 3000:
            plotted = plotted.sort_values("profile_id").sample(3000, random_state=42)
            if reference_id in set(rows.profile_id) and reference_id not in set(plotted.profile_id):
                reference_row = rows[rows.profile_id.eq(reference_id)].dropna(subset=[x, y])
                if not reference_row.empty:
                    plotted = pd.concat([plotted.head(2999), reference_row], ignore_index=True)
        plotted["role"] = np.where(plotted.profile_id.eq(reference_id), "Reference", "Filtered profile")
        label_column = color_dimension + "_labels"
        plotted["vehicle_category"] = plotted[label_column].fillna("").replace("", "Unresolved").str.replace("_", " ").str.replace(";", " / ") if label_column in plotted else "Unresolved"
        color_by = _choice("Diagram colors", ["Saved subgroup", "Vehicle category"], key + "_color", "Vehicle category" if catalog.hierarchy_receipt else "Saved subgroup")
        color_field = "vehicle_category" if color_by == "Vehicle category" else "cluster_label"
        columns = list(dict.fromkeys([*COHORT, "profile_id", "cluster_label", "vehicle_category", "matching_listings", "role", x, y]))
        encodings = {"x": alt.X(f"{x}:Q", title=x.replace("_", " ")),
            "y": alt.Y(f"{y}:Q", title=y.replace("_", " ")),
            "color": alt.Color(f"{color_field}:N", title=color_dimension.replace("_", " ").capitalize() if color_by == "Vehicle category" else "Saved subgroup"),
            "tooltip": [*COHORT, "vehicle_category", "cluster_label", "matching_listings", "role", x, y]}
        if reference_id is not None and plotted.profile_id.eq(reference_id).any():
            encodings["shape"] = alt.Shape("role:N", scale=alt.Scale(domain=["Filtered profile", "Reference"], range=["circle", "diamond"]))
        chart = alt.Chart(plotted[columns]).mark_point(filled=True, size=55, opacity=.65).encode(**encodings).properties(height=380).interactive()
        st.altair_chart(chart, alt="Filtered vehicle categories and saved subgroups within one fitted context")
        st.caption(f"{len(plotted):,} of {total:,} plottable profiles shown; {len(rows) - total:,} lack selected coordinates. Coordinates are saved profile medians. Categories are source-supported labels; subgroups are saved statistical fits. Filters never refit clusters.")
    else:
        st.info("Not enough observed numeric specifications for a two-axis diagram in this selection.")
    with st.expander("Filtered profile values and fitted model checks"):
        columns = [*COHORT, "category_summary", "body_style", "powertrain", "matching_listings", "filtered_median_price", "filtered_median_mileage", "cluster_label", "cluster_context", "profile_id"]
        st.dataframe(rows[[c for c in columns if c in rows]].head(1000), hide_index=True, width="stretch", alt="Filtered VIN-supported profile values and cluster context")
        checks = catalog.experiments[catalog.experiments.fold.eq(fold) & catalog.experiments.view.eq(view)]
        st.dataframe(checks, hide_index=True, width="stretch", alt="Saved support and stability checks for the selected fitted context")


def category_explorer(catalog, key="categories", profiles=None, reference_id=None):
    st.subheader("Vehicle categories")
    if catalog is None or not catalog.category_choices():
        st.info("Activate a completed v4 or v5 run to explore vehicle categories. The existing similarity view remains available.")
        return
    identity = str(catalog.path) + str(catalog.manifest.get("completed_at")) + str(catalog.hierarchy_receipt)
    if st.session_state.get(key + "_artifact") != identity:
        for suffix in ("_dimension", "_labels", "_unknown", "_clusters_color", "_subsegment", "_policy"):
            st.session_state.pop(key + suffix, None)
        st.session_state[key + "_artifact"] = identity
    base = catalog.profiles if profiles is None else profiles
    if base.empty:
        st.info("No vehicle profiles match the current dashboard filters.")
        return
    st.caption(f"Active saved run: {catalog.path}. Sidebar filters update matching VINs, counts and diagrams automatically. Categories remain provisional and may overlap.")
    if catalog.hierarchy_receipt:
        from MODELS.vehicle_category_hierarchy import hierarchy_rows
        policy = st.selectbox("Category evidence", ["source_trained", "verified"],
            format_func=lambda value: "Recorded evidence and learned categories (experimental)" if value == "source_trained" else "Recorded evidence only",
            key=key + "_policy")
        base = base.copy()
        hierarchy = hierarchy_rows(base, catalog.labels, catalog.predictions, include_predictions=policy == "source_trained").set_index("profile_id")
        for dim in ("epa_group", "common_segment", "size_segment", "market_positioning"):
            base[dim + "_labels"] = base.profile_id.map(hierarchy[dim]).fillna("")
        for field in ("epa_group_state", "common_segment_state", "common_segment_source", "hierarchy_path"):
            base[field] = base.profile_id.map(hierarchy[field])
        base["category_summary"] = base.hierarchy_path
        st.caption("EPA size classes are the main groups; common market terms are subsegments. Drive suffixes are combined within each EPA size class. The specification diagrams show saved fits filtered to these groups.")
        st.caption("Learned labels exclude each query's entire vehicle family. An automated check requires at least 80% source agreement across three or more development families per emitted label. Probabilities are uncalibrated; human review and peer review are optional. Table examples have unverified year scope.")
        with st.expander("Learned category probabilities and source agreement"):
            st.dataframe(catalog.predictions[catalog.predictions.profile_id.isin(base.profile_id)], hide_index=True, width="stretch", alt="Experimental source-trained category probabilities and model provenance")
            st.dataframe(pd.read_csv(catalog.hierarchy_receipt["metrics"]), hide_index=True, width="stretch", alt="Held-out family agreement with source labels, not independently reviewed accuracy")
    elif catalog.manifest.get("schema_version") == "5.0.0":
        population = "Model-level discovery" if catalog.manifest.get("grain") == "model" else "Trim-level cohorts"
        st.caption(f"Saved v5 clustering · {population}. Body-family fits and saved subgroup colors are the defaults; category labels annotate those groups.")
        base = base.copy()
        policies=["conservative","verified","broader"] if catalog.predictions is not None and not catalog.predictions.empty else ["verified"]
        if st.session_state.get(key+"_policy") not in policies:
            st.session_state.pop(key+"_policy",None)
        policy = st.selectbox("Category evidence", policies, format_func=lambda p: {"conservative": "Recorded evidence and conservative predictions", "verified": "Recorded evidence only (no predictions)", "broader": "Recorded evidence and broader predictions (experimental)"}[p], key=key + "_policy")
        if policies==["verified"]:
            st.caption("No validated predicted labels are saved in this bundle. Recorded classifications remain available.")
        if policy != "verified" and catalog.predictions is not None:
            for dim in CATEGORY_DIMENSIONS[:3]:
                selected_predictions = catalog.predictions[catalog.predictions.policy.eq(policy) & catalog.predictions.dimension.eq(dim)]
                values = selected_predictions.groupby("profile_id").label.agg(lambda x: ";".join(sorted(set(x))))
                base[dim + "_labels"] = base[dim + "_labels"].mask(base[dim + "_labels"].eq(""), base.profile_id.map(values)).fillna("")
            with st.expander("Predicted category support"):
                st.dataframe(catalog.predictions[catalog.predictions.policy.eq(policy) & catalog.predictions.profile_id.isin(base.profile_id)], hide_index=True, width="stretch", alt="Validated predicted categories, probabilities and model provenance")
        st.caption("Category coverage and statistical cluster assignment are separate. Predictions do not qualify for strict category filters.")
    dimensions = list(CATEGORY_DIMENSIONS) + (["drivetrain", "epa_class", "nhtsa_class"] if catalog.manifest.get("schema_version") == "5.0.0" else [])
    if catalog.hierarchy_receipt:
        dimensions = ["epa_group", "common_segment", *dimensions]
    dimension = st.selectbox("Category dimension", dimensions, index=0 if catalog.hierarchy_receipt else dimensions.index("body_style") if catalog.manifest.get("schema_version")=="5.0.0" else 0,
        format_func=lambda v:{"epa_group":"EPA size class (main groups)", "common_segment":"Common market subsegment", "epa_class":"EPA class (including drive)","nhtsa_class":"NHTSA NCAP class"}.get(v,v.replace("_", " ").capitalize()), key=key + "_dimension")
    if dimension=="drivetrain":
        st.caption("A profile may contain several drivetrain configurations. Missing or mixed evidence remains unresolved; 4×2 does not identify front- versus rear-wheel drive.")
    elif dimension=="nhtsa_class":
        st.caption("NCAP classes here are derived from observed body and supported curb-weight bounds. A class does not establish a safety rating or rating comparability.")
    elif dimension=="epa_class":
        st.caption("EPA classes use matched model-year and variant records and remain separate from market size segments.")
    choices = sorted({label for value in base[dimension + "_labels"].fillna("") for label in str(value).split(";") if label})
    if key + "_labels" in st.session_state:
        st.session_state[key + "_labels"] = [v for v in st.session_state[key + "_labels"] if v in choices]
    selected = st.multiselect("Category", choices, format_func=lambda v: v.replace("_", " ").capitalize(), key=key + "_labels")
    unresolved = st.checkbox("Show unresolved profiles only", key=key + "_unknown")
    data = base.copy()
    if "matching_listings" not in data:
        data["matching_listings"] = data.vin_count
    values = data[dimension + "_labels"].fillna("")
    if unresolved:
        data = data[values.eq("")]
    elif selected:
        data = data[values.map(lambda v: bool(set(v.split(";")) & set(selected)))]
    if catalog.hierarchy_receipt and dimension == "epa_group":
        segments = sorted(set(data.common_segment_labels) - {""})
        segment = _choice("Common market subsegment", ["All", *segments], key + "_subsegment", "All")
        if segment != "All":
            data = data[data.common_segment_labels.eq(segment)]
        summary = data.groupby(["epa_group_labels", "common_segment_labels"], dropna=False).agg(
            profiles=("profile_id", "size"), matching_listings=("matching_listings", "sum")).reset_index()
        summary["epa_group_labels"] = summary.epa_group_labels.replace("", "Unresolved EPA class")
        summary["common_segment_labels"] = summary.common_segment_labels.replace("", "Unresolved common segment")
        st.dataframe(summary, hide_index=True, width="stretch", alt="EPA main groups and common market subsegments with non-overlapping listing counts")
    if catalog.category_details is not None:
        reasons = catalog.category_details[catalog.category_details.dimension.eq(dimension)][["profile_id", "reason"]]
        data = data.merge(reasons, on="profile_id", how="left", validate="1:1")
    st.caption(f"{len(data):,} profiles · {data.family_block.nunique():,} vehicle families · {int(data.matching_listings.sum()):,} matching listings")
    counts = []
    for label in [*choices, "Unresolved"]:
        mask = data[dimension + "_labels"].fillna("").map(lambda value: label in str(value).split(";")) if label != "Unresolved" else data[dimension + "_labels"].fillna("").eq("")
        if mask.any():
            counts.append({"category": label.replace("_", " ").capitalize(), "matching_listings": int(data.loc[mask, "matching_listings"].sum()), "profiles": int(mask.sum())})
    if counts:
        st.altair_chart(alt.Chart(pd.DataFrame(counts)).mark_bar().encode(x=alt.X("matching_listings:Q", title="Matching listings"),
            y=alt.Y("category:N", title="Category", sort="-x"), tooltip=["category", "matching_listings", "profiles"]), alt="Category composition of the filtered listings; categories may overlap")
    preferred_view = f"{dimension}__{selected[0]}__{data.iloc[0].body_family}" if dimension in CATEGORY_DIMENSIONS[:3] and len(selected) == 1 and not data.empty and data.body_family.nunique() == 1 else None
    cluster_explorer(catalog, data, key + "_clusters", reference_id=reference_id, preferred_view=preferred_view, color_dimension=dimension)
    hierarchy_columns = ["epa_group_labels", "epa_group_state", "common_segment_labels", "common_segment_state", "common_segment_source"] if catalog.hierarchy_receipt else []
    columns = list(dict.fromkeys([*COHORT, "body_style", "powertrain",dimension+"_labels", *hierarchy_columns, "category_summary", "matching_listings", "profile_id", *(["reason"] if "reason" in data else [])]))
    st.dataframe(data[columns].head(1000), hide_index=True, width="stretch",column_config={"category_summary":"Market / performance / size"}, alt="Vehicle category examples and unresolved profiles; up to 1,000 rows")
    with st.expander("Category coverage and source evidence"):
        if catalog.category_details is not None:
            coverage = catalog.category_details.merge(data[["profile_id", "matching_listings"]], on="profile_id", how="inner", validate="m:1")
            coverage = coverage.groupby(["dimension", "reason"]).agg(profiles=("profile_id", "nunique"), families=("family_block", "nunique"), matching_listings=("matching_listings", "sum")).reset_index()
            st.dataframe(coverage, hide_index=True, width="stretch", alt="Category coverage and unresolved reasons for matching VIN-supported profiles")
        evidence = catalog.labels[catalog.labels.profile_id.isin(data.profile_id) & catalog.labels.dimension.eq(dimension)]
        st.dataframe(evidence.head(1000), hide_index=True, width="stretch", alt="Category evidence with source URLs and provisional status; up to 1,000 assignments")
