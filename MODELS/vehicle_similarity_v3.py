"""Family-separated vehicle discovery and blinded similarity evaluation.

Read a frozen v2 preparation; never rewrite source tables or historical runs.
Run ``python -m MODELS.vehicle_similarity_v3 --help`` for the three CLI stages.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import html
import importlib.metadata
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import polars as pl
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from MODELS import vehicle_profiles as r
from MODELS import vehicle_segmentation as s
from MODELS import vehicle_taxonomy as legacy
from MODELS import vehicle_taxonomy_v2 as v2

VERSION = "3.0.0"
APPROACHES = ("direct_specs", "cluster_aware")
METHODS = ("kmeans", "gower", "hierarchical")
PEER_COLUMNS = ["query_profile_id", "peer_profile_id", "approach", "rank", "distance",
                "ranking_distance", "size_distance", "powertrain_distance", "shared_size",
                "shared_powertrain", "cluster_penalty", "year_window", "evidence_scope"]


@dataclass(frozen=True)
class Config:
    seed: int = 42
    folds: int = 5
    queries: int = 100
    clusters: tuple = tuple(range(2, 9))
    screen_refits: int = 5
    stability_refits: int = 30
    starts: int = 5
    vehicle_cap: int = 10000
    medoid_cap: int = 2000
    anchor_cap: int = 1000
    minimum_profiles: int = 50
    minimum_families: int = 20
    minimum_cluster_families: int = 5
    median_ari: float = .70
    lower_ari: float = .60
    cluster_penalty: float = .10
    smoke: bool = False


def config_for(smoke=False):
    if smoke:
        return Config(queries=10, clusters=(2, 3), screen_refits=2, stability_refits=2,
                      starts=2, vehicle_cap=200, medoid_cap=100, anchor_cap=100, smoke=True)
    return Config()


def family_fold(family, folds=5):
    return int(r.digest("42|" + str(family))[:16], 16) % folds


def eligible(frame, fields):
    valid = (frame.identity_complete & frame.vin_count.ge(5)
             & ~frame.body_style.isin(["unknown", "conflicting"])
             & ~frame.powertrain.eq("conflicting"))
    for block, minimum in [("size", 2), ("powertrain", 1)]:
        selected = [f for f in fields if v2.feature_block(f) == block]
        valid &= frame[selected].notna().sum(axis=1).ge(max(minimum, int(np.ceil(len(selected) / 2)))) if selected else False
    return valid


def fit_metric(training, fields, scaling="quantile"):
    bounds = {}
    for field in fields:
        if field not in v2.CATEGORICAL:
            values = training[field].dropna()
            lo, hi = values.quantile([.01, .99]) if scaling == "quantile" else (values.min(), values.max())
            bounds[field] = (float(lo), float(hi))
    return {"fields": list(fields), "bounds": bounds, "scaling": scaling,
            "blocks": {block: [f for f in fields if v2.feature_block(f) == block]
                       for block in ["size", "powertrain"]}}


def distances(left, right, metric, details=False):
    """Normalize observed fields inside each block before giving blocks equal weight."""
    shape = (len(left), len(right))
    components, counts = {}, {}
    valid = np.ones(shape, dtype=bool)
    for block, fields in metric["blocks"].items():
        total, count = np.zeros(shape), np.zeros(shape, dtype=np.int16)
        for field in fields:
            a, b = left[field].to_numpy(), right[field].to_numpy()
            observed = pd.notna(a)[:, None] & pd.notna(b)[None, :]
            if field in v2.CATEGORICAL:
                difference = a[:, None] != b[None, :]
            else:
                lo, hi = metric["bounds"][field]
                a, b = np.clip(a.astype(float), lo, hi), np.clip(b.astype(float), lo, hi)
                difference = np.minimum(np.abs(a[:, None] - b[None, :]) / max(hi - lo, 1e-9), 1)
            total += np.where(observed, difference, 0)
            count += observed
        components[block] = np.divide(total, count, out=np.full(shape, np.nan), where=count > 0)
        counts[block] = count
        valid &= count >= max(2 if block == "size" else 1, int(np.ceil(len(fields) / 2)))
    distance = np.where(valid, (components["size"] + components["powertrain"]) / 2, np.nan)
    if details:
        return distance, components, counts
    return distance


def clip_frame(frame, metric):
    result = frame[metric["fields"]].copy()
    for field, (lo, hi) in metric["bounds"].items():
        result[field] = result[field].clip(lo, hi)
    return result


def family_weights(frame):
    key = "_bootstrap_block" if "_bootstrap_block" in frame else "family_block"
    weights = 1 / frame.groupby(key)[key].transform("size").to_numpy(float)
    return weights * len(weights) / weights.sum()


def balanced_variants(frame, cap, seed):
    """Rotate up to five observed variants per family, rather than collapsing to one."""
    shuffled = frame.sample(frac=1, random_state=seed).copy()
    shuffled["_variant_round"] = shuffled.groupby("family_block").cumcount()
    return shuffled[shuffled._variant_round.lt(5)].sort_values("_variant_round", kind="stable").head(cap).drop(columns="_variant_round")


def bootstrap_families(frame, seed):
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby("family_block", sort=True).indices.values())
    pieces = []
    for draw, index in enumerate(rng.integers(0, len(groups), len(groups))):
        part = frame.iloc[groups[index]].copy()
        part["_bootstrap_block"] = draw
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def medoids_for(matrix, assignment, k, weights=None):
    weights = np.ones(len(matrix)) if weights is None else weights
    medoids = []
    for cluster in range(k):
        indices = np.flatnonzero(assignment == cluster)
        if not len(indices):
            return None
        loss = matrix[np.ix_(indices, indices)] @ weights[indices]
        medoids.append(indices[loss.argmin()])
    return np.array(medoids)


def fit_candidates(frame, fields, method, config, seed, ks=None, balanced=False, scaling="quantile"):
    ks = list(config.clusters if ks is None else ks)
    metric = fit_metric(frame, fields, scaling)
    if method == "kmeans":
        training = frame if len(frame) <= config.vehicle_cap else frame.sample(config.vehicle_cap, random_state=seed)
        encoder = v2.encoder_for(clip_frame(training, metric), fields)
        x = encoder.transform(clip_frame(training, metric).replace({None: np.nan}))
        result = {}
        for k in ks:
            model = KMeans(k, n_init=config.starts, random_state=seed).fit(x, sample_weight=family_weights(training) if balanced else None)
            result[k] = {"method": method, "fields": fields, "metric": metric, "encoder": encoder, "model": model}
        return result
    sample = balanced_variants(frame, config.medoid_cap, seed) if balanced else frame.sample(min(config.medoid_cap, len(frame)), random_state=seed)
    sample = sample.reset_index(drop=True)
    matrix = np.nan_to_num(distances(sample, sample, metric), nan=1.)
    np.fill_diagonal(matrix, 0)
    tree = linkage(squareform(matrix, checks=False), method="average") if method == "hierarchical" else None
    weights = family_weights(sample) if balanced else np.ones(len(sample))
    result = {}
    for k in ks:
        if tree is not None:
            assignment = cut_tree(tree, n_clusters=[k]).ravel()
            medoids = medoids_for(matrix, assignment, k)
        else:
            best = None
            for start in range(config.starts):
                medoids = np.random.default_rng(seed + start).choice(len(sample), k, replace=False)
                for _ in range(50):
                    assignment = matrix[:, medoids].argmin(axis=1)
                    updated = medoids_for(matrix, assignment, k, weights)
                    if updated is None or np.array_equal(updated, medoids):
                        break
                    medoids = updated
                loss = np.average(matrix[:, medoids].min(axis=1), weights=weights)
                if best is None or loss < best[0]:
                    best = loss, medoids.copy()
            medoids = best[1]
            assignment = matrix[:, medoids].argmin(axis=1)
        centers = sample.iloc[medoids].copy()
        radius = []
        to_centers = distances(sample, centers, metric)
        for cluster in range(k):
            values = to_centers[assignment == cluster, cluster]
            values = values[np.isfinite(values)]
            radius.append(float(np.quantile(values, .95)) if len(values) else 0.)
        result[k] = {"method": method, "fields": fields, "metric": metric, "centers": centers,
                     "radius": np.array(radius), "reference_ids": sample.profile_id.tolist(),
                     "reference_assignment": assignment}
    return result


def assign(model, frame, reject=True):
    if not len(frame):
        return np.array([], dtype=int)
    valid = eligible(frame, model["fields"]).to_numpy()
    if model["method"] == "kmeans":
        labels = model["model"].predict(model["encoder"].transform(clip_frame(frame, model["metric"]).replace({None: np.nan})))
    else:
        labels = []
        for start in range(0, len(frame), 1000):
            distance = distances(frame.iloc[start:start + 1000], model["centers"], model["metric"])
            safe = np.where(np.isfinite(distance), distance, np.inf)
            nearest = safe.argmin(axis=1)
            minimum = safe[np.arange(len(safe)), nearest]
            supported = np.isfinite(minimum)
            if reject and model["method"] == "hierarchical":
                supported &= minimum <= model["radius"][nearest] + 1e-12
            labels.extend(np.where(supported, nearest, -1))
        labels = np.array(labels)
        if model["method"] == "hierarchical":
            reference = dict(zip(model["reference_ids"], model["reference_assignment"]))
            labels = np.array([reference.get(identity, label) for identity, label in zip(frame.profile_id, labels)])
    return np.where(valid, labels, -1)


def partition_score(model, development, anchor, k, config):
    labels = assign(model, development)
    anchor_labels = assign(model, anchor)
    assigned = anchor_labels >= 0
    silhouette = -1.
    if 1 < len(set(anchor_labels[assigned])) < assigned.sum():
        if model["method"] == "kmeans":
            x = model["encoder"].transform(clip_frame(anchor.loc[assigned], model["metric"]).replace({None: np.nan}))
            silhouette = silhouette_score(x, anchor_labels[assigned])
        else:
            matrix = np.nan_to_num(distances(anchor.loc[assigned], anchor.loc[assigned], model["metric"]), nan=1.)
            np.fill_diagonal(matrix, 0)
            silhouette = silhouette_score(matrix, anchor_labels[assigned], metric="precomputed")
    sizes = np.bincount(labels[labels >= 0], minlength=k)
    families = [development.loc[labels == cluster, "family_block"].nunique() for cluster in range(k)]
    return {"silhouette": float(silhouette), "minimum_count": int(sizes.min()),
            "minimum_cluster_families": int(min(families)), "assignment_coverage": float((labels >= 0).mean()),
            "required_minimum_count": max(10, int(np.ceil(.05 * len(development))))}


def stable_ari(reference, changed):
    common = (reference >= 0) & (changed >= 0)
    # Abstention must not make an unstable partition appear stable on a tiny overlap.
    return float(adjusted_rand_score(reference[common], changed[common])) if common.mean() >= .8 and common.sum() >= 10 else -1.


def choose_finalist(rows):
    supported = [row for row in rows if row["screen_median_ari"] >= .7 and row["silhouette"] > 0
                 and row["minimum_count"] >= row["required_minimum_count"] and row["minimum_cluster_families"] >= 5]
    if not supported:
        return None
    best = max(row["silhouette"] for row in supported)
    return min((row for row in supported if row["silhouette"] >= best - .02), key=lambda row: row["k"])


def discover(development, fields, target, config):
    """Screen in training families; reserve disjoint family anchors for refit stability."""
    target.mkdir(parents=True, exist_ok=False)
    anchor_mask = development.family_block.map(lambda key: int(r.digest("anchor|" + key)[:16], 16) % 5 == 0)
    training, anchor = development.loc[~anchor_mask], development.loc[anchor_mask]
    if len(development) < config.minimum_profiles or development.family_block.nunique() < config.minimum_families or training.family_block.nunique() < 10 or anchor.family_block.nunique() < 2:
        return {}, [{"method": "all", "status": "insufficient_support", "development_profiles": len(development),
                     "family_blocks": development.family_block.nunique()}], [], pd.DataFrame()
    anchor = anchor.sample(min(config.anchor_cap, len(anchor)), random_state=config.seed)
    records, repeat_rows, sensitivity_rows, accepted = [], [], [], {}
    for method in METHODS:
        print(target.name, method, "screen", len(training), "profiles", flush=True)
        base = fit_candidates(training, fields, method, config, config.seed)
        references = {k: assign(model, anchor) for k, model in base.items()}
        aris = {k: [] for k in config.clusters}
        for repeat in range(config.screen_refits):
            models = fit_candidates(bootstrap_families(training, 101 + repeat), fields, method, config, 201 + repeat)
            for k, model in models.items():
                value = stable_ari(references[k], assign(model, anchor))
                aris[k].append(value)
                repeat_rows.append({"method": method, "k": k, "phase": "screen", "repeat": repeat, "ari": value})
        method_rows = []
        for k, model in base.items():
            row = {"method": method, "k": k, "screen_median_ari": float(np.median(aris[k])),
                   **partition_score(model, development, anchor, k, config), "status": "screened_out"}
            method_rows.append(row)
        chosen = choose_finalist(method_rows)
        if chosen is None:
            method_rows[0]["status"] = "no_supported_subdivision"
        if chosen is not None:
            k, model = chosen["k"], base[chosen["k"]]
            final_aris = []
            for repeat in range(config.stability_refits):
                changed = fit_candidates(bootstrap_families(training, 1001 + repeat), fields, method, config, 2001 + repeat, [k])[k]
                value = stable_ari(references[k], assign(changed, anchor))
                final_aris.append(value)
                repeat_rows.append({"method": method, "k": k, "phase": "confirmation", "repeat": repeat, "ari": value})
            balanced_aris = []
            alternatives = [("equal_family", fields, training, seed, True, "quantile") for seed in (42, 43, 44)]
            alternatives += [("without_transmission", [f for f in fields if f != "transmission"], training, 42, False, "quantile"),
                             ("minmax_scaling", fields, training, 42, False, "minmax")]
            complete = training.dropna(subset=fields)
            if len(complete) >= config.minimum_profiles and complete.family_block.nunique() >= config.minimum_families:
                alternatives.append(("complete_case", fields, complete, 42, False, "quantile"))
            else:
                sensitivity_rows.append({"method": method, "k": k, "sensitivity": "complete_case", "status": "insufficient_support", "ari": np.nan})
            for name, alternative_fields, subset, seed, balanced, scaling in alternatives:
                changed = fit_candidates(subset, alternative_fields, method, config, seed, [k], balanced, scaling)[k]
                value = stable_ari(references[k], assign(changed, anchor))
                sensitivity_rows.append({"method": method, "k": k, "sensitivity": name, "seed": seed, "status": "evaluated", "ari": value})
                if name == "equal_family":
                    balanced_aris.append(value)
            missing_x = anchor[fields].isna().astype(float).to_numpy()
            if len(np.unique(missing_x, axis=0)) >= k:
                missing_labels = KMeans(k, n_init=5, random_state=42).fit_predict(missing_x)
                sensitivity_rows.append({"method": method, "k": k, "sensitivity": "missingness_only", "status": "evaluated",
                                         "ari": stable_ari(references[k], missing_labels)})
            chosen.update(median_ari=float(np.median(final_aris)), ari_q10=float(np.quantile(final_aris, .1)),
                          balanced_min_ari=float(min(balanced_aris)))
            supported = (chosen["median_ari"] >= config.median_ari and chosen["ari_q10"] >= config.lower_ari
                         and chosen["balanced_min_ari"] >= config.median_ari)
            chosen["status"] = "accepted_partition" if supported else "no_supported_subdivision"
            if supported:
                accepted[method] = model
                joblib.dump(model, target / (method + ".joblib"))
        records.extend(method_rows)
    pd.DataFrame(records).to_csv(target / "model_comparison.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(target / "stability.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(target / "sensitivities.csv", index=False)
    # Methods have different geometries: fixed preference, never rank their silhouettes.
    winner = next((method for method in ("gower", "hierarchical", "kmeans") if method in accepted), None)
    r.write_json(target / "selection.json", {"selected_method": winner, "method_precedence": ["gower", "hierarchical", "kmeans"],
        "anchor_families": sorted(anchor.family_block.unique()), "training_families": sorted(training.family_block.unique()),
        "semantic_names": "provisional", "reduced_budget": config.smoke})
    return accepted, records, sensitivity_rows, anchor


def freeze_queries(data, config):
    candidates = data[eligible(data, v2.NUMERIC + v2.CATEGORICAL)].copy()
    # Sampling depends on observed support, not fitted cluster or returned-slate quality.
    candidates = candidates.sample(frac=1, random_state=config.seed).drop_duplicates("family_block")
    candidates["year_band"] = (pd.to_numeric(candidates.ModelYear) // 5 * 5).astype(int)
    candidates["missing_fields"] = candidates[v2.NUMERIC + v2.CATEGORICAL].isna().sum(axis=1)
    candidates["missing_band"] = np.where(candidates.missing_fields.le(2), "0_to_2", "3_or_more")
    result = legacy.stratified_rows(candidates, config.queries, ["body_family", "powertrain", "year_band", "missing_band", "fold"])
    if len(result) != config.queries:
        raise ValueError(f"Need {config.queries} distinct supported families; found {len(result)}")
    result = result.reset_index(drop=True)
    result["card_id"] = [f"C{i:03d}" for i in range(1, len(result) + 1)]
    return result[["card_id", "profile_id", *r.COHORT, "family_block", "fold", "body_family", "powertrain", "year_band", "missing_band"]]


def retrieve(data, query_ids, fields, metric, assignments, config, ranking_policy=None, approaches=APPROACHES):
    """Rank cached arrays; avoid copying the wide profile table for every query."""
    rows, statuses = [], []
    frame = data.reset_index(drop=True)
    positions = dict(zip(frame.profile_id, range(len(frame))))
    ids = frame.profile_id.to_numpy()
    families = frame.family_block.to_numpy()
    bodies = frame.body_family.to_numpy()
    powertrains = frame.powertrain.to_numpy()
    years = pd.to_numeric(frame.ModelYear).to_numpy(dtype=float, na_value=np.nan)
    folds = frame.fold.to_numpy(int)
    is_eligible = frame.eligible.to_numpy(bool)
    cluster = np.array([assignments.get(identity, -1) for identity in ids])
    values, observed = {}, {}
    for field in fields:
        observed[field] = frame[field].notna().to_numpy()
        if field in v2.CATEGORICAL:
            values[field] = frame[field].fillna("__MISSING_EVIDENCE__").astype(str).to_numpy()
        else:
            values[field] = np.clip(frame[field].to_numpy(float), *metric["bounds"][field])
    groups = {(body, power): indices for (body, power), indices in frame[frame.eligible].groupby(["body_family", "powertrain"]).groups.items()}
    for query_id in query_ids:
        qi = positions[query_id]
        query_cluster = int(cluster[qi])
        status = {"query_profile_id": query_id, "family_block": families[qi], "fold": int(folds[qi]),
                  "eligible": bool(is_eligible[qi]), "query_cluster": query_cluster,
                  "cluster_status": "assigned" if query_cluster >= 0 else "direct_fallback",
                  "evidence_scope": "exploratory_unknown_powertrain" if powertrains[qi] == "unknown" else "supported_powertrain"}
        family = is_eligible & (families != families[qi])
        body = family & (bodies == bodies[qi])
        power = body & (powertrains == powertrains[qi])
        candidate = np.asarray(groups.get((bodies[qi], powertrains[qi]), []), dtype=int)
        candidate = candidate[(families[candidate] != families[qi]) & (np.abs(years[candidate] - years[qi]) <= 10)] if is_eligible[qi] else np.array([], dtype=int)
        components, shared = {}, {}
        valid = np.ones(len(candidate), dtype=bool)
        for block, block_fields in metric["blocks"].items():
            total, count = np.zeros(len(candidate)), np.zeros(len(candidate), dtype=int)
            for field in block_fields:
                common = observed[field][qi] & observed[field][candidate]
                if field in v2.CATEGORICAL:
                    difference = values[field][candidate] != values[field][qi]
                else:
                    lo, hi = metric["bounds"][field]
                    difference = np.minimum(np.abs(values[field][candidate] - values[field][qi]) / max(hi - lo, 1e-9), 1.)
                total += np.where(common, difference, 0.)
                count += common
            components[block] = np.divide(total, count, out=np.zeros(len(candidate)), where=count > 0)
            shared[block] = count
            valid &= count >= max(2 if block == "size" else 1, int(np.ceil(len(block_fields) / 2)))
        near = valid & (np.abs(years[candidate] - years[qi]) <= 5)
        window = 5 if len(set(families[candidate[near]])) >= 5 else 10
        valid &= near if window == 5 else True
        candidate_families = len(set(families[candidate[valid]]))
        indices = np.flatnonzero(valid)
        distance = (components["size"] + components["powertrain"]) / 2
        penalties = np.where((query_cluster >= 0) & (cluster[candidate] >= 0) & (cluster[candidate] != query_cluster), config.cluster_penalty, 0.)
        reason = "query_ineligible" if not is_eligible[qi] else "no_compatible_family" if not power.any() else "no_shared_evidence_or_year_support" if not len(indices) else "shortfall" if candidate_families < 5 else "complete"
        custom = ranking_policy(query_id, ids[candidate], penalties) if ranking_policy else None
        for approach in approaches:
            ranking = distance + (penalties if approach == "cluster_aware" else 0.)
            preference = np.zeros(len(candidate))
            applied_penalty = penalties if approach == "cluster_aware" else np.zeros(len(candidate))
            if custom is not None:
                preference, applied_penalty = custom[approach]
                ranking = distance + applied_penalty
            order = indices[np.lexsort((ids[candidate[indices]], np.abs(years[candidate[indices]] - years[qi]), ranking[indices], -preference[indices]))]
            selected, used_families = [], set()
            for index in order:
                family_id = families[candidate[index]]
                if family_id not in used_families:
                    selected.append(index)
                    used_families.add(family_id)
                if len(selected) == 5:
                    break
            statuses.append({**status, "approach": approach, "slate_size": len(selected), "status": reason,
                             "year_window": window, "eligible_alternative_profiles": int(family.sum()),
                             "body_compatible_profiles": int(body.sum()), "powertrain_compatible_profiles": int(power.sum()),
                             "supported_candidate_families": candidate_families})
            for rank, index in enumerate(selected, 1):
                rows.append({"query_profile_id": query_id, "peer_profile_id": ids[candidate[index]], "approach": approach,
                    "rank": rank, "distance": distance[index], "ranking_distance": ranking[index],
                    "size_distance": components["size"][index], "powertrain_distance": components["powertrain"][index],
                    "shared_size": shared["size"][index], "shared_powertrain": shared["powertrain"][index],
                    "cluster_penalty": applied_penalty[index],
                    **({"category_match": preference[index]} if custom is not None else {}),
                    "year_window": window, "evidence_scope": status["evidence_scope"]})
    return pd.DataFrame(rows, columns=PEER_COLUMNS + (["category_match"] if ranking_policy else [])), pd.DataFrame(statuses)


def validate_peers(data, peers, statuses, fields=None, metric=None):
    if statuses.duplicated(["query_profile_id", "approach"]).any():
        raise ValueError("Duplicate query status")
    if peers.empty:
        return
    index = data.set_index("profile_id")
    q = index.loc[peers.query_profile_id].reset_index(drop=True)
    p = index.loc[peers.peer_profile_id].reset_index(drop=True)
    if (q.family_block.eq(p.family_block) | q.body_family.ne(p.body_family) | q.powertrain.ne(p.powertrain)
            | ~q.eligible | ~p.eligible | q.powertrain.eq("conflicting")).any():
        raise ValueError("Peer compatibility violation")
    if ((pd.to_numeric(q.ModelYear) - pd.to_numeric(p.ModelYear)).abs() > peers.year_window.to_numpy()).any():
        raise ValueError("Peer year-window violation")
    if peers.assign(peer_family=p.family_block.to_numpy()).duplicated(["query_profile_id", "approach", "peer_family"]).any():
        raise ValueError("Duplicate peer family")
    if not np.allclose(peers.distance, (peers.size_distance + peers.powertrain_distance) / 2):
        raise ValueError("Block distance does not reconcile")
    if not np.allclose(peers.ranking_distance, peers.distance + peers.cluster_penalty):
        raise ValueError("Ranking penalty does not reconcile")
    if fields:
        for block in ("size", "powertrain"):
            selected = [field for field in fields if v2.feature_block(field) == block]
            shared = (q[selected].notna() & p[selected].notna()).sum(axis=1).to_numpy()
            if (shared < max(2 if block == "size" else 1, int(np.ceil(len(selected) / 2)))).any():
                raise ValueError("Insufficient shared source evidence")
            if not np.array_equal(shared, peers["shared_" + block].to_numpy()):
                raise ValueError("Shared evidence metadata differs from source profiles")
            if metric is not None:
                total = np.zeros(len(peers))
                for field in selected:
                    observed = q[field].notna().to_numpy() & p[field].notna().to_numpy()
                    if field in v2.CATEGORICAL:
                        difference = q[field].fillna("").astype(str).to_numpy() != p[field].fillna("").astype(str).to_numpy()
                    else:
                        lo, hi = metric["bounds"][field]
                        difference = np.minimum(np.abs(np.clip(q[field].to_numpy(float), lo, hi)
                            - np.clip(p[field].to_numpy(float), lo, hi)) / max(hi - lo, 1e-9), 1.)
                    total += np.where(observed, difference, 0.)
                if not np.allclose(total / shared, peers[block + "_distance"]):
                    raise ValueError("Saved distance differs from source evidence")
    actual = peers.groupby(["query_profile_id", "approach"]).size()
    for row in statuses.itertuples():
        if actual.get((row.query_profile_id, row.approach), 0) != row.slate_size:
            raise ValueError("Query status denominator mismatch")


def diagnostics(statuses, peers):
    result = []
    for approach, group in statuses.groupby("approach"):
        distance = peers[peers.approach.eq(approach)].groupby("query_profile_id").distance.mean()
        result.append({"approach": approach, "queries": len(group), "families": group.family_block.nunique(),
            "queries_with_peers": int(group.slate_size.gt(0).sum()), "complete_list_share": group.slate_size.eq(5).mean(),
            "empty_list_share": group.slate_size.eq(0).mean(), "mean_slate_size": group.slate_size.mean(),
            "mean_returned_distance": distance.mean(), "distance_is_usefulness": False})
    return pd.DataFrame(result)


def evidence_summary(prepared, profiles):
    path = Path(prepared) / "vin_profile_membership.parquet"
    group = ["profile_id", "fuel_type", "FuelTypePrimary", "FuelTypeSecondary", "ElectrificationLevel", "powertrain"]
    evidence = pl.scan_parquet(path).group_by(group).agg(pl.len().alias("vin_count")).collect().to_pandas()
    # Keep explicit source values: "Not Applicable" is not silently redefined as ICE.
    evidence["electrification_observed_vins"] = evidence.vin_count.where(evidence.ElectrificationLevel.notna(), 0)
    support = evidence.groupby("profile_id").agg(
        electrification_observed_vins=("electrification_observed_vins", "sum"),
        observed_fuels=("fuel_type", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
        electrification_evidence=("ElectrificationLevel", lambda x: ";".join(sorted(set(x.dropna().astype(str))))))
    return evidence, profiles.merge(support, left_on="profile_id", right_index=True, how="left", validate="1:1")


def coverage(data, statuses):
    direct = statuses[statuses.approach.eq("direct_specs")].set_index("query_profile_id")
    result = data.copy()
    result["has_peers"] = result.profile_id.map(direct.slate_size).fillna(0).gt(0)
    result["complete_slate"] = result.profile_id.map(direct.slate_size).fillna(0).eq(5)
    result["year_band"] = (pd.to_numeric(result.ModelYear, errors="coerce") // 5 * 5).astype("Int64").astype(str)
    result["missingness"] = pd.cut(result[v2.NUMERIC + v2.CATEGORICAL].isna().sum(axis=1), [-1, 2, 5, 11], labels=["0_to_2", "3_to_5", "6_or_more"]).astype(str)
    return result.groupby(["identity_complete", "body_family", "year_band", "powertrain", "missingness"], dropna=False).agg(
        listings=("vin_count", "sum"), profiles=("profile_id", "size"), families=("family_block", "nunique"),
        eligible_profiles=("eligible", "sum"), profiles_with_peers=("has_peers", "sum"), complete_slates=("complete_slate", "sum")).reset_index()


def lookup_profiles(directory, year, make, model, trim):
    directory = Path(directory)
    legacy.verify_artifacts(directory)
    profiles = pd.read_parquet(directory / "variant_profiles.parquet")
    mask = pd.Series(True, index=profiles.index)
    for key, value in zip(r.COHORT, [year, make, model, trim]):
        mask &= profiles[key].astype(str).str.upper().eq(str(value).upper())
    selected = profiles[mask]
    labels = pd.read_parquet(directory / "profile_labels.parquet")
    peers = pd.read_parquet(directory / "profile_peers.parquet")
    return selected, labels[labels.profile_id.isin(selected.profile_id)], peers[peers.query_profile_id.isin(selected.profile_id)].merge(
        profiles, left_on="peer_profile_id", right_on="profile_id", validate="m:1")


def execution_receipt():
    modules = [Path(__file__), Path(r.__file__), Path(s.__file__), Path(legacy.__file__), Path(v2.__file__)]
    return {"code_hashes": {p.name: r.file_hash(p) for p in modules},
            "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "polars", "scipy", "scikit-learn", "joblib", "threadpoolctl"]}}


def analyze(prepared, output, smoke=False):
    prepared, target = Path(prepared), Path(output)
    receipt = legacy.verify_artifacts(prepared)
    if receipt.get("status") != "prepared" or receipt.get("schema_version") != "2.0.0":
        raise ValueError("Requires a completed v2 preparation")
    config, execution = config_for(smoke), execution_receipt()
    r.begin_stage(target.parent, target.name)
    (target / "execution_vehicle_similarity_v3.py").write_bytes(Path(__file__).read_bytes())
    profiles = pd.read_parquet(prepared / "variant_profiles.parquet")
    names = pd.read_parquet(prepared / "identity_names.parquet")
    crosswalk = r.read_json(v2.ASSETS / "identity_crosswalk.v2.json")
    profiles = v2.resolve_families(profiles, names, crosswalk["mappings"])
    for field in v2.NUMERIC + v2.CATEGORICAL:
        profiles[field] = profiles[field].where(profiles[field + "_n"].ge(5))
    definitions, registry = legacy.load_taxonomy(v2.ASSETS / "definitions.v2.json", v2.ASSETS / "evidence.v2.json")
    profiles, labels, exceptions = v2.profile_labels(profiles, legacy.draft_mappings(registry), definitions)
    profiles["fold"] = profiles.family_block.map(family_fold)
    evidence, profiles = evidence_summary(prepared, profiles)
    evidence.to_parquet(target / "vin_evidence.parquet", index=False)
    profiles["schema_version"] = VERSION
    queries = freeze_queries(profiles, config)
    queries.to_csv(target / "evaluation_queries.csv", index=False)
    profiles[["profile_id", "family_block", "fold"]].to_csv(target / "fold_membership.csv", index=False)
    labels.to_parquet(target / "profile_labels.parquet", index=False)
    exceptions.to_csv(target / "exceptions.csv", index=False)
    for filename, value in [("identity_crosswalk.json", crosswalk), ("definitions.json", definitions), ("evidence_registry.json", registry)]:
        r.write_json(target / filename, value)
    if registry.get("reference_batch"):
        batch = registry["reference_batch"]
        for filename, expected in batch["hashes"].items():
            original = r.ROOT / batch["directory"] / filename
            if r.file_hash(original) != expected:
                raise ValueError("Changed source acquisition receipt: " + filename)
            shutil.copyfile(original, target / filename)
    r.write_json(target / "execution_configuration.json", {**asdict(config), **execution})
    peers, statuses, cluster_rows, sensitivity_rows, compositions, field_rows, membership_rows, scale_rows = [], [], [], [], [], [], [], []
    membership = pl.scan_parquet(prepared / "vin_profile_membership.parquet")
    out_of_fold_eligible = pd.Series(False, index=profiles.index)
    with threadpool_limits(limits=1):
        for fold in range(config.folds):
            print("FOLD", fold, flush=True)
            training_ids = profiles.loc[profiles.identity_complete & profiles.fold.ne(fold), ["profile_id"]]
            pairs = membership.join(pl.from_pandas(training_ids).lazy(), on="profile_id", how="inner").filter(
                pl.col("horsepower").is_finite() & pl.col("EngineHP").is_finite()
                & (pl.col("horsepower") > 0) & (pl.col("EngineHP") > 0)).select((pl.col("horsepower") / pl.col("EngineHP")).alias("ratio")).collect()["ratio"]
            verified = len(pairs) >= 30 and .9 <= pairs.median() <= 1.1
            scale_rows.append({"fold": fold, "paired_training_vins": len(pairs), "median_ratio": pairs.median(), "verified": verified})
            fold_data = profiles.copy()
            fold_data["development"] = fold_data.fold.ne(fold)
            fold_data, fields, audit = v2.select_fields(fold_data, excluded_fields=() if verified else ("horsepower",))
            audit["fold"] = fold
            field_rows.append(audit)
            out_of_fold_eligible.loc[profiles.fold.eq(fold)] = fold_data.loc[profiles.fold.eq(fold), "eligible"]
            training = fold_data[fold_data.development & fold_data.eligible]
            metric = fit_metric(training, fields)
            assignments = {}
            for view in ["global", *sorted(training.body_family.unique())]:
                subset = training if view == "global" else training[training.body_family.eq(view)]
                directory = target / "clusters" / f"fold_{fold}" / view
                models, records, sensitivity, _ = discover(subset, fields, directory, config)
                cluster_rows.extend({"fold": fold, "view": view, **row} for row in records)
                sensitivity_rows.extend({"fold": fold, "view": view, **row} for row in sensitivity)
                selected = next((method for method in ("gower", "hierarchical", "kmeans") if method in models), None)
                population = fold_data[fold_data.eligible & (True if view == "global" else fold_data.body_family.eq(view))]
                for method, model in models.items():
                    assigned = assign(model, population)
                    members = population[["profile_id", "family_block", "fold", "body_family", "ModelYear", "powertrain", "observed_fuels", "electrification_evidence"]].copy()
                    members["cluster"], members["training_fold"], members["view"], members["method"] = assigned, fold, view, method
                    membership_rows.append(members)
                    for cluster, group in population.assign(cluster=assigned).groupby("cluster"):
                        if cluster < 0:
                            continue
                        compositions.append({"fold": fold, "view": view, "method": method, "cluster": int(cluster),
                            "profiles": len(group), "families": group.family_block.nunique(), "listings": int(group.vin_count.sum()),
                            "year_min": int(group.ModelYear.min()), "year_max": int(group.ModelYear.max()),
                            "description": legacy.subgroup_name(group, population, fields),
                            "mean_missing_fields": group[fields].isna().sum(axis=1).mean(),
                            **{field + "_median": group[field].median() for field in fields if field not in v2.CATEGORICAL}})
                    if view != "global" and method == selected:
                        assignments.update(dict(zip(population.profile_id, assigned)))
            joblib.dump({"metric": metric, "fields": fields, "training_families": sorted(training.family_block.unique())}, target / f"fold_{fold}_retrieval.joblib")
            # Full population coverage uses its own held-out fold, including ineligible rows.
            query_ids = profiles.loc[profiles.fold.eq(fold), "profile_id"]
            returned, status = retrieve(fold_data, query_ids, fields, metric, assignments, config)
            validate_peers(fold_data, returned, status, fields, metric)
            peers.append(returned)
            statuses.append(status)
    profiles["eligible"] = out_of_fold_eligible
    profiles.to_parquet(target / "variant_profiles.parquet", index=False)
    peer_table, status_table = pd.concat(peers, ignore_index=True), pd.concat(statuses, ignore_index=True)
    peer_table.to_parquet(target / "profile_peers.parquet", index=False)
    status_table.to_parquet(target / "query_status.parquet", index=False)
    coverage(profiles, status_table).to_csv(target / "coverage.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(target / "cluster_experiments.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(target / "cluster_sensitivities.csv", index=False)
    pd.DataFrame(compositions).to_csv(target / "cluster_profiles.csv", index=False)
    (pd.concat(membership_rows, ignore_index=True) if membership_rows else pd.DataFrame(columns=["profile_id", "cluster"])).to_parquet(target / "cluster_memberships.parquet", index=False)
    pd.concat(field_rows, ignore_index=True).to_csv(target / "feature_support.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(target / "numeric_scale_audit.csv", index=False)
    selected_status = status_table[status_table.query_profile_id.isin(queries.profile_id)]
    selected_peers = peer_table[peer_table.query_profile_id.isin(queries.profile_id)]
    diagnostics(selected_status, selected_peers).to_csv(target / "benchmark_diagnostics.csv", index=False)
    r.write_json(target / "evaluation.json", {"schema_version": VERSION, "status": "pending_human_review", "query_families": len(queries),
        "buyer_usefulness": "not_measured", "semantic_accuracy": "not_measured", "selected_approach": "direct_specs",
        "cluster_adoption": "requires_positive_lower_95_interval_and_no_coverage_loss_or_compatibility_violations",
        "reduced_budget": smoke, "reviewer_limit": "single_reviewer_no_inter_rater_validation"})
    if execution != execution_receipt():
        raise ValueError("Execution dependencies changed during analysis")
    r.finish_stage(target, "completed", schema_version=VERSION, reduced_budget=smoke, configuration=asdict(config),
        prepared_directory=str(prepared.resolve()), prepared_manifest_sha256=r.file_hash(prepared / "manifest.json"),
        source_hashes=receipt["source_hashes"], **execution)
    return target


def profile_title(row):
    return " ".join(str(row[key]) for key in r.COHORT)


def powertrain_description(value):
    return {"unknown": "electrification unverified", "plug_in_hybrid": "plug-in hybrid",
            "battery_electric": "battery electric"}.get(value, str(value).replace("_", " "))


def export_review(run, output):
    run, target = Path(run), Path(output)
    manifest = legacy.verify_artifacts(run)
    if manifest.get("schema_version") != VERSION or manifest.get("status") != "completed":
        raise ValueError("Requires a completed v3 analysis")
    target.mkdir(parents=True, exist_ok=False)
    queries = pd.read_csv(run / "evaluation_queries.csv")
    profiles = pd.read_parquet(run / "variant_profiles.parquet").set_index("profile_id")
    peers = pd.read_parquet(run / "profile_peers.parquet")
    statuses = pd.read_parquet(run / "query_status.parquet")
    rng = np.random.default_rng(20261023)
    rubric = "Grade each vehicle: 0 unsuitable, 1 plausible, 2 strong. Leave unavailable slots blank. Choose A, B, tie, or neither. Unknown electrification is not verified combustion. Specifications are observed profile medians, not verified individual equipment."
    parts = ["<!doctype html><html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Vehicle similarity review</title>",
        "<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#20313d;background:#f5f7f8}section{background:white;padding:24px;margin:24px 0;border:1px solid #d5dce0;border-radius:12px}.slates{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{border:1px solid #d5dce0;padding:16px;margin:12px 0}small{color:#465b68}button{padding:12px;margin:8px}select,input{padding:6px;max-width:95%}.query{background:#eaf0f4;padding:12px}.warning{color:#75480b}@media(max-width:800px){.slates{grid-template-columns:1fr}}@media print{button{display:none}section{break-before:page}}</style>",
        "<h1>Blinded vehicle similarity comparisons</h1><p>" + rubric + "</p><p>Use one consistent judgment across cards. Save progress in this browser; export CSV when done. Keep the CSV as your durable copy.</p>",
        "<label>Reviewer <input id='reviewer' placeholder='Your name or initials'></label><button onclick='download()'>Export ratings CSV</button><button onclick='window.print()'>Print cards</button><span id='progress'></span>"]
    frozen_rows, key_rows = [], []
    records = []
    fold_fields = {int(fold): joblib.load(run / f"fold_{fold}_retrieval.joblib")["fields"] for fold in queries.fold.unique()}
    units = {**r.UNITS, "torque_nm": "Nm"}
    field_names = {"wheel_system": "drivetrain", "fuel_type": "fuel", "torque_nm": "torque"}
    for query in queries.to_dict("records"):
        qid, card = query["profile_id"], query["card_id"]
        a = APPROACHES if rng.integers(2) else APPROACHES[::-1]
        frozen = {"card_id": card, "query_profile_id": qid, "winner": "", "reviewer": "", "notes": ""}
        q = profiles.loc[qid]
        parts.append(f"<section id='{card}'><h2>{card}</h2><div class='query'><b>Reference: {html.escape(profile_title(q))}</b><p>{html.escape(str(q.body_style).replace('_',' '))}; {html.escape(powertrain_description(q.powertrain))}; observed fuel: {html.escape(str(q.observed_fuels))}; {int(q.vin_count):,} VINs.</p></div><div class='slates'>")
        for side, approach in zip(("A", "B"), a):
            key_rows.append({"card_id": card, "side": side, "approach": approach})
            slate = peers[peers.query_profile_id.eq(qid) & peers.approach.eq(approach)].sort_values("rank")
            status = statuses[statuses.query_profile_id.eq(qid) & statuses.approach.eq(approach)].iloc[0]
            scope = "Electrification is unverified; these are exploratory specification peers" if q.powertrain == "unknown" else "Observed powertrain matches; buyer usefulness is unverified"
            parts.append(f"<div><h3>Slate {side}</h3><p>{len(slate)} alternatives; year window +/-{int(status.year_window)}. {scope}.</p>")
            for rank in range(1, 6):
                item = slate[slate['rank'].eq(rank)]
                pid = "" if item.empty else item.iloc[0].peer_profile_id
                frozen[f"{side}{rank}_peer_id"], frozen[f"{side}{rank}_rating"] = pid, ""
                if not pid:
                    parts.append(f"<article><b>{rank}. No supported alternative</b></article>")
                    continue
                p = profiles.loc[pid]
                differences, gaps = [], []
                for field in fold_fields[int(query["fold"])]:
                    name = field_names.get(field, field.replace('_', ' '))
                    if pd.notna(q[field]) and pd.notna(p[field]):
                        if field in v2.CATEGORICAL:
                            differences.append(f"{name}: {p[field]} (reference {q[field]})")
                        else:
                            differences.append(f"{name}: {p[field]:g} {units.get(field, '')} (reference {q[field]:g})")
                    else:
                        gaps.append(name)
                details = html.escape("; ".join(differences))
                gap_text = html.escape(", ".join(gaps) or "none among displayed fields")
                parts.append(f"<article><b>{rank}. {html.escape(profile_title(p))}</b><p>{html.escape(str(p.body_style).replace('_',' '))}; {html.escape(powertrain_description(p.powertrain))}; fuel {html.escape(str(p.observed_fuels))}; {int(p.vin_count):,} VINs.</p><small>{details}</small><p class='warning'>Missing comparison evidence: {gap_text}.</p><label>Rating <select data-card='{card}' data-field='{side}{rank}_rating'><option value=''>Unreviewed</option><option value='0'>0 unsuitable</option><option value='1'>1 plausible</option><option value='2'>2 strong</option></select></label></article>")
            parts.append("</div>")
        parts.append(f"</div><label>Preferred slate <select data-card='{card}' data-field='winner'><option value=''>Unreviewed</option><option>A</option><option>B</option><option>tie</option><option>neither</option></select></label><p><label>Notes <input data-card='{card}' data-field='notes' size='100'></label></p></section>")
        frozen_rows.append(frozen)
        records.append(frozen.copy())
    frozen = pd.DataFrame(frozen_rows)
    frozen.to_csv(target / "ratings_template.csv", index=False)
    pd.DataFrame(key_rows).to_csv(target / "blind_key.csv", index=False)
    # The local review HTML contains only blinded cards, never the approach key or distances.
    payload = json.dumps(records).replace("<", "\\u003c")
    columns = json.dumps(list(frozen.columns))
    storage = "vehicle-review-" + r.file_hash(run / "manifest.json")[:20]
    script = r"""
<script>
const initial=__PAYLOAD__, columns=__COLUMNS__, storage=__STORAGE__;
let saved;
try { saved=JSON.parse(localStorage.getItem(storage)||'null'); } catch(e) {}
let rows=saved?.rows||initial;
const reviewer=document.getElementById('reviewer');
reviewer.value=saved?.reviewer||'';
function persist() {
  try { localStorage.setItem(storage,JSON.stringify({rows,reviewer:reviewer.value})); } catch(e) {}
  document.getElementById('progress').textContent=rows.filter(r=>r.winner).length+' / '+rows.length+' cards have a preference';
}
document.querySelectorAll('[data-card]').forEach(el=>{
  let row=rows.find(r=>r.card_id===el.dataset.card);
  el.value=row[el.dataset.field]||'';
  el.addEventListener('change',()=>{
    row[el.dataset.field]=el.value;
    if(el.dataset.field.endsWith('_rating')) {
      let peer=row[el.dataset.field.replace('_rating','_peer_id')];
      document.querySelectorAll('[data-card="'+el.dataset.card+'"]').forEach(other=>{
        if(other.dataset.field.endsWith('_rating') && row[other.dataset.field.replace('_rating','_peer_id')]===peer) {
          row[other.dataset.field]=el.value;
          other.value=el.value;
        }
      });
    }
    persist();
  });
});
reviewer.addEventListener('input',persist);
function download() {
  let quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
  let csv=[columns.map(quote).join(','),...rows.map(r=>columns.map(k=>quote(k==='reviewer'?reviewer.value:r[k])).join(','))].join('\r\n');
  let a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  a.download='vehicle_similarity_ratings.csv';
  a.click();
  URL.revokeObjectURL(a.href);
}
persist();
</script></html>
"""
    parts.append(script.replace("__PAYLOAD__", payload).replace("__COLUMNS__", columns).replace("__STORAGE__", json.dumps(storage)))
    (target / "review_cards.html").write_text("\n".join(parts), encoding="utf-8")
    r.write_json(target / "manifest.json", {"status": "exported", "schema_version": VERSION, "run_directory": str(run.resolve()),
        "run_manifest_sha256": r.file_hash(run / "manifest.json"), "queries": len(queries), "reduced_budget": manifest["reduced_budget"],
        "frozen_hashes": {name: r.file_hash(target / name) for name in ["ratings_template.csv", "blind_key.csv", "review_cards.html"]},
        "rubric": rubric, "review_status": "pending", **execution_receipt()})
    return target


def evaluate_review(review, ratings, output, allow_incomplete=False):
    review, target = Path(review), Path(output)
    receipt = r.read_json(review / "manifest.json")
    for filename, expected in receipt["frozen_hashes"].items():
        if r.file_hash(review / filename) != expected:
            raise ValueError("Changed frozen review artifact: " + filename)
    run = Path(receipt["run_directory"])
    legacy.verify_artifacts(run)
    if r.file_hash(run / "manifest.json") != receipt["run_manifest_sha256"]:
        raise ValueError("Review run changed")
    expected = pd.read_csv(review / "ratings_template.csv", dtype=str, keep_default_na=False)
    actual = pd.read_csv(ratings, dtype=str, keep_default_na=False)
    if set(expected.columns) != set(actual.columns) or actual.card_id.duplicated().any() or set(actual.card_id) != set(expected.card_id):
        raise ValueError("Review identities or columns changed")
    actual = actual.set_index("card_id").loc[expected.card_id].reset_index()
    identity = ["card_id", "query_profile_id"] + [f"{side}{rank}_peer_id" for side in ("A", "B") for rank in range(1, 6)]
    if not actual[identity].equals(expected[identity]):
        raise ValueError("Frozen query or slate identities changed")
    valid_preferences = ["A", "B", "tie", "neither"] + ([""] if allow_incomplete else [])
    if not actual.winner.isin(valid_preferences).all() or not actual.reviewer.str.strip().ne("").all():
        raise ValueError("All cards need a preference and reviewer")
    key = pd.read_csv(review / "blind_key.csv")
    rows, missing_judgments = [], []
    for card in actual.to_dict("records"):
        if not card["winner"]:
            missing_judgments.append({"card_id": card["card_id"], "field": "winner"})
        seen_ratings = {}
        for side in ("A", "B"):
            grades = []
            for rank in range(1, 6):
                pid, grade = card[f"{side}{rank}_peer_id"], card[f"{side}{rank}_rating"]
                if pid:
                    if not grade and allow_incomplete:
                        missing_judgments.append({"card_id": card["card_id"], "field": f"{side}{rank}_rating"})
                        grades.append(np.nan)
                        continue
                    if grade not in ("0", "1", "2"):
                        raise ValueError("Every returned vehicle needs a grade of 0, 1 or 2")
                    if pid in seen_ratings and seen_ratings[pid] != grade:
                        raise ValueError("The same vehicle must have the same grade within a card")
                    seen_ratings[pid] = grade
                    grades.append(int(grade))
                elif grade:
                    raise ValueError("Unavailable slots cannot have ratings")
                else:
                    grades.append(0)
            approach = key.loc[key.card_id.eq(card["card_id"]) & key.side.eq(side), "approach"].item()
            fully_graded = bool(np.isfinite(grades).all())
            rows.append({"card_id": card["card_id"], "query_profile_id": card["query_profile_id"], "approach": approach,
                "useful_at_5": np.mean(np.array(grades) >= 1) if fully_graded else np.nan,
                "strong_at_5": np.mean(np.array(grades) == 2) if fully_graded else np.nan,
                "fully_graded": fully_graded,
                "preference_recorded": bool(card["winner"]),
                "preferred": (.5 if card["winner"] == "tie" else float(card["winner"] == side)) if card["winner"] else np.nan,
                "reviewer": card["reviewer"]})
    scored = pd.DataFrame(rows)
    # Drop an incompletely graded query from both methods to preserve the paired comparison.
    scored["fully_graded_pair"] = scored.groupby("card_id").fully_graded.transform("all")
    scored.loc[~scored.fully_graded_pair, ["useful_at_5", "strong_at_5"]] = np.nan
    paired = scored.pivot(index="card_id", columns="approach", values="useful_at_5").dropna()
    if paired.empty:
        raise ValueError("At least one fully graded query pair is required")
    delta = (paired.cluster_aware - paired.direct_specs).to_numpy()
    intervals = []
    for seed in (42, 43):
        rng = np.random.default_rng(seed)
        draws = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
        intervals.append(np.quantile(draws, [.025, .975]).tolist())
    statuses = pd.read_parquet(run / "query_status.parquet")
    sizes = statuses.pivot(index="query_profile_id", columns="approach", values="slate_size")
    no_coverage_loss = bool((sizes.cluster_aware >= sizes.direct_specs).all())
    profiles, peers = pd.read_parquet(run / "variant_profiles.parquet"), pd.read_parquet(run / "profile_peers.parquet")
    for fold in sorted(profiles.fold.unique()):
        fitted = joblib.load(run / f"fold_{fold}_retrieval.joblib")
        fields = fitted["fields"]
        fold_profiles = profiles.assign(eligible=eligible(profiles, fields))
        fold_status = statuses[statuses.fold.eq(fold)]
        validate_peers(fold_profiles, peers[peers.query_profile_id.isin(fold_status.query_profile_id)].reset_index(drop=True), fold_status, fields, fitted["metric"])
    adopt = not receipt["reduced_budget"] and len(actual) == 100 and not missing_judgments and min(interval[0] for interval in intervals) > 0 and no_coverage_loss
    incomplete_cards = {item["card_id"] for item in missing_judgments}
    summary = {"schema_version": VERSION, "status": "partially_reviewed" if missing_judgments else "reviewed",
        "cards": len(actual), "fully_graded_pairs": len(paired),
        "preference_cards": int(actual.winner.ne("").sum()), "complete_cards": len(actual) - len(incomplete_cards),
        "excluded_usefulness_cards": sorted(scored.loc[~scored.fully_graded_pair, "card_id"].unique()),
        "missing_judgments": missing_judgments, "allow_incomplete": allow_incomplete,
        "reviewers": sorted(actual.reviewer.unique()),
        "preference_counts": {**{approach: int(((scored.approach == approach) & scored.preferred.eq(1)).sum()) for approach in APPROACHES},
                              "tie": int(actual.winner.eq("tie").sum()), "neither": int(actual.winner.eq("neither").sum())},
        "mean_useful_at_5_change": float(delta.mean()), "paired_95_intervals_seeds_42_43": intervals,
        "no_coverage_loss": no_coverage_loss, "compatibility_violations": 0,
        "selected_approach": "cluster_aware" if adopt else "direct_specs", "reduced_budget": receipt["reduced_budget"],
        "semantic_accuracy": "not_measured", "interpretation": "Reviewed catalog population and supplied reviewer; not general buyer suitability or inter-rater agreement"}
    target.mkdir(parents=True, exist_ok=False)
    scored.to_csv(target / "scored_reviews.csv", index=False)
    scored.groupby("approach")[["useful_at_5", "strong_at_5", "preferred"]].mean().to_csv(target / "review_summary.csv")
    diagnostics(statuses[statuses.query_profile_id.isin(scored.query_profile_id)], peers[peers.query_profile_id.isin(scored.query_profile_id)]).to_csv(target / "coverage_summary.csv", index=False)
    shutil.copyfile(ratings, target / "submitted_ratings.csv")
    r.write_json(target / "evaluation.json", summary)
    r.write_json(target / "manifest.json", {"status": "completed", "schema_version": VERSION,
        "review_directory": str(review.resolve()), "run_manifest_sha256": receipt["run_manifest_sha256"],
        "review_manifest_sha256": r.file_hash(review / "manifest.json"), "ratings_sha256": r.file_hash(ratings),
        "artifacts": {p.name: r.file_hash(p) for p in target.iterdir() if p.is_file()}, **execution_receipt()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["analyze", "export-review", "evaluate-review"])
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Evaluate fully graded query pairs and available preferences; incomplete reviews cannot adopt clusters")
    args = parser.parse_args()
    required = {"analyze": ["prepared"], "export-review": ["run"], "evaluate-review": ["review", "ratings"]}[args.stage]
    for field in required:
        if getattr(args, field) is None:
            parser.error(f"{args.stage} requires --{field}")
    if args.output.exists():
        parser.error("Use a fresh output directory; frozen artifacts are immutable")
    try:
        if args.stage == "analyze":
            result = analyze(args.prepared, args.output, args.smoke)
        elif args.stage == "export-review":
            result = export_review(args.run, args.output)
        else:
            result = evaluate_review(args.review, args.ratings, args.output, allow_incomplete=args.allow_incomplete)
        print(result)
    except BaseException as error:
        if args.output.is_dir() and args.stage == "analyze":
            r.finish_stage(args.output, "interrupted" if isinstance(error, KeyboardInterrupt) else "blocked",
                           schema_version=VERSION, reduced_budget=args.smoke, error=str(error), **execution_receipt())
        raise


if __name__ == "__main__":
    main()
