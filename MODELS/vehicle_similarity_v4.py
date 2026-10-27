"""Provisional vehicle categories, family-separated subgroups and explicit activation.

Stages never change source Parquets or train price models. See --help and README.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from urllib.parse import quote, urlencode

from bs4 import BeautifulSoup
import joblib
import numpy as np
import pandas as pd
import polars as pl
import requests
from threadpoolctl import threadpool_limits

from EDA.DATA_CLEANING import EXCLUDED_LISTING_VINS
from MODELS import vehicle_profiles as r
from MODELS import vehicle_taxonomy as legacy
from MODELS import vehicle_taxonomy_v2 as v2
from MODELS import vehicle_similarity_v3 as v3
from MODELS import vehicle_taxonomy_sources as sources

VERSION = "4.0.0"
DIMENSIONS = ("market_positioning", "performance_character", "size_segment", "body_style", "powertrain")
SEMANTIC = DIMENSIONS[:3]
APPROACHES = ("direct_specs", "category_specs", "category_cluster")
PRIORITY = ("gower", "hierarchical", "kmeans")
ACTIVE_STAGE = None
# Only this released ranking failure can resume across a code change. Its fitted
# coordinates/checkpoints are unchanged; all other inputs still match exactly.
RANKING_REPAIR_BASE = "c23b2b5f6ca4f4a3bcfcdee9fb6861627942ad6aade8a83e3bb2dff344f2f608"
RANKING_REPAIR_FIT = "26b27cad4c333e176b244bfa974e359636195549cb7ab754d20040aa7786c4b5"
VALIDATION_REPAIR_BASE = "875b6a0edfd6fdecd9416e48819e02763469bd5304c2b9e0ab9a33f02c8b4075"
VALIDATION_REPAIR_OUTPUT = "837bae0565d7eff0ee1afa70bdf715bba6b267edeab3cb6a0b3247412781a2d5"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        r.write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execution():
    receipt = v3.execution_receipt()
    receipt["versions"].update({name: version(name) for name in ("requests", "beautifulsoup4")})
    receipt["code_hashes"].update({Path(p).name: r.file_hash(p) for p in [__file__, sources.__file__]})
    receipt["taxonomy_hashes"] = {name: r.file_hash(v2.ASSETS / name) for name in
        ("definitions.v2.json", "evidence.v2.json", "identity_crosswalk.v2.json")}
    return receipt


def verified(path, status=None):
    receipt = legacy.verify_artifacts(path)
    if receipt.get("schema_version") != VERSION or (status and receipt.get("status") != status):
        raise ValueError(f"Requires a v4 {status or 'completed'} artifact: {path}")
    if not receipt.get("artifacts"):
        raise ValueError("Empty artifact receipt")
    return receipt


def dependency(path):
    return {"directory": str(Path(path).resolve()), "manifest_sha256": r.file_hash(Path(path) / "manifest.json")}


def check_dependency(value, status=None):
    path = Path(value["directory"])
    if r.file_hash(path / "manifest.json") != value["manifest_sha256"]:
        raise ValueError("Upstream receipt changed: " + str(path))
    return verified(path, status)


def analysis_fit_fingerprint():
    names = {"analyze", "discover_checkpoint", "fold_data", "views", "view_title", "subgroup_description"}
    constants = {"VERSION", "DIMENSIONS", "SEMANTIC", "APPROACHES", "PRIORITY"}
    parts = {}
    for node in ast.parse(Path(__file__).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            parts[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    parts[target.id] = ast.dump(node, include_attributes=False)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def compatible_ranking_repair(saved, current):
    if saved.get("stage") != "analyze" or saved.get("code_hashes", {}).get("vehicle_similarity_v4.py") != RANKING_REPAIR_BASE:
        return False
    adjusted = {**current, "code_hashes": {**current.get("code_hashes", {}), "vehicle_similarity_v4.py": RANKING_REPAIR_BASE}}
    return adjusted == saved and analysis_fit_fingerprint() == RANKING_REPAIR_FIT


def stage(path, contract, resume=False):
    global ACTIVE_STAGE
    path = Path(path)
    if resume:
        saved = r.read_json(path / "stage_state.json") if path.is_dir() else None
        repair = saved is not None and saved != contract and compatible_ranking_repair(saved, contract)
        if saved is None or (saved != contract and not repair):
            raise ValueError("Resume requires identical paths, inputs, code and configuration")
        if (path / "manifest.json").exists() and r.read_json(path / "manifest.json").get("status") == "completed":
            raise ValueError("Completed stages are immutable; use a fresh directory")
        if (path / "manifest.json").exists():
            legacy.verify_artifacts(path)
        if repair:
            atomic_json(path / "resume_compatibility.json", {
                "reason": "Numeric category indicators for empty peer candidates; fitted partitions unchanged",
                "original_code_sha256": RANKING_REPAIR_BASE,
                "resumed_code_sha256": contract["code_hashes"]["vehicle_similarity_v4.py"],
                "fit_fingerprint": RANKING_REPAIR_FIT})
            # Preserve the original stage identity used by completed fit receipts.
            # Replace only the caller's outer dictionary; its execution receipt
            # remains the current code and is recorded when analysis completes.
            contract.clear()
            contract.update(saved)
    else:
        path.mkdir(parents=True, exist_ok=False)
        atomic_json(path / "stage_state.json", contract)
    ACTIVE_STAGE = path.resolve()
    return path


def prepare(data_dir, output):
    result = v2.prepare(data_dir, None, output, excluded_vins=EXCLUDED_LISTING_VINS, schema_version=VERSION)
    receipt = r.read_json(result / "manifest.json")
    receipt.update(execution())
    atomic_json(result / "manifest.json", receipt)
    return result


def resolved_profiles(prepared):
    p = pd.read_parquet(Path(prepared) / "variant_profiles.parquet")
    names = pd.read_parquet(Path(prepared) / "identity_names.parquet")
    crosswalk = r.read_json(v2.ASSETS / "identity_crosswalk.v2.json")
    return v2.resolve_families(p, names, crosswalk["mappings"])


def source_tasks(profiles, seed):
    tasks = []
    complete = profiles[profiles.identity_complete]
    for (make, model), group in complete.groupby(["Make", "market_family"], sort=True):
        entries = [x for x in seed["sources"] if x["make"] == make and x["model"] == model]
        tasks.append({"make": make, "model": model, "year_start": int(group.ModelYear.min()),
            "year_end": int(group.ModelYear.max()), "vin_count": int(group.vin_count.sum()),
            "key": v2.stable_id([make, model]), "seed_sources": entries,
            "priority": 0 if not any(any(c["dimension"] == "market_positioning" for c in e["claims"]) for e in entries) else 1})
    return sorted(tasks, key=lambda t: (t["priority"], -t["vin_count"], t["make"], t["model"]))


def fetch_document(url, cache):
    """Cache a response and its original text; never turn failed fetches into evidence."""
    file = Path(cache) / (r.digest(url) + ".json")
    if file.exists():
        return r.read_json(file)
    response = requests.get(url, headers=sources.HEADERS, timeout=25)
    response.raise_for_status()
    response.encoding = "utf-8"
    document = {"url": response.url, "requested_url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "html": response.text,
        "content_sha256": hashlib.sha256(response.content).hexdigest()}
    atomic_json(file, document)
    return document


def parse_document(document, task):
    """Only structured class claims with explicit model-year scope can assign labels."""
    soup = BeautifulSoup(document["html"], "html.parser")
    heading = soup.select_one("h1")
    title = heading.get_text(" ", strip=True) if heading else ""
    expected = sources.normalized(task["make"] + task["model"])
    # Parenthetical generation names may narrow identity, but cannot establish years.
    actual = sources.normalized(re.sub(r"\s*\([^)]*\)\s*$", "", title))
    # A curated URL retains its original claims; it cannot establish new claims
    # from other models mentioned on that page.
    if actual != expected:
        return [], "unmatched_identity"
    entries = []
    reason = "unavailable_classification"
    vocabulary = sources.CLASSIFICATIONS + [
        ("mainstream car", "market_positioning", "mainstream"),
        ("premium car", "market_positioning", "premium"),
        ("ultra-luxury", "market_positioning", "ultra_luxury"),
        ("sport-oriented", "performance_character", "sport_oriented_variant")]
    for table in soup.select("table.infobox"):
        rows = {}
        for row in table.select("tr"):
            key, value = row.find("th"), row.find("td")
            if key and value:
                text = value.get_text(" ", strip=True)
                rows[" ".join(key.get_text(" ", strip=True).lower().split())] = text
        passage, years = rows.get("class", ""), rows.get("model years", "")
        if not passage:
            continue
        intervals = re.findall(r"(19\d{2}|20\d{2})\s*[–—-]\s*(present|19\d{2}|20\d{2})", years.lower())
        if len(intervals) != 1 or re.search(r"\([^)]*\d{4}", passage):
            reason = "ambiguous_scope"
            continue
        if re.search(r"\b(trim|variant|edition)\b", passage, re.I):
            reason = "ambiguous_scope"
            continue
        start, end = intervals[0]
        start, end = max(int(start), task["year_start"]), min(
            datetime.now(timezone.utc).year if end == "present" else int(end), task["year_end"])
        if start > end:
            reason = "out_of_scope"
            continue
        text = passage.lower().replace("‑", "-")
        claims = [{"dimension": dim, "label": label} for phrase, dim, label in vocabulary
                  if re.search(r"(?<![a-z-])" + re.escape(phrase) + r"(?![a-z-])", text)]
        if "ultra_luxury" in {c["label"] for c in claims}:
            claims = [c for c in claims if c["label"] != "luxury"]
        if not claims:
            continue
        entries.append({"id": "auto-" + v2.stable_id([task["make"], task["model"], document["url"], start, end, passage]),
            "make": task["make"], "model": task["model"], "year_start": start, "year_end": end,
            "url": document["url"], "excerpt": passage + "; model years: " + years,
            "scope_note": "Exact model identity; structured class and model-year scope. All model-line trims; variant-specific claims excluded.",
            "all_trims": True, "claims": claims, "retrieved_at": document["retrieved_at"],
            "published_at": None, "content_sha256": document["content_sha256"]})
    return entries, "source_supported" if entries else reason


def acquire_task(task, cache):
    entries, attempts = [], []
    # Curated manufacturer references are retained with their original exact scope.
    candidates = [(e["url"], e) for e in task["seed_sources"]]
    candidates.sort(key=lambda item: "wikipedia.org" in item[0])
    title = task["make"].title() + " " + task["model"]
    candidates.append(("https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")), None))
    seen = set()
    for url, seed in candidates:
        if url in seen:
            continue
        seen.add(url)
        try:
            doc = fetch_document(url, cache)
            # A curated URL proves identity only within the curated year/trim scope.
            scoped = {**task, "year_start": max(task["year_start"], seed["year_start"]),
                      "year_end": min(task["year_end"], seed["year_end"])} if seed else task
            parsed, status = parse_document(doc, scoped)
            if seed:
                for entry in parsed:
                    entry["all_trims"] = seed.get("all_trims", False)
                    entry["trims"] = seed.get("trims", [])
                    if seed.get("body_styles"):
                        entry["body_styles"] = seed["body_styles"]
            entries.extend(parsed)
            attempts.append({"url": url, "status": status})
        except (requests.RequestException, ValueError) as error:
            attempts.append({"url": url, "status": "reference_unavailable", "error": str(error)[:250]})
    if not entries:
        # Search discovers page titles, never labels or identity aliases on its own.
        api = "https://en.wikipedia.org/w/api.php?" + urlencode({"action": "query", "list": "search",
            "srsearch": task["make"] + " " + task["model"], "srlimit": 5, "format": "json"})
        try:
            response = json.loads(fetch_document(api, cache)["html"])
            hits = response.get("query", {}).get("search", [])
            expected = sources.normalized(task["make"] + task["model"])
            for hit in hits:
                title = hit["title"]
                if sources.normalized(re.sub(r"\s*\([^)]*\)\s*$", "", title)) != expected:
                    continue
                url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
                if url in seen:
                    continue
                parsed, status = parse_document(fetch_document(url, cache), task)
                entries.extend(parsed)
                attempts.append({"url": url, "status": status})
        except (requests.RequestException, ValueError, KeyError) as error:
            attempts.append({"url": api, "status": "reference_unavailable", "error": str(error)[:250]})
    return {"make": task["make"], "model": task["model"], "entries": entries, "attempts": attempts,
            "status": "source_supported" if entries else attempts[-1]["status"]}


def collect_sources(prepared, output, batch_size=100, workers=2, resume=False):
    verified(prepared, "prepared")
    contract = {"stage": "collect-sources", "prepared": dependency(prepared), "batch_size": batch_size,
                "workers": workers, **execution()}
    target = stage(output, contract, resume)
    seed = r.read_json(v2.ASSETS / "evidence.v2.json")
    tasks = source_tasks(resolved_profiles(prepared), seed)
    for name in ("results", "documents"):
        (target / name).mkdir(exist_ok=True)
    atomic_json(target / "tasks.json", tasks)
    pending = []
    for task in tasks:
        file = target / "results" / (task["key"] + ".json")
        if not file.exists():
            pending.append(task)
            continue
        cached = r.read_json(file)
        expected_hash = cached.pop("result_sha256", None)
        if (cached.get("task_contract") != r.digest(json.dumps(task, sort_keys=True))
                or r.digest(json.dumps(cached, sort_keys=True)) != expected_hash):
            raise ValueError("Changed committed family outcome: " + task["model"])
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for task, result in zip(batch, pool.map(lambda t: acquire_task(t, target / "documents"), batch)):
                result["task_contract"] = r.digest(json.dumps(task, sort_keys=True))
                result["result_sha256"] = r.digest(json.dumps(result, sort_keys=True))
                atomic_json(target / "results" / (task["key"] + ".json"), result)
        print(f"Source outcomes: {len(tasks) - len(pending) + min(offset + batch_size, len(pending))}/{len(tasks)}", flush=True)
    results = [r.read_json(target / "results" / (t["key"] + ".json")) for t in tasks]
    if contract != {"stage": "collect-sources", "prepared": dependency(prepared), "batch_size": batch_size,
                    "workers": workers, **execution()}:
        raise ValueError("Source configuration or execution dependencies changed during collection")
    atomic_json(target / "reference_results.json", results)
    pd.DataFrame([{k: value for k, value in t.items() if k != "seed_sources"} for t in tasks]).to_csv(target / "selection.csv", index=False)
    r.finish_stage(target, "completed", schema_version=VERSION, requested_families=len(tasks), outcomes=len(results),
                   dependencies=[dependency(prepared)], **execution())
    return target


def category_coverage(profiles, labels, outcomes):
    rows = []
    assigned = {(pid, dim) for pid, dim in labels[["profile_id", "dimension"]].itertuples(index=False, name=None)}
    for profile in profiles.itertuples():
        for dimension in DIMENSIONS:
            reason = "source_supported_provisional" if (profile.profile_id, dimension) in assigned else "unknown"
            if dimension in ("body_style", "powertrain") and reason != "unknown":
                reason = "observed"
            if reason == "unknown":
                if not profile.identity_complete:
                    reason = "incomplete_identity"
                else:
                    reason = outcomes.get((profile.Make, profile.market_family), "missing_reference")
                    if reason == "source_supported":
                        reason = "no_applicable_claim"
            rows.append({"profile_id": profile.profile_id, "family_block": profile.family_block,
                         "dimension": dimension, "reason": reason, "vin_count": int(profile.vin_count)})
    return pd.DataFrame(rows)


def build_categories(prepared, source_dir, output):
    verified(prepared, "prepared")
    source_receipt = verified(source_dir, "completed")
    if source_receipt["dependencies"] != [dependency(prepared)]:
        raise ValueError("Source collection belongs to a different preparation")
    target = stage(output, {"stage": "build-categories", "prepared": dependency(prepared), "sources": dependency(source_dir), **execution()})
    definitions, seed = legacy.load_taxonomy(v2.ASSETS / "definitions.v2.json", v2.ASSETS / "evidence.v2.json")
    results = r.read_json(Path(source_dir) / "reference_results.json")
    expected = {(t["make"], t["model"]) for t in r.read_json(Path(source_dir) / "tasks.json")}
    actual = {(x["make"], x["model"]) for x in results}
    if len(results) != len(actual) or actual != expected:
        raise ValueError("Source outcomes do not cover the complete family inventory")
    entries = {e["id"]: e for e in seed["sources"]}
    entries.update({e["id"]: e for result in results for e in result["entries"]})
    registry = {"taxonomy_version": definitions["version"], "sources": list(entries.values())}
    atomic_json(target / "definitions.json", definitions)
    atomic_json(target / "evidence_registry.json", registry)
    legacy.load_taxonomy(target / "definitions.json", target / "evidence_registry.json")
    p, labels, exceptions = v2.profile_labels(resolved_profiles(prepared), legacy.draft_mappings(registry), definitions)
    observed = []
    for dimension in DIMENSIONS[3:]:
        for row in p[p.identity_complete & p[dimension].isin(definitions["dimensions"][dimension])].itertuples():
            label = getattr(row, dimension)
            observed.append({"profile_id": row.profile_id, "dimension": dimension, "label": label,
                "mapping_id": "observed-" + dimension, "source_url": "snapshot://vin_profile_membership.parquet",
                "evidence": f"VIN-supported {dimension}: {label}", "scope": "Decoded cohort/body/powertrain variant",
                "status": "observed", "taxonomy_version": definitions["version"], "namespace": "observed",
                "definition": definitions["dimensions"][dimension][label]})
    labels = pd.concat([labels, pd.DataFrame(observed)], ignore_index=True).drop_duplicates()
    labels.loc[labels.dimension.isin(SEMANTIC), "status"] = "source_supported_provisional"
    for dimension in DIMENSIONS:
        grouped = labels[labels.dimension.eq(dimension)].groupby("profile_id").label.agg(lambda v: ";".join(sorted(set(v))))
        p[dimension + "_labels"] = p.profile_id.map(grouped).fillna("")
    p["category_summary"] = p[[d + "_labels" for d in SEMANTIC]].apply(
        lambda row: " · ".join(x.replace("_", " ") for x in row if x) or "Unresolved", axis=1)
    coverage = category_coverage(p, labels, {(x["make"], x["model"]): x["status"] for x in results})
    if not exceptions.empty:
        conflicts = set(exceptions.loc[exceptions.reason.eq("conflicting_evidence"), ["profile_id", "dimension"]].itertuples(index=False, name=None))
        coverage.loc[[(pid, dim) in conflicts for pid, dim in coverage[["profile_id", "dimension"]].itertuples(index=False, name=None)], "reason"] = "conflicting_evidence"
    p.to_parquet(target / "variant_profiles.parquet", index=False)
    labels.to_parquet(target / "profile_labels.parquet", index=False)
    exceptions.reindex(columns=list(dict.fromkeys([*exceptions.columns, "profile_id", "dimension", "reason"]))).to_csv(target / "exceptions.csv", index=False)
    coverage.to_parquet(target / "category_coverage.parquet", index=False)
    coverage.groupby(["dimension", "reason"]).agg(profiles=("profile_id", "nunique"), families=("family_block", "nunique"), listings=("vin_count", "sum")).reset_index().to_csv(target / "category_coverage.csv", index=False)
    atomic_json(target / "identity_crosswalk.json", r.read_json(v2.ASSETS / "identity_crosswalk.v2.json"))
    r.finish_stage(target, "completed", schema_version=VERSION, semantic_accuracy="not_measured", provisional=True,
                   dependencies=[dependency(prepared), dependency(source_dir)], **execution())
    return target


def views(data, labels):
    yield "global", data.eligible
    for body in sorted(data.body_family.unique()):
        yield body, data.eligible & data.body_family.eq(body)
    yield from ((name, mask) for name, mask in v2.profile_views(data, labels) if name != "global")


def view_title(name):
    return name.replace("market_positioning__", "").replace("performance_character__", "").replace("size_segment__", "").replace("__", " · ").replace("_", " ").capitalize()


def subgroup_description(group, parent, fields):
    original = legacy.subgroup_name(group, parent, fields)
    phrases = [] if original == "overlapping specifications" else original.split(", ")
    descriptions = {"height": ("lower height", "taller height"), "length": ("shorter length", "longer length"),
        "seating": ("fewer seats", "more seats"), "front_legroom": ("less front legroom", "more front legroom"),
        "back_legroom": ("less rear legroom", "more rear legroom"), "torque_nm": ("lower torque", "higher torque")}
    for field, (lower, upper) in descriptions.items():
        if field in fields:
            center, value = parent[field].median(), group[field].median()
            if np.isfinite(center) and center > 0 and np.isfinite(value) and abs(value / center - 1) >= .10:
                phrases.append(upper if value > center else lower)
    for field in v2.CATEGORICAL:
        if field in fields and group[field].notna().sum() >= 5:
            distribution = group[field].value_counts(normalize=True)
            value = distribution.index[0]
            parent_share = parent[field].dropna().eq(value).mean()
            if distribution.iloc[0] >= .8 and distribution.iloc[0] - parent_share >= .2:
                phrases.append(str(value) + " " + field.replace("_", " "))
    return ", ".join(phrases[:3]) or "similar observed specifications"


def ranking_policy(data, memberships, config):
    index = data.set_index("profile_id")
    groups = {view: group.set_index("profile_id").cluster for view, group in memberships.groupby("view")}

    def rank(query_id, peer_ids, unused):
        query, peers = index.loc[query_id], index.loc[peer_ids]
        matches, count = np.zeros(len(peers)), 0
        for dimension in SEMANTIC:
            wanted = set(str(query[dimension + "_labels"]).split(";")) - {""}
            if wanted:
                matches += np.fromiter((bool(wanted & set(str(value).split(";")))
                                       for value in peers[dimension + "_labels"]), dtype=np.float64, count=len(peers))
                count += 1
        scores = matches / max(1, count)
        penalties, shared = np.zeros(len(peers)), np.zeros(len(peers))
        for view, group in groups.items():
            if "__" not in view or query_id not in group or group[query_id] < 0:
                continue
            other = group.reindex(peer_ids).fillna(-1).to_numpy()
            known = other >= 0
            penalties += np.where(known, (other != group[query_id]) * config.cluster_penalty, 0)
            shared += known
        penalties = np.divide(penalties, shared, out=np.zeros(len(peers)), where=shared > 0)
        zero = np.zeros(len(peers))
        return {"direct_specs": (zero, zero), "category_specs": (scores, zero), "category_cluster": (scores, penalties)}
    return rank


def discover_checkpoint(data, fields, destination, config, contract):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for attempt in sorted(destination.glob("attempt-*")):
        if (attempt / "manifest.json").exists():
            receipt = verified(attempt, "completed")
            if receipt["contract"] != contract:
                raise ValueError("Cluster checkpoint contract changed")
            return ({m: joblib.load(attempt / (m + ".joblib")) for m in PRIORITY if (attempt / (m + ".joblib")).exists()},
                    r.read_json(attempt / "records.json"), r.read_json(attempt / "sensitivity.json"))
    attempt = destination / ("attempt-" + uuid.uuid4().hex)
    models, records, sensitivity, _ = v3.discover(data, fields, attempt, config)
    # JSON receipts must use null for unevaluated values, never NaN.
    atomic_json(attempt / "records.json", json.loads(pd.DataFrame(records).to_json(orient="records")))
    atomic_json(attempt / "sensitivity.json", json.loads(pd.DataFrame(sensitivity).to_json(orient="records")))
    r.finish_stage(attempt, "completed", schema_version=VERSION, contract=contract)
    return models, records, sensitivity


def fold_data(profiles, membership, fold):
    data = profiles.copy()
    data["development"] = data.fold.ne(fold)
    training_ids = pl.from_pandas(data.loc[data.development & data.identity_complete, ["profile_id"]])
    pairs = membership.join(training_ids.lazy(), on="profile_id", how="inner").filter(
        pl.col("horsepower").is_finite() & pl.col("EngineHP").is_finite() & (pl.col("horsepower") > 0) & (pl.col("EngineHP") > 0)
    ).select((pl.col("horsepower") / pl.col("EngineHP")).alias("ratio")).collect()["ratio"]
    verified_scale = len(pairs) >= 30 and .9 <= pairs.median() <= 1.1
    data, fields, audit = v2.select_fields(data, excluded_fields=() if verified_scale else ("horsepower",))
    metric = v3.fit_metric(data[data.development & data.eligible], fields)
    return data, fields, audit, metric, {"fold": fold, "paired_training_vins": len(pairs), "verified": bool(verified_scale)}


def analyze(prepared, categories, output, smoke=False, threads=1, resume=False):
    prep = verified(prepared, "prepared")
    cat = verified(categories, "completed")
    if cat["dependencies"][0] != dependency(prepared):
        raise ValueError("Categories belong to a different preparation")
    config, code = v3.config_for(smoke), execution()
    contract = {"stage": "analyze", "prepared": dependency(prepared), "categories": dependency(categories),
                "configuration": json.loads(json.dumps(asdict(config))), "threads": threads, **code}
    target = stage(output, contract, resume)
    profiles = pd.read_parquet(Path(categories) / "variant_profiles.parquet")
    labels = pd.read_parquet(Path(categories) / "profile_labels.parquet")
    evidence, profiles = v3.evidence_summary(prepared, profiles)
    evidence.to_parquet(target / "vin_evidence.parquet", index=False)
    profiles["fold"] = profiles.family_block.map(v3.family_fold)
    profiles["schema_version"] = VERSION
    candidates = profiles[v3.eligible(profiles, v2.NUMERIC + v2.CATEGORICAL)]
    if candidates.family_block.nunique():
        v3.freeze_queries(profiles, replace(config, queries=min(config.queries, candidates.family_block.nunique()))).to_csv(target / "evaluation_queries.csv", index=False)
    else:
        pd.DataFrame(columns=["card_id", "profile_id", "family_block"]).to_csv(target / "evaluation_queries.csv", index=False)
    membership = pl.scan_parquet(Path(prepared) / "vin_profile_membership.parquet")
    peer_rows, status_rows, member_rows, compositions, experiments, sensitivities, audits, scales = [], [], [], [], [], [], [], []
    profiles["eligible"] = False
    with threadpool_limits(limits=threads):
        for fold in range(config.folds):
            print("FOLD", fold, flush=True)
            data, fields, audit, metric, scale = fold_data(profiles, membership, fold)
            profiles.loc[profiles.fold.eq(fold), "eligible"] = data.loc[profiles.fold.eq(fold), "eligible"]
            audits.append(audit.assign(fold=fold))
            scales.append(scale)
            selected_members = []
            for view, mask in views(data, labels):
                population = data[mask]
                training = population[population.development]
                checkpoint = target / "clusters" / f"fold_{fold}" / r.digest(view)[:20]
                models, records, sensitivity = discover_checkpoint(training, fields, checkpoint, config,
                    {"stage_contract": r.digest(json.dumps(contract, sort_keys=True)), "fold": fold, "view": view, "fields": fields})
                experiments.extend({"fold": fold, "view": view, **row} for row in records)
                sensitivities.extend({"fold": fold, "view": view, **row} for row in sensitivity)
                chosen = next((m for m in PRIORITY if m in models), None)
                for method, model in models.items():
                    assigned = v3.assign(model, population)
                    members = population[["profile_id", "family_block", "fold", "body_family"]].copy()
                    members["cluster"], members["training_fold"], members["view"], members["method"] = assigned, fold, view, method
                    member_rows.append(members)
                    if chosen == method:
                        selected_members.append(members)
                    # Names and examples use development members only, never held-out characteristics.
                    for cluster, group in population.assign(cluster=assigned).query("cluster >= 0").groupby("cluster"):
                        development = group[group.development]
                        description = view_title(view) + " · " + subgroup_description(development, training, fields)
                        compositions.append({"fold": fold, "view": view, "method": method, "cluster": int(cluster),
                            "description": description, "profiles": len(group), "families": group.family_block.nunique(),
                            "listings": int(group.vin_count.sum()), "training_profiles": len(development),
                            "examples": "; ".join(" / ".join(str(row[k]) for k in r.COHORT) for row in development.sort_values("profile_id").drop_duplicates("family_block").head(3).to_dict("records")),
                            **{f + "_median": development[f].median() for f in fields if f not in v2.CATEGORICAL}})
            selected = pd.concat(selected_members, ignore_index=True) if selected_members else pd.DataFrame(columns=["view", "profile_id", "cluster"])
            callback = ranking_policy(data, selected, config)
            peers, statuses = v3.retrieve(data, data.loc[data.fold.eq(fold), "profile_id"], fields, metric, {}, config, callback, APPROACHES)
            v3.validate_peers(data, peers, statuses, fields, metric)
            peer_rows.append(peers)
            status_rows.append(statuses)
            joblib.dump({"metric": metric, "fields": fields, "training_families": sorted(data.loc[data.development, "family_block"].unique()),
                         "selected_memberships": selected}, target / f"fold_{fold}_retrieval.joblib")
    profiles.to_parquet(target / "variant_profiles.parquet", index=False)
    profiles[["profile_id", "family_block", "fold"]].to_csv(target / "fold_membership.csv", index=False)
    peers, statuses = pd.concat(peer_rows, ignore_index=True), pd.concat(status_rows, ignore_index=True)
    peers.to_parquet(target / "profile_peers.parquet", index=False)
    statuses.to_parquet(target / "query_status.parquet", index=False)
    (pd.concat(member_rows, ignore_index=True) if member_rows else pd.DataFrame(columns=["profile_id", "family_block", "fold", "body_family", "cluster", "training_fold", "view", "method"])).to_parquet(target / "cluster_memberships.parquet", index=False)
    pd.DataFrame(compositions, columns=list(dict.fromkeys(["fold", "view", "method", "cluster", "description", "profiles", "families", "listings", "training_profiles", "examples", *[f + "_median" for f in v2.NUMERIC]]))).to_csv(target / "cluster_profiles.csv", index=False)
    pd.DataFrame(experiments).to_csv(target / "cluster_experiments.csv", index=False)
    pd.DataFrame(sensitivities, columns=list(dict.fromkeys(["fold", "view", "method", "k", "sensitivity", "status", "ari", "seed", *[key for row in sensitivities for key in row]]))).to_csv(target / "cluster_sensitivities.csv", index=False)
    pd.concat(audits).to_csv(target / "feature_support.csv", index=False)
    pd.DataFrame(scales).to_csv(target / "numeric_scale_audit.csv", index=False)
    v3.coverage(profiles, statuses).to_csv(target / "coverage.csv", index=False)
    v3.diagnostics(statuses, peers).to_csv(target / "benchmark_diagnostics.csv", index=False)
    for filename in ("profile_labels.parquet", "category_coverage.parquet", "category_coverage.csv", "exceptions.csv", "definitions.json", "evidence_registry.json", "identity_crosswalk.json"):
        shutil.copyfile(Path(categories) / filename, target / filename)
    shutil.copyfile(Path(prepared) / "vin_profile_membership.parquet", target / "vin_profile_membership.parquet")
    atomic_json(target / "evaluation.json", {"selected_approach": "direct_specs", "buyer_usefulness": "not_measured",
        "semantic_accuracy": "not_measured", "category_preferences": "explicit_provisional_user_choice", "cluster_ranking": "experimental"})
    if code != execution():
        raise ValueError("Code or taxonomy changed during analysis")
    r.finish_stage(target, "completed", schema_version=VERSION, reduced_budget=smoke, configuration=asdict(config),
        prepared_directory=str(Path(prepared).resolve()), prepared_manifest_sha256=r.file_hash(Path(prepared) / "manifest.json"),
        source_directory=prep["source_directory"], source_hashes=prep["source_hashes"], excluded_vins=prep["excluded_vins"],
        dependencies=[dependency(prepared), dependency(categories)], **code)
    return target


def analysis_output_fingerprint():
    ranking = next(node for node in ast.parse(Path(__file__).read_text(encoding="utf-8")).body
                   if isinstance(node, ast.FunctionDef) and node.name == "ranking_policy")
    return hashlib.sha256((analysis_fit_fingerprint() + ast.dump(ranking, include_attributes=False)).encode()).hexdigest()


def validation_execution_compatibility(receipt):
    current = execution()
    recorded = {key: receipt[key] for key in current}
    if recorded == current:
        return None
    adjusted = {**current, "code_hashes": {**current["code_hashes"], "vehicle_similarity_v4.py": VALIDATION_REPAIR_BASE}}
    if recorded != adjusted or analysis_output_fingerprint() != VALIDATION_REPAIR_OUTPUT:
        raise ValueError("Analysis code or dependencies changed; use the recorded execution version")
    return {"reason": "Validator compares integer cluster assignments independent of integer storage width",
            "analysis_code_sha256": VALIDATION_REPAIR_BASE,
            "validator_code_sha256": current["code_hashes"]["vehicle_similarity_v4.py"],
            "analysis_output_fingerprint": VALIDATION_REPAIR_OUTPUT}


def validate_cluster_memberships(expected, actual):
    # Concatenating assignments from different estimators promotes int32 to
    # int64. Storage width is not part of a vehicle's cluster assignment.
    if (actual.index.has_duplicates or expected.index.has_duplicates
            or not pd.api.types.is_integer_dtype(actual.dtype)
            or not expected.index.equals(actual.index)
            or not np.array_equal(expected.to_numpy(), actual.to_numpy())):
        raise ValueError("Cluster memberships differ from the recorded fit")


def validate_bundle(run):
    run = Path(run)
    receipt = verified(run, "completed")
    execution_compatibility = validation_execution_compatibility(receipt)
    experiments = pd.read_csv(run / "cluster_experiments.csv")
    accepted = experiments[experiments.status.eq("accepted_partition")]
    if not accepted.empty and not (accepted.median_ari.ge(.70) & accepted.ari_q10.ge(.60)
            & accepted.balanced_min_ari.ge(.70) & accepted.silhouette.gt(0)
            & accepted.minimum_cluster_families.ge(5)
            & accepted.minimum_count.ge(accepted.required_minimum_count)).all():
        raise ValueError("Accepted cluster violates support or stability gates")
    for value in receipt["dependencies"]:
        upstream = check_dependency(value)
        for nested in upstream.get("dependencies", []):
            check_dependency(nested)
    for name, expected in receipt["source_hashes"].items():
        if r.file_hash(Path(receipt["source_directory"]) / name) != expected:
            raise ValueError("Source snapshot changed: " + name)
    if receipt["excluded_vins"] != sorted(EXCLUDED_LISTING_VINS):
        raise ValueError("Listing exclusion policy changed")
    profiles = pd.read_parquet(run / "variant_profiles.parquet")
    labels = pd.read_parquet(run / "profile_labels.parquet")
    peers = pd.read_parquet(run / "profile_peers.parquet")
    statuses = pd.read_parquet(run / "query_status.parquet")
    saved_members = pd.read_parquet(run / "cluster_memberships.parquet")
    membership = pl.scan_parquet(run / "vin_profile_membership.parquet")
    prepared, categories = [Path(d["directory"]) for d in receipt["dependencies"]]
    original = pd.read_parquet(categories / "variant_profiles.parquet")
    pd.testing.assert_frame_equal(profiles[original.columns], original, check_dtype=False)
    for name in ("profile_labels.parquet", "definitions.json", "evidence_registry.json", "identity_crosswalk.json", "category_coverage.parquet"):
        if r.file_hash(run / name) != r.file_hash(categories / name):
            raise ValueError("Category bundle differs from its checked upstream: " + name)
    if r.file_hash(run / "vin_profile_membership.parquet") != r.file_hash(prepared / "vin_profile_membership.parquet"):
        raise ValueError("VIN membership differs from preparation")
    counts = membership.select(pl.len().alias("rows"), pl.col("vin").n_unique().alias("vins"), pl.col("profile_id").null_count().alias("missing")).collect().row(0, named=True)
    if counts["rows"] != counts["vins"] or counts["missing"] or counts["rows"] != profiles.vin_count.sum() or profiles.profile_id.duplicated().any():
        raise ValueError("VIN/profile reconciliation failed")
    actual_counts = membership.group_by("profile_id").agg(pl.len().alias("actual")).collect().to_pandas().set_index("profile_id").actual
    if not profiles.set_index("profile_id").vin_count.sort_index().equals(actual_counts.sort_index().astype("int64").rename("vin_count")):
        if not np.array_equal(profiles.set_index("profile_id").vin_count.sort_index(), actual_counts.sort_index()):
            raise ValueError("Per-profile VIN counts differ")
    if membership.filter(pl.col("vin").is_in(receipt["excluded_vins"])).select(pl.len()).collect().item():
        raise ValueError("Excluded VIN in catalog")
    definitions = r.read_json(run / "definitions.json")
    if not labels.profile_id.isin(profiles.profile_id).all() or labels.duplicated(["profile_id", "dimension", "label", "mapping_id"]).any():
        raise ValueError("Invalid category identities")
    for row in labels.itertuples():
        if row.label not in definitions["dimensions"].get(row.dimension, {}) or not row.source_url or not row.evidence or not row.scope:
            raise ValueError("Invalid category evidence")
    if len(statuses) != len(profiles) * len(APPROACHES) or set(statuses.approach) != set(APPROACHES):
        raise ValueError("Incomplete query denominators")
    config = replace(v3.Config(), **{**receipt["configuration"], "clusters": tuple(receipt["configuration"]["clusters"])})
    for fold in range(config.folds):
        data, fields, _, metric, _ = fold_data(profiles, membership, fold)
        saved = joblib.load(run / f"fold_{fold}_retrieval.joblib")
        if fields != saved["fields"] or metric != saved["metric"] or saved["training_families"] != sorted(data.loc[data.development, "family_block"].unique()):
            raise ValueError("Training fold preprocessing or family receipt differs")
        held_out = set(data.loc[~data.development, "family_block"])
        if held_out & set(saved["training_families"]):
            raise ValueError("Held-out family entered training")
        masks = dict(views(data, labels))
        for checkpoint_manifest in (run / "clusters" / f"fold_{fold}").glob("*/attempt-*/manifest.json"):
            checkpoint = verified(checkpoint_manifest.parent, "completed")
            view = checkpoint["contract"]["view"]
            if checkpoint["contract"]["fold"] != fold or checkpoint["contract"]["fields"] != fields or view not in masks:
                raise ValueError("Invalid cluster context")
            directory = checkpoint_manifest.parent
            population = data[masks[view]]
            training = population[population.development]
            selection_path = directory / "selection.json"
            if selection_path.exists():
                selection = r.read_json(selection_path)
                fit_families, anchor_families = set(selection["training_families"]), set(selection["anchor_families"])
                if (fit_families & anchor_families or (fit_families | anchor_families) - set(training.family_block)):
                    raise ValueError("Cluster anchor or training families cross their fold")
            for method in PRIORITY:
                if not (directory / (method + ".joblib")).exists():
                    continue
                if len(training) < config.minimum_profiles or training.family_block.nunique() < config.minimum_families:
                    raise ValueError("Unsupported category was subdivided")
                model = joblib.load(directory / (method + ".joblib"))
                actual = saved_members[saved_members.training_fold.eq(fold) & saved_members.view.eq(view) & saved_members.method.eq(method)]
                expected_clusters = pd.Series(v3.assign(model, population), index=population.profile_id, name="cluster").sort_index()
                validate_cluster_memberships(expected_clusters, actual.set_index("profile_id").cluster.sort_index())
        relevant = peers[peers.query_profile_id.isin(data.loc[data.fold.eq(fold), "profile_id"])].reset_index(drop=True)
        status = statuses[statuses.fold.eq(fold)]
        v3.validate_peers(data, relevant, status, fields, metric)
        expected, expected_status = v3.retrieve(data, data.loc[data.fold.eq(fold), "profile_id"], fields, metric, {}, config,
            ranking_policy(data, saved["selected_memberships"], config), APPROACHES)
        pd.testing.assert_frame_equal(relevant, expected.reset_index(drop=True), check_dtype=False)
        pd.testing.assert_frame_equal(status.reset_index(drop=True), expected_status.reset_index(drop=True), check_dtype=False)
    return {"passed": True, "run": dependency(run),
        "checks": ["sources", "dependencies", "VIN_membership", "category_provenance", "family_separation", "cluster_membership", "preprocessing", "peer_reproduction"],
        "semantic_accuracy": "not_measured", "buyer_usefulness": "not_measured", "reduced_budget": receipt["reduced_budget"],
        "execution_compatibility": execution_compatibility}


def validate(run, output):
    target = stage(output, {"stage": "validate", "run": dependency(run), **execution()})
    try:
        details = validate_bundle(run)
        atomic_json(target / "validation.json", details)
        r.finish_stage(target, "completed", schema_version=VERSION, run=dependency(run), passed=True,
                       validation_code_sha256=r.file_hash(__file__))
    except BaseException as error:
        atomic_json(target / "validation.json", {"passed": False, "error": str(error), "run": dependency(run)})
        r.finish_stage(target, "failed", schema_version=VERSION, passed=False)
        raise
    return target


def activate(run, validation, registry_path=None):
    run = Path(run).resolve()
    receipt = verified(run, "completed")
    report = verified(validation, "completed")
    details = r.read_json(Path(validation) / "validation.json")
    if receipt.get("reduced_budget", True) or not report.get("passed") or not details.get("passed") or details["run"] != dependency(run) or report["run"] != dependency(run):
        raise ValueError("Activation requires matching validation and a full-budget run")
    if report.get("validation_code_sha256") != r.file_hash(__file__):
        raise ValueError("Validation code changed; validate again")
    pending = list(receipt["dependencies"])
    while pending:
        upstream = check_dependency(pending.pop())
        pending.extend(upstream.get("dependencies", []))
    for name, expected in receipt["source_hashes"].items():
        if r.file_hash(Path(receipt["source_directory"]) / name) != expected:
            raise ValueError("Source changed since validation")
    if receipt["excluded_vins"] != sorted(EXCLUDED_LISTING_VINS):
        raise ValueError("Listing exclusion policy changed")
    registry_path = Path(registry_path) if registry_path else r.ROOT / "car_app/artifact_registry.json"
    lock = registry_path.with_suffix(".lock")
    lock.open("x").close()
    try:
        before = registry_path.read_bytes()
        config = json.loads(before.decode("utf-8-sig"))
        previous = {key: config.get(key) for key in ("similarity_run", "similarity_manifest_sha256")}
        try:
            run_name = run.relative_to(r.ROOT / "MODEL_OUTPUT").as_posix()
        except ValueError:
            run_name = str(run)
        config.update(similarity_run=run_name, similarity_manifest_sha256=r.file_hash(run / "manifest.json"))
        backup = run.parent / ("activation-" + uuid.uuid4().hex)
        backup.mkdir()
        atomic_json(backup / "previous_similarity.json", previous)
        if registry_path.read_bytes() != before:
            raise ValueError("Registry changed concurrently; activation stopped")
        atomic_json(registry_path, config)
    finally:
        lock.unlink(missing_ok=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "collect-sources", "build-categories", "analyze", "validate", "activate"])
    for name in ("data-dir", "prepared", "sources", "categories", "run", "validation", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--all-families", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    required = {"prepare": ["data_dir", "output"], "collect-sources": ["prepared", "output", "all_families"],
        "build-categories": ["prepared", "sources", "output"], "analyze": ["prepared", "categories", "output"],
        "validate": ["run", "output"], "activate": ["run", "validation"]}
    for name in required[args.stage]:
        if not getattr(args, name):
            parser.error(args.stage + " requires --" + name.replace("_", "-"))
    if min(args.batch_size, args.workers, args.threads) < 1:
        parser.error("Batch size, workers and threads must be positive")
    if args.resume and args.stage not in ("collect-sources", "analyze"):
        parser.error("Only collect-sources and analyze support --resume")
    try:
        if args.stage == "prepare":
            result = prepare(args.data_dir, args.output)
        elif args.stage == "collect-sources":
            result = collect_sources(args.prepared, args.output, args.batch_size, args.workers, args.resume)
        elif args.stage == "build-categories":
            result = build_categories(args.prepared, args.sources, args.output)
        elif args.stage == "analyze":
            result = analyze(args.prepared, args.categories, args.output, args.smoke, args.threads, args.resume)
        elif args.stage == "validate":
            result = validate(args.run, args.output)
        else:
            result = activate(args.run, args.validation)
        print(result)
    except BaseException as error:
        if args.output and args.output.resolve() == ACTIVE_STAGE and args.output.is_dir():
            old = r.read_json(args.output / "manifest.json") if (args.output / "manifest.json").exists() else {}
            if old.get("status") != "completed":
                r.finish_stage(args.output, "interrupted" if isinstance(error, KeyboardInterrupt) else "blocked", schema_version=VERSION, error=str(error))
        raise


if __name__ == "__main__":
    main()
