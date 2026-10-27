"""Versioned evidence repair, reviewed classifiers and independent discovery grains.

Run ``python -m MODELS.vehicle_similarity_v5 --help``. No price training or source
table writes occur. Completed artifacts are immutable and activation is explicit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib.metadata import version
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import uuid

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from EDA.DATA_CLEANING import EXCLUDED_LISTING_VINS
from MODELS import vehicle_profiles as r
from MODELS import vehicle_taxonomy as taxonomy
from MODELS import vehicle_taxonomy_v2 as v2
from MODELS import vehicle_similarity_v3 as v3
from MODELS import vehicle_similarity_v4 as v4
from MODELS import vehicle_profiles_v5 as features
from MODELS import vehicle_sources_v5 as sources
from MODELS import vehicle_classifiers_v5 as classifiers
from MODELS import vehicle_review_v5 as peer_review

VERSION = "5.0.0"
DIMENSIONS = (*v4.DIMENSIONS, "drivetrain", "epa_class", "nhtsa_class")
LABEL_COLUMNS = ["profile_id", "dimension", "label", "status", "namespace", "source_url", "evidence", "scope", "mapping_id", "probability", "policy"]
APPROACHES = ("direct_specs", "category_specs", "category_cluster", "broader_category_cluster")


def execution():
    modules = [__file__, features.__file__, sources.__file__, classifiers.__file__, peer_review.__file__, v2.__file__, v3.__file__, v4.__file__, taxonomy.__file__, r.__file__,sources.vocabulary.__file__]
    return {"code_hashes": {Path(p).name: r.file_hash(p) for p in modules}, "taxonomy_hashes": {p.name: r.file_hash(p) for p in v2.ASSETS.glob("*.v2.json")},
        "versions": {name:version(name) for name in ["numpy","pandas","polars","scipy","scikit-learn","catboost","requests","beautifulsoup4","joblib"]}}


def analysis_fingerprint(source):
    functions={"analyze","select_fields","fit_candidates","assign","partition_score","discover","ranked_peers"}
    constants={"VERSION","DIMENSIONS","LABEL_COLUMNS","APPROACHES"}
    nodes=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in functions:
            nodes.append(ast.dump(node,include_attributes=False))
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):
            nodes.append(ast.dump(node,include_attributes=False))
    return hashlib.sha256(json.dumps(nodes).encode()).hexdigest()


def validation_compatibility(run, receipt):
    current=execution()
    recorded=receipt["execution"]
    if recorded==current:
        return None
    name=Path(__file__).name
    changed={**current,"code_hashes":{**current["code_hashes"],name:recorded["code_hashes"][name]}}
    frozen=Path(run)/"execution"/name
    if recorded!=changed or not frozen.exists() or r.file_hash(frozen)!=recorded["code_hashes"][name]:
        raise ValueError("Analysis dependencies or frozen execution changed; rebuild")
    saved=analysis_fingerprint(frozen.read_text(encoding="utf-8"))
    if saved!=analysis_fingerprint(Path(__file__).read_text(encoding="utf-8")):
        raise ValueError("Fitting or ranking code changed; rebuild")
    return {"reason":"Validator-only repair; archived fitting and ranking AST unchanged", "analysis_sha256":recorded["code_hashes"][name],"validator_sha256":current["code_hashes"][name],"analysis_fingerprint":saved}


def verify(path, status=None):
    value = taxonomy.verify_artifacts(Path(path))
    if value.get("schema_version") != VERSION or status and value.get("status") != status:
        raise ValueError("Incompatible or incomplete v5 stage: " + str(path))
    return value


def dependency(path):
    verify(path)
    return {"directory": str(Path(path).resolve()), "manifest_sha256": r.file_hash(Path(path) / "manifest.json")}


def verify_dependencies(receipt):
    pending=list(receipt.get("dependencies",[]))
    seen=set()
    while pending:
        item=pending.pop()
        path=Path(item["directory"])
        if str(path) in seen:
            continue
        seen.add(str(path))
        if r.file_hash(path/"manifest.json")!=item["manifest_sha256"]:
            raise ValueError("Dependency receipt changed: "+str(path))
        pending.extend(verify(path,"completed").get("dependencies",[]))


def begin(path, stage, inputs=(), resume=False, **configuration):
    path = Path(path)
    contract = {"stage": stage, "dependencies": [dependency(p) for p in inputs], "execution": execution(), "configuration": configuration}
    if resume:
        if not (path / "stage_state.json").exists() or r.read_json(path / "stage_state.json") != contract:
            raise ValueError("Resume requires identical dependencies, code and configuration")
        if (path / "manifest.json").exists() and r.read_json(path / "manifest.json")["status"] == "completed":
            raise ValueError("Completed stages are immutable")
    else:
        path.mkdir(parents=True, exist_ok=False)
        v4.atomic_json(path / "stage_state.json", contract)
        frozen=path/"execution"
        frozen.mkdir()
        for name in contract["execution"]["code_hashes"]:
            shutil.copyfile(Path(__file__).with_name(name),frozen/name)
        assets=frozen/"taxonomy"
        assets.mkdir()
        for file in v2.ASSETS.glob("*.v2.json"):
            shutil.copyfile(file,assets/file.name)
    r.STAGE_STARTED[str(path)] = time.perf_counter()
    return path


def finish(path, **details):
    state = r.read_json(Path(path) / "stage_state.json")
    if state["execution"] != execution():
        raise ValueError("Code changed during stage execution")
    return r.finish_stage(Path(path), "completed", schema_version=VERSION, dependencies=state["dependencies"],
                          execution=state["execution"], **details)


def prepare(data_dir, output, resume=False):
    data_dir = Path(data_dir)
    hashes = {name: r.file_hash(data_dir / name) for name in ("used_cars.parquet", "NHTSA_vPIC.parquet")}
    target = begin(output, "prepare", resume=resume, source_hashes=hashes)
    frame, conflicts = features.join_sources(data_dir)
    print("Joined retained VINs:", frame.height, flush=True)
    frame.write_parquet(target / "vin_evidence.parquet")
    for grain in ("cohort", "model"):
        path = target / grain
        path.mkdir(exist_ok=True)
        profiles, membership = features.aggregate(frame, grain)
        profiles.to_parquet(path / "variant_profiles.parquet", index=False)
        membership.write_parquet(path / "vin_profile_membership.parquet")
        pairs = frame.select("vin", (pl.col("listing_horsepower").cast(pl.Float64, strict=False) / pl.col("EngineHP").cast(pl.Float64, strict=False)).alias("ratio")).filter(pl.col("ratio").is_finite() & (pl.col("ratio") > 0)).join(membership,on="vin",validate="1:1")
        pairs.select("profile_id","ratio").write_parquet(path / "horsepower_scale_pairs.parquet")
        print(grain, len(profiles), "profiles; eligible identities", int(profiles.identity_complete.sum()), flush=True)
    for name, expected in hashes.items():
        if r.file_hash(data_dir / name) != expected:
            raise ValueError("Source changed during preparation")
    return finish(target, source_directory=str(data_dir.resolve()), source_hashes=hashes,
        excluded_vins=sorted(EXCLUDED_LISTING_VINS), retained_listings=frame.height, conflicting_decode_vins=conflicts)


def all_profiles(prepared):
    return pd.concat([pd.read_parquet(Path(prepared) / grain / "variant_profiles.parquet") for grain in ("cohort", "model")], ignore_index=True)


def collect_sources(prepared, output, reuse=None, limit=None, resume=False, offline=False):
    prep = verify(prepared, "completed")
    target = begin(output, "collect-sources", [prepared], resume, reuse=str(Path(reuse).resolve()) if reuse else None, reuse_manifest_sha256=r.file_hash(Path(reuse)/"manifest.json") if reuse else None, limit=limit, offline=offline)
    (target / "results").mkdir(exist_ok=True)
    client = sources.ReferenceClient(target / "documents")
    data = all_profiles(prepared)
    tasks = []
    for (make, model), group in data[data.identity_complete].groupby(["Make", "market_family"], sort=True):
        aliases = {sources.vocabulary.normalized(make + m) for m in group.Model.unique()}
        tasks.append({"make": make, "model": model, "year_start": int(group.ModelYear.min()), "year_end": int(group.ModelYear.max()),
            "accepted_titles": sorted(aliases), "vin_count": int(group[group.grain.eq("model")].vin_count.sum())})
    tasks.sort(key=lambda t: (-t["vin_count"], t["make"], t["model"]))
    old = {}
    old_tasks={}
    if reuse:
        reused = taxonomy.verify_artifacts(Path(reuse))
        old = {(x["make"], x["model"]): x for x in r.read_json(Path(reuse) / "reference_results.json")}
        old_tasks={(x["make"],x["model"]):x for x in r.read_json(Path(reuse)/"tasks.json")}
        for cached in (Path(reuse) / "documents").glob("*.json"):
            destination = client.directory / cached.name
            if not destination.exists():
                document = r.read_json(cached)
                if document.get("html") and document.get("content_sha256"):
                    document.setdefault("status", "success")
                sources.save(destination, document)
        for name in ("epa_vehicles.parquet","epa_receipt.json"):
            if (Path(reuse)/name).exists():
                shutil.copyfile(Path(reuse)/name,target/name)
    cached=sources.cached_pages(client.directory) if offline else {}
    # Preserve every verified success, including families outside a bounded pilot.
    for task in tasks:
        previous=old.get((task["make"],task["model"]),{})
        previous_task=old_tasks.get((task["make"],task["model"]),{})
        same_scope=all(previous_task.get(key)==task.get(key) for key in ("make","model","year_start","year_end","accepted_titles"))
        reusable_terminal=same_scope and previous.get("parser_version")==sources.PARSER_VERSION and previous.get("status") in ("ambiguous_scope","out_of_scope","unavailable_classification","unmatched_identity","missing_page")
        file=target/"results"/(v2.stable_id([task["make"],task["model"]])+".json")
        if (previous.get("entries") or reusable_terminal) and not file.exists():
            sources.save(file,{**previous,"reused_success":bool(previous.get("entries")),"reused_outcome":True})
    selected = tasks[:limit] if limit else tasks
    v4.atomic_json(target / "tasks.json", tasks)
    for number, task in enumerate(selected, 1):
        file = target / "results" / (v2.stable_id([task["make"], task["model"]]) + ".json")
        if file.exists() and r.read_json(file).get("status") not in ("deferred", "offline_pending", "pending"):
            continue
        previous = old.get((task["make"], task["model"]), {})
        if previous.get("entries"):
            result = {**previous, "reused_success": True}
        elif offline:
            entries,attempts=[],[]
            for key in set(task["accepted_titles"])|{sources.vocabulary.normalized(task["make"]+task["model"])}:
                for document in cached.get(key,[]):
                    found,status=sources.parse_page(document,task)
                    entries.extend(found)
                    attempts.append({"status":status,"url":document["url"],"offline_reparse":True})
            reasons={a["status"] for a in attempts}
            status="source_supported" if entries else next((s for s in ["ambiguous_scope","out_of_scope","unavailable_classification","unmatched_identity"] if s in reasons),previous.get("status","offline_pending"))
            result = {"make": task["make"], "model": task["model"], "entries": entries, "attempts": attempts, "status": status}
        else:
            result = sources.acquire(task, client)
        v4.atomic_json(file, result)
        if number % 10 == 0 or result["status"] == "deferred":
            print(f"References {number}/{len(selected)}: {result['status']}; requests={client.requests}; cache_hits={client.cache_hits}", flush=True)
        if result["status"] == "deferred":
            # A host cooldown pauses collection; resume uses the persisted retry queue.
            break
    results = []
    for task in tasks:
        file = target / "results" / (v2.stable_id([task["make"], task["model"]]) + ".json")
        result = r.read_json(file) if file.exists() else {"make": task["make"], "model": task["model"], "entries": [], "attempts": [], "status": "pending"}
        results.append(result)
    v4.atomic_json(target / "reference_results.json", results)
    epa_status = "success" if (target/"epa_vehicles.parquet").exists() else "offline_pending"
    if not offline and epa_status!="success":
        try:
            document = client.get(sources.EPA_URL, binary=True)
            if document["status"] == "success":
                pd.read_csv(document["payload"], compression="zip", low_memory=False).to_parquet(target / "epa_vehicles.parquet", index=False)
                v4.atomic_json(target / "epa_receipt.json", {k: document[k] for k in ("requested_url", "canonical_url", "content_sha256", "retrieved_at")})
                epa_status = "success"
            else:
                epa_status = document["status"]
        except sources.DeferredReference:
            epa_status = "deferred"
    pd.DataFrame([{k: t[k] for k in ("make", "model", "vin_count")} | {"status": result["status"]} for t, result in zip(tasks, results)]).to_csv(target / "source_coverage.csv", index=False)
    if epa_status=="success":
        epa=pd.read_parquet(target/"epa_vehicles.parquet")
        epa.groupby(["year","VClass"]).agg(vehicle_variants=("id","nunique")).reset_index().to_csv(target/"epa_class_inventory.csv",index=False)
    v4.atomic_json(target/"classification_references.json",sources.CLASSIFICATION_REFERENCES)
    pending = sum(x["status"] in ("pending", "deferred", "offline_pending") for x in results)
    return finish(target, pending_families=pending, epa_status=epa_status, requested_families=len(tasks), reduced_budget=bool(limit), source_receipt=prep["source_hashes"])


def label_row(pid, dimension, label, status, url, evidence, scope, mapping, namespace="market_terminology"):
    return dict(zip(LABEL_COLUMNS, [pid, dimension, label, status, namespace, url, evidence, scope, mapping, 1., "verified"]))


def build_categories(prepared, source_dir, output, resume=False):
    verify(prepared, "completed")
    receipt = verify(source_dir, "completed")
    if receipt["dependencies"] != [dependency(prepared)]:
        raise ValueError("References belong to different preparation")
    target = begin(output, "build-categories", [prepared, source_dir],resume)
    definitions = r.read_json(v2.ASSETS / "definitions.v2.json")
    definitions["version"] = VERSION
    definitions["dimensions"]["powertrain"].update({"combustion_unverified": "Combustion fuel observed; electrification unverified.", "fuel_cell": "Explicit fuel-cell propulsion evidence.","hybrid_unspecified":"Hybrid evidence observed; plug-in status is unverified."})
    definitions["dimensions"]["drivetrain"] = {v: v for v in ["FWD", "RWD", "2WD", "AWD", "4WD", "ALL_WHEEL"]}
    definitions["dimensions"]["nhtsa_class"]={"PC/Mi":"Passenger car mini: curb weight 1500–1999 lb", "PC/L":"Passenger car light: 2000–2499 lb", "PC/C":"Passenger car compact: 2500–2999 lb", "PC/Me":"Passenger car medium: 3000–3499 lb", "PC/H":"Passenger car heavy: 3500 lb or more", "SUV":"Sport utility vehicle", "PU":"Pickup truck", "VAN":"Van"}
    definitions["classification_references"]=sources.CLASSIFICATION_REFERENCES
    epa = pd.read_parquet(Path(source_dir) / "epa_vehicles.parquet") if (Path(source_dir) / "epa_vehicles.parquet").exists() else None
    epa_index = sources.epa_index(epa) if epa is not None else None
    definitions["dimensions"]["epa_class"] = {v: "EPA regulatory vehicle class: " + v for v in sorted(epa.VClass.dropna().unique())} if epa is not None else {}
    definitions["namespaces"].extend(["rule_derived", "predicted", "nhtsa_ncap"])
    seed = r.read_json(v2.ASSETS / "evidence.v2.json")["sources"]
    results = r.read_json(Path(source_dir) / "reference_results.json")
    entries = {e["id"]: e for e in seed}
    entries.update({e["id"]: e for result in results for e in result["entries"]})
    mappings = taxonomy.draft_mappings({"sources": list(entries.values())})
    outcomes = {(x["make"], x["model"]): x["status"] for x in results}
    labels, exceptions, profiles, coverage = [], [], [], []
    for grain in ("cohort", "model"):
        p = pd.read_parquet(Path(prepared) / grain / "variant_profiles.parquet")
        p["observed_powertrain"]=p.powertrain
        allowed = mappings if grain == "cohort" else mappings[mappings.all_trims]
        p, semantic, errors = v2.profile_labels(p, allowed, definitions)
        if not semantic.empty:
            semantic["status"], semantic["policy"], semantic["probability"] = "source_supported", "verified", 1.
            labels.extend(semantic.to_dict("records"))
        exceptions.extend(errors.to_dict("records"))
        complete=p[p.identity_complete].to_dict("records")
        for number,row in enumerate(complete,1):
            if number%1000==0:
                print(f"Category mapping {grain}: {number}/{len(complete)} profiles",flush=True)
            nhtsa=sources.nhtsa_class(row)
            if nhtsa:
                labels.append(label_row(row["profile_id"],"nhtsa_class",nhtsa,"rule_derived","https://www.nhtsa.gov/ratings",definitions["dimensions"]["nhtsa_class"][nhtsa],"NCAP class derived from observed body and supported curb-weight bounds; no safety comparison implied","nhtsa-ncap-v5","nhtsa_ncap"))
            for dimension, field in [("body_style", "body_style"), ("powertrain", "powertrain"), ("drivetrain", "wheel_system")]:
                value = row[field]
                if pd.notna(value) and value in definitions["dimensions"][dimension]:
                    status = "rule_derived" if value in ("combustion_unverified","hybrid_unspecified") else "observed"
                    labels.append(label_row(row["profile_id"], dimension, value, status, "snapshot://vin_evidence.parquet", "Reconciled VIN evidence: " + str(value), "Observed " + grain + " body/propulsion partition", "observed-" + dimension, "rule_derived" if status == "rule_derived" else "observed"))
            if epa is not None:
                match = sources.epa_claims(row, epa, epa_index)
                for dimension in ("epa_class", "powertrain"):
                    if dimension in match:
                        if dimension == "powertrain" and row["powertrain"] not in ("unknown", "combustion_unverified", "hybrid_unspecified",match[dimension]):
                            exceptions.append({"profile_id": row["profile_id"], "dimension": dimension, "reason": "conflicting_epa_evidence"})
                            continue
                        if dimension=="powertrain" and row["powertrain"] in ("unknown","combustion_unverified","hybrid_unspecified"):
                            p.loc[p.profile_id.eq(row["profile_id"]),"powertrain"]=match[dimension]
                            p.loc[p.profile_id.eq(row["profile_id"]),"powertrain_state"]="SOURCE_SUPPORTED_EPA"
                        labels.append(label_row(row["profile_id"], dimension, match[dimension], "source_supported", f"https://www.fueleconomy.gov/feg/byclass/{int(row['ModelYear'])}ClassList.shtml", "Compatible EPA IDs: " + ";".join(match["candidate_ids"]), "All compatible EPA variants agree; exact model/baseModel or documented alias and compatible specifications", "epa-" + v2.stable_id(match["candidate_ids"]), "epa_regulatory" if dimension == "epa_class" else "observed"))
                if match["reason"] != "source_supported":
                    exceptions.append({"profile_id": row["profile_id"], "dimension": "epa_class", "reason": match["reason"]})
        profiles.append(p)
    p = pd.concat(profiles, ignore_index=True)
    labels = pd.DataFrame(labels).reindex(columns=LABEL_COLUMNS).drop_duplicates(["profile_id", "dimension", "label", "mapping_id"])
    # An EPA refinement supersedes only an unverified broad propulsion label.
    refined = set(labels.loc[labels.dimension.eq("powertrain") & labels.status.eq("source_supported"), "profile_id"])
    labels = labels[~(labels.profile_id.isin(refined) & labels.label.isin(["combustion_unverified","hybrid_unspecified"]))]
    for dimension in DIMENSIONS:
        grouped = labels[labels.dimension.eq(dimension)].groupby("profile_id").label.agg(lambda x: ";".join(sorted(set(x))))
        p[dimension + "_labels"] = p.profile_id.map(grouped).fillna("")
    p["category_summary"] = p[[d + "_labels" for d in v4.SEMANTIC]].apply(lambda x: " · ".join(v for v in x if v) or "Unresolved", axis=1)
    assigned = set(labels[["profile_id", "dimension"]].itertuples(index=False, name=None))
    errors_by_key = {(e.get("profile_id"), e.get("dimension")): e.get("reason", "conflicting_evidence") for e in exceptions}
    for row in p.itertuples():
        for dimension in DIMENSIONS:
            key = row.profile_id, dimension
            reason = "assigned" if key in assigned else errors_by_key.get(key)
            if not reason:
                if not row.identity_complete:
                    reason = "incomplete_" + row.grain + "_identity"
                elif dimension=="nhtsa_class":
                    reason="insufficient_or_cross_boundary_curb_weight" if row.body_family=="passenger_car" else "insufficient_body_evidence"
                elif dimension in ("body_style", "powertrain", "drivetrain"):
                    reason = "conflicting_local_evidence" if (getattr(row, "powertrain") == "conflicting" if dimension == "powertrain" else getattr(row, "body_style") == "conflicting" if dimension == "body_style" else pd.isna(row.wheel_system)) else "insufficient_local_evidence"
                elif dimension == "epa_class":
                    reason = "epa_" + receipt.get("epa_status", "missing_reference")
                else:
                    outcome = outcomes.get((row.Make, row.market_family), "missing_reference")
                    reason = "no_applicable_claim" if outcome == "source_supported" else outcome
            coverage.append({"profile_id": row.profile_id, "grain": row.grain, "family_block": row.family_block, "dimension": dimension, "reason": reason, "vin_count": int(row.vin_count)})
    p.to_parquet(target / "variant_profiles.parquet", index=False)
    labels.to_parquet(target / "profile_labels.parquet", index=False)
    pd.DataFrame(exceptions, columns=list(dict.fromkeys(["profile_id", "dimension", "reason", *pd.DataFrame(exceptions).columns]))).to_csv(target / "exceptions.csv", index=False)
    coverage = pd.DataFrame(coverage)
    coverage.to_parquet(target / "category_coverage.parquet", index=False)
    coverage.groupby(["grain", "dimension", "reason"]).agg(profiles=("profile_id", "nunique"), families=("family_block", "nunique"), listings=("vin_count", "sum")).reset_index().to_csv(target / "category_coverage.csv", index=False)
    v4.atomic_json(target / "definitions.json", definitions)
    v4.atomic_json(target / "evidence_registry.json", {"sources": list(entries.values())})
    return finish(target, semantic_accuracy="not_measured", source_pending_families=receipt.get("pending_families", 0))


def fit_candidates(training, fields, method, config, seed, ks=None, balanced=True, scaling="quantile"):
    if not fields or not any(v2.feature_block(f) == "powertrain" for f in fields) or sum(v2.feature_block(f) == "size" for f in fields) < 2:
        return {}
    population = training if method == "kmeans" else training.dropna(subset=fields)
    if len(population) < config.minimum_profiles or population.family_block.nunique() < config.minimum_families:
        return {}
    models = v3.fit_candidates(population, fields, method, config, seed, ks, balanced, scaling)
    for k, model in models.items():
        if method == "kmeans":
            x = model["encoder"].transform(v3.clip_frame(population, model["metric"]).replace({None: np.nan}))
            assigned = model["model"].predict(x)
            distance = np.linalg.norm(x - model["model"].cluster_centers_[assigned], axis=1)
            model["radius"] = np.asarray([np.quantile(distance[assigned == c], .95) if (assigned == c).any() else 0. for c in range(k)])
        model["fit_families"] = sorted(population.family_block.unique())
    return models


def select_fields(data, scale_pairs):
    ids = set(data.loc[data.development & data.identity_complete, "profile_id"])
    ratios = scale_pairs.loc[scale_pairs.profile_id.isin(ids), "ratio"]
    verified = len(ratios) >= 30 and .9 <= ratios.median() <= 1.1
    result, fields, audit = v2.select_fields(data, excluded_fields=() if verified else ("horsepower",))
    development = result[result.development & result.identity_complete & result.vin_count.ge(5)]
    extra_audit=[]
    for field in ("width", "engine_displacement"):
        result[field]=result[field].where(result[field+"_n"].ge(5))
        values=result.loc[development.index,field]
        reason="selected" if values.notna().mean()>=.7 and values.nunique()>1 else "coverage_or_variation"
        if reason=="selected":
            for prior in fields:
                if prior not in v2.CATEGORICAL and v2.feature_block(prior)==v2.feature_block(field) and abs(result.loc[development.index,[prior,field]].corr(method="spearman").iloc[0,1])>.9:
                    reason="redundant_with_"+prior
                    break
        if reason=="selected":
            fields.append(field)
        extra_audit.append({"field":field,"block":v2.feature_block(field),"development_coverage":values.notna().mean(),"reason":reason})
    audit=pd.concat([audit,pd.DataFrame(extra_audit)],ignore_index=True)
    audit["horsepower_scale_verified"],audit["paired_training_vins"] = bool(verified),len(ratios)
    result["eligible"]=v3.eligible(result,fields)
    return result,fields,audit


def assign(model, population):
    labels = v3.assign(model, population, reject=False)
    if not len(population):
        return labels
    if model["method"] == "kmeans":
        x = model["encoder"].transform(v3.clip_frame(population, model["metric"]).replace({None: np.nan}))
        distance = np.linalg.norm(x - model["model"].cluster_centers_[np.maximum(labels, 0)], axis=1)
    else:
        matrix = v3.distances(population, model["centers"], model["metric"])
        distance = matrix[np.arange(len(population)), np.maximum(labels, 0)]
    return np.where((labels >= 0) & np.isfinite(distance) & (distance <= model["radius"][np.maximum(labels, 0)] + 1e-12), labels, -1)


def partition_score(model, population, anchor, k):
    assigned, anchors = assign(model, population), assign(model, anchor)
    valid = anchors >= 0
    silhouette = -1.
    values = anchor.loc[valid]
    if 1 < len(set(anchors[valid])) < valid.sum():
        if model["method"] == "kmeans":
            x = model["encoder"].transform(v3.clip_frame(values, model["metric"]).replace({None: np.nan}))
            silhouette = silhouette_score(x, anchors[valid])
        else:
            # Silhouette requires a fully observed pairwise reference matrix.
            complete = values.dropna(subset=model["fields"])
            matrix = v3.distances(complete, complete, model["metric"])
            labels = assign(model, complete)
            if np.isfinite(matrix).all() and 1 < len(set(labels)) < len(labels) and (labels >= 0).all():
                np.fill_diagonal(matrix, 0)
                silhouette = silhouette_score(matrix, labels, metric="precomputed")
    counts = np.bincount(assigned[assigned >= 0], minlength=k)
    return {"silhouette": float(silhouette), "minimum_count": int(counts.min()), "minimum_cluster_families": min(population.loc[assigned == c].family_block.nunique() for c in range(k)),
        "required_minimum_count": max(10, int(np.ceil(.05 * len(population)))), "assignment_coverage": float((assigned >= 0).mean())}


def discover(population, fields, config):
    split = population.family_block.map(lambda x: int(r.digest("anchor|" + x)[:16], 16) % 5 == 0)
    training, anchor = population[~split], population[split]
    if len(population) < config.minimum_profiles or population.family_block.nunique() < config.minimum_families or training.family_block.nunique() < 10 or anchor.family_block.nunique() < 2:
        return {}, [{"method": "all", "status": "insufficient_support"}], []
    anchor = anchor.sample(min(config.anchor_cap, len(anchor)), random_state=42)
    accepted, records, sensitivity = {}, [], []
    for method in v3.METHODS:
        base = fit_candidates(training, fields, method, config, 42)
        if not base:
            records.append({"method": method, "status": "insufficient_common_observations"})
            continue
        references = {k: assign(model, anchor) for k, model in base.items()}
        aris = {k: [] for k in base}
        for repeat in range(config.screen_refits):
            changed = fit_candidates(v3.bootstrap_families(training, 101+repeat), fields, method, config, 201+repeat)
            for k in base:
                aris[k].append(v3.stable_ari(references[k], assign(changed[k], anchor)) if k in changed else -1.)
        rows = [{"method": method, "k": k, "screen_median_ari": float(np.median(aris[k])), **partition_score(model, population, anchor, k), "status": "screened_out"} for k, model in base.items()]
        chosen = v3.choose_finalist(rows)
        if chosen:
            k, model = int(chosen["k"]), base[int(chosen["k"])]
            repeat_aris = []
            for repeat in range(config.stability_refits):
                changed = fit_candidates(v3.bootstrap_families(training, 1001+repeat), fields, method, config, 2001+repeat, [k])
                repeat_aris.append(v3.stable_ari(references[k], assign(changed[k], anchor)) if k in changed else -1.)
            balanced = []
            variants = [("family_balanced", fields, True, "quantile", seed) for seed in (42,43,44)] + [("profile_weighted", fields, False, "quantile", 42), ("without_transmission", [f for f in fields if f != "transmission"], True, "quantile", 42), ("minmax", fields, True, "minmax", 42)]
            for name, fs, weights, scaling, seed in variants:
                alternative = fit_candidates(training, fs, method, config, seed, [k], weights, scaling)
                ari = v3.stable_ari(references[k], assign(alternative[k], anchor)) if k in alternative else -1.
                sensitivity.append({"method": method, "k": k, "sensitivity": name, "ari": ari})
                if name == "family_balanced":
                    balanced.append(ari)
            chosen.update(median_ari=float(np.median(repeat_aris)), ari_q10=float(np.quantile(repeat_aris, .1)), balanced_min_ari=float(min(balanced)))
            supported = chosen["median_ari"] >= .7 and chosen["ari_q10"] >= .6 and chosen["balanced_min_ari"] >= .7
            chosen["status"] = "accepted_partition" if supported else "unstable_subdivision"
            if supported:
                accepted[method] = model
        elif rows:
            rows[0]["status"] = "no_supported_subdivision"
        records.extend(rows)
    return accepted, records, sensitivity


def ranked_peers(data, fields, metric, memberships, labels, predictions):
    """Vectorized full-catalog ranking with bounded category and cluster adjustments."""
    frame = data.reset_index(drop=True)
    ids, families, bodies, powers = [frame[c].to_numpy() for c in ["profile_id", "family_block", "body_family", "powertrain"]]
    years = pd.to_numeric(frame.ModelYear).to_numpy(float, na_value=np.nan)
    positions = dict(zip(ids, range(len(frame))))
    observed = {f: frame[f].notna().to_numpy() for f in fields}
    values = {f: frame[f].fillna("__UNKNOWN__").astype(str).to_numpy() if f in v2.CATEGORICAL else np.clip(frame[f].to_numpy(float), *metric["bounds"][f]) for f in fields}
    groups = {(body, power): np.asarray(indices) for (body, power), indices in frame[frame.eligible].groupby(["body_family", "powertrain"]).groups.items()}
    vectors = {}
    for policy in ("conservative", "broader"):
        for dimension in v4.SEMANTIC:
            evidence = labels[labels.dimension.eq(dimension) & labels.profile_id.isin(positions)]
            inferred = predictions[predictions.policy.eq(policy) & predictions.dimension.eq(dimension) & predictions.profile_id.isin(positions)]
            vocabulary = sorted(set(evidence.label) | set(inferred.label))
            index = {label: i for i, label in enumerate(vocabulary)}
            matrix = np.zeros((len(frame), len(vocabulary)))
            for row in inferred.itertuples():
                matrix[positions[row.profile_id], index[row.label]] = row.probability
            for row in evidence.itertuples():
                matrix[positions[row.profile_id], index[row.label]] = 1.
            vectors[policy, dimension] = matrix
    cluster_arrays = []
    for view, rows in memberships.groupby("view"):
        if view == "global":
            continue
        values_by_id = rows.set_index("profile_id").cluster
        cluster_arrays.append(values_by_id.reindex(ids).fillna(-1).to_numpy(int))
    records, statuses = [], []
    for qi in np.flatnonzero(~frame.development.to_numpy()):
        candidate = groups.get((bodies[qi], powers[qi]), np.array([], dtype=int)) if frame.iloc[qi].eligible else np.array([], dtype=int)
        candidate = candidate[(families[candidate] != families[qi]) & (np.abs(years[candidate]-years[qi]) <= 10)]
        valid, components = np.ones(len(candidate), dtype=bool), {}
        for block, fs in metric["blocks"].items():
            total, count = np.zeros(len(candidate)), np.zeros(len(candidate), dtype=int)
            for field in fs:
                common = observed[field][qi] & observed[field][candidate]
                delta = values[field][candidate] != values[field][qi] if field in v2.CATEGORICAL else np.minimum(np.abs(values[field][candidate]-values[field][qi]) / max(metric["bounds"][field][1]-metric["bounds"][field][0], 1e-9), 1.)
                total += np.where(common, delta, 0.)
                count += common
            valid &= count >= max(2 if block == "size" else 1, int(np.ceil(len(fs)/2)))
            components[block] = np.divide(total, count, out=np.zeros(len(candidate)), where=count>0)
        window = next((w for w in (2,5,10) if len(set(families[candidate[valid & (np.abs(years[candidate]-years[qi])<=w)]])) >= 5), 10)
        valid &= np.abs(years[candidate]-years[qi]) <= window
        distance = (components["size"] + components["powertrain"])/2
        category_scores = {}
        for policy in ("conservative", "broader"):
            score, dimensions = np.zeros(len(candidate)), 0
            for dimension in v4.SEMANTIC:
                matrix = vectors[policy, dimension]
                if matrix.shape[1] and matrix[qi].any():
                    score += np.max(matrix[candidate]*matrix[qi], axis=1)
                    dimensions += 1
            category_scores[policy] = score/max(1, dimensions)
        penalty, shared = np.zeros(len(candidate)), np.zeros(len(candidate))
        for cluster in cluster_arrays:
            if cluster[qi] >= 0:
                known = cluster[candidate]>=0
                penalty += np.where(known, (cluster[candidate]!=cluster[qi])*.1, 0.)
                shared += known
        penalty = np.divide(penalty, shared, out=np.zeros(len(candidate)), where=shared>0)
        for approach in APPROACHES:
            bonus = .1*category_scores["broader" if approach.startswith("broader") else "conservative"] if approach != "direct_specs" else np.zeros(len(candidate))
            applied = penalty if "cluster" in approach else np.zeros(len(candidate))
            ranking = distance-bonus+applied
            order = np.flatnonzero(valid)
            order = order[np.lexsort((ids[candidate[order]], ranking[order]))]
            selected, seen = [], set()
            for index in order:
                family = families[candidate[index]]
                if family not in seen:
                    selected.append(index)
                    seen.add(family)
                if len(selected)==5:
                    break
            statuses.append({"query_profile_id": ids[qi], "family_block": families[qi], "fold": int(frame.iloc[qi].fold), "approach": approach, "eligible": bool(frame.iloc[qi].eligible), "slate_size": len(selected), "status": "supported" if selected else "insufficient_observations" if not frame.iloc[qi].eligible else "no_supported_peers", "year_window": window})
            for rank, index in enumerate(selected, 1):
                records.append({"query_profile_id": ids[qi], "peer_profile_id": ids[candidate[index]], "approach": approach, "rank": rank, "distance": distance[index], "ranking_distance": ranking[index], "category_bonus": bonus[index], "cluster_penalty": applied[index], "year_window": window})
    columns = ["query_profile_id", "peer_profile_id", "approach", "rank", "distance", "ranking_distance", "category_bonus", "cluster_penalty", "year_window"]
    return pd.DataFrame(records, columns=columns), pd.DataFrame(statuses)


def analyze(prepared, categories, output, classifier_run=None, smoke=False, resume=False):
    prep = verify(prepared, "completed")
    verify(categories, "completed")
    inputs = [prepared, categories] + ([classifier_run] if classifier_run else [])
    target = begin(output, "analyze", inputs, resume, smoke=smoke)
    config = v3.config_for(smoke)
    labels = pd.read_parquet(Path(categories) / "profile_labels.parquet")
    if classifier_run:
        classified=verify(classifier_run,"completed")
        if classified["dependencies"][0]!=dependency(categories):
            raise ValueError("Classifiers belong to a different category stage")
        if (Path(classifier_run)/"reviewed_profile_labels.parquet").exists():
            labels=pd.read_parquet(Path(classifier_run)/"reviewed_profile_labels.parquet")
    predictions = pd.read_parquet(Path(classifier_run) / "predictions.parquet") if classifier_run else pd.DataFrame(columns=["profile_id", "policy", "dimension", "label", "probability"])
    for grain in ("cohort", "model"):
        path = target / grain
        if (path / "manifest.json").exists():
            verify(path, "completed")
            continue
        if not path.exists():
            begin(path, "analyze-grain", inputs, grain=grain, smoke=smoke)
        p = pd.read_parquet(Path(categories) / "variant_profiles.parquet")
        p = p[p.grain.eq(grain)].copy().reset_index(drop=True)
        for dimension in v4.SEMANTIC:
            values=labels[labels.dimension.eq(dimension)].groupby("profile_id").label.agg(lambda x:";".join(sorted(set(x))))
            p[dimension+"_labels"]=p.profile_id.map(values).fillna("")
        p["category_summary"]=p[[d+"_labels" for d in v4.SEMANTIC]].apply(lambda x:" · ".join(v for v in x if v) or "Unresolved",axis=1)
        p["fold"] = p.family_block.map(v3.family_fold)
        scale_pairs = pd.read_parquet(Path(prepared)/grain/"horsepower_scale_pairs.parquet")
        candidates = p[v3.eligible(p, v2.NUMERIC+v2.CATEGORICAL)]
        count = min(config.queries, candidates.family_block.nunique())
        if count:
            from dataclasses import replace
            v3.freeze_queries(p, replace(config, queries=count)).to_csv(path / "evaluation_queries.csv", index=False)
        else:
            pd.DataFrame(columns=["card_id","profile_id","family_block","fold"]).to_csv(path / "evaluation_queries.csv", index=False)
        saved_members, compositions, experiments, support, sensitivities, peer_frames, status_frames = [], [], [], [], [], [], []
        p["eligible"] = False
        with threadpool_limits(limits=1):
            for fold in range(config.folds):
                data = p.copy()
                data["development"] = data.fold.ne(fold)
                data, fields, audit = select_fields(data, scale_pairs)
                audit["fold"] = fold
                support.append(audit)
                p.loc[p.fold.eq(fold), "eligible"] = data.loc[p.fold.eq(fold), "eligible"]
                metric = v3.fit_metric(data[data.development & data.eligible], fields)
                selected = []
                views = [("global", data.eligible)] + [(body, data.eligible & data.body_family.eq(body)) for body in sorted(data.body_family.unique())]
                supported_labels=labels[labels.dimension.isin(v4.SEMANTIC) & labels.status.eq("source_supported")]
                for (dimension,label), group in supported_labels.groupby(["dimension","label"]):
                    for body in sorted(data.body_family.unique()):
                        mask=data.eligible & data.body_family.eq(body) & data.profile_id.isin(group.profile_id)
                        if mask.any():
                            views.append((f"{dimension}__{label}__{body}",mask))
                for view, mask in views:
                    # Every view learns selection/coverage from its own development population.
                    local = data.copy()
                    local["development"] = local.development & mask
                    local, local_fields, local_audit = select_fields(local, scale_pairs)
                    support.append(local_audit.assign(fold=fold,view=view))
                    population = local[mask & local.eligible]
                    training = population[population.development]
                    checkpoint = path / "checkpoints" / (str(fold) + "-" + r.digest(view)[:16])
                    checksum = checkpoint.with_suffix(".sha256")
                    if checkpoint.exists() and checksum.exists():
                        if checksum.read_text() != r.file_hash(checkpoint):
                            raise ValueError("Changed cluster checkpoint")
                        models, records, sensitivity = joblib.load(checkpoint)
                    else:
                        models, records, sensitivity = discover(training, local_fields, config)
                        checkpoint.parent.mkdir(exist_ok=True)
                        joblib.dump((models, records, sensitivity), checkpoint)
                        checksum.write_text(r.file_hash(checkpoint))
                    experiments.extend({"fold": fold, "view": view, **row} for row in records)
                    sensitivities.extend({"fold": fold, "view": view, **row} for row in sensitivity)
                    chosen = next((m for m in v4.PRIORITY if m in models), None)
                    for method, model in models.items():
                        assigned = assign(model, population)
                        rows = population[["profile_id", "family_block"]].copy()
                        rows["cluster"], rows["training_fold"], rows["view"], rows["method"] = assigned, fold, view, method
                        rows["assignment_status"] = np.where(assigned >= 0, "assigned", "out_of_domain")
                        saved_members.append(rows)
                        if method == chosen:
                            selected.append(rows)
                        for cluster in sorted(set(assigned) - {-1}):
                            group = population.loc[(assigned == cluster) & population.development]
                            compositions.append({"fold": fold, "view": view, "method": method, "cluster": cluster, "description": v4.view_title(view) + " · " + v4.subgroup_description(group, training, local_fields), "profiles": int((assigned == cluster).sum()), "families": group.family_block.nunique(), "listings": int(group.vin_count.sum())})
                    print(grain, "fold", fold, view, "accepted", list(models), flush=True)
                selected = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=["view", "profile_id", "cluster"])
                fold_predictions = predictions[predictions.outer_fold.eq(fold)] if "outer_fold" in predictions else predictions
                peers, statuses = ranked_peers(data, fields, metric, selected, labels, fold_predictions)
                peer_frames.append(peers)
                status_frames.append(statuses)
                joblib.dump({"fields": fields, "metric": metric, "training_families": sorted(data.loc[data.development].family_block.unique())}, path / f"fold_{fold}_retrieval.joblib")
        p.to_parquet(path / "variant_profiles.parquet", index=False)
        pd.concat(peer_frames).to_parquet(path / "profile_peers.parquet", index=False)
        pd.concat(status_frames).to_parquet(path / "query_status.parquet", index=False)
        pd.concat(status_frames).groupby("approach").agg(queries=("query_profile_id","size"), eligible=("eligible","sum"), returned_peers=("slate_size","sum")).reset_index().to_csv(path / "coverage.csv", index=False)
        (pd.concat(saved_members) if saved_members else pd.DataFrame(columns=["profile_id", "cluster", "training_fold", "view", "method", "assignment_status"])).to_parquet(path / "cluster_memberships.parquet", index=False)
        pd.DataFrame(compositions, columns=["fold", "view", "method", "cluster", "description", "profiles", "families", "listings"]).to_csv(path / "cluster_profiles.csv", index=False)
        pd.DataFrame(experiments).to_csv(path / "cluster_experiments.csv", index=False)
        pd.DataFrame(sensitivities, columns=["fold", "view", "method", "k", "sensitivity", "ari"]).to_csv(path / "cluster_sensitivities.csv", index=False)
        pd.concat(support).to_csv(path / "feature_support.csv", index=False)
        for name in ("profile_labels.parquet", "category_coverage.parquet", "category_coverage.csv", "definitions.json", "exceptions.csv"):
            shutil.copyfile(Path(categories) / name, path / name)
        labels[labels.profile_id.isin(p.profile_id)].to_parquet(path/"profile_labels.parquet",index=False)
        effective_coverage=pd.read_parquet(path/"category_coverage.parquet")
        effective_coverage=effective_coverage[effective_coverage.grain.eq(grain)].copy()
        if classifier_run and (Path(classifier_run)/"reviewed_decisions.parquet").exists():
            decisions=pd.read_parquet(Path(classifier_run)/"reviewed_decisions.parquet")
            reviewed_keys=set(decisions[["profile_id","dimension"]].itertuples(index=False,name=None))
            assigned_keys=set(labels[["profile_id","dimension"]].itertuples(index=False,name=None))
            for index,row in effective_coverage.iterrows():
                key=row.profile_id,row.dimension
                if key in reviewed_keys:
                    effective_coverage.loc[index,"reason"]="independently_reviewed" if key in assigned_keys else "reviewed_unknown"
        effective_coverage.to_parquet(path/"category_coverage.parquet",index=False)
        effective_coverage.groupby(["grain","dimension","reason"]).agg(profiles=("profile_id","nunique"),families=("family_block","nunique"),listings=("vin_count","sum")).reset_index().to_csv(path/"category_coverage.csv",index=False)
        shutil.copyfile(Path(prepared) / grain / "vin_profile_membership.parquet", path / "vin_profile_membership.parquet")
        predictions[predictions.profile_id.isin(p.profile_id)].to_parquet(path / "predictions.parquet", index=False)
        v4.atomic_json(path / "evaluation.json", {"selected_approach": "direct_specs", "buyer_usefulness": "pending_review", "classifier_status": "see_classifier_receipt" if classifier_run else "pending_review"})
        finish(path, reduced_budget=smoke, grain=grain, configuration=asdict(config), prepared_directory=str(Path(prepared).resolve()), prepared_manifest_sha256=r.file_hash(Path(prepared) / "manifest.json"), source_directory=prep["source_directory"], source_hashes=prep["source_hashes"], excluded_vins=prep["excluded_vins"])
    return finish(target, reduced_budget=smoke, grains=["cohort", "model"])


def validate(run, output, resume=False):
    receipt = verify(run, "completed")
    verify_dependencies(receipt)
    target = begin(output, "validate", [run],resume)
    compatibility=validation_compatibility(run,receipt)
    counts = []
    for grain in ("cohort", "model"):
        path = Path(run) / grain
        bundle = verify(path, "completed")
        p = pd.read_parquet(path / "variant_profiles.parquet").set_index("profile_id")
        coordinates=p.copy()
        for field in [*v2.NUMERIC,*v2.CATEGORICAL,"width"]:
            coordinates[field]=coordinates[field].where(coordinates[field+"_n"].ge(5))
        m = pl.read_parquet(path / "vin_profile_membership.parquet")
        actual = m.group_by("profile_id").len().to_pandas().set_index("profile_id")["len"]
        if m.height != m["vin"].n_unique() or m["profile_id"].null_count() or not p.vin_count.sort_index().eq(actual.sort_index()).all():
            raise ValueError("VIN membership reconciliation failed")
        if m.filter(pl.col("vin").is_in(sorted(EXCLUDED_LISTING_VINS))).height:
            raise ValueError("Excluded listing present")
        peers = pd.read_parquet(path / "profile_peers.parquet")
        q, c = p.loc[peers.query_profile_id], p.loc[peers.peer_profile_id]
        if (q.family_block.to_numpy() == c.family_block.to_numpy()).any() or (q.body_family.to_numpy() != c.body_family.to_numpy()).any() or (q.powertrain.to_numpy() != c.powertrain.to_numpy()).any():
            raise ValueError("Peer compatibility violated")
        if not peers.category_bonus.between(0, .10000001).all() or not peers.cluster_penalty.between(0, .10000001).all() or not np.allclose(peers.ranking_distance, peers.distance-peers.category_bonus+peers.cluster_penalty):
            raise ValueError("Unbounded or invalid soft ranking")
        statuses = pd.read_parquet(path / "query_status.parquet")
        if len(statuses) != len(p)*len(APPROACHES) or statuses.duplicated(["query_profile_id", "approach"]).any():
            raise ValueError("Incomplete query denominator")
        accepted = pd.read_csv(path / "cluster_experiments.csv")
        accepted = accepted[accepted.status.eq("accepted_partition")]
        if not accepted.empty and not (accepted.median_ari.ge(.7) & accepted.ari_q10.ge(.6) & accepted.balanced_min_ari.ge(.7) & accepted.silhouette.gt(0) & accepted.minimum_cluster_families.ge(5) & accepted.minimum_count.ge(accepted.required_minimum_count)).all():
            raise ValueError("Cluster acceptance gate violated")
        for fold in p.fold.unique():
            fitted = joblib.load(path / f"fold_{fold}_retrieval.joblib")
            if set(fitted["training_families"]) & set(p.loc[p.fold.eq(fold)].family_block):
                raise ValueError("Family leakage")
            # Reproduce the saved peer ordering and distances from the frozen fit.
            replay=coordinates.reset_index().copy()
            replay["development"]=replay.fold.ne(fold)
            replay["eligible"]=v3.eligible(replay,fitted["fields"])
            memberships=pd.read_parquet(path/"cluster_memberships.parquet")
            selected=[]
            for view,group in memberships[memberships.training_fold.eq(fold)].groupby("view"):
                method=next((m for m in v4.PRIORITY if m in set(group.method)),None)
                if method:
                    selected.append(group[group.method.eq(method)])
            selected=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame(columns=["view","profile_id","cluster"])
            predictions=pd.read_parquet(path/"predictions.parquet")
            if "outer_fold" in predictions:
                predictions=predictions[predictions.outer_fold.eq(fold)]
            labels=pd.read_parquet(path/"profile_labels.parquet")
            expected,expected_status=ranked_peers(replay,fitted["fields"],fitted["metric"],selected,labels,predictions)
            actual=peers[peers.query_profile_id.isin(set(replay.loc[~replay.development,"profile_id"]))]
            order=["query_profile_id","approach","rank"]
            pd.testing.assert_frame_equal(expected.sort_values(order).reset_index(drop=True),actual.sort_values(order).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-10)
            actual_status=statuses[statuses.fold.eq(fold)]
            pd.testing.assert_frame_equal(expected_status.sort_values(order[:2]).reset_index(drop=True),actual_status.sort_values(order[:2]).reset_index(drop=True),check_dtype=False)
        saved_members=pd.read_parquet(path/"cluster_memberships.parquet")
        for checkpoint in (path/"checkpoints").iterdir():
            if checkpoint.suffix==".sha256":
                continue
            if checkpoint.with_suffix(".sha256").read_text()!=r.file_hash(checkpoint):
                raise ValueError("Cluster checkpoint changed")
            fold=int(checkpoint.name.split("-")[0])
            models,_,_=joblib.load(checkpoint)
            for method,model in models.items():
                if set(model["fit_families"]) & set(p.loc[p.fold.eq(fold),"family_block"]):
                    raise ValueError("Cluster fit contains a held-out query family")
                for view,group in saved_members[saved_members.training_fold.eq(fold) & saved_members.method.eq(method)].groupby("view"):
                    if checkpoint.name.split("-",1)[1]!=r.digest(view)[:16]:
                        continue
                    expected=pd.Series(assign(model,coordinates.loc[group.profile_id].reset_index()),index=group.profile_id)
                    actual=group.set_index("profile_id").cluster
                    v4.validate_cluster_memberships(expected,actual)
        for name, expected in bundle["source_hashes"].items():
            if r.file_hash(Path(bundle["source_directory"]) / name) != expected:
                raise ValueError("Source snapshot changed")
        counts.append({"grain": grain, "profiles": len(p), "listings": m.height, "eligible_profiles": int(p.eligible.sum()), "assigned_clusters": len(accepted)})
    pd.DataFrame(counts).to_csv(target / "validation_counts.csv", index=False)
    v4.atomic_json(target / "validation.json", {"passed": True, "run": dependency(run), "human_gates": "pending unless separately reviewed","execution_compatibility":compatibility})
    return finish(target, passed=True, run=dependency(run))


def activate(run, validation, registry_path=None, evaluation=None):
    receipt, checked = verify(run, "completed"), verify(validation, "completed")
    if receipt.get("reduced_budget", True) or not checked.get("passed") or checked.get("run") != dependency(run) or checked["execution"] != execution():
        raise ValueError("Activation requires current matching full-budget validation")
    verify_dependencies(receipt)
    approaches = {"cohort":"direct_specs", "model":"direct_specs"}
    if evaluation:
        result = verify(evaluation, "completed")
        if result.get("run") != dependency(run):
            raise ValueError("Peer evaluation belongs to a different run")
        approaches = {grain:value["selected_approach"] for grain,value in r.read_json(Path(evaluation)/"evaluation.json").items()}
    for grain in ("cohort", "model"):
        bundle = verify(Path(run)/grain,"completed")
        for name, expected in bundle["source_hashes"].items():
            if r.file_hash(Path(bundle["source_directory"])/name) != expected:
                raise ValueError("Source changed since validation")
    registry_path = Path(registry_path or r.ROOT / "car_app/artifact_registry.json")
    lock = registry_path.with_suffix(".lock")
    lock.open("x").close()
    try:
        original = registry_path.read_bytes()
        config = json.loads(original.decode("utf-8-sig"))
        config.update(similarity_run=str(Path(run).resolve().relative_to(r.ROOT / "MODEL_OUTPUT")).replace("\\", "/"), similarity_manifest_sha256=r.file_hash(Path(run) / "manifest.json"))
        config["preferred_similarity_approaches"] = approaches
        backup = Path(run).parent / ("activation-" + uuid.uuid4().hex)
        backup.mkdir()
        (backup / "previous_registry.json").write_bytes(original)
        if registry_path.read_bytes() != original:
            raise ValueError("Registry changed concurrently")
        v4.atomic_json(registry_path, config)
    finally:
        lock.unlink(missing_ok=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "collect-sources", "build-categories", "export-review", "import-review", "train-classifiers", "analyze", "validate", "export-peer-review", "evaluate-peer-review", "activate"])
    for name in ("data-dir", "prepared", "sources", "categories", "output", "reuse-sources", "review", "ratings", "classifiers", "run", "validation", "evaluation"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    required = {"prepare": ["data_dir", "output"], "collect-sources": ["prepared", "output"], "build-categories": ["prepared", "sources", "output"], "export-review": ["categories", "output"], "import-review": ["categories", "review", "ratings", "output"], "train-classifiers": ["categories", "output"], "analyze": ["prepared", "categories", "output"], "validate": ["run", "output"], "export-peer-review":["run","output"], "evaluate-peer-review":["run","review","ratings","validation","output"], "activate": ["run", "validation"]}
    for name in required[args.stage]:
        if getattr(args, name) is None:
            parser.error("Missing --" + name.replace("_", "-"))
    if args.stage == "prepare":
        result = prepare(args.data_dir, args.output,args.resume)
    elif args.stage == "collect-sources":
        result = collect_sources(args.prepared, args.output, args.reuse_sources, args.limit, args.resume, args.offline)
    elif args.stage == "build-categories":
        result = build_categories(args.prepared, args.sources, args.output,args.resume)
    elif args.stage in ("export-review", "import-review", "train-classifiers"):
        verify(args.categories, "completed")
        inputs = [args.categories] + ([args.review] if args.review else [])
        if args.review:
            verify(args.review, "completed")
        config={"ratings_sha256":r.file_hash(args.ratings)} if args.stage=="import-review" else {}
        target = begin(args.output, args.stage, inputs,args.resume,**config)
        data = pd.read_parquet(args.categories / "variant_profiles.parquet")
        labels = pd.read_parquet(args.categories / "profile_labels.parquet")
        definitions = r.read_json(args.categories / "definitions.json")
        if args.stage == "export-review":
            classifiers.export_review(data, labels, target)
        elif args.stage == "import-review":
            inventory = pd.read_csv(args.review / "review_inventory.csv", keep_default_na=False)
            classifiers.import_review(inventory, args.ratings, definitions).to_parquet(target / "reviewed_labels.parquet", index=False)
            shutil.copyfile(args.ratings,target/"ratings_input.csv")
            if r.file_hash(target/"ratings_input.csv")!=config["ratings_sha256"]:
                raise ValueError("Review ratings changed during import")
        else:
            reviewed = pd.read_parquet(args.review / "reviewed_labels.parquet") if args.review else pd.DataFrame()
            predicted, _ = classifiers.train(data, labels, reviewed, definitions, target)
            predicted.to_parquet(target / "predictions.parquet", index=False)
            if not reviewed.empty:
                reviewed.to_parquet(target/"reviewed_decisions.parquet",index=False)
                keys=set(reviewed[["profile_id","dimension"]].itertuples(index=False,name=None))
                reconciled=labels[[pair not in keys for pair in labels[["profile_id","dimension"]].itertuples(index=False,name=None)]].copy()
                additions=[]
                for row in reviewed.itertuples():
                    for label in set(str(row.labels).split(";"))-{ "" }:
                        additions.append(label_row(row.profile_id,row.dimension,label,"source_supported","review://"+row.review_id,"Independent category judgment by "+row.reviewer,"Reviewed "+row.grain+" profile; original evidence: "+str(row.source_urls),"reviewed-"+row.review_id))
                reconciled=pd.concat([reconciled,pd.DataFrame(additions,columns=LABEL_COLUMNS)],ignore_index=True)
                reconciled.to_parquet(target/"reviewed_profile_labels.parquet",index=False)
        result = finish(target)
    elif args.stage == "analyze":
        result = analyze(args.prepared, args.categories, args.output, args.classifiers, args.smoke, args.resume)
    elif args.stage == "validate":
        result = validate(args.run, args.output,args.resume)
    elif args.stage == "export-peer-review":
        verify(args.run,"completed")
        target=begin(args.output,args.stage,[args.run],args.resume)
        details=peer_review.export(args.run,target)
        result=finish(target,run=dependency(args.run),**details)
    elif args.stage == "evaluate-peer-review":
        receipt=verify(args.run,"completed")
        reviewed=verify(args.review,"completed")
        checked=verify(args.validation,"completed")
        if reviewed.get("run") != dependency(args.run) or checked.get("run") != dependency(args.run) or not checked.get("passed"):
            raise ValueError("Review and validation must match the run")
        target=begin(args.output,args.stage,[args.run,args.review,args.validation],args.resume, ratings_sha256={grain:r.file_hash(args.ratings/grain/"ratings.csv") for grain in ("cohort","model")})
        result=peer_review.evaluate(args.review,args.ratings,args.run,receipt.get("reduced_budget",True))
        v4.atomic_json(target/"evaluation.json",result)
        result=finish(target,run=dependency(args.run))
    else:
        result = activate(args.run, args.validation, evaluation=args.evaluation)
    print(result, flush=True)


if __name__ == "__main__":
    main()
