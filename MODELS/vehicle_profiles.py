"""Retained-listing profiles and conditional, ecological complaint comparisons.

Run stages with ``python -m MODELS.vehicle_profiles --help``. Sources are read-only;
each stage owns a new directory and records an explicit completion or blocker state.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, silhouette_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ["ModelYear", "Make", "Model"]
COHORT = FAMILY + ["Trim"]
NUMERIC = ["seating", "wheelbase", "length", "height", "horsepower", "engine_displacement"]
EXTRA_NUMERIC = ["width", "front_legroom", "back_legroom"]
RESEARCH_NUMERIC = NUMERIC + EXTRA_NUMERIC
CATEGORICAL = ["fuel_type", "wheel_system", "transmission"]
BLOCKS = {"size": NUMERIC[:4], "powertrain": NUMERIC[4:] + CATEGORICAL}
UNITS = {"seating": "seats", "wheelbase": "in", "length": "in", "height": "in",
         "horsepower": "hp", "engine_displacement": "cc", "width": "in", "front_legroom": "in", "back_legroom": "in"}
OPTIONS = ["Navigation System", "Leather Seats", "Heated Seats", "Backup Camera",
           "Bluetooth", "Sunroof/Moonroof", "Remote Start", "Alloy Wheels",
           "Adaptive Cruise Control", "Blind Spot Monitoring"]
COMPONENTS = {"powertrain": "ENGINE|POWER TRAIN", "brakes": "BRAKE", "steering": "STEERING",
              "electrical": "ELECTRICAL", "airbags": "AIR BAG", "fuel": "FUEL SYSTEM",
              "structure": "STRUCTURE", "visibility": "VISIBILITY|EXTERIOR LIGHTING"}
FILES = {"used_cars": "used_cars.parquet", "vpic": "NHTSA_vPIC.parquet",
         "complaints": "NHTSA_Complaints.parquet"}
CUTOFF = pd.Timestamp("2026-09-30")
STAGE_STARTED = {}
REPORTING_WORDS = set("""contact consumer consumers stated states statement owns owned vehicle vehicles car cars
failure failed failing problem problems issue issues dealer dealership manufacturer contacted notified
information provide provided details told taken repaired repair repairs replaced replacement updated
update writes regards regarding mileage miles mph approximate approximately campaign nhtsa number
notification received available unavailable reasonable time times date vin tl ak jb ts smd mr nlm yh
skd tt tr dt cw js aw bw cc recall recalls safety complaint complaints""".split())
MECHANICAL_WORDS = set("""engine engines motor stalling stalled stall oil coolant overheating leak leaking
transmission gear gears shift shifting shifter clutch fluid steering wheel rack column turn turning
brake brakes braking pedal rotors pads abs accelerator acceleration accelerate cruise control
airbag airbags air bags bag deploy deployed deployment collision impact injuries passenger driver
seat seats belt belts buckle seatbelt child retract electrical battery ignition key switch lights
light headlight headlights beam beams signal sensor sensors warning illuminated start starting
fuel tank pump gas gasoline gauge pressure fire burning smoke tire tires tread sidewall rim spare
wear worn blowout flat alignment vibration suspension axle wheel wheels rear front noise grinding
door doors lock latch sliding open opened window windows windshield wiper wipers glass sunroof
crack cracked shattered visibility roof structure frame rust corrosion park parking reverse neutral
power loss shut restart exhaust emissions catalyst catalytic throttle speed speedometer acceleration
conditioning conditioner compressor air hvac heat heater defrost""".split())


@dataclass(frozen=True)
class Config:
    seed: int = 42
    vehicle_cap: int = 10000
    medoid_cap: int = 2000
    text_cap: int = 50000
    starts: int = 5
    stability_refits: int = 10
    topic_refits: int = 5
    bootstrap: int = 1000
    clusters: tuple = (2, 3, 4, 5, 6, 7, 8)
    topics: tuple = (10, 20, 30)
    smoke: bool = False


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_table(path, frame):
    frame.to_parquet(path, index=False)


def canonical(value):
    if pd.isna(value):
        return None
    text = " ".join(str(value).strip().upper().split())
    return text if text and text not in {"UNKNOWN", "NOT APPLICABLE", "N/A", "--"} else None


def normalize_keys(frame):
    result = frame.copy()
    result["ModelYear"] = pd.to_numeric(result.ModelYear, errors="coerce").astype("Int64")
    for column in [c for c in COHORT[1:] if c in result]:
        result[column] = result[column].map(canonical)
    return result


def is_development(make, model, seed=42):
    return int(digest(f"{seed}|{make}|{model}")[:16], 16) % 5 != 0


def option_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or not str(value).strip():
        return "missing", None
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError, TypeError):
        return "malformed", None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return "malformed", None
    return ("populated" if parsed else "valid_empty"), set(parsed)


def mode_agreement(series):
    counts = series.dropna().value_counts()
    if counts.empty:
        return None, 0.0
    # A tied mode never passes the 90% agreement rule.
    return sorted(counts[counts == counts.max()].index.astype(str))[0], float(counts.max() / counts.sum())


def prepare_vehicles(listings, decodes, config):
    listings = listings.copy()
    if listings.vin.isna().any() or listings.vin.duplicated().any():
        raise ValueError("Listing vin must be non-null and unique")
    decodes = normalize_keys(decodes).drop_duplicates()
    conflicting = decodes.loc[decodes.vin.duplicated(False), "vin"].dropna().unique()
    decodes = decodes[~decodes.vin.isin(conflicting) & decodes.vin.notna()]
    joined = listings.merge(decodes, on="vin", how="left", validate="one_to_one", indicator=True)
    audit = {"retained_listings": len(listings), "conflicting_decode_vins": len(conflicting),
             "unmatched_listing_vins": int((joined._merge == "left_only").sum()),
             "incomplete_cohort_vins": int(joined[COHORT].isna().any(axis=1).sum())}
    joined = joined.dropna(subset=COHORT).drop(columns="_merge")
    joined["seating"] = pd.to_numeric(joined.maximum_seating.astype("string").str.extract(
        r"^(\d+)\s+seats?$", expand=False), errors="coerce")
    for field in EXTRA_NUMERIC + ["bed_length", "price", "mileage", "is_new"]:
        if field not in joined:
            joined[field] = np.nan
    for field in RESEARCH_NUMERIC + ["bed_length"]:
        joined[field] = pd.to_numeric(joined[field], errors="coerce")
        joined.loc[~np.isfinite(joined[field]) | (joined[field] <= 0), field] = np.nan
    from EDA.DATA_CLEANING import USED_CARS_UNITS
    for field in ["wheelbase", "length", "height", *EXTRA_NUMERIC]:
        if USED_CARS_UNITS.get(field) != UNITS[field]:
            raise ValueError(f"Unverified export unit for {field}")
    unit_checks = {}
    unit_development = np.array([is_development(a, b, config.seed) for a, b in zip(joined.Make, joined.Model)])
    for field, reference in [("horsepower", "EngineHP"), ("engine_displacement", "DisplacementCC")]:
        reference_values = pd.to_numeric(joined.get(reference, pd.Series(np.nan, index=joined.index)), errors="coerce")
        ratio = (joined[field] / reference_values.where(reference_values > 0)).where(unit_development).replace([np.inf, -np.inf], np.nan).dropna()
        median_ratio = float(ratio.median()) if len(ratio) else None
        verified = len(ratio) >= 30 and .9 <= median_ratio <= 1.1
        unit_checks[field] = {"paired_vins": len(ratio), "median_source_reference_ratio": median_ratio,
                              "paired_ratio_outside_20_percent_share": float(((ratio < .8) | (ratio > 1.2)).mean()) if len(ratio) else None,
                              "reference": reference, "verified": verified, "population": "development VINs"}
        if not verified:
            joined[field] = np.nan
    audit["unit_checks"] = unit_checks
    for field in CATEGORICAL + ["body_type"]:
        joined[field] = joined[field].map(canonical)
    parsed = joined.major_options.map(option_list)
    joined["option_state"] = parsed.map(lambda item: item[0])
    joined["option_count"] = parsed.map(lambda item: len(item[1]) if item[1] is not None else np.nan)
    for i, label in enumerate(OPTIONS):
        joined[f"advertised_{i}"] = parsed.map(lambda item: float(label in item[1]) if item[1] is not None else np.nan)
    records, representatives = [], []
    for key, group in joined.groupby(COHORT, sort=True, observed=True):
        row = dict(zip(COHORT, key))
        row.update(vin_count=int(group.vin.nunique()), development=is_development(key[1], key[2], config.seed))
        electric = group.fuel_type.fillna("").str.upper().eq("ELECTRIC")
        for field in RESEARCH_NUMERIC + ["bed_length"]:
            values = group[field].dropna()
            median = float(values.median()) if len(values) else np.nan
            iqr = float(values.quantile(.75) - values.quantile(.25)) if len(values) else np.nan
            row.update({field: median if len(values) and iqr <= .2 * median else np.nan,
                        field + "_median": median, field + "_iqr": iqr, field + "_n": len(values)})
            row[field + "_state"] = ("not_applicable" if field == "engine_displacement" and electric.all()
                                      else "unknown" if not len(values) else "observed" if iqr <= .2 * median else "heterogeneous")
        for field in ["price", "mileage"]:
            values = pd.to_numeric(group[field], errors="coerce")
            values = values[np.isfinite(values) & ((values > 0) if field == "price" else (values >= 0))]
            row.update({field + "_median": values.median(), field + "_q25": values.quantile(.25),
                        field + "_q75": values.quantile(.75), field + "_n": len(values)})
        row["new_listing_share"] = group.is_new.astype("string").str.lower().map({"true": 1., "false": 0.}).mean()
        for field in CATEGORICAL + ["body_type"]:
            mode, agreement = mode_agreement(group[field])
            row.update({field: mode if agreement >= .9 else None,
                        field + "_mode": mode, field + "_agreement": agreement,
                        field + "_n": int(group[field].notna().sum())})
        for field in ["city_fuel_economy", "highway_fuel_economy"]:
            values = pd.to_numeric(group.get(field, pd.Series(dtype=float)), errors="coerce")
            row[field + "_median_unverified"] = values.where(np.isfinite(values) & (values > 0)).median()
        for state in ["missing", "malformed", "valid_empty", "populated"]:
            row["option_" + state + "_count"] = int((group.option_state == state).sum())
        row["option_valid_count"] = int(group.option_count.notna().sum())
        row["option_valid_share"] = row["option_valid_count"] / len(group)
        row["option_mean_length"] = group.option_count.mean()
        row["dealer_distinct"] = int(group.dealer_zip.nunique())
        for field in [f"advertised_{i}" for i in range(len(OPTIONS))]:
            row[field] = group[field].mean()
        row["heterogeneous"] = any(pd.isna(row[f]) and group[f].notna().any() for f in NUMERIC + CATEGORICAL)
        # Select an actual VIN closest to the numeric medians; do not fabricate a configuration.
        values = group[NUMERIC].to_numpy(float)
        medians = np.array([row[f + "_median"] for f in NUMERIC])
        deviations = np.abs(values - medians) / np.where(medians > 0, medians, 1)
        observed = np.isfinite(deviations).sum(axis=1)
        distance = np.divide(np.nansum(deviations, axis=1), observed,
                             out=np.full(len(group), np.inf), where=observed > 0)
        if np.isfinite(distance).any():
            index = int(np.argmin(distance))
            representative = group.iloc[index][COHORT + ["vin"] + NUMERIC + CATEGORICAL].to_dict()
            representative.update(representative_status="numeric_match", numeric_fields_observed=int(observed[index]))
        else:
            representative = {**dict(zip(COHORT, key)), **{f: None for f in ["vin", *NUMERIC, *CATEGORICAL]},
                              "representative_status": "unavailable_no_numeric_specs", "numeric_fields_observed": 0}
        representatives.append(representative)
        records.append(row)
    cohorts = pd.DataFrame(records, columns=list(records[0]) if records else COHORT + ["vin_count", "development"])
    audit.update(complete_cohorts=len(cohorts), primary_size_cohorts=int((cohorts.vin_count >= 5).sum()),
                 heterogeneous_cohorts=int(cohorts.get("heterogeneous", pd.Series(dtype=bool)).sum()))
    audit["option_state_counts"] = joined.option_state.value_counts().to_dict()
    audit["representative_unavailable_cohorts"] = sum(r["vin"] is None for r in representatives)
    return cohorts, pd.DataFrame(representatives), audit


def product_status(row):
    try:
        products = json.loads(row.products) if isinstance(row.products, str) else row.products
    except (ValueError, TypeError):
        return "unsupported"
    if not isinstance(products, list) or not products:
        return "unsupported"
    vehicles = [p for p in products if isinstance(p, dict) and str(p.get("type", "")).upper() == "VEHICLE"]
    for product in vehicles:
        if (str(product.get("productYear", "")) == str(row.ModelYear)
                and canonical(product.get("productMake")) == row.Make
                and canonical(product.get("productModel")) == row.Model):
            return "exact_supported"
    return "conflict" if vehicles else "unsupported"


def primary_window(year, filed):
    if pd.isna(year) or pd.isna(filed):
        return False
    year = int(year)
    return year + 4 <= 2025 and year <= filed.year <= year + 4 and filed <= CUTOFF


def prepare_complaints(raw, cohorts, config):
    frame = normalize_keys(raw.rename(columns={"model_year": "ModelYear", "make": "Make", "model": "Model"}))
    for field in ["dateComplaintFiled", "dateOfIncident"]:
        frame[field] = pd.to_datetime(frame[field], errors="coerce")
    missing_identity = frame.odiNumber.isna() | frame[FAMILY].isna().any(axis=1)
    frame = frame[~missing_identity].copy()
    # Repeated query rows must not choose a filing date by incidental row order.
    frame["dateComplaintFiled"] = frame.groupby("odiNumber").dateComplaintFiled.transform("min")
    frame["normalized_text"] = frame.summary.fillna("").map(lambda x: " ".join(str(x).lower().split()))
    frame["text_hash"] = frame.normalized_text.map(digest)
    conflicting = frame.groupby("odiNumber").text_hash.nunique()
    conflicts = set(conflicting[conflicting > 1].index)
    frame["mapping_status"] = frame.apply(product_status, axis=1)
    frame["narrative_conflict"] = frame.odiNumber.isin(conflicts)
    frame["development"] = [is_development(a, b, config.seed) for a, b in zip(frame.Make, frame.Model)]
    frame["filing_eligible"] = frame.dateComplaintFiled.notna() & (frame.dateComplaintFiled <= CUTOFF)
    frame["primary_window"] = [primary_window(y, d) for y, d in zip(frame.ModelYear, frame.dateComplaintFiled)]
    frame["approximate_age"] = frame.dateComplaintFiled.dt.year - frame.ModelYear
    # Preserve differences in applicability while collapsing duplicate collection rows.
    associations = frame.drop_duplicates(FAMILY + ["odiNumber", "mapping_status", "text_hash"])
    supported = associations[associations.mapping_status == "exact_supported"]
    multi = supported.drop_duplicates(["odiNumber", *FAMILY]).groupby("odiNumber").size()
    associations["multi_family"] = associations.odiNumber.isin(multi[multi > 1].index)
    memberships = associations.groupby("text_hash").development.nunique()
    cross = set(memberships[memberships > 1].index)
    associations["cross_split_text"] = associations.text_hash.isin(cross)
    source_families = cohorts[FAMILY].drop_duplicates()
    source_families["vehicle_family_present"] = True
    associations = associations.merge(source_families, on=FAMILY, how="left", validate="many_to_one")
    associations["vehicle_family_present"] = associations.vehicle_family_present.fillna(False).astype(bool)
    associations["primary_eligible"] = (associations.filing_eligible & associations.primary_window
        & (associations.mapping_status == "exact_supported") & ~associations.narrative_conflict
        & ~associations.multi_family & associations.vehicle_family_present)
    associations["text_eligible"] = (associations.primary_eligible & ~associations.cross_split_text
                                    & (associations.normalized_text.str.len() >= 20))
    audit_samples = []
    for accepted in [True, False]:
        subset = associations[(associations.mapping_status == "exact_supported") == accepted]
        subset = subset.sort_values(FAMILY + ["odiNumber"]).drop_duplicates(FAMILY + ["odiNumber"])
        sample = subset.sample(min(50, len(subset)), random_state=config.seed).copy()
        sample["review_id"] = [digest("|".join(map(str, values))) for values in sample[FAMILY + ["odiNumber", "mapping_status"]].to_numpy()]
        sample["judgment"] = ""
        sample["notes"] = ""
        audit_samples.append(sample)
    audit = {"source_rows": len(raw), "distinct_complaints": int(frame.odiNumber.nunique()),
             "missing_identity_rows": int(missing_identity.sum()),
             "conflicting_narratives": len(conflicts), "multi_family_ids": int((multi > 1).sum()),
             "cross_split_text_hashes": len(cross), "primary_family_associations": int(associations.primary_eligible.sum()),
             "incident_date_missing": int(frame.drop_duplicates("odiNumber").dateOfIncident.isna().sum()),
             "maximum_filing_date": str(frame.dateComplaintFiled.max()),
             "mapping_status_counts": associations.mapping_status.value_counts().to_dict()}
    audit["overlapping_vehicle_families"] = len(associations.loc[associations.vehicle_family_present, FAMILY].drop_duplicates())
    audit["supported_primary_families"] = len(associations.loc[associations.primary_eligible, FAMILY].drop_duplicates())
    return associations, pd.concat(audit_samples, ignore_index=True), audit


def load_sources(data_dir, dictionary=ROOT / "DATA_DICTIONARY.md"):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    documentation = Path(dictionary).read_text(encoding="utf-8")
    sources, receipt = {}, {"manifest_sha256": file_hash(data_dir / "manifest.json")}
    required = {"used_cars": ["vin", "maximum_seating", *NUMERIC[1:], *CATEGORICAL, "major_options", "body_type", "dealer_zip", "city_fuel_economy", "highway_fuel_economy"],
                "vpic": ["vin", *COHORT, "EngineHP", "DisplacementCC"],
                "complaints": ["odiNumber", "model_year", "make", "model", "products", "summary", "components", "dateComplaintFiled", "dateOfIncident"]}
    for name, filename in FILES.items():
        path = data_dir / filename
        schema = pl.scan_parquet(path).collect_schema()
        metadata = next((v for v in manifest["tables"].values() if v["file"] == filename), None)
        if metadata is None or metadata["schema"] != {k: str(v) for k, v in schema.items()}:
            raise ValueError(f"Manifest schema mismatch: {filename}")
        count = pl.scan_parquet(path).select(pl.len()).collect().item()
        section = re.search(rf"## `{re.escape(path.stem)}`\s+.*?\*\*([\d,]+) rows;", documentation, re.S)
        if section is None or count != int(section[1].replace(",", "")) or count != metadata["rows"]:
            raise ValueError(f"Dictionary/manifest row count mismatch: {filename}")
        missing = set(required[name]) - set(schema)
        if missing:
            raise ValueError(f"Missing columns in {filename}: {sorted(missing)}")
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        receipt[filename] = {"sha256": file_hash(path), "rows": count, "schema": metadata["schema"]}
        if "sha256" in metadata and metadata["sha256"] != receipt[filename]["sha256"]:
            raise ValueError(f"Manifest hash mismatch: {filename}")
        optional = {"used_cars": EXTRA_NUMERIC + ["bed_length", "price", "mileage", "is_new"], "vpic": [], "complaints": []}[name]
        sources[name] = pl.scan_parquet(path).select(required[name] + [f for f in optional if f in schema]).collect(engine="streaming").to_pandas()
        if before != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f"Source changed during read: {filename}")
    return sources, receipt


def begin_stage(run_dir, stage):
    target = Path(run_dir) / stage
    target.mkdir(parents=True, exist_ok=False)
    STAGE_STARTED[str(target)] = time.perf_counter()
    return target


def finish_stage(target, status, **details):
    artifacts = {str(p.relative_to(target)).replace("\\", "/"): file_hash(p) for p in target.rglob("*")
                 if p.is_file() and p != target / "manifest.json"}
    write_json(target / "manifest.json", {"stage": target.name, "status": status, "artifacts": artifacts,
        "completed_at": datetime.now(timezone.utc).isoformat(), "code_sha256": file_hash(__file__),
        "elapsed_seconds": time.perf_counter() - STAGE_STARTED.get(str(target), time.perf_counter()),
        "segmentation_code_sha256": file_hash(ROOT / "MODELS/vehicle_segmentation.py") if (ROOT / "MODELS/vehicle_segmentation.py").is_file() else None,
        **details})
    return target


def verified_stage(run_dir, stage):
    target = Path(run_dir) / stage
    manifest = read_json(target / "manifest.json")
    if manifest.get("code_sha256") != file_hash(__file__):
        raise ValueError("Research code changed; prepare a fresh run")
    if manifest.get("segmentation_code_sha256") and manifest["segmentation_code_sha256"] != file_hash(ROOT / "MODELS/vehicle_segmentation.py"):
        raise ValueError("Segmentation code changed; prepare a fresh run")
    for name, expected in manifest["artifacts"].items():
        if file_hash(target / name) != expected:
            raise ValueError(f"Changed frozen artifact: {stage}/{name}")
    return target, manifest


def prepare(data_dir, run_dir, config=Config()):
    target = begin_stage(run_dir, "prepare")
    base = {"config": asdict(config), "source_dir": str(Path(data_dir).resolve()),
            "code_sha256": file_hash(__file__), "cutoff": str(CUTOFF.date()), "units": UNITS,
            "population": "retained listings after existing price/mileage export exclusions",
            "versions": {p: importlib.metadata.version(p) for p in ["pandas", "polars", "scikit-learn", "numpy", "scipy"]}}
    try:
        sources, receipt = load_sources(data_dir)
        cohorts, representatives, vehicle_audit = prepare_vehicles(sources["used_cars"], sources["vpic"], config)
        if cohorts.empty:
            return finish_stage(target, "insufficient_cohorts", sources=receipt, **base)
        complaints, mapping_review, complaint_audit = prepare_complaints(sources["complaints"], cohorts, config)
        for name, frame in [("cohorts", cohorts), ("representatives", representatives), ("complaints", complaints)]:
            save_table(target / f"{name}.parquet", frame)
        mapping_review.to_csv(target / "mapping_review_template.csv", index=False)
        _, _, specification_support = select_fields(cohorts)
        specification_support.to_csv(target / "specification_support.csv", index=False)
        complaints.groupby(["ModelYear", "mapping_status", "primary_window", "filing_eligible"], dropna=False).odiNumber.nunique().rename("distinct_complaints").to_csv(target / "complaint_coverage.csv")
        complaints.assign(filing_year=complaints.dateComplaintFiled.dt.year).groupby(["ModelYear", "filing_year"], dropna=False).odiNumber.nunique().rename("distinct_complaints").to_csv(target / "filing_periods.csv")
        complaints.loc[complaints.mapping_status != "exact_supported", FAMILY + ["odiNumber", "products", "mapping_status"]].drop_duplicates().to_csv(target / "unmatched_product_mappings.csv", index=False)
        cohorts[COHORT + ["development"]].to_csv(target / "split_receipt.csv", index=False)
        pd.DataFrame([{"measure": k, "value": str(v)} for k, v in {**vehicle_audit, **complaint_audit}.items()]).to_csv(target / "feasibility.csv", index=False)
        return finish_stage(target, "prepared", sources=receipt, vehicles=vehicle_audit, complaints=complaint_audit, **base)
    except (PermissionError, FileNotFoundError, ValueError, KeyError) as error:
        status = "access_denied" if isinstance(error, PermissionError) else "missing_source" if isinstance(error, FileNotFoundError) else "invalid_source"
        return finish_stage(target, status, error=str(error), **base)


def run_config(manifest):
    values = manifest["config"].copy()
    values["clusters"], values["topics"] = tuple(values["clusters"]), tuple(values["topics"])
    return Config(**values)


def relaxed_cohorts(cohorts):
    result = cohorts.copy()
    for field in NUMERIC:
        result[field] = result[field + "_median"].where(result[field + "_iqr"] <= .4 * result[field + "_median"])
    for field in CATEGORICAL:
        result[field] = result[field + "_mode"].where(result[field + "_agreement"] >= .8)
    return result


def select_fields(cohorts, minimum_vins=5, options=False):
    development = cohorts[cohorts.development & (cohorts.vin_count >= minimum_vins)]
    fields, profile = [], []
    candidates = NUMERIC + CATEGORICAL + ([f"advertised_{i}" for i in range(len(OPTIONS))] if options else [])
    for field in candidates:
        coverage = float(development[field].notna().mean()) if len(development) else 0.0
        reason = "selected" if coverage >= .7 and development[field].nunique() > 1 else "coverage_or_variation"
        block = next((b for b, fs in BLOCKS.items() if field in fs), "advertised")
        if reason == "selected" and field in NUMERIC:
            for prior in fields:
                if prior in BLOCKS[block] and prior in NUMERIC:
                    correlation = development[[prior, field]].corr(method="spearman").iloc[0, 1]
                    if abs(correlation) > .9:
                        reason = "redundant_with_" + prior
                        break
        if reason == "selected":
            fields.append(field)
        profile.append({"field": field, "coverage": coverage, "reason": reason, "block": block})
    eligible = (cohorts.vin_count >= minimum_vins) & (cohorts[fields].notna().mean(axis=1) >= .7) if fields else pd.Series(False, index=cohorts.index)
    return fields, eligible, pd.DataFrame(profile)


def feature_block(field):
    if field in EXTRA_NUMERIC:
        return "size"
    if field in ["log_price", "log_mileage"]:
        return "market"
    if field.startswith("component_"):
        return "components"
    if field.startswith("topic_"):
        return "topics"
    return next((b for b, fs in BLOCKS.items() if field in fs), "advertised")


def fit_encoding(frame, fields, equal_blocks=True):
    transformers = []
    weights = []
    for field in fields:
        if field in CATEGORICAL:
            transform = make_pipeline(SimpleImputer(strategy="constant", fill_value="__missing__"),
                                      OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        else:
            transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        transformers.append((field, transform, [field]))
    counts = {b: sum(feature_block(f) == b for f in fields) for b in set(map(feature_block, fields))}
    for field in fields:
        weights.append(1 / counts[feature_block(field)] if equal_blocks else 1.0)
    encoder = ColumnTransformer(transformers, transformer_weights={f: np.sqrt(w / (2 if f in CATEGORICAL else 1)) for f, w in zip(fields, weights)})
    encoder.fit(frame[fields].replace({None: np.nan}))
    return encoder


def fit_gower(frame, fields, equal_blocks=True):
    counts = {b: sum(feature_block(f) == b for f in fields) for b in set(map(feature_block, fields))}
    ranges = {f: max(float(frame[f].max() - frame[f].min()), 1e-9) if frame[f].notna().any() else 1.
              for f in fields if f not in CATEGORICAL}
    return {"fields": fields, "ranges": ranges,
            "weights": np.array([1 / counts[feature_block(f)] if equal_blocks else 1. for f in fields])}


def gower(left, right, metric):
    result = np.zeros((len(left), len(right)))
    denominator = np.zeros_like(result)
    for field, weight in zip(metric["fields"], metric["weights"]):
        a, b = left[field].to_numpy(), right[field].to_numpy()
        observed = pd.notna(a)[:, None] & pd.notna(b)[None, :]
        if field in CATEGORICAL:
            difference = a[:, None] != b[None, :]
        else:
            difference = np.minimum(np.abs(a.astype(float)[:, None] - b.astype(float)[None, :]) / metric["ranges"][field], 1)
        result += np.where(observed, difference * weight, 0)
        denominator += observed * weight
    return np.divide(result, denominator, out=np.ones_like(result), where=denominator > 0)


def fit_partition(frame, fields, method, k, config, seed, equal_blocks=True):
    if method == "kmeans":
        encoder = fit_encoding(frame, fields, equal_blocks)
        x = encoder.transform(frame[fields].replace({None: np.nan}))
        model = KMeans(k, n_init=config.starts, random_state=seed).fit(x)
        return {"method": method, "encoder": encoder, "model": model, "fields": fields}
    sample = frame.sample(min(config.medoid_cap, len(frame)), random_state=seed).reset_index(drop=True)
    metric = fit_gower(frame, fields, equal_blocks)
    distance = gower(sample, sample, metric)
    best = None
    for start in range(config.starts):
        medoids = np.random.default_rng(seed + start).choice(len(sample), k, replace=False)
        for _ in range(50):
            labels = distance[:, medoids].argmin(axis=1)
            updated = medoids.copy()
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                if len(members):
                    updated[cluster] = members[np.argmin(distance[np.ix_(members, members)].sum(axis=1))]
            if np.array_equal(updated, medoids):
                break
            medoids = updated
        loss = distance[:, medoids].min(axis=1).mean()
        if best is None or loss < best[0]:
            best = loss, medoids
    return {"method": method, "metric": metric, "centers": sample.iloc[best[1]], "fields": fields}


def assign_partition(model, frame):
    if model["method"] == "kmeans":
        return model["model"].predict(model["encoder"].transform(frame[model["fields"]].replace({None: np.nan})))
    return np.concatenate([gower(frame.iloc[i:i + 1000], model["centers"], model["metric"]).argmin(axis=1) for i in range(0, len(frame), 1000)]) if len(frame) else np.array([], dtype=int)


def resample_blocks(frame, rng):
    groups = list(frame.groupby(["Make", "Model"], sort=True).indices.values())
    indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    return frame.iloc[indices]


def stratified_sample(frame, maximum, seed):
    if len(frame) <= maximum:
        return frame.copy()
    frame = frame.copy()
    frame["_stratum"] = frame.body_type.fillna("UNKNOWN") + "|" + (frame.ModelYear.astype(int) // 5).astype(str)
    frame["_random"] = np.random.default_rng(seed).random(len(frame))
    # Round-robin sampling prevents large strata from eliminating small ones.
    frame = frame.sort_values("_random")
    frame["_round"] = frame.groupby("_stratum").cumcount()
    return frame.sort_values(["_round", "_random"]).head(maximum).drop(columns=["_stratum", "_random", "_round"])


def sample_documents(frame, maximum, seed):
    frame = frame.copy()
    frame["_random"] = np.random.default_rng(seed).random(len(frame))
    frame = frame.sort_values("_random")
    frame["_round"] = frame.groupby(FAMILY).cumcount()
    return frame.sort_values(["_round", "_random"]).head(maximum).drop(columns=["_round", "_random"])


def vehicles(run_dir):
    target = begin_stage(run_dir, "vehicles")
    source, manifest = verified_stage(run_dir, "prepare")
    if manifest["status"] != "prepared":
        return finish_stage(target, "skipped", reason=manifest["status"])
    config = run_config(manifest)
    cohorts = pd.read_parquet(source / "cohorts.parquet")
    fields, eligible, profile = select_fields(cohorts)
    profile.to_csv(target / "field_profiles.csv", index=False)
    development = cohorts[eligible & cohorts.development]
    if not all(any(f in fields for f in block) for block in BLOCKS.values()) or len(development) < 10:
        return finish_stage(target, "insufficient_specifications", fields=fields)
    training = stratified_sample(development, config.vehicle_cap, config.seed)
    anchor = development.sample(min(len(development), 1000), random_state=config.seed)
    candidates, fitted = [], {}
    for method in ["kmeans", "gower"]:
        for k in config.clusters:
            if k >= len(training):
                continue
            model = fit_partition(training, fields, method, k, config, config.seed)
            labels = assign_partition(model, anchor)
            all_labels = assign_partition(model, development)
            sizes = np.bincount(all_labels, minlength=k) / len(all_labels)
            scores = []
            rng = np.random.default_rng(config.seed)
            for repeat in range(config.stability_refits):
                sample = resample_blocks(training, rng)
                refit = fit_partition(sample, fields, method, k, config, config.seed + repeat + 1)
                scores.append(adjusted_rand_score(labels, assign_partition(refit, anchor)))
            separation = -1.0
            if 1 < len(np.unique(labels)) < len(labels):
                if method == "gower":
                    distance = gower(anchor, anchor, model["metric"])
                    np.fill_diagonal(distance, 0)
                    separation = silhouette_score(distance, labels, metric="precomputed")
                else:
                    separation = silhouette_score(model["encoder"].transform(anchor[fields].replace({None: np.nan})), labels)
            row = {"method": method, "k": k, "median_ari": float(np.median(scores)),
                   "silhouette": float(separation), "minimum_fraction": float(sizes.min())}
            row["accepted"] = row["median_ari"] >= .7 and row["minimum_fraction"] >= .05
            candidates.append(row)
            fitted[(method, k)] = model
    comparison = pd.DataFrame(candidates)
    comparison.to_csv(target / "model_comparison.csv", index=False)
    accepted = comparison[comparison.accepted].sort_values(["median_ari", "silhouette", "k", "method"], ascending=[False, False, True, True])
    if accepted.empty:
        return finish_stage(target, "no_accepted_partition", fields=fields)
    selected = accepted.iloc[0]
    model = fitted[(selected.method, int(selected.k))]
    cohorts["eligible"] = eligible
    cohorts["cluster"] = -1
    cohorts.loc[eligible, "cluster"] = assign_partition(model, cohorts.loc[eligible])
    save_table(target / "assignments.parquet", cohorts)
    joblib.dump(model, target / "partition.joblib")
    cohorts.groupby("cluster")[NUMERIC + ["vin_count", "option_valid_share", "option_mean_length", "dealer_distinct"]].median().to_csv(target / "cluster_profiles.csv")
    pd.crosstab(cohorts.cluster, cohorts.body_type).to_csv(target / "body_composition.csv")
    pd.crosstab(cohorts.cluster, cohorts.ModelYear).to_csv(target / "year_composition.csv")
    diagnostics = []
    for split, subset in cohorts[eligible].groupby("development"):
        for field in ["option_valid_share", "option_mean_length", "dealer_distinct", "vin_count"]:
            for cluster, group in subset.groupby("cluster"):
                diagnostics.append({"development": split, "cluster": cluster, "measure": field,
                                    "count": int(group[field].notna().sum()), "median": group[field].median()})
    pd.DataFrame(diagnostics).to_csv(target / "reporting_diagnostics.csv", index=False)
    within_body = []
    for body in ["SEDAN", "SUV / CROSSOVER"]:
        subset = development[development.body_type == body]
        if len(subset) > 2:
            labels = assign_partition(model, subset)
            within_body.append({"body_type": body, "cohorts": len(subset), "clusters": len(np.unique(labels)),
                                "largest_cluster_share": float(pd.Series(labels).value_counts(normalize=True).max())})
    pd.DataFrame(within_body, columns=["body_type", "cohorts", "clusters", "largest_cluster_share"]).to_csv(target / "within_body_profiles.csv", index=False)
    sensitivities = []
    for name, changed, minimum, options, equal in [
        ("single_vin", cohorts, 1, False, True), ("relaxed_agreement", relaxed_cohorts(cohorts), 5, False, True),
        ("advertised_options", cohorts, 5, True, True), ("equal_fields", cohorts, 5, False, False),
        ("complete_cases", cohorts.dropna(subset=fields), 5, False, True)]:
        sf, se, _ = select_fields(changed, minimum, options)
        dev = changed[se & changed.development]
        common = anchor.index.intersection(changed.index[se])
        if len(dev) <= int(selected.k) or len(common) < 2 or not sf:
            sensitivities.append({"sensitivity": name, "status": "insufficient_support"})
            continue
        sensitivity = fit_partition(stratified_sample(dev, config.vehicle_cap, config.seed), sf, selected.method, int(selected.k), config, config.seed, equal)
        sensitivities.append({"sensitivity": name, "status": "evaluated", "common_cohorts": len(common),
                              "ari": adjusted_rand_score(assign_partition(model, cohorts.loc[common]), assign_partition(sensitivity, changed.loc[common]))})
    pd.DataFrame(sensitivities).to_csv(target / "sensitivities.csv", index=False)
    cautions = [f"{row['sensitivity']} agreement below 0.70 (ARI={row['ari']:.3f})"
                for row in sensitivities if row.get("ari", 1) < .7]
    charts = cluster_charts(cohorts, model, training, target)
    return finish_stage(target, "accepted_partition", selected=selected.to_dict(), fields=fields,
                        reduced_budget=config.smoke, preparation_sha256=file_hash(source / "manifest.json"),
                        sensitivity_cautions=cautions, charts=charts)


def cluster_charts(cohorts, model, development, target):
    """Project the fitted inputs for display; labels remain the original assignments."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import plotly.express as px

    fields = model["fields"]
    encoder = model["encoder"] if model["method"] == "kmeans" else fit_encoding(development, fields)
    projection = PCA(n_components=2).fit(encoder.transform(development[fields].replace({None: np.nan})))
    assigned = cohorts[cohorts.cluster >= 0].copy()
    coordinates = projection.transform(encoder.transform(assigned[fields].replace({None: np.nan})))
    assigned[["profile_axis_1", "profile_axis_2"]] = coordinates
    assigned["cluster_label"] = assigned.cluster.map(lambda value: f"Cluster {value}")
    assigned["cohort_label"] = assigned[COHORT].astype(str).agg(" / ".join, axis=1)
    columns = COHORT + ["cluster", "vin_count", "eligible", *fields]
    membership = cohorts[columns].merge(assigned[COHORT + ["profile_axis_1", "profile_axis_2"]],
                                       on=COHORT, how="left", validate="one_to_one")
    membership.to_csv(target / "cohort_membership.csv", index=False)
    colors = ["#2563a6", "#d17b16", "#68883e", "#b34e87", "#7357a4", "#168482", "#a45632", "#707070"]
    palette = {f"Cluster {c}": colors[i % len(colors)] for i, c in enumerate(sorted(assigned.cluster.unique()))}
    variance = projection.explained_variance_ratio_
    axis_labels = {f"profile_axis_{i + 1}": f"Specification axis {i + 1} ({v:.1%} variance)" for i, v in enumerate(variance)}
    figure = px.scatter(assigned, x="profile_axis_1", y="profile_axis_2", color="cluster_label",
                        color_discrete_map=palette, hover_name="cohort_label",
                        hover_data={"ModelYear": True, "Make": True, "Model": True, "Trim": True,
                                    "vin_count": True, "cluster_label": False,
                                    "profile_axis_1": ":.2f", "profile_axis_2": ":.2f"},
                        labels={**axis_labels, "cluster_label": "Vehicle cluster", "vin_count": "Listing VINs"},
                        title=f"Vehicle clusters · {len(assigned):,} assigned cohorts; {(cohorts.cluster < 0).sum():,} unassigned")
    figure.update_traces(marker={"size": 6, "opacity": .55})
    figure.update_layout(template="plotly_white", height=720, margin={"b": 110}, legend={"orientation": "h", "y": 1.08})
    figure.add_annotation(text="One point = decoded year / make / model / trim. Hover for identity; click legend to filter.<br>"
                          "2D development-fitted PCA is a display projection; overlap does not change cluster membership.",
                          x=0, y=-.16, xref="paper", yref="paper", showarrow=False, xanchor="left", align="left")
    figure.write_html(target / "cluster_cohorts.html", include_plotlyjs=True, full_html=True)
    fig = Figure(figsize=(15, max(6, len(palette) * 2.8)))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1]})
    examples = []
    for cluster, group in assigned.groupby("cluster", sort=True):
        label = f"Cluster {cluster}"
        axes[0].scatter(group.profile_axis_1, group.profile_axis_2, s=8, alpha=.35,
                        color=palette[label], label=f"{label}: {len(group):,} cohorts", rasterized=True)
        center = group[["profile_axis_1", "profile_axis_2"]].median().to_numpy()
        # Examples span distinct Make/Model families near the displayed cluster center.
        ranked = group.assign(_distance=np.linalg.norm(group[["profile_axis_1", "profile_axis_2"]].to_numpy() - center, axis=1))
        sample = ranked.sort_values(["_distance", *COHORT]).drop_duplicates(["Make", "Model"]).head(5)
        examples.extend(sample.to_dict("records"))
    axes[0].set(xlabel=axis_labels["profile_axis_1"], ylabel=axis_labels["profile_axis_2"], title="All assigned cohorts")
    axes[0].legend(loc="best", markerscale=2)
    axes[1].axis("off")
    axes[1].set_title("Example cohorts near each displayed center", loc="left")
    y = .94
    row_height = .88 / (len(examples) + 2 * len(palette))
    for cluster in sorted(assigned.cluster.unique()):
        axes[1].text(0, y, f"Cluster {cluster}", color=palette[f"Cluster {cluster}"], weight="bold", transform=axes[1].transAxes)
        y -= row_height
        for example in [r for r in examples if r["cluster"] == cluster]:
            axes[1].text(0, y, example["cohort_label"], fontsize=9, transform=axes[1].transAxes)
            y -= row_height
        y -= row_height
    fig.suptitle(f"Vehicle specification clusters · {len(assigned):,} cohorts", fontsize=16)
    fig.text(.055, .025, f"Each point is a year / make / model / trim cohort. {(cohorts.cluster < 0).sum():,} unassigned cohorts excluded. "
             "Axes are a 2D projection, not safety or reliability scores.", fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .95])
    fig.savefig(target / "cluster_cohorts.png", dpi=160)
    fig.clear()
    return {"projection": "development_fitted_PCA", "explained_variance": variance.tolist(),
            "assigned_cohorts": len(assigned), "unassigned_cohorts": int((cohorts.cluster < 0).sum())}


def normalized_mixtures(weights):
    total = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)


def topic_stop_words(fitting):
    # Learn identity vocabulary only from development documents; preserve system terms.
    identities = {token for field in ["Make", "Model"] for value in fitting[field].dropna().unique()
                  for token in re.findall(r"[a-z]{3,}", str(value).lower())}
    return sorted((set(ENGLISH_STOP_WORDS) | REPORTING_WORDS | identities) - MECHANICAL_WORDS)


def complaints(run_dir, embedding_model_dir=None, embedding_revision=None):
    target = begin_stage(run_dir, "complaints")
    source, manifest = verified_stage(run_dir, "prepare")
    if manifest["status"] != "prepared":
        return finish_stage(target, "skipped", reason=manifest["status"])
    config = run_config(manifest)
    records = pd.read_parquet(source / "complaints.parquet")
    history = records[records.filing_eligible & (records.mapping_status == "exact_supported") & ~records.narrative_conflict & records.vehicle_family_present]
    for name, subset in [("all_history", history), ("primary", records[records.primary_eligible])]:
        summaries = family_complaints(subset)
        save_table(target / f"{name}_components.parquet", summaries)
    fitting = records[records.text_eligible & records.development].drop_duplicates("text_hash")
    fitting = sample_documents(fitting, config.text_cap, config.seed)
    if len(fitting) < 40:
        return finish_stage(target, "components_only", reason="insufficient_development_text")
    stop_words = topic_stop_words(fitting)
    write_json(target / "text_preprocessing.json", {"stop_words": stop_words,
               "identity_vocabulary": "development Make/Model tokens; mechanical words preserved",
               "token_pattern": r"(?u)\b[a-z][a-z]+\b", "source_narratives": "unchanged"})
    vectorizer = TfidfVectorizer(stop_words=stop_words, token_pattern=r"(?u)\b[a-z][a-z]+\b",
                                 min_df=2, max_df=.95, max_features=20000)
    try:
        x = vectorizer.fit_transform(fitting.normalized_text)
    except ValueError as error:
        return finish_stage(target, "components_only", reason=str(error))
    models, candidates, examples, topic_checks = {}, [], [], []
    for count in config.topics:
        if count >= min(x.shape):
            continue
        model = NMF(count, init="nndsvda", random_state=config.seed, max_iter=400).fit(x)
        similarities = []
        for repeat in range(config.topic_refits):
            indices = np.random.default_rng(config.seed + repeat).integers(0, len(fitting), len(fitting))
            refit = NMF(count, init="random", random_state=config.seed + repeat, max_iter=400).fit(x[indices])
            similarity = normalize(model.components_) @ normalize(refit.components_).T
            a, b = linear_sum_assignment(-similarity)
            similarities.append(float(np.median(similarity[a, b])))
        weights = normalized_mixtures(model.transform(x))
        for topic in range(count):
            representative = np.argsort(-weights[:, topic])[:5]
            pool = np.setdiff1d(np.flatnonzero(weights.argmax(axis=1) == topic), representative)
            random = np.random.default_rng(config.seed + topic).choice(pool, min(5, len(pool)), replace=False)
            top_terms = vectorizer.get_feature_names_out()[np.argsort(-model.components_[topic])[:12]]
            terms = ", ".join(top_terms)
            mechanical_share = float(np.mean([term in MECHANICAL_WORDS for term in top_terms]))
            support = int((weights.argmax(axis=1) == topic).sum())
            topic_checks.append({"count": count, "topic": topic, "terms": terms,
                                 "mechanical_term_share": mechanical_share, "dominant_documents": support,
                                 "screen_passed": mechanical_share >= .4 and support >= 10 and len(random) == 5})
            for kind, indices in [("representative", representative), ("random", random)]:
                for index in indices:
                    examples.append({"count": count, "topic": topic, "kind": kind,
                                     "text_hash": fitting.iloc[index].text_hash, "summary": fitting.iloc[index].summary,
                                     "terms": terms, "judgment": "", "notes": ""})
        models[count] = model
        passing = float(np.mean([row["screen_passed"] for row in topic_checks if row["count"] == count]))
        candidates.append({"count": count, "median_similarity": float(np.median(similarities)),
                           "screened_topic_share": passing,
                           "automatically_eligible": passing >= .8 and float(np.median(similarities)) >= .7})
    pd.DataFrame(candidates, columns=["count", "median_similarity"]).to_csv(target / "topic_candidates.csv", index=False)
    pd.DataFrame(examples, columns=["count", "topic", "kind", "text_hash", "summary", "terms", "judgment", "notes"]).to_csv(target / "topic_review_template.csv", index=False)
    pd.DataFrame(topic_checks).to_csv(target / "topic_checks.csv", index=False)
    selected = next((row["count"] for row in candidates if row["automatically_eligible"]), None)
    write_json(target / "topic_selection.json", {"method": "automated_lexical_and_stability_screen",
               "selected_topics": selected, "semantic_validation": "not_established",
               "thresholds": {"matched_refit_similarity": .7, "passing_topic_share": .8,
                              "mechanical_share_of_top_12_terms": .4, "dominant_documents": 10,
                              "random_examples": 5},
               "interpretation": "exploratory themes; lexical screening is not human coherence validation"})
    joblib.dump({"vectorizer": vectorizer, "models": models}, target / "topics.joblib")
    embedding_status = embedding_sensitivity(records, fitting, target, config, embedding_model_dir, embedding_revision)
    return finish_stage(target, "automated_topics" if selected is not None else "components_only",
                        reason="automated exploratory screening" if selected is not None else "no topic candidate passed automated screening",
                        selected_topics=selected, semantic_validation="not_established",
                        development_documents=len(fitting), reduced_budget=config.smoke, embedding=embedding_status)


def spherical_fit(values, count, starts, seed):
    best = None
    for start in range(starts):
        centers = values[np.random.default_rng(seed + start).choice(len(values), count, replace=False)].copy()
        for _ in range(100):
            labels = (values @ centers.T).argmax(axis=1)
            updated = np.array([values[labels == c].mean(axis=0) if (labels == c).any() else centers[c] for c in range(count)])
            updated = normalize(updated)
            if np.allclose(updated, centers):
                break
            centers = updated
        score = (values @ centers.T).max(axis=1).mean()
        if best is None or score > best[0]:
            best = score, centers
    return best[1]


def token_chunks(text, tokenizer, maximum, overlap=32):
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    window = maximum - tokenizer.num_special_tokens_to_add(pair=False)
    if window <= overlap:
        raise ValueError("Token window must exceed overlap")
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + window, len(ids))
        chunk = tokenizer.decode(ids[start:end], skip_special_tokens=True)
        # Decoding and retokenizing subwords can change the length at boundaries.
        while len(tokenizer.encode(chunk, add_special_tokens=True, truncation=False)) > maximum:
            end -= 1
            if end <= start:
                raise ValueError("Tokenizer cannot produce a bounded text chunk")
            chunk = tokenizer.decode(ids[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        if end == len(ids):
            break
        start = max(start + 1, end - overlap)
    return chunks or [""]


def embedding_sensitivity(records, fitting, target, config, model_dir=None, revision=None):
    """Local-only MiniLM sensitivity; token windows preserve long narratives."""
    if model_dir is None:
        return {"status": "skipped", "reason": "supply local MiniLM snapshot with --embedding-model-dir and --embedding-revision"}
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Embedding revision must be the pinned 40-character Hugging Face commit")
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {"status": "unavailable", "reason": "sentence-transformers is not installed"}
    model = SentenceTransformer(str(model_dir), local_files_only=True)
    tokenizer = model.tokenizer
    window = int(model.max_seq_length) - tokenizer.num_special_tokens_to_add(pair=False)
    if window < 32:
        raise ValueError("Unexpected MiniLM token window")
    vocabulary = sorted({str(v) for c in ["Make", "Model"] for v in records[c].dropna().unique()}, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(v) for v in vocabulary) + r")\b", re.I)
    documents = records[records.text_eligible].drop_duplicates("text_hash").sort_values("text_hash")
    fitting_hashes = set(fitting.text_hash)
    fit_mask = documents.text_hash.isin(fitting_hashes).to_numpy()
    receipts, assignments = [], []
    cache = {}
    for variant in ["raw", "identity_masked"]:
        vectors, chunk_counts = [], []
        for text in documents.normalized_text:
            text = pattern.sub("vehicle", text) if variant == "identity_masked" else text
            chunks = token_chunks(text, tokenizer, int(model.max_seq_length))
            embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
            vectors.append(normalize(np.asarray(embeddings).mean(axis=0, keepdims=True))[0])
            chunk_counts.append(len(chunks))
        vectors = np.asarray(vectors)
        np.savez_compressed(target / f"minilm_{variant}.npz", vectors=vectors, text_hash=documents.text_hash.to_numpy(dtype=str))
        cache[variant] = {}
        for count in config.topics:
            if fit_mask.sum() <= count:
                continue
            centers = spherical_fit(vectors[fit_mask], count, config.starts, config.seed)
            labels = (vectors @ centers.T).argmax(axis=1)
            cache[variant][count] = labels
            for text_hash, development, label in zip(documents.text_hash, documents.development, labels):
                assignments.append({"text_hash": text_hash, "development": development, "variant": variant, "count": count, "cluster": int(label)})
        receipts.append({"variant": variant, "documents": len(documents), "multi_chunk_documents": int((np.array(chunk_counts) > 1).sum()),
                         "chunks": sum(chunk_counts), "truncated_documents": 0})
    pd.DataFrame(assignments).to_csv(target / "minilm_assignments.csv", index=False)
    pd.DataFrame([{"count": count, "development_raw_masked_ari": adjusted_rand_score(cache["raw"][count][fit_mask], cache["identity_masked"][count][fit_mask])}
                  for count in cache["raw"]]).to_csv(target / "minilm_identity_sensitivity.csv", index=False)
    return {"status": "descriptive_sensitivity", "model": "sentence-transformers/all-MiniLM-L6-v2", "revision": revision,
            "files": {str(p.relative_to(model_dir)): file_hash(p) for p in model_dir.rglob("*") if p.is_file()},
            "token_window": window, "overlap_tokens": 32, "coverage": receipts}


def family_complaints(records, topic_weights=None):
    component_fields = ["component_" + name for name in COMPONENTS]
    columns = FAMILY + ["complaint_count", *component_fields]
    if topic_weights is not None:
        columns += ["topic_complaint_count", *topic_weights.columns]
    if records.empty:
        return pd.DataFrame(columns=columns)
    incident_keys = FAMILY + ["odiNumber"]
    flags = records[incident_keys].copy()
    components = records.components.fillna("").astype(str)
    for name, pattern in COMPONENTS.items():
        flags["component_" + name] = components.str.contains(pattern, case=False, regex=True).to_numpy()
    # Union component labels before counting each incident once within its family.
    incidents = flags.groupby(incident_keys, observed=True, sort=False)[component_fields].max().reset_index()
    grouped = incidents.groupby(FAMILY, observed=True, sort=True)
    result = grouped[component_fields].mean()
    result.insert(0, "complaint_count", grouped.size())
    result = result.reset_index()
    if topic_weights is not None:
        identities = records[incident_keys + ["text_hash"]].drop_duplicates(incident_keys)
        # Join once: repeated .isin/.loc against a large string index is quadratic in families.
        linked = identities.merge(topic_weights.rename_axis("text_hash").reset_index(),
                                  on="text_hash", how="left", validate="many_to_one", indicator=True)
        linked["topic_complaint_count"] = (linked._merge == "both").astype(int)
        grouped_topics = linked.groupby(FAMILY, observed=True, sort=True)
        topics = grouped_topics[list(topic_weights.columns)].mean()
        topics.insert(0, "topic_complaint_count", grouped_topics.topic_complaint_count.sum())
        result = result.merge(topics.reset_index(), on=FAMILY, validate="one_to_one")
    result["ModelYear"] = result.ModelYear.astype("int64")
    return result[columns]


def family_profiles(assignments):
    rows = []
    clusters = sorted(assignments.loc[assignments.cluster >= 0, "cluster"].unique())
    for key, group in assignments.groupby(FAMILY, sort=True, observed=True):
        row = dict(zip(FAMILY, key))
        assigned = group[group.cluster >= 0]
        row.update(trim_count=len(group), assigned_trim_count=len(assigned), assignment_coverage=len(assigned) / len(group),
                   excluded_trim_count=int((~group.eligible).sum()), unassigned_trim_count=int((group.cluster < 0).sum()))
        for cluster in clusters:
            row[f"cluster_{cluster}"] = float((group.cluster == cluster).mean())
            row[f"listing_cluster_{cluster}"] = float(group.loc[group.cluster == cluster, "vin_count"].sum() / group.vin_count.sum())
        row["concentration"] = max([(assigned.cluster == c).mean() for c in clusters], default=0) if len(assigned) else 0
        for field in NUMERIC:
            row[field] = group[field].median()
        for field in ["body_type", *CATEGORICAL]:
            row[field] = mode_agreement(group[field])[0]
        row["development"] = bool(group.development.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def validate_mapping_review(template, reviewed):
    if reviewed is None:
        return False, "mapping_review_pending"
    if reviewed.review_id.duplicated().any() or set(template.review_id) != set(reviewed.review_id):
        raise ValueError("Mapping review IDs must match the complete template exactly")
    joined = template[["review_id", "mapping_status"]].merge(reviewed[["review_id", "judgment"]], on="review_id", validate="one_to_one")
    if not joined.judgment.isin(["correct", "incorrect", "unclear"]).all():
        return False, "mapping_review_incomplete"
    if ((joined.mapping_status == "exact_supported") & (joined.judgment != "correct")).any():
        return False, "mapping_rule_blocked"
    return True, "reviewed"


def choose_topics(source, review):
    if review is None:
        selection = source / "topic_selection.json"
        return read_json(selection)["selected_topics"] if selection.is_file() else None
    if not (source / "topic_candidates.csv").exists():
        return None
    template = pd.read_csv(source / "topic_review_template.csv")
    identity = ["count", "topic", "kind", "text_hash"]
    if review.duplicated(identity).any() or set(map(tuple, template[identity].to_numpy())) != set(map(tuple, review[identity].to_numpy())):
        raise ValueError("Topic review must match every template example exactly")
    for row in pd.read_csv(source / "topic_candidates.csv").sort_values("count").itertuples():
        subset = review[review["count"] == row.count]
        if not subset.judgment.isin(["coherent", "mixed", "unclear"]).all():
            continue
        coherent = []
        for topic in range(row.count):
            examples = subset[subset.topic == topic]
            sufficient = all((examples.kind == kind).sum() == 5 for kind in ["representative", "random"])
            coherent.append(sufficient and (examples.judgment == "coherent").mean() >= .8)
        if np.mean(coherent) >= .8 and row.median_similarity >= .7:
            return int(row.count)
    return None


def automatic_mapping_checks(records):
    primary = records[records.primary_eligible]
    unique = records.loc[records.mapping_status == "exact_supported", FAMILY + ["products"]].drop_duplicates()
    invalid_products = 0
    for row in unique.itertuples(index=False):
        try:
            products = json.loads(row.products) if isinstance(row.products, str) else row.products
            identities = {(str(p.get("productYear")), canonical(p.get("productMake")), canonical(p.get("productModel")))
                          for p in products if isinstance(p, dict) and str(p.get("type", "")).upper() == "VEHICLE"}
            invalid_products += (str(row.ModelYear), row.Make, row.Model) not in identities
        except (ValueError, TypeError):
            invalid_products += 1
    checks = {"exact_product_mismatches": int(invalid_products),
              "duplicate_primary_incidents": int(primary.duplicated(FAMILY + ["odiNumber"]).sum()),
              "ambiguous_primary_incidents": int((primary.multi_family | primary.narrative_conflict).sum()),
              "invalid_primary_windows": int((~primary.dateComplaintFiled.notna()
                  | (primary.dateComplaintFiled > CUTOFF) | (primary.ModelYear > 2021)
                  | (primary.dateComplaintFiled.dt.year < primary.ModelYear)
                  | (primary.dateComplaintFiled.dt.year > primary.ModelYear + 4)).sum()),
              "text_split_leaks": int(records.loc[records.text_eligible].groupby("text_hash").development.nunique().gt(1).sum())}
    return {"method": "automated_product_identity_and_eligibility_checks", "passed": not any(checks.values()),
            "unique_supported_product_mappings": len(unique), "primary_family_incidents": len(primary),
            "checks": checks, "manual_validation": "not_performed"}


def ridge_prediction(training, heldout, fields, target, seed):
    numeric = [f for f in fields if f not in ["body_type", *CATEGORICAL]]
    categorical = [f for f in fields if f not in numeric]
    transform = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="constant", fill_value="__missing__"), OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical)])
    pipeline = make_pipeline(transform, Ridge())
    blocks = training.Make.astype(str) + "|" + training.Model.astype(str)
    folds = min(5, blocks.nunique())
    if folds < 2:
        raise ValueError("At least two development Make/Model blocks required")
    selection = GridSearchCV(pipeline, {"ridge__alpha": [.1, 1., 10., 100.]}, cv=GroupKFold(folds), scoring="neg_mean_absolute_error")
    selection.fit(training[fields].replace({None: np.nan}), training[target], groups=blocks)
    return selection.predict(heldout[fields].replace({None: np.nan})), selection.best_params_["ridge__alpha"]


def paired_intervals(errors, blocks, draws, seed):
    unique = np.unique(blocks)
    if len(unique) < 20:
        return None
    groups = [np.flatnonzero(blocks == block) for block in unique]
    rng = np.random.default_rng(seed)
    means = [errors[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(draws)]
    return np.quantile(means, [.025, .975]).tolist()


def compare(run_dir, mapping_review=None, topic_review=None):
    name = "compare" if not (Path(run_dir) / "compare").exists() else "compare-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = begin_stage(run_dir, name)
    prepared, preparation = verified_stage(run_dir, "prepare")
    vehicle_dir, vehicle_manifest = verified_stage(run_dir, "vehicles")
    if vehicle_manifest["status"] != "accepted_partition":
        return finish_stage(target, "skipped", reason=vehicle_manifest["status"])
    complaint_dir, complaint_manifest = verified_stage(run_dir, "complaints")
    if complaint_manifest["status"] == "skipped":
        return finish_stage(target, "skipped", reason="complaint_stage_unavailable")
    records = pd.read_parquet(prepared / "complaints.parquet")
    mapping_checks = automatic_mapping_checks(records)
    write_json(target / "mapping_checks.json", mapping_checks)
    review = pd.read_csv(mapping_review) if mapping_review else None
    valid, reason = mapping_checks["passed"], "automated_mapping_checks_failed"
    if review is not None:
        template = pd.read_csv(prepared / "mapping_review_template.csv")
        reviewed_valid, reason = validate_mapping_review(template, review)
        valid = valid and reviewed_valid
        review.to_csv(target / "mapping_review.csv", index=False)
    families = family_profiles(pd.read_parquet(vehicle_dir / "assignments.parquet"))
    save_table(target / "family_profiles.parquet", families)
    if not valid:
        return finish_stage(target, "skipped", reason=reason)
    config = run_config(preparation)
    reviewed_topics = pd.read_csv(topic_review) if topic_review else None
    count = choose_topics(complaint_dir, reviewed_topics)
    if reviewed_topics is not None:
        reviewed_topics.to_csv(target / "topic_review.csv", index=False)
    weights = None
    if count is not None:
        saved = joblib.load(complaint_dir / "topics.joblib")
        documents = records[records.text_eligible].drop_duplicates("text_hash")
        values = normalized_mixtures(saved["models"][count].transform(saved["vectorizer"].transform(documents.normalized_text)))
        weights = pd.DataFrame(values, index=documents.text_hash, columns=[f"topic_{i}" for i in range(count)])
        weights = weights.loc[weights.sum(axis=1) > 0]
    outcomes = family_complaints(records[records.primary_eligible], weights)
    data = families.merge(outcomes, on=FAMILY, validate="one_to_one")
    save_table(target / "family_evidence.parquet", data)
    data = data[(data.complaint_count >= 20) & (data.assignment_coverage > 0)].copy()
    training, heldout = data[data.development], data[~data.development]
    if len(training) < 10 or len(heldout) < 2:
        return finish_stage(target, "insufficient_overlap", families=len(data), selected_topics=count)
    baseline = ["body_type", "ModelYear", *CATEGORICAL]
    variants = {"baseline": baseline, "clusters": baseline + [c for c in data if c.startswith("cluster_")],
                "continuous": baseline + NUMERIC}
    reports, predictions, intervals = [], [], []
    blocks = (heldout.Make.astype(str) + "|" + heldout.Model.astype(str)).to_numpy()
    targets = [c for c in outcomes if c.startswith(("component_", "topic_")) and c != "topic_complaint_count"]
    for outcome in targets:
        fit = training.dropna(subset=[outcome])
        test = heldout.dropna(subset=[outcome])
        if outcome.startswith("topic_"):
            fit, test = fit[fit.topic_complaint_count >= 20], test[test.topic_complaint_count >= 20]
        if len(fit) < 10 or len(test) < 2:
            continue
        errors = {}
        for name, fields in variants.items():
            predicted, alpha = ridge_prediction(fit, test, fields, outcome, config.seed)
            errors[name] = np.abs(test[outcome].to_numpy() - predicted)
            reports.append({"outcome": outcome, "variant": name, "mae": errors[name].mean(), "alpha": alpha, "heldout_families": len(test)})
            for key, actual, value in zip(test[FAMILY].to_numpy(), test[outcome], predicted):
                predictions.append({**dict(zip(FAMILY, key)), "outcome": outcome, "variant": name, "actual": actual, "prediction": value})
        test_blocks = (test.Make.astype(str) + "|" + test.Model.astype(str)).to_numpy()
        for name in ["clusters", "continuous"]:
            difference = errors[name] - errors["baseline"]
            for seed in [42, 43]:
                bounds = paired_intervals(difference, test_blocks, config.bootstrap, seed)
                intervals.append({"outcome": outcome, "variant": name, "mean_mae_change": difference.mean(), "seed": seed,
                                  "lower": bounds[0] if bounds else None, "upper": bounds[1] if bounds else None,
                                  "heldout_blocks": len(np.unique(test_blocks)), "interval_status": "available" if bounds else "fewer_than_20_blocks"})
    pd.DataFrame(reports).to_csv(target / "comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(target / "predictions.csv", index=False)
    pd.DataFrame(intervals).to_csv(target / "paired_intervals.csv", index=False)
    save_table(target / "homogeneous_family_descriptive.parquet", data[data.concentration >= .9])
    return finish_stage(target, "evaluated", selected_topics=count,
                        mapping_status="automated_checks_passed" if review is None else "reviewed_and_checked",
                        topic_status=("reviewed" if reviewed_topics is not None else "automated_exploratory") if count else "components_only",
                        semantic_validation="not_established" if reviewed_topics is None else "review_provided",
                        sensitivity_cautions=vehicle_manifest.get("sensitivity_cautions", []),
                        heldout_blocks=len(np.unique(blocks)), reduced_budget=config.smoke,
                        interpretation="ecological associations among reported complaints; not failure rates or safety rankings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "vehicles", "complaints", "compare", "segments", "complaint_profiles", "relationships"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "DATA/CAR_DATA_FINAL")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Reduced computation; never final study evidence")
    parser.add_argument("--mapping-review", type=Path)
    parser.add_argument("--topic-review", type=Path)
    parser.add_argument("--embedding-model-dir", type=Path, help="Local all-MiniLM-L6-v2 snapshot; no implicit download")
    parser.add_argument("--embedding-revision", help="Pinned 40-character source commit for local embedding snapshot")
    args = parser.parse_args()
    if args.stage != "prepare" and args.run_dir is None:
        parser.error("--run-dir is required for subsequent stages")
    run = args.run_dir or ROOT / "MODEL_OUTPUT/vehicle_profiles" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config = Config(vehicle_cap=200, medoid_cap=100, text_cap=200, starts=2, stability_refits=2,
                    topic_refits=2, bootstrap=20, clusters=(2, 3), topics=(2, 3), smoke=True) if args.smoke else Config()
    if args.stage == "prepare":
        result = prepare(args.data_dir, run, config)
    elif args.stage == "compare":
        result = compare(run, args.mapping_review, args.topic_review)
    elif args.stage == "complaints":
        result = complaints(run, args.embedding_model_dir, args.embedding_revision)
    elif args.stage in ["segments", "complaint_profiles", "relationships"]:
        from MODELS import vehicle_segmentation
        result = getattr(vehicle_segmentation, args.stage)(run)
    else:
        result = globals()[args.stage](run)
    print(json.dumps({"run_dir": str(run.resolve()), **read_json(result / "manifest.json")}, indent=2, default=str))


if __name__ == "__main__":
    main()
