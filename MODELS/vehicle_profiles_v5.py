"""VIN-level evidence reconciliation and independent cohort/model discovery grains."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

from EDA.DATA_CLEANING import EXCLUDED_LISTING_VINS
from MODELS import vehicle_profiles as r
from MODELS import vehicle_taxonomy as taxonomy
from MODELS import vehicle_taxonomy_v2 as v2

NUMERIC = v2.NUMERIC + ["width"]
EXTRA = ["engine_displacement", "engine_cylinders", "curb_weight_lb", "doors", "engine_hp_low", "engine_hp_high", "wheelbase_low", "wheelbase_high"]
CATEGORICAL = v2.CATEGORICAL
SOURCE_FIELDS = ["seating_source", "horsepower_source", "wheelbase_source", "displacement_source"]


def reconcile_displacement(frame):
    listing = pl.col("engine_displacement").cast(pl.Float64, strict=False)
    decoded = pl.col("DisplacementCC").cast(pl.Float64, strict=False)
    a = listing.is_finite() & (listing > 0)
    b = decoded.is_finite() & (decoded > 0)
    # Advertised liters are rounded; accept 5% or 50 cc before flagging a disagreement.
    conflict = a & b & ((listing-decoded).abs() > pl.max_horizontal(pl.lit(50.), .05*decoded))
    inapplicable = pl.col("powertrain").is_in(["battery_electric", "fuel_cell"])
    return frame.with_columns(
        pl.col("engine_displacement").alias("listing_engine_displacement"),
        pl.when(inapplicable | conflict).then(None).when(a).then(listing).when(b).then(decoded).otherwise(None).alias("engine_displacement"),
        pl.when(inapplicable).then(pl.lit("not_applicable")).when(conflict).then(pl.lit("conflicting"))
          .when(a & b).then(pl.lit("compatible_observed")).when(a).then(pl.lit("listing"))
          .when(b).then(pl.lit("vpic")).otherwise(pl.lit("unknown")).alias("displacement_source"))


def drive(value):
    value = str(value or "").upper().strip()
    for prefix, label in [("FWD", "FWD"), ("RWD", "RWD"), ("AWD", "AWD"), ("4WD", "4WD"), ("4X2", "2WD")]:
        if value.startswith(prefix):
            return label
    return "UNKNOWN"


def reconcile_drive(listing, decoded):
    a, b = drive(listing), drive(decoded)
    broad = lambda x: "TWO_WHEEL" if x in ("FWD", "RWD", "2WD") else "ALL_WHEEL" if x in ("AWD", "4WD") else "UNKNOWN"
    if a == "UNKNOWN" and b == "UNKNOWN":
        return None, "UNKNOWN", "unknown"
    if a == "UNKNOWN" or b == "UNKNOWN":
        value = b if a == "UNKNOWN" else a
        return value, broad(value), "vpic" if a == "UNKNOWN" else "listing"
    if a == b:
        return a, broad(a), "agreed"
    if broad(a) != broad(b):
        return None, "CONFLICTING", "conflicting"
    if "2WD" in (a, b):
        return b if a == "2WD" else a, "TWO_WHEEL", "compatible_broad"
    if {a, b} == {"AWD", "4WD"}:
        return "ALL_WHEEL", "ALL_WHEEL", "detail_unresolved"
    return None, "TWO_WHEEL", "detail_conflicting"


def propulsion(fuel, primary, secondary, electrification, displacement):
    values = [str(x or "").upper() for x in (fuel, primary, secondary, electrification)]
    listed, first, second, level = values
    combustion = any(any(t in x for t in ("GASOLINE", "DIESEL", "FLEX", "ETHANOL", "PROPANE", "NATURAL GAS", "LPG")) for x in values[:3])
    if "FCEV" in level or "FUEL CELL" in first:
        if combustion or pd.notna(displacement) and displacement>0:
            return "conflicting","conflicting",combustion
        return "fuel_cell", "observed", combustion
    if "PHEV" in level or "PLUG-IN" in level:
        return "plug_in_hybrid", "observed", True
    if "BEV" in level:
        if combustion or listed=="HYBRID" or pd.notna(displacement) and displacement > 0:
            return "conflicting", "conflicting", combustion
        return "battery_electric", "observed", False
    if "HEV" in level or "HYBRID" in level:
        return "hybrid", "observed", True
    if "HYBRID" in listed or first == "ELECTRIC" and combustion or second == "ELECTRIC" and combustion:
        return "hybrid_unspecified","rule_derived",True
    if (listed=="ELECTRIC" or first=="ELECTRIC") and not second and not combustion:
        return ("conflicting","conflicting",False) if pd.notna(displacement) and displacement>0 else ("battery_electric","observed",False)
    if "NOT ELECTRIFIED" in level and combustion:
        return "combustion", "observed", True
    return ("combustion_unverified", "rule_derived", True) if combustion else ("unknown", "unknown", False)


def join_sources(data_dir):
    data_dir = Path(data_dir)
    listing_fields = ["vin", "make_name", "model_name", "trim_name", "year", "body_type", "fuel_type", "engine_displacement", "maximum_seating", "wheelbase", "length", "height", "width", "front_legroom", "back_legroom", "horsepower", "torque", "transmission", "wheel_system", "price", "mileage", "listed_date"]
    decoder_fields = ["vin", *r.COHORT, "BodyClass", "VehicleType", "DriveType", "Seats", "EngineCylinders", "FuelTypePrimary", "FuelTypeSecondary", "ElectrificationLevel", "DisplacementCC", "EngineHP", "EngineHP_to", "WheelBaseShort", "WheelBaseLong", "CurbWeightLB", "Doors"]
    left = pl.scan_parquet(data_dir / "used_cars.parquet").select(listing_fields)
    if left.select(pl.len()).collect().item() != left.select(pl.col("vin").n_unique()).collect().item():
        raise ValueError("Duplicate listing VINs")
    left = left.filter(~pl.col("vin").is_in(sorted(EXCLUDED_LISTING_VINS)))
    right = pl.scan_parquet(data_dir / "NHTSA_vPIC.parquet").select(decoder_fields).unique()
    duplicates = right.group_by("vin").len().filter(pl.col("len") > 1).select("vin")
    conflicts = duplicates.collect().height
    right = right.join(duplicates, on="vin", how="anti").with_columns([v2.canonical_expr(c).alias(c) for c in r.COHORT[1:]])
    year = pl.col("ModelYear").cast(pl.Float64, strict=False)
    right = right.with_columns(pl.when(year.is_finite() & year.eq(year.floor())).then(year.cast(pl.Int64, strict=False)).alias("ModelYear"))
    joined = left.join(right, on="vin", how="left", validate="1:1").collect(engine="streaming")
    if joined.height != left.select(pl.len()).collect().item():
        raise ValueError("VIN join changed listing population")
    # Retain raw source fields alongside reconciled evidence in a separate VIN audit.
    joined = joined.with_columns(pl.col("wheel_system").alias("listing_wheel_system"), pl.col("horsepower").alias("listing_horsepower"), pl.col("wheelbase").alias("listing_wheelbase"))
    drives = {pair: reconcile_drive(*pair) for pair in joined.select("wheel_system", "DriveType").unique().iter_rows()}
    powers = {pair: propulsion(*pair) for pair in joined.select("fuel_type", "FuelTypePrimary", "FuelTypeSecondary", "ElectrificationLevel", "engine_displacement").unique().iter_rows()}
    joined = joined.with_columns(
        pl.struct("wheel_system", "DriveType").map_elements(lambda x: dict(zip(["detail", "broad", "state"], drives[(x["wheel_system"], x["DriveType"])])), return_dtype=pl.Struct({"detail": pl.String, "broad": pl.String, "state": pl.String})).alias("drive_result"),
        pl.struct("fuel_type", "FuelTypePrimary", "FuelTypeSecondary", "ElectrificationLevel", "engine_displacement").map_elements(lambda x: dict(zip(["type", "state", "combustion"], powers[tuple(x.values())])), return_dtype=pl.Struct({"type": pl.String, "state": pl.String, "combustion": pl.Boolean})).alias("power_result"))
    joined = taxonomy_body(joined).with_columns(
        pl.col("drive_result").struct.field("detail").alias("wheel_system"), pl.col("drive_result").struct.field("broad").alias("drive_family"), pl.col("drive_result").struct.field("state").alias("drivetrain_state"),
        pl.col("power_result").struct.field("type").alias("powertrain"), pl.col("power_result").struct.field("state").alias("powertrain_state"), pl.col("power_result").struct.field("combustion").alias("combustion_observed"),
        pl.col("maximum_seating").cast(pl.String).str.extract(r"^(\d+)\s+seats?$", 1).cast(pl.Float64, strict=False).alias("listing_seats"))
    joined = reconcile_displacement(joined)
    for original, normalized in [("EngineHP", "engine_hp_low"), ("EngineHP_to", "engine_hp_high"), ("WheelBaseShort", "wheelbase_low"), ("WheelBaseLong", "wheelbase_high"), ("EngineCylinders", "engine_cylinders"), ("CurbWeightLB", "curb_weight_lb"), ("Doors", "doors")]:
        joined = joined.with_columns(pl.col(original).cast(pl.Float64, strict=False).alias(normalized))
    seat_conflict = pl.col("listing_seats").is_not_null() & (pl.col("Seats") > 0) & (pl.col("listing_seats") != pl.col("Seats"))
    joined = joined.with_columns(
        pl.when(seat_conflict).then(None).otherwise(pl.coalesce("listing_seats", pl.when(pl.col("Seats") > 0).then(pl.col("Seats")))).alias("seating"),
        pl.when(seat_conflict).then(pl.lit("conflicting")).when(pl.col("listing_seats").is_not_null()).then(pl.lit("listing")).when(pl.col("Seats") > 0).then(pl.lit("vpic")).otherwise(pl.lit("unknown")).alias("seating_source"))
    for field, low, high in [("horsepower", "engine_hp_low", "engine_hp_high"), ("wheelbase", "wheelbase_low", "wheelbase_high")]:
        exact = (pl.col(low) > 0) & (pl.col(low) == pl.col(high))
        tolerance = .05 * pl.col(high) if field == "horsepower" else pl.lit(1.)
        disagrees = (pl.col(field) > 0) & (pl.col(low) > 0) & (pl.col(high) >= pl.col(low)) & ((pl.col(field) < pl.col(low)-tolerance) | (pl.col(field) > pl.col(high)+tolerance))
        joined = joined.with_columns(pl.when(pl.col(field) > 0).then(pl.col(field)).when(exact).then(pl.col(low)).otherwise(None).alias(field),
            pl.when(pl.col(field) > 0).then(pl.lit("listing")).when(exact).then(pl.lit("vpic_equal_endpoints")).otherwise(pl.lit("unknown")).alias(field + "_source"),
            disagrees.fill_null(False).alias(field + "_decoder_bounds_conflict"))
    torque = {v: taxonomy.parse_torque(v) for v in joined["torque"].unique().drop_nulls()}
    joined = joined.with_columns(pl.col("torque").replace_strict(torque, default=None, return_dtype=pl.Float64).alias("torque_nm"))
    joined = joined.with_columns(pl.all_horizontal([pl.col(c).is_not_null() for c in r.COHORT]).alias("cohort_complete"),
                                pl.all_horizontal([pl.col(c).is_not_null() for c in r.FAMILY]).alias("model_complete"))
    return joined.drop("drive_result", "power_result"), conflicts


def taxonomy_body(frame):
    return v2.body_expressions(frame)


def aggregate(frame, grain):
    data = frame if grain == "cohort" else frame.with_columns(pl.lit(None, dtype=pl.String).alias("Trim"))
    data = data.with_columns(pl.col("cohort_complete" if grain == "cohort" else "model_complete").alias("identity_complete"))
    groups = [*r.COHORT, "identity_complete", "body_style", "powertrain"]
    aggregations = [pl.len().alias("vin_count")]
    for field in NUMERIC + EXTRA:
        value = pl.col(field).cast(pl.Float64, strict=False)
        value = pl.when(value.is_finite() & (value > 0)).then(value)
        aggregations.extend([value.median().alias(field), value.count().alias(field + "_n"), (value.quantile(.75) - value.quantile(.25)).alias(field + "_iqr")])
        if field=="curb_weight_lb":
            aggregations.extend([value.min().alias(field+"_min"),value.max().alias(field+"_max")])
    for field in CATEGORICAL + ["drive_family", "VehicleType", "powertrain_state", "drivetrain_state", *SOURCE_FIELDS]:
        value = v2.canonical_expr(field)
        aggregations.extend([value.drop_nulls().mode().sort().first().alias(field), value.count().alias(field + "_n"), value.drop_nulls().value_counts(sort=True).struct.field("count").first().alias(field + "_mode_n")])
    aggregations.extend([pl.col("price").median().alias("price_median"), pl.col("mileage").median().alias("mileage_median")])
    aggregations.extend(pl.col(field + "_decoder_bounds_conflict").sum().alias(field + "_decoder_bounds_conflict_n") for field in ("horsepower", "wheelbase") if field + "_decoder_bounds_conflict" in data.columns)
    profiles = data.group_by(groups).agg(aggregations).sort(groups).to_pandas()
    profiles["ModelYear"] = pd.to_numeric(profiles.ModelYear).astype("Int64")
    profiles["grain"] = grain
    profiles["profile_id"] = [v2.stable_id([grain, *[None if pd.isna(v) else v for v in row]]) for row in profiles[groups].itertuples(index=False, name=None)]
    profiles["parent_id"] = profiles.profile_id
    for field in NUMERIC + EXTRA:
        valid = profiles[field + "_n"].ge(5) & profiles[field + "_iqr"].le(.2 * profiles[field])
        profiles[field + "_state"] = np.where(valid, "observed", np.where(profiles[field + "_n"].eq(0), "unknown", "insufficient_or_heterogeneous"))
        profiles.loc[~valid, field] = np.nan
    for field in CATEGORICAL + ["drive_family", "VehicleType", "powertrain_state", "drivetrain_state", *SOURCE_FIELDS]:
        profiles[field + "_agreement"] = profiles[field + "_mode_n"] / profiles[field + "_n"].replace(0, np.nan)
        profiles.loc[profiles[field + "_agreement"].lt(.9), field] = None
    profiles["body_family"] = profiles.body_style.map(taxonomy.body_family)
    profiles.loc[profiles.powertrain.isin(["battery_electric", "fuel_cell"]), "engine_displacement"] = np.nan
    profiles.loc[profiles.powertrain.isin(["battery_electric", "fuel_cell"]), "engine_displacement_state"] = "not_applicable"
    crosswalk = r.read_json(v2.ASSETS / "identity_crosswalk.v2.json")["mappings"]
    profiles = v2.resolve_families(profiles, None, crosswalk)
    keys = pl.from_pandas(profiles[groups + ["profile_id"]])
    membership = data.join(keys, on=groups, how="left", nulls_equal=True, validate="m:1").select("vin", "profile_id")
    if membership["profile_id"].null_count() or membership["vin"].n_unique() != frame.height or int(profiles.vin_count.sum()) != frame.height:
        raise ValueError("VIN/profile reconciliation failed: " + grain)
    return profiles, membership
