"""Train reproducible advertised-price models for all vehicle listings.

The two feature sets are listing-only ``base`` and NHTSA-enriched ``enriched``.
The listing-only and NHTSA-enriched feature sets use matching samples and
splits. Preprocessing is fitted on training rows only.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timezone
import json
import hashlib
import importlib.metadata
import sys
import warnings
from pathlib import Path
from typing import Any

import polars as pl

from EDA.DATA_CLEANING import (
    _load_cohort_exclusion_vins, clean_used_cars_batch, load_final_data, scan_final_table,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
class PricePreprocessor(TransformerMixin, BaseEstimator):
    """Keep unknown categories and numeric missingness explicit at inference.

    Numeric medians and categorical vocabularies are fitted on training rows
    only. Missing indicators exist for every numeric field, even when its
    training values are complete. Source category strings are prefixed so
    literal source values cannot collide with the reserved missing tokens.
    """

    def __init__(self, categorical_columns, numeric_columns):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns

    @staticmethod
    def _category(series):
        values = series.astype("string").str.strip()
        return ("value:" + values).where(values.notna() & values.ne(""), "__MISSING__")

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer_.fit(X[self.numeric_columns].replace([np.inf, -np.inf], np.nan))
        self.categories_ = {}
        for column in self.categorical_columns:
            observed = set(self._category(X[column]).tolist())
            self.categories_[column] = sorted(observed | {"__MISSING__", "__UNKNOWN__"})
        self.output_columns_ = list(X.columns) + [f"{c}__missing" for c in self.numeric_columns]
        return self

    def transform(self, X):
        check_is_fitted(self, "categories_")
        absent = set(self.feature_names_in_) - set(X.columns)
        if absent:
            raise ValueError(f"Missing input features: {sorted(absent)}")
        result = X.loc[:, self.feature_names_in_].copy()
        numbers = result[self.numeric_columns].replace([np.inf, -np.inf], np.nan)
        result[self.numeric_columns] = self.imputer_.transform(numbers)
        indicators = numbers.isna().astype("float64")
        indicators.columns = [f"{c}__missing" for c in self.numeric_columns]
        result = pd.concat([result, indicators], axis=1)
        for column in self.categorical_columns:
            values = self._category(result[column])
            vocabulary = self.categories_[column]
            values = values.where(values.isin(vocabulary), "__UNKNOWN__")
            result[column] = pd.Categorical(values, categories=vocabulary)
        return result.loc[:, self.output_columns_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)


class ExpLogPricePredictor(RegressorMixin, BaseEstimator):
    """Expose dollar predictions from an estimator trained on ln(price)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.estimator.fit(X, np.log(np.asarray(y, dtype=float)), **fit_params)
        return self

    def __sklearn_is_fitted__(self):
        if hasattr(self.estimator, "is_fitted"):
            return self.estimator.is_fitted()
        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            return False
        return True

    def predict(self, X):
        return np.exp(self.estimator.predict(X))


# The CLI runs this file as ``__main__``. Give saved joblib predictors a stable
# import path that resolves to this same module in both CLI and notebook use.
sys.modules.setdefault("MODELS.Price_ML_Models", sys.modules[__name__])
PricePreprocessor.__module__ = "MODELS.Price_ML_Models"
ExpLogPricePredictor.__module__ = "MODELS.Price_ML_Models"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_fingerprint(manifest):
    columns = ["vin", "partition", "split_group"]
    ordered = manifest[columns].sort_values("vin")
    digest = hashlib.sha256()
    for start in range(0, len(ordered), 100_000):
        digest.update(ordered.iloc[start:start + 100_000].to_csv(index=False, header=False).encode())
    return digest.hexdigest()


PRICE_WEIGHTING_MODES = ("none", "high_price")
PRICE_WEIGHT_REFERENCE = 50_000.0
PRICE_WEIGHT_CAP = 5.0
PRICE_BAND_WIDTH = 25_000
HIGH_PRICE_THRESHOLD = 100_000
FOCUS_MAKES = ("TOYOTA", "FERRARI", "LAMBORGHINI")


def evaluation_dimensions(predictions):
    """Left-closed $25k bands with no upper cap, plus an inclusive $100k+ cut."""
    data = predictions.copy()
    prices = data["actual_price"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("Evaluation prices must be finite and nonnegative")
    lower = (prices // PRICE_BAND_WIDTH).astype(np.int64) * PRICE_BAND_WIDTH
    data["price_band_lower_usd"] = lower
    data["price_band_upper_usd"] = lower + PRICE_BAND_WIDTH
    data["price_band"] = [f"{lo // 1000}k-{(lo + PRICE_BAND_WIDTH) // 1000}k" for lo in lower]
    data["price_tier"] = np.where(prices >= HIGH_PRICE_THRESHOLD, "100k+", "below100k")
    return data


def segment_summary(part):
    family_column = "family_block" if "family_block" in part else "split_group"
    family_count = int(part[family_column].nunique(dropna=False))
    return {"rows": len(part), "family_blocks": family_count,
            "median_actual_price": float(part["actual_price"].median()),
            **regression_metrics(part["actual_price"], part["prediction"]),
            "support_status": "supported" if len(part) >= 100 and family_count >= 5 else "low_support"}


def training_price_weights(prices, mode="none"):
    """Weight training labels only; evaluation always retains ordinary rows."""
    if mode not in PRICE_WEIGHTING_MODES:
        raise ValueError(f"Unknown price weighting: {mode}")
    if mode == "none":
        return None
    values = np.asarray(prices, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Training prices must be finite and positive")
    return np.clip(values / PRICE_WEIGHT_REFERENCE, 1.0, PRICE_WEIGHT_CAP)


def regression_metrics(actual, predicted):
    """Report unweighted errors and R² on the original dollar scale."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    errors = predicted - actual
    absolute = np.abs(errors)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 and np.ptp(actual) > 0 else None,
        "wape_pct": float(absolute.sum() / actual.sum() * 100) if actual.sum() > 0 else None,
        "mean_signed_error": float(errors.mean()),
        "p95_absolute_error": float(np.quantile(absolute, .95)),
    }


def validation_metrics(actual, predicted):
    return {f"validation_{key}": value for key, value in regression_metrics(actual, predicted).items()}


def prediction_metrics(actual, predicted, groups):
    error = np.abs(np.asarray(actual) - np.asarray(predicted))
    grouped = pd.DataFrame({"group": groups, "error": error}).groupby("group")["error"].agg(["sum", "count"])
    # Resample whole split blocks, preserving within-block dependence.
    rng = np.random.default_rng(42)
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    bootstrap = []
    if len(grouped) >= 2:
        for _ in range(200):
            sampled = rng.integers(0, len(grouped), len(grouped))
            bootstrap.append(sums[sampled].sum() / counts[sampled].sum())
    return {
        **regression_metrics(actual, predicted),
        "median_absolute_error": float(np.median(error)),
        "p90_absolute_error": float(np.quantile(error, .9)),
        "p95_absolute_error": float(np.quantile(error, .95)),
        "mean_signed_error": float(np.mean(np.asarray(predicted) - np.asarray(actual))),
        "negative_prediction_count": int((np.asarray(predicted) < 0).sum()),
        "macro_split_group_mae": float((sums / counts).mean()),
        "mae_group_bootstrap_low": float(np.quantile(bootstrap, .025)) if bootstrap else None,
        "mae_group_bootstrap_high": float(np.quantile(bootstrap, .975)) if bootstrap else None,
    }


def segment_metrics(predictions, model_name):
    data = evaluation_dimensions(predictions)
    rows = []
    for dimension in ("is_new", "cohort_complete", "split_identity_source", "price_band",
                      "price_tier", "mileage_review_flag", "price_review_flag", "make_name", "Make"):
        if dimension not in data.columns:
            continue
        for value, part in data.groupby(dimension, observed=True, dropna=False):
            rows.append({"model": model_name, "dimension": dimension, "segment": str(value),
                         **segment_summary(part),
                         **({"price_band_lower_usd": int(part.price_band_lower_usd.iloc[0]),
                             "price_band_upper_usd": int(part.price_band_upper_usd.iloc[0])}
                            if dimension == "price_band" else {})})
    return rows


def make_price_metrics(predictions, model_name, *, focus_only=False, makes=None):
    """Keep decoded Make and price band separate; expose absent spotlight makes."""
    data = evaluation_dimensions(predictions)
    rows = []
    report_empty = focus_only or makes is not None
    makes = FOCUS_MAKES if focus_only else data["Make"].dropna().unique() if makes is None else makes
    empty = {"rows": 0, "family_blocks": 0, "median_actual_price": None,
             **{k: None for k in regression_metrics([1], [1])}, "support_status": "no_rows"}
    for make in makes:
        part = data[data.Make == make]
        for tier in ("all", "below100k", "100k+"):
            selected = part if tier == "all" else part[part.price_tier == tier]
            if len(selected) or report_empty:
                rows.append({"model": model_name, "Make": make, "dimension": "price_tier", "price_band": None,
                             "price_tier": tier, **(segment_summary(selected) if len(selected) else empty)})
        for lower, selected in part.groupby("price_band_lower_usd", sort=True):
            rows.append({"model": model_name, "Make": make, "dimension": "price_band",
                         "price_band": selected.price_band.iloc[0], "price_tier": selected.price_tier.iloc[0],
                         "price_band_lower_usd": int(lower), "price_band_upper_usd": int(lower + PRICE_BAND_WIDTH),
                         **segment_summary(selected)})
    return rows


def year_make_metrics(predictions, model_name):
    """Score held-out rows by decoded year, make, and year/make with support."""
    data = predictions.copy()
    data["absolute_error"] = (data["prediction"] - data["actual_price"]).abs()
    data["signed_error"] = data["prediction"] - data["actual_price"]
    dimensions = (
        ("year", ["ModelYear"]),
        ("make", ["Make"]),
        ("year_make", ["ModelYear", "Make"]),
    )
    rows = []
    for level, columns in dimensions:
        for key, part in data.groupby(columns, observed=True, dropna=False, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            row = {"model": model_name, "metric_level": level,
                   "ModelYear": None, "Make": None}
            row.update(dict(zip(columns, values)))
            mae = float(part["absolute_error"].mean())
            median_price = float(part["actual_price"].median())
            family_count = int(part["family_block"].nunique(dropna=False))
            row.update({
                "r2": regression_metrics(part["actual_price"], part["prediction"])["r2"],
                "rows": len(part),
                "family_blocks": family_count,
                "median_actual_price": median_price,
                "mae_usd": mae,
                "rmse_usd": float(np.sqrt((part["signed_error"] ** 2).mean())),
                "mean_signed_error_usd": float(part["signed_error"].mean()),
                "p90_absolute_error_usd": float(part["absolute_error"].quantile(.90)),
                "wape_pct": float(part["absolute_error"].sum() / part["actual_price"].sum() * 100),
                "mae_pct_of_median_price": float(mae / median_price * 100) if median_price > 0 else None,
                "support_status": "supported" if len(part) >= 100 and family_count >= 5 else "low_support",
            })
            rows.append(row)
    return rows


def compatible_run_metadata(left, right, *, allow_feature_set=False, allow_price_weighting=False):
    """Old runs without content/split/code identities cannot be auto-combined."""
    required = ("dataset_fingerprint", "split_fingerprint", "code_fingerprint",
                "preprocessing_version", "package_versions", "tuning_rows", "max_trees",
                "split_strategy")
    if not all(left.get(key) is not None and left.get(key) == right.get(key) for key in required):
        return False
    if not allow_price_weighting and left.get("price_weighting", "none") != right.get("price_weighting", "none"):
        return False
    if allow_feature_set:
        return left.get("model_type") in MODEL_FEATURE_SETS and right.get("model_type") in MODEL_FEATURE_SETS
    return left.get("model_type") is not None and left.get("model_type") == right.get("model_type")



PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_OUTPUT_DIR = PROJECT_ROOT / "MODEL_OUTPUT"

RANDOM_SEED = 42
SEARCH_ITERATIONS = 10
SEARCH_N_JOBS = 1  # Keep parallel model copies from increasing peak memory.

TARGET_COLUMN = "price"
COHORT_COLUMNS = ("ModelYear", "Make", "Model", "Trim")
PREPROCESSING_VERSION = "advertised-price-log-v3"
OPTION_LABELS = {
    "option_navigation": "Navigation System", "option_leather": "Leather Seats",
    "option_heated_seats": "Heated Seats", "option_backup_camera": "Backup Camera",
    "option_bluetooth": "Bluetooth", "option_sunroof": "Sunroof/Moonroof",
    "option_remote_start": "Remote Start", "option_alloy_wheels": "Alloy Wheels",
    "option_adaptive_cruise": "Adaptive Cruise Control",
}
DERIVED_FEATURE_COLUMNS = ("maximum_seating_numeric", "log1p_mileage", *OPTION_LABELS)

# Listing features from used_cars plus deterministic listing-derived features.
BASE_FEATURE_COLUMNS = (
    "engine_displacement", "engine_type", "exterior_color", "fuel_type",
    "horsepower", "interior_color", "is_new", "listing_color", "make_name",
    "mileage", "model_name", "owner_count", "transmission", "trim_name",
    "wheel_system", "wheelbase", "width", "year",
    "has_accidents", "frame_damaged", "salvage", "fleet", "theft_title",
    "latitude", "longitude", "body_type", "city_fuel_economy", "highway_fuel_economy",
) + DERIVED_FEATURE_COLUMNS

# General NHTSA vPIC fields requested in addition to the listing baseline.
# Includes general, truck, and battery/electrification fields in ``enriched``.
VPIC_FEATURE_COLUMNS = (
    "ABS", "AirBagLocCurtain", "AirBagLocFront", "AirBagLocSide",
    "AutoReverseSystem", "Axles", "BlindSpotMon", "BodyClass",
    "BrakeSystemType", "CoolingType", "DaytimeRunningLight", "DisplacementL",
    "Doors", "DynamicBrakeSupport", "ESC", "EngineConfiguration", "EngineCycles",
    "EngineCylinders", "EngineManufacturer", "FuelTypePrimary",
    "ForwardCollisionWarning", "FuelInjectionType", "FuelTypeSecondary",
    "KeylessIgnition", "LaneCenteringAssistance", "LaneDepartureWarning",
    "LaneKeepSystem", "ParkAssist", "PedestrianAutomaticEmergencyBraking",
    "RearCrossTrafficAlert", "RearVisibilitySystem", "SeatRows", "Seats",
    "TPMS", "TopSpeedMPH", "TransmissionSpeeds", "Turbo", "WheelSizeFront",
    "WheelSizeRear", "Wheels", "Windows",
)

COMPLAINT_VALUE_FEATURES = (
    "complaint_count", "crash_report_count", "fire_report_count",
    "injury_report_count", "death_report_count", "reported_injuries",
    "reported_deaths", "component_count",
)
RECALL_VALUE_FEATURES = ("recall_campaign_count",)
COMPLAINT_COVERAGE_FEATURES = ("complaint_source_records_present",)
RECALL_COVERAGE_FEATURES = ("recall_source_records_present",)
TRUCK_FEATURE_COLUMNS = ("BedType", "BodyCabType", "BedLengthIN")
EV_HEV_FEATURE_COLUMNS = (
    "BatteryKWh", "BatteryType", "BatteryPacks", "BatteryV", "EVDriveUnit",
    "ElectrificationLevel", "BatteryInfo", "ChargerLevel", "ChargerPowerKW",
    "EngineKW",
)

SAFETY_FEATURE_COLUMNS = (
    "safety_variant_count", "safety_overall_rating_mean", "safety_front_rating_mean",
    "safety_side_rating_mean", "safety_rollover_rating_mean", "safety_rollover_probability_mean",
    "safety_complaints_count", "safety_recalls_count", "safety_investigations_count",
)
COMPLAINT_RECALL_FEATURE_COLUMNS = set(
    COMPLAINT_VALUE_FEATURES + RECALL_VALUE_FEATURES
    + COMPLAINT_COVERAGE_FEATURES + RECALL_COVERAGE_FEATURES + SAFETY_FEATURE_COLUMNS
)
ENRICHED_FEATURE_COLUMNS = tuple(dict.fromkeys(
    BASE_FEATURE_COLUMNS + VPIC_FEATURE_COLUMNS + TRUCK_FEATURE_COLUMNS
    + EV_HEV_FEATURE_COLUMNS + COMPLAINT_VALUE_FEATURES + RECALL_VALUE_FEATURES
    + COMPLAINT_COVERAGE_FEATURES + RECALL_COVERAGE_FEATURES + SAFETY_FEATURE_COLUMNS
))
MODEL_FEATURE_SETS = {
    "base": BASE_FEATURE_COLUMNS,
    "enriched": ENRICHED_FEATURE_COLUMNS,
}


def add_vehicle_subtype_flags(frame: pl.DataFrame) -> pl.DataFrame:
    """Add independent truck and EV/HEV cohort flags from decoded vPIC fields.

    EV/HEV evidence is an explicit decoded hybrid/BEV/PHEV/FCEV level,
    an EV-like ``VehicleType``, or a positive ``BatteryKWh``.
    The flags are independent, so an electric truck is in both cohorts.
    """
    required = {"VehicleType", "ElectrificationLevel", "BatteryKWh"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Missing vPIC subtype columns: " + ", ".join(sorted(missing)))

    vehicle_type = pl.col("VehicleType").cast(pl.String).str.strip_chars().str.to_uppercase()
    electrification = (
        pl.col("ElectrificationLevel").cast(pl.String)
        .str.strip_chars().str.to_uppercase()
    )
    has_electrification = (
        electrification.is_in([
            "STRONG HEV (HYBRID ELECTRIC VEHICLE)", "MILD HEV (HYBRID ELECTRIC VEHICLE)",
            "HEV (HYBRID ELECTRIC VEHICLE) - LEVEL UNKNOWN", "BEV (BATTERY ELECTRIC VEHICLE)",
            "PHEV (PLUG-IN HYBRID ELECTRIC VEHICLE)", "FCEV (FUEL CELL ELECTRIC VEHICLE)",
        ])
    )
    is_ev_vehicle_type = vehicle_type.is_in(["EV", "ELECTRICVEHICLE", "ELECTRIC VEHICLE"])
    return frame.with_columns(
        (vehicle_type == "TRUCK").fill_null(False).alias("is_truck"),
        (
            has_electrification | is_ev_vehicle_type
            | (pl.col("BatteryKWh") > 0)
        ).fill_null(False).alias("is_ev_hev"),
    )


def filter_vehicle_subtype(frame: pl.DataFrame, subtype: str) -> pl.DataFrame:
    """Filter an already listing-to-vPIC-joined frame to truck or EV/HEV rows."""
    if subtype not in {"truck", "ev_hev"}:
        raise ValueError("subtype must be 'truck' or 'ev_hev'")
    flagged = add_vehicle_subtype_flags(frame)
    flag_column = "is_truck" if subtype == "truck" else "is_ev_hev"
    return flagged.filter(pl.col(flag_column))


COMPLAINT_RECALL_AS_OF = date(2020, 12, 31)
COMPLAINT_GROUP_KEYS = ("model_year", "make", "model")


def _normalized_nhtsa_keys(frame: pl.DataFrame, *, year_column: str = "model_year") -> pl.DataFrame:
    """Normalize only case and whitespace for exact NHTSA cohort joins."""
    return frame.with_columns(
        pl.col(year_column).cast(pl.Int64, strict=False).alias("_nhtsa_year"),
        pl.col("make").cast(pl.String).str.strip_chars().str.to_uppercase()
            .str.replace_all(r"\s+", " ").alias("_nhtsa_make"),
        pl.col("model").cast(pl.String).str.strip_chars().str.to_uppercase()
            .str.replace_all(r"\s+", " ").alias("_nhtsa_model"),
    )


def _complaint_product_matches(record: dict[str, Any]) -> bool:
    """Require exact cohort agreement when explicit vehicle products exist."""
    products = record.get("products")
    if isinstance(products, str):
        try:
            products = json.loads(products)
        except (TypeError, json.JSONDecodeError):
            return False if products else True
    if not products:
        return True
    if not isinstance(products, list):
        return False
    vehicles = [item for item in products if isinstance(item, dict)
                and str(item.get("type", "")).strip().upper() == "VEHICLE"]
    if not vehicles:
        return False
    year, make, model = record.get("model_year"), record.get("make"), record.get("model")
    make, model = str(make or "").strip().upper(), str(model or "").strip().upper()
    make, model = " ".join(make.split()), " ".join(model.split())
    return any(str(item.get("productYear", "")).strip() == str(year)
               and " ".join(str(item.get("productMake", "")).strip().upper().split()) == make
               and " ".join(str(item.get("productModel", "")).strip().upper().split()) == model
               for item in vehicles)


def build_complaint_recall_features(
    as_of: date = COMPLAINT_RECALL_AS_OF,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Aggregate documented NHTSA Parquet sources in memory; write no sidecars.

    A missing source cohort is represented by a zero-valued coverage flag and
    null counts. The Parquet snapshot has no query ledger, so absence cannot be
    called a confirmed zero or distinguished as successful-empty/unattempted.
    """
    complaints = scan_final_table("nhtsa_complaints").select(
        "model_year", "make", "model", "odiNumber", "dateComplaintFiled",
        "crash", "fire", "numberOfInjuries", "numberOfDeaths", "components",
        "summary", "products",
    ).collect()
    recalls = scan_final_table("nhtsa_recalls").select(
        "model_year", "make", "model", "NHTSACampaignNumber",
        "ReportReceivedDate", "Component", "Consequence", "Remedy",
    ).collect()
    safety = scan_final_table("nhtsa_safety_rating_values").select(
        pl.col("ModelYear").alias("model_year"), pl.col("Make").alias("make"),
        pl.col("Model").alias("model"), "OverallRating", "OverallFrontCrashRating",
        "OverallSideCrashRating", "RolloverRating", "RolloverPossibility",
        "ComplaintsCount", "RecallsCount", "InvestigationCount",
    ).collect()

    complaint_presence = (_normalized_nhtsa_keys(complaints)
        .group_by("_nhtsa_year", "_nhtsa_make", "_nhtsa_model")
        .agg(pl.lit(1, dtype=pl.Int8).alias("complaint_source_records_present")))
    recalls_presence = (_normalized_nhtsa_keys(recalls)
        .group_by("_nhtsa_year", "_nhtsa_make", "_nhtsa_model")
        .agg(pl.lit(1, dtype=pl.Int8).alias("recall_source_records_present")))

    eligible_complaints = _normalized_nhtsa_keys(complaints).filter(
        pl.col("dateComplaintFiled").is_not_null()
        & (pl.col("dateComplaintFiled") <= pl.lit(as_of))
        & pl.col("odiNumber").is_not_null()
        & pl.struct("products", "model_year", "make", "model")
            .map_elements(_complaint_product_matches, return_dtype=pl.Boolean)
    )
    complaint_ids = eligible_complaints.group_by(
        "_nhtsa_year", "_nhtsa_make", "_nhtsa_model", "odiNumber"
    ).agg(
        pl.col("crash").first(), pl.col("fire").first(),
        pl.col("numberOfInjuries").first(), pl.col("numberOfDeaths").first(),
        pl.col("components").first(),
        pl.any_horizontal([
            pl.col(column).n_unique() > 1
            for column in ("summary", "crash", "fire", "numberOfInjuries", "numberOfDeaths", "components")
        ]).alias("_conflicting_version"),
    )
    complaint_features = complaint_ids.group_by(
        "_nhtsa_year", "_nhtsa_make", "_nhtsa_model"
    ).agg(
        pl.len().alias("complaint_count"),
        pl.when(~pl.col("_conflicting_version"))
            .then(pl.col("crash").fill_null(False).cast(pl.Int64)).otherwise(0)
            .sum().alias("crash_report_count"),
        pl.when(~pl.col("_conflicting_version"))
            .then(pl.col("fire").fill_null(False).cast(pl.Int64)).otherwise(0)
            .sum().alias("fire_report_count"),
        pl.when(~pl.col("_conflicting_version") & (pl.col("numberOfInjuries").fill_null(0) > 0))
            .then(1).otherwise(0).sum().alias("injury_report_count"),
        pl.when(~pl.col("_conflicting_version") & (pl.col("numberOfDeaths").fill_null(0) > 0))
            .then(1).otherwise(0).sum().alias("death_report_count"),
        pl.when(~pl.col("_conflicting_version"))
            .then(pl.col("numberOfInjuries").fill_null(0)).otherwise(0)
            .sum().alias("reported_injuries"),
        pl.when(~pl.col("_conflicting_version"))
            .then(pl.col("numberOfDeaths").fill_null(0)).otherwise(0)
            .sum().alias("reported_deaths"),
    )
    component_counts = (complaint_ids.filter(~pl.col("_conflicting_version"))
        .select("_nhtsa_year", "_nhtsa_make", "_nhtsa_model", "components")
        .with_columns(pl.col("components").fill_null("").str.split(","))
        .explode("components").with_columns(pl.col("components").str.strip_chars())
        .filter(pl.col("components") != "")
        .group_by("_nhtsa_year", "_nhtsa_make", "_nhtsa_model")
        .agg(pl.col("components").n_unique().alias("component_count")))
    complaint_features = complaint_features.join(component_counts,
        on=["_nhtsa_year", "_nhtsa_make", "_nhtsa_model"], how="left", validate="1:1")
    if "component_count" in complaint_features.columns:
        complaint_features = complaint_features.with_columns(pl.col("component_count").fill_null(0))
    else:
        complaint_features = complaint_features.with_columns(pl.lit(0).alias("component_count"))

    eligible_recalls = _normalized_nhtsa_keys(recalls).filter(
        pl.col("ReportReceivedDate").is_not_null()
        & (pl.col("ReportReceivedDate") <= pl.lit(as_of))
        & pl.col("NHTSACampaignNumber").is_not_null()
        & (pl.col("NHTSACampaignNumber").str.strip_chars() != "")
    )
    distinct_campaigns = eligible_recalls.unique(
        subset=["_nhtsa_year", "_nhtsa_make", "_nhtsa_model", "NHTSACampaignNumber"],
        maintain_order=True,
    )
    recall_features = distinct_campaigns.group_by(
        "_nhtsa_year", "_nhtsa_make", "_nhtsa_model"
    ).agg(pl.len().alias("recall_campaign_count"))

    safety_features = (_normalized_nhtsa_keys(safety)
        .group_by("_nhtsa_year", "_nhtsa_make", "_nhtsa_model")
        .agg(
            pl.len().alias("safety_variant_count"),
            pl.col("OverallRating").cast(pl.Float64, strict=False).mean().alias("safety_overall_rating_mean"),
            pl.col("OverallFrontCrashRating").cast(pl.Float64, strict=False).mean().alias("safety_front_rating_mean"),
            pl.col("OverallSideCrashRating").cast(pl.Float64, strict=False).mean().alias("safety_side_rating_mean"),
            pl.col("RolloverRating").cast(pl.Float64, strict=False).mean().alias("safety_rollover_rating_mean"),
            pl.col("RolloverPossibility").cast(pl.Float64, strict=False).mean().alias("safety_rollover_probability_mean"),
            pl.col("ComplaintsCount").cast(pl.Float64, strict=False).mean().alias("safety_complaints_count"),
            pl.col("RecallsCount").cast(pl.Float64, strict=False).mean().alias("safety_recalls_count"),
            pl.col("InvestigationCount").cast(pl.Float64, strict=False).mean().alias("safety_investigations_count"),
        ))

    features = (complaint_presence.join(recalls_presence,
            on=["_nhtsa_year", "_nhtsa_make", "_nhtsa_model"], how="full", coalesce=True)
        .join(complaint_features, on=["_nhtsa_year", "_nhtsa_make", "_nhtsa_model"], how="left", validate="1:1")
        .join(recall_features, on=["_nhtsa_year", "_nhtsa_make", "_nhtsa_model"], how="left", validate="1:1")
        .join(safety_features, on=["_nhtsa_year", "_nhtsa_make", "_nhtsa_model"], how="full", coalesce=True)
        .rename({"_nhtsa_year": "ModelYear", "_nhtsa_make": "Make", "_nhtsa_model": "Model"}))
    stats = {
        "as_of_date": as_of.isoformat(),
        "price_observation_date_verified": False,
        "interpretation": "Retrospective associations; filing/receipt cutoff does not establish listing-price observation time.",
        "complaint_source_rows": complaints.height,
        "recall_source_rows": recalls.height,
        "safety_source_rows": safety.height,
        "complaint_rows_after_cutoff": eligible_complaints.height,
        "distinct_eligible_complaint_cohort_ids": complaint_ids.height,
        "conflicting_complaint_cohort_ids": complaint_ids.filter(pl.col("_conflicting_version")).height,
        "recall_rows_after_cutoff": eligible_recalls.height,
        "distinct_eligible_cohort_campaigns": distinct_campaigns.height,
        "complaint_source_cohorts": complaint_presence.height,
        "recall_source_cohorts": recalls_presence.height,
        "absence_semantics": "No saved records is unknown coverage, not a confirmed zero; source Parquet has no query ledger.",
    }
    return features, stats


def select_model_columns(
    frame: pl.DataFrame,
    model_type: str = "base",
    *,
    require_all: bool = True,
) -> tuple[pl.DataFrame, pl.Series]:
    """Return selected predictors and ``price`` after checking the schema.

    Set ``require_all=False`` to return the intersection with currently
    available predictors while retaining the full feature-set constants.
    Missing predictors are reported rather than silently dropped by default.
    """
    if model_type not in MODEL_FEATURE_SETS:
        choices = ", ".join(MODEL_FEATURE_SETS)
        raise ValueError(f"Unknown model_type {model_type!r}; choose from: {choices}.")
    required = (TARGET_COLUMN,) + MODEL_FEATURE_SETS[model_type]
    missing = [column for column in required if column not in frame.columns]
    if TARGET_COLUMN in missing:
        raise ValueError(f"Target column {TARGET_COLUMN!r} is missing from the input.")
    missing_features = [column for column in missing if column != TARGET_COLUMN]
    if missing_features and require_all:
        raise ValueError("Missing requested predictors: " + ", ".join(missing_features))
    available = [column for column in MODEL_FEATURE_SETS[model_type] if column in frame.columns]
    return frame.select(available), frame.get_column(TARGET_COLUMN)

TREE_MODEL_NAMES = ("catboost", "lightgbm", "xgboost")
CLASSICAL_MODEL_NAMES = (
    "random_forest", "multiple_linear_regression", "lasso", "ridge", "elastic_net",
)
PILOT_MODEL_NAMES = TREE_MODEL_NAMES + CLASSICAL_MODEL_NAMES

# Full-run CLI search ranges retained after the 50,000-listing base pilot.
# Compare model complexity at fixed learning rate and regularization first,
# then refine the best structure without a full Cartesian search.
TREE_PARAMETER_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "catboost": [
        {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
        {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
        {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
        {"depth": 10, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
    ],
    "lightgbm": [
        {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 200},
        {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 200},
        {"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 200},
        {"num_leaves": 255, "learning_rate": 0.05, "min_child_samples": 200},
    ],
    "xgboost": [
        {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 1,
         "subsample": 0.8, "colsample_bytree": 0.8,
         "objective": "reg:squarederror"},
        {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1,
         "subsample": 0.8, "colsample_bytree": 0.8,
         "objective": "reg:squarederror"},
        {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 1,
         "subsample": 0.8, "colsample_bytree": 0.8,
         "objective": "reg:squarederror"},
        {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 1,
         "subsample": 0.8, "colsample_bytree": 0.8,
         "objective": "reg:squarederror"},
    ],
}
TREE_REFINEMENT_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "catboost": [
        {"learning_rate": 0.08}, {"l2_leaf_reg": 3.0},
        {"l2_leaf_reg": 20.0},
    ],
    "lightgbm": [
        {"learning_rate": 0.08}, {"min_child_samples": 100},
        {"min_child_samples": 400},
    ],
    "xgboost": [
        {"learning_rate": 0.08}, {"min_child_weight": 10},
        {"min_child_weight": 25}, {"objective": "reg:absoluteerror"},
    ],
}

CLASSICAL_PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "model__n_estimators": [200],
        "model__max_features": [0.3, 0.5, 0.7],
        "model__min_samples_leaf": [2, 3, 5],
    },
    "multiple_linear_regression": {},
    "lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
    "elastic_net": {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "model__l1_ratio": [0.5, 0.9, 0.95],
    },
}
# Used only by the older generic tune_models() helper, not the CLI search above.
TREE_PARAMETER_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "catboost": {"model__depth": [6, 8, 10],
                 "model__learning_rate": [0.03, 0.05],
                 "model__l2_leaf_reg": [3.0, 5.0, 10.0]},
    "lightgbm": {"model__num_leaves": [15, 31, 63],
                 "model__learning_rate": [0.03, 0.05],
                 "model__min_child_samples": [50, 100, 200]},
    "xgboost": {"model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05],
                "model__min_child_weight": [1, 10, 25],
                "model__subsample": [0.8, 0.9],
                "model__colsample_bytree": [0.7, 0.8]},
}
PARAMETER_DISTRIBUTIONS: dict[str, Any] = {
    **TREE_PARAMETER_DISTRIBUTIONS,
    **CLASSICAL_PARAMETER_GRIDS,
}


def iter_cleaned_batches() -> Iterator[tuple[str, pl.DataFrame]]:
    """Yield saved CAR_DATA_FINAL tables as Polars batches without re-cleaning."""
    yield from load_final_data()


def prepare_training_data(
    batches: Iterator[tuple[str, pl.DataFrame]],
) -> tuple[Any, Any, Any, Any]:
    """TODO: define target, features, joins, sampling, and CV groups after EDA.

    Return ``X, y, preprocessor, cv``. Choose a bounded sample or a suitable
    out-of-core strategy after measuring the real data shape and memory needs.
    The CV splitter should match the eventual row grain and leakage risks.
    """
    del batches
    raise NotImplementedError(
        "Complete prepare_training_data() after EDA determines the target, "
        "feature set, joins, sampling, preprocessing, and validation strategy."
    )


def build_estimators() -> dict[str, Any]:
    """Create the seven requested estimators with conservative parallelism."""
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from xgboost import XGBRegressor

    return {
        "lightgbm": LGBMRegressor(n_jobs=1, random_state=RANDOM_SEED),
        "xgboost": XGBRegressor(
            n_jobs=1, random_state=RANDOM_SEED, tree_method="hist",
            objective="reg:squarederror",
        ),
        "random_forest": RandomForestRegressor(n_jobs=1, random_state=RANDOM_SEED),
        "multiple_linear_regression": LinearRegression(n_jobs=1),
        "lasso": Lasso(max_iter=10_000, random_state=RANDOM_SEED),
        "ridge": Ridge(),
        "elastic_net": ElasticNet(max_iter=10_000, random_state=RANDOM_SEED),
    }


def build_searches(
    preprocessor: Any,
    cv: Any,
    scoring: Any,
    model_names: list[str] | None = None,
    iterations: int = SEARCH_ITERATIONS,
) -> dict[str, Any]:
    """Build one randomized search per model once preprocessing and CV are set."""
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline

    estimators = build_estimators()
    selected_names = model_names or list(estimators)
    unknown = set(selected_names) - set(estimators)
    if unknown:
        raise ValueError("Unknown model names: " + ", ".join(sorted(unknown)))
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    searches = {}
    for name in selected_names:
        distributions = PARAMETER_DISTRIBUTIONS[name]
        if not distributions:
            raise NotImplementedError(
                f"Fill PARAMETER_DISTRIBUTIONS[{name!r}] after EDA before tuning."
            )
        pipeline = Pipeline(
            [("preprocess", preprocessor), ("model", estimators[name])]
        )
        searches[name] = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=distributions,
            n_iter=iterations,
            scoring=scoring,
            cv=cv,
            n_jobs=SEARCH_N_JOBS,
            pre_dispatch=SEARCH_N_JOBS,
            random_state=RANDOM_SEED,
            return_train_score=False,
            refit=True,
        )
    return searches


def tune_models(
    X: Any,
    y: Any,
    preprocessor: Any,
    cv: Any,
    scoring: Any,
    groups: Any = None,
    model_names: list[str] | None = None,
    iterations: int = SEARCH_ITERATIONS,
    model_type: str = "base",
) -> dict[str, Any]:
    """Fit configured searches and save reusable models plus run reports.

    Each call writes to a unique UTC-stamped directory under
    ``MODEL_OUTPUT/<model_type>/``. The directory contains a joblib pipeline,
    per-model CV results, and metadata describing features and best parameters.
    """
    if model_type not in MODEL_FEATURE_SETS:
        choices = ", ".join(MODEL_FEATURE_SETS)
        raise ValueError(f"Unknown model_type {model_type!r}; choose from: {choices}.")
    searches = build_searches(preprocessor, cv, scoring, model_names, iterations)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = MODEL_OUTPUT_DIR / model_type / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    fitted = {}
    run_metadata: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "target_column": TARGET_COLUMN,
        "feature_columns": list(MODEL_FEATURE_SETS[model_type]),
        "scoring": scoring if isinstance(scoring, str) else repr(scoring),
        "iterations": iterations,
        "models": {},
    }
    for name, search in searches.items():
        print(f"Tuning {name}...")
        if groups is None:
            search.fit(X, y)
        else:
            search.fit(X, y, groups=groups)
        fitted[name] = search
        print(f"  best CV score: {search.best_score_:,.4f}")
        print(f"  parameters: {search.best_params_}")
        model_dir = run_dir / name
        model_dir.mkdir()

        import joblib

        joblib.dump(search.best_estimator_, model_dir / "model.joblib", compress=3)
        import pandas as pd

        pd.DataFrame(search.cv_results_).to_csv(
            model_dir / "cv_results.csv", index=False
        )
        run_metadata["models"][name] = {
            "best_cv_score": float(search.best_score_),
            "best_parameters": search.best_params_,
            "artifact": str((model_dir / "model.joblib").relative_to(run_dir)),
            "cv_results": str((model_dir / "cv_results.csv").relative_to(run_dir)),
        }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved models and run details to: {run_dir}")
    return fitted


SUBTYPE_SUPPORT_COLUMNS = ("VehicleType", "ElectrificationLevel", "BatteryKWh")
MISSING_COLOR_LABELS = {
    "", "NONE", "NULL", "UNKNOWN", "N/A", "NA", "NOT APPLICABLE",
}
PILOT_MAX_TREES = 6400
TREE_EARLY_STOPPING_ROUNDS = 120
TREE_THREADS = 4


def _validate_snapshot_columns(model_type: str) -> tuple[list[str], list[str]]:
    """Return selected listing and vPIC columns, failing on stale feature names."""
    listing_schema = scan_final_table("used_cars").collect_schema().names()
    vpic_schema = scan_final_table("nhtsa_vpic_decodes").collect_schema().names()
    features = list(MODEL_FEATURE_SETS[model_type])
    listing_features = [column for column in features if column in listing_schema]
    vpic_features = [column for column in features if column in vpic_schema]
    missing = [column for column in features
               if column not in listing_schema and column not in vpic_schema
               and column not in DERIVED_FEATURE_COLUMNS
               and column not in COMPLAINT_RECALL_FEATURE_COLUMNS]
    missing += [c for c in ("major_options", "maximum_seating") if c not in listing_schema]
    if missing:
        raise ValueError(
            "Requested predictors absent from the saved Parquet schemas: "
            + ", ".join(missing)
        )
    missing_cohort = [column for column in COHORT_COLUMNS if column not in vpic_schema]
    if missing_cohort:
        raise ValueError(
            "vPIC cohort columns missing from the saved Parquet schema: "
            + ", ".join(missing_cohort)
        )
    missing_subtype = [column for column in SUBTYPE_SUPPORT_COLUMNS if column not in vpic_schema]
    if missing_subtype:
        raise ValueError(
            "vPIC subtype columns missing from the saved Parquet schema: "
            + ", ".join(missing_subtype)
        )
    return listing_features, vpic_features


def add_cohort_groups(frame: pl.DataFrame) -> pl.DataFrame:
    """Retain decoded dimensions; use conservative year/make/model split blocks.

    Listing identity is a grouping-only fallback, never an imputed decoded
    predictor. Missing trim cannot send members of one known family to different
    partitions. Unresolvable identities share one explicit unknown block.
    """
    expressions = []
    for column in ("Make", "Model", "Trim", "make_name", "model_name"):
        value = pl.col(column).cast(pl.String).str.strip_chars()
        expressions.append(pl.when(value == "").then(None).otherwise(value).alias(column))
    frame = frame.with_columns(expressions)
    complete = pl.all_horizontal([pl.col(c).is_not_null() for c in COHORT_COLUMNS])
    identity_complete = pl.all_horizontal([pl.col(c).is_not_null() for c in COHORT_COLUMNS[:3]])
    # Normalize case/spacing for split identity only. Do not guess model aliases.
    identity = [
        pl.coalesce("ModelYear", "year").cast(pl.String),
        pl.coalesce("Make", "make_name").str.to_uppercase().str.replace_all(r"\s+", " "),
        pl.coalesce("Model", "model_name").str.to_uppercase().str.replace_all(r"\s+", " "),
    ]
    known_identity = pl.all_horizontal([v.is_not_null() for v in identity])
    return frame.with_columns(
        complete.alias("cohort_complete"),
        pl.when(~known_identity).then(pl.lit("unresolved"))
        .when(identity_complete).then(pl.lit("vpic_year_make_model"))
        .otherwise(pl.lit("listing_identity_fallback")).alias("split_identity_source"),
        pl.when(known_identity)
        .then(pl.struct([v.alias(f"key_{i}") for i, v in enumerate(identity)]).hash(seed=RANDOM_SEED))
        .otherwise(pl.lit(0, dtype=pl.UInt64)).alias("_cohort_group"),
    )


def add_listing_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Deterministic extraction; no population statistics or target information."""
    import re

    options = pl.col("major_options").str.strip_chars()
    expressions = [
        pl.col("maximum_seating").str.strip_chars().str.extract(r"^(\d+)\s+seats?$", 1)
        .cast(pl.Float64, strict=False).alias("maximum_seating_numeric"),
        pl.when(pl.col("mileage") >= 0).then(pl.col("mileage").log1p())
        .otherwise(None).alias("log1p_mileage"),
    ]
    for name, label in OPTION_LABELS.items():
        # Match an individual quoted label, not an arbitrary substring.
        expressions.append(
            pl.when(options.is_null() | ~options.str.starts_with("[") | ~options.str.ends_with("]")).then(None)
            .otherwise(options.str.contains("['\"]" + re.escape(label) + "['\"]").cast(pl.Float64))
            .alias(name)
        )
    return frame.with_columns(expressions)


def _normalize_model_feature_strings(frame: pl.DataFrame,
                                     features: list[str]) -> pl.DataFrame:
    """Apply the same blank and color-sentinel rules in training and scoring."""
    string_features = [column for column in features if frame.schema[column] == pl.String]
    expressions = []
    for column in string_features:
        value = pl.col(column).str.strip_chars()
        empty = value.is_null() | (value == "")
        if column in {"exterior_color", "interior_color", "listing_color"}:
            empty = empty | value.str.to_uppercase().is_in(sorted(MISSING_COLOR_LABELS))
        expressions.append(pl.when(empty).then(None).otherwise(value).alias(column))
    return frame.with_columns(expressions) if expressions else frame


def load_model_frame(
    model_type: str, sample_size: int | None, *, issue_as_of: date | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load only requested columns, VIN-join listings/vPIC, and optionally sample."""
    if model_type not in MODEL_FEATURE_SETS:
        raise ValueError(f"Unknown model_type {model_type!r}; choose from: "
                         + ", ".join(MODEL_FEATURE_SETS))
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive or None for the full dataset")

    listing_features, vpic_features = _validate_snapshot_columns(model_type)
    features = list(MODEL_FEATURE_SETS[model_type])
    listing_columns = list(dict.fromkeys([
        "vin", TARGET_COLUMN, *listing_features, "maximum_seating", "major_options",
    ]))
    vpic_columns = list(dict.fromkeys([
        "vin", *vpic_features, *COHORT_COLUMNS, *SUBTYPE_SUPPORT_COLUMNS,
    ]))
    listings = scan_final_table("used_cars").select(listing_columns)
    # Reapply the cleaning eligibility contract so older exports follow the
    # same cohort-ratio and null-mileage rules as a fresh export.
    excluded_vins = _load_cohort_exclusion_vins(None)
    listings = listings.filter(
        pl.col("mileage").is_not_null() & ~pl.col("vin").is_in(list(excluded_vins))
    )
    vpic = (
        scan_final_table("nhtsa_vpic_decodes")
        .select(vpic_columns)
        .with_columns(pl.lit(True).alias("_vpic_match"))
    )

    sampled_listing_count = None
    if sample_size is not None:
        # Hash only the VIN column to choose a stable, uniform sample without
        # materializing the wide listing table in memory first.
        sampled_vins = (
            listings.select("vin")
            .with_columns(pl.col("vin").hash(seed=RANDOM_SEED).alias("_sample_key"))
            .sort("_sample_key")
            .head(sample_size)
            .collect()["vin"].to_list()
        )
        sampled_listing_count = len(sampled_vins)
        listings = listings.filter(pl.col("vin").is_in(sampled_vins))
        vpic = vpic.filter(pl.col("vin").is_in(sampled_vins))

    # Both source tables are expected to have unique VIN rows. Polars validates
    # this at join time instead of silently multiplying listing rows.
    frame = listings.join(vpic, on="vin", how="left", validate="1:1", maintain_order="left").collect()
    joined_rows = frame.height
    unmatched_vpic_rows = frame.filter(pl.col("_vpic_match").is_null()).height
    issue_feature_stats = None
    if model_type == "enriched":
        feature_frame, issue_feature_stats = build_complaint_recall_features(
            as_of=issue_as_of or COMPLAINT_RECALL_AS_OF,
        )
        # Match case and repeated whitespace only. The original vPIC cohort
        # columns remain unchanged and separate in every output.
        frame = frame.with_columns(
            pl.col("ModelYear").cast(pl.Int64, strict=False).alias("_feature_year"),
            pl.col("Make").cast(pl.String).str.strip_chars().str.to_uppercase()
                .str.replace_all(r"\s+", " ").alias("_feature_make"),
            pl.col("Model").cast(pl.String).str.strip_chars().str.to_uppercase()
                .str.replace_all(r"\s+", " ").alias("_feature_model"),
        )
        frame = frame.join(feature_frame,
            left_on=["_feature_year", "_feature_make", "_feature_model"],
            right_on=["ModelYear", "Make", "Model"],
            how="left", validate="m:1", maintain_order="left")
        frame = frame.with_columns(
            pl.col("complaint_source_records_present").fill_null(0),
            pl.col("recall_source_records_present").fill_null(0),
        ).drop(["_feature_year", "_feature_make", "_feature_model"])
    if model_type.endswith("_truck"):
        frame = filter_vehicle_subtype(frame, "truck")
    elif model_type.endswith("_ev_hev"):
        frame = filter_vehicle_subtype(frame, "ev_hev")

    invalid_price = frame.filter(
        pl.col(TARGET_COLUMN).is_null() | (pl.col(TARGET_COLUMN) <= 0)
    ).height
    frame = frame.filter(pl.col(TARGET_COLUMN).is_not_null() & (pl.col(TARGET_COLUMN) > 0))

    frame = add_listing_features(add_cohort_groups(frame))

    # Match the dictionary's missing-value rules in both training and scoring.
    frame = _normalize_model_feature_strings(frame, features)

    # Review flags do not exclude, cap, or alter any finite observed values.
    frame = frame.with_columns(
        ((pl.col("mileage") < 0) | (pl.col("mileage") > 1_000_000))
        .fill_null(False).alias("mileage_review_flag"),
        ((pl.col("price") < 500) | (pl.col("price") > 500_000))
        .fill_null(False).alias("price_review_flag"),
    )
    selected = [*features, *COHORT_COLUMNS, TARGET_COLUMN, "_cohort_group", "vin", "cohort_complete",
                "split_identity_source", "mileage_review_flag", "price_review_flag"]
    result = frame.select(selected)
    info = {
        "joined_rows_before_price_and_subtype_filters": joined_rows,
        "listing_rows_without_vpic_match": unmatched_vpic_rows,
        "rows_removed_for_nonpositive_or_missing_price": invalid_price,
        "cohort_ratio_exclusion_vins": len(excluded_vins),
        "rows_after_filters": result.height,
        "requested_sample_size": sample_size,
        "sampled_listing_count_before_join": sampled_listing_count,
        "unique_split_groups": result["_cohort_group"].n_unique(),
        "unique_complete_cohorts": result.filter(pl.col("cohort_complete"))
            .select(COHORT_COLUMNS).unique().height,
        "complete_cohort_rows": result["cohort_complete"].sum(),
        "incomplete_cohort_rows": result.height - result["cohort_complete"].sum(),
        "split_identity_counts": result.group_by("split_identity_source").len().to_dicts(),
        "extreme_value_policy": "exclude null mileage and review-file VINs with price/cohort-median ratio >2 or <=0.75, or mileage/cohort-median ratio >20; no absolute price/mileage bounds",
        "review_flag_counts": {c: result[c].sum() for c in ("mileage_review_flag", "price_review_flag")},
        "feature_columns": features,
        "listing_feature_columns": listing_features,
        "vpic_feature_columns": vpic_features,
    }
    if model_type == "enriched":
        info["complaint_recall_feature_build"] = issue_feature_stats
        info["complaint_recall_coverage"] = {
            c: frame[c].value_counts().to_dicts()
            for c in ("complaint_source_records_present", "recall_source_records_present")
        }
    return result, info


def prepare_prediction_frame(model_type: str, listings: pl.DataFrame) -> pl.DataFrame:
    """Assemble model inputs from listing rows without requiring a price label.

    Rows join to the saved vPIC snapshot on lowercase ``vin``. Missing source
    fields and unavailable decodes are left null for the fitted preprocessor.
    """
    if model_type not in MODEL_FEATURE_SETS:
        raise ValueError(f"Unknown model_type {model_type!r}; choose from: "
                         + ", ".join(MODEL_FEATURE_SETS))
    if not isinstance(listings, pl.DataFrame):
        listings = pl.DataFrame(listings)
    if "vin" not in listings.columns:
        raise ValueError("Prediction listings must include lowercase vin")

    listing_features, vpic_features = _validate_snapshot_columns(model_type)
    listing_schema = scan_final_table("used_cars").collect_schema()
    model_features = list(MODEL_FEATURE_SETS[model_type])
    listing_columns = list(dict.fromkeys([
        "vin", *listing_features, "maximum_seating", "major_options",
    ]))
    available_columns = [column for column in listing_columns if column in listings.columns]
    frame = clean_used_cars_batch(listings.select(available_columns))
    frame = frame.with_columns(pl.col("vin").cast(pl.String).str.strip_chars())
    for column in listing_columns:
        if column == "vin" or column in frame.columns:
            continue
        frame = frame.with_columns(pl.lit(None, dtype=listing_schema[column]).alias(column))
    frame = frame.with_columns([
        pl.col(column).cast(listing_schema[column], strict=False).alias(column)
        for column in listing_columns if column != "vin"
    ])

    vpic_columns = list(dict.fromkeys([
        "vin", *vpic_features, *COHORT_COLUMNS, *SUBTYPE_SUPPORT_COLUMNS,
    ]))
    vins = frame.get_column("vin").drop_nulls().unique().to_list()
    vpic = (scan_final_table("nhtsa_vpic_decodes").select(vpic_columns)
            .filter(pl.col("vin").is_in(vins)).collect())
    frame = frame.join(vpic, on="vin", how="left", validate="m:1", maintain_order="left")

    if model_type == "enriched":
        feature_frame, _ = build_complaint_recall_features()
        frame = frame.with_columns(
            pl.col("ModelYear").cast(pl.Int64, strict=False).alias("_feature_year"),
            pl.col("Make").cast(pl.String).str.strip_chars().str.to_uppercase()
                .str.replace_all(r"\s+", " ").alias("_feature_make"),
            pl.col("Model").cast(pl.String).str.strip_chars().str.to_uppercase()
                .str.replace_all(r"\s+", " ").alias("_feature_model"),
        )
        frame = frame.join(feature_frame,
            left_on=["_feature_year", "_feature_make", "_feature_model"],
            right_on=["ModelYear", "Make", "Model"],
            how="left", validate="m:1", maintain_order="left")
        frame = frame.with_columns(
            pl.col("complaint_source_records_present").fill_null(0),
            pl.col("recall_source_records_present").fill_null(0),
        ).drop(["_feature_year", "_feature_make", "_feature_model"])

    frame = _normalize_model_feature_strings(
        add_listing_features(add_cohort_groups(frame)), model_features
    )
    receipt_columns = list(dict.fromkeys(["vin", *COHORT_COLUMNS, "is_new", *model_features]))
    return frame.select(receipt_columns)


def predict_listings(model_type: str, predictor_path: Path,
                     listings: pl.DataFrame) -> pl.DataFrame:
    """Score listing rows and return VIN, decoded identity, scope, and USD price."""
    import joblib

    frame = prepare_prediction_frame(model_type, listings)
    predictor = joblib.load(predictor_path)
    predictions = predictor.predict(frame.select(MODEL_FEATURE_SETS[model_type]).to_pandas())
    return frame.select(["vin", *COHORT_COLUMNS, "is_new"]).with_columns(
        pl.Series("predicted_price_usd", predictions)
    )


def split_by_vehicle_cohort(
    groups: Any,
    test_fraction: float = 0.20,
) -> tuple[Any, Any, Any, Any]:
    """Partition supplied vehicle-family blocks without cross-partition overlap."""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    group_values = np.asarray(groups)
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 2:
        raise ValueError(
            "A group holdout needs at least two distinct vehicle-family blocks. "
            "Use a larger or broader dataset."
        )
    test_group_count = max(1, int(np.ceil(len(unique_groups) * test_fraction)))
    test_group_count = min(test_group_count, len(unique_groups) - 1)
    outer = GroupShuffleSplit(
        n_splits=1, test_size=test_group_count, random_state=RANDOM_SEED
    )
    train_val_index, test_index = next(
        outer.split(np.zeros(len(group_values)), groups=group_values)
    )
    train_val_groups = group_values[train_val_index]
    remaining_group_count = len(np.unique(train_val_groups))
    if remaining_group_count < 2:
        # A two-cohort dataset can still produce a valid, disjoint test set,
        # but has no separate cohort available for early stopping.
        return train_val_index, np.asarray([], dtype=int), test_index, {
            "test_group_count": test_group_count,
            "validation_group_count": 0,
            "early_stopping_validation_available": False,
        }
    validation_group_count = max(1, int(np.ceil(remaining_group_count * test_fraction)))
    validation_group_count = min(validation_group_count, remaining_group_count - 1)
    inner = GroupShuffleSplit(
        n_splits=1, test_size=validation_group_count, random_state=RANDOM_SEED + 1
    )
    fit_relative, validation_relative = next(
        inner.split(np.zeros(len(train_val_index)), groups=train_val_groups)
    )
    fit_index = train_val_index[fit_relative]
    validation_index = train_val_index[validation_relative]
    return fit_index, validation_index, test_index, {
        "test_group_count": test_group_count,
        "validation_group_count": validation_group_count,
        "early_stopping_validation_available": True,
    }


def _make_tree_model(
    name: str,
    iterations: int,
    *,
    early_stopping: bool = True,
    parameters: dict[str, Any] | None = None,
) -> Any:
    parameters = parameters or {}
    if name == "catboost":
        from catboost import CatBoostRegressor

        extra_parameters = {
            key: value for key, value in parameters.items()
            if key not in {"depth", "learning_rate"}
        }
        return CatBoostRegressor(
            iterations=iterations, depth=parameters.get("depth", 8),
            learning_rate=parameters.get("learning_rate", 0.05),
            loss_function="MAE", eval_metric="MAE", random_seed=RANDOM_SEED,
            thread_count=TREE_THREADS, allow_writing_files=False, verbose=False,
            **extra_parameters,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        extra_parameters = {
            key: value for key, value in parameters.items()
            if key not in {"learning_rate", "num_leaves", "min_child_samples"}
        }
        return LGBMRegressor(
            objective="regression_l1", n_estimators=iterations,
            learning_rate=parameters.get("learning_rate", 0.05),
            num_leaves=parameters.get("num_leaves", 31), max_depth=-1,
            min_child_samples=parameters.get("min_child_samples", 100), n_jobs=TREE_THREADS,
            random_state=RANDOM_SEED, verbosity=-1,
            **extra_parameters,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        extra_parameters = {
            key: value for key, value in parameters.items()
            if key not in {"learning_rate", "max_depth", "min_child_weight",
                           "subsample", "colsample_bytree", "objective"}
        }
        return XGBRegressor(
            objective=parameters.get("objective", "reg:absoluteerror"),
            eval_metric="mae", n_estimators=iterations,
            learning_rate=parameters.get("learning_rate", 0.05),
            max_depth=parameters.get("max_depth", 8),
            min_child_weight=parameters.get("min_child_weight", 10),
            subsample=parameters.get("subsample", 0.8),
            colsample_bytree=parameters.get("colsample_bytree", 0.8), tree_method="hist",
            enable_categorical=True, n_jobs=TREE_THREADS,
            random_state=RANDOM_SEED,
            **({"early_stopping_rounds": TREE_EARLY_STOPPING_ROUNDS} if early_stopping else {}),
            **extra_parameters,
        )
    raise ValueError(f"Unsupported pilot model {name!r}.")


def _fit_with_early_stopping(
    name: str,
    x_fit: Any,
    y_fit: Any,
    x_validation: Any,
    y_validation: Any,
    categorical_columns: list[str],
    max_trees: int = PILOT_MAX_TREES,
    diagnostics: list[dict[str, Any]] | None = None,
    sample_weight: Any = None,
    selected_evaluation: dict[str, Any] | None = None,
) -> tuple[int, float, dict[str, Any], bool]:
    from sklearn.metrics import mean_absolute_error
    import numpy as np

    best: tuple[float, int, dict[str, Any], bool] | None = None
    cap_limited_count = 0
    actual_prices = np.exp(y_validation)

    def evaluate(parameters: dict[str, Any], stage: str) -> None:
        nonlocal best, cap_limited_count
        model = _make_tree_model(name, max_trees, parameters=parameters)
        if name == "catboost":
            from catboost import Pool

            model.fit(
                x_fit, y_fit, cat_features=categorical_columns,
                sample_weight=sample_weight,
                eval_set=Pool(x_validation, y_validation, cat_features=categorical_columns),
                early_stopping_rounds=TREE_EARLY_STOPPING_ROUNDS, verbose=False,
            )
            best_iteration = model.best_iteration_
            iterations = int(best_iteration + 1) if best_iteration is not None and best_iteration >= 0 \
                else max(1, int(model.tree_count_))
        elif name == "lightgbm":
            import lightgbm as lgb

            model.fit(
                x_fit, y_fit, categorical_feature=categorical_columns,
                sample_weight=sample_weight,
                eval_X=x_validation, eval_y=y_validation, eval_metric="l1",
                callbacks=[lgb.early_stopping(TREE_EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            iterations = max(1, int(model.best_iteration_ or model.n_estimators))
        else:
            model.fit(x_fit, y_fit, sample_weight=sample_weight,
                      eval_set=[(x_validation, y_validation)], verbose=False)
            iterations = max(1, int(model.best_iteration + 1))
        predicted = np.exp(model.predict(x_validation))
        scores = validation_metrics(actual_prices, predicted)
        score = scores["validation_mae"]
        history = model.evals_result_ if name != "xgboost" else model.evals_result()
        validation_key = {"catboost": "validation", "lightgbm": "valid_0",
                          "xgboost": "validation_0"}[name]
        rounds_trained = len(next(iter(history[validation_key].values())))
        rounds_after_best = rounds_trained - iterations
        cap_limited = (rounds_trained >= max_trees
                       and rounds_after_best < TREE_EARLY_STOPPING_ROUNDS)
        cap_limited_count += cap_limited
        if diagnostics is not None:
            diagnostics.append({"model": name, "stage": stage, "parameters": parameters,
                                **scores, "best_iteration": iterations,
                                "rounds_trained": rounds_trained,
                                "rounds_after_best": rounds_after_best,
                                "early_stopping_rounds": TREE_EARLY_STOPPING_ROUNDS,
                                "tree_cap_reached": cap_limited,
                                "learning_curve": history})
        if best is None or score < best[0]:
            best = (score, iterations, parameters, cap_limited)
            if selected_evaluation is not None:
                selected_evaluation.clear()
                selected_evaluation.update(scores, predictions=predicted)

    for parameters in TREE_PARAMETER_CANDIDATES[name]:
        evaluate(parameters, "structure")
    assert best is not None
    best_structure = best[2]
    for override in TREE_REFINEMENT_OVERRIDES[name]:
        evaluate({**best_structure, **override}, "refinement")

    if cap_limited_count:
        selected_status = ("The selected candidate is budget-limited."
                           if best[3] else "The selected candidate stopped earlier.")
        warnings.warn(
            f"{name}: {cap_limited_count} candidate(s) exhausted the {max_trees}-tree "
            f"budget before {TREE_EARLY_STOPPING_ROUNDS} validation rounds passed "
            f"their best iteration. {selected_status} Inspect learning curves and "
            "raise --max-trees if needed."
        )
    assert best is not None
    return best[1], best[0], best[2], best[3]


def _make_classical_pipeline(
    name: str,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> Any:
    """Build sklearn preprocessing and estimator for non-boosting models.

    Categories are one-hot encoded with infrequent levels grouped to bound
    dimensionality. Linear predictors scale numeric columns before fitting;
    Random Forest uses the same encoding without scaling.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    scale_numeric = name != "random_forest"
    numeric_steps = [("scale", StandardScaler())] if scale_numeric else []
    numeric_pipeline = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", OneHotEncoder(
                handle_unknown="ignore", min_frequency=20, sparse_output=True,
            ), categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    estimators = {
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_features=0.5, min_samples_leaf=5,
            max_depth=24, max_samples=0.7,
            n_jobs=TREE_THREADS, random_state=RANDOM_SEED,
        ),
        "multiple_linear_regression": LinearRegression(),
        "lasso": Lasso(alpha=100.0, max_iter=20_000, tol=1e-3,
                       selection="cyclic", random_state=RANDOM_SEED),
        "ridge": Ridge(alpha=10.0, solver="lsqr"),
        "elastic_net": ElasticNet(alpha=100.0, l1_ratio=0.5, max_iter=20_000,
                                  tol=1e-3, random_state=RANDOM_SEED),
    }
    return Pipeline([("preprocess", preprocessor), ("model", estimators[name])])


def _fit_classical_model(
    name: str,
    x_fit: Any,
    y_fit: Any,
    x_validation: Any | None,
    y_validation: Any | None,
    categorical_columns: list[str],
    numeric_columns: list[str],
    diagnostics: list[dict[str, Any]] | None = None,
    sample_weight: Any = None,
    selected_evaluation: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], float | None]:
    """Tune classical estimators on validation cohorts and refit on outer train."""
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import ParameterGrid
    from sklearn.exceptions import ConvergenceWarning

    pipeline = _make_classical_pipeline(name, categorical_columns, numeric_columns)
    grid = CLASSICAL_PARAMETER_GRIDS[name]
    candidates = list(ParameterGrid(grid)) if grid else [{}]
    best_params: dict[str, Any] = {}
    best_score: float | None = None
    if x_validation is not None:
        for params in candidates:
            candidate = _make_classical_pipeline(
                name, categorical_columns, numeric_columns
            ).set_params(**params)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                candidate.fit(x_fit, y_fit, model__sample_weight=sample_weight)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            predicted = np.exp(candidate.predict(x_validation))
            scores = validation_metrics(np.exp(y_validation), predicted)
            score = scores["validation_mae"]
            if diagnostics is not None:
                diagnostics.append({"model": name, "parameters": params, **scores,
                                    "converged": converged, "warnings": [str(w.message) for w in caught]})
            if converged and (best_score is None or score < best_score):
                pipeline, best_params, best_score = candidate, params, score
                if selected_evaluation is not None:
                    selected_evaluation.clear()
                    selected_evaluation.update(scores, predictions=predicted)
        if best_score is None:
            raise RuntimeError(f"No converged {name} candidate; inspect tuning diagnostics and adjust regularization/iterations")
    elif candidates:
        # The first grid entry is a deterministic fallback when a disjoint
        # validation cohort cannot be formed; no validation-based tuning occurs.
        pipeline.set_params(**candidates[0])
        best_params = candidates[0]
    return pipeline, best_params, best_score


def _save_model(model: Any, name: str, path: Path) -> str:
    if name == "catboost":
        path = path.with_suffix(".cbm")
        model.save_model(path)
    elif name == "xgboost":
        path = path.with_suffix(".json")
        model.save_model(path)
    else:
        import joblib

        path = path.with_suffix(".joblib")
        joblib.dump(model, path, compress=3)
    return path.name


def _write_shap_outputs(
    model: Any,
    name: str,
    x_test: Any,
    y_test: Any,
    features: list[str],
    categorical_columns: list[str],
    output_dir: Path,
    rows: int,
) -> dict[str, str] | None:
    """Save a bounded row-level SHAP sample and global mean-absolute values."""
    import numpy as np
    import pandas as pd

    if rows <= 0 or x_test.empty:
        return None
    count = min(rows, len(x_test))
    selected = np.random.default_rng(RANDOM_SEED).choice(len(x_test), count, replace=False)
    explain_x = x_test.iloc[selected].copy()
    explain_y = np.asarray(y_test)[selected]
    try:
        if name == "catboost":
            from catboost import Pool

            raw = model.get_feature_importance(
                Pool(explain_x, cat_features=categorical_columns), type="ShapValues"
            )
            shap_values, base_values = raw[:, :-1], raw[:, -1]
        else:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(explain_x)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.asarray(shap_values)
            base = explainer.expected_value
            base_values = np.full(count, float(np.asarray(base).reshape(-1)[0]))
        shap_values = np.asarray(shap_values)
        global_importance = pd.DataFrame({
            "feature": features,
            "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_absolute_shap", ascending=False)
        global_path = output_dir / f"{name}_shap_global.csv"
        global_importance.to_csv(global_path, index=False)
        row_values = pd.DataFrame(shap_values, columns=features)
        row_values.insert(0, "base_value", base_values)
        row_values.insert(0, "predicted_log_price", model.predict(explain_x))
        row_values.insert(0, "actual_price", np.exp(explain_y))
        row_values.insert(0, "actual_log_price", explain_y)
        row_values.insert(0, "test_row", selected)
        row_values.rename(columns={"base_value": "base_value_log_price"}, inplace=True)
        row_path = output_dir / f"{name}_shap_values.parquet"
        row_values.to_parquet(row_path, index=False)
        return {"global_importance": global_path.name, "row_values": row_path.name}
    except Exception as exc:
        print(f"SHAP unavailable for {name}; model training completed. Reason: {exc}")
        return None


def run_price_model_pilot(
    model_type: str = "base",
    sample_size: int | None = 100_000,
    model_names: list[str] | None = None,
    shap_rows: int = 1_000,
    tuning_rows: int = 100_000,
    max_trees: int = PILOT_MAX_TREES,
    split_strategy: str = "unseen_family",
    price_weighting: str = "none",
) -> Path:
    """Select on validation data, refit on outer training, report untouched tests.

    All listings predict advertised listing price, with ``is_new`` retained as
    a predictor and a reported segment. Full-data mode holds test groups out.
    """
    import gc
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyRegressor
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import mean_absolute_error
    from sklearn.pipeline import Pipeline
    from EDA.DATA_CLEANING import FINAL_DATASET_PATH

    model_names = model_names or list(PILOT_MODEL_NAMES)
    unknown = sorted(set(model_names) - set(PILOT_MODEL_NAMES))
    if unknown:
        raise ValueError("Unsupported models: " + ", ".join(unknown))
    if len(set(model_names)) != len(model_names):
        raise ValueError("Model names must be unique")
    if tuning_rows <= 0 or max_trees <= 0:
        raise ValueError("tuning_rows and max_trees must be positive")
    if split_strategy not in {"unseen_family", "known_family"}:
        raise ValueError("split_strategy must be 'unseen_family' or 'known_family'")

    # Fingerprint the same documented Parquet inputs for every feature set so
    # notebook comparisons can verify a shared data snapshot.
    source_paths = [FINAL_DATASET_PATH / name for name in (
        "used_cars.parquet", "NHTSA_vPIC.parquet",
        "NHTSA_Complaints.parquet", "NHTSA_Recalls.parquet", "NHTSA_Safety.parquet",
    )]
    source_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in source_paths}
    source_hashes = {p.name: file_sha256(p) for p in source_paths}
    dataset_fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    code_hashes = {"Price_ML_Models.py": file_sha256(Path(__file__))}
    code_fingerprint = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()
    package_versions = {p: importlib.metadata.version(p) for p in
                        ("polars", "pandas", "numpy", "scikit-learn", "catboost", "lightgbm", "xgboost", "pyarrow")}

    polars_frame, data_info = load_model_frame(model_type, sample_size)
    if any((p.stat().st_size, p.stat().st_mtime_ns) != source_stats[str(p)] for p in source_paths):
        raise RuntimeError("Source snapshot changed during loading; retry against a stable export")
    raw_features = data_info["feature_columns"]
    categorical_columns = [c for c in raw_features if polars_frame.schema[c] == pl.String]
    raw_numeric_columns = [c for c in raw_features if polars_frame.schema[c].is_numeric()]
    frame = polars_frame.to_pandas()
    del polars_frame
    family_groups = frame["_cohort_group"].to_numpy()
    groups = family_groups if split_strategy == "unseen_family" else frame["vin"].to_numpy()
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    y_log = np.log(y)
    fit_index, validation_index, test_index, split_info = split_by_vehicle_cohort(groups)
    outer_train_index = np.sort(np.concatenate([fit_index, validation_index]))

    # Keep evaluation/split identities and raw decoded dimensions apart from
    # predictor imputation. Every valid row is accounted for in the manifest.
    receipt_columns = ["vin", *COHORT_COLUMNS, "make_name", "is_new", "mileage", "cohort_complete", "split_identity_source",
                       "mileage_review_flag", "price_review_flag",
                       *[c for c in ("complaint_source_records_present", "recall_source_records_present") if c in frame.columns]]
    manifest = frame[receipt_columns].copy()
    manifest["split_group"] = groups
    manifest["family_block"] = family_groups
    manifest["partition"] = "fit"
    manifest.loc[validation_index, "partition"] = "validation"
    manifest.loc[test_index, "partition"] = "test"
    split_hash = split_fingerprint(manifest)
    x = frame[raw_features].copy()
    del frame

    # Bounded, target-independent tuning samples. Vocabulary/imputation fit
    # only on the chosen fit rows; validation remains transform-only.
    rng = np.random.default_rng(RANDOM_SEED)
    def bounded(index):
        return np.sort(rng.choice(index, min(tuning_rows, len(index)), replace=False))
    tune_fit_index = bounded(fit_index)
    tune_validation_index = bounded(validation_index)
    tune_weights = training_price_weights(y[tune_fit_index], price_weighting)
    final_weights = training_price_weights(y[outer_train_index], price_weighting)
    tune_preprocessor = PricePreprocessor(categorical_columns, raw_numeric_columns)
    x_fit = tune_preprocessor.fit_transform(x.iloc[tune_fit_index])
    x_validation = (tune_preprocessor.transform(x.iloc[tune_validation_index])
                    if len(tune_validation_index) else None)
    final_preprocessor = PricePreprocessor(categorical_columns, raw_numeric_columns)
    x_train = final_preprocessor.fit_transform(x.iloc[outer_train_index])
    x_test = final_preprocessor.transform(x.iloc[test_index])
    features = final_preprocessor.output_columns_
    numeric_columns = raw_numeric_columns + [f"{c}__missing" for c in raw_numeric_columns]
    del x
    gc.collect()

    output_dir = MODEL_OUTPUT_DIR / model_type / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + (f"-pilot-{sample_size}" if sample_size else "-full")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest.to_parquet(output_dir / "split_manifest.parquet", index=False)
    flagged = manifest[manifest["mileage_review_flag"] | manifest["price_review_flag"]].copy()
    flagged["actual_price"] = y[flagged.index]
    # Mileage itself remains a predictor in the source; flags are audit-only.
    flagged.to_parquet(output_dir / "extreme_value_review.parquet", index=False)
    test_receipt = manifest.iloc[test_index].copy().reset_index(drop=True)
    test_receipt["actual_price"] = y[test_index]
    validation_receipt = manifest.iloc[tune_validation_index].copy().reset_index(drop=True)
    validation_receipt["actual_price"] = y[tune_validation_index]
    del manifest, flagged
    metrics, segments, year_make_segments, diagnostics = [], [], [], []
    validation_segments, validation_year_make = [], []
    artifacts = {}
    metadata = {
        **data_info, "status": "running", "model_type": model_type,
        "target_column": TARGET_COLUMN,
        "target_transform": "natural_log",
        "price_weighting": price_weighting,
        "price_weighting_policy": {"reference_price_usd": PRICE_WEIGHT_REFERENCE,
                                   "max_training_weight": PRICE_WEIGHT_CAP,
                                   "formula": "clip(training_price / reference_price, 1, cap)",
                                   "evaluation_weighting": "none"},
        "target_model_scale": "ln(price in USD)",
        "target_output_scale": "price in USD; inverse transform is exp(prediction)",
        "shap_output_scale": "ln(price in USD)",
        "target_interpretation": "advertised listing price in USD across all listings; is_new is a predictor and reporting segment; not a sale price",
        "raw_feature_columns": raw_features, "feature_columns": features,
        "categorical_features": categorical_columns, "numeric_features": numeric_columns,
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset_fingerprint": dataset_fingerprint, "source_file_sha256": source_hashes,
        "split_fingerprint": split_hash, "code_fingerprint": code_fingerprint,
        "package_versions": package_versions,
        "split_strategy": split_strategy,
        "split_method": ("year/make/model blocks; all trims together" if split_strategy == "unseen_family"
                         else "VIN groups split within the full population; year/make/model families may appear in all partitions"),
        "split_seed": RANDOM_SEED, "split_details": split_info,
        "train_rows": len(outer_train_index), "validation_rows_for_early_stopping": len(validation_index),
        "test_rows": len(test_index),
        "train_split_groups": len(np.unique(groups[outer_train_index])),
        "validation_split_groups": len(np.unique(groups[validation_index])),
        "test_split_groups": len(np.unique(groups[test_index])),
        "tuning_rows": tuning_rows, "actual_tuning_fit_rows": len(tune_fit_index),
        "actual_tuning_validation_rows": len(tune_validation_index), "max_trees": max_trees,
        "early_stopping_rounds": TREE_EARLY_STOPPING_ROUNDS,
        "tree_iteration_selection_metric": "validation MAE on ln(price)",
        "tree_candidate_selection_metric": "validation MAE in USD",
        "imputation": {"method": "training-only numeric median plus all-numeric missing indicators; categorical missing/unknown tokens",
                       "all_missing_numeric_fallback": 0,
                       "test_cohort_imputation": "none; transform using stored training statistics only"},
        "uncertainty": "95% percentile interval from 200 resamples of held-out split blocks; conditional on this fitted model/split",
        "models": artifacts,
    }
    def checkpoint():
        pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
        pd.DataFrame(segments).to_csv(output_dir / "segment_metrics.csv", index=False)
        pd.DataFrame(year_make_segments).to_csv(output_dir / "year_make_metrics.csv", index=False)
        pd.DataFrame(validation_segments).to_csv(output_dir / "validation_segment_metrics.csv", index=False)
        pd.DataFrame(validation_year_make).to_csv(output_dir / "validation_year_make_metrics.csv", index=False)
        pd.DataFrame([{k: v for k, v in row.items() if k == "model" or k.startswith("validation_")}
                      for row in metrics]).to_csv(output_dir / "validation_metrics.csv", index=False)
        (output_dir / "tuning_results.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
        # Publish metadata last. Notebook accepts completed runs only.
        (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    def record(name, predictions, validation_mae, params, iterations, selected_evaluation=None):
        result = prediction_metrics(y[test_index], predictions, groups[test_index])
        result.update(model=name, test_rows=len(test_index), validation_mae=validation_mae,
                      selected_parameters=json.dumps(params, default=str), best_iteration=iterations)
        if selected_evaluation:
            result.update({k: v for k, v in selected_evaluation.items() if k != "predictions"})
            validation_predictions = validation_receipt.copy()
            validation_predictions["prediction"] = selected_evaluation["predictions"]
            validation_segments.extend(segment_metrics(validation_predictions, name))
            validation_year_make.extend(year_make_metrics(validation_predictions, name))
        metrics.append(result)
        receipt = test_receipt.copy()
        receipt["prediction"] = predictions
        receipt["residual"] = receipt["actual_price"] - receipt["prediction"]
        receipt.to_parquet(output_dir / f"{name}_test_predictions.parquet", index=False)
        segments.extend(segment_metrics(receipt, name))
        year_make_segments.extend(year_make_metrics(receipt, name))
        print(f"{name}: test MAE=${result['mae']:,.0f}; test R²={result['r2']}; validation MAE={validation_mae}; "
              f"trees={iterations}; parameters={params}")

    baseline = DummyRegressor(strategy="median").fit(np.zeros((len(outer_train_index), 1)), y_log[outer_train_index])
    validation_baseline = DummyRegressor(strategy="median").fit(np.zeros((len(tune_fit_index), 1)), y_log[tune_fit_index])
    baseline_evaluation = {}
    if len(tune_validation_index):
        baseline_predictions = np.exp(validation_baseline.predict(np.zeros((len(tune_validation_index), 1))))
        baseline_evaluation.update(validation_metrics(y[tune_validation_index], baseline_predictions), predictions=baseline_predictions)
    record("training_median_baseline", np.exp(baseline.predict(np.zeros((len(test_index), 1)))),
           baseline_evaluation.get("validation_mae"), {}, None, baseline_evaluation)
    checkpoint()
    try:
        for name in model_names:
            params, validation_mae, iterations = {}, None, None
            selected_evaluation = {}
            if name in TREE_MODEL_NAMES:
                if x_validation is not None:
                    iterations, validation_mae, params, selected_cap_limited = _fit_with_early_stopping(
                        name, x_fit, y_log[tune_fit_index], x_validation, y_log[tune_validation_index],
                        categorical_columns, max_trees, diagnostics,
                        sample_weight=tune_weights, selected_evaluation=selected_evaluation,
                    )
                else:
                    iterations, params = min(400, max_trees), TREE_PARAMETER_CANDIDATES[name][0]
                    selected_cap_limited = False
                final_model = _make_tree_model(name, iterations, early_stopping=False, parameters=params)
                if name == "catboost":
                    final_model.fit(x_train, y_log[outer_train_index], sample_weight=final_weights,
                                    cat_features=categorical_columns, verbose=False)
                elif name == "lightgbm":
                    final_model.fit(x_train, y_log[outer_train_index], sample_weight=final_weights,
                                    categorical_feature=categorical_columns)
                else:
                    final_model.fit(x_train, y_log[outer_train_index], sample_weight=final_weights, verbose=False)
            else:
                final_model, params, validation_mae = _fit_classical_model(
                    name, x_fit, y_log[tune_fit_index], x_validation,
                    y_log[tune_validation_index] if x_validation is not None else None,
                    categorical_columns, numeric_columns, diagnostics,
                    sample_weight=tune_weights, selected_evaluation=selected_evaluation,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    final_model.fit(x_train, y_log[outer_train_index], model__sample_weight=final_weights)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    diagnostics.append({"model": name, "phase": "final_refit", "converged": False,
                                        "warnings": [str(w.message) for w in caught]})
                    raise RuntimeError(f"{name} final refit did not converge; successful earlier model artifacts are preserved")
            model_dir = output_dir / name
            model_dir.mkdir()
            native_file = _save_model(final_model, name, model_dir / "model")
            # This bundle accepts the raw, joined feature columns in the saved
            # schema; transformations never inspect other prediction rows.
            bundle = Pipeline([("input", final_preprocessor),
                               ("estimator", ExpLogPricePredictor(final_model))])
            joblib.dump(bundle, model_dir / "predictor.joblib", compress=3)
            record(name, np.exp(final_model.predict(x_test)), validation_mae, params, iterations, selected_evaluation)
            artifacts[name] = {
                "model_file": native_file, "predictor_file": "predictor.joblib",
                "best_iteration": iterations, "selected_parameters": params,
                "validation_mae": validation_mae,
                **{k: v for k, v in selected_evaluation.items() if k != "predictions"},
                "validation_tuning_performed": x_validation is not None,
                "tree_cap_reached": selected_cap_limited if name in TREE_MODEL_NAMES else False,
                "shap_files": (_write_shap_outputs(final_model, name, x_test, y_log[test_index], features,
                    categorical_columns, model_dir, shap_rows) if name in TREE_MODEL_NAMES else None),
            }
            checkpoint()
            del bundle, final_model
            gc.collect()
        candidates = [row for row in metrics
                      if row["validation_mae"] is not None
                      and row["model"] != "training_median_baseline"]
        metadata["selected_by_validation_mae"] = min(candidates, key=lambda row: row["validation_mae"])["model"] if candidates else None
        metadata["status"] = "completed"
    except Exception as exc:
        metadata.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        checkpoint()
    print(f"Saved outputs to: {output_dir}")
    return output_dir


def run_price_forecast_suite(args: Any) -> int:
    """Run feature sets sequentially in fresh instances of this script."""
    import subprocess
    import sys

    failures = []
    combinations = [(kind, mode) for kind in args.model_types for mode in args.price_weightings]
    for model_type, price_weighting in combinations:
        command = [sys.executable, "-m", "MODELS.Price_ML_Models",
                   "--model-type", model_type, "--sample-size", str(args.sample_size),
                   "--tuning-rows", str(args.tuning_rows), "--max-trees", str(args.max_trees),
                   "--shap-rows", str(args.shap_rows), "--split-strategy", args.split_strategy,
                   "--price-weightings", price_weighting,
                   "--models", *args.models]
        print(f"\n=== Running {model_type} / weighting={price_weighting} ===", flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            failures.append((model_type, price_weighting, result.returncode))
            print(f"{model_type} / {price_weighting} failed with exit code {result.returncode}; continuing.",
                  flush=True)
    if failures:
        print("Failed forecast runs:", failures, file=sys.stderr)
        return 1
    print("All requested forecast runs completed.")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train group-held-out advertised-price models for all listings.")
    parser.add_argument("--model-type", choices=MODEL_FEATURE_SETS, default="base")
    parser.add_argument("--model-types", nargs="+", choices=MODEL_FEATURE_SETS,
                        help="Run several feature sets sequentially.")
    parser.add_argument("--split-strategy", choices=("unseen_family", "known_family"),
                        default="unseen_family",
                        help="Hold out year/make/model families or VINs within known families.")
    parser.add_argument("--sample-size", type=int, default=100_000,
                        help="Uniform VIN-hash sample size; use 0 for the full filtered data.")
    parser.add_argument("--models", nargs="+", choices=PILOT_MODEL_NAMES,
                        default=list(PILOT_MODEL_NAMES))
    parser.add_argument("--price-weightings", nargs="+", choices=PRICE_WEIGHTING_MODES,
                        default=["none"], help="Compare ordinary training and capped price weighting in separate runs.")
    parser.add_argument("--shap-rows", type=int, default=1_000,
                        help="Maximum held-out rows explained per model; use 0 to skip SHAP.")
    parser.add_argument("--tuning-rows", type=int, default=100_000,
                        help="Maximum rows in each tuning partition; final refit uses all outer-training rows.")
    parser.add_argument("--max-trees", type=int, default=PILOT_MAX_TREES,
                        help="Boosting search ceiling (default 6400); early stopping selects fewer "
                             "trees when validation MAE plateaus.")
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error("--sample-size must be 0 or positive")
    args.price_weightings = list(dict.fromkeys(args.price_weightings))
    if args.model_types or len(args.price_weightings) > 1:
        args.model_types = list(dict.fromkeys(args.model_types or [args.model_type]))
        raise SystemExit(run_price_forecast_suite(args))
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_price_model_pilot(
        model_type=args.model_type,
        sample_size=None if args.sample_size == 0 else args.sample_size,
        model_names=args.models,
        shap_rows=args.shap_rows,
        tuning_rows=args.tuning_rows,
        max_trees=args.max_trees,
        split_strategy=args.split_strategy,
        price_weighting=args.price_weightings[0],
    )
