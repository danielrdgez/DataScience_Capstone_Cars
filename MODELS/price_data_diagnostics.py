"""Read-only preflight diagnostics for the car price modeling sources.

Run from the project root with::

    python -m MODELS.price_data_diagnostics

Source Parquet files are only scanned. Reports and candidate rows are written
under ``MODEL_OUTPUT/diagnostics/<run-id>/`` (ignored by Git). Review flags are
heuristics for inspection; this module never filters or edits source rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import polars as pl

from EDA.DATA_CLEANING import FINAL_DATASET_PATH, PARQUET_FILES
from MODELS.Price_ML_Models import BASE_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MODEL_OUTPUT" / "diagnostics"
EXPECTED_TABLES = tuple(PARQUET_FILES)
REVIEW_CATEGORY_COLUMNS = (
    "model_name", "trim_name", "exterior_color", "interior_color",
    "listing_color", "major_options",
)
RARE_CATEGORY_COLUMNS = set(REVIEW_CATEGORY_COLUMNS) - {"major_options"}
COHORT_COLUMNS = ("ModelYear", "Make", "Model", "Trim")
COLOR_MISSING_LABELS = {"NONE", "NULL", "UNKNOWN", "N/A", "NA", "NOT APPLICABLE"}


def _collect(frame: pl.LazyFrame | pl.Expr) -> pl.DataFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame.collect(engine="streaming")
    raise TypeError("Expected a Polars LazyFrame")


def _write_csv(frame: pl.DataFrame, path: Path) -> None:
    frame.write_csv(path)


def _scope_expression() -> pl.Expr:
    value = pl.col("is_new").cast(pl.String).str.strip_chars().str.to_lowercase()
    return (
        pl.when(value.is_in(["true", "1", "new"]))
        .then(pl.lit("new"))
        .when(value.is_in(["false", "0", "used"]))
        .then(pl.lit("used"))
        .otherwise(pl.lit("unknown"))
        .alias("vehicle_scope")
    )


def _normalized_string(column: str) -> pl.Expr:
    return (
        pl.col(column).cast(pl.String).str.strip_chars().str.to_uppercase()
        .str.replace_all(r"\s+", " ")
    )


def _missing_expression(column: str, dtype: pl.DataType) -> pl.Expr:
    value = pl.col(column)
    if dtype == pl.String:
        missing = value.is_null() | (value.str.strip_chars() == "")
        if column in {"exterior_color", "interior_color", "listing_color"}:
            missing = missing | value.str.strip_chars().str.to_uppercase().is_in(
                sorted(COLOR_MISSING_LABELS)
            )
        return missing
    if dtype.is_float():
        return value.is_null() | ~value.is_finite()
    return value.is_null()


def _date_expression(column: str, schema: dict[str, pl.DataType]) -> pl.Expr:
    value = pl.col(column)
    dtype = schema[column]
    if dtype == pl.Date:
        return value
    if dtype.is_temporal():
        return value.cast(pl.Date, strict=False)
    return value.cast(pl.String).str.to_date(strict=False, exact=False)


def _table_inventory(data_dir: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name, filename in PARQUET_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            rows.append({
                "table": table_name, "file": filename, "exists": False,
                "rows": None, "columns": None, "size_mib": None,
                "modified_utc": None, "field_names": None,
            })
            continue
        lazy = pl.scan_parquet(path)
        schema = lazy.collect_schema()
        count = _collect(lazy.select(pl.len().alias("rows"))).item()
        rows.append({
            "table": table_name, "file": filename, "exists": True,
            "rows": int(count), "columns": len(schema),
            "size_mib": round(path.stat().st_size / (1024 * 1024), 2),
            "modified_utc": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "field_names": ", ".join(schema.names()),
        })
    return pl.DataFrame(rows)


def _manifest_check(data_dir: Path, inventory: pl.DataFrame) -> pl.DataFrame:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        return pl.DataFrame([{
            "manifest_present": False, "export_created_at_utc": None,
            "tables_with_count_mismatch": None,
            "note": "No export manifest was found; compare source inventory to the expected snapshot manually.",
        }])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_manifest = manifest.get("tables", {})
    mismatches = []
    for row in inventory.to_dicts():
        if not row["exists"] or row["table"] not in table_manifest:
            continue
        expected = table_manifest[row["table"]].get("rows")
        if expected is not None and int(expected) != int(row["rows"]):
            mismatches.append({
                "table": row["table"], "manifest_rows": int(expected),
                "parquet_rows": int(row["rows"]),
            })
    return pl.DataFrame([{
        "manifest_present": True,
        "export_created_at_utc": manifest.get("created_at_utc"),
        "tables_with_count_mismatch": len(mismatches),
        "mismatch_details": json.dumps(mismatches),
        "note": "Manifest row counts are checked against the current Parquet files.",
    }])


def _scope_summary(listings: pl.LazyFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    expressions: list[pl.Expr] = [pl.len().alias("rows")]
    if "vin" in schema:
        expressions.extend([
            pl.col("vin").n_unique().alias("distinct_vins_including_null"),
            pl.col("vin").is_null().sum().alias("null_vins"),
            (pl.col("vin").cast(pl.String).str.len_chars() != 17)
                .fill_null(False).sum().alias("non_17_character_vins"),
        ])
    for column in ("listing_id", "price", "mileage", "year"):
        if column in schema:
            expressions.append(pl.col(column).is_null().sum().alias(f"null_{column}"))
    for column in ("price", "mileage"):
        if column in schema and schema[column].is_numeric():
            expressions.extend([
                pl.col(column).min().alias(f"{column}_min"),
                pl.col(column).quantile(0.50).alias(f"{column}_p50"),
                pl.col(column).quantile(0.90).alias(f"{column}_p90"),
                pl.col(column).quantile(0.99).alias(f"{column}_p99"),
                pl.col(column).max().alias(f"{column}_max"),
            ])
    if "price" in schema:
        expressions.extend([
            (pl.col("price") <= 0).fill_null(False).sum().alias("nonpositive_price_rows"),
            (pl.col("price") < 500).fill_null(False).sum().alias("price_below_500_rows"),
            (pl.col("price") > 500_000).fill_null(False).sum().alias("price_above_500k_rows"),
        ])
    if "mileage" in schema:
        expressions.extend([
            (pl.col("mileage") < 0).fill_null(False).sum().alias("negative_mileage_rows"),
            (pl.col("mileage") == 0).fill_null(False).sum().alias("zero_mileage_rows"),
            (pl.col("mileage") > 1_000_000).fill_null(False).sum().alias("mileage_above_1m_rows"),
        ])
    return _collect(
        listings.with_columns(_scope_expression())
        .group_by("vehicle_scope").agg(expressions).sort("vehicle_scope")
    )


def _listing_key_quality(listings: pl.LazyFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    checks = []
    for key in ("vin", "listing_id"):
        if key not in schema:
            continue
        duplicate_keys = _collect(
            listings.filter(pl.col(key).is_not_null())
            .group_by(key).len().filter(pl.col("len") > 1)
        )
        total = int(_collect(listings.select(pl.len().alias("rows"))).item())
        nulls = int(_collect(listings.select(pl.col(key).is_null().sum().alias("n"))).item())
        duplicate_rows = int(duplicate_keys["len"].sum()) if not duplicate_keys.is_empty() else 0
        checks.append({
            "key": key, "source_rows": total, "null_key_rows": nulls,
            "duplicate_key_values": duplicate_keys.height,
            "rows_with_a_duplicate_key": duplicate_rows,
            "duplicate_row_rate": duplicate_rows / total if total else None,
        })
    return pl.DataFrame(checks)


def _missingness(listings: pl.LazyFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    columns = [column for column in BASE_FEATURE_COLUMNS if column in schema]
    if not columns:
        return pl.DataFrame()
    expressions = [
        _missing_expression(column, schema[column]).sum().alias(column)
        for column in columns
    ]
    rows = _collect(
        listings.with_columns(_scope_expression())
        .group_by("vehicle_scope").agg(pl.len().alias("rows"), *expressions)
        .sort("vehicle_scope")
    )
    long_rows: list[dict[str, Any]] = []
    for record in rows.to_dicts():
        group_rows = record["rows"]
        for column in columns:
            missing_rows = int(record[column] or 0)
            long_rows.append({
                "vehicle_scope": record["vehicle_scope"], "feature": column,
                "rows": group_rows, "missing_rows": missing_rows,
                "missing_rate": missing_rows / group_rows if group_rows else None,
            })
    return pl.DataFrame(long_rows).sort(
        ["vehicle_scope", "missing_rate"], descending=[False, True]
    )


def _category_cardinality(
    listings: pl.LazyFrame, schema: dict[str, pl.DataType], min_rare_count: int
) -> pl.DataFrame:
    columns = [
        column for column in REVIEW_CATEGORY_COLUMNS
        if column in schema and schema[column] == pl.String
    ]
    if not columns:
        return pl.DataFrame(schema={
            "feature": pl.String, "used_nonmissing_rows": pl.Int64,
            "raw_distinct_levels": pl.Int64, "normalized_distinct_levels": pl.Int64,
            "levels_at_or_below_rare_count": pl.Int64, "rows_in_rare_levels": pl.Int64,
            "rare_count_threshold": pl.Int64,
        })
    used = listings.with_columns(_scope_expression()).filter(pl.col("vehicle_scope") == "used")
    raw_exprs, normalized_exprs = [], []
    for column in columns:
        raw_value = pl.col(column).str.strip_chars()
        normalized_value = _normalized_string(column)
        raw_exprs.append(raw_value.drop_nulls().filter(raw_value != "").n_unique().alias(column))
        normalized_exprs.append(
            normalized_value.drop_nulls().filter(normalized_value != "").n_unique().alias(column)
        )
    raw_stats = _collect(used.select(raw_exprs)).to_dicts()[0]
    normalized_stats = _collect(used.select(normalized_exprs)).to_dicts()[0]
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column in RARE_CATEGORY_COLUMNS:
            normalized = _normalized_string(column)
            level_counts = _collect(
                used.filter(normalized.is_not_null() & (normalized != ""))
                .group_by(normalized.alias("_level")).len()
            )
            rare = level_counts.filter(pl.col("len") <= min_rare_count)
            used_nonmissing = int(level_counts["len"].sum())
            rare_levels = rare.height
            rare_rows = int(rare["len"].sum()) if not rare.is_empty() else 0
        else:
            # major_options is structured text with many unique combinations.
            # Profile its cardinality without materializing a huge rare-level table.
            used_nonmissing = int(_collect(
                used.select((~_missing_expression(column, schema[column])).sum().alias("n"))
            ).item())
            rare_levels = None
            rare_rows = None
        rows.append({
            "feature": column,
            "used_nonmissing_rows": used_nonmissing,
            "raw_distinct_levels": int(raw_stats[column]),
            "normalized_distinct_levels": int(normalized_stats[column]),
            "levels_at_or_below_rare_count": rare_levels,
            "rows_in_rare_levels": rare_rows,
            "rare_count_threshold": min_rare_count if column in RARE_CATEGORY_COLUMNS else None,
        })
    return pl.DataFrame(rows).sort("normalized_distinct_levels", descending=True)


def _vpic_join(
    listings: pl.LazyFrame, listing_schema: dict[str, pl.DataType],
    vpic_path: Path, output_dir: Path,
) -> tuple[pl.LazyFrame | None, pl.DataFrame, str | None]:
    vpic = pl.scan_parquet(vpic_path)
    vpic_schema = vpic.collect_schema()
    required = {"vin", *COHORT_COLUMNS}
    missing = sorted(required - set(vpic_schema.names()))
    if missing:
        return None, pl.DataFrame(), "vPIC is missing required fields: " + ", ".join(missing)

    duplicate_vins = _collect(
        vpic.group_by("vin").len().filter(pl.col("len") > 1)
    )
    if not duplicate_vins.is_empty():
        _write_csv(duplicate_vins.head(500), output_dir / "duplicate_vpic_vins_sample.csv")
        return None, pl.DataFrame([{
            "check": "vPIC VIN uniqueness", "status": "failed",
            "duplicate_vin_keys": duplicate_vins.height,
            "duplicate_rows": int(duplicate_vins["len"].sum()),
            "join_rows": None, "listing_rows": None,
        }]), "vPIC VINs are not unique; downstream VIN-linked checks were skipped."

    listing_vins = _collect(
        listings.select(
            pl.len().alias("listing_rows"),
            pl.col("vin").n_unique().alias("distinct_vins"),
            pl.col("listing_id").n_unique().alias("distinct_listing_ids")
            if "listing_id" in listing_schema else pl.lit(None).alias("distinct_listing_ids"),
        )
    ).to_dicts()[0]
    distinct_listing_vins = int(listing_vins["distinct_vins"])
    vpic_vins = int(_collect(vpic.select(pl.col("vin").n_unique().alias("n"))).item())

    vpic_match = vpic.select(
        "vin", *COHORT_COLUMNS,
        *(["ErrorCode"] if "ErrorCode" in vpic_schema else []),
        pl.lit(True).alias("_vpic_match"),
    )
    used_join_columns = [column for column in (
        "vin", "price", "mileage", "listed_date", "daysonmarket", "year",
        "make_name", "model_name", "trim_name", "is_new", "listing_id",
    ) if column in listing_schema]
    joined = listings.select(used_join_columns).join(
        vpic_match, on="vin", how="left", validate="m:1", maintain_order="left"
    )
    report_exprs: list[pl.Expr] = [
        pl.len().alias("join_rows"),
        pl.col("_vpic_match").is_not_null().sum().alias("matched_listing_rows"),
        pl.col("_vpic_match").is_null().sum().alias("unmatched_listing_rows"),
    ]
    for column in COHORT_COLUMNS:
        report_exprs.append(
            (pl.col(column).is_null() | (pl.col(column).cast(pl.String).str.strip_chars() == ""))
            .fill_null(True).sum().alias(f"missing_{column}_rows")
        )
    identity_checks = []
    for listing_column, vpic_column in (
        ("year", "ModelYear"), ("make_name", "Make"), ("model_name", "Model"),
    ):
        if listing_column not in listing_schema:
            continue
        left = (
            pl.col(listing_column).cast(pl.String).str.strip_chars().str.to_uppercase()
            .str.replace_all(r"\s+", " ")
        )
        right = _normalized_string(vpic_column)
        mismatch = left.is_not_null() & (left != "") & right.is_not_null() & (right != "") & (left != right)
        report_exprs.append(mismatch.sum().alias(f"{listing_column}_vpic_disagreement_rows"))
        identity_checks.append(mismatch)
    linked_counts = _collect(joined.with_columns(_scope_expression()).select(
        *report_exprs,
        pl.col("vehicle_scope").eq("used").sum().alias("used_rows"),
    ))
    summary = pl.DataFrame([{
        "check": "listing to vPIC VIN join", "status": "passed",
        "listing_rows": int(listing_vins["listing_rows"]),
        "distinct_listing_vins": distinct_listing_vins,
        "distinct_listing_ids": listing_vins["distinct_listing_ids"],
        "vpic_distinct_vins": vpic_vins,
        **linked_counts.to_dicts()[0],
        "join_row_expansion": int(linked_counts["join_rows"][0]) - int(listing_vins["listing_rows"]),
    }])

    if identity_checks:
        mismatch_any = identity_checks[0]
        for check in identity_checks[1:]:
            mismatch_any = mismatch_any | check
        sample = _collect(
            joined.with_columns(_scope_expression())
            .filter(mismatch_any & (pl.col("vehicle_scope") == "used"))
            .select([column for column in (
                "vin", "listing_id", "is_new", "year", "ModelYear", "make_name",
                "Make", "model_name", "Model", "Trim", "ErrorCode",
            ) if column in joined.collect_schema().names()])
            .head(500)
        )
        _write_csv(sample, output_dir / "used_listing_vpic_name_mismatch_sample.csv")
    return joined, summary, None


def _date_profile(
    listings: pl.LazyFrame, listing_schema: dict[str, pl.DataType], output_dir: Path
) -> tuple[pl.DataFrame, bool]:
    if "listed_date" not in listing_schema:
        return pl.DataFrame([{"check": "listing date proxy", "status": "not run",
                              "detail": "listed_date is missing."}]), False
    listed_date = _date_expression("listed_date", listing_schema)
    expressions: list[pl.Expr] = [
        pl.len().alias("rows"), listed_date.min().alias("listed_date_min"),
        listed_date.max().alias("listed_date_max"),
        listed_date.is_null().sum().alias("unparsed_or_missing_listed_date_rows"),
    ]
    proxy: pl.Expr | None = None
    if "daysonmarket" in listing_schema:
        days = pl.col("daysonmarket").cast(pl.Int64, strict=False)
        proxy = (listed_date + pl.duration(days=days)).alias("observation_date_proxy")
        expressions.extend([
            days.min().alias("daysonmarket_min"), days.max().alias("daysonmarket_max"),
            days.is_null().sum().alias("unparsed_or_missing_daysonmarket_rows"),
            proxy.min().alias("observation_date_proxy_min"),
            proxy.max().alias("observation_date_proxy_max"),
        ])
    base = listings.with_columns(_scope_expression(), listed_date.alias("_listed_date"))
    if proxy is not None:
        base = base.with_columns(proxy)
    date_summary = _collect(base.group_by("vehicle_scope").agg(expressions).sort("vehicle_scope"))
    _write_csv(date_summary, output_dir / "listing_date_profile.csv")
    return pl.DataFrame([{
        "check": "listing date proxy", "status": "profiled",
        "detail": "listed_date + daysonmarket is a proxy, not a verified price capture timestamp."
                 if proxy is not None else "Only listed_date is available; no capture-date proxy was calculated.",
    }]), proxy is not None


def _source_date_profile(
    table_name: str, filename: str, date_column: str,
    data_dir: Path,
) -> pl.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        return pl.DataFrame([{"source": table_name, "date_field": date_column,
                              "status": "missing file", "rows": None}])
    source = pl.scan_parquet(path)
    schema = source.collect_schema()
    if date_column not in schema:
        return pl.DataFrame([{"source": table_name, "date_field": date_column,
                              "status": "missing field", "rows": None}])
    date_value = _date_expression(date_column, schema)
    stats = _collect(source.select(
        pl.len().alias("rows"), date_value.min().alias("date_min"),
        date_value.max().alias("date_max"), date_value.is_null().sum().alias("missing_date_rows"),
    )).to_dicts()[0]
    return pl.DataFrame([{
        "source": table_name, "date_field": date_column, "status": "profiled", **stats,
    }])


def _issue_availability(
    joined: pl.LazyFrame | None, joined_error: str | None,
    data_dir: Path, output_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    profiles = pl.concat([
        _source_date_profile("nhtsa_complaints", PARQUET_FILES["nhtsa_complaints"],
                             "dateComplaintFiled", data_dir),
        _source_date_profile("nhtsa_recalls", PARQUET_FILES["nhtsa_recalls"],
                             "ReportReceivedDate", data_dir),
    ], how="diagonal_relaxed")
    _write_csv(profiles, output_dir / "nhtsa_event_date_profile.csv")

    if joined is None:
        return profiles, pl.DataFrame([{
            "source": "complaints and recalls", "status": "not run",
            "reason": joined_error or "VIN-linked listing/vPIC join is unavailable.",
        }])
    joined_schema = joined.collect_schema()
    required_proxy = {"observation_date_proxy", "ModelYear", "Make", "Model"}
    if not required_proxy.issubset(joined_schema.names()):
        return profiles, pl.DataFrame([{
            "source": "complaints and recalls", "status": "not run",
            "reason": "The join does not have the date proxy and vPIC year/make/model keys.",
        }])

    vehicle_rows = joined.with_columns(_scope_expression()).filter(
        (pl.col("vehicle_scope") == "used") & pl.col("observation_date_proxy").is_not_null()
    ).with_columns(
        pl.col("ModelYear").cast(pl.Int64, strict=False).alias("_year"),
        _normalized_string("Make").alias("_make"),
        _normalized_string("Model").alias("_model"),
    )
    timing_rows: list[dict[str, Any]] = []
    sources = (
        ("nhtsa_complaints", "dateComplaintFiled"),
        ("nhtsa_recalls", "ReportReceivedDate"),
    )
    for table_name, date_column in sources:
        path = data_dir / PARQUET_FILES[table_name]
        if not path.is_file():
            timing_rows.append({"source": table_name, "status": "missing file"})
            continue
        source = pl.scan_parquet(path)
        schema = source.collect_schema()
        keys = ("model_year", "make", "model")
        required = set(keys) | {date_column}
        if not required.issubset(schema.names()):
            timing_rows.append({"source": table_name, "status": "missing required fields",
                                "detail": ", ".join(sorted(required - set(schema.names())))})
            continue
        source_max = (
            source.select(
                pl.col("model_year").cast(pl.Int64, strict=False).alias("_year"),
                _normalized_string("make").alias("_make"),
                _normalized_string("model").alias("_model"),
                _date_expression(date_column, schema).alias("_event_date"),
            )
            .filter(pl.col("_event_date").is_not_null())
            .group_by("_year", "_make", "_model")
            .agg(pl.col("_event_date").max().alias("_latest_event_date"))
        )
        has_later = (
            vehicle_rows.join(source_max, on=["_year", "_make", "_model"],
                              how="left", validate="m:1")
            .with_columns(
                (pl.col("_latest_event_date") > pl.col("observation_date_proxy"))
                .fill_null(False).alias("has_post_proxy_records")
            )
        )
        result = _collect(has_later.select(
            pl.len().alias("used_rows_with_proxy"),
            pl.col("_latest_event_date").is_not_null().sum().alias("rows_with_source_records"),
            pl.col("has_post_proxy_records").sum().alias("rows_with_later_source_records"),
            pl.col("_latest_event_date").max().alias("latest_source_date_in_joined_keys"),
        )).to_dicts()[0]
        timing_rows.append({
            "source": table_name, "status": "profiled against date proxy",
            "date_field": date_column, **result,
            "interpretation": "Potential future-information exposure only; the proxy is not a verified price timestamp.",
        })
    return profiles, pl.DataFrame(timing_rows)


def _review_candidates(
    joined: pl.LazyFrame | None, listing_schema: dict[str, pl.DataType],
    output_dir: Path, *, price_low: float, price_high: float,
    mileage_high: float, peer_ratio: float, peer_min_count: int,
) -> pl.DataFrame:
    if joined is None:
        return pl.DataFrame([{"status": "not run", "reason": "VIN-linked vPIC join unavailable."}])
    schema = joined.collect_schema()
    if not {"price", "mileage"}.issubset(schema.names()):
        return pl.DataFrame([{"status": "not run", "reason": "price or mileage field is missing."}])
    family_keys = ["ModelYear", "Make", "Model"]
    can_peer_compare = set(family_keys).issubset(schema.names())
    used = joined.with_columns(_scope_expression()).filter(pl.col("vehicle_scope") == "used")
    if can_peer_compare:
        peers = (
            used.filter(pl.all_horizontal([
                pl.col(c).is_not_null()
                & (pl.col(c).cast(pl.String).str.strip_chars() != "")
                for c in family_keys
            ]))
            .group_by(family_keys)
            .agg(
                pl.col("price").median().alias("peer_family_price_median"),
                pl.len().alias("peer_family_listing_count"),
            )
        )
        used = used.join(peers, on=family_keys, how="left", validate="m:1")
        ratio = pl.col("price") / pl.col("peer_family_price_median")
        peer_flag = (pl.col("peer_family_listing_count") >= peer_min_count) & (ratio >= peer_ratio)
        used = used.with_columns(ratio.alias("price_to_peer_family_median"), peer_flag.alias("peer_price_review_flag"))
    else:
        used = used.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("price_to_peer_family_median"),
            pl.lit(False).alias("peer_price_review_flag"),
        )
    low = (pl.col("price") < price_low).fill_null(False)
    high = (pl.col("price") > price_high).fill_null(False)
    mileage_invalid = ((pl.col("mileage") < 0) | (pl.col("mileage") > mileage_high)).fill_null(False)
    zero_mileage = (pl.col("mileage") == 0).fill_null(False)
    reasons = pl.concat_str([
        pl.when(pl.col("price_below_threshold_flag")).then(pl.lit("price_below_threshold")).otherwise(None),
        pl.when(pl.col("price_above_threshold_flag")).then(pl.lit("price_above_threshold")).otherwise(None),
        pl.when(pl.col("peer_price_review_flag")).then(pl.lit("high_vs_peer_family")).otherwise(None),
        pl.when(pl.col("mileage_out_of_range_flag")).then(pl.lit("mileage_out_of_range")).otherwise(None),
        pl.when(pl.col("zero_mileage_review_flag")).then(pl.lit("zero_mileage_check")).otherwise(None),
    ], separator=";", ignore_nulls=True)
    used = used.with_columns(
        low.alias("price_below_threshold_flag"), high.alias("price_above_threshold_flag"),
        mileage_invalid.alias("mileage_out_of_range_flag"),
        zero_mileage.alias("zero_mileage_review_flag"),
    ).with_columns(
        pl.when(reasons == "").then(None).otherwise(reasons).alias("review_reasons")
    )
    candidates = used.filter(pl.col("review_reasons").is_not_null())
    select_columns = [column for column in (
        "vin", "listing_id", "price", "mileage", "year", *COHORT_COLUMNS,
        "make_name", "model_name", "trim_name", "listed_date", "daysonmarket",
        "peer_family_price_median", "peer_family_listing_count", "price_to_peer_family_median",
        "price_below_threshold_flag", "price_above_threshold_flag", "peer_price_review_flag",
        "mileage_out_of_range_flag", "zero_mileage_review_flag", "review_reasons",
    ) if column in candidates.collect_schema().names()]
    candidates_df = _collect(candidates.select(select_columns))
    _write_csv(candidates_df, output_dir / "used_listing_review_candidates.csv")
    summary = candidates_df.select(
        pl.len().alias("candidate_rows"),
        pl.col("price_below_threshold_flag").sum().alias("price_below_threshold_rows"),
        pl.col("price_above_threshold_flag").sum().alias("price_above_threshold_rows"),
        pl.col("peer_price_review_flag").sum().alias("high_vs_peer_family_rows"),
        pl.col("mileage_out_of_range_flag").sum().alias("mileage_out_of_range_rows"),
        pl.col("zero_mileage_review_flag").sum().alias("zero_mileage_review_rows"),
    ) if not candidates_df.is_empty() else pl.DataFrame([{
        "candidate_rows": 0, "price_below_threshold_rows": 0,
        "price_above_threshold_rows": 0, "high_vs_peer_family_rows": 0,
        "mileage_out_of_range_rows": 0, "zero_mileage_review_rows": 0,
    }])
    return summary.with_columns(
        pl.lit("review heuristics only; no rows excluded or changed").alias("policy")
    )


def run_diagnostics(
    data_dir: Path = FINAL_DATASET_PATH,
    output_dir: Path | None = None,
    *, price_low: float = 500, price_high: float = 500_000,
    mileage_high: float = 1_000_000, peer_ratio: float = 10,
    peer_min_count: int = 20, rare_count: int = 20,
) -> Path:
    """Write bounded diagnostic reports without mutating source data."""
    data_dir = Path(data_dir).resolve()
    if output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DEFAULT_OUTPUT_ROOT / run_id
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    inventory = _table_inventory(data_dir)
    _write_csv(inventory, output_dir / "source_inventory.csv")
    _write_csv(_manifest_check(data_dir, inventory), output_dir / "export_manifest_check.csv")
    required_files = {"used_cars", "nhtsa_vpic_decodes"}
    missing_required = sorted(
        required_files - set(inventory.filter(pl.col("exists"))["table"].to_list())
    )
    if missing_required:
        raise FileNotFoundError(
            f"Required Parquet tables are missing in {data_dir}: {', '.join(missing_required)}. "
            f"Source inventory was saved to {output_dir}."
        )

    listings_path = data_dir / PARQUET_FILES["used_cars"]
    vpic_path = data_dir / PARQUET_FILES["nhtsa_vpic_decodes"]
    listings = pl.scan_parquet(listings_path)
    listing_schema = listings.collect_schema()
    for column in ("vin", "price", "is_new"):
        if column not in listing_schema:
            raise ValueError(f"used_cars.parquet is missing required diagnostic field {column!r}.")

    scope_summary = _scope_summary(listings, listing_schema)
    _write_csv(scope_summary, output_dir / "listing_scope_and_value_summary.csv")
    _write_csv(_listing_key_quality(listings, listing_schema), output_dir / "listing_key_quality.csv")
    missingness = _missingness(listings, listing_schema)
    _write_csv(missingness, output_dir / "base_feature_missingness.csv")
    cardinality = _category_cardinality(listings, listing_schema, rare_count)
    _write_csv(cardinality, output_dir / "used_category_cardinality.csv")

    joined, join_summary, join_error = _vpic_join(
        listings, listing_schema, vpic_path, output_dir
    )
    _write_csv(join_summary, output_dir / "vin_join_and_identity_summary.csv")
    date_profile, has_date_proxy = _date_profile(listings, listing_schema, output_dir)
    if joined is not None and has_date_proxy:
        joined_schema = joined.collect_schema()
        listed_date = _date_expression("listed_date", joined_schema)
        days = pl.col("daysonmarket").cast(pl.Int64, strict=False)
        joined = joined.with_columns(
            (listed_date + pl.duration(days=days)).alias("observation_date_proxy")
        )
    _write_csv(date_profile, output_dir / "date_proxy_notes.csv")

    issue_profile, timing = _issue_availability(joined, join_error, data_dir, output_dir)
    _write_csv(timing, output_dir / "issue_availability_vs_date_proxy.csv")
    review = _review_candidates(
        joined, listing_schema, output_dir, price_low=price_low,
        price_high=price_high, mileage_high=mileage_high,
        peer_ratio=peer_ratio, peer_min_count=peer_min_count,
    )
    _write_csv(review, output_dir / "used_label_and_mileage_review_summary.csv")

    safety_path = data_dir / PARQUET_FILES["nhtsa_safety_rating_values"]
    safety_note: list[dict[str, Any]] = []
    if safety_path.is_file():
        safety = pl.scan_parquet(safety_path)
        safety_schema = safety.collect_schema()
        if "OverallRating" in safety_schema:
            safety_note = _collect(safety.select(
                pl.len().alias("rows"), pl.col("OverallRating").is_null().sum().alias("missing_overall_rating_rows"),
                pl.col("ModelYear").min().alias("minimum_model_year")
                if "ModelYear" in safety_schema else pl.lit(None).alias("minimum_model_year"),
                pl.col("ModelYear").max().alias("maximum_model_year")
                if "ModelYear" in safety_schema else pl.lit(None).alias("maximum_model_year"),
            )).to_dicts()
    _write_csv(pl.DataFrame(safety_note or [{
        "status": "safety table missing or OverallRating unavailable",
    }]), output_dir / "safety_source_profile.csv")

    run_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(data_dir),
        "output_directory": str(output_dir),
        "source_parquets_read_only": True,
        "review_rows_excluded": False,
        "listing_grain_assumption": "one source listing row; VIN is a join key, not a predictor",
        "cohort_dimensions": list(COHORT_COLUMNS),
        "vpic_join_key": "lowercase vin",
        "date_proxy": "listed_date + daysonmarket; not a verified price observation timestamp",
        "review_thresholds": {
            "price_below": price_low, "price_above": price_high,
            "mileage_above": mileage_high, "peer_ratio_at_least": peer_ratio,
            "minimum_peer_family_rows": peer_min_count, "rare_category_count_at_most": rare_count,
        },
        "join_issue": join_error,
        "reports": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "readiness_note": (
            "This is a data review only. It does not approve training eligibility, "
            "confirm source timestamps, or resolve suspicious rows."
        ),
    }
    (output_dir / "diagnostic_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create read-only data-quality reports before price-model training."
    )
    parser.add_argument("--data-dir", type=Path, default=FINAL_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; defaults to MODEL_OUTPUT/diagnostics/<UTC run id>.")
    parser.add_argument("--price-low", type=float, default=500,
                        help="Review listings below this price; does not filter them.")
    parser.add_argument("--price-high", type=float, default=500_000,
                        help="Review listings above this price; does not filter them.")
    parser.add_argument("--mileage-high", type=float, default=1_000_000,
                        help="Review mileage above this value or below zero; does not filter rows.")
    parser.add_argument("--peer-ratio", type=float, default=10,
                        help="Review price / year-make-model peer median at or above this ratio.")
    parser.add_argument("--peer-min-count", type=int, default=20,
                        help="Minimum used listings in the peer group before ratio review.")
    parser.add_argument("--rare-count", type=int, default=20,
                        help="Summarize category levels at or below this used-row count.")
    args = parser.parse_args()
    if args.price_low < 0 or args.price_high <= args.price_low:
        parser.error("price thresholds must satisfy 0 <= --price-low < --price-high")
    if args.mileage_high <= 0 or args.peer_ratio <= 0 or args.peer_min_count < 1 or args.rare_count < 1:
        parser.error("mileage, peer ratio, peer minimum, and rare count must be positive")
    output = run_diagnostics(
        data_dir=args.data_dir, output_dir=args.output_dir,
        price_low=args.price_low, price_high=args.price_high,
        mileage_high=args.mileage_high, peer_ratio=args.peer_ratio,
        peer_min_count=args.peer_min_count, rare_count=args.rare_count,
    )
    print(f"Diagnostics saved to: {output}")
    for path in sorted(output.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
