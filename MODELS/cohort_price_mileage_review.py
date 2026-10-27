"""List used listings with price or mileage far from their vPIC cohort median.

The review grain is the four separate vPIC columns ``ModelYear``, ``Make``,
``Model``, and ``Trim``. Listings join to the lowercase ``vin`` column. The
script is read-only with respect to source Parquet and never filters rows.

Run from the project root::

    python -m MODELS.cohort_price_mileage_review
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import polars as pl

from EDA.DATA_CLEANING import FINAL_DATASET_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MODEL_OUTPUT" / "diagnostics" / "cohort_outliers"
COHORT_COLUMNS = ("ModelYear", "Make", "Model", "Trim")


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


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


def _normalized_cohort_exprs() -> list[pl.Expr]:
    year = pl.col("ModelYear").cast(pl.Int64, strict=False).alias("ModelYear")
    return [
        year,
        pl.col("Make").cast(pl.String).str.strip_chars().str.to_uppercase()
        .str.replace_all(r"\s+", " ").alias("Make"),
        pl.col("Model").cast(pl.String).str.strip_chars().str.to_uppercase()
        .str.replace_all(r"\s+", " ").alias("Model"),
        pl.col("Trim").cast(pl.String).str.strip_chars().str.to_uppercase()
        .str.replace_all(r"\s+", " ").alias("Trim"),
    ]


def _cohort_complete_expression() -> pl.Expr:
    return pl.all_horizontal([
        pl.col("ModelYear").is_not_null(),
        *[
            pl.col(column).is_not_null() & (pl.col(column).cast(pl.String).str.strip_chars() != "")
            for column in COHORT_COLUMNS[1:]
        ],
    ])


def _outlier_expressions(price_multiple: float, mileage_multiple: float,
                         iqr_multiple: float) -> list[pl.Expr]:
    price = pl.col("price")
    price_median = pl.col("price_median")
    mileage = pl.col("mileage")
    mileage_median = pl.col("mileage_median")
    mileage_iqr = pl.col("mileage_q75") - pl.col("mileage_q25")
    price_enough = pl.col("price_value_count") >= pl.col("minimum_cohort_size")
    mileage_enough = pl.col("mileage_value_count") >= pl.col("minimum_cohort_size")

    price_flag = (
        price_enough & price.is_not_null() & (price > 0) & (price_median > 0)
        & ((price >= price_median * price_multiple) | (price <= price_median / price_multiple))
    ).fill_null(False)
    mileage_ratio_flag = (
        mileage_enough & mileage.is_not_null() & (mileage >= 0) & (mileage_median > 0)
        & ((mileage >= mileage_median * mileage_multiple)
           | (mileage <= mileage_median / mileage_multiple))
    ).fill_null(False)
    mileage_zero_median_flag = (
        mileage_enough & mileage.is_not_null() & (mileage >= 0)
        & (mileage_median == 0) & (mileage > pl.col("mileage_q75") + iqr_multiple * mileage_iqr)
    ).fill_null(False)
    return [
        price_flag.alias("price_median_multiple_flag"),
        mileage_ratio_flag.alias("mileage_median_multiple_flag"),
        mileage_zero_median_flag.alias("mileage_zero_median_iqr_flag"),
        ((price.is_not_null()) & (price <= 0)).fill_null(False).alias("nonpositive_price_flag"),
        ((mileage.is_not_null()) & (mileage < 0)).fill_null(False).alias("negative_mileage_flag"),
        pl.when(price_median > 0).then(price / price_median)
        .otherwise(None).alias("price_to_cohort_median"),
        pl.when(mileage_median > 0).then(mileage / mileage_median)
        .otherwise(None).alias("mileage_to_cohort_median"),
        (price - price_median).alias("price_minus_cohort_median"),
        (mileage - mileage_median).alias("mileage_minus_cohort_median"),
    ]


def run_review(
    data_dir: Path = FINAL_DATASET_PATH,
    output_dir: Path | None = None,
    *, scope: str = "used", price_multiple: float = 3.0,
    mileage_multiple: float = 3.0, iqr_multiple: float = 3.0,
    min_cohort_size: int = 5,
) -> Path:
    if scope not in {"used", "new", "all"}:
        raise ValueError("scope must be 'used', 'new', or 'all'")
    if min(price_multiple, mileage_multiple, iqr_multiple) <= 1:
        raise ValueError("outlier multiples must be greater than 1")
    if min_cohort_size < 2:
        raise ValueError("min_cohort_size must be at least 2")

    data_dir = Path(data_dir).resolve()
    if output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = DEFAULT_OUTPUT_ROOT / run_id
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    listings_path = data_dir / "used_cars.parquet"
    vpic_path = data_dir / "NHTSA_vPIC.parquet"
    for path in (listings_path, vpic_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}")

    listings = pl.scan_parquet(listings_path)
    vpic = pl.scan_parquet(vpic_path)
    listing_schema = listings.collect_schema()
    vpic_schema = vpic.collect_schema()
    required_listing = {"vin", "price", "mileage", "is_new"}
    required_vpic = {"vin", *COHORT_COLUMNS}
    missing_listing = sorted(required_listing - set(listing_schema.names()))
    missing_vpic = sorted(required_vpic - set(vpic_schema.names()))
    if missing_listing or missing_vpic:
        details = []
        if missing_listing:
            details.append("used_cars missing: " + ", ".join(missing_listing))
        if missing_vpic:
            details.append("NHTSA_vPIC missing: " + ", ".join(missing_vpic))
        raise ValueError("; ".join(details))

    duplicate_vpic = _collect(vpic.group_by("vin").len().filter(pl.col("len") > 1))
    if not duplicate_vpic.is_empty():
        duplicate_vpic.head(500).write_csv(output_dir / "duplicate_vpic_vins_sample.csv")
        raise ValueError(
            f"vPIC has {duplicate_vpic.height} duplicate VIN keys. Review the sample in {output_dir}; "
            "the listing join was stopped to avoid multiplying rows."
        )

    listing_columns = [column for column in (
        "vin", "listing_id", "price", "mileage", "year", "make_name",
        "model_name", "trim_name", "is_new",
    ) if column in listing_schema]
    decoded = vpic.select("vin", *COHORT_COLUMNS).with_columns(pl.lit(True).alias("_vpic_match"))
    joined = (
        listings.select(listing_columns)
        .with_columns(_scope_expression())
        .join(decoded, on="vin", how="left", validate="m:1", maintain_order="left")
        .with_columns(_normalized_cohort_exprs())
        .with_columns(_cohort_complete_expression().alias("cohort_complete"))
    )
    if scope != "all":
        joined = joined.filter(pl.col("vehicle_scope") == scope)

    # Keep new, used, and unknown listing scopes from sharing a median when
    # --scope all is selected. Vehicle scope remains a separate output field.
    cohort_keys = ["vehicle_scope", *COHORT_COLUMNS]
    valid = joined.filter(pl.col("cohort_complete"))
    medians = valid.group_by(cohort_keys).agg(
        pl.len().alias("cohort_listing_count"),
        pl.col("price").filter(pl.col("price") > 0).count().alias("price_value_count"),
        pl.col("price").filter(pl.col("price") > 0).median().alias("price_median"),
        pl.col("price").filter(pl.col("price") > 0).quantile(0.25).alias("price_q25"),
        pl.col("price").filter(pl.col("price") > 0).quantile(0.75).alias("price_q75"),
        pl.col("mileage").filter(pl.col("mileage") >= 0).count().alias("mileage_value_count"),
        pl.col("mileage").filter(pl.col("mileage") >= 0).median().alias("mileage_median"),
        pl.col("mileage").filter(pl.col("mileage") >= 0).quantile(0.25).alias("mileage_q25"),
        pl.col("mileage").filter(pl.col("mileage") >= 0).quantile(0.75).alias("mileage_q75"),
    ).with_columns(pl.lit(min_cohort_size).alias("minimum_cohort_size"))

    scored = valid.join(medians, on=cohort_keys, how="left", validate="m:1")
    scored = scored.with_columns(_outlier_expressions(price_multiple, mileage_multiple, iqr_multiple))
    scored = scored.with_columns(
        pl.concat_str([
            pl.when(pl.col("price_median_multiple_flag"))
            .then(pl.lit("price_far_from_median")).otherwise(None),
            pl.when(pl.col("mileage_median_multiple_flag"))
            .then(pl.lit("mileage_far_from_median")).otherwise(None),
            pl.when(pl.col("mileage_zero_median_iqr_flag"))
            .then(pl.lit("mileage_far_from_zero_median")).otherwise(None),
            pl.when(pl.col("nonpositive_price_flag"))
            .then(pl.lit("nonpositive_price")).otherwise(None),
            pl.when(pl.col("negative_mileage_flag"))
            .then(pl.lit("negative_mileage")).otherwise(None),
        ], separator=";", ignore_nulls=True).alias("review_reasons")
    ).filter(pl.col("review_reasons").is_not_null() & (pl.col("review_reasons") != ""))

    output_columns = [column for column in (
        "vin", "listing_id", "vehicle_scope", *COHORT_COLUMNS,
        "cohort_listing_count", "price_value_count", "price", "price_median",
        "price_q25", "price_q75", "price_to_cohort_median", "price_minus_cohort_median",
        "mileage_value_count", "mileage", "mileage_median", "mileage_q25", "mileage_q75",
        "mileage_to_cohort_median", "mileage_minus_cohort_median",
        "price_median_multiple_flag", "mileage_median_multiple_flag",
        "mileage_zero_median_iqr_flag", "nonpositive_price_flag", "negative_mileage_flag",
        "review_reasons", "year", "make_name", "model_name", "trim_name",
    ) if column in scored.collect_schema().names()]
    cleaning_candidate = (
        (pl.col("price_to_cohort_median") > 2.0)
        | (pl.col("price_to_cohort_median") <= 0.75)
        | (pl.col("mileage_to_cohort_median") > 20.0)
    ).fill_null(False)
    scored = scored.with_columns(
        cleaning_candidate.alias("cleaning_exclusion_flag"),
        pl.when(pl.col("price_to_cohort_median") > 2.0).then(pl.lit("price_ratio_gt_2"))
        .when(pl.col("price_to_cohort_median") <= 0.75).then(pl.lit("price_ratio_le_0_75"))
        .when(pl.col("mileage_to_cohort_median") > 20.0).then(pl.lit("mileage_ratio_gt_20"))
        .otherwise(None).alias("cleaning_exclusion_reason"),
    )
    candidates = _collect(scored.filter(cleaning_candidate).select(
        *output_columns, "cleaning_exclusion_flag", "cleaning_exclusion_reason"
    ).sort(
        ["price_median_multiple_flag", "mileage_median_multiple_flag", "price"],
        descending=[True, True, True], nulls_last=True,
    ))
    candidates.write_csv(output_dir / "cohort_outlier_candidates.csv")
    medians_df = _collect(medians.sort(cohort_keys))
    medians_df.write_csv(output_dir / "cohort_medians.csv")
    small_groups = medians_df.filter(
        (pl.col("price_value_count") < min_cohort_size)
        | (pl.col("mileage_value_count") < min_cohort_size)
    )
    small_groups.write_csv(output_dir / "small_cohort_groups.csv")

    unscored = joined.filter(~pl.col("cohort_complete")).with_columns(
        pl.when(pl.col("_vpic_match").is_null()).then(pl.lit("no_vpic_vin_match"))
        .otherwise(pl.lit("incomplete_vpic_cohort")).alias("unscored_reason")
    )
    unscored_columns = [column for column in (
        "vin", "listing_id", "vehicle_scope", *COHORT_COLUMNS, "price", "mileage",
        "year", "make_name", "model_name", "trim_name", "unscored_reason",
    ) if column in unscored.collect_schema().names()]
    unscored_df = _collect(unscored.select(unscored_columns))
    unscored_df.write_csv(output_dir / "unscored_incomplete_cohorts.csv")

    scope_counts = _collect(joined.group_by("vehicle_scope").len().sort("vehicle_scope"))
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(data_dir),
        "output_directory": str(output_dir),
        "scope": scope,
        "cohort_columns": list(COHORT_COLUMNS),
        "join_key": "lowercase vin",
        "price_outlier_rule": (
            f"positive price at least {price_multiple:g}x above or below its cohort median; "
            f"at least {min_cohort_size} valid price rows"
        ),
        "mileage_outlier_rule": (
            f"nonnegative mileage at least {mileage_multiple:g}x above or below its cohort median; "
            f"at least {min_cohort_size} valid mileage rows; when median is zero, flag above Q3 + "
            f"{iqr_multiple:g}x IQR"
        ),
        "candidate_rows": candidates.height,
        "cleaning_exclusion_policy": "price/cohort median > 2 or <= 0.75, or mileage/cohort median > 20; missing mileage is separately excluded by export",
        "unscored_incomplete_cohort_rows": unscored_df.height,
        "cohort_groups": medians_df.height,
        "small_cohort_groups": small_groups.height,
        "listing_scope_counts": scope_counts.to_dicts(),
        "source_parquets_read_only": True,
        "rows_filtered_or_modified": False,
        "note": "Candidates are for human review; medians are descriptive and do not determine validity.",
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find listing price or mileage values far from their four-field vPIC cohort medians."
    )
    parser.add_argument("--data-dir", type=Path, default=FINAL_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; defaults below MODEL_OUTPUT/diagnostics/cohort_outliers.")
    parser.add_argument("--scope", choices=("used", "new", "all"), default="used",
                        help="Vehicle scope to compare; default: used.")
    parser.add_argument("--price-multiple", type=float, default=3.0,
                        help="Flag price at or beyond this factor above/below its cohort median.")
    parser.add_argument("--mileage-multiple", type=float, default=3.0,
                        help="Flag mileage at or beyond this factor above/below its cohort median.")
    parser.add_argument("--iqr-multiple", type=float, default=3.0,
                        help="For zero-median mileage cohorts, flag above Q3 + this times the IQR.")
    parser.add_argument("--min-cohort-size", type=int, default=5,
                        help="Minimum valid price or mileage rows needed to score that measure.")
    args = parser.parse_args()
    try:
        output = run_review(
            data_dir=args.data_dir, output_dir=args.output_dir, scope=args.scope,
            price_multiple=args.price_multiple, mileage_multiple=args.mileage_multiple,
            iqr_multiple=args.iqr_multiple, min_cohort_size=args.min_cohort_size,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Cohort review saved to: {output}")
    for path in sorted(output.iterdir()):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
