"""Read-only, grain-aware access to the documented Parquet snapshot."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from functools import reduce
import hashlib
import json
from pathlib import Path

import polars as pl
import streamlit as st

from EDA.DATA_CLEANING import EXCLUDED_LISTING_VINS

ROOT = Path(__file__).resolve().parents[1]
COHORT = ["ModelYear", "Make", "Model", "Trim"]
FAMILY = COHORT[:3]
MIN_MAKE_ROWS = 10
LISTING_COLUMNS = [
    "vin", "listing_id", "price", "mileage", "is_new", "body_type", "fuel_type",
    "wheel_system", "transmission", "maximum_seating", "horsepower",
    "city_fuel_economy", "highway_fuel_economy", "has_accidents", "frame_damaged",
    "salvage", "theft_title", "major_options", "make_name", "model_name",
    "trim_name", "year", "listed_date", "city", "daysonmarket",
]
TABLE_NAMES = {
    "used_cars": "used_cars", "NHTSA_vPIC": "nhtsa_vpic_decodes",
    "NHTSA_Complaints": "nhtsa_complaints", "NHTSA_Recalls": "nhtsa_recalls",
    "NHTSA_Safety": "nhtsa_safety_rating_values",
    "Youtube_Comments": "youtube_comments_sentiment",
}


@dataclass(frozen=True)
class Filters:
    makes: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    trims: tuple[str, ...] = ()
    body_types: tuple[str, ...] = ()
    fuels: tuple[str, ...] = ()
    drivetrains: tuple[str, ...] = ()
    condition: str = "All"
    min_price: float | None = None
    max_price: float | None = None
    max_mileage: float | None = None

    def active(self):
        return {k: v for k, v in asdict(self).items()
                if v not in ((), None, "All")}

    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def filter_expression(filters: Filters):
    expressions = []
    for key, column in [
        ("makes", "Make"), ("models", "Model"), ("years", "ModelYear"),
        ("trims", "Trim"), ("body_types", "body_type"), ("fuels", "fuel_type"),
        ("drivetrains", "wheel_system"),
    ]:
        values = getattr(filters, key)
        if values:
            expressions.append(pl.col(column).is_in(values))
    if filters.min_price is not None:
        expressions.append(pl.col("price") >= filters.min_price)
    if filters.max_price is not None:
        expressions.append(pl.col("price") <= filters.max_price)
    if filters.max_mileage is not None:
        expressions.append(pl.col("mileage") <= filters.max_mileage)
    if filters.condition != "All":
        expressions.append(pl.col("is_new") == ("True" if filters.condition == "New" else "False"))
    return reduce(lambda a, b: a & b, expressions, pl.lit(True)).fill_null(False)


def normalize_family(frame: pl.LazyFrame, columns=FAMILY):
    return frame.with_columns(
        pl.col(columns[0]).cast(pl.Int64, strict=False).alias("_year"),
        *[pl.col(source).cast(pl.String).str.strip_chars().str.to_uppercase()
          .str.replace_all(r"\s+", " ").alias(target)
          for source, target in zip(columns[1:], ["_make", "_model"])],
    )


class Snapshot:
    def __init__(self, directory: Path | str = ROOT / "DATA/CAR_DATA_FINAL"):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8-sig"))
        signature = []
        for table, key in TABLE_NAMES.items():
            path = self.directory / f"{table}.parquet"
            entry = self.manifest["tables"][key]
            actual = pl.read_parquet_schema(path)
            if list(actual) != list(entry["schema"]):
                raise ValueError(f"{table}: snapshot columns differ from manifest")
            for name, dtype in actual.items():
                if str(dtype) != entry["schema"][name]:
                    raise ValueError(f"{table}.{name}: snapshot type differs from manifest")
            stat = path.stat()
            signature.append((table, stat.st_size, stat.st_mtime_ns))
        self.identity = hashlib.sha256(json.dumps(
            {"tables": signature, "excluded_listing_vins": sorted(EXCLUDED_LISTING_VINS)}
        ).encode()).hexdigest()

    def scan(self, table):
        if table not in TABLE_NAMES:
            raise ValueError("Unsupported snapshot table")
        return pl.scan_parquet(self.directory / f"{table}.parquet")

    def catalog(self, filters=Filters()):
        listings = self.scan("used_cars").filter(
            ~pl.col("vin").is_in(sorted(EXCLUDED_LISTING_VINS))).select(LISTING_COLUMNS)
        decoded = self.scan("NHTSA_vPIC").select("vin", *COHORT).with_columns(
            *[pl.when(pl.col(c).cast(pl.String).str.strip_chars() == "").then(None)
              .otherwise(pl.col(c)).alias(c) for c in COHORT[1:]],
            pl.lit(True).alias("vpic_matched"),
        )
        result = listings.join(decoded, on="vin", how="left", validate="1:1")
        return result.with_columns(
            pl.all_horizontal(pl.col(c).is_not_null() for c in COHORT).alias("cohort_complete"),
            pl.col("maximum_seating").str.strip_chars().str.extract(r"^(\d+)\s+seats?$", 1)
              .cast(pl.Float64, strict=False).alias("seats"),
        ).filter(filter_expression(filters))

    def families(self, filters):
        return self.catalog(filters).select(FAMILY).drop_nulls().unique()

    def source_records(self, table, filters, families=None):
        if families is None:
            families = normalize_family(self.families(filters)).select("_year", "_make", "_model")
        columns = ["model_year", "make", "model"] if table in ("NHTSA_Complaints", "NHTSA_Recalls") else FAMILY
        records = normalize_family(self.scan(table), columns)
        records = records.join(families, on=["_year", "_make", "_model"], how="semi")
        if table == "NHTSA_Complaints":
            from MODELS.Price_ML_Models import _complaint_product_matches
            records = records.filter(pl.col("dateComplaintFiled").is_not_null()
                                     & (pl.col("dateComplaintFiled") <= date(2020, 12, 31))
                                     & pl.col("odiNumber").is_not_null())
            records = records.filter(pl.struct("products", "model_year", "make", "model")
                .map_elements(_complaint_product_matches, return_dtype=pl.Boolean))
        if table == "NHTSA_Recalls":
            # Source applicability can differ from the API query family.
            records = records.filter(pl.col("ReportReceivedDate").is_not_null()
                                     & (pl.col("ReportReceivedDate") <= date(2020, 12, 31))
                                     & pl.col("NHTSACampaignNumber").is_not_null()
                                     & (pl.col("NHTSACampaignNumber").str.strip_chars() != ""))
            source = normalize_family(records, FAMILY)
            records = source.join(families, on=["_year", "_make", "_model"], how="semi")
        return records

    def summary(self, filters):
        return self.catalog(filters).select(
            pl.len().alias("listings"), pl.col("vin").n_unique().alias("unique_vins"),
            pl.struct(COHORT).filter(pl.col("cohort_complete")).n_unique().alias("cohorts"),
            pl.col("price").median().alias("median_price"),
            pl.col("mileage").median().alias("median_mileage"),
            *[stat for column in ("price", "mileage") for stat in (
                pl.col(column).filter(pl.col(column).is_finite()).min().alias(f"min_{column}"),
                pl.col(column).filter(pl.col(column).is_finite()).max().alias(f"max_{column}"),
            )],
            pl.col("vpic_matched").fill_null(False).sum().alias("decoded_listings"),
            pl.col("cohort_complete").sum().alias("complete_cohort_listings"),
            pl.col("listed_date").min().alias("first_listing"),
            pl.col("listed_date").max().alias("last_listing"),
        ).collect().to_dicts()[0]

    def cohort_summary(self, filters, limit=200):
        return self.catalog(filters).filter(pl.col("cohort_complete")).group_by(COHORT).agg(
            pl.len().alias("listings"), pl.col("price").median().alias("median_price"),
            pl.col("price").quantile(.25).alias("price_q25"),
            pl.col("price").quantile(.75).alias("price_q75"),
            pl.col("mileage").median().alias("median_mileage"),
        ).sort(["listings", *COHORT], descending=[True, False, False, False, False]).head(limit).collect()

    def distribution(self, filters, column, width):
        return (self.catalog(filters).filter(pl.col(column).is_finite())
                .group_by((pl.col(column) // width * width).alias("lower"))
                .agg(pl.len().alias("listings")).sort("lower").collect()
                .with_columns((pl.col("lower") + width).alias("upper")))

    def youtube(self, filters):
        makes = self.catalog(filters).select("Make").drop_nulls().unique().collect()["Make"]
        path = self.directory.parents[1] / "YOUTUBE/outputs/price_features_v1/make_features.parquet"
        if not path.is_file():
            return pl.DataFrame()
        return pl.read_parquet(path).filter(pl.col("Make").is_in(makes.to_list()))


    def youtube_comments(self, filters, limit=10):
        makes = self.catalog(filters).select("Make").drop_nulls().unique().collect()["Make"].to_list()
        paths = sorted((self.directory.parents[1] / "YOUTUBE/outputs/text_scored_v1/shards").glob("*.parquet"))
        if not paths or not makes:
            return pl.DataFrame()
        columns = ["Make", "comment_id", "video_title", "published_date", "text"]
        return (pl.scan_parquet(paths).filter(
            pl.col("Make").is_in(makes)
            & (pl.col("attribution_scope") == "explicit_comment")
            & pl.col("feature_eligible") & ~pl.col("development_excluded")
            & ~pl.col("direct_price_mention")
            & (pl.col("published_date") <= "2026-10-01")
            & (pl.col("updated_date") <= "2026-10-01")
        ).select(columns).unique(subset=["Make", "comment_id"], maintain_order=True)
          .head(limit).collect())


@st.cache_resource(max_entries=2, show_spinner=False)
def snapshot_at(signature: tuple):
    return Snapshot()


def snapshot():
    directory = ROOT / "DATA/CAR_DATA_FINAL"
    paths = [directory / "manifest.json", *sorted(directory.glob("*.parquet"))]
    signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    return snapshot_at(signature)


@st.cache_data(max_entries=32, ttl=900, show_spinner=False)
def query_snapshot(identity: str, filters: Filters, operation: str, *args):
    source = snapshot()
    if source.identity != identity:
        raise ValueError("Snapshot changed; refresh the app")
    if operation == "scope":
        return (source.catalog(filters).group_by(FAMILY).agg(pl.len().alias("listings"))
                .sort(FAMILY, nulls_last=True).collect())
    if operation == "summary":
        return source.summary(filters)
    if operation == "cohorts":
        return source.cohort_summary(filters, *args)
    if operation == "distribution":
        return source.distribution(filters, *args)
    if operation == "distribution_extremes":
        column, = args
        if column not in ("price", "mileage"):
            raise ValueError("Unsupported distribution column")
        return (source.catalog(filters).filter(pl.col(column).is_finite())
                .select(*COHORT, "price", "mileage", "vin", "listing_id")
                .sort([column, "vin"], descending=[True, False])
                .head(10).collect())
    if operation == "listings":
        column, descending, page = args
        if column not in ["price", "mileage", "ModelYear", "listing_id"]:
            raise ValueError("Unsupported sort column")
        listing_page = source.catalog(filters).sort(
            [column, "listing_id"], descending=[descending, False], nulls_last=True
        ).slice(page * 100, 100)
        decoded = source.scan("NHTSA_vPIC")
        transmission = "Transmission" if "Transmission" in decoded.collect_schema() else "TransmissionStyle"
        return listing_page.join(
            decoded.select("vin", pl.col(transmission).alias("Transmission")),
            on="vin", how="left", validate="1:1",
        ).collect()
    if operation == "scatter":
        return source.catalog(filters).select("vin", *COHORT, "price", "mileage").sort(
            pl.col("vin").hash(seed=42)).head(3000).collect()
    if operation == "youtube_comments":
        return source.youtube_comments(filters)
    if operation == "youtube":
        return source.youtube(filters)
    raise ValueError("Unsupported query")


def supported_make_cohorts(rows):
    supported = rows.group_by("Make").len().filter(pl.col("len") >= MIN_MAKE_ROWS).select("Make")
    return rows.join(supported, on="Make", how="semi")


@st.cache_data(max_entries=4, ttl=900, show_spinner=False)
def filter_options(identity):
    source = snapshot()
    decoded = source.scan("NHTSA_vPIC").select("vin", *COHORT)
    cohorts = source.scan("used_cars").select("vin").join(decoded, on="vin", how="inner", validate="1:1")
    cohorts = cohorts.collect()
    cohorts = supported_make_cohorts(cohorts).select(COHORT).unique()
    categories = source.scan("used_cars").select(
        *[pl.col(c).drop_nulls().unique().implode() for c in ["body_type", "fuel_type", "wheel_system", "transmission"]]
    ).collect().to_dicts()[0]
    return cohorts, categories


def evidence_bundle(source, filters):
    """Deduplicate source identities before counts; conflicting versions stay visible."""
    result = {}
    families = normalize_family(source.families(filters)).select("_year", "_make", "_model").collect()
    for table, key in [("NHTSA_Complaints", "odiNumber"), ("NHTSA_Recalls", "NHTSACampaignNumber"),
                       ("NHTSA_Safety", "VehicleId")]:
        records = source.source_records(table, filters, families.lazy()).collect()
        if table == "NHTSA_Safety":
            ratings = [c for c in records.columns if "Rating" in c]
            records = records.with_columns(
                *[pl.when(pl.col(c).is_between(1, 5)).then(pl.col(c)).otherwise(None).alias(c)
                  for c in ratings])
            for rating, possibility in [("RolloverRating", "RolloverPossibility"),
                                        ("RolloverRating2", "RolloverPossibility2")]:
                records = records.with_columns(
                    pl.when((pl.col(possibility) == 0) & pl.col(rating).is_null())
                    .then(None).otherwise(pl.col(possibility)).alias(possibility))
        payload = [c for c in records.columns if c not in ["query_id", "record_key", "_year", "_make", "_model"]]
        versions = records.select(payload).unique()
        group = ["_year", "_make", "_model", key]
        conflict_columns = {
            "NHTSA_Complaints": ["summary", "crash", "fire", "numberOfInjuries", "numberOfDeaths", "components"],
            "NHTSA_Recalls": ["Component", "Summary", "Consequence", "Remedy"],
            "NHTSA_Safety": [c for c in records.columns if "Rating" in c],
        }[table]
        conflicts = records.group_by(group).agg(
            pl.any_horizontal([pl.col(c).n_unique() > 1 for c in conflict_columns]).alias("conflicting")
        ).filter(pl.col("conflicting")).select(group)
        component = "components" if table == "NHTSA_Complaints" else "Component"
        components = pl.DataFrame()
        if table != "NHTSA_Safety":
            stable = records.join(conflicts, on=group, how="anti", nulls_equal=True)
            components = (stable.select(key, component)
                .with_columns(pl.col(component).str.split(",")).explode(component, empty_as_null=True)
                .with_columns(pl.col(component).str.strip_chars())
                .filter(pl.col(component).is_not_null() & (pl.col(component) != ""))
                .group_by(component).agg(pl.col(key).n_unique().alias("source_ids"))
                .sort("source_ids", descending=True).head(20))
        result[table] = {
            "count": records[key].drop_nulls().n_unique(),
            "families": records.select("_year", "_make", "_model").unique().height,
            "conflicts": conflicts[key].n_unique(),
            "reference_families": families.height,
            "components": components,
            "records": versions.sort(key, nulls_last=True).head(500),
        }
    return result


@st.cache_data(max_entries=16, ttl=900, show_spinner=False)
def cached_evidence(identity, filters):
    return evidence_bundle(snapshot(), filters)
