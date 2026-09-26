"""Read selected modeling tables in bounded batches for the cleaning workflow.

This script reads source tables from SQLite and applies the used-car and vPIC
renames and type conversions in memory. It never modifies the database.
Batches are yielded one at a time to bound memory use.
"""

from pathlib import Path
import argparse
import sqlite3
from collections.abc import Iterator
from contextlib import closing

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "DATA" / "CAR_DATA.db"

# Keep this list aligned with the selected tables documented in
# DATA_DICTIONARY.md. The query ledger and vPIC lookup tables are excluded.
TABLES_TO_LOAD = (
    "used_cars",
    "nhtsa_vpic_decodes",
    "nhtsa_complaints",
    "nhtsa_recalls",
    "nhtsa_safety_rating_values",
)
BATCH_SIZE = 100_000
INTEGER_COLUMNS = (
    "Axles", "Doors", "EngineCycles", "EngineCylinders", "EngineHP",
    "ModelYear", "SeatRows", "Seats", "TransmissionSpeeds", "Wheels", "Windows",
)
FLOAT_COLUMNS = ("BasePrice", "DisplacementCC", "DisplacementCI", "DisplacementL")
USED_CARS_FLOAT_COLUMNS = (
    "back_legroom", "bed_length", "city_fuel_economy", "combine_fuel_economy",
    "engine_displacement", "front_legroom", "fuel_tank_volume", "height",
    "highway_fuel_economy", "horsepower", "latitude", "length", "longitude",
    "mileage", "price", "seller_rating", "width", "wheelbase",
)
USED_CARS_INTEGER_COLUMNS = (
    "daysonmarket", "dealer_zip", "maximum_seating", "owner_count",
    "savings_amount", "year",
)
USED_CARS_UNIT_SUFFIXES = {
    "back_legroom": "in", "bed_length": "in", "front_legroom": "in",
    "fuel_tank_volume": "gal", "height": "in", "length": "in",
    "width": "in", "wheelbase": "in", "maximum_seating": "seats",
}


def clean_used_cars_batch(batch: pl.DataFrame) -> pl.DataFrame:
    """Strip measurement suffixes and parse listing numbers and dates in memory.

    Empty or invalid values become NULL. Integral decimal strings such as
    owner_count '3.0' become integers; fractional counts become NULL.
    Converting dealer_zip to an integer removes leading zeros.
    """
    conversions: list[pl.Expr] = []
    for column_name in USED_CARS_FLOAT_COLUMNS + USED_CARS_INTEGER_COLUMNS:
        if column_name not in batch.columns:
            continue
        value = pl.col(column_name).cast(pl.String).str.strip_chars()
        if column_name in USED_CARS_UNIT_SUFFIXES:
            suffix = USED_CARS_UNIT_SUFFIXES[column_name]
            value = value.str.replace(rf"\s*{suffix}$", "").str.strip_chars()
        value = value.replace("", None)
        if column_name in USED_CARS_INTEGER_COLUMNS:
            # Accept integer-valued decimal strings without truncating fractions.
            value = value.str.replace(r"\.0+$", "").cast(pl.Int64, strict=False)
        else:
            value = value.cast(pl.Float64, strict=False)
        conversions.append(value.alias(column_name))
    if "listed_date" in batch.columns:
        conversions.append(
            pl.col("listed_date").cast(pl.String).str.strip_chars()
            .replace("", None).str.to_date(format="%Y-%m-%d", strict=False)
            .alias("listed_date")
        )
    return batch.with_columns(conversions) if conversions else batch


def clean_vpic_batch(batch: pl.DataFrame) -> pl.DataFrame:
    """Rename vPIC fields and cast the requested columns in a batch."""
    rename_map = {
        column: column.replace("vpic_", "", 1)
        for column in batch.columns
        if "vpic_" in column
    }
    batch = batch.rename(rename_map)

    conversions: list[pl.Expr] = []
    for column_name in INTEGER_COLUMNS:
        if column_name in batch.columns:
            conversions.append(
                pl.col(column_name).cast(pl.String).str.strip_chars()
                .replace("", None).cast(pl.Int64, strict=False).alias(column_name)
            )
    for column_name in FLOAT_COLUMNS:
        if column_name in batch.columns:
            conversions.append(
                pl.col(column_name).cast(pl.String).str.strip_chars()
                .replace("", None).cast(pl.Float64, strict=False).alias(column_name)
            )
    if "fetched_at" in batch.columns:
        conversions.append(
            pl.col("fetched_at").cast(pl.String)
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
            .alias("fetched_at")
        )

    return batch.with_columns(conversions) if conversions else batch


def load_data(
    database_path: Path = DATABASE_PATH,
    batch_size: int = BATCH_SIZE,
    table_names: tuple[str, ...] = TABLES_TO_LOAD,
) -> Iterator[tuple[str, pl.DataFrame]]:
    """Yield (table name, Polars batch) for each selected table.

    The caller can process each batch before requesting the next one. The
    caller must avoid accumulating batches to keep memory use bounded.
    ``used_cars`` batches have units stripped and numeric/date fields parsed.
    ``nhtsa_vpic_decodes`` batches have prefixes removed and selected columns
    cast in memory; the SQLite source database is never modified.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    unknown_tables = sorted(set(table_names) - set(TABLES_TO_LOAD))
    if unknown_tables:
        raise ValueError("Tables are not in the selected data dictionary: " + ", ".join(unknown_tables))

    with closing(sqlite3.connect(database_path)) as connection:
        available_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = sorted(set(table_names) - available_tables)
        if missing_tables:
            raise RuntimeError(
                "Expected tables are missing from the database: "
                + ", ".join(missing_tables)
            )

        for table_name in table_names:
            cursor = connection.execute(f'SELECT * FROM "{table_name}"')
            columns = [description[0] for description in cursor.description]
            while rows := cursor.fetchmany(batch_size):
                batch = pl.DataFrame(
                    rows, schema=columns, orient="row", infer_schema_length=None
                )
                # Release SQLite's Python row objects before Polars cleaning.
                del rows
                if table_name == "used_cars":
                    batch = clean_used_cars_batch(batch)
                elif table_name == "nhtsa_vpic_decodes":
                    batch = clean_vpic_batch(batch)
                yield table_name, batch
                # Release this frame before fetching the next batch.
                del batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Rows per Polars batch (default: 100000); lower to reduce memory use.",
    )
    args = parser.parse_args()
    batch_counts = {table_name: 0 for table_name in TABLES_TO_LOAD}
    for table_name, batch in load_data(batch_size=args.batch_size):
        batch_counts[table_name] += len(batch)
        print(
            f"Read {len(batch):,} rows from {table_name} "
            f"(running total: {batch_counts[table_name]:,})"
        )
        del batch
