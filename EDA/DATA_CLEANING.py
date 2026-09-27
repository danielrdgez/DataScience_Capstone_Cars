"""Export the selected SQLite tables to a reusable CAR_DATA_FINAL Parquet dataset.

The source database is opened read-only. Numeric conversions have fixed dtypes;
JSON records are expanded and safety fields are widened with stable schemas.
Rows are processed in bounded batches. Modeling reads the saved Parquet files.
"""

from pathlib import Path
import argparse
import sqlite3
import json
import math
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
import shutil
import uuid
from collections.abc import Iterator
from contextlib import closing

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "DATA" / "CAR_DATA.db"
FINAL_DATASET_PATH = PROJECT_ROOT / "DATA" / "CAR_DATA_FINAL"

# Keep this list aligned with the selected tables documented in
# DATA_DICTIONARY.md. The query ledger and vPIC lookup tables are excluded.
TABLES_TO_LOAD = (
    "used_cars",
    "nhtsa_vpic_decodes",
    "nhtsa_complaints",
    "nhtsa_recalls",
    "nhtsa_safety_rating_values",
    "youtube_comments_sentiment",
)
TABLES_TO_EXPORT = TABLES_TO_LOAD
BATCH_SIZE = 100_000
INTEGER_COLUMNS = (
    "model_year_hint", "Axles", "BatteryA", "BatteryCells", "BatteryKWh",
    "BatteryKWh_to", "BatteryPacks", "BatteryV", "BedLengthIN", "ChargerPowerKW",
    "CurbWeightLB", "Doors", "EngineCycles", "EngineCylinders", "EngineHP",
    "EngineHP_to", "EngineKW", "ModelYear", "SeatRows", "Seats", "TopSpeedMPH",
    "TrailerLength", "TransmissionSpeeds", "WheelSizeFront", "WheelSizeRear",
    "Wheels", "Windows",
)
FLOAT_COLUMNS = (
    "BasePrice", "DisplacementCC", "DisplacementCI", "DisplacementL",
    "TrackWidth", "WheelBaseLong", "WheelBaseShort",
)
USED_CARS_FLOAT_COLUMNS = (
    "back_legroom", "bed_length", "city_fuel_economy", "engine_displacement",
    "front_legroom", "fuel_tank_volume", "height", "highway_fuel_economy",
    "horsepower", "latitude", "length", "longitude", "mileage",
    "seller_rating", "wheelbase", "width",
)
USED_CARS_INTEGER_COLUMNS = (
    "daysonmarket", "dealer_zip", "owner_count", "price", "savings_amount", "year",
)
USED_CARS_UNITS = {
    "back_legroom": "in", "bed_length": "in", "front_legroom": "in",
    "fuel_tank_volume": "gal", "height": "in", "length": "in",
    "wheelbase": "in", "width": "in",
}
PARQUET_FILES = {
    "used_cars": "used_cars.parquet",
    "nhtsa_vpic_decodes": "NHTSA_vPIC.parquet",
    "nhtsa_complaints": "NHTSA_Complaints.parquet",
    "nhtsa_recalls": "NHTSA_Recalls.parquet",
    "nhtsa_safety_rating_values": "NHTSA_Safety.parquet",
    "youtube_comments_sentiment": "Youtube_Comments.parquet",
}
JSON_TABLES = ("nhtsa_complaints", "nhtsa_recalls")
DROP_COLUMNS = frozenset((
    "bed_height", "combine_fuel_economy", "is_certified", "main_picture_url",
    "sp_id", "sp_name", "trimID", "vehicle_damage_category", "DriverAssist",
    "FuelTankMaterial", "FuelTankType", "GCWR", "GCWR_to", "MakeID",
    "Manufacturer_ID", "ModelID", "MotorcycleChassisType",
    "MotorcycleSuspensionType", "NCSAMapExcApprovedBy", "NCSAMapExcApprovedOn",
    "NCSAMappingException", "NCSAModel", "NCSANote", "NonLandUse",
    "OtherBusInfo", "OtherMotorcycleInfo", "OtherTrailerInfo", "CleanDecode",
    "WheelieMitigation", "VehicleDescriptor", "SuggestedVIN",
    "SAEAutomationLevel_to", "SAEAutomationLevel", "PossibleValues",
))


def _convert_if_representable(batch: pl.DataFrame, column: str, expression: pl.Expr) -> pl.DataFrame:
    """Keep the source column if conversion would lose a non-NULL value.

    Expressions must only parse types, never strip, replace, or normalize values.
    A failed parse anywhere in the batch leaves the entire column unchanged.
    """
    source = batch[column]
    converted = batch.select(expression.alias(column)).to_series()
    if converted.null_count() != source.null_count():
        return batch
    if converted.dtype.is_float() and source.dtype == pl.String:
        # A finite numeric string outside Float64 range must not become infinity.
        overflow = converted.is_infinite() & ~source.str.to_lowercase().is_in(
            ["inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"]
        )
        if overflow.any():
            return batch
    # Numeric casts must not truncate fractional inputs or overflow finite values.
    if source.dtype.is_numeric() and converted.dtype.is_numeric():
        if not converted.cast(source.dtype, strict=False).equals(source):
            return batch
    return batch.with_columns(converted)


def _numeric_expressions(batch: pl.DataFrame, float_columns: tuple[str, ...],
                         integer_columns: tuple[str, ...], units: dict) -> list[pl.Expr]:
    """Shared nullable casts with the user's nearest-integer rounding policy."""
    expressions = []
    for column in float_columns + integer_columns:
        if column not in batch.columns:
            continue
        value = pl.col(column).cast(pl.String).str.strip_chars()
        if column in units:
            value = value.str.replace(r"\s*" + units[column] + r"$", "").str.strip_chars()
        if column in integer_columns:
            # The first fractional digit determines nearest-integer rounding.
            # Trim subsequent digits before Decimal conversion so its scale
            # cannot round 2.499... to 2.5 and cause a second rounding to 3.
            decimal_text = value.str.replace(r"(\.\d)\d+$", "${1}")
            decimal = pl.when(value.str.contains(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")) \
                .then(decimal_text.cast(pl.Decimal(38, 1), strict=False)).otherwise(None)
            converted = decimal.round(0, mode="half_away_from_zero").cast(pl.Int64, strict=False)
        else:
            number = value.cast(pl.Float64, strict=False)
            converted = pl.when(number.is_finite()).then(number).otherwise(None)
        expressions.append(converted.alias(column))
    return expressions


def clean_used_cars_batch(batch: pl.DataFrame) -> pl.DataFrame:
    """Apply requested listing types, units and integer rounding; invalids are null."""
    expressions = _numeric_expressions(batch, USED_CARS_FLOAT_COLUMNS,
                                       USED_CARS_INTEGER_COLUMNS, USED_CARS_UNITS)
    if "listed_date" in batch.columns:
        expressions.append(
            pl.col("listed_date").cast(pl.String).str.strip_chars()
            .str.to_date(format="%Y-%m-%d", strict=False).alias("listed_date")
        )
    return batch.with_columns(expressions)


def clean_vpic_batch(batch: pl.DataFrame) -> pl.DataFrame:
    """Drop vpic_ prefixes and apply fixed numeric types with rounded integers."""
    rename_map = {
        column: column.removeprefix("vpic_")
        for column in batch.columns
        if column.startswith("vpic_")
    }
    batch = batch.rename(rename_map)

    batch = batch.with_columns(_numeric_expressions(batch, FLOAT_COLUMNS, INTEGER_COLUMNS, {}))
    if "fetched_at" in batch.columns:
        batch = _convert_if_representable(
            batch, "fetched_at",
            pl.col("fetched_at").cast(pl.String)
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
        )

    return batch


def _validate_tables(table_names: tuple[str, ...]) -> None:
    if not table_names or len(set(table_names)) != len(table_names):
        raise ValueError("Select at least one table, without duplicates")
    unknown = sorted(set(table_names) - set(TABLES_TO_LOAD))
    if unknown:
        raise ValueError("Tables are not in the selected data dictionary: " + ", ".join(unknown))


def _source_schema(connection: sqlite3.Connection, table: str) -> dict:
    columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    if not columns:
        raise RuntimeError(f"Expected table is missing from the database: {table}")
    # SQLite DATE fields in the imported YouTube table are stored as text.
    types = {"TEXT": pl.String, "INTEGER": pl.Int64, "REAL": pl.Float64,
             "DATE": pl.String, "DATETIME": pl.String}
    schema = {}
    for _, name, declared, *_ in columns:
        if declared.upper() not in types:
            raise ValueError(f"Unsupported SQLite type for {table}.{name}: {declared}")
        schema[name] = types[declared.upper()]
    return schema


def _query_batches(connection: sqlite3.Connection, query: str, schema: dict,
                   batch_size: int, parameters: tuple = ()) -> Iterator[pl.DataFrame]:
    cursor = connection.execute(query, parameters)
    try:
        while rows := cursor.fetchmany(batch_size):
            batch = pl.DataFrame(rows, schema=schema, orient="row")
            del rows
            yield batch
            del batch
    finally:
        cursor.close()


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _decode_record(raw: str | None, table: str) -> dict:
    if raw is None:
        return {}

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    try:
        record = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
        if not isinstance(record, dict):
            raise ValueError("record_json must be a JSON object")
        return record
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid record_json in {table}: {exc}") from exc


def _json_kind(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and -(2**63) <= value < 2**63:
        return "integer"
    if isinstance(value, float) and math.isfinite(value):
        return "float"
    return "text"


def _json_columns(connection: sqlite3.Connection, table: str, source_schema: dict,
                  batch_size: int) -> dict:
    """Discover every top-level key and its type, including keys in later batches."""
    if "record_json" not in source_schema:
        raise ValueError(f"Missing record_json in {table}")
    kinds = {}
    for batch in _query_batches(connection, f'SELECT record_json FROM "{table}"',
                                {"record_json": pl.String}, batch_size):
        for raw in batch["record_json"]:
            for key, value in _decode_record(raw, table).items():
                observed = kinds.setdefault(key, set())
                kind = _json_kind(value)
                if kind != "null":
                    observed.add(kind)
        del batch
    columns = {}
    # Reserve every original JSON key before allocating collision-safe aliases.
    occupied = set(source_schema) | set(kinds)
    for key in sorted(kinds):
        name = key
        if name in source_schema:
            name = "json_" + key
            while name in occupied:
                name = "json_" + name
        occupied.add(name)
        dtype = {frozenset({"boolean"}): pl.Boolean,
                 frozenset({"integer"}): pl.Int64,
                 frozenset({"float"}): pl.Float64}.get(frozenset(kinds[key]), pl.String)
        columns[key] = (name, dtype)
    return columns


def _expand_json(batch: pl.DataFrame, table: str, columns: dict) -> pl.DataFrame:
    values = {key: [] for key in columns}
    for raw in batch["record_json"]:
        record = _decode_record(raw, table)
        for key, (_, dtype) in columns.items():
            value = record.get(key)
            if dtype == pl.String and value is not None and not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            values[key].append(value)
    return batch.with_columns([
        pl.Series(name, values[key], dtype=dtype)
        for key, (name, dtype) in columns.items()
    ])


def _table_plan(connection: sqlite3.Connection, table: str, schema: dict,
                batch_size: int) -> dict:
    plan = {"query": f'SELECT * FROM "{table}"', "parameters": (),
            "read_schema": schema, "json_columns": {}, "timestamp_as_datetime": False}
    plan["source_rows"] = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    plan["expected_rows"] = plan["source_rows"]
    if table in JSON_TABLES:
        plan["json_columns"] = _json_columns(connection, table, schema, batch_size)
    elif table == "nhtsa_vpic_decodes" and "fetched_at" in schema:
        # Preserve the existing timestamp guard across the entire saved column.
        plan["timestamp_as_datetime"] = True
        for batch in _query_batches(connection, f'SELECT fetched_at FROM "{table}"',
                                    {"fetched_at": schema["fetched_at"]}, batch_size):
            if clean_vpic_batch(batch).schema["fetched_at"] != pl.Datetime("us"):
                plan["timestamp_as_datetime"] = False
                break
    elif table == "nhtsa_safety_rating_values":
        required = {"query_id", "vehicle_id", "field_name", "field_value"}
        if set(schema) != required:
            raise ValueError("Safety source must have query_id, vehicle_id, field_name, field_value")
        invalid = connection.execute(f'SELECT 1 FROM "{table}" WHERE query_id IS NULL OR vehicle_id IS NULL OR field_name IS NULL LIMIT 1').fetchone()
        if invalid:
            raise ValueError("Safety identifiers and field_name must be non-null")
        duplicate = connection.execute(f'SELECT 1 FROM "{table}" GROUP BY query_id, vehicle_id, field_name HAVING COUNT(*) > 1 LIMIT 1').fetchone()
        if duplicate:
            raise ValueError("Duplicate safety field within (query_id, vehicle_id); refusing to discard values")
        fields = [row[0] for row in connection.execute(f'SELECT DISTINCT field_name FROM "{table}" ORDER BY field_name')]
        # Reject ambiguous aliases rather than silently merging distinct fields.
        if any(name.lower() in {"query_id", "vehicle_id"} for name in fields) or len({name.lower() for name in fields}) != len(fields):
            raise ValueError("Safety field names collide with identifiers or with each other")
        expressions = [f'MAX(CASE WHEN field_name = ? THEN field_value END) AS {_quote(name)}' for name in fields]
        selection = ', '.join(['query_id', 'vehicle_id'] + expressions)
        plan["query"] = f'SELECT {selection} FROM "{table}" GROUP BY query_id, vehicle_id ORDER BY query_id, vehicle_id'
        plan["parameters"] = tuple(fields)
        plan["read_schema"] = {"query_id": schema["query_id"], "vehicle_id": schema["vehicle_id"], **{name: pl.String for name in fields}}
        plan["expected_rows"] = connection.execute(f'SELECT COUNT(*) FROM (SELECT query_id, vehicle_id FROM "{table}" GROUP BY query_id, vehicle_id)').fetchone()[0]
    return plan


def _transform_for_export(table: str, batch: pl.DataFrame, plan: dict) -> pl.DataFrame:
    if table == "used_cars":
        batch = clean_used_cars_batch(batch)
    elif table == "nhtsa_vpic_decodes":
        # Remove the timestamp while numeric casts run to avoid a batch fallback.
        timestamp = batch["fetched_at"] if "fetched_at" in batch.columns else None
        batch = clean_vpic_batch(batch.drop("fetched_at") if timestamp is not None else batch)
        if timestamp is not None:
            if plan["timestamp_as_datetime"]:
                timestamp = timestamp.cast(pl.String).str.to_datetime(format="%Y-%m-%d %H:%M:%S")
            batch = batch.with_columns(timestamp)
            # Preserve original source column order after renaming.
            batch = batch.select([name.removeprefix("vpic_") for name in plan["read_schema"]])
    elif table in JSON_TABLES:
        batch = batch.with_columns(_numeric_expressions(batch, (), ("model_year",), {}))
        batch = _expand_json(batch, table, plan["json_columns"])
    # Apply exclusions after all derived columns are present; this also removes
    # matching keys discovered in expanded complaint/recall JSON.
    drop = [name for name in batch.columns if name in DROP_COLUMNS or name == "record_json"]
    return batch.drop(drop) if drop else batch


def load_data(database_path: Path = DATABASE_PATH, batch_size: int = BATCH_SIZE,
              table_names: tuple[str, ...] = TABLES_TO_LOAD) -> Iterator[tuple[str, pl.DataFrame]]:
    """Inspect transformed source batches with the same schemas as the exporter."""
    _validate_tables(table_names)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    with closing(sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        schemas = {table: _source_schema(connection, table) for table in table_names}
        for table, schema in schemas.items():
            plan = _table_plan(connection, table, schema, batch_size)
            for batch in _query_batches(connection, plan["query"], plan["read_schema"], batch_size, plan["parameters"]):
                yield table, _transform_for_export(table, batch, plan)
                del batch


def export_parquet(database_path: Path = DATABASE_PATH,
                   output_dir: Path = FINAL_DATASET_PATH,
                   batch_size: int = BATCH_SIZE,
                   table_names: tuple[str, ...] = TABLES_TO_EXPORT) -> dict:
    """Write a complete new dataset, publishing only after all files are verified.

    The complete validated dataset replaces any existing output directory.
    A read transaction keeps schema discovery and data aligned.
    """
    import pyarrow.parquet as pq

    _validate_tables(table_names)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_database": str(database_path.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_size": batch_size,
        "integer_policy": "Round nearest, ties away from zero; blanks/invalids/overflow become null",
        "tables": {},
    }
    with TemporaryDirectory(prefix=".car-data-export-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        with closing(sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("BEGIN")
            schemas = {table: _source_schema(connection, table) for table in table_names}
            for table, schema in schemas.items():
                print(f"Preparing {table}...", flush=True)
                plan = _table_plan(connection, table, schema, batch_size)
                empty = _transform_for_export(table, pl.DataFrame(schema=plan["read_schema"]), plan)
                path = staging / PARQUET_FILES[table]
                rows = 0
                column_names = {
                    name: name.removeprefix("vpic_") if table == "nhtsa_vpic_decodes" else name
                    for name in schema
                }
                column_names = {
                    source_name: output_name
                    for source_name, output_name in column_names.items()
                    if output_name not in DROP_COLUMNS and output_name != "record_json"
                }
                new_nulls = {name: 0 for name in column_names.values()} if table != "nhtsa_safety_rating_values" else {}
                with pq.ParquetWriter(path, empty.to_arrow().schema, compression="zstd") as writer:
                    for batch in _query_batches(connection, plan["query"], plan["read_schema"], batch_size, plan["parameters"]):
                        source_nulls = batch.null_count().row(0, named=True) if new_nulls else {}
                        result = _transform_for_export(table, batch, plan)
                        del batch
                        if new_nulls:
                            result_nulls = result.null_count().row(0, named=True)
                            for source_name, output_name in column_names.items():
                                new_nulls[output_name] += result_nulls[output_name] - source_nulls[source_name]
                        writer.write_table(result.to_arrow(), row_group_size=batch_size)
                        rows += len(result)
                        del result
                        print(f"  {table}: {rows:,} rows written", flush=True)
                with pq.ParquetFile(path) as parquet:
                    if parquet.metadata.num_rows != rows or rows != plan["expected_rows"]:
                        raise RuntimeError(f"Parquet row count mismatch: {table}")
                manifest["tables"][table] = {
                    "file": path.name, "source_rows": plan["source_rows"], "rows": rows, "bytes": path.stat().st_size,
                    "json_key_columns": {key: name for key, (name, _) in plan["json_columns"].items()},
                    "schema": {name: str(dtype) for name, dtype in empty.schema.items()},
                    "new_nulls_from_conversion": {name: count for name, count in new_nulls.items() if count},
                }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        backup = output_dir.with_name(f".{output_dir.name}.backup-{uuid.uuid4().hex}")
        had_existing_output = output_dir.exists()
        if had_existing_output:
            if not output_dir.is_dir():
                raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
            output_dir.rename(backup)
        try:
            staging.rename(output_dir)
        except Exception:
            if had_existing_output and backup.exists():
                backup.rename(output_dir)
            raise
        if had_existing_output:
            shutil.rmtree(backup)
    print(f"Saved dataset: {output_dir}", flush=True)
    return manifest


def scan_final_table(table_name: str = "used_cars",
                     dataset_path: Path = FINAL_DATASET_PATH) -> pl.LazyFrame:
    """Return a Polars lazy scan so models can select/filter before collecting."""
    _validate_tables((table_name,))
    path = dataset_path / PARQUET_FILES[table_name]
    if not path.is_file():
        raise FileNotFoundError(f"Run EDA/DATA_CLEANING.py first; missing {path}")
    return pl.scan_parquet(path)


def load_final_data(dataset_path: Path = FINAL_DATASET_PATH,
                    batch_size: int = BATCH_SIZE,
                    table_names: tuple[str, ...] = TABLES_TO_EXPORT) -> Iterator[tuple[str, pl.DataFrame]]:
    """Read saved Parquet in bounded batches and return Polars DataFrames."""
    import pyarrow.parquet as pq

    _validate_tables(table_names)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    paths = {table: dataset_path / PARQUET_FILES[table] for table in table_names}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Run EDA/DATA_CLEANING.py first; missing {path}")
    for table, path in paths.items():
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=batch_size):
                yield table, pl.from_arrow(batch)
                del batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=FINAL_DATASET_PATH)
    parser.add_argument("--table", action="append", choices=TABLES_TO_LOAD,
                        help="Export selected tables; repeat to select several. Default: all six tables.")
    args = parser.parse_args()
    export_parquet(args.database, args.output_dir, args.batch_size,
                   tuple(args.table) if args.table else TABLES_TO_EXPORT)
