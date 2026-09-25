"""Build CAR_DATA.db from used-car listings and NHTSA data sources.

The listing CSV is streamed in batches. Vehicle-level NHTSA endpoints are
queried once per distinct (year, make_name, model_name); VIN decode values are
queried locally in vPIC SQL Server batches of up to 100 VINs. Listings are only
loaded when --load-listings is specified, so query-only runs do not overwrite
or replace source listing rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "DATA" / "used_cars_data.csv"
DB_PATH = ROOT / "DATA" / "CAR_DATA.db"
LOG = logging.getLogger("nhtsa_import")
BASE = "https://api.nhtsa.gov"
VPIC_BATCH_LIMIT = 100


def request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    from urllib.error import HTTPError
    from urllib.parse import urlencode
    full_url = f"{url}?{urlencode(params)}"
    req = Request(full_url, headers={"User-Agent": "CarDataCapstone/1.0"})
    try:
        with urlopen(req, timeout=45) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail[:800] or 'no response body'}") from exc


def result_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results", payload.get("Results", []))
    return rows if isinstance(rows, list) else []


def normalize_catalog_key(value: str) -> str:
    """Normalize punctuation and case only for catalog matching."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def catalog_results(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    payload = request_json(f"{BASE}{path}", params)
    return result_list(payload)


def find_catalog_value(rows: list[dict[str, Any]], input_value: str,
                       field_names: tuple[str, ...]) -> str | None:
    wanted = normalize_catalog_key(input_value)
    for row in rows:
        for field in field_names:
            value = row.get(field)
            if isinstance(value, str) and normalize_catalog_key(value) == wanted:
                return value
    return None


class VehicleCatalog:
    """Caches NHTSA's issue-specific make/model catalog values."""

    def __init__(self) -> None:
        self._makes: dict[tuple[int, str], list[dict[str, Any]]] = {}
        self._models: dict[tuple[int, str, str], list[dict[str, Any]]] = {}

    def canonical_pair(self, year: int, make: str, model: str,
                       issue_type: str) -> tuple[str, str] | None:
        make_key = (year, issue_type)
        if make_key not in self._makes:
            self._makes[make_key] = catalog_results(
                "/products/vehicle/makes", {"modelYear": year, "issueType": issue_type})
        canonical_make = find_catalog_value(self._makes[make_key], make, ("make", "Make"))
        if canonical_make is None:
            return None

        model_key = (year, issue_type, canonical_make.casefold())
        if model_key not in self._models:
            self._models[model_key] = catalog_results(
                "/products/vehicle/models",
                {"modelYear": year, "make": canonical_make, "issueType": issue_type})
        canonical_model = find_catalog_value(self._models[model_key], model, ("model", "Model"))
        if canonical_model is None:
            return None
        return canonical_make, canonical_model


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS nhtsa_vpic_values (
        vin TEXT NOT NULL, model_year_hint INTEGER, variable_id TEXT NOT NULL,
        variable_name TEXT, value TEXT, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(vin, model_year_hint, variable_id)
    );
    CREATE TABLE IF NOT EXISTS nhtsa_recalls (
        query_id INTEGER NOT NULL, record_key TEXT NOT NULL, model_year INTEGER,
        make TEXT, model TEXT, record_json TEXT NOT NULL,
        PRIMARY KEY(query_id, record_key)
    );
    CREATE TABLE IF NOT EXISTS nhtsa_complaints (
        query_id INTEGER NOT NULL, record_key TEXT NOT NULL, model_year INTEGER,
        make TEXT, model TEXT, record_json TEXT NOT NULL,
        PRIMARY KEY(query_id, record_key)
    );
    CREATE TABLE IF NOT EXISTS nhtsa_safety_rating_values (
        query_id INTEGER NOT NULL, vehicle_id TEXT NOT NULL, field_name TEXT NOT NULL,
        field_value TEXT, PRIMARY KEY(query_id, vehicle_id, field_name)
    );
    CREATE TABLE IF NOT EXISTS nhtsa_queries (
        query_id INTEGER PRIMARY KEY AUTOINCREMENT, query_type TEXT NOT NULL,
        model_year INTEGER, make TEXT, model TEXT, vin TEXT,
        status TEXT NOT NULL, result_count INTEGER NOT NULL DEFAULT 0,
        error TEXT, fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(query_type, model_year, make, model, vin)
    );
    """)


def load_listings(conn: sqlite3.Connection, csv_path: Path, limit: int | None, batch_size: int) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    count = 0
    batch = []
    with csv_path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        required = {"vin", "year", "make_name", "model_name"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns {sorted(required)}")
        columns = list(reader.fieldnames or [])
        # Source CSV columns are retained individually as text so the resulting
        # database remains useful for analysis without losing source formatting.
        quoted = ",".join('"' + name.replace('"', '""') + '" TEXT' for name in columns)
        id_col = '"listing_id"' if "listing_id" in columns else '"vin"'
        conn.execute(f"CREATE TABLE IF NOT EXISTS used_cars ({quoted}, PRIMARY KEY ({id_col}))")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_used_cars_vin ON used_cars(vin)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_used_cars_mmy ON used_cars(year, make_name, model_name)")
        insert_sql = (f"INSERT OR REPLACE INTO used_cars ({','.join('"'+c.replace('"','""')+'"' for c in columns)}) "
                      f"VALUES ({','.join('?' for _ in columns)})")
        for row in reader:
            if limit is not None and count >= limit:
                break
            count += 1
            batch.append(tuple(row.get(c) for c in columns))
            if len(batch) >= batch_size:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()
                if count % (batch_size * 10) == 0:
                    LOG.info("Loaded %s listing rows", count)
    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()
    return count


def insert_query(conn: sqlite3.Connection, kind: str, year: int | None, make: str | None,
                 model: str | None, vin: str | None, status: str, n: int = 0,
                 error: str | None = None) -> int:
    conn.execute("""INSERT INTO nhtsa_queries
      (query_type,model_year,make,model,vin,status,result_count,error)
      VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(query_type,model_year,make,model,vin)
      DO UPDATE SET status=excluded.status,result_count=excluded.result_count,
      error=excluded.error,fetched_at=CURRENT_TIMESTAMP""",
      (kind, year, make, model, vin, status, n, error))
    return int(conn.execute("""SELECT query_id FROM nhtsa_queries WHERE query_type=?
      AND model_year IS ? AND make IS ? AND model IS ? AND vin IS ?""",
      (kind, year, make, model, vin)).fetchone()[0])


def fetch_vehicle_tables(conn: sqlite3.Connection, delay: float,
                         kinds: tuple[str, ...] = ("recalls", "complaints", "safety"),
                         limit: int | None = None,
                         catalog: VehicleCatalog | None = None) -> None:
    cursor = conn.execute("""SELECT DISTINCT year,make_name,model_name FROM used_cars
      WHERE year IS NOT NULL AND make_name IS NOT NULL AND model_name IS NOT NULL
      ORDER BY year,make_name,model_name""")
    catalog = catalog or VehicleCatalog()
    total = 0
    while mmy_batch := cursor.fetchmany(1000):
        for year_text, source_make, source_model in mmy_batch:
            if limit is not None and total >= limit:
                return
            try:
                year = int(str(year_text).strip())
            except (TypeError, ValueError):
                continue
            for kind in kinds:
                table_kind = "safety_ratings" if kind == "safety" else kind
                try:
                    if kind in ("recalls", "complaints"):
                        issue_type = "r" if kind == "recalls" else "c"
                        pair = catalog.canonical_pair(year, source_make, source_model, issue_type)
                        if pair is None:
                            insert_query(conn, kind, year, source_make, source_model, None,
                                         "unmatched", error="No exact make/model match in the NHTSA product catalog.")
                            conn.commit()
                            continue
                        query_make, query_model = pair
                        url = f"{BASE}/{kind}/{kind}ByVehicle"
                        params = {"modelYear": year, "make": query_make,
                                  "model": query_model, "format": "json"}
                    else:
                        query_make, query_model = source_make, source_model
                        url = (f"{BASE}/SafetyRatings/modelyear/{year}/make/"
                               f"{quote(query_make, safe='')}/model/{quote(query_model, safe='')}")
                        params = {"format": "json"}

                    rows = result_list(request_json(url, params))
                    qid = insert_query(conn, table_kind, year, source_make, source_model,
                                       None, "success", len(rows))
                    if kind in ("recalls", "complaints"):
                        # Keep the listing's original make/model values in SQLite;
                        # canonical values are used only to form the API request.
                        table = "nhtsa_" + kind
                        conn.execute(f"DELETE FROM {table} WHERE query_id=?", (qid,))
                        conn.executemany(
                            f"INSERT OR REPLACE INTO {table} (query_id,record_key,model_year,make,model,record_json) VALUES (?,?,?,?,?,?)",
                            [(qid, str(j), year, source_make, source_model,
                              json.dumps(row, ensure_ascii=False)) for j, row in enumerate(rows)])
                    else:
                        conn.execute("DELETE FROM nhtsa_safety_rating_values WHERE query_id=?", (qid,))
                        values = []
                        for row in rows:
                            vehicle_id = str(row.get("VehicleId", "unknown"))
                            values.extend((qid, vehicle_id, str(k), None if v is None else str(v))
                                          for k, v in row.items())
                        conn.executemany("INSERT OR REPLACE INTO nhtsa_safety_rating_values VALUES (?,?,?,?)", values)
                    conn.commit()
                except Exception as exc:
                    LOG.warning("%s query failed for %s %s %s: %s",
                                kind, year, source_make, source_model, exc)
                    insert_query(conn, table_kind, year, source_make, source_model,
                                 None, "error", error=str(exc)[:1000])
                    conn.commit()
                time.sleep(delay)
            total += 1
            if total % 100 == 0:
                LOG.info("Queried NHTSA for %s distinct vehicle identities", total)


def connect_vpic(server: str, database: str, driver: str):
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError("Local vPIC decoding requires pyodbc. Install it with: python -m pip install pyodbc") from exc
    connection_string = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        "Trusted_Connection=yes;TrustServerCertificate=yes"
    )
    return pyodbc.connect(connection_string, autocommit=True, timeout=30)


def fetch_vpic(conn: sqlite3.Connection, server: str, database: str, driver: str,
               batch_size: int = VPIC_BATCH_LIMIT, limit: int | None = None) -> None:
    """Decode distinct VINs with the locally restored vPIC SQL Server database."""
    if not 1 <= batch_size <= VPIC_BATCH_LIMIT:
        raise ValueError(f"vPIC batch size must be between 1 and {VPIC_BATCH_LIMIT}")
    mssql = connect_vpic(server, database, driver)
    source = conn.execute("""SELECT vin, MIN(year) FROM used_cars
      WHERE vin IS NOT NULL AND trim(vin) <> '' GROUP BY vin ORDER BY vin""")
    total = processed = 0
    try:
        while True:
            source_rows = source.fetchmany(batch_size)
            if not source_rows or (limit is not None and processed >= limit):
                break
            if limit is not None:
                source_rows = source_rows[:max(0, limit - processed)]
            batch: list[tuple[str, int | None]] = []
            invalid: list[tuple[str, int | None]] = []
            for vin, year_text in source_rows:
                try:
                    year = int(str(year_text).strip())
                except (TypeError, ValueError):
                    year = None
                if len(vin) < 3 or len(vin) > 17 or any(ch in vin.upper() for ch in "IOQ"):
                    invalid.append((vin, year))
                else:
                    batch.append((vin, year))

            try:
                decoded: dict[str, list[tuple[str, str | None]]] = {
                    vin.casefold(): [] for vin, _ in batch
                }
                decoded_keys = set(decoded)
                if batch:
                    placeholders = ",".join("(?)" for _ in batch)
                    sql = ("SET NOCOUNT ON; DECLARE @VinList dbo.tblVinList; "
                           f"INSERT INTO @VinList (Vin) VALUES {placeholders}; "
                           "EXEC dbo.spVinDecodeMultiple @VinList;")
                    cursor = mssql.cursor()
                    cursor.execute(sql, *(vin for vin, _ in batch))
                    result_columns: list[str] | None = None
                    while True:
                        if cursor.description:
                            columns = [column[0] for column in cursor.description]
                            normalized = {re.sub(r"[^a-z0-9]", "", name.casefold()): i
                                          for i, name in enumerate(columns)}
                            vin_col = normalized.get("vin")
                            name_col = normalized.get("variable", normalized.get("variablename"))
                            value_col = normalized.get("value")
                            if vin_col is not None and name_col is not None and value_col is not None:
                                result_columns = columns
                                for row in cursor.fetchall():
                                    source_vin = str(row[vin_col])
                                    key = source_vin.casefold()
                                    if key in decoded_keys:
                                        variable = str(row[name_col])
                                        value = None if row[value_col] is None else str(row[value_col])
                                        decoded[key].append((variable, value))
                        if not cursor.nextset():
                            break
                    if result_columns is None:
                        raise RuntimeError(
                            "spVinDecodeMultiple returned no result set with VIN, Variable, and Value columns; "
                            "check that the restored vPIC database is the expected NHTSA release."
                        )

                for vin, year in invalid:
                    insert_query(conn, "vpic_decode", year, None, None, vin, "invalid_vin")
                for vin, year in batch:
                    values = decoded.get(vin.casefold(), [])
                    qid = insert_query(conn, "vpic_decode", year, None, None, vin,
                                       "success", 1 if values else 0)
                    conn.execute("DELETE FROM nhtsa_vpic_values WHERE vin=? AND model_year_hint IS ?",
                                 (vin, year))
                    conn.executemany(
                        "INSERT OR REPLACE INTO nhtsa_vpic_values"
                        "(vin,model_year_hint,variable_id,variable_name,value) VALUES(?,?,?,?,?)",
                        [(vin, year, variable, variable, value) for variable, value in values])
                conn.commit()
            except Exception as exc:
                conn.rollback()
                for vin, year in batch:
                    insert_query(conn, "vpic_decode", year, None, None, vin,
                                 "error", error=str(exc)[:1000])
                for vin, year in invalid:
                    insert_query(conn, "vpic_decode", year, None, None, vin, "invalid_vin")
                conn.commit()
                LOG.error("Local vPIC batch failed for %s VINs: %s", len(batch), exc)
            processed += len(source_rows)
            total += len(source_rows)
            if total % 10000 == 0 or (limit is not None and processed >= limit):
                LOG.info("Decoded %s unique VINs locally", total)
    finally:
        mssql.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--load-listings", action="store_true",
                        help="Load/replace listing rows from the CSV before querying. Omit for query-only runs.")
    parser.add_argument("--load-only", action="store_true",
                        help="Load listings from CSV and do not query NHTSA or the local vPIC database.")
    parser.add_argument("--limit", type=int,
                        help="Maximum listings to load and/or unique VINs or vehicle identities to query.")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between NHTSA API requests.")
    parser.add_argument("--only", action="append", choices=("all", "recalls", "complaints", "vpic", "safety"),
                        help="Query one source. Repeat for multiple sources; defaults to local vPIC decoding only.")
    parser.add_argument("--vpic-server", default="localhost", help="Local SQL Server instance hosting the restored vPIC backup.")
    parser.add_argument("--vpic-database", default="vPICList_Lite", help="Restored local vPIC database name.")
    parser.add_argument("--vpic-driver", default="ODBC Driver 18 for SQL Server", help="Installed SQL Server ODBC driver name.")
    parser.add_argument("--vpic-batch-size", type=int, default=VPIC_BATCH_LIMIT,
                        help=f"VINs per local vPIC stored-procedure call (1-{VPIC_BATCH_LIMIT}).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")
    if args.load_only and args.only:
        parser.error("--load-only cannot be combined with --only")
    if args.load_only:
        args.load_listings = True
    if args.load_listings:
        if not args.csv.exists():
            parser.error(f"CSV not found: {args.csv}")
    # The default path is local VIN decoding. Remote NHTSA endpoints are opt-in
    # because they are separate, slower data collection tasks.
    selections = [] if args.load_only else (args.only or ["vpic"])
    if "all" in selections and len(selections) != 1:
        parser.error("--only all cannot be combined with other --only selections")
    if "vpic" in selections and not 1 <= args.vpic_batch_size <= VPIC_BATCH_LIMIT:
        parser.error(f"--vpic-batch-size must be between 1 and {VPIC_BATCH_LIMIT}")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db, timeout=60)
    try:
        create_schema(conn)
        if args.load_listings:
            n = load_listings(conn, args.csv, args.limit, args.batch_size)
            LOG.info("Loaded %s listing rows into %s", n, args.db)
        has_listings = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='used_cars'").fetchone()
        if not has_listings:
            parser.error("used_cars table is missing; run once with --load-listings")

        if selections == ["all"]:
            selections = ["recalls", "complaints", "safety", "vpic"]
        catalog = VehicleCatalog()
        for selection in selections:
            if selection in ("recalls", "complaints", "safety"):
                fetch_vehicle_tables(conn, args.delay, (selection,), args.limit, catalog)
            elif selection == "vpic":
                fetch_vpic(conn, args.vpic_server, args.vpic_database, args.vpic_driver,
                           args.vpic_batch_size, args.limit)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
