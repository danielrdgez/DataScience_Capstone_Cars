"""Refresh saved-Parquet profiles without changing data or field definitions."""

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import tempfile

import duckdb
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "used_cars", "NHTSA_vPIC", "NHTSA_Complaints", "NHTSA_Recalls",
    "NHTSA_Safety", "Youtube_Comments",
)
COLOR_COLUMNS = {"exterior_color", "interior_color", "listing_color"}
COLOR_PLACEHOLDERS = {"none", "null", "unknown", "n/a", "na", "not applicable"}


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


def missing_condition(table, name, dtype):
    column = quoted(name)
    conditions = [f"{column} IS NULL"]
    if dtype == pl.String:
        conditions.append(f"regexp_full_match({column}, '\\s*')")
        if table == "used_cars" and name in COLOR_COLUMNS:
            labels = ", ".join(f"'{label}'" for label in sorted(COLOR_PLACEHOLDERS))
            conditions.append(f"lower(trim({column})) IN ({labels})")
        if table == "used_cars" and name == "maximum_seating":
            conditions.append(f"trim({column}) = '--'")
    if dtype.is_float():
        conditions.append(f"NOT isfinite({column})")
    if table == "NHTSA_Safety":
        if "Rating" in name:
            conditions.append(f"trim(cast({column} AS VARCHAR)) IN ('0', 'Not Rated')")
        if name in {"RolloverPossibility", "RolloverPossibility2"}:
            rating = quoted("RolloverRating2" if name.endswith("2") else "RolloverRating")
            conditions.append(f"({column} = 0 AND ({rating} IS NULL OR {rating} = 0))")
    return "(" + " OR ".join(conditions) + ")"


def example_cell(values):
    rendered = []
    for value in values:
        if isinstance(value, bool):
            value = str(value).lower()
        value = str(value).replace("\r", r"\r").replace("\n", r"\n")
        if len(value) > 110:
            value = value[:107] + "..."
        rendered.append("<code>" + escape(value).replace("|", "&#124;") + "</code>")
    return "<br>".join(rendered) if rendered else "(no non-missing values)"


def refresh_table(connection, dataset, table, document):
    path = dataset / f"{table}.parquet"
    original_stat = path.stat()
    schema = pl.scan_parquet(path).collect_schema()
    source = "read_parquet('" + path.as_posix().replace("'", "''") + "')"
    rows = connection.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    metadata = next(info for info in manifest['tables'].values() if info['file'] == path.name)
    if rows != metadata['rows'] or {name: str(dtype) for name, dtype in schema.items()} != metadata['schema']:
        raise ValueError(f"Saved data differs from export manifest: {table}")
    section_pattern = rf"(## `{re.escape(table)}`\n)(.*?)(?=\n## |\Z)"
    match = re.search(section_pattern, document, re.S)
    if match is None:
        raise ValueError(f"Missing dictionary section: {table}")
    section = match.group(2)
    documented = re.findall(r"^\| `([^`]+)` \| `[^`]+` \|", section, re.M)
    if documented != list(schema):
        raise ValueError(f"Dictionary fields differ from saved schema: {table}")
    stats = {}
    names = list(schema)
    for start in range(0, len(names), 4):
        expressions = []
        for name in names[start:start + 4]:
            missing = missing_condition(table, name, schema[name])
            expressions.extend([
                f"count(*) FILTER (WHERE {missing})",
                f"count(DISTINCT CASE WHEN NOT {missing} THEN {quoted(name)} END)",
            ])
        measured = connection.execute(f"SELECT {', '.join(expressions)} FROM {source}").fetchone()
        for index, name in enumerate(names[start:start + 4]):
            missing = missing_condition(table, name, schema[name])
            candidates = connection.execute(
                f"SELECT DISTINCT {quoted(name)} FROM {source} WHERE NOT {missing} LIMIT 5"
            ).fetchall()
            stats[name] = (measured[index * 2], measured[index * 2 + 1], [v[0] for v in candidates])
        print(f"{table}: profiled {min(start + 4, len(names))}/{len(names)} fields", flush=True)
    if (path.stat().st_size, path.stat().st_mtime_ns) != (original_stat.st_size, original_stat.st_mtime_ns):
        raise ValueError(f"Source file changed during profiling: {table}")

    def replace_row(row_match):
        cells = row_match.group(0).strip("| ").split(" | ")
        name = cells[0].strip("`")
        missing, distinct, examples = stats[name]
        cells[1] = f"`{schema[name]}`"
        cells[4:] = [f"{missing:,}", f"{rows - missing:,}", f"{distinct:,}", example_cell(examples)]
        return "| " + " | ".join(cells) + " |"

    section = re.sub(r"^\| `[^`]+` \| `[^`]+` \|.*$", replace_row, section, flags=re.M)
    section = re.sub(r"\*\*[\d,]+ rows; [\d,]+ columns\*\*", f"**{rows:,} rows; {len(schema)} columns**", section)
    document = document[:match.start(2)] + section + document[match.end(2):]
    inventory = rf"^\| `{re.escape(table)}` \| [^\n]+$"
    def replace_inventory(row_match):
        cells = row_match.group(0).strip("| ").split(" | ")
        cells[1:4] = [f"{rows:,}", str(len(schema)), f"{path.stat().st_size / 2**20:.2f}"]
        return "| " + " | ".join(cells) + " |"
    document = re.sub(inventory, replace_inventory, document, count=1, flags=re.M)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "DATA/CAR_DATA_FINAL")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "DATA_DICTIONARY.md")
    parser.add_argument("--table", action="append", choices=TABLES)
    args = parser.parse_args()
    document = args.dictionary.read_text(encoding="utf-8")
    tables = args.table or list(TABLES)
    with tempfile.TemporaryDirectory(prefix="car-dictionary-") as scratch:
        with duckdb.connect(":memory:") as connection:
            connection.execute("SET memory_limit='1GB'")
            connection.execute("SET threads=1")
            connection.execute("SET preserve_insertion_order=true")
            connection.execute("SET temp_directory=?", [scratch])
            for table in tables:
                document = refresh_table(connection, args.dataset, table, document)
    timestamp = datetime.now(timezone.utc).isoformat()
    note = f"**Profile verification:** {timestamp}; refreshed {', '.join('`' + t + '`' for t in tables)} from saved Parquet."
    if re.search(r"^\*\*Profile verification:\*\*.*$", document, re.M):
        document = re.sub(r"^\*\*Profile verification:\*\*.*$", note, document, flags=re.M)
    else:
        document = document.replace("\n## Classification guide", "\n" + note + "\n\n## Classification guide", 1)
    args.dictionary.write_text(document, encoding="utf-8")
    print(f"Updated {args.dictionary}; source data unchanged.", flush=True)


if __name__ == "__main__":
    main()
