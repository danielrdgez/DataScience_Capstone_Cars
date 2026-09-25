# DataScience Capstone Cars

This repository contains the working files for a car-focused data science capstone project.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `DATA/` | Source CSV and generated `CAR_DATA.db` unified SQLite database. Large datasets are ignored by Git by default. |
| `NHTSA/` | `build_car_data_db.py` streams used-car CSV listings and collects NHTSA vPIC, recall, complaint, and safety-rating values. |
| `YOUTUBE/` | `import_youtube_comments.py` copies the raw YouTube comment table into `DATA/CAR_DATA.db`. |
| `EDA/` | Exploratory data analysis notebooks, scripts, and outputs. |
| `MODELS/` | Model training, evaluation, and saved model workflow files. Large model artifacts are ignored by Git by default. |
| `AGENTS.md` | Instructions for Codex and other coding agents working in this repository. |
| `DATA_DICTIONARY.md` | Dataset documentation, field definitions, transformations, and known quality notes. |

## Notes

- Keep project documentation current as the dataset, scripts, and models evolve.
- Store large raw data, processed data, and model artifacts locally unless a tracked sample or metadata file is intentionally added.

## Unified database workflow

The local source listing file is `DATA/used_cars_data.csv`, sourced from the
[US Used Cars Dataset on Kaggle](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset/data).
To load its rows into SQLite without any NHTSA queries, run:

```powershell
.\.venv\Scripts\python.exe NHTSA/build_car_data_db.py --load-only
```

Query commands use the existing `DATA/CAR_DATA.db` listing table and do not
reload or replace listing rows. Use `--load-listings` only when you intend to
load the CSV. Recall and complaint queries are checked against NHTSA's
issue-specific make/model product catalog first. VIN decoding uses the locally
restored vPIC database in batches of up to 100 VINs; it makes no vPIC API calls.

Restore `DATA/vPICList_lite_2026_09.bak` into a local SQL Server 2019 or newer
instance as `vPICList_Lite` (SSMS: Databases → Restore Database → Device →
select the backup). Start the SQL Server service if it is stopped. Install the
[Microsoft ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
and the Python ODBC package:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If starting `MSSQLSERVER` reports event 17051, the SQL Server Evaluation
period has expired. On this machine, SQL Server 2019 Enterprise Evaluation is
installed. From an Administrator session, run the cached SQL Server setup at
`C:\Program Files\Microsoft SQL Server\150\Setup Bootstrap\SQL2019\setup.exe`,
then choose **Maintenance → Edition Upgrade**, select the `MSSQLSERVER`
instance, and change it to the free Developer edition. Developer edition is
licensed for development and test use. If Setup cannot upgrade the expired
instance, install a separate SQL Server Developer instance and restore vPIC
there; pass its instance name with `--vpic-server`.

The project currently needs only `pyodbc` beyond Python's standard library.
The `.venv` folder is local to this checkout and is ignored by Git.

The local SQL Server service was stopped when this setup was prepared. From an
Administrator PowerShell window, start it and run the guarded restore script:

```powershell
Start-Service MSSQLSERVER
& 'C:\Program Files\Microsoft SQL Server\Client SDK\ODBC\170\Tools\Binn\SQLCMD.EXE' -S localhost -E -b -i NHTSA\restore_vpic_database.sql
```

The restore script creates `vPICList_Lite` only when it does not already exist.
If SQL Server reports that it cannot read the `.bak`, copy the backup into a
folder readable by the SQL Server service account and update the `FROM DISK`
path in `NHTSA/restore_vpic_database.sql`.

Run each data source separately:

```powershell
.\.venv\Scripts\python.exe NHTSA/build_car_data_db.py --only recalls
.\.venv\Scripts\python.exe NHTSA/build_car_data_db.py --only complaints
.\.venv\Scripts\python.exe NHTSA/build_car_data_db.py --only vpic
```

Or run the three sequentially in one process, sharing catalog lookups:

```powershell
.\.venv\Scripts\python.exe NHTSA/build_car_data_db.py --only recalls --only complaints --only vpic
```

Use `--limit 100` for a small trial. Set `--vpic-server` and
`--vpic-database` for a named SQL Server instance or a different restored
database. Add `--load-listings` only when the listing table needs to be loaded
or refreshed. Safety ratings remain available with `--only safety`; `--only all`
runs recalls, complaints, safety ratings, then local vPIC decoding. The
YouTube importer adds `youtube_comments_sentiment` from the configured
`CAR_DATA_FINAL.db`:

```powershell
.\.venv\Scripts\python.exe YOUTUBE/import_youtube_comments.py
```

Both scripts write to `DATA/CAR_DATA.db`. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
for source fields, table grains, and known limitations.
