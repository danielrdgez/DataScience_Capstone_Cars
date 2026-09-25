# Data Dictionary

This document describes the source fields and the SQLite tables created by the
project ingestion scripts. The CSV's original values are retained as text in
`used_cars`; this avoids silently changing source formatting during import.

## Dataset sources

| Source | Description | Location | Grain |
|---|---|---|---|
| Used car listings | [US Used Cars Dataset on Kaggle](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset/data); listing attributes, vehicle identity, price and seller/listing information | `DATA/used_cars_data.csv` | One source listing per row; `listing_id` is the key when populated |
| NHTSA vPIC | VIN decode result values | Local SQL Server restore of `DATA/vPICList_lite_2026_09.bak`, using `dbo.spVinDecodeMultiple` | One value per VIN and returned decode variable; up to 100 VINs per procedure call |
| NHTSA recalls | Recalls by model year, make and model | NHTSA Recalls API `recallsByVehicle`; make/model are validated and canonicalized against the recalls product catalog | One returned record per MMY query |
| NHTSA complaints | Complaints by model year, make and model | NHTSA Complaints API `complaintsByVehicle`; make/model are validated and canonicalized against the complaints product catalog | One returned record per MMY query |
| NHTSA safety ratings | Safety ratings by model year, make and model | NHTSA SafetyRatings API | One returned field per query and vehicle variant |
| YouTube comments | Raw comment records and metadata | `CAR_DATA_FINAL.db`, table `youtube_comments_sentiment` in `E:\Car-Price-Data-Visualization-Learning\CAR_DATA_OUTPUT` | One comment per `comment_id` |

## Used car CSV fields

The following fields are copied to identically named columns in `used_cars`.
They are stored as SQLite `TEXT` to preserve the CSV representation (including
units, blank values, and list-like strings). `year` is interpreted as an
integer only for the NHTSA query key; its stored source value remains text.

| Field | Description / source meaning |
|---|---|
| `vin` | Vehicle identification number supplied with the listing; used for vPIC lookup |
| `back_legroom` | Rear-seat legroom, source value and unit |
| `bed` | Truck bed configuration/description |
| `bed_height` | Bed height, source value and unit |
| `bed_length` | Bed length, source value and unit |
| `body_type` | Listing body-style category |
| `cabin` | Cab configuration |
| `city` | Listing city |
| `city_fuel_economy` | City fuel economy, source units |
| `combine_fuel_economy` | Combined fuel economy, source units |
| `daysonmarket` | Days listed at the source snapshot |
| `dealer_zip` | Dealer postal code |
| `description` | Free-text listing description |
| `engine_cylinders` | Engine cylinder configuration |
| `engine_displacement` | Engine displacement, source units |
| `engine_type` | Engine type/configuration |
| `exterior_color` | Exterior color description |
| `fleet` | Fleet vehicle indicator when supplied |
| `frame_damaged` | Frame damage indicator when supplied |
| `franchise_dealer` | Franchise-dealer indicator |
| `franchise_make` | Franchise make associated with dealer |
| `front_legroom` | Front-seat legroom, source value and unit |
| `fuel_tank_volume` | Fuel tank capacity, source value and unit |
| `fuel_type` | Fuel type |
| `has_accidents` | Reported accident indicator |
| `height` | Vehicle height, source value and unit |
| `highway_fuel_economy` | Highway fuel economy, source units |
| `horsepower` | Engine horsepower |
| `interior_color` | Interior color description |
| `isCab` | Source cab indicator |
| `is_certified` | Certified vehicle indicator |
| `is_cpo` | Certified pre-owned indicator |
| `is_new` | New vehicle indicator |
| `is_oemcpo` | OEM certified pre-owned indicator |
| `latitude` | Listing/dealer latitude |
| `length` | Vehicle length, source value and unit |
| `listed_date` | Listing date supplied by source |
| `listing_color` | Source listing color category/code |
| `listing_id` | Listing identifier; primary key in `used_cars` |
| `longitude` | Listing/dealer longitude |
| `main_picture_url` | Main listing image URL |
| `major_options` | Source representation of major options |
| `make_name` | Listing make; used with `year` and `model_name` for NHTSA queries |
| `maximum_seating` | Maximum seating capacity/description |
| `mileage` | Odometer mileage at listing |
| `model_name` | Listing model; used with `year` and `make_name` for NHTSA queries |
| `owner_count` | Reported prior-owner count |
| `power` | Source engine power description |
| `price` | Asking/listing price, not a confirmed transaction price |
| `salvage` | Salvage indicator/title status when supplied |
| `savings_amount` | Source-reported savings amount |
| `seller_rating` | Seller rating |
| `sp_id` | Source/provider identifier |
| `sp_name` | Source/provider name |
| `theft_title` | Theft-title indicator when supplied |
| `torque` | Source engine torque description |
| `transmission` | Transmission code/type |
| `transmission_display` | Display transmission description |
| `trimId` | Source trim identifier |
| `trim_name` | Trim description |
| `vehicle_damage_category` | Source vehicle damage category |
| `wheel_system` | Drivetrain/wheel system code |
| `wheel_system_display` | Display drivetrain description |
| `wheelbase` | Wheelbase, source value and unit |
| `width` | Vehicle width, source value and unit |
| `year` | Model year; used with make and model for NHTSA vehicle endpoints |

## SQLite table fields

| Table | Fields | Meaning |
|---|---|---|
| `used_cars` | All 66 CSV fields above | Listing-grain source data; `listing_id` primary key |
| `nhtsa_queries` | `query_id`, `query_type`, `model_year`, `make`, `model`, `vin`, `status`, `result_count`, `error`, `fetched_at` | Request ledger; records success/error/invalid VIN and unmatched vehicle catalog keys. For vehicle endpoints one query per source MMY and endpoint; for vPIC one per VIN. |
| `nhtsa_vpic_values` | `vin`, `model_year_hint`, `variable_id`, `variable_name`, `value`, `fetched_at` | Local vPIC decode values stored as text. `variable_id` remains the returned variable name to keep the existing table key and values compatible. |
| `nhtsa_recalls` | `query_id`, `record_key`, `model_year`, `make`, `model`, `record_json` | Full NHTSA recall record serialized as JSON; `record_key` is the row position within its query. |
| `nhtsa_complaints` | `query_id`, `record_key`, `model_year`, `make`, `model`, `record_json` | Full NHTSA complaint record serialized as JSON; `record_key` is the row position within its query. |
| `nhtsa_safety_rating_values` | `query_id`, `vehicle_id`, `field_name`, `field_value` | Safety response flattened to field/value rows so dynamic NHTSA fields are retained. Values are text. |
| `youtube_comments_sentiment` | `video_id`, `playlist_id`, `video_title`, `source`, `text`, `extracted_at`, `comment_id`, `author`, `like_count`, `reply_count`, `published_at`, `updated_at` | Imported raw YouTube comment table. `comment_id` is the primary key; this import does not calculate sentiment. |

## Scripts and run sequence

1. `python NHTSA/build_car_data_db.py --load-only` streams the configured CSV
   into `used_cars` without making queries. Listing values remain unchanged as
   text. Query-only runs do not reload or replace listing rows.
2. Run `python NHTSA/build_car_data_db.py --only recalls`, `--only complaints`,
   or `--only vpic` to query one source. Repeat `--only` flags to run selected
   sources sequentially. The vPIC task uses the restored local SQL Server
   backup and batches up to 100 VINs per stored-procedure call. Recall and
   complaint calls remain remote and use NHTSA's issue-specific make/model
   catalogs to validate and canonicalize query parameters. Use `--limit N` for
   a small trial.
3. `python YOUTUBE/import_youtube_comments.py` copies comments from the
   configured `CAR_DATA_FINAL.db` into the same `DATA/CAR_DATA.db`.

## Transformation and quality notes

- CSV ingestion is batched; original columns are retained as text. Re-running
  replaces rows with a matching non-null `listing_id`.
- NHTSA recall/complaint records are model-year/make/model results. They are not
  confirmed VIN-specific events and should not be joined to each listing as if
  they were per-car histories.
- Safety ratings can contain multiple vehicle variants for one MMY query.
- VIN format screening for vPIC only rejects lengths outside 3–17 and excluded
  letters I/O/Q; it does not verify check digits.
- Recall/complaint keys without an exact normalized make/model match in the
  corresponding NHTSA product catalog are recorded as `unmatched` and are not
  sent to that endpoint. Catalog canonicalization is used only for the request;
  source make/model values in the SQLite result tables are retained unchanged.
- Recall/complaint API failures are recorded in `nhtsa_queries` with the
  response body when NHTSA provides one; re-running retries requests.
- The standalone vPIC backup supports VIN decoding only. Recalls and complaints
  continue to come from their separate NHTSA APIs.
- The local `spVinDecodeMultiple` procedure accepts VINs but no model-year
  override. `model_year_hint` remains the listing's year for reference; the
  local procedure determines decoded values from the VIN.
- YouTube comments are observational collected comments, not a representative
  sample of owners. `text` and `author` are free text / display-name fields.
