# Data Dictionary: Intended Modeling Dataset

## Dataset identity and grain

The intended model dataset combines six sources: the **US Used Cars Dataset** published on [Kaggle](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset/data), NHTSA vPIC VIN decoding, NHTSA safety ratings, NHTSA complaints, NHTSA recalls, and YouTube comments. The Kaggle dataset is described as used-car listing data collected from CarGurus. Download the CSV from Kaggle and place it at `DATA/used_cars_data.csv`.

The listings are the base data and each source row represents one advertised vehicle listing. `listing_id` identifies a listing when present and is the primary key used by the loader. `price` is the advertised listing price, not a verified sale price. This dataset is a historical snapshot; it should not be interpreted as current market inventory or completed transactions. The final modeling row grain and join/aggregation rules have not yet been implemented and must be made explicit when the integrated dataset is assembled.

The listing import streams the CSV into `used_cars` in `DATA/CAR_DATA.db`. It requires `vin`, `year`, `make_name`, and `model_name`; retains all source columns and values as SQLite `TEXT`; and inserts/replaces rows by `listing_id` (or VIN when no listing ID column exists). NHTSA values are collected separately and YouTube comment records are copied from the pre-existing external database, then all source tables feed the final modeling dataset.

## Modeling source components

| Component | Source and grain | Linkage available for final assembly | Data-specific cautions |
|---|---|---|---|
| Used-car listings | Kaggle CSV at `DATA/used_cars_data.csv`; one row per advertised listing. Fields are listed below. | `vin`; listing `year`, `make_name`, and `model_name`; `listing_id` identifies the source listing. | A VIN can occur on multiple listings. `price` is asking price, not transaction price. |
| NHTSA vPIC | NHTSA standalone SQL Server backup restored as `vPICList_Lite`; variable/value pairs per distinct VIN in `nhtsa_vpic_values`. | VIN. `model_year_hint` stores the listing year for reference; the decoder derives values from VIN. | The standalone backup provides VIN decoding. VIN screening is basic and does not verify check digits. |
| NHTSA safety ratings | NHTSA SafetyRatings API; potentially multiple vehicle variants/vehicles per model-year, make, and model query. | Query key uses listing `year`, `make_name`, and `model_name`; response `VehicleId` distinguishes variants. | Ratings describe queried vehicle variants and should not be assumed to identify a specific listed VIN. |
| NHTSA complaints | NHTSA `complaintsByVehicle` API; zero or more complaint records per model-year, make, and model query. | Query key uses listing year, make, and model after catalog validation/canonicalization. | These are model-level query results, not verified complaint histories for each listed vehicle. Aggregate or represent one-to-many results explicitly. |
| NHTSA recalls | NHTSA `recallsByVehicle` API; zero or more recall records per model-year, make, and model query. | Query key uses listing year, make, and model after catalog validation/canonicalization. | These are model-level query results, not verified recall histories for each listed vehicle. Aggregate or represent one-to-many results explicitly. |
| YouTube comments | Collected by `YOUTUBE/YOUTUBE_COMMENTS_API.py` from playlist/video/comment resources in the YouTube Data API. An already-collected table is also available in `E:\Car-Price-Data-Visualization-Learning\CAR_DATA_OUTPUT\CAR_DATA_FINAL.db` and can be copied with the project importer. One comment per `comment_id`. | The API collector writes into `youtube_comments_sentiment` in `DATA/CAR_DATA.db`; the importer copies the existing external table there. Comment rows include `video_id`, `playlist_id`, and `video_title`, not listing VINs. A vehicle-to-video mapping/aggregation rule is needed to connect comments to listings. | Comments are user-generated text around selected videos and are not a representative sample of vehicle owners. The collector stores comment text and metadata; it does not calculate sentiment scores. |

The project-local collector discovers videos from its configured playlists through the YouTube Data API, obtains video titles and top-level comments, records extraction timestamps and persistent fetch status, and saves comment records. The pre-existing external database contains previously collected comments; importing it copies those records and does not rerun collection. Neither workflow calculates sentiment scores.

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

## Modeling considerations

- The table grain is a listing, so the same VIN can appear in multiple rows. Decide whether the modeling question is listing-level or vehicle-level before splitting training and evaluation data; repeated VINs across splits can leak vehicle-specific information.
- vPIC values are VIN-level, safety/complaint/recall data are model-year/make/model-level, and YouTube comments are video-level. Define aggregation and linkage before joining these sources so one-to-many records do not multiply listing rows or imply unsupported vehicle-specific facts.
- A vehicle-to-video association is not provided by the YouTube comment table itself. Record the mapping source and rule used to connect video/comment features to a vehicle or listing.
- `price` is the natural candidate target for a listing-price model, but the project has not yet defined a final target, inclusion rules, or feature set. Keep those choices explicit in the model workflow.
- Numeric and categorical values are stored as source text on import. Parsing, missing-value handling, unit normalization, and category cleanup belong in a documented modeling transformation.
- Free-text fields such as `description`, image URLs, and seller/location fields may need to be excluded or specially processed depending on the model objective.
- The source describes listings and asking prices. Predictions should be interpreted as estimates of listed prices within this dataset's collection period and coverage.
