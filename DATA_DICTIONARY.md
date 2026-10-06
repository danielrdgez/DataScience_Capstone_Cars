# Project Data Dictionary

This dictionary profiles the six model-development tables saved in `DATA/CAR_DATA_FINAL/`. See [EDA/DATA_CLEANING.py](EDA/DATA_CLEANING.py) for transformations and filename mappings. Table names below omit the file extension. The saved export manifest is dated **2026-09-29T19:10:32.788840+00:00**. All six saved schemas and row totals were verified against that manifest during the profile refresh below. The listing snapshot excludes null mileage and VINs selected by the cohort-ratio policy; source SQLite and Parquet files were not modified by this documentation refresh.

**Column** is the exact saved column name. **Type** is the Polars dtype read from Parquet. **Variable classification** describes the field's statistical meaning, independently of its storage dtype.

**Current listing profile:** 2,792,999 rows and 59 columns. Missing, present, distinct, and example values were recomputed from the saved Parquet files for all six source tables. Derived comparison features and expert-reference tables retain their separately dated profiles below.

**Profile verification:** 2026-10-06T12:30:25.803757+00:00; refreshed `used_cars`, `NHTSA_vPIC`, `NHTSA_Complaints`, `NHTSA_Recalls`, `NHTSA_Safety`, `Youtube_Comments` from saved Parquet.

## Classification guide

| Classification | Meaning and modeling interpretation |
|---|---|
| Categorical: nominal | Unordered labels, such as make, model, fuel type, or equipment availability. |
| Categorical: binary nominal | Two meaningful states, such as true/false. |
| Categorical: ordinal | Ordered levels; numeric-looking codes do not establish equal distances. |
| Categorical: ordinal + status | Ordered rating levels may coexist with source statuses such as `Not Rated`. |
| Identifier: nominal | Keys/codes used for linkage or identity; integer storage does not make them measures. |
| Quantitative: discrete | Counts or quantities recorded in whole units. |
| Quantitative: continuous | Physical measurements, monetary amounts, averages, or probabilities. |
| Temporal: discrete year (interval) | Calendar/model years; differences can be meaningful, but ratios are not. |
| Temporal: date/time | Dates or timestamps; some remain stored as strings. |
| Text / structured text | Narratives, compound measurements, or JSON; require explicit feature extraction. |
| Reference: URL | A link to media, not a numeric measurement or encoded model feature. |
| Unspecified | Legacy fields whose meaning was not established by available source documentation. |

**Profile rules:** Missing Values counts nulls, empty/whitespace-only text, and non-finite floating-point values; table-specific placeholders are also described below. Total rows with values equals rows minus missing rows. Distinct values and examples exclude those missing values. Examples show up to five distinct values from the source scan, not a random sample. These documentation rules do not alter saved values.

- **Listings:** `--` is missing for `maximum_seating`; selected `None`, `NULL`, `Unknown`, `N/A`, `NA`, and `Not Applicable` labels are missing in color fields. Numeric conversion failures become null; fractional integers round to nearest with ties away from zero; ZIP codes lose leading zeros. Export excludes null mileage and review-file VINs with price/cohort-median ratio above 2 or at/below 0.75, or mileage/cohort-median ratio above 20. Modeling also excludes invalid target prices and imputes remaining predictor missingness using training data.
- **vPIC:** `vpic_` prefixes are removed and selected numeric fields are converted to fixed types. Missing decoded values do not establish that equipment is absent; explicit `Not Applicable`, `Not Available`, `No`, `Standard`, and `Optional` values remain distinct. `fetched_at` is converted only if the whole source column parses.
- **Complaints and recalls:** top-level JSON values become columns and raw `record_json` is dropped afterward. Missing keys and explicit JSON null become null; arrays/objects such as `products` remain JSON text. Malformed JSON stops the export. `query_id`, `record_key`, and model-year fields are nullable `Int64`; complaint `dateComplaintFiled` and `dateOfIncident` parse from `MM/DD/YYYY` to nullable `Date`, and recall `ReportReceivedDate` parses from `DD/MM/YYYY` to nullable `Date`. Invalid or blank requested numeric/date values become null. `false`, zero counts, and partial complaint VINs remain values.
- **Safety:** one row is saved per `(query_id, vehicle_id)` with field names as columns. The requested identifiers, counts, years, and numeric ratings are nullable `Int64`; `RolloverPossibility` and `RolloverPossibility2` are nullable `Float64`. Unparseable values (including non-numeric rating statuses) become null during conversion. For profile missingness only, `Not Rated` and `0` count as missing for star ratings; zero rollover probability is treated as missing when its paired rating is unavailable. These profile rules do not alter saved Parquet values.
- **YouTube:** `youtube_comments_sentiment` is exported without sentiment transformation. Date fields remain source text; comment text is not a sentiment score.
- Configured excluded fields are not present in any exported table. No cross-table joins, complaint/recall aggregation, imputation, or feature selection are performed.

## Exported dataset inventory

| Table | Rows | Columns | Size (MiB) | Row grain |
|---|---:|---:|---:|---|
| `used_cars` | 2,792,999 | 59 | 1175.22 | One exported listing row after type conversion, null-mileage exclusion, and cohort-ratio VIN exclusion. |
| `NHTSA_vPIC` | 2,999,875 | 133 | 22.23 | One source VIN/model-year-hint decode row. |
| `NHTSA_Complaints` | 1,390,528 | 17 | 239.59 | One query/record-key complaint record; top-level JSON keys are separate columns. |
| `NHTSA_Recalls` | 26,993 | 20 | 1.95 | One query/record-key recall record; top-level JSON keys are separate columns. |
| `NHTSA_Safety` | 8,144 | 35 | 0.26 | One query/vehicle-variant pair; rating fields are wide columns. |
| `Youtube_Comments` | 1,981,686 | 12 | 166.87 | One imported YouTube comment record. |

**Available auxiliary expert baseline:** DougScore is selected as the separate
professional-opinion reference for YouTube feature development. Its imported
`expert_review_scores.parquet` has 585 review records and 46 columns, and
`expert_aspect_scores.parquet` has 2,925 review/aspect records and ten columns. These existing
tables live under `EXPERT_REVIEWS/outputs/dougscore_v1/`, outside the six-file
source snapshot; the current price model does not consume them. Full metric,
identity, availability, and proposed feature definitions are in
[DougScore expert-opinion baseline](#dougscore-expert-opinion-baseline).

## Price-model complaint/recall aggregates

`MODELS/Price_ML_Models.py` builds these features in memory from `NHTSA_Complaints.parquet` and `NHTSA_Recalls.parquet`; it writes no intermediate feature files. The NHTSA grain is normalized (`model_year`, `make`, `model`) and joined to listings only after the VIN join to `NHTSA_vPIC`. The four vPIC cohort columns, including `Trim`, remain separate in the listing dataset; NHTSA features describe the broader year/make/model group.

The inclusive cutoff is `2020-12-31`. This is a provisional evidence cutoff because the listing price observation date is unverified; these features support a retrospective association comparison, not a historical prediction claim. Complaint `dateComplaintFiled` and `dateOfIncident` are saved as `Date` values parsed from `MM/DD/YYYY`; recall `ReportReceivedDate` is saved as `Date` parsed from `DD/MM/YYYY`. Missing/invalid dates and records after cutoff are excluded. Complaint `products` stays JSON text in Parquet. If it contains explicit `Vehicle` entries, at least one must exactly match the row's year, make, and model; equipment/tire-only entries do not establish vehicle applicability.

Complaint IDs (`odiNumber`) are counted distinctly within cohort. Duplicate IDs count once; if duplicate versions conflict in summary, crash/fire flags, injury/death counts, or components, the ID remains in the total count but is excluded from those structured summaries. Recall campaigns are counted distinctly by `NHTSACampaignNumber` within cohort. Report flags and reported injury/death counts describe submitted reports, not population failure risks. No ratios or rates are derived.

| Feature | Type / meaning | Interpretation |
|---|---|---|
| `complaint_count` | Quantitative: discrete | Distinct eligible complaint IDs associated with this cohort. |
| `crash_report_count`, `fire_report_count` | Quantitative: discrete | Distinct eligible complaints with the source flag set true. |
| `injury_report_count`, `death_report_count` | Quantitative: discrete | Distinct eligible complaints reporting at least one injury/death. |
| `reported_injuries`, `reported_deaths` | Quantitative: discrete | Sum of reported counts across distinct non-conflicting complaints. |
| `component_count` | Quantitative: discrete | Distinct component labels across eligible non-conflicting complaints. |
| `recall_campaign_count` | Quantitative: discrete | Distinct eligible NHTSA campaign identifiers associated with this cohort. |
| `complaint_source_records_present`, `recall_source_records_present` | Quantitative: binary indicator | `1` means source rows for this cohort are present in the saved Parquet; `0` means none are present. This is not query success/coverage, and absence is not a confirmed zero. |

Counts are null where no qualifying aggregate can be derived from saved records. Training-only imputation and missingness indicators preserve that distinction. Source Parquet snapshots remain unchanged. See feature-set definitions and comparison output in `MODELS/Price_ML_Models.py` and `MODEL_OUTPUT.ipynb`.

## Experimental NHTSA text features (outside the source snapshot)

`NHTSA/text_features.py` uses the existing complaint/recall Parquet files without modifying their fields or the six-table model-development snapshot. Derived files are stored separately under `MODEL_OUTPUT/nhtsa_text/<workspace>/`. They are experimental candidates for price modeling, not new production price inputs. The automatic route uses source components, text-support features, and unlabeled TF-IDF/SVD or embedding representations; human annotation is optional and applies only to named semantic labels.

The experimental comparison selects predictor subsets from the same frozen listing dataset. Every non-base subset includes all selected vPIC attributes. `base_features` in the legacy `workflow.json` names the 113-column fully structured input list, not the listing-only base; new comparison metadata records each actual subset explicitly.

| Experimental feature set | Predictor sources and current pilot width |
|---|---|
| `base` | 39 listing and deterministic listing-derived fields; decoded vPIC fields are retained only for audit/grouping, not predictors. |
| `vpic_enriched` | All 39 base fields plus 54 selected vPIC fields, including the truck and battery/electrification fields: 93 predictors. |
| `nhtsa_structured` | All vPIC-enriched fields plus eight complaint values, one complaint presence flag, one recall count, one recall presence flag, and nine safety fields: 113 predictors. This equals the old comparison's `enriched` set. |
| `complaints` | vPIC-enriched fields plus complaint values/presence, eight complaint component shares, four complaint support fields, and complaint-summary vectors when requested. Recall and safety fields are excluded. |
| `recalls` | vPIC-enriched fields plus recall count/presence, eight recall component counts, four recall support fields, and summary/consequence/remedy vectors when requested. Complaint and safety fields are excluded. |
| `safety` | vPIC-enriched fields plus the nine safety fields: 102 predictors. |
| `components`, `text_support` | Full structured inputs plus the 16 component fields (129 predictors), then eight automatic support fields (137 predictors). Annotated builds can additionally contain qualified labels. |
| `tfidf`, `embeddings` | Full structured, component, and support inputs plus the four narrative-vector channels. The first real TF-IDF run has 64 vector coordinates and 201 total predictors. |
| `interpretable` | Full structured, component, support, and qualified narrative-label inputs from an annotated build. |

Widths count raw selected predictors; preprocessing adds numeric-missingness indicators. Source joins, row eligibility, lowercase VIN linkage, and fit/validation/test assignments remain identical across predictor subsets. Individual source sets measure an addition relative to vPIC; the three-set base/vPIC/full-text design measures a combined addition, while including `nhtsa_structured` separately isolates the text/component/support increment.

The default inclusive evidence cutoff is `2020-12-31`. Eligibility uses complaint `dateComplaintFiled` and recall `ReportReceivedDate`, never incident date as an availability substitute. Listing-price observation dates and historical narrative versions remain unverified. The interpretation is retrospective association; a received-date filter alone does not establish historical availability of current text or enriched safety values.

Records are associated at separate normalized `ModelYear`, `Make`, and `Model` fields, with case/whitespace normalization only. Complaints must pass the existing `products` vehicle-applicability check. Recall returned identity must agree with the query identity. Repeated complaint IDs and campaign IDs are resolved within family; conflicting narrative/component versions are excluded from text aggregation. Complaint conflict checks also include crash/fire/injury/death fields. A campaign may legitimately map to multiple families and counts once within each. Complaint partial VINs do not establish listing-level identity. Listing linkage remains lowercase `vin` to `NHTSA_vPIC`, retaining all four separate cohort fields including `Trim`.

| Derived artifact | Grain and data definitions |
|---|---|
| `listings.parquet` | Frozen eligible listing sample with current enriched predictors, positive advertised `price`, lowercase `vin`, four raw cohort columns, grouping/audit fields, and `partition` (`fit`, `validation`, `test`). `split_group` follows the selected unseen-family or known-family strategy. |
| `records.parquet` | One non-conflicting `(ModelYear, Make, Model, source, source_id)` association. `source` is `complaint` or `recall`; `source_id` is a string ODI number or campaign identifier. `evidence_date` is filing/receipt date. Unified `summary`, `consequence`, `remedy`, and `components` preserve the corresponding source text after whitespace trimming; absent fields are empty strings. |
| `text_hash`, `document_id` | SHA-256 of lowercase/whitespace-normalized narrative fields, and a document identifier combining source, source ID, and text hash. These are provenance/grouping keys, never numerical predictors. A shared campaign document can have multiple family associations. |
| `annotation_sample.parquet`, `labels.csv` | Optional frozen source/sample records and their editable annotation copy, unused by the automatic route. One sampled source ID, selected from price fit families. `annotation_split` is development or holdout and groups normalized exact narrative matches. `suggested_labels` are unvalidated rules; `human_labels` are exhaustive pipe-separated positive labels or explicit `none`, conditional on `review_status=reviewed`. Positive annotations require verbatim `evidence`; blank annotations are unfinished. `language` is English/other/unclear; only reviewed English records enter extraction evaluation. No inferred annotation is a human label. |
| `family_features.parquet` | One `(ModelYear, Make, Model, as_of_date)` row with numerical component and support features; annotated builds also include qualified narrative labels. Automatic builds have 16 component columns and eight support columns. These describe broader family evidence and are not trim-specific or individual-VIN defect/repair status. |

The component vocabulary is powertrain, brakes, steering, electrical, airbags, fuel, structure, and visibility. It maps source component text using the explicit patterns in the script. Multiple component labels can apply, so component shares need not sum to one. Other components remain outside the eight indicators rather than being relabeled as known categories.

| Feature pattern | Definition and interpretation |
|---|---|
| `complaint_component_<category>_share` | Fraction of distinct eligible, non-conflicting family complaint IDs whose source component categories match the named category. |
| `recall_component_<category>_campaign_count` | Distinct eligible family campaigns whose component description matches the named category. |
| `complaint_<label>_share` | Fraction of distinct family complaints with a usable summary (at least 20 characters) assigned the qualified symptom/disruption label. Labels include stall, oil leak, brake/steering loss, transmission/electrical failure, towing, repeated failure, repair delay, and unsuccessful repair. These are shares of submitted narratives, not vehicle failure probabilities. |
| `recall_<label>_campaign_count` | Count of distinct family campaigns assigned a qualified label in its corresponding usable field (at least 20 characters). Summary labels describe mechanisms; consequence labels describe potential hazards; remedy labels describe corrective actions or explicit unavailability. No usable field produces null, not a confirmed zero. |
| `<source>_eligible_record_count` | Number of distinct eligible non-conflicting family records available to text preparation. |
| `<source>_usable_summary_count`, `<source>_usable_summary_share` | Count and fraction of those records with a summary at least 20 characters long. Character length is a usability heuristic, not proof of language or relevance. |
| `<source>_low_text_support` | `1` when usable summary count is below 20, otherwise `0`, conditional on a source aggregate being present. |
| `tfidf_<channel>_<index>`, `embeddings_<channel>_<index>` | Mean document-vector coordinate within family, separately for complaint, defect, hazard, and remedy channels. SVD/PCA axes are learned numerical representations, not named severity scales or calibrated probabilities. Missing channel evidence remains null. |

Automatic builds have no named narrative labels, no label-file dependency, and no claim of semantic-label accuracy. Their `feature_manifest.json` records `method=automatic`, an empty `qualified_labels` list, null `label_sha256`, and `semantic_label_accuracy=not_evaluated`. Preparation and feature-building code hashes are distinct provenance fields. Component features come from source categories rather than inferred narrative labels.

For optional annotated builds, only labels meeting the documented holdout support/precision/recall rules appear; their manifests record actual qualified labels and columns. Rules are binary matches with limited negation/hypothetical-context handling. Optional supervised labels use fixed-threshold logistic predictions trained on reviewed development rows. Precision is established only for reviewed English pilot records; corpus-wide language coverage and paraphrase handling remain unverified.

Vector vocabularies and reducers use only documents associated with price fit families during validation; final refitting may use fit plus validation families. Vectors require no human labels. Different-channel vectors are not concatenated into one narrative before encoding. Text cleanup masks the record's own make/model and selected identifiers, but can leave other vehicle identities in a multi-vehicle recall. Means can dilute rare hazards; qualified hazard indicators can supplement vectors in an annotated build.

Source-free or ineligible families receive null feature values through a validated many-to-one join; training-only imputation preserves missingness. Absence does not establish zero defects or successful-empty collection. Remedy text does not establish repair completion, open-recall status, expected owner expense, or an ordinal repair-cost/severity score. Reported allegations and potential recall consequences remain distinct from verified mechanical condition.

**Prepared pilot profile (2026-10-02):** `pilot_v1` contains 50,000 eligible listings: 31,542 fit, 9,834 validation, and 8,624 test rows under the unseen-family strategy. Its source hashes and eligibility details are recorded in `workflow.json`.

| Derived pilot population | Rows / identifiers |
|---|---|
| Complaint family records | 531,342 associations; 531,340 distinct ODI IDs. Of 532,072 distinct eligible mapped family IDs before conflict exclusion, 730 were conflicting. |
| Recall family records | 10,845 associations; 3,117 distinct campaign IDs; no within-family conflicting versions found. |
| Complaint annotation sample | 400 rows: 320 development, 80 extraction holdout. |
| Recall annotation sample | 200 rows: 160 development, 40 extraction holdout. |

The automatic build `features_automatic` contains 2,914 family rows and 24 numerical features: 16 component features and eight text-support features, alongside the three family keys and `as_of_date`. All 600 human annotations remain pending and are unused by this build. The first real comparison fitted 16 TF-IDF/SVD coordinates per channel, using 20,000 complaint documents, 2,510 recall-summary documents, 2,504 consequence documents, and 2,511 remedy documents associated with fit families. Its 64 vector coordinates are generated during comparison and its encoders are saved with the run. The price test remains unscored. These counts describe the separate experimental workspace, not a replacement profile for the six source tables.

### Price-weighting experiment and evaluation fields

The completed `pilot_v2_price_weights` sample contains 100,000 uniformly VIN-hash-selected eligible listings: 63,045 fit, 16,548 validation, and 20,407 test rows. Its automatic feature build and 24-candidate validation comparison are saved; its price test remains unscored. Full NHTSA preparation selects all eligible listings with sample size zero. Sources, eligibility, lowercase VIN linkage, separate cohort dimensions, predictor definitions, and family splitting are unchanged. The earlier `pilot_v1` counts above describe that preserved workspace.

`price_weighting` identifies a training treatment, never a predictor: `none` uses equal row weights; `high_price` uses `clip(training_price / 50000, 1, 5)`. Weights are calculated only for the labels in the current fit partition (fit plus validation during final refitting), never from test labels. The log-price target and all unweighted evaluation denominators remain unchanged. The fixed reference is $50,000 and the maximum relative training weight is 5. The underlying model-development Parquet tables gain no fields or row transformations from this experiment.

Derived evaluation tables add `validation_r2` or `r2` on the original dollar-price scale, `validation_rmse`, `wape_pct` (100 times total absolute error divided by total actual price), and `mean_signed_error` (prediction minus actual). R² is null when fewer than two rows or no actual-price variation make it undefined; negative values remain negative. New price-band reports use $25,000 intervals starting at zero and continuing above $100,000 without a ceiling. `price_band_lower_usd=floor(actual_price/25000)*25000`; `price_band_upper_usd=lower+25000`; intervals are `[lower, upper)`. The display label `100k-125k` includes exactly $100,000 and excludes $125,000. `price_tier` separately identifies `below100k` and `100k+` (actual price at least $100,000). Preserved older reports keep their original intervals. Listing make is reported under `make_name`; decoded make remains separately reported under `Make`. These are reporting dimensions, not predictors derived from the target.

Segment reports retain `rows`, `family_blocks`, `median_actual_price`, and `support_status`; `low_support` means fewer than 100 rows or five distinct normalized year/make/model family blocks. No rows for a make/band produce no segment row rather than zero error. Matched weighting comparisons have one `(variant, model, dimension, segment)` row with ordinary/weighted metric columns and `*_change` defined as weighted minus ordinary; source tables and predictor schema are unaffected.

### Selected YouTube predictors in the full comparison dataset

The derived comparison assembly additionally joins `YOUTUBE/outputs/price_features_v1/make_features.parquet` by normalized canonical decoded `Make`, after the listing-to-vPIC linkage on lowercase `vin`. The selected source view is `explicit_comment` attribution and the `cumulative` window at `2026-10-01`; it contains 83 unique make rows, of which 73 pass the existing semantic/topic support requirement. These fields describe makes, not individual year/make/model/trim cohorts. All four cohort dimensions remain separate. Listings and source Parquet schemas are unchanged; the derived columns are assembled during comparison. The selected manifest records source/export hashes, field blocks, cutoff, and sentiment limitations. These are retrospective candidate inputs with `validation_status=not_independently_validated`; the text was collected/fitted after the historical listing period.

| Selected field block | Definitions and missingness |
|---|---|
| `yt_feature_row_present` | Binary: 1 when a selected canonical-make row matches, 0 for an unmatched listing. This is a saved-feature match indicator, not YouTube collection completeness. |
| `yt_comment_count`, `yt_eligible_count`, `yt_video_count` | Existing deduplicated comment count, eligible comment count, and eligible video count for the selected make/view. Unmatched makes receive null counts, not confirmed zero. |
| `yt_support_suppressed`, `yt_semantic_missing`, eight `yt_<aspect>_topic_missing` flags | Existing support/missing flags. An unmatched make sets these flags true. Suppressed semantic/topic measurements stay null. Support requires at least 20 eligible comments and five eligible videos. |
| `yt_semantic_01` through `yt_semantic_08` | Existing training-reducer semantic coordinates, balanced over videos; learned representation axes, not named sentiment or severity values. |
| Eight `yt_<aspect>_mention_share` values | Video-balanced discussion shares for design, performance, comfort, technology, reliability, maintenance cost, fuel economy/range, and safety. These measure discussion, not approval. |
| Four `yt_<aspect>_video_balanced_net` values | Existing `(positive-negative)/(positive+negative+neutral+mixed)` per-video nets averaged across supported opinion videos for comfort, maintenance cost, performance, and reliability. Unsupported scores remain null. |
| Four each of `yt_<aspect>_opinion_count`, `yt_<aspect>_opinion_video_count`, `yt_<aspect>_missing` | Opinion support counts and existing suppression indicators. An unmatched make has null counts and true missing flags. Opinion support requires 20 resolved comments and five opinion videos. |

The export has 46 candidate predictors: 14 coverage/support fields, 16 semantic/topic values, and 16 aspect sentiment/support fields. Current selected sentiment has no populated comfort, maintenance-cost, or performance scores and only one populated reliability score, equal to zero. The manifest marks `directional_sentiment_available=false`; any measured increment of that block may reflect opinion support/missingness, not directional sentiment. No cached classifications, confidence thresholds, or source opinions are altered by selection.

The four main comparison subsets are `base`, `enriched` (alias of `nhtsa_structured`), `nhtsa_text` (the chosen narrative representation plus structured/component/support inputs), and `nhtsa_text_youtube` (all 46 selected YouTube inputs in addition). Optional `nhtsa_text_youtube_coverage` and `nhtsa_text_youtube_topics` isolate the 14-field coverage block and its 16 semantic/topic additions. Safety contributes its nine structured fields; no safety narrative is present in the selected source snapshot. Training-only price imputation/scaling remains unchanged. The externally fitted YouTube coordinates are fixed 2026 descriptors, with the retrospective limitation recorded rather than a claim that they were available before historical prices.

Derived `make_price_metrics.csv` reports every decoded make in the prepared dataset, with `Make`, `price_band`, and `price_tier` separate. Overall and tier rows for a make absent from an evaluation partition have `rows=0`, null errors, and `support_status=no_rows`; empty make-by-band combinations have no row. The matching `validation_*` file uses the capped tuning-validation sample. `partition_make_price_support.csv` describes the full frozen partitions, and `youtube_join_coverage.csv` reports match/support counts by partition and make. Supplementary focus tables show the named example makes without limiting the all-make reports. Matched feature comparisons define `*_change` as candidate minus reference on the same rows.

**Eligibility profile measured 2026-10-05:** the saved listing snapshot contains 2,792,999 unique lowercase VIN rows meeting the loader's positive-price, non-null-mileage, and cohort-review exclusion rules. It includes 14,896 rows at or above $100,000. Example decoded make counts are Toyota 227,078 (two at $100k+), Ferrari 1,026 (852 at $100k+), and Lamborghini 339 (309 at $100k+). The full workspace `full_youtube_20261006T015226424766Z`, verified 2026-10-06, freezes 1,720,358 fit, 496,219 validation, and 576,422 test rows. Completed full validation/test reports are not present at this check; preparation counts do not establish model performance.

## Conversion nulls recorded by the export

These are additional nulls introduced by conversion in the saved export manifest, before mileage and cohort-ratio VIN row filters. They are not total missingness in the saved output. In particular, the 144,387 converted-null mileage values were excluded from exported listings.

| Column | Additional nulls in `used_cars` |
|---|---:|
| `back_legroom` | 242,724 |
| `bed_length` | 2,579,826 |
| `city_fuel_economy` | 491,278 |
| `dealer_zip` | 209 |
| `engine_displacement` | 172,383 |
| `front_legroom` | 175,454 |
| `fuel_tank_volume` | 160,674 |
| `height` | 159,734 |
| `highway_fuel_economy` | 491,278 |
| `horsepower` | 172,383 |
| `length` | 159,723 |
| `mileage` | 144,387 |
| `owner_count` | 1,516,996 |
| `seller_rating` | 40,872 |
| `wheelbase` | 159,699 |
| `width` | 159,747 |

## `used_cars`

One source listing row after type conversion, null-mileage exclusion, and cohort-ratio VIN exclusion. **2,792,999 rows; 59 columns.**

| Column | Type | Variable classification | Definition / interpretation | Missing Values | Total rows with values | Distinct values | 5 distinct values |
|---|---|---|---|---:|---:|---:|---|
| `vin` | `String` | Identifier: nominal | Source identifier `vin`; treat as a key rather than a numeric quantity. | 0 | 2,792,999 | 2,792,999 | <code>ZACNJABB5KPJ92081</code><br><code>SALCJ2FX1LH858117</code><br><code>SALRR2RV0L2433391</code><br><code>SALCJ2FXXLH862327</code><br><code>SALYK2EX1LA261711</code> |
| `back_legroom` | `Float64` | Quantitative: continuous | Numeric source field `back_legroom`; storage type is `Float64`. | 224,310 | 2,568,689 | 218 | <code>35.1</code><br><code>38.1</code><br><code>37.6</code><br><code>37.1</code><br><code>33.8</code> |
| `bed` | `String` | Categorical: ordinal | Source-provided `bed` value from `used_cars`; exact meaning follows the upstream field definition. | 2,773,920 | 19,079 | 3 | <code>Short</code><br><code>Long</code><br><code>Regular</code> |
| `bed_length` | `Float64` | Quantitative: continuous | Numeric source field `bed_length`; storage type is `Float64`. | 2,404,024 | 388,975 | 82 | <code>78.9</code><br><code>81.9</code><br><code>74.0</code><br><code>67.4</code><br><code>69.9</code> |
| `body_type` | `String` | Categorical: nominal | Source-provided `body_type` value from `used_cars`; exact meaning follows the upstream field definition. | 11,937 | 2,781,062 | 9 | <code>SUV / Crossover</code><br><code>Sedan</code><br><code>Coupe</code><br><code>Hatchback</code><br><code>Pickup Truck</code> |
| `cabin` | `String` | Categorical: nominal | Source-provided `cabin` value from `used_cars`; exact meaning follows the upstream field definition. | 2,731,225 | 61,774 | 4 | <code>Crew Cab</code><br><code>Extended Cab</code><br><code>Regular Cab</code><br><code>Large Crew Cab</code> |
| `city` | `String` | Categorical: nominal | Source-provided `city` value from `used_cars`; exact meaning follows the upstream field definition. | 0 | 2,792,999 | 4,685 | <code>Bayamon</code><br><code>San Juan</code><br><code>Guaynabo</code><br><code>Bay Shore</code><br><code>Bronx</code> |
| `city_fuel_economy` | `Float64` | Quantitative: continuous | Numeric source field `city_fuel_economy`; storage type is `Float64`. | 452,372 | 2,340,627 | 100 | <code>27.0</code><br><code>18.0</code><br><code>15.0</code><br><code>20.0</code><br><code>24.0</code> |
| `daysonmarket` | `Int64` | Quantitative: discrete | Source numeric measure in `daysonmarket`; storage type is `Int64`. | 0 | 2,792,999 | 1,718 | <code>522</code><br><code>207</code><br><code>196</code><br><code>137</code><br><code>242</code> |
| `dealer_zip` | `Int64` | Identifier: nominal | Dealer ZIP/location code, not a numerical magnitude. Integer export removes leading zeros; categorical use requires explicit formatting. | 209 | 2,792,790 | 8,231 | <code>960</code><br><code>922</code><br><code>969</code><br><code>11706</code><br><code>10466</code> |
| `description` | `String` | Text / structured text | Source narrative or compound text in `description`; extract features explicitly before modeling. | 73,872 | 2,719,127 | 2,366,593 | <code>[!@@Additional Info@@!]4-Wheel Disc Brakes,ABS,AM/FM Stereo,Adaptive Cruise Control,Adjustable Steering Whe...</code><br><code>4WD. Diamond Black Crystal Pearlcoat 2020 Jeep Compass Latitude 4WD 9-Speed 948TE Automatic 2.4L I422/30 Ci...</code><br><code>2014 NISSAN ROGUE SL ALL WHEEL DRIVE, LOCAL TRADE IN, 2.5/ 4 CYL. ENGINE, LEATHER INTERIOR, NAVIGATION SYST...</code><br><code>Slate Blue Pearlcoat 2020 Jeep Cherokee Limited 4WD 9-Speed 948TE Automatic 3.2L V619/27 City/Highway MPG R...</code><br><code>[!@@Additional Info@@!]Door Handle Color - Body-Color, Exhaust Tip Color - Chrome, Front Bumper Color - Bod...</code> |
| `engine_cylinders` | `String` | Categorical: nominal | Source-provided `engine_cylinders` value from `used_cars`; exact meaning follows the upstream field definition. | 92,595 | 2,700,404 | 39 | <code>I4</code><br><code>V6</code><br><code>V6 Diesel</code><br><code>I6</code><br><code>V8</code> |
| `engine_displacement` | `Float64` | Quantitative: continuous | Numeric source field `engine_displacement`; storage type is `Float64`. | 157,878 | 2,635,121 | 67 | <code>1300.0</code><br><code>2000.0</code><br><code>3000.0</code><br><code>2500.0</code><br><code>1700.0</code> |
| `engine_type` | `String` | Categorical: nominal | Source-provided `engine_type` value from `used_cars`; exact meaning follows the upstream field definition. | 92,595 | 2,700,404 | 39 | <code>I4</code><br><code>V6</code><br><code>V6 Diesel</code><br><code>I6</code><br><code>V8</code> |
| `exterior_color` | `String` | Categorical: nominal | Source-provided `exterior_color` value from `used_cars`; exact meaning follows the upstream field definition. Missing: `none`, `null`, `unknown`, `n/a`, `na`, `not applicable` (case-insensitive, trimmed). | 45,907 | 2,747,092 | 27,947 | <code>Solar Yellow</code><br><code>Narvik Black</code><br><code>Eiger Gray</code><br><code>Kaikoura Stone</code><br><code>SONIC SILVER</code> |
| `fleet` | `String` | Categorical: binary nominal | Source-provided `fleet` value from `used_cars`; exact meaning follows the upstream field definition. | 1,246,819 | 1,546,180 | 2 | <code>False</code><br><code>True</code> |
| `frame_damaged` | `String` | Categorical: binary nominal | Source-provided `frame_damaged` value from `used_cars`; exact meaning follows the upstream field definition. | 1,246,819 | 1,546,180 | 2 | <code>False</code><br><code>True</code> |
| `franchise_dealer` | `String` | Categorical: binary nominal | Source-provided `franchise_dealer` value from `used_cars`; exact meaning follows the upstream field definition. | 0 | 2,792,999 | 2 | <code>True</code><br><code>False</code> |
| `franchise_make` | `String` | Categorical: nominal | Source-provided `franchise_make` value from `used_cars`; exact meaning follows the upstream field definition. | 565,412 | 2,227,587 | 48 | <code>Jeep</code><br><code>Land Rover</code><br><code>FIAT</code><br><code>Chevrolet</code><br><code>Cadillac</code> |
| `front_legroom` | `Float64` | Quantitative: continuous | Numeric source field `front_legroom`; storage type is `Float64`. | 161,910 | 2,631,089 | 100 | <code>41.2</code><br><code>39.1</code><br><code>39.0</code><br><code>40.2</code><br><code>42.3</code> |
| `fuel_tank_volume` | `Float64` | Quantitative: continuous | Numeric source field `fuel_tank_volume`; storage type is `Float64`. | 147,681 | 2,645,318 | 181 | <code>12.7</code><br><code>17.7</code><br><code>23.5</code><br><code>16.6</code><br><code>13.2</code> |
| `fuel_type` | `String` | Categorical: nominal | Source-provided `fuel_type` value from `used_cars`; exact meaning follows the upstream field definition. | 75,692 | 2,717,307 | 8 | <code>Gasoline</code><br><code>Diesel</code><br><code>Biodiesel</code><br><code>Flex Fuel Vehicle</code><br><code>Electric</code> |
| `has_accidents` | `String` | Categorical: binary nominal | Source-provided `has_accidents` value from `used_cars`; exact meaning follows the upstream field definition. | 1,246,819 | 1,546,180 | 2 | <code>False</code><br><code>True</code> |
| `height` | `Float64` | Quantitative: continuous | Numeric source field `height`; storage type is `Float64`. | 146,816 | 2,646,183 | 471 | <code>66.5</code><br><code>68.0</code><br><code>73.0</code><br><code>66.3</code><br><code>56.9</code> |
| `highway_fuel_economy` | `Float64` | Quantitative: continuous | Numeric source field `highway_fuel_economy`; storage type is `Float64`. | 452,372 | 2,340,627 | 98 | <code>36.0</code><br><code>27.0</code><br><code>22.0</code><br><code>25.0</code><br><code>34.0</code> |
| `horsepower` | `Float64` | Quantitative: continuous | Numeric source field `horsepower`; storage type is `Float64`. | 157,878 | 2,635,121 | 455 | <code>177.0</code><br><code>246.0</code><br><code>340.0</code><br><code>247.0</code><br><code>186.0</code> |
| `interior_color` | `String` | Categorical: nominal | Source-provided `interior_color` value from `used_cars`; exact meaning follows the upstream field definition. Missing: `none`, `null`, `unknown`, `n/a`, `na`, `not applicable` (case-insensitive, trimmed). | 366,576 | 2,426,423 | 43,223 | <code>Black</code><br><code>Black (Ebony)</code><br><code>Gray (Ebony/Ebony/Ebony)</code><br><code>Brown (Ebony / Ebony)</code><br><code>White (Eclipse / Ebony / Ebony)</code> |
| `isCab` | `String` | Categorical: binary nominal | Source-provided `isCab` value from `used_cars`; exact meaning follows the upstream field definition. | 1,246,819 | 1,546,180 | 2 | <code>False</code><br><code>True</code> |
| `is_cpo` | `String` | Categorical: binary nominal | Source-provided `is_cpo` value from `used_cars`; exact meaning follows the upstream field definition. | 2,611,166 | 181,833 | 1 | <code>True</code> |
| `is_new` | `String` | Categorical: binary nominal | Source-provided `is_new` value from `used_cars`; exact meaning follows the upstream field definition. | 0 | 2,792,999 | 2 | <code>True</code><br><code>False</code> |
| `is_oemcpo` | `String` | Categorical: binary nominal | Source-provided `is_oemcpo` value from `used_cars`; exact meaning follows the upstream field definition. | 2,658,335 | 134,664 | 1 | <code>True</code> |
| `latitude` | `Float64` | Quantitative: continuous | Numeric source field `latitude`; storage type is `Float64`. | 0 | 2,792,999 | 23,429 | <code>18.3988</code><br><code>18.4439</code><br><code>18.3467</code><br><code>40.7333</code><br><code>40.8847</code> |
| `length` | `Float64` | Quantitative: continuous | Numeric source field `length`; storage type is `Float64`. | 146,807 | 2,646,192 | 835 | <code>166.6</code><br><code>181.0</code><br><code>195.1</code><br><code>188.9</code><br><code>183.5</code> |
| `listed_date` | `Date` | Temporal: date/time | Date/time value in `listed_date`; interpretation follows the source field. | 0 | 2,792,999 | 1,714 | <code>2019-04-06</code><br><code>2020-02-15</code><br><code>2020-02-26</code><br><code>2020-04-25</code><br><code>2020-01-11</code> |
| `listing_color` | `String` | Categorical: nominal | Source-provided `listing_color` value from `used_cars`; exact meaning follows the upstream field definition. Missing: `none`, `null`, `unknown`, `n/a`, `na`, `not applicable` (case-insensitive, trimmed). | 366,053 | 2,426,946 | 14 | <code>YELLOW</code><br><code>BLACK</code><br><code>GRAY</code><br><code>SILVER</code><br><code>WHITE</code> |
| `listing_id` | `String` | Identifier: nominal | Source identifier `listing_id`; treat as a key rather than a numeric quantity. | 0 | 2,792,999 | 2,792,999 | <code>237132766</code><br><code>265946296</code><br><code>266911050</code><br><code>270957414</code><br><code>262940541</code> |
| `longitude` | `Float64` | Quantitative: continuous | Numeric source field `longitude`; storage type is `Float64`. | 0 | 2,792,999 | 23,229 | <code>-66.1582</code><br><code>-66.0785</code><br><code>-66.1098</code><br><code>-73.2587</code><br><code>-73.8317</code> |
| `major_options` | `String` | Categorical: nominal, multiple labels | Source list of advertised vehicle options, retained as text; may contain multiple feature labels. | 188,599 | 2,604,400 | 269,230 | <code>[&#x27;Quick Order Package&#x27;]</code><br><code>[&#x27;Adaptive Cruise Control&#x27;]</code><br><code>[&#x27;Leather Seats&#x27;, &#x27;Sunroof/Moonroof&#x27;, &#x27;Navigation System&#x27;, &#x27;Adaptive Cruise Control&#x27;, &#x27;Alloy Wheels&#x27;, &#x27;Blue...</code><br><code>[&#x27;Alloy Wheels&#x27;, &#x27;Bluetooth&#x27;, &#x27;Backup Camera&#x27;]</code><br><code>[&#x27;Navigation System&#x27;, &#x27;Backup Camera&#x27;, &#x27;Blind Spot Monitoring&#x27;]</code> |
| `make_name` | `String` | Categorical: nominal | Source-provided `make_name` value from `used_cars`; exact meaning follows the upstream field definition. | 0 | 2,792,999 | 97 | <code>Jeep</code><br><code>Land Rover</code><br><code>Mazda</code><br><code>Alfa Romeo</code><br><code>Hyundai</code> |
| `maximum_seating` | `String` | Quantitative: discrete | Source seating-capacity text; retained unchanged pending further transformation instructions. Missing: `--` (case-insensitive, trimmed). | 146,851 | 2,646,148 | 11 | <code>5 seats</code><br><code>7 seats</code><br><code>2 seats</code><br><code>8 seats</code><br><code>4 seats</code> |
| `mileage` | `Float64` | Quantitative: continuous | Source numeric measure in `mileage`; storage type is `Float64`. | 0 | 2,792,999 | 197,171 | <code>7.0</code><br><code>8.0</code><br><code>11.0</code><br><code>12.0</code><br><code>14.0</code> |
| `model_name` | `String` | Categorical: nominal | Source-provided `model_name` value from `used_cars`; exact meaning follows the upstream field definition. | 0 | 2,792,999 | 1,396 | <code>Renegade</code><br><code>Discovery Sport</code><br><code>Discovery</code><br><code>Range Rover Velar</code><br><code>MAZDA3</code> |
| `owner_count` | `Int64` | Quantitative: discrete | Source numeric measure in `owner_count`; storage type is `Int64`. | 1,323,908 | 1,469,091 | 18 | <code>2</code><br><code>1</code><br><code>3</code><br><code>4</code><br><code>6</code> |
| `power` | `String` | Text / structured text | Source narrative or compound text in `power`; extract features explicitly before modeling. | 436,393 | 2,356,606 | 2,030 | <code>177 hp @ 5,750 RPM</code><br><code>246 hp @ 5,500 RPM</code><br><code>340 hp @ 6,500 RPM</code><br><code>247 hp @ 5,500 RPM</code><br><code>186 hp @ 6,000 RPM</code> |
| `price` | `Int64` | Quantitative: discrete | Advertised listing price in USD for new or used vehicles; not an observed transaction/sale price. Stored as a rounded integer by the export. | 0 | 2,792,999 | 86,040 | <code>23141</code><br><code>46500</code><br><code>67430</code><br><code>48880</code><br><code>66903</code> |
| `salvage` | `String` | Categorical: binary nominal | Source-provided `salvage` value from `used_cars`; exact meaning follows the upstream field definition. | 1,246,819 | 1,546,180 | 2 | <code>False</code><br><code>True</code> |
| `savings_amount` | `Int64` | Quantitative: discrete | Numeric source field `savings_amount`; storage type is `Int64`. | 0 | 2,792,999 | 10,824 | <code>0</code><br><code>2709</code><br><code>1749</code><br><code>3500</code><br><code>4238</code> |
| `seller_rating` | `Float64` | Quantitative: continuous | Source rating measure in `seller_rating`; review its scale before treating it as interval data. | 38,193 | 2,754,806 | 1,815 | <code>2.8</code><br><code>3.0</code><br><code>3.4477611940298507</code><br><code>2.963636363636364</code><br><code>4.666666666666667</code> |
| `theft_title` | `String` | Categorical: binary nominal | Source-provided `theft_title` value from `used_cars`; exact meaning follows the upstream field definition. | 1,246,819 | 1,546,180 | 2 | <code>False</code><br><code>True</code> |
| `torque` | `String` | Text / structured text | Source narrative or compound text in `torque`; extract features explicitly before modeling. | 471,716 | 2,321,283 | 2,054 | <code>200 lb-ft @ 1,750 RPM</code><br><code>269 lb-ft @ 1,400 RPM</code><br><code>332 lb-ft @ 3,500 RPM</code><br><code>269 lb-ft @ 1,200 RPM</code><br><code>186 lb-ft @ 4,000 RPM</code> |
| `transmission` | `String` | Categorical: nominal | Source-provided `transmission` value from `used_cars`; exact meaning follows the upstream field definition. | 58,707 | 2,734,292 | 4 | <code>A</code><br><code>CVT</code><br><code>M</code><br><code>Dual Clutch</code> |
| `transmission_display` | `String` | Categorical: nominal | Source-provided `transmission_display` value from `used_cars`; exact meaning follows the upstream field definition. | 58,707 | 2,734,292 | 44 | <code>9-Speed Automatic Overdrive</code><br><code>8-Speed Automatic Overdrive</code><br><code>6-Speed Automatic Overdrive</code><br><code>6-Speed Automatic</code><br><code>9-Speed Automatic</code> |
| `trimId` | `String` | Identifier: nominal | Source identifier `trimId`; treat as a key rather than a numeric quantity. | 105,884 | 2,687,115 | 41,009 | <code>t83804</code><br><code>t86759</code><br><code>t86074</code><br><code>t85614</code><br><code>t85256</code> |
| `trim_name` | `String` | Categorical: nominal | Source-provided `trim_name` value from `used_cars`; exact meaning follows the upstream field definition. | 106,336 | 2,686,663 | 8,989 | <code>Latitude FWD</code><br><code>S AWD</code><br><code>V6 HSE AWD</code><br><code>P250 R-Dynamic S AWD</code><br><code>Sedan FWD</code> |
| `wheel_system` | `String` | Categorical: nominal | Source-provided `wheel_system` value from `used_cars`; exact meaning follows the upstream field definition. | 133,721 | 2,659,278 | 5 | <code>FWD</code><br><code>AWD</code><br><code>RWD</code><br><code>4WD</code><br><code>4X2</code> |
| `wheel_system_display` | `String` | Categorical: nominal | Source-provided `wheel_system_display` value from `used_cars`; exact meaning follows the upstream field definition. | 133,721 | 2,659,278 | 5 | <code>Front-Wheel Drive</code><br><code>All-Wheel Drive</code><br><code>Rear-Wheel Drive</code><br><code>Four-Wheel Drive</code><br><code>4X2</code> |
| `wheelbase` | `Float64` | Quantitative: continuous | Numeric source field `wheelbase`; storage type is `Float64`. | 146,785 | 2,646,214 | 482 | <code>101.2</code><br><code>107.9</code><br><code>115.0</code><br><code>113.1</code><br><code>107.3</code> |
| `width` | `Float64` | Quantitative: continuous | Numeric source field `width`; storage type is `Float64`. | 146,827 | 2,646,172 | 284 | <code>79.6</code><br><code>85.6</code><br><code>87.4</code><br><code>84.4</code><br><code>70.7</code> |
| `year` | `Int64` | Temporal: discrete year (interval) | Vehicle model year in `year`. | 0 | 2,792,999 | 96 | <code>2019</code><br><code>2020</code><br><code>2015</code><br><code>2017</code><br><code>2018</code> |

## `NHTSA_vPIC`

One source VIN/model-year-hint decode row. **2,999,875 rows; 133 columns.**

| Column | Type | Variable classification | Definition / interpretation | Missing Values | Total rows with values | Distinct values | 5 distinct values |
|---|---|---|---|---:|---:|---:|---|
| `vin` | `String` | Identifier: nominal | vPIC decoded field `vin`; consult the NHTSA variable metadata for its official label, ID, group, and data type. | 0 | 2,999,875 | 2,999,875 | <code>00000000011111111</code><br><code>00000000011615922</code><br><code>00000000012631735</code><br><code>00000000012646077</code><br><code>00000000012686460</code> |
| `model_year_hint` | `Int64` | Identifier: nominal | vPIC decoded field `model_year_hint`; consult the NHTSA variable metadata for its official label, ID, group, and data type. | 0 | 2,999,875 | 98 | <code>1967</code><br><code>1925</code><br><code>1926</code><br><code>1924</code><br><code>1968</code> |
| `fetched_at` | `Datetime(time_unit='us', time_zone=None)` | Categorical: nominal | vPIC decoded field `fetched_at`; consult the NHTSA variable metadata for its official label, ID, group, and data type. | 0 | 2,999,875 | 34,268 | <code>2026-09-25 20:29:15</code><br><code>2026-09-25 20:29:16</code><br><code>2026-09-25 20:29:17</code><br><code>2026-09-25 20:29:18</code><br><code>2026-09-25 20:29:19</code> |
| `ABS` | `String` | Categorical: nominal | NHTSA variable `ABS`; label: Antilock Braking System (ABS); variable ID: 86; group: Active Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,057,509 | 1,942,366 | 3 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code> |
| `ActiveSafetySysNote` | `String` | Text / structured text | NHTSA variable `ActiveSafetySysNote`; label: Active Safety System Note; variable ID: 169; group: Active Safety System; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,774,053 | 225,822 | 499 | <code>Lane Keep System, Lane Departure Warning, Forward Collision Warning, Crash Imminent Braking, Adaptive Cruis...</code><br><code>Acura Watch Plus: Lane Keep System, Lane Departure Warning, Forward Collision Warning, Crash Imminent Braki...</code><br><code>Lane Keep System, Lane Departure Warning, Forward Collision Warning, Crash Imminent Braking, Adaptive Cruis...</code><br><code>Rear Cross Traffic Monitor: Standard for Premium; Back-Up Sensors: Optional for Technology Plus; Rear Cross...</code><br><code>Automatic Crash Notification: Standard for Technology Package</code> |
| `AdaptiveCruiseControl` | `String` | Categorical: nominal | NHTSA variable `AdaptiveCruiseControl`; label: Adaptive Cruise Control (ACC); variable ID: 81; group: Active Safety System/Maintaining Safe Distance; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,137,888 | 861,987 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `AdaptiveDrivingBeam` | `String` | Categorical: nominal | NHTSA variable `AdaptiveDrivingBeam`; label: Adaptive Driving Beam (ADB); variable ID: 180; group: Active Safety System/Lighting Technologies; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,432,288 | 567,587 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `AdaptiveHeadlights` | `String` | Categorical: nominal | NHTSA variable `AdaptiveHeadlights`; label: AdaptiveHeadlights; variable ID: not captured; group: not captured; DataType: `None`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,875 | 0 | 0 | (no non-missing values) |
| `AdditionalErrorText` | `String` | Categorical: nominal | NHTSA variable `AdditionalErrorText`; label: Additional Error Text; variable ID: 156; group: not captured; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,519,979 | 479,896 | 356 | <code>Invalid character(s): 14:S, 15:V, 16:R, 17:A.</code><br><code>The Model Year decoded for this VIN may be incorrect. If you know the Model year, please enter it and decod...</code><br><code>Invalid character(s): 14:S. The Model Year decoded for this VIN may be incorrect. If you know the Model yea...</code><br><code>Invalid character(s): 16:E.</code><br><code>Invalid character(s): 14:E, 15:R, 16:E, 17:D.</code> |
| `AirBagLocCurtain` | `String` | Categorical: nominal | NHTSA variable `AirBagLocCurtain`; label: Curtain Air Bag Locations; variable ID: 55; group: Passive Safety System/Air Bag Location; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,117,143 | 1,882,732 | 6 | <code>Not Applicable</code><br><code>1st and 2nd Rows</code><br><code>1st Row (Driver and Passenger)</code><br><code>All Rows</code><br><code>1st and 2nd and 3rd Rows</code> |
| `AirBagLocFront` | `String` | Categorical: nominal | NHTSA variable `AirBagLocFront`; label: Front Air Bag Locations; variable ID: 65; group: Passive Safety System/Air Bag Location; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 77,654 | 2,922,221 | 3 | <code>Not Applicable</code><br><code>1st Row (Driver and Passenger)</code><br><code>Driver Seat Only</code> |
| `AirBagLocKnee` | `String` | Categorical: nominal | NHTSA variable `AirBagLocKnee`; label: Knee Air Bag Locations; variable ID: 69; group: Passive Safety System/Air Bag Location; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,920,354 | 1,079,521 | 4 | <code>Not Applicable</code><br><code>Driver Seat Only</code><br><code>1st Row (Driver and Passenger)</code><br><code>Passenger Seat Only</code> |
| `AirBagLocSeatCushion` | `String` | Categorical: nominal | NHTSA variable `AirBagLocSeatCushion`; label: Seat Cushion Air Bag Locations; variable ID: 56; group: Passive Safety System/Air Bag Location; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,815,337 | 184,538 | 5 | <code>Not Applicable</code><br><code>1st Row (Driver and Passenger)</code><br><code>Passenger Seat Only</code><br><code>Driver Seat Only</code><br><code>1st and 2nd Rows</code> |
| `AirBagLocSide` | `String` | Categorical: nominal | NHTSA variable `AirBagLocSide`; label: Side Air Bag Locations; variable ID: 107; group: Passive Safety System/Air Bag Location; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 154,134 | 2,845,741 | 7 | <code>Not Applicable</code><br><code>1st Row (Driver and Passenger)</code><br><code>All Rows</code><br><code>1st and 2nd Rows</code><br><code>1st and 2nd and 3rd Rows</code> |
| `AutoReverseSystem` | `String` | Categorical: nominal | NHTSA variable `AutoReverseSystem`; label: Auto-Reverse System for Windows and Sunroofs; variable ID: 172; group: Active Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,161,446 | 1,838,429 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `AutomaticPedestrianAlertingSound` | `String` | Categorical: nominal | NHTSA variable `AutomaticPedestrianAlertingSound`; label: Automatic Pedestrian Alerting Sound (for Hybrid and EV only); variable ID: 173; group: Active Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,826,159 | 173,716 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Not Available</code><br><code>Optional</code> |
| `AxleConfiguration` | `String` | Categorical: nominal | NHTSA variable `AxleConfiguration`; label: Axle Configuration; variable ID: 145; group: Mechanical/Drivetrain; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,998,950 | 925 | 3 | <code>Single</code><br><code>SFA - Set-Forward Axle</code><br><code>Tandem</code> |
| `Axles` | `Int64` | Quantitative: discrete | NHTSA variable `Axles`; label: Axles; variable ID: 41; group: Mechanical/Drivetrain; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,240,501 | 1,759,374 | 10 | <code>5</code><br><code>0</code><br><code>2</code><br><code>1</code><br><code>3</code> |
| `BasePrice` | `Float64` | Quantitative: continuous | NHTSA variable `BasePrice`; label: Base Price ($); variable ID: 136; group: General; DataType: `decimal`; stored in Parquet as `Float64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,130,821 | 869,054 | 2,046 | <code>25900.0</code><br><code>21140.0</code><br><code>21700.0</code><br><code>23400.0</code><br><code>23800.0</code> |
| `BatteryA` | `Int64` | Quantitative: discrete | NHTSA variable `BatteryA`; label: Battery Current (Amps) From; variable ID: 57; group: Mechanical/Battery; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,998,797 | 1,078 | 3 | <code>244</code><br><code>180</code><br><code>120</code> |
| `BatteryA_to` | `String` | Quantitative: discrete | NHTSA variable `BatteryA_to`; label: Battery Current (Amps) To; variable ID: 132; group: Mechanical/Battery; DataType: `int`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,875 | 0 | 0 | (no non-missing values) |
| `BatteryCells` | `Int64` | Quantitative: discrete | NHTSA variable `BatteryCells`; label: Number of Battery Cells per Module; variable ID: 48; group: Mechanical/Battery; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,660 | 215 | 1 | <code>34</code> |
| `BatteryInfo` | `String` | Text / structured text | NHTSA variable `BatteryInfo`; label: Other Battery Info; variable ID: 1; group: Mechanical/Battery; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,984,603 | 15,272 | 45 | <code>w/ DC Fast Charging</code><br><code>Motor (BAT:24kw)</code><br><code>Motor (BAT:30kw)</code><br><code>3NM</code><br><code>3 Phase A/C Induction</code> |
| `BatteryKWh` | `Int64` | Quantitative: continuous | NHTSA variable `BatteryKWh`; label: Battery Energy (KWh) From; variable ID: 59; group: Mechanical/Battery; DataType: `decimal`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,997,614 | 2,261 | 10 | <code>24</code><br><code>30</code><br><code>31</code><br><code>51</code><br><code>81</code> |
| `BatteryKWh_to` | `Int64` | Quantitative: continuous | NHTSA variable `BatteryKWh_to`; label: Battery Energy (KWh) To; variable ID: 134; group: Mechanical/Battery; DataType: `decimal`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,254 | 621 | 4 | <code>40</code><br><code>60</code><br><code>90</code><br><code>9</code> |
| `BatteryModules` | `String` | Quantitative: discrete | NHTSA variable `BatteryModules`; label: Number of Battery Modules per Pack; variable ID: 137; group: Mechanical/Battery; DataType: `int`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,875 | 0 | 0 | (no non-missing values) |
| `BatteryPacks` | `Int64` | Quantitative: discrete | NHTSA variable `BatteryPacks`; label: Number of Battery Packs per Vehicle; variable ID: 138; group: Mechanical/Battery; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,979,524 | 20,351 | 2 | <code>1</code><br><code>2</code> |
| `BatteryType` | `String` | Categorical: nominal | NHTSA variable `BatteryType`; label: Battery Type; variable ID: 2; group: Mechanical/Battery; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,979,983 | 19,892 | 4 | <code>Not Applicable</code><br><code>Lithium-ion/Li-Ion</code><br><code>Nickel-Metal-Hydride/NiMH</code><br><code>Lead Acid/Lead</code> |
| `BatteryV` | `Int64` | Quantitative: discrete | NHTSA variable `BatteryV`; label: Battery Voltage (Volts) From; variable ID: 58; group: Mechanical/Battery; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,983,011 | 16,864 | 11 | <code>144</code><br><code>24</code><br><code>4</code><br><code>244</code><br><code>240</code> |
| `BatteryV_to` | `String` | Quantitative: discrete | NHTSA variable `BatteryV_to`; label: Battery Voltage (Volts) To; variable ID: 133; group: Mechanical/Battery; DataType: `int`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,875 | 0 | 0 | (no non-missing values) |
| `BedLengthIN` | `Int64` | Quantitative: discrete | NHTSA variable `BedLengthIN`; label: Bed Length (inches); variable ID: 49; group: Exterior/Dimension; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,995,414 | 4,461 | 9 | <code>61</code><br><code>60</code><br><code>66</code><br><code>67</code><br><code>74</code> |
| `BedType` | `String` | Categorical: nominal | NHTSA variable `BedType`; label: Bed Type; variable ID: 3; group: Exterior/Truck; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,945,697 | 1,054,178 | 5 | <code>Not Applicable</code><br><code>Short</code><br><code>Long</code><br><code>Extended</code><br><code>Standard</code> |
| `BlindSpotIntervention` | `String` | Categorical: nominal | NHTSA variable `BlindSpotIntervention`; label: Blind Spot Intervention (BSI); variable ID: 193; group: Active Safety System/Lane and Side Assist; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,989,261 | 10,614 | 2 | <code>Optional</code><br><code>Standard</code> |
| `BlindSpotMon` | `String` | Categorical: nominal | NHTSA variable `BlindSpotMon`; label: Blind Spot Warning (BSW); variable ID: 88; group: Active Safety System/Lane and Side Assist; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,947,166 | 1,052,709 | 4 | <code>Not Applicable</code><br><code>Optional</code><br><code>Standard</code><br><code>Not Available</code> |
| `BodyCabType` | `String` | Categorical: nominal | NHTSA variable `BodyCabType`; label: Cab Type; variable ID: 4; group: Exterior/Truck; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,569,390 | 1,430,485 | 7 | <code>Not Applicable</code><br><code>Extra/Super/Quad/Double/King/Extended</code><br><code>Crew/Super Crew/Crew Max</code><br><code>Regular</code><br><code>Mega</code> |
| `BodyClass` | `String` | Categorical: nominal | NHTSA variable `BodyClass`; label: Body Class; variable ID: 5; group: Exterior/Body; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 4,841 | 2,995,034 | 28 | <code>Truck</code><br><code>Trailer</code><br><code>Convertible/Cabriolet</code><br><code>Sedan/Saloon</code><br><code>Coupe</code> |
| `BrakeSystemDesc` | `String` | Categorical: nominal | NHTSA variable `BrakeSystemDesc`; label: Brake System Description; variable ID: 52; group: Mechanical/Brake; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,944,036 | 55,839 | 16 | <code>4-Wheel ABS</code><br><code>Incomplete Vehicle with Hydraulic Brake</code><br><code>Incomplete Vehicle with Hydraulic Brakes</code><br><code>Calipers / Front Vented Discs</code><br><code>Calipers / Front Vented Discs &#124; Rear Drums</code> |
| `BrakeSystemType` | `String` | Categorical: nominal | NHTSA variable `BrakeSystemType`; label: Brake System Type; variable ID: 42; group: Mechanical/Brake; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,771,313 | 1,228,562 | 2 | <code>Hydraulic</code><br><code>Air</code> |
| `BusFloorConfigType` | `String` | Categorical: nominal | NHTSA variable `BusFloorConfigType`; label: Bus Floor Configuration Type; variable ID: 148; group: Exterior/Bus; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 31,332 | 2,968,543 | 2 | <code>Not Applicable</code><br><code>Low Floor</code> |
| `BusLength` | `String` | Quantitative: discrete | NHTSA variable `BusLength`; label: Bus Length (feet); variable ID: 147; group: Exterior/Bus; DataType: `int`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,875 | 0 | 0 | (no non-missing values) |
| `BusType` | `String` | Categorical: nominal | NHTSA variable `BusType`; label: Bus Type; variable ID: 149; group: Exterior/Bus; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 31,333 | 2,968,542 | 1 | <code>Not Applicable</code> |
| `CAN_AACN` | `String` | Categorical: nominal | NHTSA variable `CAN_AACN`; label: Automatic Crash Notification (ACN)/Advanced Automatic Crash Notification (AACN); variable ID: 174; group: Active Safety System/911 Notification; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,910,208 | 1,089,667 | 4 | <code>Not Applicable</code><br><code>Optional</code><br><code>Standard</code><br><code>Not Available</code> |
| `CIB` | `String` | Categorical: nominal | NHTSA variable `CIB`; label: Crash Imminent Braking (CIB); variable ID: 87; group: Active Safety System/Forward Collision Prevention; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,901,790 | 1,098,085 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `CashForClunkers` | `String` | Categorical: nominal | NHTSA variable `CashForClunkers`; label: CashForClunkers; variable ID: not captured; group: not captured; DataType: `None`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,875 | 0 | 0 | (no non-missing values) |
| `ChargerLevel` | `String` | Categorical: nominal | NHTSA variable `ChargerLevel`; label: Charger Level; variable ID: 127; group: Mechanical/Battery/Charger; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,998,828 | 1,047 | 3 | <code>Not Applicable</code><br><code>Level 2 AC Charger (up to 80A, 208-240V AC, up to 20kW from single- or three-phase AC) cables permanently f...</code><br><code>Level 3 DC Charger or fast charger (up to 400A, up to 600V DC, up to 240KW)</code> |
| `ChargerPowerKW` | `Int64` | Quantitative: discrete | NHTSA variable `ChargerPowerKW`; label: Charger Power (KW); variable ID: 128; group: Mechanical/Battery/Charger; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,993,490 | 6,385 | 9 | <code>10</code><br><code>20</code><br><code>150</code><br><code>32</code><br><code>88</code> |
| `CombinedBrakingSystem` | `String` | Categorical: nominal | NHTSA variable `CombinedBrakingSystem`; label: Combined Braking System (CBS); variable ID: 202; group: Exterior/Motorcycle; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,875 | 0 | 0 | (no non-missing values) |
| `CoolingType` | `String` | Categorical: nominal | NHTSA variable `CoolingType`; label: Cooling Type; variable ID: 122; group: Engine; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,684,187 | 315,688 | 3 | <code>Not Applicable</code><br><code>Water</code><br><code>Air</code> |
| `CurbWeightLB` | `Int64` | Quantitative: discrete | NHTSA variable `CurbWeightLB`; label: Curb Weight (pounds); variable ID: 54; group: Exterior/Dimension; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,906,275 | 93,600 | 216 | <code>3095</code><br><code>3014</code><br><code>3115</code><br><code>3124</code><br><code>3148</code> |
| `CustomMotorcycleType` | `String` | Categorical: nominal | NHTSA variable `CustomMotorcycleType`; label: Custom Motorcycle Type; variable ID: 151; group: Exterior/Motorcycle; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 3,248 | 2,996,627 | 1 | <code>Not Applicable</code> |
| `DaytimeRunningLight` | `String` | Categorical: nominal | NHTSA variable `DaytimeRunningLight`; label: Daytime Running Light (DRL); variable ID: 177; group: Active Safety System/Lighting Technologies; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,227,320 | 1,772,555 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `DestinationMarket` | `String` | Categorical: nominal | NHTSA variable `DestinationMarket`; label: Destination Market; variable ID: 10; group: General; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,963,031 | 36,844 | 18 | <code>Canada</code><br><code>Mexico</code><br><code>U.S., Mexico, Other Export Market (BUX)</code><br><code>U.S., Canada, Mexico</code><br><code>U.S.</code> |
| `DisplacementCC` | `Float64` | Quantitative: continuous | NHTSA variable `DisplacementCC`; label: Displacement (CC); variable ID: 11; group: Engine; DataType: `decimal`; stored in Parquet as `Float64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 25,401 | 2,974,474 | 301 | <code>5700.0</code><br><code>6500.0</code><br><code>6600.0</code><br><code>6200.0</code><br><code>2359.737216</code> |
| `DisplacementCI` | `Float64` | Quantitative: continuous | NHTSA variable `DisplacementCI`; label: Displacement (CI); variable ID: 12; group: Engine; DataType: `decimal`; stored in Parquet as `Float64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 25,401 | 2,974,474 | 338 | <code>347.83534133997</code><br><code>396.65433661575</code><br><code>402.75671102523</code><br><code>378.34721338734</code><br><code>144.0</code> |
| `DisplacementL` | `Float64` | Quantitative: continuous | NHTSA variable `DisplacementL`; label: Displacement (L); variable ID: 13; group: Engine; DataType: `decimal`; stored in Parquet as `Float64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 25,401 | 2,974,474 | 210 | <code>5.7</code><br><code>6.5</code><br><code>6.6</code><br><code>6.2</code><br><code>2.359737216</code> |
| `Doors` | `Int64` | Quantitative: discrete | NHTSA variable `Doors`; label: Doors; variable ID: 14; group: Exterior/Body; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 542,302 | 2,457,573 | 5 | <code>4</code><br><code>2</code><br><code>5</code><br><code>3</code><br><code>6</code> |
| `DriveType` | `String` | Categorical: nominal | NHTSA variable `DriveType`; label: Drive Type; variable ID: 15; group: Mechanical/Drivetrain; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 447,801 | 2,552,074 | 9 | <code>Not Applicable</code><br><code>4x2</code><br><code>4WD/4-Wheel Drive/4x4</code><br><code>FWD/Front-Wheel Drive</code><br><code>AWD/All-Wheel Drive</code> |
| `DynamicBrakeSupport` | `String` | Categorical: nominal | NHTSA variable `DynamicBrakeSupport`; label: Dynamic Brake Support (DBS); variable ID: 170; group: Active Safety System/Forward Collision Prevention; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,211,779 | 1,788,096 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `EDR` | `String` | Categorical: nominal | NHTSA variable `EDR`; label: Event Data Recorder (EDR); variable ID: 175; group: Active Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,901,043 | 98,832 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `ESC` | `String` | Categorical: nominal | NHTSA variable `ESC`; label: Electronic Stability Control (ESC); variable ID: 99; group: Active Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,130,410 | 1,869,465 | 2 | <code>Not Applicable</code><br><code>Standard</code> |
| `EVDriveUnit` | `String` | Categorical: nominal | NHTSA variable `EVDriveUnit`; label: EV Drive Unit; variable ID: 72; group: Mechanical/Battery; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,989,378 | 10,497 | 3 | <code>Not Applicable</code><br><code>Single Motor</code><br><code>Dual Motor</code> |
| `ElectrificationLevel` | `String` | Categorical: nominal | NHTSA variable `ElectrificationLevel`; label: Electrification Level; variable ID: 126; group: Engine; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,884,757 | 115,118 | 7 | <code>Not Applicable</code><br><code>Strong HEV (Hybrid Electric Vehicle)</code><br><code>Mild HEV (Hybrid Electric Vehicle)</code><br><code>HEV (Hybrid Electric Vehicle) - Level Unknown</code><br><code>BEV (Battery Electric Vehicle)</code> |
| `EngineConfiguration` | `String` | Categorical: nominal | NHTSA variable `EngineConfiguration`; label: Engine Configuration; variable ID: 64; group: Engine; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,055,988 | 1,943,887 | 6 | <code>Not Applicable</code><br><code>V-Shaped</code><br><code>In-Line</code><br><code>Rotary</code><br><code>Horizontally opposed (boxer)</code> |
| `EngineCycles` | `Int64` | Quantitative: discrete | NHTSA variable `EngineCycles`; label: Engine Stroke Cycles; variable ID: 17; group: Engine; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,669,166 | 330,709 | 4 | <code>4</code><br><code>8</code><br><code>6</code><br><code>2</code> |
| `EngineCylinders` | `Int64` | Quantitative: discrete | NHTSA variable `EngineCylinders`; label: Engine Number of Cylinders; variable ID: 9; group: Engine; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 301,306 | 2,698,569 | 9 | <code>8</code><br><code>4</code><br><code>6</code><br><code>10</code><br><code>3</code> |
| `EngineHP` | `Int64` | Quantitative: continuous | NHTSA variable `EngineHP`; label: Engine Brake (hp) From; variable ID: 71; group: Engine; DataType: `decimal`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,102,652 | 1,897,223 | 397 | <code>300</code><br><code>165</code><br><code>201</code><br><code>500</code><br><code>225</code> |
| `EngineHP_to` | `Int64` | Quantitative: continuous | NHTSA variable `EngineHP_to`; label: Engine Brake (hp) To; variable ID: 125; group: Engine; DataType: `decimal`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,910,326 | 89,549 | 119 | <code>205</code><br><code>129</code><br><code>155</code><br><code>175</code><br><code>160</code> |
| `EngineKW` | `Int64` | Quantitative: continuous | NHTSA variable `EngineKW`; label: Engine Power (KW); variable ID: 21; group: Engine; DataType: `decimal`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,992,437 | 7,438 | 46 | <code>107</code><br><code>143</code><br><code>171</code><br><code>103</code><br><code>55</code> |
| `EngineManufacturer` | `String` | Categorical: nominal | NHTSA variable `EngineManufacturer`; label: Engine Manufacturer; variable ID: 146; group: Engine; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,148,451 | 1,851,424 | 117 | <code>GEP</code><br><code>GM-Isuzu</code><br><code>GM</code><br><code>Honda</code><br><code>HONDA</code> |
| `EngineModel` | `String` | Categorical: nominal | NHTSA variable `EngineModel`; label: Engine Model; variable ID: 18; group: Engine; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,129,271 | 1,870,604 | 1,433 | <code>K24V7</code><br><code>JNC1</code><br><code>J32A1</code><br><code>J32A2</code><br><code>J32A3</code> |
| `EntertainmentSystem` | `String` | Categorical: nominal | NHTSA variable `EntertainmentSystem`; label: Entertainment System; variable ID: 23; group: Interior; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,911,081 | 88,794 | 3 | <code>Not Applicable</code><br><code>CD + stereo</code><br><code>Rear Entertainment System</code> |
| `ErrorCode` | `String` | Categorical: nominal | NHTSA variable `ErrorCode`; label: Error Code; variable ID: 143; group: not captured; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 0 | 2,999,875 | 60 | <code>7,12</code><br><code>1,7,12</code><br><code>1,7,12,400</code><br><code>1,7,11,400</code><br><code>6,7,12,400</code> |
| `ErrorText` | `String` | Categorical: nominal | NHTSA variable `ErrorText`; label: Error Text; variable ID: 191; group: not captured; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 0 | 2,999,875 | 61 | <code>7 - Manufacturer is not registered with NHTSA for sale or importation in the U.S. for use on U.S roads; Ple...</code><br><code>1 - Check Digit (9th position) does not calculate properly; 7 - Manufacturer is not registered with NHTSA f...</code><br><code>1 - Check Digit (9th position) does not calculate properly; 7 - Manufacturer is not registered with NHTSA f...</code><br><code>1 - Check Digit (9th position) does not calculate properly; 7 - Manufacturer is not registered with NHTSA f...</code><br><code>6 - Incomplete VIN; 7 - Manufacturer is not registered with NHTSA for sale or importation in the U.S. for u...</code> |
| `ForwardCollisionWarning` | `String` | Categorical: nominal | NHTSA variable `ForwardCollisionWarning`; label: Forward Collision Warning (FCW); variable ID: 101; group: Active Safety System/Forward Collision Prevention; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,716,740 | 1,283,135 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `FuelInjectionType` | `String` | Categorical: nominal | NHTSA variable `FuelInjectionType`; label: Fuel Delivery/Fuel Injection Type; variable ID: 67; group: Engine; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,453,072 | 546,803 | 7 | <code>Not Applicable</code><br><code>Stoichiometric gasoline direct injection (SGDI)</code><br><code>Sequential Fuel Injection (SFI)</code><br><code>Multipoint Fuel Injection (MPFI)</code><br><code>Throttle Body Fuel Injection (TBI)</code> |
| `FuelTypePrimary` | `String` | Categorical: nominal | NHTSA variable `FuelTypePrimary`; label: Fuel Type - Primary; variable ID: 24; group: Engine; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 25,648 | 2,974,227 | 9 | <code>Not Applicable</code><br><code>Diesel</code><br><code>Gasoline</code><br><code>Compressed Natural Gas (CNG)</code><br><code>Flexible Fuel Vehicle (FFV)</code> |
| `FuelTypeSecondary` | `String` | Categorical: nominal | NHTSA variable `FuelTypeSecondary`; label: Fuel Type - Secondary; variable ID: 66; group: Engine; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,711,678 | 288,197 | 8 | <code>Not Applicable</code><br><code>Electric</code><br><code>Flexible Fuel Vehicle (FFV)</code><br><code>Ethanol (E85)</code><br><code>Compressed Natural Gas (CNG)</code> |
| `GVWR` | `String` | Categorical: nominal | NHTSA variable `GVWR`; label: Gross Vehicle Weight Rating From; variable ID: 25; group: Exterior/Dimension; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 10,256 | 2,989,619 | 16 | <code>Class 3: 10,001 - 14,000 lb (4,536 - 6,350 kg)</code><br><code>Class 1C: 4,001 - 5,000 lb (1,814 - 2,268 kg)</code><br><code>Class 1: 6,000 lb or less (2,722 kg or less)</code><br><code>Class 1B: 3,001 - 4,000 lb (1,360 - 1,814 kg)</code><br><code>Class 1A: 3,000 lb or less (1,360 kg or less)</code> |
| `GVWR_to` | `String` | Categorical: nominal | NHTSA variable `GVWR_to`; label: Gross Vehicle Weight Rating To; variable ID: 190; group: Exterior/Dimension; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,926,807 | 1,073,068 | 7 | <code>Class 1: 6,000 lb or less (2,722 kg or less)</code><br><code>Class 1C: 4,001 - 5,000 lb (1,814 - 2,268 kg)</code><br><code>Class 1B: 3,001 - 4,000 lb (1,360 - 1,814 kg)</code><br><code>Class 1D: 5,001 - 6,000 lb (2,268 - 2,722 kg)</code><br><code>Class 2E: 6,001 - 7,000 lb (2,722 - 3,175 kg)</code> |
| `KeylessIgnition` | `String` | Categorical: nominal | NHTSA variable `KeylessIgnition`; label: Keyless Ignition; variable ID: 176; group: Active Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,704,562 | 1,295,313 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `LaneCenteringAssistance` | `String` | Categorical: nominal | NHTSA variable `LaneCenteringAssistance`; label: Lane Centering Assistance; variable ID: 194; group: Active Safety System/Lane and Side Assist; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,949,634 | 50,241 | 2 | <code>Standard</code><br><code>Optional</code> |
| `LaneDepartureWarning` | `String` | Categorical: nominal | NHTSA variable `LaneDepartureWarning`; label: Lane Departure Warning (LDW); variable ID: 102; group: Active Safety System/Lane and Side Assist; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,873,167 | 1,126,708 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `LaneKeepSystem` | `String` | Categorical: nominal | NHTSA variable `LaneKeepSystem`; label: Lane Keeping Assistance (LKA); variable ID: 103; group: Active Safety System/Lane and Side Assist; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,952,376 | 1,047,499 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `LowerBeamHeadlampLightSource` | `String` | Categorical: nominal | NHTSA variable `LowerBeamHeadlampLightSource`; label: Headlamp Light Source; variable ID: 178; group: Active Safety System/Lighting Technologies; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,247,864 | 752,011 | 6 | <code>Not Applicable</code><br><code>LED</code><br><code>Halogen</code><br><code>HID</code><br><code>Halogen, LED</code> |
| `Make` | `String` | Categorical: nominal | NHTSA variable `Make`; label: Make; variable ID: 26; group: General; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 3,294 | 2,996,581 | 166 | <code>EAGER BEAVER</code><br><code>SHERMAN + REILLY</code><br><code>IDEAL</code><br><code>HUMMER</code><br><code>TRAVALONG</code> |
| `Manufacturer` | `String` | Categorical: nominal | NHTSA variable `Manufacturer`; label: Manufacturer Name; variable ID: 27; group: General; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 3,200 | 2,996,675 | 210 | <code>GENERAL ENGINES CO., INC </code><br><code>SHERMAN + REILLY, INC.</code><br><code>IDEAL WELDING CO INC</code><br><code>AM GENERAL LLC</code><br><code>TRAVALONG INC.</code> |
| `ManufacturerId` | `String` | Categorical: nominal | NHTSA variable `ManufacturerId`; label: ManufacturerId; variable ID: not captured; group: not captured; DataType: `None`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,875 | 0 | 0 | (no non-missing values) |
| `Model` | `String` | Categorical: nominal | NHTSA variable `Model`; label: Model; variable ID: 28; group: General; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 4,733 | 2,995,142 | 1,109 | <code>H1</code><br><code>Travalong Trailer</code><br><code>KARI-KOOL TRANSPORTS</code><br><code>Transglobal Industries</code><br><code>Asher Series</code> |
| `ModelYear` | `Int64` | Temporal: discrete year (interval) | NHTSA variable `ModelYear`; label: Model Year; variable ID: 29; group: General; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 3,518 | 2,996,357 | 42 | <code>2006</code><br><code>2007</code><br><code>2002</code><br><code>2013</code><br><code>2001</code> |
| `NCSABodyType` | `String` | Categorical: nominal | NHTSA variable `NCSABodyType`; label: NCSA Body Type; variable ID: 96; group: Internal; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 4,748 | 2,995,127 | 33 | <code>Single-unit straight truck or Cab-Chassis (GVWR range 10,001 to 19,500 lbs.)</code><br><code>Not Reported</code><br><code>Convertible(excludes sun-roof,t-bar)</code><br><code>4-door sedan, hardtop</code><br><code>2-door sedan,hardtop,coupe</code> |
| `NCSAMake` | `String` | Categorical: nominal | NHTSA variable `NCSAMake`; label: NCSA Make; variable ID: 97; group: Internal; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 4,729 | 2,995,146 | 48 | <code>AM General</code><br><code>Not Reported</code><br><code>Other Domestic Manufacturers</code><br><code>Acura</code><br><code>Honda</code> |
| `Note` | `String` | Text / structured text | NHTSA variable `Note`; label: Note; variable ID: 114; group: General; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,434,741 | 565,134 | 1,021 | <code>0= standard  length  , 2= 2ft 0&quot; longer than standard etc and letters will represent shorter deck length i....</code><br><code>ENLARGED CAB</code><br><code>8&quot; LED Display, Apple CarPlay, Android Auto, HD Radio, SiriusXM Satellite Radio, Bluetooth, USB port</code><br><code>Curb Weight: 3,803-3,875 lbs</code><br><code>Curb Weight: 3,878-3,945 lbs</code> |
| `OtherEngineInfo` | `String` | Text / structured text | NHTSA variable `OtherEngineInfo`; label: Other Engine Info; variable ID: 129; group: Engine; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,107,926 | 891,949 | 1,731 | <code>NATURALLY ASPIRATED</code><br><code>Direct Fuel Injection</code><br><code>Direct Fuel Injection, Sequential Multiport Fuel Injection</code><br><code>Sequential Multiport Fuel Injection</code><br><code>Engine: 107HP / Motor: 129HP; Sequential Multiport Fuel Injection</code> |
| `OtherRestraintSystemInfo` | `String` | Text / structured text | NHTSA variable `OtherRestraintSystemInfo`; label: Other Restraint System Info; variable ID: 121; group: Passive Safety System; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,984,419 | 1,015,456 | 626 | <code>seat belts: front, rear, rear center</code><br><code>Seat Belt (Rear)</code><br><code>Seat Belt ( Rear Center Position)</code><br><code>seat belts: front, rear and rear center</code><br><code>Rear Side Curtain Air bag (Rr outer positions). Rear Seat Belts (Rr center position).</code> |
| `ParkAssist` | `String` | Categorical: nominal | NHTSA variable `ParkAssist`; label: Parking Assist; variable ID: 105; group: Active Safety System/Backing Up and Parking; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,720,185 | 279,690 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `PedestrianAutomaticEmergencyBraking` | `String` | Categorical: nominal | NHTSA variable `PedestrianAutomaticEmergencyBraking`; label: Pedestrian Automatic Emergency Braking (PAEB); variable ID: 171; group: Active Safety System/Forward Collision Prevention; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,149,181 | 850,694 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `PlantCity` | `String` | Categorical: nominal | NHTSA variable `PlantCity`; label: Plant City; variable ID: 31; group: General; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 104,612 | 2,895,263 | 343 | <code>CHATTANOOGA</code><br><code>MISHAWAKA</code><br><code>BLUE RAPIDS</code><br><code>CARPINTERIA</code><br><code>MARYSVILLE</code> |
| `PlantCompanyName` | `String` | Categorical: nominal | NHTSA variable `PlantCompanyName`; label: Plant Company Name; variable ID: 76; group: General; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 981,134 | 2,018,741 | 389 | <code>PMC</code><br><code>C.D.V.</code><br><code>St Louis Assembly South</code><br><code>Newark Assembly</code><br><code>STERLING HEIGHTS ASSEMBLY PLANT</code> |
| `PlantCountry` | `String` | Categorical: nominal | NHTSA variable `PlantCountry`; label: Plant Country; variable ID: 75; group: General; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 4,986 | 2,994,889 | 34 | <code>UNITED STATES (USA)</code><br><code>JAPAN</code><br><code>VENEZUELA</code><br><code>CHINA</code><br><code>GERMANY</code> |
| `PlantState` | `String` | Categorical: nominal | NHTSA variable `PlantState`; label: Plant State; variable ID: 77; group: General; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 730,287 | 2,269,588 | 105 | <code>TENNESSEE</code><br><code>INDIANA</code><br><code>KANSAS</code><br><code>CALIFORNIA</code><br><code>OHIO</code> |
| `Pretensioner` | `String` | Categorical: nominal | NHTSA variable `Pretensioner`; label: Pretensioner; variable ID: 78; group: Passive Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,815,190 | 184,685 | 3 | <code>Not Applicable</code><br><code>Yes</code><br><code>No</code> |
| `RearAutomaticEmergencyBraking` | `String` | Categorical: nominal | NHTSA variable `RearAutomaticEmergencyBraking`; label: Rear Automatic Emergency Braking; variable ID: 192; group: Active Safety System/Backing Up and Parking; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,993,600 | 6,275 | 2 | <code>Optional</code><br><code>Standard</code> |
| `RearCrossTrafficAlert` | `String` | Categorical: nominal | NHTSA variable `RearCrossTrafficAlert`; label: Rear Cross Traffic Alert; variable ID: 183; group: Active Safety System/Backing Up and Parking; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,415,798 | 584,077 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `RearVisibilitySystem` | `String` | Categorical: nominal | NHTSA variable `RearVisibilitySystem`; label: Backup Camera; variable ID: 104; group: Active Safety System/Backing Up and Parking; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,139,712 | 1,860,163 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `SeatBeltsAll` | `String` | Categorical: nominal | NHTSA variable `SeatBeltsAll`; label: Seat Belt Type; variable ID: 79; group: Passive Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 82,862 | 2,917,013 | 4 | <code>Not Applicable</code><br><code>Manual</code><br><code>Automatic</code><br><code>Manual and Automatic</code> |
| `SeatRows` | `Int64` | Quantitative: discrete | NHTSA variable `SeatRows`; label: Number of Seat Rows; variable ID: 61; group: Interior/Seat; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,221,289 | 1,778,586 | 4 | <code>2</code><br><code>1</code><br><code>3</code><br><code>4</code> |
| `Seats` | `Int64` | Quantitative: discrete | NHTSA variable `Seats`; label: Number of Seats; variable ID: 33; group: Interior/Seat; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,301,926 | 1,697,949 | 8 | <code>4</code><br><code>5</code><br><code>2</code><br><code>7</code><br><code>6</code> |
| `SemiautomaticHeadlampBeamSwitching` | `String` | Categorical: nominal | NHTSA variable `SemiautomaticHeadlampBeamSwitching`; label: Semiautomatic Headlamp Beam Switching; variable ID: 179; group: Active Safety System/Lighting Technologies; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,220,490 | 1,779,385 | 4 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code><br><code>Not Available</code> |
| `Series` | `String` | Categorical: nominal | NHTSA variable `Series`; label: Series; variable ID: 34; group: General; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,351,041 | 648,834 | 778 | <code>1 1/4 ton</code><br><code>1 1/4 ton HMCS</code><br><code>1 1/4 ton K-SERIES</code><br><code>1 1/4 ton OPEN BODY W/FULL HARD DOORS</code><br><code>w/Leather</code> |
| `Series2` | `String` | Categorical: nominal | NHTSA variable `Series2`; label: Series2; variable ID: 110; group: General; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,405,210 | 594,665 | 184 | <code>Wagon body style</code><br><code>Special</code><br><code>High</code><br><code>Premium</code><br><code>H Series</code> |
| `SteeringLocation` | `String` | Categorical: nominal | NHTSA variable `SteeringLocation`; label: Steering Location; variable ID: 36; group: Interior; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,005,495 | 1,994,380 | 3 | <code>Not Applicable</code><br><code>Left-Hand Drive (LHD)</code><br><code>Right-Hand Drive (RHD)</code> |
| `TPMS` | `String` | Categorical: nominal | NHTSA variable `TPMS`; label: Tire Pressure Monitoring System (TPMS) Type; variable ID: 168; group: Active Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 423,643 | 2,576,232 | 3 | <code>Not Applicable</code><br><code>Direct</code><br><code>Indirect</code> |
| `TopSpeedMPH` | `Int64` | Quantitative: discrete | NHTSA variable `TopSpeedMPH`; label: Top Speed (MPH); variable ID: 139; group: Engine; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,672,385 | 327,490 | 95 | <code>120</code><br><code>121</code><br><code>110</code><br><code>111</code><br><code>119</code> |
| `TrackWidth` | `Float64` | Quantitative: continuous | NHTSA variable `TrackWidth`; label: Track Width (inches); variable ID: 159; group: Exterior/Body; DataType: `decimal`; stored in Parquet as `Float64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,967,696 | 32,179 | 27 | <code>71.6</code><br><code>60.3</code><br><code>64.4</code><br><code>61.8</code><br><code>61.4</code> |
| `TractionControl` | `String` | Categorical: nominal | NHTSA variable `TractionControl`; label: Traction Control; variable ID: 100; group: Active Safety System; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,159,546 | 1,840,329 | 3 | <code>Not Applicable</code><br><code>Standard</code><br><code>Optional</code> |
| `TrailerBodyType` | `String` | Categorical: nominal | NHTSA variable `TrailerBodyType`; label: Trailer Body Type; variable ID: 117; group: Exterior/Trailer; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 3,616 | 2,996,259 | 5 | <code>Special Purpose Trailer</code><br><code>Not Applicable</code><br><code>Camping or Travel Trailer</code><br><code>Utility Trailer</code><br><code>Tank Trailer</code> |
| `TrailerLength` | `Int64` | Quantitative: continuous | NHTSA variable `TrailerLength`; label: Trailer Length (feet); variable ID: 118; group: Exterior/Trailer; DataType: `decimal`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,999,589 | 286 | 93 | <code>1</code><br><code>78</code><br><code>75</code><br><code>96</code><br><code>86</code> |
| `TrailerType` | `String` | Categorical: nominal | NHTSA variable `TrailerType`; label: Trailer Type Connection; variable ID: 116; group: Exterior/Trailer; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 3,624 | 2,996,251 | 3 | <code>Not Applicable</code><br><code>Straight Semi/Semi Trailer</code><br><code>Ball Type Pull</code> |
| `TransmissionSpeeds` | `Int64` | Quantitative: discrete | NHTSA variable `TransmissionSpeeds`; label: Transmission Speeds; variable ID: 63; group: Mechanical/Transmission; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,827,579 | 1,172,296 | 10 | <code>4</code><br><code>5</code><br><code>3</code><br><code>8</code><br><code>9</code> |
| `TransmissionStyle` | `String` | Categorical: nominal | NHTSA variable `TransmissionStyle`; label: Transmission Style; variable ID: 37; group: Mechanical/Transmission; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,392,108 | 1,607,767 | 8 | <code>Not Applicable</code><br><code>Automatic</code><br><code>Dual-clutch Transmission (DCT)</code><br><code>Manual/Standard</code><br><code>Continuously Variable Transmission (CVT)</code> |
| `Trim` | `String` | Categorical: nominal | NHTSA variable `Trim`; label: Trim; variable ID: 38; group: General; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 537,755 | 2,462,120 | 3,340 | <code>Base/Acura Watch Plus</code><br><code>ILX</code><br><code>Special Edition</code><br><code>Premium/Tech</code><br><code>Premium Package/Technology Plus Package</code> |
| `Trim2` | `String` | Categorical: nominal | NHTSA variable `Trim2`; label: Trim2; variable ID: 109; group: General; DataType: `string`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,916,766 | 83,109 | 173 | <code>NAVI</code><br><code>w/ NAVI</code><br><code>Honda Sensing</code><br><code>L (Low Line)</code><br><code>H (High Line)</code> |
| `Turbo` | `String` | Categorical: nominal | NHTSA variable `Turbo`; label: Turbo; variable ID: 135; group: Engine; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,279,528 | 720,347 | 3 | <code>Not Applicable</code><br><code>Yes</code><br><code>No</code> |
| `VIN` | `String` | Identifier: nominal | NHTSA variable `VIN`; label: VIN; variable ID: not captured; group: not captured; DataType: `None`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 0 | 2,999,875 | 2,999,875 | <code>00000000011111111</code><br><code>00000000011615922</code><br><code>00000000012631735</code><br><code>00000000012646077</code><br><code>00000000012686460</code> |
| `ValveTrainDesign` | `String` | Categorical: nominal | NHTSA variable `ValveTrainDesign`; label: Valve Train Design; variable ID: 62; group: Engine; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,848,779 | 1,151,096 | 5 | <code>Not Applicable</code><br><code>Dual Overhead Cam (DOHC)</code><br><code>Single Overhead Cam (SOHC)</code><br><code>Overhead Valve (OHV)</code><br><code>Camless Valve Actuation (CVA)</code> |
| `VehicleType` | `String` | Categorical: nominal | NHTSA variable `VehicleType`; label: Vehicle Type; variable ID: 39; group: General; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 3,200 | 2,996,675 | 9 | <code>TRAILER</code><br><code>MULTIPURPOSE PASSENGER VEHICLE (MPV)</code><br><code>PASSENGER CAR</code><br><code>INCOMPLETE VEHICLE</code><br><code>TRUCK</code> |
| `WheelBaseLong` | `Float64` | Quantitative: continuous | NHTSA variable `WheelBaseLong`; label: Wheelbase (inches) To; variable ID: 112; group: Exterior/Dimension; DataType: `decimal`; stored in Parquet as `Float64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,929,869 | 70,006 | 32 | <code>107.1</code><br><code>106.6</code><br><code>148.0</code><br><code>176.0</code><br><code>164.2</code> |
| `WheelBaseShort` | `Float64` | Quantitative: continuous | NHTSA variable `WheelBaseShort`; label: Wheelbase (inches) From; variable ID: 111; group: Exterior/Dimension; DataType: `decimal`; stored in Parquet as `Float64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,440,900 | 1,558,975 | 241 | <code>105.1</code><br><code>106.3</code><br><code>108.0</code><br><code>95.4</code><br><code>116.0</code> |
| `WheelBaseType` | `String` | Categorical: nominal | NHTSA variable `WheelBaseType`; label: Wheelbase Type; variable ID: 60; group: Exterior/Body; DataType: `lookup`; stored in Parquet as `String`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,893,100 | 106,775 | 6 | <code>Short</code><br><code>Long</code><br><code>Extra Long</code><br><code>Standard</code><br><code>Medium</code> |
| `WheelSizeFront` | `Int64` | Quantitative: discrete | NHTSA variable `WheelSizeFront`; label: Wheel Size Front (inches); variable ID: 119; group: Exterior/Wheel Tire; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,805,140 | 1,194,735 | 9 | <code>16</code><br><code>17</code><br><code>18</code><br><code>19</code><br><code>20</code> |
| `WheelSizeRear` | `Int64` | Quantitative: discrete | NHTSA variable `WheelSizeRear`; label: Wheel Size Rear (inches); variable ID: 120; group: Exterior/Wheel Tire; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,805,741 | 1,194,134 | 9 | <code>16</code><br><code>17</code><br><code>18</code><br><code>19</code><br><code>20</code> |
| `Wheels` | `Int64` | Quantitative: discrete | NHTSA variable `Wheels`; label: Number of Wheels; variable ID: 115; group: Exterior/Wheel Tire; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 1,182,140 | 1,817,735 | 4 | <code>4</code><br><code>6</code><br><code>5</code><br><code>1</code> |
| `Windows` | `Int64` | Quantitative: discrete | NHTSA variable `Windows`; label: Windows; variable ID: 40; group: Exterior/Body; DataType: `int`; stored in Parquet as `Int64`. See the [NHTSA variable list](https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableList?format=json). | 2,905,772 | 94,103 | 2 | <code>4</code><br><code>6</code> |
## `NHTSA_Complaints`

One query/record-key complaint record; top-level JSON keys are separate columns. **1,390,528 rows; 17 columns.**

| Column | Type | Variable classification | Definition / interpretation | Missing Values | Total rows with values | Distinct values | 5 distinct values |
|---|---|---|---|---:|---:|---:|---|
| `query_id` | `Int64` | Identifier: nominal | Source or project identifier `query_id`; use as a key, not a numeric measure. | 0 | 1,390,528 | 6,707 | <code>9989</code><br><code>11348</code><br><code>11354</code><br><code>11357</code><br><code>11366</code> |
| `record_key` | `Int64` | Identifier: nominal | Source or project identifier `record_key`; use as a key, not a numeric measure. | 0 | 1,390,528 | 5,128 | <code>0</code><br><code>1</code><br><code>2</code><br><code>3</code><br><code>4</code> |
| `model_year` | `Int64` | Temporal: discrete year (interval) | Model or product year reported in `model_year`; retained as an interval year. | 0 | 1,390,528 | 60 | <code>2000</code><br><code>2002</code><br><code>2003</code><br><code>2004</code><br><code>2005</code> |
| `make` | `String` | Categorical: nominal | Source NHTSA record field `make`; local query context and returned record fields can have different meanings. | 0 | 1,390,528 | 62 | <code>Toyota</code><br><code>Mercedes-Benz</code><br><code>Dodge</code><br><code>Chevrolet</code><br><code>Ford</code> |
| `model` | `String` | Categorical: nominal | Source NHTSA record field `model`; local query context and returned record fields can have different meanings. | 0 | 1,390,528 | 917 | <code>MR2 Spyder</code><br><code>C-Class</code><br><code>CLK-Class</code><br><code>E-Class</code><br><code>S-Class</code> |
| `components` | `String` | Categorical: nominal, multiple labels | Comma-separated NHTSA component categories associated with the complaint; split labels only after resolving repeated complaint IDs. | 0 | 1,390,528 | 9,114 | <code>ENGINE AND ENGINE COOLING</code><br><code>STRUCTURE</code><br><code>STEERING</code><br><code>ENGINE AND ENGINE COOLING,UNKNOWN OR OTHER</code><br><code>ENGINE AND ENGINE COOLING,POWER TRAIN</code> |
| `crash` | `Boolean` | Categorical: binary nominal | Boolean status returned in `crash`. | 0 | 1,390,528 | 2 | <code>false</code><br><code>true</code> |
| `dateComplaintFiled` | `Date` | Temporal: date/time | NHTSA complaint filing date; parsed from `MM/DD/YYYY` and used as the provisional feature eligibility date. | 0 | 1,390,528 | 11,691 | <code>2004-01-15</code><br><code>2002-11-24</code><br><code>2002-11-06</code><br><code>2002-10-10</code><br><code>2002-10-02</code> |
| `dateOfIncident` | `Date` | Temporal: date/time | NHTSA-reported incident date parsed from `MM/DD/YYYY`; not used as the provisional evidence availability date. | 54,591 | 1,335,937 | 12,916 | <code>2003-08-15</code><br><code>2002-11-15</code><br><code>2002-08-15</code><br><code>2002-09-20</code><br><code>2002-09-01</code> |
| `fire` | `Boolean` | Categorical: binary nominal | Boolean status returned in `fire`. | 0 | 1,390,528 | 2 | <code>false</code><br><code>true</code> |
| `manufacturer` | `String` | Categorical: nominal | Source NHTSA record field `manufacturer`; local query context and returned record fields can have different meanings. | 0 | 1,390,528 | 69 | <code>Toyota Motor Corporation</code><br><code>Mercedes-Benz USA, LLC</code><br><code>Chrysler (FCA US, LLC)</code><br><code>Continental Tire the Americas, LLC</code><br><code>Mercedes-Benz USA, LLC - DBA Sprinter</code> |
| `numberOfDeaths` | `Int64` | Quantitative: discrete | Numeric value expanded from the NHTSA record field `numberOfDeaths`. | 0 | 1,390,528 | 18 | <code>0</code><br><code>1</code><br><code>3</code><br><code>4</code><br><code>2</code> |
| `numberOfInjuries` | `Int64` | Quantitative: discrete | Numeric value expanded from the NHTSA record field `numberOfInjuries`. | 0 | 1,390,528 | 28 | <code>0</code><br><code>1</code><br><code>2</code><br><code>3</code><br><code>7</code> |
| `odiNumber` | `Int64` | Identifier: nominal | Complaint identifier; use to deduplicate repeated complaint versions within each vehicle cohort, not as a magnitude. | 0 | 1,390,528 | 1,226,727 | <code>10071023</code><br><code>770060</code><br><code>8022295</code><br><code>8020642</code><br><code>767980</code> |
| `products` | `String` | Text / structured text | JSON array describing associated vehicles/equipment/tires; remains JSON text in Parquet and is used only to check explicit vehicle applicability. | 0 | 1,390,528 | 23,303 | <code>[{&quot;type&quot;:&quot;Equipment&quot;,&quot;productYear&quot;:&quot;2000&quot;,&quot;productMake&quot;:&quot;TOYOTA&quot;,&quot;productModel&quot;:&quot;MR2 SPYDER&quot;,&quot;manufacturer&quot;...</code><br><code>[{&quot;type&quot;:&quot;Equipment&quot;,&quot;productYear&quot;:&quot;2001&quot;,&quot;productMake&quot;:&quot;TOYOTA&quot;,&quot;productModel&quot;:&quot;MR2 SPYDER&quot;,&quot;manufacturer&quot;...</code><br><code>[{&quot;type&quot;:&quot;Equipment&quot;,&quot;productYear&quot;:&quot;2002&quot;,&quot;productMake&quot;:&quot;TOYOTA&quot;,&quot;productModel&quot;:&quot;MR2 SPYDER&quot;,&quot;manufacturer&quot;...</code><br><code>[{&quot;type&quot;:&quot;Vehicle&quot;,&quot;productYear&quot;:&quot;2002&quot;,&quot;productMake&quot;:&quot;MERCEDES BENZ&quot;,&quot;productModel&quot;:&quot;C230&quot;,&quot;manufacturer&quot;:...</code><br><code>[{&quot;type&quot;:&quot;Vehicle&quot;,&quot;productYear&quot;:&quot;2002&quot;,&quot;productMake&quot;:&quot;MERCEDES BENZ&quot;,&quot;productModel&quot;:&quot;C320&quot;,&quot;manufacturer&quot;:...</code> |
| `summary` | `String` | Text / structured text | NHTSA narrative or structured source field `summary`; extract features explicitly before modeling. | 6 | 1,390,522 | 1,225,203 | <code>THE AIRMATIC SYSTEM FAILS AND THE BODY OF VEHICLE CAME DOWN TO TOUCH ALL 4 WHEELS WHILE DRIVING HIGH SPEED ...</code><br><code>WHEN CONSUMER EXITED FROM HIGHWAY, AT A LOW RATE OF SPEED, THE VEHICLE&#x27;S ESP(ANTI-TILT ) SYSTEM  ENGAGED.  ...</code><br><code>WHILE MAKING A LEFT TURN TO PULL INTO THE GARAGE, THE VEHICLE SUDDENLY ACCELERATED AT A HIGH SPEED.  THE CO...</code><br><code>I WAS DRIVING AND WHEN I TRIED TO BRAKE. THE BRAKES WENT ALL THE WAY TO THE FLOOR. RED LIGHT LET UP IN THE ...</code><br><code>RECENTLY ON TODAY 7-12-16 AT 174,000 MILES THE DISPLAY ON MY 2003 E320 4S MERCEDES BENZ ALL OF A SUDDENLY D...</code> |
| `vin` | `String` | Categorical: nominal | Source NHTSA record field `vin`; local query context and returned record fields can have different meanings. | 174,683 | 1,215,845 | 280,131 | <code>JTDFR3206Y0</code><br><code>WDBRN47J52A</code><br><code>WDBRN47J12A</code><br><code>WDBRH64J92F</code><br><code>WDBRH64J02F</code> |

## `NHTSA_Recalls`

One query/record-key recall record; top-level JSON keys are separate columns. **26,993 rows; 20 columns.**

| Column | Type | Variable classification | Definition / interpretation | Missing Values | Total rows with values | Distinct values | 5 distinct values |
|---|---|---|---|---:|---:|---:|---|
| `query_id` | `Int64` | Identifier: nominal | Source or project identifier `query_id`; use as a key, not a numeric measure. | 0 | 26,993 | 6,606 | <code>1114</code><br><code>1294</code><br><code>1405</code><br><code>1516</code><br><code>1528</code> |
| `record_key` | `Int64` | Identifier: nominal | Source or project identifier `record_key`; use as a key, not a numeric measure. | 0 | 26,993 | 33 | <code>0</code><br><code>1</code><br><code>2</code><br><code>3</code><br><code>4</code> |
| `model_year` | `Int64` | Temporal: discrete year (interval) | Model or product year reported in `model_year`; retained as an interval year. | 0 | 26,993 | 60 | <code>1961</code><br><code>1963</code><br><code>1964</code><br><code>1965</code><br><code>1966</code> |
| `make` | `String` | Categorical: nominal | Source NHTSA record field `make`; local query context and returned record fields can have different meanings. | 0 | 26,993 | 68 | <code>Chevrolet</code><br><code>Buick</code><br><code>Opel</code><br><code>Pontiac</code><br><code>Dodge</code> |
| `model` | `String` | Categorical: nominal | Source NHTSA record field `model`; local query context and returned record fields can have different meanings. | 0 | 26,993 | 912 | <code>Corvair</code><br><code>Wildcat</code><br><code>Biscayne</code><br><code>Corvette</code><br><code>Impala</code> |
| `Component` | `String` | Categorical: nominal | NHTSA recall component label; not used as a narrative text feature in the first ablation. | 0 | 26,993 | 396 | <code>VISIBILITY:DEFROSTER/DEFOGGER/HVAC SYSTEM</code><br><code>VEHICLE SPEED CONTROL</code><br><code>ENGINE AND ENGINE COOLING:EXHAUST SYSTEM:MANIFOLD/HEADER/MUFFLER/TAIL PIPE</code><br><code>STEERING:LINKAGES:ARM:IDLER AND ATTACHMENT</code><br><code>SERVICE BRAKES, AIR:DISC:CALIPER</code> |
| `Consequence` | `String` | Text / structured text | NHTSA narrative or structured source field `Consequence`; extract features explicitly before modeling. | 844 | 26,149 | 4,663 | <code>UNDER THESE CONDITIONS, VEHICLE OCCUPANTS MAY EXPERIENCECARBON MONOXIDE POISONING.</code><br><code>UNDER THESE CONDITIONS, THE VEHICLE OPERATOR MAY NOT BE ABLE TO STOP THE CAR, POSSIBLY RESULTING IN A VEHIC...</code><br><code>UNEXPECTED LOSS OF THROTTLE CONTROL MAY CAUSE LOSS OFVEHICLE CONTROL AND AN ACCIDENT.</code><br><code>INCREASED STEERING EFFORT OR INTERFERENCE WITH STEERINGCONTROL MAY LEAD TO LOSS OF VEHICLE CONTROL AND AN A...</code><br><code>IF THE SHOULDER BELT IS SEPARATED FROM THE SEAT BELTBUCKLE, THE SHOULDER BELT WILL NOT PROPERLY PROTECT SEA...</code> |
| `Make` | `String` | Categorical: nominal | Source NHTSA record field `Make`; local query context and returned record fields can have different meanings. | 0 | 26,993 | 68 | <code>Chevrolet</code><br><code>Buick</code><br><code>Opel</code><br><code>Pontiac</code><br><code>Dodge</code> |
| `Manufacturer` | `String` | Categorical: nominal | Source NHTSA record field `Manufacturer`; local query context and returned record fields can have different meanings. | 0 | 26,993 | 282 | <code>GENERAL MOTORS CORP.</code><br><code>CARDONE INDUSTRIES, INC.</code><br><code>GEN MTRS CORP-OVERSEAS</code><br><code>CHRYSLER CORPORATION</code><br><code>FORD MOTOR COMPANY</code> |
| `Model` | `String` | Categorical: nominal | Source NHTSA record field `Model`; local query context and returned record fields can have different meanings. | 0 | 26,993 | 912 | <code>Corvair</code><br><code>Wildcat</code><br><code>Biscayne</code><br><code>Corvette</code><br><code>Impala</code> |
| `ModelYear` | `Int64` | Temporal: discrete year (interval) | Model or product year reported in `ModelYear`; retained as an interval year. | 0 | 26,993 | 60 | <code>1961</code><br><code>1963</code><br><code>1964</code><br><code>1965</code><br><code>1966</code> |
| `NHTSAActionNumber` | `String` | Categorical: nominal | Source NHTSA record field `NHTSAActionNumber`; local query context and returned record fields can have different meanings. | 20,944 | 6,049 | 576 | <code>SQ99001</code><br><code>EA88005</code><br><code>EA91002</code><br><code>PE89093</code><br><code>EA89030</code> |
| `NHTSACampaignNumber` | `String` | Identifier: nominal | Source or project identifier `NHTSACampaignNumber`; use as a key, not a numeric measure. | 0 | 26,993 | 6,117 | <code>71V224000</code><br><code>76V120000</code><br><code>69V030000</code><br><code>67V098000</code><br><code>03E032000</code> |
| `Notes` | `String` | Text / structured text | NHTSA narrative or structured source field `Notes`; extract features explicitly before modeling. | 701 | 26,292 | 2,708 | <code>GENERAL MOTOR CAMPAIGN NO 69-C-13.  POSSIBILITY THAT FUMES IN  DIRECT AIRHEATING SYSTEM MAY BE TRANSFERRED ...</code><br><code>GENERAL MOTOR CAMPAIGN NO 67C3.  POSSIBILITY THAT SEPARATED MOTOR MOUNT MAYALLOW MOTOR TO LIFT AND ROTATE, ...</code><br><code>SYSTEM: EXHAUST.VEHICLE DESCRIPTION: PASSENGER CARS.</code><br><code>POSSIBILITY THAT, AFTER EXTENDED MILEAGE, THE IDLER ARM ASSEMBLY COULD SEPARATE DURING VEHICLE OPERATION AN...</code><br><code>ALSO, CUSTOMERS CAN CONTACT THE NATIONAL HIGHWAY TRAFFIC SAFETY ADMINISTRATION&#x27;S AUTO SAFETY HOTLINE AT 1-8...</code> |
| `Remedy` | `String` | Text / structured text | NHTSA narrative or structured source field `Remedy`; extract features explicitly before modeling. | 416 | 26,577 | 5,783 | <code>RESEAL NEARBY BODY JOINTS AND REPLACE THE DRAW PLUG IN THE BODY WHICH WAS USED TO REMOVE SURPLUS BONDERIZIN...</code><br><code>CARDONE WILL NOTIFY ITS CUSTOMERS AND ALL UNSOLD INVENTORY WILL BE REPURCHASED AND WILL PROVIDE A FULL REFU...</code><br><code>INSPECT ENGINE MOUNTS AND INSTALL RESTRAINTS WHICH WILL LIMIT ENGINE LIFT AND ELIMINATE POSSIBLE SECONDARY ...</code><br><code>DEALER WILL INSPECT AND, IF NECESSARY, INSTALL CLAMPS ON FUEL LINE AND REPAIR FUEL INLET PIPE TO CARBURETOR...</code><br><code>THE DEALER WILL REPLACE THE DRIVER&quot;S SEAT PIVOT PIN BRACKET ON ALL AFFECTED VEHICLES, FREE OF CHARGE.</code> |
| `ReportReceivedDate` | `Date` | Temporal: date/time | NHTSA recall report-received date parsed from `DD/MM/YYYY`. | 0 | 26,993 | 3,886 | <code>1971-12-03</code><br><code>1976-08-02</code><br><code>1969-02-28</code><br><code>1967-10-23</code><br><code>2003-05-12</code> |
| `Summary` | `String` | Text / structured text | NHTSA narrative or structured source field `Summary`; extract features explicitly before modeling. | 436 | 26,557 | 5,783 | <code>EXHAUST EMISSIONS (CARBON MONOXIDE) MAY ENTER THE INTERIOR OF THE VEHICLE WHEN THE EXHAUST PIPE IS SUBSTANT...</code><br><code>REMANUFACTURERD REAR BRAKE CALIPERS, PART NOS. 18-7019, 18-7020, 16-7019, AND 16-7020, MANUFACTURED FROM FE...</code><br><code>THERE IS A POSSIBILITY THAT A SEPARATED MOTOR MOUNT MAY ALLOW THE ENGINE TO LIFT, WHICH MAY AFFECT THE THRO...</code><br><code>ON THE INVOLVED VEHICLES, BOTH THE FUEL LINE TO THE FUEL INLET PIPE CONNECTION AND FUEL INLET PIPE TO CARBU...</code><br><code>THE SEAT BACK PIVOT PIN BRACKET ON THE INVOLVED VEHICLES MAY FAIL. CONSEQUENTLY, THE SEAT BACK WILL ROTATE ...</code> |
| `overTheAirUpdate` | `Boolean` | Categorical: binary nominal | Boolean over-the-air-update indicator returned in the recall record; not a date. | 4,517 | 22,476 | 2 | <code>false</code><br><code>true</code> |
| `parkIt` | `Boolean` | Categorical: binary nominal | Boolean status returned in `parkIt`. | 4,517 | 22,476 | 2 | <code>false</code><br><code>true</code> |
| `parkOutSide` | `Boolean` | Categorical: binary nominal | Boolean status returned in `parkOutSide`. | 4,517 | 22,476 | 2 | <code>false</code><br><code>true</code> |

## `NHTSA_Safety`

One query/vehicle-variant pair; rating fields are wide columns. **8,144 rows; 35 columns.**

| Column | Type | Variable classification | Definition / interpretation | Missing Values | Total rows with values | Distinct values | 5 distinct values |
|---|---|---|---|---:|---:|---:|---|
| `query_id` | `Int64` | Identifier: nominal | Identifier for the query or NHTSA vehicle variant; retained as a linkage key. | 0 | 8,144 | 5,460 | <code>5415</code><br><code>5421</code><br><code>5439</code><br><code>5454</code><br><code>5487</code> |
| `vehicle_id` | `Int64` | Identifier: nominal | Identifier for the query or NHTSA vehicle variant; retained as a linkage key. | 0 | 8,144 | 8,144 | <code>2860</code><br><code>2898</code><br><code>2852</code><br><code>2900</code><br><code>2859</code> |
| `ComplaintsCount` | `Int64` | Quantitative: discrete | NHTSA reported count in `ComplaintsCount`; not VIN-specific. | 4 | 8,140 | 1,017 | <code>94</code><br><code>43</code><br><code>63</code><br><code>88</code><br><code>114</code> |
| `FrontCrashDriversideRating` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `FrontCrashDriversideRating`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 2,617 | 5,527 | 5 | <code>5</code><br><code>1</code><br><code>4</code><br><code>2</code><br><code>3</code> |
| `FrontCrashPassengersideRating` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `FrontCrashPassengersideRating`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 2,650 | 5,494 | 5 | <code>5</code><br><code>4</code><br><code>1</code><br><code>3</code><br><code>2</code> |
| `FrontCrashPicture` | `String` | Reference: URL | NHTSA safety-test media link in `FrontCrashPicture`. | 4,565 | 3,579 | 2,712 | <code>https://static.nhtsa.gov/crashTest/images/1997/LeSabre.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/1997/deville_a.gif</code><br><code>https://static.nhtsa.gov/crashTest/images/1997/Blazer.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/1997/Cavalier4.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/1997/Cavalier.jpg</code> |
| `FrontCrashVideo` | `String` | Reference: URL | NHTSA safety-test media link in `FrontCrashVideo`. | 5,220 | 2,924 | 2,120 | <code>https://static.nhtsa.gov/crashTest/videos/1995/camaro.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/1995/caprice.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/1995/caravan.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/1995/intrepid.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/1995/aerostar.wmv</code> |
| `InvestigationCount` | `Int64` | Quantitative: discrete | NHTSA reported count in `InvestigationCount`; not VIN-specific. | 4 | 8,140 | 20 | <code>3</code><br><code>9</code><br><code>5</code><br><code>1</code><br><code>4</code> |
| `Make` | `String` | Categorical: nominal | NHTSA safety field `Make` retained as source text. | 4 | 8,140 | 53 | <code>CADILLAC</code><br><code>CHEVROLET</code><br><code>DODGE</code><br><code>FORD</code><br><code>GEO</code> |
| `Model` | `String` | Categorical: nominal | NHTSA safety field `Model` retained as source text. | 4 | 8,140 | 723 | <code>DEVILLE</code><br><code>BLAZER</code><br><code>CAVALIER</code><br><code>DAKOTA</code><br><code>MUSTANG</code> |
| `ModelYear` | `Int64` | Temporal: discrete year (interval) | NHTSA model year converted to nullable `Int64`. | 4 | 8,140 | 32 | <code>1990</code><br><code>1991</code><br><code>1992</code><br><code>1993</code><br><code>1994</code> |
| `NHTSAElectronicStabilityControl` | `String` | Categorical: nominal | NHTSA safety field `NHTSAElectronicStabilityControl` retained as source text. | 4 | 8,140 | 3 | <code>No</code><br><code>Standard</code><br><code>Optional</code> |
| `NHTSAForwardCollisionWarning` | `String` | Categorical: nominal | NHTSA safety field `NHTSAForwardCollisionWarning` retained as source text. | 4 | 8,140 | 4 | <code>No</code><br><code>Optional</code><br><code>Standard</code><br><code>Standard &amp; Optional</code> |
| `NHTSALaneDepartureWarning` | `String` | Categorical: nominal | NHTSA safety field `NHTSALaneDepartureWarning` retained as source text. | 4 | 8,140 | 3 | <code>No</code><br><code>Optional</code><br><code>Standard</code> |
| `OverallFrontCrashRating` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `OverallFrontCrashRating`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 5,416 | 2,728 | 4 | <code>4</code><br><code>5</code><br><code>3</code><br><code>2</code> |
| `OverallRating` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `OverallRating`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 5,505 | 2,639 | 4 | <code>5</code><br><code>4</code><br><code>3</code><br><code>2</code> |
| `OverallSideCrashRating` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `OverallSideCrashRating`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 5,452 | 2,692 | 4 | <code>5</code><br><code>2</code><br><code>3</code><br><code>4</code> |
| `RecallsCount` | `Int64` | Quantitative: discrete | NHTSA reported count in `RecallsCount`; not VIN-specific. | 4 | 8,140 | 32 | <code>1</code><br><code>4</code><br><code>3</code><br><code>5</code><br><code>9</code> |
| `RolloverPossibility` | `Float64` | Quantitative: continuous | NHTSA rollover probability-like value converted to nullable `Float64`; interpretation and paired-rating availability should be checked before analysis. | 3,861 | 4,283 | 113 | <code>0.0</code><br><code>0.107</code><br><code>0.191</code><br><code>0.263</code><br><code>0.13</code> |
| `RolloverPossibility2` | `Float64` | Quantitative: continuous | NHTSA rollover probability-like value converted to nullable `Float64`; interpretation and paired-rating availability should be checked before analysis. | 6,385 | 1,759 | 48 | <code>0.0</code><br><code>0.13</code><br><code>0.09</code><br><code>0.105</code><br><code>0.113</code> |
| `RolloverRating` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `RolloverRating`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 3,862 | 4,282 | 4 | <code>2</code><br><code>3</code><br><code>4</code><br><code>5</code> |
| `RolloverRating2` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `RolloverRating2`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 6,385 | 1,759 | 5 | <code>3</code><br><code>1</code><br><code>4</code><br><code>2</code><br><code>5</code> |
| `SideCrashDriversideRating` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `SideCrashDriversideRating`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 3,286 | 4,858 | 5 | <code>1</code><br><code>4</code><br><code>3</code><br><code>2</code><br><code>5</code> |
| `SideCrashPassengersideRating` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `SideCrashPassengersideRating`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 3,599 | 4,545 | 5 | <code>3</code><br><code>4</code><br><code>2</code><br><code>1</code><br><code>5</code> |
| `SideCrashPicture` | `String` | Reference: URL | NHTSA safety-test media link in `SideCrashPicture`. | 4,746 | 3,398 | 2,629 | <code>https://static.nhtsa.gov/crashTest/images/1997/Devill-s.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/1997/Camaro-s.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/1997/Lumina-s.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/1997/Intrep-s.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/1997/Stratu-s.jpg</code> |
| `SideCrashVideo` | `String` | Reference: URL | NHTSA safety-test media link in `SideCrashVideo`. | 5,416 | 2,728 | 1,962 | <code>https://static.nhtsa.gov/crashTest/videos/2003/03BMWZ4_s.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/2003/03Astro_S.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/2003/03CrownVic_s.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/2003/03Accord_2Dr_S.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/2003/03Element_s.wmv</code> |
| `SidePoleCrashRating` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `SidePoleCrashRating`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 5,441 | 2,703 | 5 | <code>5</code><br><code>4</code><br><code>2</code><br><code>1</code><br><code>3</code> |
| `SidePolePicture` | `String` | Reference: URL | NHTSA safety-test media link in `SidePolePicture`. | 6,147 | 1,997 | 1,227 | <code>https://static.nhtsa.gov/crashTest/images/2011/v07136P039.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/2011/v07414P039.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/2011/v07033P002.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/2011/v07369P039.jpg</code><br><code>https://static.nhtsa.gov/crashTest/images/2011/v07067P067.jpg</code> |
| `SidePoleVideo` | `String` | Reference: URL | NHTSA safety-test media link in `SidePoleVideo`. | 6,157 | 1,987 | 1,223 | <code>https://static.nhtsa.gov/crashTest/videos/2011/v07136C012.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/2011/v07414C012.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/2011/v07033C014.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/2011/v07369C012.wmv</code><br><code>https://static.nhtsa.gov/crashTest/videos/2011/v07067C012.wmv</code> |
| `VehicleDescription` | `String` | Categorical: nominal | NHTSA safety field `VehicleDescription` retained as source text. | 0 | 8,144 | 8,144 | <code>1990 Cadillac Deville 4-DR.</code><br><code>1990 Chevrolet S-10 Blazer 2-DR. 4x4</code><br><code>1990 Chevrolet Cavalier 4-DR.</code><br><code>1990 Dodge Dakota 2-DR. 4x4</code><br><code>1990 Ford Mustang 2-DR. Convertible</code> |
| `VehicleId` | `Int64` | Categorical: nominal | NHTSA safety variant identifier converted to nullable `Int64`; use as an identifier, not a measure. | 0 | 8,144 | 8,144 | <code>2860</code><br><code>2898</code><br><code>2852</code><br><code>2900</code><br><code>2859</code> |
| `combinedSideBarrierAndPoleRating-Front` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `combinedSideBarrierAndPoleRating-Front`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 5,452 | 2,692 | 5 | <code>5</code><br><code>4</code><br><code>1</code><br><code>2</code><br><code>3</code> |
| `combinedSideBarrierAndPoleRating-Rear` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `combinedSideBarrierAndPoleRating-Rear`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 5,541 | 2,603 | 5 | <code>5</code><br><code>4</code><br><code>2</code><br><code>3</code><br><code>1</code> |
| `dynamicTipResult` | `String` | Categorical: nominal | NHTSA safety field `dynamicTipResult` retained as source text. | 4,088 | 4,056 | 2 | <code>No Tip</code><br><code>Tip</code> |
| `sideBarrierRating-Overall` | `Int64` | Categorical: ordinal + status | NHTSA safety rating field `sideBarrierRating-Overall`; integer rating codes are stored as `Int64`; non-numeric source statuses such as `Not Rated` become null during export. | 5,452 | 2,692 | 4 | <code>5</code><br><code>2</code><br><code>4</code><br><code>3</code> |

## `Youtube_Comments`

One imported YouTube comment record. **1,981,686 rows; 12 columns.**

Read-only date/text check (2026-10-05): all publication dates parse as
`MM-DD-YYYY`, spanning 2012-01-24 through 2026-08-26. There are 675,792 comments
published on or before the provisional `2020-12-31` comparison cutoff, 7,920 rows
with an update date later than publication, and 94 empty/whitespace-only texts.
Extraction dates span 2026-06-04 through 2026-08-26. Historical text versions,
historical engagement values, and the listing price-observation date remain
unverified; the date-filtered count does not establish historical availability.

| Column | Type | Variable classification | Definition / interpretation | Missing Values | Total rows with values | Distinct values | 5 distinct values |
|---|---|---|---|---:|---:|---:|---|
| `video_id` | `String` | Identifier: nominal | YouTube video identifier associated with the comment. | 0 | 1,981,686 | 4,224 | <code>-MXV5VjFV54</code><br><code>JH300fr1KO4</code><br><code>SdrvFgqY4zM</code><br><code>l8WHZ_2XAkA</code><br><code>6wX4kPl4cj0</code> |
| `playlist_id` | `String` | Identifier: nominal | YouTube playlist identifier associated with the video. | 0 | 1,981,686 | 59 | <code>PLdmWqCdsjCu6XXTGcfwQE-v-8qZKOUbPv</code><br><code>PLdmWqCdsjCu4IAZtt7cFYCI-5wLOu-W5v</code><br><code>PLdmWqCdsjCu6vdUJzxywmjNa9IMv9E4bP</code><br><code>PLdmWqCdsjCu4Ikxn-hvSiATgUkX5qo2iI</code><br><code>PLdmWqCdsjCu7Y-UCY7jFP1f86lI-IFBES</code> |
| `video_title` | `String` | Categorical: nominal | Title of the YouTube video associated with the comment. | 0 | 1,981,686 | 4,223 | <code>2026 Dodge Charger Daytona Review &#124; What Hellcat Fans Never Asked For (and that&#x27;s a pity)</code><br><code>The Craziest Honda Ever Is The 2023 Honda Civic Type R  &#124; First Drive Review</code><br><code>2022 Toyota GR86 &#124; Why The Best Isn&#x27;t Always The One To Get</code><br><code>The 2023 Integra Is Fantastic... If You Set Your Expectations Right...</code><br><code>Caddy&#x27;s Not Done With Gas... Yet... 2022 Cadillac CT5-V Blackwing Review</code> |
| `source` | `String` | Categorical: nominal | Source label recorded by the importer. | 0 | 1,981,686 | 1 | <code>comment</code> |
| `text` | `String` | Text / structured text | Comment text as imported; no sentiment score is calculated by this project. | 94 | 1,981,592 | 1,942,766 | <code>About the efficiency, especially my comment around 35:00. The EPA&#x27;s 72 MPGe rating isn&#x27;t really far off the...</code><br><code>very nice car.  i would never use the fake exhaust tho</code><br><code>If you could choose the sound the speakers play, it could be hilarious.</code><br><code>Your Venn diagram was hysterical, Alex!</code><br><code>i&#x27;m glad you being you and show this as you see it too</code> |
| `extracted_at` | `String` | Temporal: date/time | Date recorded when the comment was extracted; retained as source text. | 0 | 1,981,686 | 11 | <code>06-04-2026</code><br><code>06-08-2026</code><br><code>06-16-2026</code><br><code>06-18-2026</code><br><code>06-20-2026</code> |
| `comment_id` | `String` | Identifier: nominal | YouTube comment identifier; source primary key. | 0 | 1,981,686 | 1,981,686 | <code>UgzhGuCfKKwM7LgPkUN4AaABAg</code><br><code>Ugyf3dBDEdW_-rwDb194AaABAg</code><br><code>Ugxe_DDU9kw-gxc0XzF4AaABAg</code><br><code>UgxJxWL-6i8FouHKaFV4AaABAg</code><br><code>Ugy_A68z-9sHzbvmgT14AaABAg</code> |
| `author` | `String` | Categorical: nominal | Comment author name as supplied by YouTube. | 23 | 1,981,663 | 808,182 | <code>@AAutoBuyersGuide</code><br><code>@AntLive29</code><br><code>@wpelfeta</code><br><code>@HerrLT</code><br><code>@tallll70</code> |
| `like_count` | `Int64` | Quantitative: discrete | Like count recorded for the comment. | 0 | 1,981,686 | 3,450 | <code>22</code><br><code>27</code><br><code>21</code><br><code>3</code><br><code>0</code> |
| `reply_count` | `Int64` | Quantitative: discrete | Reply count recorded for the comment. | 0 | 1,981,686 | 291 | <code>10</code><br><code>2</code><br><code>4</code><br><code>1</code><br><code>0</code> |
| `published_at` | `String` | Temporal: date/time | Comment publication date/time; retained as source text. | 0 | 1,981,686 | 5,075 | <code>01-14-2026</code><br><code>01-15-2026</code><br><code>01-20-2026</code><br><code>01-19-2026</code><br><code>01-16-2026</code> |
| `updated_at` | `String` | Temporal: date/time | Comment update date/time; retained as source text. | 0 | 1,981,686 | 5,075 | <code>01-14-2026</code><br><code>01-15-2026</code><br><code>01-20-2026</code><br><code>01-19-2026</code><br><code>01-16-2026</code> |

## YouTube relevance annotation data (outside the source snapshot)

`YOUTUBE/relevance/prepare_relevance_round.py` produces annotation CSVs from read-only `Youtube_Comments.parquet` and canonical `Make` values from `NHTSA_vPIC.parquet`. These files do not change the six model-development tables and contain no inferred sentiment. The first round defaults to 400 enriched non-filler training candidates and 300 comment-uniform blind rows from seeded reserved videos. Videos already in the Toyota label sample are excluded from the blind population. Selected blind text matching prior labels after word normalization is also excluded. Every reserved video stays out of training; normalized exact text matches are blocked between the selected splits. This prevents normalized exact duplicate leakage, not semantic paraphrase leakage. Future prevalence estimates describe the eligible reserved-video population; enriched training proportions do not estimate source prevalence. Actual sample counts, populations, source SHA-256 hashes, and dependency/script versions are recorded in `sample_manifest.json`; the real-source round 01 audit and sample-integrity review are complete.

`train_sample.csv` and `blind_sample.csv` are frozen originals. Their `*_labels.csv` copies have the same rows/columns with three editable annotation fields. `second_reviewer_labels.csv` contains an independently selected 15% subset of each split. The row grain is one `comment_id`, not one make/aspect record. As of October 1, 2026, `relevance_round_01_reviewed` contains 400 training rows with 179 completed binary labels (92 yes, 87 no), 300 blind rows with 107 completed relevance labels, and 105 second-reviewer rows with no completed labels. Four training negatives lack a subtype and one uses the unsupported `question` subtype. The blind label distribution remains withheld during candidate selection.

| Annotation field | Definition |
|---|---|
| `comment_id`, `video_id`, `video_title`, `published_at`, `text` | Unchanged source identifiers/context/text; dates remain source strings. |
| `candidate_makes` | Pipe-separated canonical vPIC makes explicitly matched in the normalized title. Blank means unmatched; this is not confirmed comment-level make attribution. No aliases are inferred. |
| `split` | `train` or `blind`; assignment is based on video reservation before comment sampling. |
| `sample_stratum` | Blind: `comment_uniform_reserved_videos`. Training: `creator_spam_candidate`, `short_nonfiller`, or `general_nonfiller`. These sampling categories are not human labels. |
| `prefilter_reason` | Existing exact filler-rule flag for audit; flagged rows remain eligible for blind sampling and are excluded from the new non-filler training pool. |
| `normalized_text_sha256` | SHA-256 of case-folded Unicode words joined with spaces, ignoring punctuation; used for exact normalized-text duplicate protection. Original text is retained. |
| `vehicle_relevant` | Human `yes`, `no`, or `unclear`; blank means not yet labeled. Questions/facts about a vehicle are `yes` even without an opinion. |
| `negative_subtype` | Only for `no`: `creator_video`, `spam`, `generic_reaction`, or `other`; blank for other relevance outcomes. |
| `labeler_notes` | Human explanations, language uncertainty, or reconciled labeling disagreements. |

`source_audit.json` profiles source row/video counts, empty and normalized-duplicate text, publication dates/months, length bands, proposed filler flags, tentative title make coverage, and the top video counts. It fails preparation for duplicate/missing comment IDs or missing video IDs. Date profiling accepts the source `MM-DD-YYYY`, ISO dates, and ISO timestamps; invalid values are counted and original strings remain unchanged. No language model is used in this audit. `toyota_error_review.csv` preserves baseline prediction/error fields and adds `review_notes`; predictions are not gold labels. Label progress JSON describes training support/independent-review agreement and blind completion, while withholding blind label distributions until model selection is locked. Execution steps belong in the README and runbook.

Partial-annotation recovery creates a separate working copy with the same comment grain and schema. `recovery_manifest.json` records hashes of preserved input CSVs, user-confirmed spreadsheet row bounds (header is row 1), the meaning of reviewed blank relevance cells, and source line-ending restoration counts. Only the three annotation columns are carried across onto checksum-verified original source fields; substantive source edits/conflicting labels fail recovery. In round 01, the user explicitly confirmed reviewed blanks as `no` through training row 180 and blind row 108. Later blanks stay unlabeled. The working training file has 179 labels (95 yes, 84 no); 221 training rows and 193 blind rows remain unlabeled. No unknown negative subtype is inferred; missing subtypes and the unsupported `question` subtype stay in the annotations for human review.

Partial diagnostic outputs are outside the model-development snapshot. `relevance_cv_predictions.csv` and `cv_error_review.csv` have one labeled training `comment_id`, original annotation fields, original working-file `csv_row`, video-grouped `cv_fold`, uncalibrated `prob_vehicle_relevant`, and the model's `predicted_vehicle_relevant`; the error file adds `review_notes`. `training_review_queue.csv` has one pending training `comment_id` with the same source/annotation fields and suggested predictions from a model fit on all explicit training labels; `distance_from_half` is the absolute distance of that uncalibrated score from 0.5, used only to prioritize human review. `annotation_issues.csv` lists training annotations needing subtype review without changing their binary labels. The `run_manifest.json` preserves input-label/script hashes, source snapshot, folds, seed, and actual runtime dependency versions. None of these predictions is a human label, production qualification, or sentiment feature; blind annotations are not read by the evaluator.

User-reviewed corrections are preserved in `relevance_round_01_reviewed/`, a separate annotation snapshot. Exact `yes`, `no`, or `unclear` entries in training error `review_notes` are imported as user relevance judgments, alongside changed annotation fields. Corrections are joined by `comment_id`, checked against the original source fields, and logged in `review_changes.csv` with `field`, `old`, `new`, and `review_file`. Unchanged export fields do not overwrite newer corrections; inconsistent overlapping changes fail import. Negative subtypes are cleared when a corrected label ceases to be `no`. `review_manifest.json` records input hashes and class/change counts. Round 01 has 92 yes and 87 no labels after review, with 221 training rows still unlabeled. Remaining subtype issues preserve the user's binary labels. Blind annotations are copied unchanged without entering training.

Local model-comparison outputs have one `(model, comment_id)` prediction for each labeled training comment. `cv_predictions.csv` adds `model`, `cv_fold`, `prob_vehicle_relevant`, and `predicted_vehicle_relevant` to source/annotation fields; `cv_errors.csv` adds human `review_notes`. `fold_assignments.csv` has one training `comment_id` and `video_id` with its validation fold, shared by all candidates. `model_comparison.csv` reports confusion counts, relevant/irrelevant recall, rejection precision, Brier score, and fit/validation time at 0.5. `comparison_metrics.json` adds approximate video-cluster-bootstrap 95% intervals and `production_ready=false`. `threshold_diagnostics.csv` compares fixed exploratory cutoffs; a row is classified negative when its score is below the cutoff. No cutoff is automatically approved. `calibration_bins.csv` records support, mean uncalibrated score, and observed relevance in 0.1-width bins on the enriched training sample. `segment_metrics.csv` reports training stratum and tentative title-make context; this is not unseen-make evaluation. `encoder_input_audit.json` counts comments exceeding the neural encoder token limit; truncation affects model inputs only, not source text. Neural weights/revisions, input-label and script hashes, seed, folds, dependency versions, and training limits are recorded in the run manifest. Fold checkpoint JSONs contain held-out comment IDs, probabilities, and elapsed time. CV after correcting model-selected errors is descriptive; blind evaluation, language/near-duplicate audits, and release validation remain separate.

## Saved relevance candidate and preserving pilot (outside the source snapshot)

`relevance_candidate.py fit` saves the SetFit encoder/head in a versioned model directory. `training_snapshot.csv` contains only completed binary training rows and their original source/annotation fields plus `csv_row`. `protected_train_sample.csv` and `protected_blind_sample.csv` retain all frozen pool rows, including unfinished training examples, for leakage protection. `candidate_manifest.json` records these hashes, model file hashes, the pinned encoder revision, canonical makes from the source sampling manifest, token limit, dependency versions, cutoff, and evaluation policy. `label_guide_snapshot.md` preserves the guide used for the candidate. None of these artifacts changes the six source Parquet tables or supplies a sentiment label.

`pilot_sample.csv` has one selected `comment_id` with the five source fields and tentative `candidate_makes`. Its enriched sampling strata combine tentative title-make context, publication year, and text-length bands: `short` below 40 characters, `medium` from 40 through 199, and `long` at least 200. Frozen training-pool/blind videos, all seed-reserved videos, and normalized exact text matching the protected pools are excluded. Sampling gives eligible strata equal allocation up to capacity; it does not estimate full-source prevalence. `pilot_manifest.json` records the actual source hash and stratum quotas. Batch checkpoint JSONs contain stable IDs, scores, token counts, timing, and run configuration.

| Pilot score field | Definition |
|---|---|
| `comment_id`, `video_id`, `video_title`, `published_at`, `text` | Original source context/text, with source null strings represented as empty CSV cells; no source row is modified. |
| `candidate_makes`, `normalized_text_sha256`, `prefilter_reason` | Tentative canonical title context, normalized exact text hash, and proposed exact-rule audit reason. None is a confirmed target, relevance label, or sentiment. |
| `published_month`, `length_band` | Derived publication month (`YYYY-MM`, or `unknown`) and the sampling length band. Original publication text remains present. |
| `prob_vehicle_relevant` | Uncalibrated saved-model score. It is not a human label or a verified probability for the source population. |
| `proposed_rejection` | `yes` only for scores strictly below the frozen candidate cutoff, initially 0.20. No row is actually excluded. |
| `relevance_decision` | Always `review_or_pass` in the pilot and current blind predictions; automatic rejection is disabled. |
| `input_tokens`, `input_truncated` | Token count including special tokens, and `yes` when it exceeds the frozen 256-token limit. Only model input is truncated; original text remains present. |
| `model_version`, `candidate_manifest_sha256` | Saved-model directory version and exact candidate manifest lineage. |

`pilot_audit_sample.csv` selects up to 100 rows from each of three disjoint strata: proposed rejection; retained scores below 0.50; retained scores at least 0.50. `audit_stratum_rows` and `audit_stratum_sampled` record the population/sample support for weighting or separate stratum reporting. `pilot_audit_labels.csv` adds human `vehicle_relevant`, `negative_subtype`, `language`, and `labeler_notes`; blanks mean unfinished. Resume preserves this editable file. `pilot_summary.json` reports actual row counts, proposed exclusions, truncation, runtime, end-of-run process RSS when available, and tentative make/video/month/length coverage. Actual exclusions stay zero.

The blind evaluator produces one frozen blind `comment_id` prediction and `blind_metrics.json` after complete relevance annotations. `unclear` comments are kept and reported separately from binary metrics; unfinished comments block evaluation. The report includes the fixed-cutoff confusion counts, exact-prefilter comparison, row/video uncertainty summaries, support screens, and independent-review limitations. The round-level evaluation receipt records model/input hashes and test consumption. No release or source-wide inferred label is published by these files. Execution and review steps belong in the README/runbook.

**Completed pilot profile (2026-10-01):** `qualification_pilot_v1` contains
10,000 distinct comment IDs from 1,496 videos. Score strata contain 2,243
proposed rejections, 1,991 scores in [0.20, 0.50), and 5,766 scores >=0.50;
actual exclusions are zero. The enriched pilot is not a prevalence estimate.
There are 97 truncated model inputs, with original text retained, and 128
excess rows across 54 repeated normalized-text groups within the sample;
these are different comment IDs rather than duplicate IDs. No protected
training/blind ID, video, or normalized-text overlaps were found. The audit
contains 100 rows per stratum, with zero completed human relevance labels.
Candidate/model hashes, source fields, and all 79 score checkpoints agree in
[the artifact review](YOUTUBE/outputs/pipeline_run_review_01/run_review.json).

## Make/aspect development annotations (outside the source snapshot)

`prepare_aspect_round.py` selects human-confirmed relevant comments from the reviewed round, optionally supplemented by confirmed Toyota feedback. Frozen relevance blind IDs/videos/normalized text are excluded; IDs and normalized texts are deduplicated. The sample is enriched across tentative make contexts and is intended for development, not prevalence or held-out performance. `comment_sample.csv` retains the five original source fields, `candidate_makes`, `label_pool` (`reviewed_round_01` or `toyota_confirmed`), and `normalized_text_sha256`. `canonical_makes.csv` copies the exact vPIC make vocabulary recorded by the source sample manifest. Both are checksum-protected references.

`target_labels.csv` starts with one row per sampled comment. `target_index` is a positive per-comment integer used to attach additional human targets for comparisons. Each target has editable `target_resolution_status` (`resolved`, `ambiguous`, `unresolved`), canonical `Make`, `opinion_expressed` (`yes`, `no`, `unclear`), and notes. Pending/resolved-without-opinion rows are unfinished; ambiguous/unresolved targets carry no confirmed Make/opinion. These statuses describe proposed annotation targets; the later per-comment resolution status is not inferred by this preparation tool.

`aspect_labels.csv` starts with nine rows per target, keyed by `(comment_id, target_index, aspect)`. The aspect vocabulary is `reliability`, `maintenance_cost`, `price_value`, `fuel_economy_range`, `performance`, `comfort`, `safety`, `technology`, and `design`. Editable `aspect_present` is `yes`, `no`, or `unclear`; blank means unfinished. Only `yes` has `polarity` (`positive`, `negative`, `neutral`, `mixed`) and a verbatim source `evidence_quote`. A present reliability aspect also requires `reliability_claim_basis` (`firsthand_owner`, `secondhand`, `speculation`, `unclear`). Aspect labels require a resolved Make with opinion=yes. Targets with no opinion need no aspect labels and create no neutral sentiment records.

Label checks preserve source/context and export only valid resolved records: `labeled_targets.csv` has one `(comment_id, Make)` with human opinion status, and `labeled_aspects.csv` has one `(comment_id, Make, aspect)` with human polarity/evidence/basis. Multiple target indexes resolving to the same Make for one comment are rejected. Individual valid partial records can be exported while other comments/aspects remain unfinished; `annotation_progress.json` records completion and support, with training readiness false until independent review and a separate held-out split are established. These human annotations do not populate the planned model-development Parquet tables.

**Completed starter profile (2026-10-01):** `aspect_round_01` contains 100
distinct, normalized-text-deduplicated comments: 66 from reviewed-round
confirmed positives and 34 from Toyota confirmed positives. There are 100
target placeholders and 900 aspect rows. Frozen hashes and source fields
validate, with no relevance-blind ID/video/normalized-text overlap. All target
and aspect annotations remain blank, so there are zero exported human
target/aspect labels and 100 unfinished comments. These counts describe
development templates, not classified sentiment or a sentiment holdout.

## Planned YouTube make-level features (not in the current dataset)

The first YouTube sentiment pilot uses `Youtube_Comments.parquet` without changing its 12 source columns or the six-file model-development snapshot. All collected videos are car reviews. A title match supplies candidate vPIC `Make` context for sampling, not a video exclusion rule or proof that every comment concerns that make. The local Toyota pilot has 240 user-confirmed vehicle relevance labels (175 positive and 65 negative); no labels or scored make features have been added to the model-development snapshot. The [pipeline plan](YOUTUBE/docs/MAKE_SENTIMENT_PIPELINE_PLAN.md) proposes four new Parquet tables after validation; none exists yet.

| Planned table | Grain | Meaning |
|---|---|---|
| `youtube_comment_relevance.parquet` | One `comment_id` | Vehicle qualification decision, score, prefilter reason, later target-resolution status, and model/source versions. |
| `youtube_comment_targets.parquet` | One `(comment_id, Make)` | Confirmed canonical vPIC make target and opinion status. A relevant comment with unresolved make has no target row. Multiple makes are permitted for comparisons. |
| `youtube_comment_aspect_sentiment.parquet` | One `(comment_id, Make, aspect)` | One supported aspect opinion and its polarity, score, evidence, and reliability-claim basis when applicable. |
| `youtube_make_sentiment_features.parquet` | One `(Make, aspect, as_of_date, window)` | Time-bounded counts, shares, video-balanced sentiment, and support indicators. |

| Planned field | Grain / meaning |
|---|---|
| `comment_id`, `video_id` | Source identifiers preserving the comment-to-video link. Do not use spreadsheet row numbers as join keys. |
| `Make` | Canonical vPIC make being assessed. The pilot proposes it from the video title, then comment-level target labels determine whether a comment is about that make. No `ModelYear`, `Model`, or `Trim` is inferred from a make-only video. |
| `vehicle_relevant`, `vehicle_relevance_score`, `relevance_decision`, `comment_topic` | The pilot feedback file maps each sampled `comment_id` to `vehicle_relevant=yes/no`. The future classifier records a score and `accept`, `reject`, or `review_or_pass`. `vehicle_specific` is a confirmed positive relevance label without a confirmed target make or opinion; `not_vehicle_specific` is a confirmed negative. A review title alone is not proof of comment relevance. |
| `target_resolution_status`, `target_score`, `target_method`, `opinion_expressed` | `target_resolution_status` is per comment (`resolved`, `ambiguous`, or `unresolved`); the other fields are per `(comment_id, Make)` target. Questions and factual comments can be relevant without an opinion; they are not neutral sentiment. An unresolved comment has no make-target row. |
| `aspect`, `polarity`, `polarity_score`, `evidence_quote` | One car aspect and its positive, negative, neutral, or mixed opinion, with supporting text. Multiple aspect records may arise from one comment. An absent aspect has no record, not a neutral score. |
| `reliability_claim_basis` | Firsthand owner report, secondhand report, speculation, or unclear basis; never a measured failure rate. |
| `prefilter_reason` | Auditable reason for a proposed deterministic skip, such as empty or exact creator-focused text. A skip is not treated as a sentiment label. |
| `as_of_date`, `window`, `window_start`, `window_end` | Time boundary for a cumulative or trailing 12-month make aggregate. Only comments published by `as_of_date` and within the selected window contribute. |
| `comment_count`, `opinion_count`, `video_count`, `author_count_approx` | Distinct support counts for a make/aspect/window; `author_count_approx` uses display names rather than verified author IDs. These are not present in the current Parquet snapshot. |
| `positive_count`, `negative_count`, `neutral_count`, `mixed_count`, `net_sentiment`, `video_balanced_net` | Planned aspect-polarity counts; `net_sentiment = (positive_count - negative_count) / opinion_count` when support is sufficient. `video_balanced_net` averages per-video sentiment so a heavily commented video has less influence. |
| `support_suppressed`, `source_snapshot`, `model_version`, `label_schema_version` | Preserve low-support/null status and the exact data and model lineage for every derived output. |

Future make-level aggregates will have `(Make, aspect, as_of_date, window)` as their key, with aspect-specific support and sentiment measures. Missing or suppressed coverage is null, not a neutral opinion. If such aggregates are joined to a listing cohort, its `ModelYear`, `Make`, `Model`, and `Trim` must remain separate columns and the YouTube signal must remain labeled as **make-level**. Historical price prediction additionally requires a verified price-observation cutoff and comments published no later than that cutoff; the current listing dates do not establish one.

### Implemented experimental sentiment extension (2026-10-01)

These separate artifacts are not inputs to the current price model or additions
to the six model-development source tables. The 1,000-comment sample is drawn
from the enriched 10,000-comment relevance pilot; its population is
`enriched_pilot_engineering_only`. The expert reference contains 2,925 rows
across 585 reviews. Saved pretrained inference and video/make/wide feature tables
are complete for the 1,000- and 10,000-comment engineering samples under
`YOUTUBE/outputs/sentiment_inference_v1/`, `sentiment_inference_10k_v1/`,
`sentiment_features_v1/`, and `sentiment_features_10k_v1/`. They retain
`enriched_pilot_engineering_only` and `not_independently_validated` status and are
not consumed by the current price model.

| Artifact | Grain and fields |
| --- | --- |
| `sample.parquet` | One comment with original source fields and relevance score. `score_band`, cell counts, and `conditional_sampling_fraction` describe the seeded engineering sample; they are not source inclusion probabilities. |
| `youtube_experimental_predictions.parquet` | Comment/Make/attribution scope/aspect; original text, relevance score, `polarity`, `language_status`, preprocessing flags, normalized text hash, and `evidence_json` containing original clause offsets/text, topic/aspect/polarity scores, tone scores, target support, truncation and windowing flags. |
| `youtube_experimental_video_features.parquet` | Video/Make/attribution scope/aspect/as-of/window/policy; deduplicated polarity and coverage counts, raw and duplicate counts, and per-video `net_sentiment`. |
| `youtube_experimental_make_features.parquet` | Make/attribution scope/aspect/as-of/window/policy; video-balanced and comment-weighted net, across-video standard deviation, polarity shares, opinion/video support, unresolved/duplicate/language/truncation coverage, and `support_suppressed`. |
| `youtube_experimental_make_features_wide.parquet` | Make/attribution scope/as-of/window/policy; `yt_<aspect>_video_balanced_net`, comment-weighted net, opinion/video counts, suppression, unresolved/duplicate shares, and explicit `yt_<aspect>_missing` flags. |
| `dougscore_experimental_reference.parquet` | Review/aspect; imported raw review identity and scores, original mapped components, proposed mapping status, approved identity counts, snapshot target scope, and descriptive expert display index. |
| `expert_community_make_panel.parquet` | Community make-feature row plus the selective expert review-sample mean/count where canonical identity is approved. Populations remain explicitly different; no blended score or same-vehicle pairing is defined. |

`attribution_scope` is `explicit_comment`, `title_context`, or `unresolved`.
Title context cannot receive a clause explicitly attributed to another make.
Canonical makes come from the frozen relevance manifest with versioned explicit
aliases. Trim and model identity are not inferred by this make-level route.
`polarity` is positive, negative, neutral, mixed, no_opinion,
aspect_not_present, or unresolved. Unknown/unsupported languages and truncated
or bounded-window evidence stay unresolved; unknown language is not neutral.
Nine community aspects are retained; DougScore covers only design, performance,
comfort, technology, and price_value. NLI and tone outputs are uncalibrated.

Deduplicate normalized exact text within video/Make/scope/aspect for aggregation;
preserve all prediction rows and duplicate counts. Net is
`(positive_count-negative_count)/opinion_count`; its denominator includes resolved
neutral and mixed opinions. No-opinion, absent-aspect and unresolved rows are
excluded. Video-balanced net averages equally over videos with resolved opinions;
comment-weighted net pools those opinions. `video_count` counts opinion-bearing
videos; `observed_video_count` also includes videos with no resolved opinion.
Suppress make scores and polarity shares below 20 opinions or five videos while
retaining support counts. Missing and suppressed scores are null, never zero.
Across-video standard deviation is descriptive spread, not a confidence interval.

Dates parse the existing MM-DD-YYYY source representation and ISO dates/timestamps.
Missing/invalid dates are excluded and counted in feature manifests. As-of dates
include the complete UTC day; trailing-12-month windows exclude the previous
year's matching calendar cutoff and include the current cutoff. Monthly cutoffs
are month ends, with the final month capped at the requested day. Policies retain
all scores or require relevance >=0.20 or >=0.50; a sensitivity policy can exclude
exact normalized text repeated across videos. Relevance probabilities are never
statistical weights. Source-scored rows retain `development_protected` for training,
blind/reserved videos, protected IDs and protected normalized text; these rows are
retained in predictions and excluded from aggregates.

Community outputs carry `method=pretrained_experimental`,
`validation_status=not_independently_validated`, sampling population, and inference
manifest hash. Manifests freeze source/sample/code hashes, model commit revisions,
package versions, hypothesis text and inference thresholds. The expert display
index `(expert_score-5.5)/4.5` aligns 1..10 with -1..1 for display only. Expert Value
is flagged out of the first price experiment. Review model year is not a review
publication or score-availability date. Time evidence and approved identities
remain necessary before historical price joins; lowercase `vin` and separate
`ModelYear`, `Make`, `Model`, `Trim` requirements continue to apply.

### Raw-source GPU text feature extension (2026-10-05)

`YOUTUBE/text_pipeline.py` implements the raw-source semantic/topic and reduced
sentiment experiments. Derived files are separate `YOUTUBE/outputs/text_*`
artifacts: **no six-source Parquet or price-input field is replaced**. Canonical
`Make` is retained; make features are not cohort measurements.

| Artifact / grain | Derived data contract |
|---|---|
| Prepared shards; source comment | Original ID/video/title/text/publication/update/extraction strings, parsed `published_date`/`updated_date`, `normalized_text_sha256` preserving punctuation/emojis, `split` (`train`, `validation`, `test`, `duplicate_excluded`), `direct_price_mention`. |
| Score shards; comment/evidence span | Exact `evidence_start`/`evidence_end`/`evidence_text`; canonical `Make` or null; `attribution_scope` (`explicit_comment`, `title_context`, `unresolved`); language status, `windowed`, `multiple_targets`, `model_truncated`, `development_excluded`, `feature_eligible`. |
| Score-model fields | `content_label`, JSON content scores (independent entailment for teachers, class softmax for students), `tone_scores_json`, eight-topic cosine `topic_scores_json`, thresholded `topics`, 384-coordinate normalized `embedding`, `polarity` (positive/negative/neutral/unresolved). These are predictions, not human labels or calibrated confidence. |
| `youtube_text_video_features.parquet`; video/Make/scope/aspect/cutoff/window/method | Deduplicated eligible/mention/opinion counts, unresolved opinion count, raw/duplicate counts and per-video net. |
| `youtube_text_make_features.parquet`; Make/scope/aspect/cutoff/window/method | Equal-video `mention_share`, support/polarity counts, opinion/video counts, equal-video and comment-weighted net, negative/mixed shares, separate topic/sentiment suppression. |
| `youtube_text_make_features_wide.parquet`; Make/scope/cutoff/window/method | `yt_semantic_01`..`08`/`yt_semantic_missing`; eight `yt_<aspect>_mention_share`/`topic_missing` fields; four `yt_<aspect>_video_balanced_net`/`opinion_count`/`opinion_video_count`/`missing` blocks; comment/eligible/video counts, duplicate/unknown-language/truncation shares, global support suppression. |

Clause scores can yield comment/aspect `mixed` when separate resolved positive
and negative clauses concern that aspect. Incomplete evidence stays unresolved.
Facts/questions contribute to topics/semantics, never neutral sentiment. Topics
use eight existing non-price aspects; sentiment initially covers reliability,
maintenance cost, performance and comfort. Direct currency/numerical price
mentions are excluded. Cosine 0.30 and prediction 0.75/margin 0.15 thresholds are
experimental operating rules, not independently validated probabilities.

The content student learns only classes supplied by confident teachers. If no
confident `irrelevant` training examples exist, it learns vehicle opinion,
vehicle fact/question and creator discussion, and records `irrelevant` in
`unlearned_labels`. Teacher abstentions are not converted into irrelevant labels.
Unsupported/low-confidence and unusable short-language text remains unresolved.

Preparation applies publication **and update** cutoffs; invalid dates and protected
IDs/videos/text/reserved videos are excluded without using annotations. A 25%
development-video pool supplies rotating video quotas and video-blocked splits;
exact text cannot cross partitions. Student features exclude all development
videos/text. Canonical make tokens/aliases are masked for semantic encoding;
model names can still encode identity. Eight PCA coordinates fit training text
only. Manifests freeze source/shard/model/code hashes and runtime/model revisions.

Aggregation deduplicates normalized exact text within video/Make/scope, counts one
comment per topic/aspect, then averages within video and equally across videos.
Topics/semantics are null below 20 eligible comments/five videos; sentiment needs
20 resolved opinions/five opinion-bearing videos. Positive, negative, neutral
and mixed opinions remain in the net denominator. Topic shares use all eligible
targeted comments, including facts/questions. Cumulative/trailing-12-month windows
and optional cross-video-repeat exclusion are explicit policies. Missing or
suppressed scores are null, not zero. Population and
`validation_status=not_independently_validated` persist.

Later-fit students/reducers cannot generate earlier-cutoff features. The 2026
extraction does not verify historical availability or price-observation dates.
Static vectors can re-encode `Make`. Join one selected wide make/scope/date/window
row per listing after lowercase vPIC `vin` linkage, keeping `ModelYear`, `Make`,
`Model`, `Trim` separate. DougScore remains separate, with Value excluded from the
first price experiment. The selected make view is assembled into the retrospective
comparison dataset as defined above; independent automotive accuracy remains unmeasured.

**Executed extension profile:** 12,000 development comments (9,691 train, 1,153
validation, 1,143 test, 13 cross-partition duplicate exclusions). The separate
2,000-comment engineering sample exports 119 wide, 952 long and 4,928 video rows.
Six cumulative make/scope cells meet semantic/topic support; zero sentiment cells
meet the 20-opinion/five-video rule. These are sample coverage counts, not
full-source prevalence or accuracy. Expert coverage stays missing because no
review identities are approved. The later completed full-source run contains
1,287,155 scored comments and 2,872,698 clauses; aggregation exports 273 wide,
2,184 long, and 312,696 video rows across its attribution/window policies.
The selected 83-make comparison view and its coverage/sentiment limitations are
defined under [selected YouTube predictors](#selected-youtube-predictors-in-the-full-comparison-dataset).
Source Parquet schemas remain unchanged.

### Other research candidates without new manual labels

The [2026-10-05 research review](YOUTUBE/docs/NO_LABEL_TEXT_FEATURE_RESEARCH.md)
considers semantic/topic features alongside the existing experimental sentiment
fields. Optional `yt_<aspect>_mention_share` would average per-video fractions of
eligible targeted vehicle comments discussing an aspect, allowing facts/questions
without treating them as neutral opinions. Optional `yt_semantic_01` through
`yt_semantic_08` would be quantitative coordinates of a fixed reduced text
representation, not sentiment scores. The GPU extension implements mention
shares and eight training-fitted PCA coordinates. TF-IDF/SVD, BERTopic, JASen and
Snorkel remain optional comparators. The research review itself changed no source
column or price input.

Candidate aggregates retain canonical `Make`, attribution scope, as-of date,
window, feature method, support, and missingness. A static make vector is a
deterministic function of an already encoded `Make`; it does not supply observed
within-make or cohort-specific sentiment. The saved 10,000-comment cumulative
2026-10-01 no-relevance-cutoff view has supported explicit-attribution cells for
six makes and title-context cells for 42 makes under the current 20-opinion/
five-video rule. Those are enriched-pilot coverage diagnostics, not source
prevalence or independent target/sentiment validation.

## DougScore expert-opinion baseline

<!-- Legacy anchor retained for existing project links. -->
<a id="planned-dougscore-expert-review-data"></a>

This is a separately imported expert baseline source alongside the YouTube feature
tables. It is not included in the current six-table Parquet snapshot or
price-model inputs. The source CSV remains at
`C:/Users/danie/Downloads/DougScore - DougScore.csv`; its SHA-256 is
`5b93a406ad014c7b473b3dcef69360d69c93249bf89815416899b373030f1600`.
[EXPERT_REVIEWS/DOUGSCORE_AUDIT.ipynb](EXPERT_REVIEWS/DOUGSCORE_AUDIT.ipynb)
contains the executed profile. Integration workflow belongs in
[README.md](README.md#dougscore-expert-reference) and
[the expert plan](EXPERT_REVIEWS/DOUGSCORE_INTEGRATION_PLAN.md).

**Baseline role (2026-10-05):** Doug DeMuro's professional vehicle ratings are
the expert-opinion reference alongside YouTube audience features. They are
existing observed ratings, not synthetic comment labels or a professional
consensus. Preserve review-level ratings and compare only compatible aspects
and supported identity scopes. Make-level expert averages describe the selected
review sample, with one contribution per review and explicit review counts.

**Profile:** 585 vehicle records, 20 source columns, model years 1954–2023.
There are two header rows and one blank row before the records. All identity
and score fields are populated; all ten category scores are integer values
in 1–10, and all daily/weekend/overall totals reconcile. Overall scores range
25–74, Weekend 8–49, and Daily 8–40. There are zero exact duplicates and zero
duplicate year/make/raw-model keys after trimming and case folding. Seventeen
make fields have surrounding whitespace; 93 raw make labels become 84
trimmed/case-folded labels. These are not canonical manufacturer counts.
Review publication dates, score-version dates, separate trim, and usable
video URLs/IDs are unavailable in the CSV. Model years do not supply those
dates. vPIC/YouTube overlap and post-crosswalk uniqueness remain unmeasured.

### Proposed source-field normalization

Column positions are one-based and refer to this exact CSV layout. Raw values
remain available alongside any normalized values. These transformations are implemented in the separate expert snapshot; the
original six model-development tables remain unchanged.

| Source position / header | Proposed field | Classification and meaning |
| --- | --- | --- |
| 1 / Year (unnamed CSV header) | `model_year_raw`; parsed `ModelYear` | Raw text plus temporal discrete model year; never a review date. |
| 2 / Make (unnamed CSV header) | `make_raw` | Nominal manufacturer/brand text; canonical vPIC `Make` requires a reviewed crosswalk. |
| 3 / Model (unnamed CSV header) | `model_raw` | Compound model/variant text; canonical `Model` and `Trim` require separate evidence. |
| 4–8 / Styling, Acceleration, Handling, Fun Factor, Cool Factor | `styling`, `acceleration`, `handling`, `fun_factor`, `cool_factor` | Original ordered 1–10 expert category ratings; do not assume equal score intervals. |
| 9 / WEEKEND TOTAL | `weekend_total` | Sum of the five original Weekend categories; theoretical 5–50. |
| 10–14 / Features, Comfort, Quality, Practicality, Value | `features`, `comfort`, `quality`, `practicality`, `value` | Original ordered 1–10 expert category ratings. Quality and Practicality are compound judgments; Value is price-informed. |
| 15 / DAILY TOTAL | `daily_total` | Sum of the five original Daily categories; theoretical 5–50. |
| 16 / DOUGSCORE | `dougscore` | Sum of all ten categories; theoretical 10–100. |
| 17 / Video Link | `video_link_raw` | 584 populated time-formatted strings and one blank; not a verified URL, video ID, duration, or timestamp. |
| 18 / Filming Location | `filming_city_raw` | Filming locality text; one blank. Not vehicle market location. |
| 19 / unnamed | `filming_region_raw` | Filming region text; all populated. |
| 20 / Vehicle Country | `vehicle_country_raw` | Source country label; not a verified production location or sales market. |

### Imported expert snapshot profile (2026-10-01)

`EXPERT_REVIEWS/outputs/dougscore_v1/expert_review_scores.parquet` contains
585 unique snapshot review records. First three source columns mean Year,
Make, Model. Raw identity is `model_year_raw`, `make_raw`, `model_raw`; parsed
`ModelYear` is integer. `make_normalized` and `model_normalized` collapse whitespace
and casefold text for candidate search only; they are not canonical vPIC values.
Original score text is preserved as `<score_field>_raw`; parsed category and total
fields are integers. `video_id`, `review_url`, `review_published_at`,
`score_available_at`, and `score_version_evidence` are nullable strings, all null.
Video Link, filming locality, region, and Vehicle Country remain raw provenance
and are unused in current integration. `ingested_at` is UTC ingestion time.

Read-only recheck (2026-10-05): 585 unique review IDs, zero Weekend/Daily/overall
reconciliation errors, and 585 missing values in each of `video_id`, `review_url`,
`review_published_at`, and `score_available_at`. No expert/source records were
changed or added by this review.

### Original DougScore metrics

Definitions follow [Doug DeMuro's methodology](https://www.dougdemuro.com/dougscore).
Category fields are `Int64` ordered expert ratings on 1–10; totals are `Int64`
additive indices, not calibrated interval measures of quality. The methodology
and snapshot represent a selected enthusiast-oriented review sample.

| Saved field | Group | Meaning | Observed range | Missing |
| --- | --- | --- | --- | ---: |
| `styling` | Weekend | Appearance and styling judgment. | 2–10 | 0 |
| `acceleration` | Weekend | Acceleration rating based on 0–60 performance. | 1–10 | 0 |
| `handling` | Weekend | Handling and driving-control assessment. | 1–10 | 0 |
| `fun_factor` | Weekend | Driving enjoyment. | 1–10 | 0 |
| `cool_factor` | Weekend | Desirability and enthusiast/historical interest. | 1–10 | 0 |
| `features` | Daily | Equipment and feature assessment. | 1–10 | 0 |
| `comfort` | Daily | Ride, smoothness, and comfort/luxury judgment. | 1–10 | 0 |
| `quality` | Daily | Materials/build quality plus perceived reliability. | 1–10 | 0 |
| `practicality` | Daily | Cargo/usability assessment including fuel economy. | 1–10 | 0 |
| `value` | Daily | Worth relative to market price. | 1–10 | 0 |
| `weekend_total` | Summary | Sum of the five Weekend ratings; theoretical 5–50. | 8–49 | 0 |
| `daily_total` | Summary | Sum of the five Daily ratings, including Value; theoretical 5–50. | 8–40 | 0 |
| `dougscore` | Summary | Weekend plus Daily; theoretical 10–100. | 25–74 | 0 |

Each original field also has a retained `<field>_raw` string. The original
ten scores and totals remain the expert baseline; the five community-aspect
mappings are derived views. Score revisions may occur, and `ModelYear` is not
evidence of review publication or score availability.

`expert_aspect_scores.parquet` has 2,925 rows, exactly five per review. Its
`aspect_mapping_version` is `dougscore-community-v1-proposed`; `mapping_status`
is `proposed`. `expert_categories` and `expert_component_scores` are ordered
lists of original components. `expert_score` is a float on 1–10; performance is
an equal-weight mean of Acceleration, Handling, Fun Factor named
`equal_weight_performance_mean`. Other mappings retain one original category.
`partial_proxy` is true only for technology. No expert reliability, maintenance,
fuel-economy/range, or safety records are created. These are expert opinions,
not observed community polarity or validated sentiment labels.

`vehicle_video_crosswalk.csv` contains 585 pending snapshot records with blank
canonical identity, match evidence, reviewer and optional future video/date
fields. `outputs/dougscore_coverage_v1/vehicle_candidate_queue.csv` contains 863
unapproved candidates for 308 reviews; 277 reviews have none under the initial
text rule. Candidate matching uses normalized year/make and exact canonical
Model or a Model prefix followed by a space in source Model. No aliases or
Trim inference are applied. Repeated listing cohorts are deduplicated; 18,816
distinct four-column vPIC identity tuples include incomplete tuples. These are
candidate counts rather than verified review overlap. Canonical cohort fields
remain separate; no listing-frequency weighting or listing join is performed.
Both manifests record SHA-256 provenance. No approved expert-vehicle or paired
expert/community table exists yet. Field definitions below retain those planned
contracts until identity and sentiment validation are complete.

### Proposed derived tables and fields

| Planned table | Candidate key / grain |
| --- | --- |
| `expert_review_scores.parquet` | One `(source_snapshot_sha256, source_row_number)` review record; snapshot-only identity until durable review IDs are recovered. |
| `expert_vehicle_matches.parquet` | One `(review_record_id, match_candidate_index)` candidate cohort; canonical `ModelYear`, `Make`, `Model`, `Trim` stay separate and unsupported fields remain null. |
| `expert_aspect_scores.parquet` | One `(review_record_id, aspect, aspect_mapping_version)` supported mapped expert index. |
| `expert_community_comparison.parquet` | One `(review_record_id, comparison_target_id, target_scope, aspect, as_of_date, window)` comparison; target dimensions stay separate and the internal target ID is not a modeling predictor. |

| Planned field | Definition |
| --- | --- |
| `review_record_id`, `reviewer`, `source_snapshot_sha256`, `source_row_number`, `ingested_at` | Snapshot identity, named expert, input hash, logical CSV record locator, and ingestion timestamp. Ingestion time is not first score availability. |
| `video_id`, `review_url`, `review_published_at` | Recovered verified video identity/link/publication metadata; all unavailable in this CSV. |
| `score_version_at`, `score_available_at`, `score_time_evidence`, `source_version` | Documented score revision/availability time and its evidence. A video's publication date alone cannot establish the availability of this snapshot's score. |
| `ModelYear`, `Make`, `Model`, `Trim`, `match_candidate_index`, `match_level`, `match_status`, `match_evidence` | Reviewed vPIC identity candidate with explicit approved/pending/rejected status and `exact_cohort`, `year_make_model`, `make_only`, `ambiguous`, or `unmatched` level. Broader matches do not establish trim applicability. |
| `aspect`, `expert_category_components`, `expert_score_raw`, `expert_index_name`, `aspect_mapping_version` | Supported mapped aspect, original contributing categories, retained score/index, descriptive index name, and mapping version. Any mean of ordinal components is an explicit equal-spacing/index assumption. |
| `comparison_target_id`, `target_scope`, target dimensions | Internal reviewed target key plus `reviewed_variant`, `year_make_model`, or `make` scope and separate year/make/model/trim columns. Make-only evidence is not model-specific sentiment. |
| `community_net_sentiment`, `community_video_balanced_net`, `comment_count`, `opinion_count`, `video_count`, `support_suppressed`, uncertainty fields | Dated community measures and support from validated target/aspect records; inherit the YouTube feature definitions and retain null suppression. |
| `same_review_comment_count`, `other_video_comment_count`, `audience_overlap_status` | Distinct counts/status separating comments under the expert's review from other verified videos, without asserting independent opinions or verified author identities. |
| `as_of_date`, `window`, `window_start`, `window_end`, source/model/schema versions | Comparison cutoff, community window, and exact lineage; score availability must also be established for historical comparisons. |

The proposed mapping uses Styling for `design`, the three
Acceleration/Handling/Fun Factor components for a named `performance` index,
Comfort for `comfort`, Features as a partial `technology` proxy, and Value
for `price_value`. The original Quality, Practicality, and Cool Factor scores
remain separate expert measurements. There are no dedicated source scores
for `reliability`, `maintenance_cost`, `fuel_economy_range`, or `safety`.
Expert scores do not constitute human comment polarity labels. Missing expert
coverage and unsupported comparisons are null rather than zero/neutral.

### Proposed expert price-feature block (not implemented)

Use original non-Value categories as a separate expert predictor block only
where vehicle identity and score availability support the listing observation.
`value`, `daily_total`, and `dougscore` contain price-informed Value and are
excluded from the initial expert price baseline. `weekend_total` does not
contain Value. Optional derived summaries, not new original ratings, are:

| Proposed field | Definition | Theoretical range | Recomputed snapshot range |
| --- | --- | --- | --- |
| `expert_daily_without_value` | `daily_total - value`; four-category Daily index. | 4–40 | 4–35; field not materialized. |
| `expert_total_without_value` | `dougscore - value`; nine-category overall index. | 9–90 | 21–68; field not materialized. |

These fields have not been added to an expert Parquet or model input. Subtracting
Value does not establish historical availability or eliminate review selection
bias. Keep expert and community predictors separate and compare baseline,
expert-only, community-only, and combined additions on identical listing samples.
Expert/community rank agreement is descriptive consistency, not comment-label
accuracy; no new manual comment annotation is required for that comparison.

## Price-model input and derived-field contract

The price population includes both new and used **advertised listings**. `is_new` describes the listed vehicle, not evidence of a completed sale. The current saved listings span `listed_date` 2010-11-03 through 2020-09-13; listing date is not asserted to be the price-observation date. No current-market or realized-sale target is available.

Only positive, non-missing `price` values are modeled. Estimators are fit to the natural-log target `ln(price)`. The self-contained saved predictors exponentiate model outputs, and reported predictions and error metrics use USD.

The export removes rows with missing mileage and VINs whose review ratios exceed the specified cohort-relative rules: price/cohort median >2 or <=0.75, or mileage/cohort median >20. Model preparation repeats these exclusions for older snapshots. No absolute price or mileage cutoff is applied. Missing or nonpositive target prices remain ineligible for modeling.

The `base` feature set uses listing predictors from `used_cars` plus deterministic listing-derived fields. The `enriched` set includes those predictors and vPIC, complaint/recall, and safety features. Both use all listings, and `is_new` is a predictor whose errors are reported as a separate segment. Decoded `ModelYear`, `Make`, `Model`, and `Trim` remain separate in assembled frames and split/prediction receipts, but they are not predictors in `base`. `ModelYear` is numeric; the other three are categorical. Original decoded missing values remain missing in receipts. Linkage uses lowercase `vin`, with a one-to-one left join. The default unseen-family split groups normalized year/make/model, with all trims together. The optional known-family split groups by VIN so families may appear across partitions. Listing `year`, `make_name`, and `model_name` fill missing default split-key dimensions only; they do not overwrite decoded values. Completely unresolved identities share one explicit unknown block. Alias mismatches across listing/vPIC naming are not automatically resolved.

The following fields are derived in memory by the modeling loader, **not added to the six source Parquet tables**:

| Field | Meaning / transformation |
|---|---|
| `maximum_seating_numeric` | Numeric capacity extracted from `maximum_seating` text matching a number followed by `seat`/`seats`; other values become missing. |
| `log1p_mileage` | Natural log of one plus nonnegative mileage; negative/missing inputs become missing for this derived field only. The original mileage remains unchanged. |
| `option_navigation`, `option_leather`, `option_heated_seats`, `option_backup_camera`, `option_bluetooth`, `option_sunroof`, `option_remote_start`, `option_alloy_wheels`, `option_adaptive_cruise` | Indicators of exact quoted advertised option labels in `major_options`: Navigation System, Leather Seats, Heated Seats, Backup Camera, Bluetooth, Sunroof/Moonroof, Remote Start, Alloy Wheels, Adaptive Cruise Control. A valid empty list yields zero; missing/non-list text yields missing. Zero means not listed, not proof the equipment is absent. |
| `<numeric_feature>__missing` | One missingness indicator for every numeric model feature, computed before training-median filling. Non-finite numbers are treated as missing. |
| `cohort_complete` | Audit flag: all four decoded dimensions are non-missing after trimming whitespace. Not a predictor. |
| `split_identity_source` | Audit category: `vpic_year_make_model`, `listing_identity_fallback`, or `unresolved`. Not a predictor. |
| `split_group` | Internal deterministic split identifier saved in receipts: year/make/model family for `unseen_family`, or VIN for `known_family`. Not a cohort label or a predictor. Cohort dimensions remain separate. |
| `family_block` | Internal year/make/model grouping key used to report group support; not a cohort label or a predictor. |
| `mileage_review_flag` | Audit-only flag for mileage below zero or above 1,000,000. These thresholds trigger review, not automatic rejection. |
| `price_review_flag` | Audit-only flag for price below $500 or above $500,000; never a predictor because it uses the target. |

All three listing color columns apply the dictionary's missing-placeholder rules. Categorical missing values map to a dedicated missing token; categories absent from training map to an unknown token. Source category values receive a prefix to prevent collisions with reserved tokens. Numerical imputation uses training medians with zero plus a missing indicator for an entirely missing training column. These transformations never infer that missing equipment is absent.

Values selected by the stated cohort-ratio policy are excluded rather than winsorized. Review outputs preserve VIN, separate decoded dimensions, actual price, mileage, and candidate flags. Model evaluation includes `is_new`, complete/incomplete cohort, grouping-source, price-band, and extreme-flag segments. `year_make_metrics.csv` reports errors by decoded year, make, and year/make, with support counts and low-support flags. Model training and execution details belong in README.md.

## Sources and maintenance

The saved Parquet files and `manifest.json` are authoritative for schemas, row counts, examples, and conversion audit counts. Field classifications are modeling guidance; confirm NHTSA source definitions before treating numeric codes or status values as measurements.

Regenerate source profiles after another export with [EDA/profile_data_dictionary.py](EDA/profile_data_dictionary.py). It checks saved schemas and row counts against the manifest and preserves definitions and separate experiment sections. Add or revise field definitions before profiling a schema change. [README.md](README.md) describes the workflow; [EDA/DATA_CLEANING.py](EDA/DATA_CLEANING.py) defines the transformations.
