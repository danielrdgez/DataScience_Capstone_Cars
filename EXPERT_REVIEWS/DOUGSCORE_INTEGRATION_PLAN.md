# DougScore expert baseline integration

**Implemented route (October 9, 2026):** `price_features.py` now generates separate price-model expert features with automatic **year/make/model** matching, per-family category means, and an equal-model make fallback requiring five distinct models. Trim is ignored for DougScore linkage by user instruction; separate vPIC cohort columns remain intact. The `full` variant includes these features alongside existing YouTube inputs. See [README](../README.md#dougscore-expert-features) and the [selected predictor definitions](../DATA_DICTIONARY.md#selected-dougscore-predictors).

The first family artifact (`outputs/price_features_v2/`) has 302 families and 22 supported makes from 308 resolved reviews; 277 reviews remain unmatched. Automatic matches are not manually verified. Historical score dates remain unverified, Value-derived predictors are excluded, and no comment labels or blended scores are introduced.

## Earlier supervised comparison design

The design below records the earlier trim-specific/manual-review proposal. The implemented year/make/model price-feature route above supersedes its matching and price-integration deferrals. Video-linked, independently validated expert/community sentiment comparisons remain future work.

Started 2026-10-01 after user approval. This is an expert extension alongside the current
[YouTube sentiment plan](../YOUTUBE/docs/MAKE_SENTIMENT_PIPELINE_PLAN.md). The source
audit and snapshot ingestion are complete; unapproved vPIC candidates and
expert-aspect baselines are generated. Verified vehicle matching and paired
expert/community scoring remain pending. The six model-development Parquet sources and the price
model do not currently consume DougScore.

## Recommended use

Maintain an expert-only baseline with the ten original category scores,
Daily total, Weekend total, and overall DougScore. Compare supported expert
categories with community aspect sentiment, showing both scores and their
coverage. These are different measurements: a review rates a vehicle, while
comment sentiment measures expressed opinions. Agreement is a useful external
comparison, not proof that a sentiment label is correct.

For an everyday-car recommendation, show Daily and its component scores first;
show Weekend separately for enthusiast preferences. Keep the original overall
score available, but do not make it the default definition of a good used car.
This is a proposed product choice, not a validated ranking model.

## Source profile

Input: `C:/Users/danie/Downloads/DougScore - DougScore.csv`.

SHA-256: `5b93a406ad014c7b473b3dcef69360d69c93249bf89815416899b373030f1600`.
The [executed audit notebook](DOUGSCORE_AUDIT.ipynb) contains the exact checks
and saved outputs. [The machine-readable profile](outputs/dougscore_profile.json)
records execution time, missingness, ranges, and source hash. The input remains
unchanged in Downloads; the exact hashed snapshot is now imported into
`outputs/dougscore_v1/` without altering the audit notebook.

| Check | Observed result | Consequence |
| --- | --- | --- |
| Layout | 588 logical CSV rows: two header rows, one blank row, 585 vehicle rows; 20 columns. | Use an explicit parser; the first line is not a conventional field header. |
| Years | 1954–2023; 187 before 2010, 286 during 2010–2020, 112 after 2020. | Model year is not review publication time or evidence of current coverage. |
| Required values | Year, make, raw model/variant, all ten scores, and all three totals are populated for every review. | The expert baseline has complete ratings for this snapshot. |
| Score checks | All ten scores are integers in 1–10; every five-score subtotal and overall total reconciles. | No score correction is needed in this snapshot. |
| Duplicates | Zero exact duplicates and zero duplicate year/make/raw-model keys after trimming and case folding. | Recheck uniqueness after aliases and model/trim resolution; variants can then share a broader key. |
| Make names | 93 raw labels, 84 after trimming/case folding; 17 rows have make whitespace. | These are text-label counts, not verified counts of manufacturers. Alias review remains necessary. |
| Variant identity | No separate Trim column; examples include Toyota `4Runner TRD Pro`, `Supra 3.0`, and `Land Cruiser V6 TT`. | Preserve full raw names and curate a crosswalk to vPIC rather than treating the raw field as canonical Model. |
| Video references | 584 time-formatted strings in `Video Link`, one blank, zero HTTP(S) URLs. | The CSV does not provide usable review video links or IDs. The time strings' meaning is unverified. |
| Dates | No review publication or score-version timestamp columns. | This file cannot establish historical score availability. |

The initial vPIC candidate scan found 308 reviews with text candidates and
277 without, producing 863 candidate rows. Approved vehicle overlap and trim
coverage remain unmeasured; video linkage is deferred by user instruction.
The first three source columns are Year, Make, and Model. The full Model text
is retained in `model_raw`, including any embedded variant information.

## Category mapping

The official [DougScore methodology](https://www.dougdemuro.com/dougscore)
describes ten 1–10 ratings, divided equally into Daily and Weekend. It favors
enthusiast appeal; Quality combines materials with perceived reliability,
Practicality includes cargo/usefulness and fuel economy, and Value considers
market price. Scores can be revised. Those definitions inform the following
**proposed mapping**, which still needs validation against labeled comments.

| DougScore field | Current community aspect | Proposed treatment |
| --- | --- | --- |
| Styling | `design` | Compare styling opinions; broader design opinions may concern different features. |
| Acceleration, Handling, Fun Factor | `performance` | Keep all three expert components. An optional mean is an explicitly equal-weighted performance index, not an original DougScore category. |
| Comfort | `comfort` | Compare ride, noise, and comfort opinions with supporting text. |
| Features | `technology` | Partial proxy: equipment breadth differs from technology usability or dependability. |
| Value | `price_value` | Compare value opinions; retain the applicable price context when available. |
| Quality | None directly | Keep as `expert_quality`; it cannot isolate `reliability` from materials and build quality. |
| Practicality | None directly | Keep as `expert_practicality`; it cannot isolate `fuel_economy_range`. |
| Cool Factor | None directly | Keep as `expert_cool_factor` for desirability/enthusiast preference. |

The current nine-aspect comment schema stays unchanged. Dedicated expert
baselines for `reliability`, `maintenance_cost`, `fuel_economy_range`, and
`safety` are unavailable in this source. Do not manufacture them from Quality,
Practicality, Handling, or overall DougScore. Expanding the community schema to
practicality/build quality/desirability can be considered later with new label
definitions and examples; it is not necessary for initial integration.

## Vehicle and video linkage

1. Preserve all source text and score fields. Attach a snapshot hash and CSV
   logical row locator. Until usable review IDs are recovered, identify a
   snapshot record by `(source_snapshot_sha256, source_row_number)`; this is
   not a durable identity across updated spreadsheets.
2. **Deferred by the user for initial integration:** recover the source-sheet hyperlinks or manually verify review videos. Store
   `video_id`, review URL, publication date, and evidence for the vehicle
   identity. A time-formatted CSV value cannot establish a video match.
3. Curate a reviewed crosswalk with separate `ModelYear`, `Make`, `Model`, and
   `Trim`. Fix whitespace and propose aliases without altering raw fields.
   Entries such as `Mecedes-AMG`, `Mercedes`, `Mercedes - Benz`, Maybach, and
   Range Rover require vehicle/year-specific review; do not apply an unchecked
   universal replacement. Exotic or non-listing vehicles may remain unmatched.
4. Store match level and evidence: `exact_cohort`, `year_make_model`,
   `make_only`, `ambiguous`, or `unmatched`. Keep Trim null when unsupported.
   One review can have multiple candidate cohorts, but unresolved candidates
   are not approved matches. Do not copy one trim's performance score across
   every trim of the same model.
5. Use the verified video ID to identify comments under the reviewed video.
   Then verify that each comment actually discusses that reviewed vehicle;
   comments about another vehicle, the presenter, or a comparison are not
   opinions about the review subject.
6. Extend comment target labels to year/model/trim only for the paired-review
   experiment, with explicit evidence and unresolved states. The existing
   reusable sentiment features remain **make-level** until this extension is
   independently checked. A make-level score cannot validate a specific
   reviewed variant.

The current vPIC snapshot describes decoded listing vehicles, rather than all
vehicles ever produced. Lack of a match can reflect population coverage.
Report source reviews, approved matches, ambiguous/unmatched counts, and
distinct matched cohorts without multiplying reviews by listing frequency.
Any listing price/mileage comparison joins listings to vPIC using lowercase
`vin`; expert scores enter only after the vehicle crosswalk is established.

## Scoring and evaluation template

Use this display/analysis schema for each supported pairing:

| Vehicle dimensions | Aspect | Expert category/index and raw score | Community sentiment and support | Match and time evidence |
| --- | --- | --- | --- | --- |
| Separate `ModelYear`, `Make`, `Model`, `Trim` | Supported mapped aspect | Reviewer, original 1–10 category values, optional named derived index | Positive/negative/neutral/mixed counts, net sentiment, distinct comments/videos, suppression and uncertainty | Review/video IDs, approved match level, window, source versions |

Keep unknown, unmatched, and low-support results null. Retain both sources'
scales and meanings. Optional display rescaling of a category with `(score-1)/9`
or sentiment with `(net_sentiment+1)/2` only aligns endpoints; it does not
calibrate the measurements, establish equal spacing of ordinal ratings, or
justify averaging them.

Evaluate in this order:

1. **Expert baseline:** original daily/weekend/category scores on approved
   reviewed vehicles. Report coverage and vehicle mix rather than a simple
   brand leaderboard over this selective sample.
2. **Community baseline:** validated aspect sentiment with the existing
   support and date rules. Model-specific comparisons require validated
   model-specific attribution; brand-only analysis must be labeled separately.
3. **Paired comparison:** inspect aspect rankings, high-disagreement examples,
   evidence quotes, and coverage on matched vehicles. The same-review audience
   may be influenced by the expert's verdict. Compare with comments from other
   verified videos where available; mark same-review and other-video evidence
   separately. One Doug review alone does not pass the current proposed
   five-video community support rule, and that rule is not automatically
   relaxed for this experiment.
4. **Optional combined system:** choose a defined goal and validate weights
   using held-out vehicles/review videos before publishing a combined score.
   Keep all comments from the same video out of opposite training/test splits;
   avoid splitting repeated variants of the same vehicle family across folds
   when evaluating transfer to unseen vehicles.

Expert scores are external comparison measurements, not comment sentiment
gold labels. Human aspect/polarity annotations and an independent sentiment
holdout are still required. Differences may reflect review selection, variant,
time, price context, or commenter priorities, rather than a classifier error.

Historical price modeling has an additional requirement: verify both the
price-observation cutoff and the availability/version of each expert score.
Retrieving a video publication date does not prove that today's spreadsheet
score existed then. Value is price-informed and therefore needs particular
target-leakage review; omit it from the first price-feature experiment until
its provenance and timing are defensible. Initial use should be descriptive
expert/community analysis or recommendations for the supplied snapshot.

## Planned tables and execution order

Field meanings and candidate keys are recorded in the
[data dictionary](../DATA_DICTIONARY.md#selected-dougscore-predictors).

| Planned artifact | Grain | Role |
| --- | --- | --- |
| `expert_review_scores.parquet` | One snapshot review record | Raw identity, original ten scores and totals, provenance, recovered video/date metadata. |
| `expert_vehicle_matches.parquet` | One snapshot review / candidate cohort pair | Separate four cohort dimensions, match level/status/evidence; ambiguous candidates are retained for review. |
| `expert_aspect_scores.parquet` | One snapshot review / supported mapped aspect | Mapping version, contributing original categories, named index and score. |
| `expert_community_comparison.parquet` | One approved review / target scope / aspect / time window | Paired values, provenance, support, coverage, and same-review audience status. |

`expert_review_scores.parquet` and `expert_aspect_scores.parquet` are generated;
`expert_vehicle_matches.parquet` and `expert_community_comparison.parquet` remain
planned. A blank `vehicle_video_crosswalk.csv` and a separate unapproved
`vehicle_candidate_queue.csv` start identity review. The existing sentiment fit,
qualification audit, and aspect-labeling commands remain the next runnable
steps. Build the expert integration alongside them:

1. Import this exact hashed snapshot with explicit headers; retain the original
   file and run the score/identity checks for each new snapshot.
2. Prepare a blank vehicle crosswalk and measure vPIC candidate coverage.
   Completed; manually review candidates and aliases next. Video linkage,
   filming location, and Vehicle Country are outside the initial focus.
   Do not train or assign inferred matches.
3. Choose a small supported matched-vehicle subset, label its comment targets
   and aspects, and validate the attribution and sentiment models.
4. Produce the expert-only and community-only baseline tables plus a paired
   comparison, preserving Daily/Weekend views and missing coverage.
5. Consider new aspects, blending, or price-model features only after that
   comparison and the applicable validation gates.

The audit notebook remains the original read-only source audit. Use
`prepare_dougscore.py import-snapshot` for validated ingestion and
`prepare_dougscore.py coverage` for unapproved vPIC candidates; commands and
current workflow are in README. Import manifests bind source and output hashes;
coverage reports bind the import manifest and vPIC hash. New runs require fresh
output directories and never overwrite annotations. No combined-score,
verified-match publication, or price-feature command is implemented yet.

Completed artifacts:

- `outputs/dougscore_v1/`: 585 reviews, 2,925 proposed expert-aspect records,
  585 pending crosswalk rows, and an import manifest.
- `outputs/dougscore_coverage_v1/`: 863 candidate rows and a coverage report.

Continue with manual identity review before generating approved expert vehicle
baselines. The unchanged nine community aspects still require human labels,
model validation and independent support. The proposed 20-opinion/five-video
floor remains in force for paired comparisons; unsupported cells remain null.
