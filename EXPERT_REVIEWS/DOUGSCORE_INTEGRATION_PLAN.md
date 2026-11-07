# DougScore expert baseline integration

`price_features.py` builds year/make/model expert features with an equal-model make fallback; trim is not a match key. The [dictionary](../DATA_DICTIONARY.md#selected-dougscore-predictors) owns field definitions, support thresholds, matching rules, and coverage. The [README](../README.md#dougscore-expert-features) owns price-export commands.

## Source profile

Source snapshot: `outputs/dougscore_v1/`; original input SHA-256 below.

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

## Source import and candidate coverage

Use fresh directories; these commands import a validated snapshot and propose vPIC candidates without approving matches:

```powershell
.\.venv\Scripts\python.exe EXPERT_REVIEWS/prepare_dougscore.py import-snapshot --source "<DougScore CSV>" --output-dir EXPERT_REVIEWS/outputs/dougscore_next
.\.venv\Scripts\python.exe EXPERT_REVIEWS/prepare_dougscore.py coverage --snapshot-dir EXPERT_REVIEWS/outputs/dougscore_next --output-dir EXPERT_REVIEWS/outputs/dougscore_coverage_next
```

`outputs/dougscore_v1/` contains 585 reviews, 2,925 mapped aspect records, and 585 pending crosswalk rows. `outputs/dougscore_coverage_v1/` contains 863 candidate rows: 308 reviews have candidates and 277 do not. Import/coverage manifests bind source and artifact hashes. Candidates are distinct from the automatic matches used by the separate price-feature export.

## Expert/community comparison

This optional paired-review study remains unimplemented. It requires verified review/video identity and supported comment targets beyond the current make-only features. Keep original category ratings and Daily/Weekend totals separate from community polarity and coverage.

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

For a paired study:

1. Preserve raw names/scores and `(source_snapshot_sha256, source_row_number)` provenance. Row locators are snapshot-specific.
2. Verify separate decoded cohort fields and recover usable video IDs, review dates, and score-version availability. Ambiguous candidates stay unresolved.
3. Label comment targets and aspects on a small supported subset; reserve fresh review videos for independent evaluation. Distinguish same-review audiences from other videos.
4. Compare compatible aspect ranks, distributions, and disagreements with support counts. Keep unmatched/low-support values null. Expert ratings are comparison measurements, not sentiment gold labels.

Optional display rescaling only aligns endpoints; it does not calibrate ordinal ratings to sentiment or justify blending them. Daily and overall totals include price-informed Value. The selected price model excludes Value-derived totals, but historical score availability remains unverified.

Planned comparison artifacts are `expert_vehicle_matches.parquet` (review/candidate-cohort grain) and `expert_community_comparison.parquet` (approved review/target/aspect/window grain). Existing `expert_review_scores.parquet` and `expert_aspect_scores.parquet` preserve source ratings and proposed mappings. Publication of a combined expert/community score needs a defined objective and held-out validation.
