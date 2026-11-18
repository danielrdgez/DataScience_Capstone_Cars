# DataScience Capstone Cars

Predict historical advertised vehicle prices and explore similar vehicles using listing attributes, NHTSA data, YouTube opinions, and DougScore expert reviews. Saved price and discovery artifacts support a local Streamlit recommendation app.

## Project files and sources

| Path | Purpose |
| --- | --- |
| `DATA/` | Listing CSV, vPIC backup, SQLite database, and six-table Parquet snapshot in `CAR_DATA_FINAL/`. |
| `NHTSA/`, `YOUTUBE/`, `EXPERT_REVIEWS/` | Source collection, text processing, and expert-feature generation. |
| `EDA/` | Cleaning, cohort review, export, and exploratory analysis. |
| `MODELS/` | Price training, diagnostics, vehicle profiles, and similarity research. |
| `app.py`, `app_pages/`, `car_app/` | Local Streamlit dashboard and deterministic recommendation search. |
| `MODEL_OUTPUT/`, `MODEL_OUTPUT.ipynb` | Versioned model artifacts, evaluation receipts, and results review. |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Selected dataset schemas, profiles, transformations, and predictor definitions. |

| Source | Input or collection |
| --- | --- |
| [Kaggle US Used Cars](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset) | `DATA/used_cars_data.csv`; includes new and used listings. |
| [NHTSA vPIC](https://vpic.nhtsa.dot.gov/Downloads) | `DATA/vPICList_lite_2026_09.bak`, restored as SQL Server database `vPICList_Lite`. |
| [NHTSA APIs](https://api.nhtsa.gov/) | Recalls, complaints, and safety ratings. |
| [YouTube Data API](https://developers.google.com/youtube/v3) | Video metadata and top-level comments. |
| [DougScore](https://www.dougdemuro.com/dougscore) | Imported expert-review snapshot under `EXPERT_REVIEWS/outputs/`. |

## Workflow

<!-- mermaid:id=car_workflow -->
```mermaid
flowchart LR
  S["Listings, NHTSA, YouTube"] --> P["Cleaned Parquets"]
  P --> F["Selected predictor blocks"]
  E["DougScore"] --> F
  F --> M["Price fits and held-out evaluation"]
  P --> V["Specification profiles and dated evidence"]
  M --> C["Verified saved checkpoints"]
  C --> G["Model diagnostics"]
  F --> O["Full CatBoost: ten OOF folds and final fit"]
  O --> B["Validate and activate price bundle"]
  B --> Q["Optional OOF pricing policy"]
  Q --> A
  C --> T["Best tree and linear by validation MAE"]
  T --> A["Dashboard and recommendation"]
  V --> A
```

## Setup and data preparation

Run these PowerShell commands from the repository root. Git ignores large data, model outputs, virtual environments, credentials, and browser receipts. Keep run logs and manifests. Python caches and `.playwright-cli/` are disposable.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-app.txt
```

Collection needs the listing CSV, SQL Server 2019+, and the Microsoft ODBC Driver. Set the paths and instance in [restore_vpic_database.sql](NHTSA/restore_vpic_database.sql), then restore the vPIC backup before running collection.

```powershell
.\.venv\Scripts\python.exe NHTSA/build_car_data_db.py --load-only
.\.venv\Scripts\python.exe NHTSA/build_car_data_db.py --only all
.\.venv\Scripts\python.exe YOUTUBE/YOUTUBE_COMMENTS_API.py
```

`--load-only` replaces listing rows. Other collection resumes saved work. Pass `--only vpic`, `--only recalls`, `--only complaints`, or `--only safety` to collect one NHTSA source. The local vPIC decoder saves values and variable metadata to SQLite. YouTube reads `YOUTUBE_API_KEY` or `GOOGLE_API_KEY` from the environment or ignored root `.env` and consumes quota.

Export and profile the selected six tables:

```powershell
.\.venv\Scripts\python.exe EDA/DATA_CLEANING.py
.\.venv\Scripts\python.exe EDA/profile_data_dictionary.py
```

Export vPIC with listings. The exporter applies the eligibility rules, validates the replacement Parquets, and writes `manifest.json` while reading SQLite in read-only mode. Missing decodes get null source fields. Duplicate retained decode keys stop publication.

Use `.\.venv\Scripts\python.exe EDA/DATA_CLEANING.py --align-vpic-only` to realign vPIC in an existing snapshot. It saves a rollback copy and logs match counts in the manifest. Use `MODELS/cohort_price_mileage_review.py` for cohort outliers and `MODELS/price_data_diagnostics.py` for source integrity and event timing. The profiler records which tables it refreshed in the [data dictionary](DATA_DICTIONARY.md#data-rules).

```mermaid
flowchart LR
  L[Retained listing VINs] --> A[Align vPIC on lowercase vin]
  V[Source vPIC records] --> A
  A --> P[One row per listing: decoded fields or nulls]
  P --> C[Validate unique keys, equal populations, and manifest]
```

`EDA/price_mileage_plotly.py` writes full-population histograms and boxplots of `ln(1 + price)` and `ln(1 + mileage)`. HTML, PNG, and summary receipts for the presentation slide go to `outputs/slide13_update/charts/`. The log transform compresses the right tails and retains zeros. PNG export needs Kaleido and Chrome.

```mermaid
flowchart LR
  P[Cleaned listing Parquet] --> L[Nonnegative finite values: ln of 1 plus x]
  L --> C[Plotly distributions and summary receipt]
  C --> S[Presentation price and mileage slide]
```

## Text and expert features

### YouTube GPU text features

The selected export, `YOUTUBE/outputs/sentiment_argmax_20261027T155951/price/`, contains 75 candidate fields for 83 makes. Price models use 40. See the runbook for setup, training, scoring, and recovery, and the dictionary for exact fields and coverage. [GPU runbook](YOUTUBE/docs/GPU_TEXT_RUNBOOK.md) · [Selected YouTube predictors](DATA_DICTIONARY.md#selected-youtube-predictors)

<a id="previous-youtube-sentiment-benchmark-workflow"></a>
Earlier experiments remain available through the [legacy sentiment plan](YOUTUBE/docs/EXPERIMENTAL_SENTIMENT_FEATURE_PLAN.md), [relevance runbook](YOUTUBE/docs/RELEVANCE_ROUND_RUNBOOK.md), and [research review](YOUTUBE/docs/NO_LABEL_TEXT_FEATURE_RESEARCH.md).

### DougScore expert features

[Expert integration](EXPERT_REVIEWS/DOUGSCORE_INTEGRATION_PLAN.md) documents source import and year/make/model matching with a supported make fallback. Build the default export only if absent:

```powershell
.\.venv\Scripts\python.exe -m EXPERT_REVIEWS.price_features --output-dir EXPERT_REVIEWS/outputs/price_features_v2
```

Give later exports a fresh directory and pass it to price training with `--expert-feature-dir`. Matching rules and the eight ratings and two source fields are defined in the dictionary. [Selected DougScore predictors](DATA_DICTIONARY.md#selected-dougscore-predictors)

## Listing-price models

`MODELS/Price_ML_Models.py` compares CatBoost, LightGBM, XGBoost, Random Forest, multiple linear regression, Lasso, Ridge, and Elastic Net with a training-only median baseline. PCR is opt-in for `full`.

| Variant | Inputs | Raw predictors |
| --- | --- | ---: |
| `base` | Listings and deterministic derived fields | 32 |
| `enhanced` | Base plus vPIC specifications | 79 |
| `enhanced_safety` (`enriched`) | Enhanced plus structured NHTSA aggregates | 98 |
| `full_youtube`, `full_youtube_sentiment` | Enhanced safety plus YouTube | 138 |
| `enhanced_safety_expert` | Enhanced safety plus expert fields | 108 |
| `enhanced_safety_expert_coverage` | Enhanced safety plus expert coverage | 100 |
| `full_youtube_expert_coverage` | Full YouTube plus expert coverage | 140 |
| `full` | Enhanced safety, YouTube, and expert fields | 148 |

Counts exclude generated missingness indicators. The default split assigns distinct VINs to fit, validation, and test at about 60/20/20 within decoded year/make/model strata with at least 25 eligible VINs. Pool trims for splitting. Use `unseen_family` to test transfer to held-out families. Fit preprocessing on training rows and select settings on validation. Refit on fit plus validation before predicting the test partition. Models fit `ln(price)` and predict USD. Compare runs with matching populations, sources, partitions, and protocols.

```powershell
.\.venv\Scripts\python.exe -m MODELS.Price_ML_Models `
  --model-types base enhanced enhanced_safety full `
  --models catboost lightgbm xgboost random_forest multiple_linear_regression lasso ridge elastic_net pcr `
  --sample-size 0 --tuning-rows 200000 --max-trees 0 --full-validation
```

`--max-trees 0` expands boosting budgets until early stopping and grows forests to a validation plateau. `--full-validation` evaluates all validation VINs after sample tuning; `--validation-only` avoids test exposure. Device probes and CPU fallbacks are recorded. Completed estimators survive later failures; interrupted ensembles do not resume automatically.

CPU LightGBM uses native training and validated shards of at most 100 trees to avoid Windows whole-model serialization crashes. It retains the L1 objective, one CPU thread, column-wise histograms, a 512 MiB cache, and 100-tree progress receipts. For recovery, run the matching training settings with `--models lightgbm --device cpu --full-validation`; `--lightgbm-selection-run MODEL_OUTPUT/full/20261105T014105524672Z-full` reuses a compatible saved selection. Recovery writes a new run with `lightgbm_serialization=bounded_tree_shards_v1`; progress receipts cannot resume a partially fitted ensemble.

Use the printed suite directory for compatible-run comparison and deferred explanations:

```powershell
.\.venv\Scripts\python.exe -m MODELS.Price_ML_Models --compare-suite '<printed-suite-directory>'
.\.venv\Scripts\python.exe -m MODELS.Price_ML_Models --explain-suite '<printed-suite-directory>'
```

Reports retain USD errors, R², WAPE, bias, block-bootstrap intervals, subgroup errors, and source/split/predictor receipts. SHAP explains exact saved predictors on log-price; linear coefficients and forest impurity retain their own scales. Source/schema changes require compatible retraining.

### Recommendation pricing

The dashboard policy `best_tree_linear_v1` takes the completed tree and linear checkpoints with the lowest validation MAE in `price_run_selection`, including compatible Full continuations. It currently selects Full CatBoost and Full multiple linear regression. The AI receives separate estimates and verified explanations for each checkpoint. Only saved test predictions support model-relative value ranking. Supplemental inference can include fitting VINs and is labeled accordingly.

The app checks source, schema, package versions, and frozen inference replay. The receipt for the corrected snapshot compares all retained listing and vPIC values with the original files, checks that removed VINs were outside the model split, and replays frozen inputs and predictions.

```powershell
.\.venv-youtube\Scripts\python.exe -m streamlit run app.py
```

#### Optional Full CatBoost OOF workflow

`MODELS.price_oof` uses the existing Full inputs and eligible population. Ten deterministic VIN folds rotate within decoded year/make/model strata; Trim stays separate and pools only for splitting. Every historical prediction excludes its VIN from fitting, preprocessing and inner parameter selection. Each fold selects settings on its own training portion with an inner 80/20 split, then refits on the other nine outer folds. A separate final selection/refit uses all eligible rows for new-vehicle inference. Training is sequential and does not activate itself.

By default, each of the ten folds and the final workflow uses 200,000-row tuning samples with cohort coverage. The tree budget expands until early stopping on the log-price objective. Tuning selects by USD validation MAE. The workflow then measures the selected settings on full inner validation before refitting. Cross-validation uses all eligible rows, leaving no separate final test partition.

Run from the repository root in the dashboard environment:

```powershell
$OofRun = "MODEL_OUTPUT/recommendation_price/" + (Get-Date -Format "yyyyMMddTHHmmss") + "-full-catboost-oof"
.\.venv-youtube\Scripts\python.exe -m MODELS.price_oof train `
  --output "$OofRun" --folds 10 --tuning-rows 200000 --max-trees 0 --device auto
if ($LASTEXITCODE -ne 0) { throw 'OOF training failed; retain this run directory for resume.' }

.\.venv-youtube\Scripts\python.exe -m MODELS.price_oof validate --run "$OofRun"
if ($LASTEXITCODE -ne 0) { throw 'OOF validation failed.' }

.\.venv-youtube\Scripts\python.exe -m car_app.model_registry --activate-recommendation "$OofRun"
if ($LASTEXITCODE -ne 0) { throw 'Recommendation activation failed.' }

.\.venv-youtube\Scripts\python.exe -m streamlit run app.py
```

Save the printed run path. To resume after interruption, set `$OofRun` to that same path and run:

```powershell
.\.venv-youtube\Scripts\python.exe -m MODELS.price_oof train --output "$OofRun" --resume
```

Resume reuses the saved configuration and completed folds only when source, code, package and device identities match; incomplete fits restart. Do not create a new timestamp when resuming. `--device gpu` requires a successful GPU probe; `auto` records a CPU fallback when needed. Sampled runs, finite tree caps and fewer than ten folds are development bundles and cannot activate for recommendations.

Artifacts include `folds.parquet`, `oof_predictions.parquet`, pooled/per-fold `metrics.json`, subgroup CSVs, `fold_00/` through `fold_09/`, `final/predictor.joblib`, and hash-bound manifests. Validation checks coverage, disjoint selection roles, prediction provenance, metrics and frozen raw/prepared/USD replay for each predictor. A changed inference implementation needs successful revalidation and reactivation; source/schema/package changes require retraining. Default bundles contain no SHAP explanations.

Explicit OOF activation replaces the current tree/linear policy with the activated Full CatBoost bundle. Training VINs use saved OOF prices; an absent OOF price never falls back to the all-data predictor. Other catalog VINs may receive clearly labeled supplemental inference. Historical value ranking uses OOF predictions only. When the OOF policy is selected without a valid activated run, its pricing is unavailable while observed-price filtering remains usable. Dashboard evaluation checkpoints remain separate.

Score genuinely new VINs from listing Parquet rows without a price label:

```powershell
.\.venv-youtube\Scripts\python.exe -m MODELS.price_oof predict `
  --run "$OofRun" --input "DATA/new_listings.parquet" --output "MODEL_OUTPUT/new_listing_prices.parquet"
```

Supply lowercase `vin` and observed listing fields. Enrichment reads saved vPIC, YouTube, and expert data and passes missing inputs to fitted preprocessing. The output reports missing-feature counts and historical-scope warnings. Use the OOF bank for training VINs and choose a fresh output path. Prediction uses the saved decodes and historical price scope.

OOF metrics are conditional on the previously selected feature set and existing price-informed dataset exclusions. They measure predictions for unseen VINs within represented vehicle strata, not unseen families or independently validated future prices. Later YouTube/expert snapshots remain retrospective context; the final all-data model has no separate test score.

### Diagnostics and PCA

Use the project `.venv` Jupyter kernel. Saved outputs retain their original schema and run identities.

| Notebook | Purpose |
| --- | --- |
| [EDA.ipynb](EDA/EDA.ipynb) | Source exploration and cohort summaries |
| [Full_Feature_Multicollinearity.ipynb](MODELS/Full_Feature_Multicollinearity.ipynb) | Fit-only associations, missingness, VIF, and rank checks |
| [PCA.ipynb](MODELS/PCA.ipynb) | Fit-only numerical scaling, spectrum, and loadings |
| [Price_Data_Diagnostics.ipynb](MODELS/Price_Data_Diagnostics.ipynb) | Completed checkpoints, receipt-reconciled errors, baselines, and explanations |

PCR uses 31 unwhitened numerical components; flags, missing indicators, and encoded categories bypass PCA. Historical PCA and multicollinearity outputs predate current feature selection and need recomputation. `MODELS/price_error_audit.py` supports optional saved-error inspection.

<a id="full-comparison-including-youtube-and-all-makes"></a>
### NHTSA and YouTube feature comparison

`NHTSA/text_features.py` separately tests complaint/recall components, support, text vectors, and optional YouTube ablations. These vectors are outside standard price variants; its default split holds out families:

```powershell
.\.venv\Scripts\python.exe -m NHTSA.text_features run-full --sample-size 0 --coverage-ablations
```

This creates a validation workspace with eight estimators and both weighting treatments. Inspect validation/convergence before a separate `compare --evaluate-test` with identical workspace, features, models, and budgets; see subcommand `--help`. Test results must not guide tuning.

## Vehicle discovery and recommendation

### Vehicle discovery v7: exploratory groups without abstention

The dashboard's **Vehicle categories** page uses validated full v7 run `20261114T204920` through the registry's `discovery_run`. Both model and trim-cohort views load this bundle, show a **Discovery v7** caption, and use its saved assignments and projections. The bundle contains 10,827 model profiles and 21,226 trim-cohort profiles, each covering the same 2,792,993 VINs across 45 body/fold contexts per grain.

V7 assigns every model and trim-cohort profile within its body category, including unresolved bodies. It skips v6's identity, five-VIN, powertrain-conflict, block coverage, partition acceptance, and 95th-percentile distance checks. Family-weighted K-means uses development-fitted median imputation, scaling, and encoding, with missing markers for categorical values. The highest sampled development silhouette wins among feasible cluster counts (2–8 for full runs). V7 applies no minimum silhouette or stability cutoff. A context with no usable development variation receives one exploratory group.

Preparation keeps available medians and modes without masking for fewer than five observations, IQR, or categorical agreement. Source reconciliation preserves unknowns, conflicts, and audit flags. Clustering uses fields with development variation, skipping coverage, redundancy, and horsepower-scale checks. It still uses family-held-out folds, lowercase-`vin` joins, separate cohort columns, and the existing NHTSA text and direct-peer rules.

The saved map uses development-median imputation to place every profile. Contexts that cannot fit two numeric axes place all profiles at the origin. Feature charts use observed values. V7 records `grouping_version=7` in the v6 artifact schema, which also reads existing v6 bundles. Treat the groups as exploratory.

```mermaid
flowchart LR
  P[VIN-linked model and trim profiles] --> D[Development-only encoding and imputation]
  D --> K[Exploratory K-means or single group]
  K --> G[Assign every profile without rejection]
  G --> E[Attach dated evidence]
  E --> V[Validate coverage, replay and family isolation]
  V --> A[Activate validated bundle]
```

Run from the repository root. The runner performs dependency checks, regression tests, preparation, smoke/full fits, text processing, evidence attachment, peer comparison, validation and activation. Existing v6 artifacts are preserved. A full local MiniLM cache is required; use `-EmbeddingModel '<local-snapshot>'` if automatic lookup fails. The runner stops on failure.

```powershell
Set-Location E:\DataScience_Capstone_Cars
.\.venv-youtube\Scripts\python.exe -m pip install -r requirements-discovery.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
$DiscoveryV7Run = 'MODEL_OUTPUT/vehicle_similarity_v7/' + (Get-Date -Format 'yyyyMMddTHHmmss')
Write-Host "Save this run path: $DiscoveryV7Run"
.\MODELS\run_vehicle_discovery.ps1 -Version 7 -RunRoot $DiscoveryV7Run
```

After interruption, set `$DiscoveryV7Run` to the saved original path and run `.\MODELS\run_vehicle_discovery.ps1 -Version 7 -RunRoot $DiscoveryV7Run -Resume`. Resume requires unchanged inputs, code, environment, encoder and device. Do not use a v6 run directory. After successful activation, launch `.\.venv-youtube\Scripts\python.exe -m streamlit run app.py`.

Resume keeps completed stages and body/fold fits, then restarts the unfinished context. For JSON receipts, the runner writes to unique sibling temporary files and retries permission-denied replacements for about five seconds. Persistent locks stop the run and preserve the previous receipt. Resume compares configuration as saved JSON, so tuples and their serialized lists match.

```mermaid
flowchart LR
  R[Resume original run] --> C[Verify and retain completed contexts]
  C --> F[Refit unfinished context and continue]
  F --> J[Write receipts with bounded file-lock retries]
  J --> V[Attach evidence, validate and activate]
```

### Vehicle discovery v6: specifications and NHTSA text

`MODELS/vehicle_similarity_v6.py` builds separate model and trim-cohort specification profiles with dated NHTSA evidence. The prior v6 bundle `20261114-snapshot-refresh` preserves the validated fits from `20261104T204904-listing-first` after exact source-compatibility replay. Its [validation receipt](MODEL_OUTPUT/vehicle_similarity_v6/20261114-snapshot-refresh/validation/validation.json) binds membership, folds, dates, text isolation, and vectors. The dashboard now selects v7 above; the app verifies the selected bundle/validation hashes and source identity.

Listing body labels take precedence over vPIC after lowercase-VIN linkage; blank labels fall back, while nonblank unrecognized labels remain unresolved. Source disagreements remain audit flags. The saved bundle and validation receipt record body provenance, support, family-held-out contexts, text channels, safety comparability, and coverage.

```powershell
.\.venv-youtube\Scripts\python.exe -m pip install -r requirements-discovery.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
.\MODELS\run_vehicle_discovery.ps1
```

The runner uses the same preflight, test, build, and activation sequence as above. Pass `-Device cpu` to force CPU. It uses a complete local MiniLM cache without downloading models. Outputs include `prepared/`, `text/`, `specs/`, `bundle/`, `peer-comparison/`, and `validation/`. [Discovery runner](MODELS/run_vehicle_discovery.ps1)

Resume using `-RunRoot '<printed-directory>' -Resume` with matching inputs, code, environment, encoder, and device. Completed stages are immutable. After a repair, preserve a failed stage under a new name before restarting it; keep completed prerequisite manifests unchanged. Encoder lookup honors Hugging Face cache variables and accepts transformer exports without `modules.json`, recording mean pooling and a 256-token window.

After a listing/vPIC snapshot correction, `refresh-sources --run <old-bundle> --prepared <fresh-preparation> --output <new-bundle>` can reuse saved fits only when a fresh `prepare` produces exactly the same nine discovery tables and all three NHTSA evidence files retain their hashes. Run these stages with `python -m MODELS.vehicle_similarity_v6`, then `validate` and `activate` the new bundle as usual. Changed values or populations require a full rebuild. Original artifacts remain immutable; `source_refresh.json` records the comparison.

```mermaid
flowchart LR
  S[Corrected snapshot] --> P[Fresh preparation]
  P --> C[Exact table replay and unchanged NHTSA hashes]
  C --> B[New bundle preserving saved fits]
  B --> V[Validate and activate]
```

The November 14 source refresh (`20261114-snapshot-refresh`) preserves the November 4 text cutoff and existing fitted groups.

To inspect body coverage before a rebuild:

```powershell
$BodyRun = 'MODEL_OUTPUT/vehicle_similarity_v6/' + (Get-Date -Format 'yyyyMMddTHHmmss') + '-listing-first'
.\.venv-youtube\Scripts\python.exe -m MODELS.vehicle_similarity_v6 prepare --data-dir DATA/CAR_DATA_FINAL --output "$BodyRun/prepared"
if ($LASTEXITCODE -ne 0) { throw 'Body preparation failed' }
Import-Csv "$BodyRun/prepared/coverage.csv" | Format-Table -AutoSize
.\MODELS\run_vehicle_discovery.ps1 -RunRoot $BodyRun -Resume
```

### Vehicle evidence and discovery v5

Legacy registry entries retain v5 `20261026-repair/run-reconciled` and EPA layer `20261026-epa-hierarchy-supported-validated`. They are used only when v6 is unconfigured; an unreadable configured v6 bundle warns rather than substituting legacy peers. Direct specification retrieval remains selected; cluster methods and source-trained category predictions are experimental. EPA, market, and NCAP classes retain separate definitions. Use the v5 CLI's `--help` for legacy stages and `MODELS/vehicle_category_hierarchy.py` for EPA-layer deployment checks.

<a id="vehicle-categories-and-category-aware-clustering-v4"></a>
Earlier alternatives are inspected in [Vehicle_Profiles.ipynb](MODELS/Vehicle_Profiles.ipynb); they are retained research, not extra setup steps. Discovery never adds clusters to price training. Buyer usefulness remains unmeasured for v5–v7; the historical v3 review did not establish a cluster-ranking gain.

### Local Streamlit dashboard

Launch with the activated discovery build environment:

```powershell
.\.venv-youtube\Scripts\python.exe -m streamlit run app.py
```

For the standard environment, use `.\.venv\Scripts\python.exe -m streamlit run app.py`. Open [127.0.0.1:8501](http://127.0.0.1:8501); checked-in configuration binds to loopback.

| View | Behavior |
| --- | --- |
| Dashboard | Cascading filters, historical distributions, bounded detail tables, evidence, downloads, and saved diagnostics. Totals cover the full matching scope. |
| Listings & cohorts | The Listings table and page download omit city, city/highway fuel economy, accident/title flags, and major options. |
| Vehicle categories | The selected v7 bundle includes unresolved bodies as a separate category. Advanced exploration defaults to selectable X/Y specification features in original units. The optional PCA map retains saved projections and feature directions. Both views retain saved groups and selected-vehicle markers; missing feature pairs are omitted. |
| NHTSA | Overview and comparison show complaint component shares, recall campaign counts, or recorded side/pole/rollover stars within one rating era. Complaint and recall views also offer similar vehicles and saved source evidence. |
| YouTube | Independent make selector defaults to the first alphabetical make. Two distinct charts show discussion-topic shares and overall sentiment, with an optional second make. A collapsed pipeline walkthrough traces one primary-make comment through its saved clauses, topics, sentiment, and eligibility; unsupported measurements are omitted. |
| Model performance | Registry-pinned Base/Enhanced runs and CatBoost Full; newest compatible completed continuation per other Full estimator. Availability refreshes every 30 seconds. Metric headers use MAE, RMSE, MSE, WAPE, R², and P95, with definitions above the overall table. The overall table omits Run; the metrics download retains it. |
| Similar vehicles | Saved reference-held-out distances from the selected discovery bundle; same body, ±10 years, distinct Make/Model alternatives. Filters reset the reference; coordinates never refit. |
| Vehicle Recommendation | Six-question search, one AI best match, up to five alternatives, follow-up refinement, and expandable evidence/research. |

The selected v7 bundle assigns every profile to an exploratory specification group. Group letters refer to the current reference fit; missing source observations remain unknown. The earlier v6 view also distinguishes ungrouped profiles by data support and assignment distance.

```mermaid
flowchart LR
  N[Filtered NHTSA evidence] --> C[Component or safety overview and comparison]
  N --> S[Similar vehicles from complaint or recall text]
  N --> E[Selected vehicle source evidence]
```

YouTube choices come from the selected Full run's verified export and eligible catalog makes. Overall sentiment uses a fixed −1 to 1 scale. The pipeline walkthrough shows saved content labels and the opinion-clause combination rule. It explains date filtering, deduplication, and equal-video aggregation. An eligible comment can still be dropped as a duplicate. Cached readers refresh when source data or `car_app/data.py` changes.

```mermaid
flowchart LR
  Y[Verified make-level YouTube export] --> M[Local make selector]
  M --> C[Optional second make]
  C --> T[Topic shares]
  C --> S[Overall sentiment]
  M --> E[One primary-make saved comment]
  E --> Q[Saved clauses, topics, tone and eligibility]
  Q --> A[Explain comment and video aggregation]
```

[artifact_registry.json](car_app/artifact_registry.json) owns exact checkpoint policies. Completion requires metadata, predictor, nonempty validation/test receipts, and metrics per estimator, including estimators from active runs. Incomplete retries cannot hide completed checkpoints. Full CPU LightGBM additionally requires its shard-serialization receipt. Pending estimators remain visible without an incompatible fallback.

Historical diagnostics can filter saved receipts by decoded cohort, advertised price, mileage, and condition when those fields are present. Filters on other fields, such as body, fuel, or drivetrain, need matching source hashes. Inference compatibility is checked separately. Feature importance caches bind to the checkpoint and frozen inputs. Refresh missing caches after changing checkpoints.

```powershell
.\.venv\Scripts\python.exe -m car_app.model_registry --refresh-diagnostic-explanations
```

```mermaid
flowchart LR
  P[Saved evaluation receipts] --> F[Filter frozen attributes]
  F --> D[Historical prediction diagnostics]
  M[Selected predictors and frozen inputs] --> E[Offline explanation cache]
  E --> I[Checkpoint-matched feature importance]
```

Inference verifies feature order, preprocessing, packages, source and predictor identities. A changed code fingerprint requires replay of all frozen test inputs and VIN-aligned USD predictions (`rtol=1e-7`, absolute tolerance $0.01). Hash-bound passing receipts are cached under `MODEL_OUTPUT/diagnostics/inference_compatibility/`; replay does not retrain or relax strict paired-experiment rules.

### Conversational vehicle recommendation

Budget is inclusive. Custom budget and mileage caps accept values such as `18k` and `75k`. Use **Additional preferences** for optional requirements. Hard filters require observed listing facts. [Tools, evidence, and validation](MODELS/AI_AGENT_RECOMMENDATION_PLAN.md)

**Get AI recommendations**, follow-ups, and retries call OpenAI. Root `.env` supplies `OPENAI_API_KEY` and `OPENAI_MODEL` (currently `gpt-6-luna`). Process values take precedence. On failure, the app reports AI unavailability and shows basic catalog results.

The AI retrieves dictionary definitions, eligible candidates, saved specifications, price estimates, and dated source evidence. Each card explains the match, tradeoff, and comparison with the winner. **Explore supporting evidence** shows separate estimates, global importance, and per-vehicle baseline and signed contributions on the log-price scale. It also provides provenance and downloads. Enable the group-research explorer to load it. **Refine these recommendations** updates the list and form. Resubmit after a dictionary, source, or model change.

```mermaid
flowchart LR
  R[Submitted preferences] --> E[Retrieve eligible candidates and source evidence]
  E --> P[Verify price estimates and signed feature contributions]
  P --> V[Validate AI response]
  V -->|Rejected| C[One correction attempt]
  C --> V
  V -->|Accepted| A[Recommendations]
```

## Verification

Run the offline suites from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest car_app.test_dashboard
.\.venv\Scripts\python.exe -m unittest discover -s YOUTUBE/tests
.\.venv\Scripts\python.exe -m unittest discover -s EDA
.\.venv\Scripts\python.exe -m unittest discover -s NHTSA
.\.venv\Scripts\python.exe -m unittest EXPERT_REVIEWS.test_prepare_dougscore MODELS.test_price_evaluation MODELS.test_vehicle_profiles
```

The explicit `python -m car_app.agent --live-smoke` command makes at most two real API submissions and writes an ignored receipt. Human relevance/usability evaluation and calibrated price intervals remain future work.

## Research Papers

Literature review dated October 27, 2026, with implementation comparisons updated November 4. The presentation comparisons for Lessmann and Voß, Fayyaz et al., and Le et al. were checked against publisher abstracts or full manuscripts on November 10. Some comparisons rely on abstracts or previews. The project did not reproduce the papers' metrics, which use different datasets, targets, currencies, and splits.

### Vehicle recommendation

| Paper | Similarity to this project | Applicable difference |
| --- | --- | --- |
| Ngoc Luyen Le, Marie-Hélène Abel, and Philippe Gouspillou (2023), [A Constraint-based Recommender System via RDF Knowledge Graphs](https://arxiv.org/abs/2307.10702), CSCWD. | Uses RDF graphs, SWRL rules, and SPARQL to match vehicles to buyer constraints such as budget, mileage, and seating. | Preference matching leaves room to add saved price estimates and dated NHTSA evidence to candidate explanations. This project supplies those separate evidence channels; buyer usefulness remains unmeasured. |
| Sabri et al. (2025), [Responsive Web-Based Malaysian Car Recommendation System With AI-Based Filtering](https://doi.org/10.1109/ISIEA65768.2025.11138107), ISIEA. | Normalized attributes and cosine similarity match user preferences to vehicles. | This project adds separate cohort/model discovery, evidence states, and family-held-out checks. Paper comparison is limited to its abstract. |
| Shi et al. (2022), [RecNet: A Resource-Constraint Aware Neural Network for Used Car Recommendation](https://doi.org/10.1007/s44196-022-00155-9), International Journal of Computational Intelligence Systems. | Vehicle recommendation under inventory and financial constraints. | Learns sequential user/item representations from auction interactions, with chronological evaluation. This project uses stated requirements and catalog attributes without equivalent buyer-history data. |
| Huang et al. (2023; revised 2024), [Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations](https://arxiv.org/abs/2308.16505), preprint. | An LLM coordinates recommendation tools, conversation, and memory. | Related to the implemented local agent; its behavioral recommenders differ from this catalog's deterministic retrieval and bounded evidence tools. |
| [PFMAR: A multi-agent conversational recommender system for high-involvement products](https://doi.org/10.1016/j.dss.2026.114714) (2026), Decision Support Systems. | Automotive recommendations and changing preferences across dialogue turns. | Studies planner/actor/critic/reflection agents. This project uses one agent with explicit preference revisions, deterministic requirements and source checks. |

### Price prediction and integrated systems

| Paper | Similarity to this project | Applicable difference |
| --- | --- | --- |
| Stefan Lessmann and Stefan Voß (2017), [Car resale price forecasting: The impact of regression method, private information, and heterogeneity on forecast accuracy](https://doi.org/10.1016/j.ijforecast.2017.04.003), International Journal of Forecasting. | Compares resale-price regression methods, market heterogeneity, and sellers' private information. | Resale forecasting leaves a different target: advertised US prices with linked NHTSA, opinion, and expert context. This project's source comparisons explore that opportunity. Comparison uses the publisher abstract and does not establish improved forecasting. |
| Imran Fayyaz, G. G. Md. Nawaz Ali, and Samantha S. Khairunnesa (2025), [Advanced Feature Engineering and Machine Learning Techniques for High Accurate Price Prediction of Heterogeneous Pre-Own Cars](https://doi.org/10.3390/vehicles7030094), Vehicles. | Benchmarks regression models and feature engineering on 4,009 vehicle records, including linear models, forests, boosting, stacking, and neural networks. | Listing features leave room to test added vPIC, NHTSA, YouTube, and DougScore inputs on matched VIN splits. This project uses cross-source linkage and source ablations; the algorithms themselves are established. |
| Dutulescu et al. (2023), [Car Price Quotes Driven by Data-Comprehensive Predictions Grounded in Deep Learning Techniques](https://doi.org/10.3390/electronics12143083), Electronics. | Predicts advertised prices from vehicle attributes and equipment. | Studies German/Romanian markets, vehicle images, neural models, and transfer learning. This project uses US listings with structured evidence and aggregated opinion/expert features. |
| Kang and Fu (2026), [The pricing of used electric vehicles via machine learning and hedonic price model: A comparative study](https://doi.org/10.1016/j.cie.2026.111964), Computers & Industrial Engineering. | Combines vehicle attributes, review sentiment, clustering, and interpretable price models. | A close conceptual match using used-EV transaction samples and market segmentation. Here, specification clustering supports discovery separately from price fitting; YouTube evidence is make-level. |
| Youssef et al. (2026), [An Intelligent Multi-Task Framework for Used Vehicle Valuation Marketability Classification and Budget-Aware Recommendation](https://doi.org/10.66279/svgz0r91), Engineering Systems and Intelligent Technologies. | Integrates valuation, marketability, and budget-aware recommendation. | Pricing uses synthetic Egyptian listings; simulated recommendation labels share the utility formula. This project uses historical observed listings but still needs independent buyer-usefulness evaluation. |

Related deployed examples include [CarGurus conversational search](https://www.cargurus.com/about/press/ai-search-experience) and [Cars.com Carson](https://www.cars.com/articles/meet-carson-cars-coms-new-ai-engine-for-car-shopping-518222/). Company announcements establish product availability, not independently validated recommendation quality.

**Research focus:** quantify the incremental predictive value of VIN-linked specifications, NHTSA evidence, make-level YouTube opinions, and expert ratings, then evaluate whether constraint-respecting vehicle discovery is useful. This source combination is a potential distinction; the review does not establish first-of-its-kind novelty or superior performance.

## Data conventions and limits

- A vehicle cohort is separate vPIC `ModelYear`, `Make`, `Model`, and `Trim` columns. Join listings to vPIC on lowercase `vin`, never decoder `VIN`. Combined identities are display labels only.
- Missing values and absent source records do not establish no defects, neutral sentiment, or equipment absence. Make-level opinions and expert averages are not observations about individual vehicles.
- Listings end in September 2020; the provisional NHTSA cutoff is `2020-12-31`, and YouTube uses `2026-10-01`. Historical text/score availability is unverified. Enriched results describe retrospective associations, not demonstrated historical forecasting or current inventory.
- Prices are advertised USD amounts, not verified transactions or ownership costs. Mileage units remain unverified. Complaint counts, source-trained categories, and text opinions are not validated reliability, safety, or buyer-suitability measures.
- Feature selection, clustering stability, and SHAP do not establish predictive improvement or causality. Model comparisons require compatible populations/protocols; recommendation relevance requires a separate evaluation.
