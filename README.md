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
  D["DougScore"] --> F
  F --> M["Price fits and held-out evaluation"]
  P --> V["Specification profiles and dated evidence"]
  M --> C["Verified saved checkpoints"]
  C --> A["Dashboard and recommendation"]
  V --> A
```

## Setup and data preparation

Run PowerShell commands from the repository root. Large data, model outputs, environments, credentials, and browser receipts are ignored by Git. Keep saved run logs and manifests for reproducibility; Python caches and `.playwright-cli/` are disposable.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-app.txt
```

Collection needs the listing CSV, SQL Server 2019+, and the Microsoft ODBC Driver. Restore the vPIC backup using [restore_vpic_database.sql](NHTSA/restore_vpic_database.sql), adjusting paths and instance first:

```powershell
.\.venv\Scripts\python.exe NHTSA/build_car_data_db.py --load-only
.\.venv\Scripts\python.exe NHTSA/build_car_data_db.py --only all
.\.venv\Scripts\python.exe YOUTUBE/YOUTUBE_COMMENTS_API.py
```

`--load-only` replaces listing rows; other collection resumes saved work. `--only vpic`, `recalls`, `complaints`, or `safety` limits NHTSA collection. YouTube uses `YOUTUBE_API_KEY` or `GOOGLE_API_KEY` from the environment or ignored root `.env` and consumes quota.

Export and profile the selected six tables:

```powershell
.\.venv\Scripts\python.exe EDA/DATA_CLEANING.py
.\.venv\Scripts\python.exe EDA/profile_data_dictionary.py
```

The exporter applies shared eligibility exclusions, validates the replacement snapshot, and writes `manifest.json`; SQLite stays read-only. Review cohort outliers with `MODELS/cohort_price_mileage_review.py` and source integrity with `MODELS/price_data_diagnostics.py`. Exact fields and eligibility rules belong in the [dictionary](DATA_DICTIONARY.md#data-rules).

## Text and expert features

### YouTube GPU text features

Use the [GPU runbook](YOUTUBE/docs/GPU_TEXT_RUNBOOK.md) for CUDA setup, teacher/student training, scoring, relabeling, aggregation, calibration, and recovery. The selected export is `YOUTUBE/outputs/sentiment_argmax_20261027T155951/price/`: 75 candidate fields for 83 makes, of which price models select 40. Coverage and field definitions are in the [dictionary](DATA_DICTIONARY.md#selected-youtube-predictors).

<a id="previous-youtube-sentiment-benchmark-workflow"></a>
Earlier experiments remain available through the [legacy sentiment plan](YOUTUBE/docs/EXPERIMENTAL_SENTIMENT_FEATURE_PLAN.md), [relevance runbook](YOUTUBE/docs/RELEVANCE_ROUND_RUNBOOK.md), and [research review](YOUTUBE/docs/NO_LABEL_TEXT_FEATURE_RESEARCH.md).

### DougScore expert features

[Expert integration](EXPERT_REVIEWS/DOUGSCORE_INTEGRATION_PLAN.md) documents source import and year/make/model matching with a supported make fallback. Build the default export only if absent:

```powershell
.\.venv\Scripts\python.exe -m EXPERT_REVIEWS.price_features --output-dir EXPERT_REVIEWS/outputs/price_features_v2
```

Later builds need a fresh directory and `--expert-feature-dir` during price training. The [dictionary](DATA_DICTIONARY.md#selected-dougscore-predictors) owns matching, support, and the eight-rating/two-source-field predictor definition.

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

Counts exclude generated missingness indicators. Default splits hold out disjoint VINs within decoded year/make/model strata with at least 25 eligible VINs, approximately 60/20/20. Trim pools only for splitting. `unseen_family` provides a separate transfer experiment. Preprocessing fits training rows; validation selects settings, then fit-plus-validation refits produce final test predictions. Models fit `ln(price)` and predict USD. Comparisons require matching populations, sources, partitions, and protocols.

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

### Vehicle discovery v6: specifications and NHTSA text

`MODELS/vehicle_similarity_v6.py` builds separate model and trim-cohort specification profiles with dated NHTSA evidence. The registry selects `20261104T204904-listing-first`, validated and activated November 5. Its [validation receipt](MODEL_OUTPUT/vehicle_similarity_v6/20261104T204904-listing-first/validation/validation.json) binds membership, folds, dates, text isolation, and vectors. The app verifies bundle/validation hashes and source identity.

Listing body labels take precedence over vPIC after lowercase-VIN linkage; blank labels fall back, while nonblank unrecognized labels remain unresolved. Source disagreements remain audit flags. The [dictionary](DATA_DICTIONARY.md#vehicle-discovery-v6-derived-features) defines body provenance, numerical support, family-held-out contexts, text channels, safety comparability, and coverage denominators.

```powershell
.\.venv-youtube\Scripts\python.exe -m pip install -r requirements-discovery.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
.\MODELS\run_vehicle_discovery.ps1
```

The [runner](MODELS/run_vehicle_discovery.ps1) performs preflight, regression checks, smoke/full fitting, evidence attachment, peer comparison, validation, and activation, stopping on failure. It needs a complete locally cached MiniLM snapshot; `-EmbeddingModel '<local-snapshot>'` selects one and `-Device cpu` forces CPU. No model downloads occur. Outputs include `prepared/`, `text/`, `specs/`, `bundle/`, `peer-comparison/`, and `validation/`.

Resume using `-RunRoot '<printed-directory>' -Resume` with matching inputs, code, environment, encoder, and device. Completed stages are immutable. After a repair, preserve a failed stage under a new name before restarting it; keep completed prerequisite manifests unchanged. Encoder lookup honors Hugging Face cache variables and accepts transformer exports without `modules.json`, recording mean pooling and a 256-token window.

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
Earlier alternatives are inspected in [Vehicle_Profiles.ipynb](MODELS/Vehicle_Profiles.ipynb); they are retained research, not extra setup steps. Discovery never adds clusters to price training. Buyer usefulness remains unmeasured for v5/v6; the historical v3 review did not establish a cluster-ranking gain.

### Local Streamlit dashboard

Launch with the activated v6 build environment:

```powershell
.\.venv-youtube\Scripts\python.exe -m streamlit run app.py
```

For the standard environment, use `.\.venv\Scripts\python.exe -m streamlit run app.py`. Open [127.0.0.1:8501](http://127.0.0.1:8501); checked-in configuration binds to loopback.

| View | Behavior |
| --- | --- |
| Dashboard | Cascading filters, historical distributions, bounded detail tables, evidence, downloads, and saved diagnostics. Totals cover the full matching scope. |
| YouTube | The selected Full run's verified 40 pricing inputs, original support counts, comparison/export, and optional topic chart; excluded evidence stays separate. |
| Model performance | Registry-pinned Base/Enhanced runs and CatBoost Full; newest compatible completed continuation per other Full estimator. Availability refreshes every 30 seconds. |
| Similar vehicles | Saved v6 reference-held-out distances; same body, ±10 years, distinct Make/Model alternatives. Filters reset the reference; coordinates never refit. |
| Vehicle Recommendation | Six-question search, one AI best match, up to five alternatives, follow-up refinement, and expandable evidence/research. |

[artifact_registry.json](car_app/artifact_registry.json) owns exact checkpoint policies. Completion requires metadata, predictor, nonempty validation/test receipts, and metrics per estimator, including estimators from active runs. Incomplete retries cannot hide completed checkpoints. Full CPU LightGBM additionally requires its shard-serialization receipt. Pending estimators remain visible without an incompatible fallback.

Inference verifies feature order, preprocessing, packages, source and predictor identities. A changed code fingerprint requires replay of all frozen test inputs and VIN-aligned USD predictions (`rtol=1e-7`, absolute tolerance $0.01). Hash-bound passing receipts are cached under `MODEL_OUTPUT/diagnostics/inference_compatibility/`; replay does not retrain or relax strict paired-experiment rules.

### Conversational vehicle recommendation

The [agent contract](MODELS/AI_AGENT_RECOMMENDATION_PLAN.md) owns tools, evidence requirements, checkpoint selection, request limits, and validation. Budget is an inclusive ceiling. Preset/custom budget and mileage caps accept values such as `18k` and `75k`; optional requirements stay under **Additional preferences**. Hard requirements use observed listing facts; unknown required facts fail.

Only **Get AI recommendations**, follow-up, or retry calls OpenAI. Root `.env` supplies `OPENAI_API_KEY` and `OPENAI_MODEL` (currently `gpt-6-luna`); process values take precedence. Navigation, downloads, checkpoint refreshes, and optional local tree/linear estimates make no provider request. Failures show AI-unavailable status and separately labeled basic catalog results.

The AI retrieves dictionary definitions, qualifying candidates, saved specification context, compatible price estimates, and dated source evidence before returning validated IDs. Every card explains its match, tradeoff, and comparison with the winner. **Refine these recommendations** updates the result set and form. **Explore supporting evidence** exposes coverage/provenance/downloads; the group-research explorer loads only when enabled. Dictionary/source/model changes require fresh submission.

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

Focused literature review, October 27, 2026; implementation comparisons updated November 4. Some studies were available only through abstracts or publisher previews. Their reported metrics were not reproduced and are not directly comparable across datasets, targets, currencies, and splits.

### Vehicle recommendation

| Paper | Similarity to this project | Applicable difference |
| --- | --- | --- |
| Le et al. (2023), [A Constraint-based Recommender System via RDF Knowledge Graphs](https://arxiv.org/abs/2307.10702), CSCWD. | Explicit budget, mileage, seating, and vehicle preferences constrain candidates. | Uses RDF, SWRL rules, and SPARQL; this project uses VIN-linked tables and specification profiles. Finding a feasible match does not establish buyer usefulness. |
| Sabri et al. (2025), [Responsive Web-Based Malaysian Car Recommendation System With AI-Based Filtering](https://doi.org/10.1109/ISIEA65768.2025.11138107), ISIEA. | Normalized attributes and cosine similarity match user preferences to vehicles. | This project adds separate cohort/model discovery, evidence states, and family-held-out checks. Paper comparison is limited to its abstract. |
| Shi et al. (2022), [RecNet: A Resource-Constraint Aware Neural Network for Used Car Recommendation](https://doi.org/10.1007/s44196-022-00155-9), International Journal of Computational Intelligence Systems. | Vehicle recommendation under inventory and financial constraints. | Learns sequential user/item representations from auction interactions, with chronological evaluation. This project uses stated requirements and catalog attributes without equivalent buyer-history data. |
| Huang et al. (2023; revised 2024), [Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations](https://arxiv.org/abs/2308.16505), preprint. | An LLM coordinates recommendation tools, conversation, and memory. | Related to the implemented local agent; its behavioral recommenders differ from this catalog's deterministic retrieval and bounded evidence tools. |
| [PFMAR: A multi-agent conversational recommender system for high-involvement products](https://doi.org/10.1016/j.dss.2026.114714) (2026), Decision Support Systems. | Automotive recommendations and changing preferences across dialogue turns. | Studies planner/actor/critic/reflection agents. This project uses one agent with explicit preference revisions, deterministic requirements and source checks. |

### Price prediction and integrated systems

| Paper | Similarity to this project | Applicable difference |
| --- | --- | --- |
| Lessmann and Voß (2017), [Car resale price forecasting: The impact of regression method, private information, and heterogeneity on forecast accuracy](https://doi.org/10.1016/j.ijforecast.2017.04.003), International Journal of Forecasting. | Compares forecasting methods, additional vehicle information, and heterogeneous markets. | Focuses on resale forecasting and sellers' information advantages. This project predicts advertised listing prices and tests specific NHTSA, YouTube, and expert additions. |
| Fayyaz et al. (2025), [Advanced Feature Engineering and Machine Learning Techniques for High Accurate Price Prediction of Heterogeneous Pre-Own Cars](https://doi.org/10.3390/vehicles7030094), Vehicles. | Compares linear models, forests, boosting, stacking, neural networks, and SHAP. | Uses 4,009 records and emphasizes feature engineering. This project's distinctions are cross-source linkage, matched source ablations, and reproducible receipts; the algorithms themselves are established. |
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
