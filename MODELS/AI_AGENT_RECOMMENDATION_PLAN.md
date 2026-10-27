# AI Agent Recommendation: research and system design

Status: October 26, 2026. Independent requirements search, filter-aware v4 category/cluster similarity, explanatory diagrams and saved-model price inference are connected. Price training remains unchanged. Conversational AI calls remain disabled.

## Recommendation and boundaries

Build a constraint-based vehicle recommender with an LLM conversation layer. The agent elicits preferences and explains results. Deterministic services enforce requirements, retrieve candidates, and calculate scores. Existing price predictors estimate advertised prices; they do not predict user preference or vehicle reliability.

Support two equal views: year/make/model/trim cohorts and individual historical listings. Use observed prices for hard budget limits and predicted prices as an additional value signal. Evaluate Jev for narrow semantic decisions before making it a dependency.

The local dashboard implements deterministic historical search, similarity, clusters and price estimates. The remaining conversational and utility-scoring designs below are future work. Hosted APIs remain deferred. The existing historical snapshot is not current inventory. Recommendation outputs do not extend the selected price-model dataset.

## Dashboard integration plan

### Provisional category workflow (v4)

The opt-in v4 CLI prepares VIN-supported variants with shared exclusions, attempts whole-catalog source evidence in resumable batches, builds scoped provisional labels, and discovers category × body-family subgroups in five family folds. Coverage, conflicting/unavailable evidence, subgroup support and source provenance remain inspectable. Small categories do not require a statistical subdivision.

The dashboard supports completed v3 and v4 bundles. v4 adds category exploration and explicit recommendation preferences: OR within a dimension, match fraction across selected dimensions, then the existing listing sort. Strict mode requires all selected dimensions and excludes unknowns. The reference panel defaults to saved v4 category-and-cluster peers, with category/specification and direct comparisons selectable. Buyer usefulness remains unmeasured. No hosted inference or classifier training is added.

All sidebar predicates apply to listing VINs before membership/profile aggregation. Cached scope statistics retain matching-listing counts and filtered price/mileage medians. Category charts, saved subgroup composition, profile tables and reference options follow that scope; the former 200-cohort reference cap is removed. Plot coordinates and assignments remain frozen. Reference changes refresh results automatically, including method-specific ranks/status; reference market summaries/examples follow the filters while alternatives retain broader catalog scope. The independent recommendation draft remains separate.

Validation reproduces preprocessing/peer results and checks immutable dependencies before explicit similarity-only activation. The registry preserves all price settings. Category changes clear stale category selections and affected results. The [README](../README.md#vehicle-categories-and-category-aware-clustering-v4) contains the commands. Future price-feature ablations require matched splits, training-only feature generation and separate predictor activation; current model training and selected dataset are unchanged.

### Implemented boundary

The local Streamlit app provides independent requirements search, reference-variant selection, direct peers and a cluster explorer under **Vehicle Recommendation**. It returns up to five cohorts and ten listings after hard filtering. Observed prices enforce budget. Explicit observed-price/mileage/seating/horsepower sorts determine ordering; broad priorities stay unscored. Compatible price estimates are supporting information only. No hosted AI client or Jev call is connected. Proposed utility/value blending below remains a deferred experiment.

| UI state | Owner and future behavior |
| --- | --- |
| `dashboard_filters` | Cascading vehicle and attribute scope for actuals and saved evaluation receipts. |
| `recommendation_session` | Independent requirements, messages, revision, and results. Dashboard changes must never update this session. |
| `similarity_session` | Separate requirements/messages/results plus dashboard reference identity. Filter or snapshot changes clear stale results and advance its revision. |

**Similar vehicles to the filter in the dashboard** is a separate section within the Vehicle Recommendation tab. Its reference comes from active dashboard filters. Those filters will not become automatic hard restrictions on alternatives. The main recommender starts from its own questions and answers.

### Current question-first increment

User choices: guided questions with a review screen; cover everyday buyers and enthusiasts. The user explicitly deferred all OpenAI calls and API-key tests. The supplied credential is stored in root `.env`, ignored by Git. The intended model is `gpt-6-luna`; credential validity and project model access have not been tested.

The grouped questionnaire validates a draft on **Review my requirements**, advances its revision, clears results and offers JSON download. Dashboard filters never supply its answers. Checked limits become hard requirements; unchecked defaults impose none. **Generate recommendations** requires a reviewed budget and matching snapshot. Unresolved priorities remain visible and unscored; unknown hard facts fail their filters. Result tables report qualifying-member cohort summaries and listing-level predictions separately.

| Round | Question | Contract and follow-up |
| --- | --- | --- |
| 1. Purpose | What will you use the vehicle for most often? | Commute, family, road trips, work/hauling, enthusiast/weekend, or undecided. Context only; no inferred restriction. |
| 1. Budget | What maximum advertised USD price should we use? | Clarify before search if missing. Apply inclusive observed price later. Taxes, fees, financing, and ownership costs are excluded. |
| 2. Basic needs | New, used, or either? Which body styles and minimum seats? | Undecided/empty categories impose no restriction. Unknown required seating fails. Cargo dimensions and towing need separate supported definitions. |
| 2. Limits | Maximum advertised mileage and decoded years? | Explicit optional limits. Mileage remains in source units pending verification. Contradictory years fail validation. |
| 2. Categories | Must-have makes, fuel types, or drivetrains? | Observed categories only. AWD and 4WD remain distinct. Empty selections mean no restriction. |
| 2. History | Require reports of no accidents, damage, salvage, or theft title? | Future filter requires explicit false for each history field; unknowns fail. Seller reports are not verified condition. |
| 3. Enthusiast/equipment | Transmission, minimum horsepower, or advertised equipment? | Optional for everyone. Exact transmission categories, raw horsepower, and existing quoted `major_options` labels. No inferred acceleration, handling, towing, or installed decoded equipment. |
| 3. Priorities | Which three priorities matter most? | Price, mileage, space, efficiency, performance, comfort, safety, reliability, or running costs. Ranking weights are not assigned yet. |
| 4. Clarification | What does an ambiguous priority mean? | Safety: crash protection vs assistance vs reported issues. Efficiency: city vs highway and comparable fuels. Comfort/performance: measurable features. Reliability/cost: explain missing validated targets and ask whether source-linked themes help. |
| 4. Review | Are the must-haves and priorities correct? | Show the submitted draft, unresolved questions, snapshot identity, revision, and limitations. JSON is a requirements receipt, not a recommendation. |

`car_app/requirements.py` owns validation and the version-1 draft contract. Its export contains `hard_requirements`, `soft_priorities`, unset `priority_weights`, unresolved questions, source mappings, limitations, and the intended provider/model with `api_calls_enabled=false`. These recommendation-only fields do not change the selected model-development dataset or its dictionary.

**Decisions still to review:** wording/order, whether enthusiast questions should be collapsible or a separate step, priority order/weights, and precise follow-ups for safety, comfort, performance, and efficiency. Free-text extraction and conversational history are not implemented.

### Pipeline and correlating outputs

| Stage | Output | State / acceptance |
| --- | --- | --- |
| Guided intake and review | Versioned `vehicle_requirements.json` download | Implemented; local validation of categories, limits, ambiguity, revisions, and independent state. |
| Deterministic search | Eligible listing IDs, complete cohorts, cumulative filter counts | Next; raw observed requirements and lowercase `vin` join cardinality. Never relax silently. |
| Compatible price adapter | Optional USD estimates and exact run/source receipt | Planned; explicit completed run and compatibility checks. Observed-price results survive missing estimates. |
| Comparison | Up to five diverse cohorts and ten listings; support and score contributions | Planned; frozen scoring, separate cohort dimensions, qualifying-member distributions. |
| Luna reasoning | Structured clarifications/explanations with candidate IDs and source references | Deferred by user instruction. No key validation or provider requests until questions are settled and live use is authorized. Bound context/tokens, validate IDs, retain deterministic fallback. |
| Streamlit results/export | Matching on-screen results and Markdown/JSON receipts | Planned; snapshot/revision checks; no calls on navigation or sidebar reruns. |
| Similarity | Independent reference/alternative comparison | Planned; narrow dashboard reference, broader alternatives, separate state. |

Offline checks may use fixtures and Streamlit AppTest. They must not instantiate an OpenAI client, read the key for testing, or send provider requests during this phase. No achieved recommendation relevance or explanation quality is claimed.

### Integration sequence once the engine is ready

1. **Freeze a backend contract.** Implement the proposed preference, search, price, evidence, and comparison interfaces in a separate service module. Accept validated requirements, explicit snapshot/run identities, request revision, and bounded limits. Return catalog IDs, observed attributes/prices, optional model estimates, support, evidence references, reasons, unresolved questions, and computed no-match information. Reuse the existing source/model adapters where their contracts fit; keep Streamlit imports out of backend logic.
2. **Connect the independent recommender.** Replace its disabled prompt/action with editable requirements, a separate conversation, and explicit submit/regenerate actions. Ask only unresolved questions that affect selection. Call the backend only on user submission; sidebar reruns and navigation must never trigger API calls. Display up to five cohorts and ten listings, evidence coverage, unavailable estimates, and a source-referenced Markdown export.
3. **Connect similarity separately.** Require a sufficiently narrow, coherent reference; otherwise ask the user to narrow it. Summarize reference counts and distributions before asking what must stay similar and what may change. Search outside the reference selection across the broader catalog. Display similarity reasons and differences in its own results/conversation. Exclude reference members according to the agreed cohort/listing identity.
4. **Protect against stale responses.** Attach snapshot identity, session revision, and reference hash to each request/response. Render only responses that still match their originating session. A dashboard change invalidates similarity work, including an in-flight response; it must leave the independent recommender unchanged.
5. **Add hosted AI after deterministic acceptance.** Load credentials server-side from the ignored local environment, without logging or rendering them. Validate structured tool arguments and returned IDs against bounded retrieved candidates; treat source text as evidence, not instructions. Keep Jev default-off with independent controls in each section. Revalidate IDs and hard requirements after any reranking; retain deterministic results when a provider fails.
6. **Enable only after end-to-end checks.** Use controlled fixtures and provider stubs, then a small explicitly triggered live request. Confirm UI/export agreement, grounded claims, bounded context, no unintended calls on filters, responsive layout, and secure error handling. Update this status and README with the actual engine/API setup when enabled.

### Backend readiness and acceptance

- Enforce inclusive hard requirements in code. Positive observed prices enforce budgets; unknown required evidence fails. Explain computed zero matches and apply a relaxed constraint only after user selection.
- Use compatible saved predictors and their USD outputs. Preserve observed-price candidates when inference fails; estimates stay unavailable. Do not retrain or borrow explanations from another run.
- Preserve this design's support, scoring, diversity, and evidence rules. Complaint counts and unsupported discussion signals must never become reliability scores.
- Test contradictions, revisions, boundary values, missing evidence, low support, no matches, and provider/Jev failures. Acceptance requires zero invented candidate IDs and zero hard-filter violations.
- Test that dashboard changes clear similarity results and reject late responses while preserving independent recommendation requirements, messages, and results.
- Keep the dashboard usable without credentials or a functioning recommendation backend. Model hosting, DougScore, calibrated price intervals, and learned preference ranking remain deferred.

## Research review

Searches used SciSpace and Consensus for conversational, constraint-based, and vehicle recommendation research. Relevant records were checked against paper or publisher pages. Hugging Face supplied public paper/model pages; its model-search connector was unavailable. Duplicate preprint and conference records are treated as one study. This is a focused design review, not an exhaustive systematic review.

| Study | Evidence and limitation | Design implication |
| --- | --- | --- |
| [A Constraint-based Recommender System via RDF Knowledge Graphs](https://arxiv.org/abs/2307.10702), CSCWD 2023; [conference DOI](https://doi.org/10.1109/CSCWD57460.2023.10152701) | Applies constraints and knowledge graphs to vehicle purchase/sale recommendations. Its results do not establish performance on this catalog. | Use explicit requirements. Start with existing tables; introduce a graph only for a demonstrated relational-retrieval limitation. |
| [Recommender AI Agent / InteRecAgent](https://arxiv.org/abs/2308.16505), 2023, revised 2024 | Combines an LLM with recommendation tools, memory, and planning. Its recommenders use behavioral information absent from this project's documented sources. | Reuse orchestration, replacing behavioral retrieval with catalog filters and transparent preference scoring. |
| [Leveraging Large Language Models in Conversational Recommender Systems / RecLLM](https://arxiv.org/abs/2305.07961), 2023 preprint | Describes preference understanding, dialogue management, explainability, and user simulation. The proof of concept does not validate vehicle recommendations. | Keep session preferences explicit; evaluate dialogue and retrieval separately. |
| [Logit Space Constrained Fine-Tuning for Mitigating Hallucinations in LLM-Based Recommender Systems](https://aclanthology.org/2025.emnlp-main.1491/), EMNLP 2025 | Reports reductions in hallucination through specialized fine-tuning across recommendation benchmarks. | Validate catalog IDs and claims first; defer fine-tuning until labeled failures establish a need. |
| [Conformalized Quantile Regression](https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html), NeurIPS 2019 | Combines quantile regression and conformal calibration for adaptive intervals, subject to statistical assumptions. | Evaluate calibrated price intervals independently of ranking and AI confidence. |
| [Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation](https://arxiv.org/abs/2403.09738), 2024 preprint | Identifies deviations between simulated and human conversational behavior. | Use synthetic dialogues for coverage; use human judgments for relevance and usability conclusions. |
| [Beyond Answer Confidence: A Controlled Audit of Self-Knowledge in a Black-Box Decision Model](https://arxiv.org/abs/2610.01006), October 2026 preprint | Audits Jev and reports overconfidence with absent evidence and beyond a knowledge boundary. This is recent, preliminary independent evidence. | Test missing evidence explicitly; require source-backed decisions even at high reported confidence. |

This architecture is an engineering inference from the research. No reviewed paper establishes the best model, ranking weights, or confidence thresholds for this dataset. Collaborative filtering and learned ranking are deferred because the documented sources contain no user-item preference histories or suitability labels.

## Data-to-preference contract

The authoritative definitions are in [DATA_DICTIONARY.md](../DATA_DICTIONARY.md). Read the six tables from `DATA/CAR_DATA_FINAL/` and check their schemas against `manifest.json`. Reuse existing cleaning and eligibility rules rather than silently broadening the population.

| User question or requirement | Source fields | Proposed handling |
| --- | --- | --- |
| Sedan, SUV, truck, or another body style? | `used_cars.body_type`; vPIC `BodyClass` for context | Map language to observed categories, including `SUV / Crossover`. Do not silently substitute a different classification for missing listing evidence. |
| Maximum price? | `used_cars.price` | Positive observed advertised USD price must be at or below the inclusive maximum. Taxes, fees, financing, and ownership costs are excluded. |
| New/used? Preferred years? | `is_new`; decoded `ModelYear`; listing `year` for discrepancy review | Normalize boolean strings explicitly. Use decoded year for cohort filters; surface conflicting or missing identities. |
| Maximum mileage? | `mileage` | Validate finite nonnegative values; apply an inclusive maximum. Verify the source unit before exposing a miles-labeled control. |
| How many seats? | `maximum_seating`, existing `maximum_seating_numeric`; vPIC `Seats` for context | Reuse deterministic extraction. Keep disagreements and unknown capacity visible. |
| AWD, 4WD, or another drivetrain? | `wheel_system`, `wheel_system_display` | Match explicit categories; do not equate AWD and 4WD without user agreement. |
| Fuel type or transmission? | `fuel_type`, `transmission`, `transmission_display`; decoded electrification fields | Use documented catalog aliases. Fuel category alone cannot establish range or charging speed. |
| Fuel economy a priority? | `city_fuel_economy`, `highway_fuel_economy` | Ask which measure matters. Verify units and fuel-specific comparability; do not invent a combined rating or equate MPG and MPGe. |
| Accident or title restrictions? | `has_accidents`, `frame_damaged`, `salvage`, `theft_title` | Require explicit reported false values for an exclusion requirement. Unknown history does not pass; reports are not independent verification. |
| Particular equipment? | `major_options`, existing `option_*`; vPIC equipment fields | Distinguish advertised, decoded, standard, and optional evidence. Optional equipment does not establish installation on a VIN. |
| Safety a priority? | Raw `NHTSA_Safety` variants; existing family aggregates | Clarify crash protection, driver assistance, or reported issues. Preserve applicability and coverage. |
| Reliability or maintenance cost? | Complaint/recall records; optional make-level YouTube evidence | Explain reported themes and limitations. No validated reliability probability, repair-cost target, or failure-rate denominator exists here. |

### Identity, joins, and time

- Cohort identity is the separate decoded `ModelYear`, `Make`, `Model`, and `Trim` columns. Join listings to vPIC on lowercase `vin`, never the decoder field `VIN`.
- Validate join cardinality: one retained listing per VIN and one selected decode per VIN. Multiple decodes require an explicit selection rule, not an arbitrary first row. Family aggregates join many-to-one after documented normalization.
- Preserve listing IDs and original decoded columns. Incomplete identities may appear as individual listings with warnings; do not silently fill them to create complete cohorts.
- Listing dates span `2010-11-03` to `2020-09-13`; price-observation dates are unverified. All results state that prices and availability are historical.
- Complaint/recall model features use provisional cutoff `2020-12-31`; YouTube features use `2026-10-01`. Mixed-date evidence supports retrospective exploration, not a historically available recommendation experiment.
- Keep raw evidence separate from predictor inputs. Median imputation enables inference but cannot establish hard-filter compliance.

## Conversation and tool boundaries

Session state contains a revision number, hard filters, weighted soft priorities, unresolved questions, and supporting user statements. Explicit requirements become filters immediately. Ambiguous descriptions trigger clarification instead of inferred restrictions. Each preference revision invalidates previous results.

Start with use case and budget/body-style questions, then ask only missing questions that could change the shortlist. Users may skip optional preferences or request results. For zero matches, report computed filter exclusions and alternatives changing one requirement at a time. Apply a relaxation only after the user selects it.

These are proposed internal interfaces, not implemented APIs:

| Interface | Input | Bounded output |
| --- | --- | --- |
| `update_preferences` | Revision; explicit additions, replacements, or removals with user-statement references | Validated next revision and ambiguities; reject unsupported fields/units. |
| `search_candidates` | Validated filters and result limit | Eligible IDs, relevant raw attributes, filter outcomes, total counts, and exclusion reasons. |
| `score_prices` | Candidate IDs and explicit compatible run selection | USD predictions, run/source/preprocessing provenance, and failures. No agent-supplied artifact paths. |
| `retrieve_evidence` | Candidate IDs, topic, optional text flag | Bounded source records, IDs, dates, grain, applicability, and missingness. |
| `compare_vehicles` | Eligible IDs, weights, cohort/listing view | Ranked supplied IDs, score contributions, summaries, coverage, and tradeoffs. |

Responses carry the session revision and snapshot identity. Allowlist argument fields and validate types. Give the agent no arbitrary SQL, filesystem, shell, or training access. Send bounded vehicle context rather than the dataset. Treat descriptions, comments, and retrieved documents as untrusted evidence, never instructions.

### Recommendation-only outputs

| Field | Definition |
| --- | --- |
| `candidate_id` | Stable listing identity within the snapshot, with lowercase `vin` and `listing_id`. Cohort IDs resolve to the four separate decoded dimensions. |
| `observed_price_usd` | Source `price`, retained independently of predictor inputs. |
| `predicted_price_usd` | Existing model's estimated advertised price in USD. |
| `value_gap_usd`, `value_gap_pct` | Predicted minus observed price; percentage is `100 * (predicted - observed) / predicted`, only for finite positive predictions. |
| `filter_results` | Per-requirement pass/fail/unknown with supporting field. Unknown required evidence does not pass. |
| `preference_score`, `score_components` | Utility and weighted contributions; not satisfaction probability. |
| `evidence_coverage` | Known/requested evidence, grain, and ambiguity; do not collapse into an overall confidence percentage. |
| `source_refs`, `model_ref` | Snapshot/table/record references and model identity sufficient to reproduce results. |

Default to up to five cohorts and ten individual listings; return fewer when eligibility/evidence is insufficient. Cohort summaries use qualifying members and show count, median, and interquartile range for price/mileage. These are filtered-sample distributions, not prediction intervals or cohort-wide guarantees. Show member-level budget compliance; an affordable median does not make every member affordable.

### Vehicle-profile research boundary

The offline `MODELS/Vehicle_Profiles.ipynb` pilot studies retained-listing specification profiles and conditional year/make/model complaint associations. It replaces the retired safety-equipment experiment. Cohort medians and advertised-option shares do not establish an individual vehicle's installed configuration; complaint associations do not identify trim-specific failures.

Recommendation integration remains outside this pilot. Any later use requires a separate review of source compatibility, profile stability, applicability and held-out usefulness. Apply must-have equipment filters to verified applicable records; Optional cannot establish installation. Research clusters and complaint themes provide no safety ranking, crash-risk prediction, or deployment readiness.

## Implemented similarity, model registry and completion workflow

`car_app/artifact_registry.json` pins the activated full-budget similarity run, selected price metadata/predictor hashes and comparison receipt. v4 defaults to frozen category-and-cluster peers in the reference panel, with direct/category baselines available. Displayed memberships use the reference's fitted context, including unsupported states. Filtered explorers select one fitted fold/view/method and display provisional category labels, subgroup descriptions, matching support and saved observed coordinates. Cluster numbers never align automatically across models.

`car_app.model_registry` inventories all eight feature variants, compares only compatible completed full-budget runs using the existing comparison routine, and records pending/pilot/failed cases. Matched-estimator source additions and expert-coverage controls use saved validation receipts; test scores never select. Source/code/package/feature compatibility is checked before executable predictors load. The base CatBoost full run is the provisional default. SHAP runs on saved bounded inputs in report copies. No training directory is modified.

When training finishes, build a fresh report with `--output <fresh-directory> --explain-rows 100 --publish`; inspect it in Dashboard → Feature comparisons. Activate a candidate explicitly with `--report <directory> --activate <variant/run/estimator>`. Publishing alone preserves the price-model pin. App session model selection is explicit and clears stale results. `MODEL_OUTPUT.ipynb` reads the same pinned comparison.

No additional human labels are required for this exploratory historical prototype. The partial similarity review remains unchanged; it does not establish general buyer suitability. New enrichment, value-score blending, conversational AI and deployment remain future work.

## Price integration and ranking

Reuse [Price_ML_Models.py](Price_ML_Models.py), especially `prepare_prediction_frame()` and `predict_listings()`. The saved predictor already applies the inverse log transform and returns USD. Do not exponentiate its predictions again.

Read-only inventory found completed `base`/`enriched` runs with `advertised-price-log-v3`, `unseen_family` splits, and validation-selected CatBoost. This establishes artifact availability, not compatibility or current performance. Before future inference:

1. Require completed metadata, recorded validation-selected estimator, and a trusted local predictor. Ignore running, failed, and unsupported legacy runs.
2. Check raw feature names/order, preprocessing, packages, source hashes, dataset fingerprint, code provenance, and feature-build cutoffs. Schema agreement alone is insufficient; review changed feature contracts rather than bypassing mismatches.
3. Compare base/enriched candidates only on compatible populations/splits using validation criteria. Select an explicit run, not the newest directory or the best exposed test result.
4. Verify stable identities, finite positive USD predictions, and the same result for a row alone or in a batch. On failure, retain observed-price browsing with an estimate-unavailable explanation; do not generate substitute prices with an LLM.

For the future baseline, filter first and calculate utility only for eligible rows. Use exact matches for categorical preferences and clipped percentile utilities for directional numeric preferences, with reference distributions frozen from development data. Explicit weights override equal weights across stated priorities. Missing preferred evidence contributes no utility and stays labeled unknown; do not renormalize each candidate over only observed fields.

Use 80% preference utility and 20% value utility as initial experimental weights when soft priorities exist. Value utility is the clipped development-reference percentile of `value_gap_pct`. Without soft priorities, display observed-price order and value comparisons rather than implying a personalized ordering. When the model is unavailable, use preference utility alone and record the limitation. Freeze weights before evaluation and include a zero-value-weight ablation.

Rank cohorts by mean qualifying-member utility, show support counts, and flag fewer than five members as low support. Prefer distinct year/make/model families in the first results, then fill remaining positions by score without duplicate IDs. Observed price and listing ID provide stable listing tie breakers; the separate cohort dimensions provide stable cohort tie breakers. Diversity never admits ineligible candidates. Optional AI reranking receives the same bounded eligible shortlist and passes a final deterministic ID/filter check.

A positive value gap is model-relative, not proof of a bargain, transaction value, or condition. Assess it with held-out or out-of-fold predictions. Training-row residuals cannot establish value-ranking performance.

## Safety and optional text evidence

Preserve raw safety variant IDs/descriptions for explanations. Normalize star-rating `0`/`Not Rated` as missing. Treat zero rollover values as missing only when paired ratings are unavailable. Recommendation normalization does not change sources or the fitted model's feature contract.

Family means are descriptive aggregates, not official trim/VIN ratings. An exact safety requirement needs unambiguous applicable evidence. If linkage is unavailable, classify the requirement as unverified and exclude the candidate from the strict eligible set. Offer family-level evidence separately only if the user chooses that interpretation. Family aggregates cannot establish installed equipment.

[NHTSA guidance](https://www.nhtsa.gov/ratings) restricts overall/frontal comparisons to comparable classes and weights within 250 pounds. Missing class, curb weight, rating-era comparability, or variant linkage prevents those claims. `CurbWeightLB` is sparse; `GVWR` is not curb weight. Do not create a universal safety score. Complaint counts are not failure rates; campaign counts do not establish unresolved VIN recalls. Missing source rows mean unknown coverage, not zero defects.

Text is optional and evaluated separately. Begin complaint/recall retrieval with a lexical baseline. Preserve record IDs, family applicability, cutoff, and conflicting-version exclusions. Add semantic retrieval only after improvement on labeled examples. Source-linked summaries distinguish reported allegations, campaigns, and structured facts.

YouTube features describe makes, not cohorts/VINs. The selected export lacks a usable variable directional sentiment signal; independent automotive accuracy is unmeasured. Show supported discussion themes with dates/coverage, not validated reliability or safety scores. DougScore stays outside the initial design while approved matches and availability dates are pending.

## Model comparison and Jev experiment

| Role | Candidate | Proposed evaluation |
| --- | --- | --- |
| Hosted conversation baseline | [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) | Structured extraction, tool arguments, clarification, and grounded explanation. |
| Hosted quality comparison | [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) | Same prompts/tools/scenarios; compare quality, latency, and cost. |
| Open-weight alternative | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Tool use and extraction with output validation; record revision/runtime/hardware. |
| Price estimator | Compatible existing `base`/`enriched` bundle | Validation-led selection and held-out price/value diagnostics. |
| Optional semantic decisions | [Jev `jev-1.13.0`](https://docs.typesafe.ai/models) | Narrow judgments added to the same recommendation baseline. |

Candidates are not measured winners. Recheck availability/pricing before experiments and record returned model IDs/configuration. The question-design phase makes no paid requests. A supplied credential is stored in ignored root `.env`; validity and model access remain untested. Future live setup sends only necessary context and bounds retries, token use, and spend.

Jev's [Choice/Score confidence](https://docs.typesafe.ai/confidence) summarizes concentration over supplied outcomes. Noul returns a yes/no probability without separate confidence. Neither establishes suitability or factual evidence. Evaluate three isolated tasks: select a legal next clarification, judge support for a supplied claim, and rerank a small eligible shortlist. Include uncertain/insufficient-evidence outcomes and retain full probabilities.

[Documented Jev limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13) include arithmetic, date comparisons, irrelevant context, adversarial content, and option-order sensitivity. Keep calculations in code, bound context, permute options, and test missing evidence. Pin the version for threshold experiments. Timeouts, unstable decisions, or low confidence lead to deterministic ranking or clarification. Jev cannot alter hard filters; retain it only if evaluation demonstrates useful improvement.

### Three separate uncertainties

- **Price:** independently calibrated prediction intervals. Existing `mae_group_bootstrap_low/high` measure aggregate MAE uncertainty, not individual price intervals.
- **Preference:** unresolved priorities or several acceptable alternatives. Clarification and comparisons address this uncertainty.
- **Evidence:** missing, ambiguous, outdated, or mismatched source records. Report coverage and applicability explicitly.

A future interval experiment needs calibration observations excluded from fitting and tuning. Existing validation rows used in the final refit cannot independently calibrate it. Compare split-conformal residual intervals with quantile/CQR alternatives, assess family dependence and temporal shift, and report empirical coverage/width rather than unconditional guarantees.

## Example conversations

These illustrate behavior, not actual catalog search results.

| Step | Dialogue and state change |
| --- | --- |
| Broad request | User: "I need a family vehicle under $25,000. Safety matters." Agent: "Would you consider a sedan, SUV, or either? How many seats do you need, and do crash ratings or driver-assistance features matter most?" Budget becomes a filter; family/safety descriptions remain unresolved. |
| Requirements | User: "SUV, seven seats, used, no more than 60,000 miles. Crash ratings matter most." Record the explicit requirements after confirming source mileage units; crash protection remains a soft priority until a specific evidence requirement is stated. |
| Evidence limit | Agent: "The saved ratings cover variants and some are unrated. I can show family-level evidence, but cannot verify a trim rating when the variant match is ambiguous." Unknown evidence cannot satisfy a later strict rating requirement. |
| No matches | Agent shows query-computed counts for alternatives changing only mileage, seating, or body style. Original requirements stay active; do not generate counts or relax silently. |
| Revision/results | User: "Allow 80,000 miles; keep the other limits." Recompute and show cohort comparisons plus individual listings, observed/predicted prices, tradeoffs, evidence coverage, and historical labels. |

## Evaluation protocol and acceptance

All experiments here are future work. Freeze snapshot, eligibility, price run, prompts, preference definitions, and scoring references for each comparison. Use development scenarios for changes and a locked set for final reporting. Do not tune on previously exposed price-model test results.

Build a reviewed scenario set spanning the cases below, with expected filters and acceptable evidence. Keep paraphrases of a scenario together in one partition. Obtain independently reviewed relevance grades and claim-support labels; adjudicate disagreements and report guidance, sample sizes, disagreements, and uncertainty. Synthetic users supplement coverage; they are not the sole judge.

| Comparison | Fixed inputs | Measures |
| --- | --- | --- |
| Structured preference baseline vs LLM conversation | Catalog and requirements | Extraction precision/recall, conflict handling, clarification burden, human usability. |
| Deterministic vs LLM vs LLM + Jev ranking | Eligible shortlist and prices | Human relevance/NDCG at 5 and 10 when graded labels exist, diversity, constraints, catalog validity. |
| Preference-only vs preference + value | Preferences and split-safe predictions | Relevance, value stability, subgroup behavior, sensitivity to price error. Price accuracy is not preference relevance. |
| Base vs enriched prices | Compatible populations/splits | USD error by price band, body style, make/year, and evidence coverage, with support counts. |
| Structured vs optional text evidence | Candidates and questions | Retrieval relevance, claim/source accuracy, unsupported-claim rate. |
| Jev ungated vs gated | Labeled decisions; development-only threshold selection | Brier score, log loss, reliability diagrams, sample-sensitive calibration error, and error-versus-abstention. Assess Choice confidence separately from outcome probabilities. |
| Price intervals | Independent calibration/evaluation | Nominal/empirical coverage, width, subgroup support, family/temporal shift. |

Record successful-session cost, tokens, retries, failures, and median/p95 latency. Cluster uncertainty estimates by scenario or family where dependent. Inspect make/body-style and missing-evidence subgroups. Set nonzero quality targets before locked evaluation; this document claims no achieved quality, latency, or cost targets.

| Acceptance scenario | Required future behavior |
| --- | --- |
| Exact budget/mileage boundary; missing or invalid values | Inclusive limits; unknown required facts fail; predictions never override observed budget. |
| Contradictions or zero matches | Clarify or explain computed exclusions; never relax silently. |
| Unknown accident/title history | Does not satisfy an explicit reported-history exclusion; distinguish reports from verified condition. |
| Missing ratings, ambiguous variants, optional equipment | Exact claims remain unverified; family evidence cannot establish a VIN requirement. |
| Incomparable safety classes/weights/eras | No unsupported overall/frontal comparative ranking. |
| Duplicate VINs/decodes or aggregate joins | Refuse row multiplication; preserve IDs and separate cohort dimensions. |
| Unsupported preference, units, or current-market request | State the missing data/capability and request a supported interpretation. |
| Requirement revision/removal | Increment revision, recompute, and reject stale results. |
| Injected instructions in retrieved text | Treat as content; no arbitrary action or filter change. |
| Artifact/provider failure or insufficient evidence | Explicit limitation and deterministic fallback; no fabricated prices or records. |
| Reordered Jev options or ties | Measure stability; use deterministic fallback/tie breakers. |

Hard-filter violations and invented candidate IDs must be zero in the acceptance suite. Revalidate after reranking and before display. Trace numerical comparisons to sources or calculations. Human review checks explanation support; fluency is not evidence of correctness.

## Staged implementation roadmap

| Stage | Deliverable / state | Completion gate |
| --- | --- | --- |
| 0. This change | Research/design document and README proposed diagram | Source/local-link, Markdown, and diagram static checks; no runtime/data changes. |
| 0a. Question intake | Guided questionnaire, validated review, and JSON export implemented; provider calls disabled | Offline boundary/ambiguity/state checks; user review of questions. |
| 1. Catalog and prediction adapter | Implemented read-only retrieval and pinned compatible price inference | Join/identity checks, raw/imputed separation, historical labels, estimate fallback. |
| 2. Deterministic recommender | Implemented requirements, explicit sorts, diverse cohort/listing results, direct peers and cluster views; utility blending deferred | Boundary, no-match, revision, context-consistent clusters and automated acceptance checks. |
| 3. Conversation layer | Typed tools, clarification, grounded explanation | Locked extraction/claim-support evaluation; zero invalid IDs/constraint violations. |
| 4. Optional experiments | Jev, text retrieval, intervals, provider comparison | Each addition improves a predefined outcome without constraint regressions; disclose costs/limits. |
| 5. Capstone demonstration | Reproducible historical demo and reviewed user study | Human relevance/usability evidence, pinned configuration, reproducible outputs, historical scope. |

Current inventory, transaction-price prediction, reliability modeling, collaborative filtering, learned personalized ranking, and deployment need separate data/evaluation decisions. Update `README.md` as workflows are implemented. Update `DATA_DICTIONARY.md` only when selected model-development sources, fields, or transformations change; keep recommendation-only definitions here.
