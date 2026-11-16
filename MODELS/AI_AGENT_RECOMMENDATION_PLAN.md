# AI vehicle recommendation: implementation and evidence contract

Status: November 15, 2026. Recommendation pricing currently selects the best completed tree and linear checkpoints by validation MAE. Full CatBoost OOF remains an optional offline workflow. Implemented behavior does not establish buyer suitability.

## Architecture and sources

This is a retrieval-augmented recommendation agent with structured tools. Python applies requirements and retrieves bounded catalog/model/evidence context; GPT-6 Luna clarifies preferences, explains tradeoffs, and reorders eligible IDs. Full CatBoost OOF training and the final inference fit are explicit offline workflows. No training, vector fitting, ensemble or learned preference score runs inside an AI request.

The [data dictionary](../DATA_DICTIONARY.md) defines source fields, transformations, populations and missingness. Runtime loaders validate actual schemas, saved manifests and artifact receipts. The dictionary describes data; it is not an embedding index or the catalog itself.

| Input | Grain and role | Interpretation |
| --- | --- | --- |
| Six selected Parquets and manifest | Listing facts, VIN decodes, NHTSA records, raw YouTube comments | Historical catalog; applicable observed facts satisfy hard requirements. |
| Selected v7 bundle `20261114T204920` | Listing-first body reconciliation; 10,827 model profiles and 21,226 trim profiles over 2,792,993 VINs | Shared dashboard/AI registry and validation receipt. Groups are exploratory; direct specification distance remains the default for peers. |
| NHTSA discovery text outputs | Dated family/component support, TF-IDF/SVD and MiniLM mixtures, narratives | Context-local unreviewed groups; allegations/campaigns, not risk estimates. |
| Selected YouTube sentiment-v2 export | 83 makes; 75 exported measurements, 40 selected price inputs | Supported discussion/tone/uncertainty at make level; independent accuracy unmeasured. |
| DougScore v2 | Model-year family or make fallback supported by five distinct models | Supplemental expert opinion; no trim match or verified historical availability. |
| Selected tree and linear checkpoints | Completed saved predictors, validation selection, test predictions and split/source/replay receipts | Historical advertised-price estimates; not current valuations, transaction prices or preference labels. |

Read `DATA/CAR_DATA_FINAL/`. Join listings to vPIC on lowercase `vin`, never decoder `VIN`. Keep `ModelYear`, `Make`, `Model`, and `Trim` separate. Missing trim may support a labeled model-level discovery fallback; never fill it to fabricate a complete cohort. Model/cohort populations overlap and must not be added.

## Discovery and text evidence

- Load the selected discovery bundle through existing validation and source checks. Grouping retains grain, body partition and excluded query-family fold. Prices and NHTSA outcomes do not enter specification coordinates.
- Reference comparisons use one saved reference fit for ranks and shared-cluster labels. Other subgroup descriptions retain their own contexts. IDs and axes never align across fits, periods or representations. Unknown semantic categories do not prevent specification retrieval.
- Dashboard filters select reference VINs. Alternatives use independent requirements over the broader catalog, retaining same-body, plus/minus-ten-year and distinct Make/Model restrictions.
- Historical NHTSA through `2020-12-31` is default. `latest_saved` is separate: currently built through `2026-11-04`, newest retained record `2026-09-23`. Collection completeness and historical version availability remain unverified.
- Retrieve named component counts/shares, supported channel-specific TF-IDF/MiniLM mixtures, and at most two dated source examples per family (first distinct ID per source, text fields limited to 240 characters). Examples do not measure prevalence. Opaque coordinates do not acquire automotive labels.
- YouTube uses the selected explicit-comment/cumulative argmax export through `2026-10-01`. Overall sentiment varies across 73 supported makes; aspect support is narrower and performance has no supported scores. Missing sentiment is not neutral. Confidence and teacher agreement do not establish automotive accuracy.
- Expert retrieval preserves family/make fallback, review/model support, nulls and dating limitations. Existing price feature selection remains unchanged.
- Crash comparisons retain applicable tested variants, rating eras, body classes, supported curb weights and the 250-pound comparison restriction. Campaign counts do not establish unresolved VIN recalls; complaint counts have no failure-rate denominator.
- Component avoidance leaves insufficient source support unranked in personalized output. Ordinary hard-filtered browsing remains available. Text never becomes a reliability or universal safety score.

## Selected tree and linear prices

`recommendation_price.kind = best_tree_linear_v1` selects one completed tree and one completed linear checkpoint from the dashboard's configured `price_run_selection`. Compatible Full continuations remain eligible per estimator. Select by validation MAE, never test error; the current winners are Full CatBoost and Full multiple linear regression. Both estimates reach the mandatory evidence tool and evidence table separately. No averaging or training occurs during recommendation.

Saved unique test predictions must belong to the saved test VIN partition before they support historical model-relative ranking. The tree is the primary ranking model, with the linear model used if no tree checkpoint exists. Other requested VINs may receive supplemental inference using the verified feature assembly; these may be fitting VINs, so they cannot support held-out value ranking or evaluation. The corrected source snapshot requires a hash-bound equivalence proof against original source files and a frozen-input/prediction replay for each estimator. Exact-checkpoint global TreeSHAP and linear coefficients accompany the AI context when their explanation receipts match the predictor and frozen inputs. These are global, noncausal explanations, not vehicle-specific weights. Estimates retain model, feature set, run, hashes, validation MAE and prediction type. Missing evidence is disclosed without removing otherwise eligible observed-price browsing.

The explicit `MODELS.price_oof` training/validation/activation commands remain available. Activating an OOF bundle switches the recommendation policy back to that bundle. Existing source fields, feature definitions and cohort joins are unchanged.

## Agent, tools and state

`car_app.agent` uses one Agents SDK `Agent` with `OpenAIResponsesModel` and local tools. There is no hosted shell, arbitrary filesystem/SQL tool, training tool or handoff.

Read `OPENAI_API_KEY` and `OPENAI_MODEL=gpt-6-luna` server-side from ignored root `.env`; process environment values take precedence. Reuse the existing key. Navigation, imports, filtering and downloads create no provider client. Disable provider tracing and response storage. Missing credentials/provider failure retain deterministic search; never substitute a model automatically.

| Tool | Contract |
| --- | --- |
| `update_preferences` | Atomic changes with exact supporting quotes from the current user message; validate observed labels and questionnaire rules. Numeric values must occur in quoted text. |
| `search_eligible_catalog` | Apply raw requirements and return bounded IDs, facts, exclusions and separate specification/price/safety signals. |
| `get_specification_peers` | Read saved reference-context specification evidence for up to six retrieved listing IDs. |
| `get_price_evidence` | Read selected tree and linear held-out or supplemental inference USD estimates, prediction type and per-candidate predictor provenance for up to six IDs. |
| `get_recommendation_evidence` | Required before answering: retrieve each final candidate's price, specification and dated source evidence, retaining its own fold provenance. |
| `get_source_evidence` | Retrieve separately dated NHTSA, make-level YouTube and expert evidence for up to six listing or complete-cohort IDs. Cohorts retain family/make scope and never inherit an individual member's subgroup assignment. |

Strict tool schemas and application validation allow only retrieved IDs and allowlisted fields. Callers cannot supply artifact paths. Narratives and earlier assistant prose are untrusted data and cannot authorize preference changes.

Keep version-3 requirements and use the version-4 agent envelope/recommendation policy, including dictionary identity and delivered-evidence IDs. Invalidate older results without discarding buyer requirements. Record snapshot/artifact identities, candidate facts, deterministic lists, selected IDs, source references, warnings and telemetry. Send at most six recent messages; retain twenty in the UI. Current structured requirements remain authoritative.

- The form asks for an observed-price budget and primary priority; mileage, seating, use and additional preferences may remain unspecified. Ask at most one blocking clarification, rather than repeating optional questions. Retain up to three ordered priorities for compatible conversations. Validate clear edits; clarify broad descriptions rather than inferring constraints. Treat mileage/fuel-economy units as unverified and reject unsupported unit claims in candidate reasons. No numerical suitability score or arbitrary preference/value blend is used.
- Pool distinct-family priority retrieval with supported specification, held-out price and safety lists; deduplicate VINs and cap at 24 listings. Performance uses advertised horsepower; passenger-space retrieval uses seating; lower-mileage and lower-price priorities use their named fields. Default retrieval uses mileage. These are limited proxies, not general suitability scores. Attach saved profiles and their own held-out subgroup contexts. Retrieve at most ten complete cohorts and aggregate all qualifying members.
- Return one best match followed by up to five alternatives from distinct decoded Make/Model families. Every selection requires `reason`, `tradeoff`, `comparison` and candidate-specific `source_refs`. The winner explains why it leads; alternatives explain when to choose them instead. General recommendation requests leave personalized cohort output empty; cohort aggregates and deterministic lists remain supporting evidence. Ordering is preference-based and limited to the retrieved pool.
- Recheck IDs, raw requirements, source signatures and selected model hashes before display. Changes invalidate candidate context. Invalid/provider-failed replies return available deterministic results; changed sources/models require fresh submission.
- Limits: five model turns per provider run, ten tools across a submission, 128,000 application-context/tool-output characters, 4,000 characters per user message, 3,000 output tokens per call, 120-second submission deadline, twenty-second transport timeout, one transport retry. A final response rejected by an application-owned validation rule can receive one correction attempt within the same submission limits; unknown errors do not retry. Safe validation reasons appear in telemetry. Checks surround local operations; blocking file reads may finish before cancellation is observed.
- Record requested model, available token usage, requests/tools, duration and safe failure category. Never expose keys, provider exception bodies or unvalidated model output.

## Streamlit workflow

The [README](../README.md#conversational-vehicle-recommendation) documents launch, questions, cards, refinement, and supporting-evidence controls. Only an explicit submission, follow-up, or retry invokes OpenAI. Reruns and local estimates make no provider request. Request/revision/artifact checks discard late replies; independent and reference conversations retain separate state.

## Verification and remaining evaluation

Provider stubs in the existing dashboard suite exercise preference edits, retrieval, ID/reference validation, fallback, session isolation and native reruns. Existing discovery checks cover reference folds, missing support, safety comparability and held-out prices.

```powershell
.\.venv\Scripts\python.exe -m unittest car_app.test_dashboard
```

The optional smoke command explicitly makes at most two real submissions with the configured key/model. It changes a used-Sedan budget from $20,000 to $18,000 while preserving seating/priorities and saves safe receipts under ignored `MODEL_OUTPUT/agent_smoke/latest.json`:

```powershell
.\.venv\Scripts\python.exe -m car_app.agent --live-smoke
```

Offline tests cover preference edits, hard limits, distinct-family cards, evidence delivery, provider-call isolation, fallback labels, and source/model compatibility. The optional smoke checks request validity and a follow-up budget edit; it does not measure recommendation relevance or guarantee latency. Human usability evaluation, independent text validation, calibrated price intervals, and deployment remain separate work.

Implementation references: [Agents SDK](https://developers.openai.com/api/docs/guides/agents/sdk), [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [function calling](https://developers.openai.com/api/docs/guides/function-calling). Retrieval/orchestration research includes [InteRecAgent](https://arxiv.org/abs/2308.16505) and [RecLLM](https://arxiv.org/abs/2305.07961); these studies do not validate this catalog or ranking policy.
