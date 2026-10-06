# YouTube comment qualification and make-level sentiment plan

**Current direction (October 6, 2026):** the user has deferred labeling.
Follow [the GPU text runbook](GPU_TEXT_RUNBOOK.md) for new raw-source runs.
Full-source scoring and make features are complete; the selected make export
is available to the retrospective price comparison. The
[experimental feature plan](EXPERIMENTAL_SENTIMENT_FEATURE_PLAN.md) preserves
the earlier pretrained benchmark using existing scores and DougScore.
The supervised validation sequence below is retained for later use and is not
required before this prototype. Inference and aggregation are implemented in `../sentiment_pipeline.py`; see
[the previous benchmark workflow](../../README.md#previous-youtube-sentiment-benchmark-workflow).


Status as of October 1, 2026: the source audit, annotation recovery, user-review
import, three-model relevance comparison, saved candidate fit, 10,000-comment
qualification pilot, and initial aspect-template preparation are complete. The current working
annotation snapshot is `outputs/relevance_round_01_reviewed`; the completed
comparison is its `model_comparison_v1` directory. The saved SetFit candidate
is `outputs/relevance_candidate_v1`. The independent relevance validation gate remains incomplete.
Candidate fitting/saving, preserving pilot scoring, a one-use blind evaluator,
and target/aspect annotation preparation/checks are now implemented in
`relevance_candidate.py` and `prepare_aspect_round.py`. Sentiment models and
aggregate feature building still need implementation. The six
model-development Parquet tables remain source snapshots.

## Next steps from the completed candidate and pilot

This section is the current execution order. The phase descriptions below
provide the longer-term design; their preparation commands describe earlier
stages and do not need to be rerun now.

### Current checkpoint

| Item | Verified state |
| --- | --- |
| Source | 1,981,686 comments from 4,224 review videos. |
| Reviewed training labels | 179 of 400 complete: 92 yes, 87 no; 221 unfinished. |
| Blind labels | 107 of 300 complete; 193 unfinished. Class outcomes remain withheld during candidate selection. |
| Independent reviewer subset | 0 of 105 complete. |
| Training subtype cleanup | Four missing negative subtypes and one unsupported `question` subtype. Binary labels remain the user's judgments. |
| Model comparison | TF-IDF, embeddings, and SetFit completed on identical five video-grouped folds. |
| Proposed operating point | SetFit, reject only scores strictly below 0.20 after validation. |
| Training CV at 0.20 | 91/92 relevant retained; 41/87 irrelevant rejected; 41/42 rejected comments were irrelevant. |
| Input length | Two labeled comments exceed the encoder's 256-token limit. Source text is preserved. |
| Saved candidate | `relevance_candidate_v1` fit complete on 179 labels; 449 seconds; save/load maximum score difference 0.0; model/input hashes verified. |
| Qualification pilot | 10,000 distinct comments from 1,496 videos, with 79 completed matching checkpoints. Proposed rejection: 2,243; scores 0.20–<0.50: 1,991; scores >=0.50: 5,766. All rows remain `review_or_pass`. |
| Pilot isolation | No protected training/blind ID, video, or normalized-text overlap; no seed-reserved videos. |
| Pilot input/duplicate audit | 97 inputs exceed 256 tokens; none are proposed rejections. There are 128 excess rows across 54 repeated normalized-text groups within the pilot. Source text and IDs stay intact; semantic duplicate review is still pending. |
| Pilot human audit | 300 rows, 100 per score stratum; all human fields are unfinished. |
| Aspect starter | `aspect_round_01`: 100 comments (66 reviewed-round, 34 Toyota), 100 target placeholders, 900 aspect rows; templates valid, all annotations unfinished. |

The [completed artifact review](../outputs/pipeline_run_review_01/run_review.json)
verifies candidate integrity, score/checkpoint agreement, frozen source fields,
and sample isolation. Start human work with the pilot's 100 proposed-rejection
audit rows, then the boundary/retained strata. Label aspect targets before
polarity/evidence. Finish relevance blind/subtype/reviewer annotations before
the one-use blind evaluation. Do not rerun fit, pilot, or preparation into the
existing output directories.

The 98.9% relevant recall and 97.6% rejection precision are descriptive training
CV estimates, following model-selected error review. They are not independent
test results. The saved bootstrap intervals describe the 0.50 cutoff, not 0.20.
The scores are uncalibrated; 0.20 is an operating cutoff, not a verified
20% probability of relevance. The comparison saves fold predictions, not a
reusable fitted classifier.

### 1. Freeze and save the relevance candidate — completed for v1

- Preserve the reviewed labels and `model_comparison_v1`. Freeze their hashes,
  the current relevance definition, and the proposed cutoff before consulting
  blind outcomes. Review discrepancies between the guide and user-confirmed
  labels explicitly; do not silently relabel the user's work.
- Run the fitter using all 179 completed training labels, the same pinned
  MiniLM revision, balanced logistic head, normalized embeddings, CPU default,
  256-token limit, five pair-sampling iterations, and 100 body-training steps.
  Record the full-fit seed and actual dependency versions. Keep unfinished and
  `unclear` labels out of fitting and keep blind labels out of the fitter.
- Save both the encoder and classifier head locally, along with preprocessing,
  source/label hashes, threshold, training counts, and model version. Verify that
  loading the saved artifact reproduces scores on a fixed test batch.
- Write a candidate configuration with automatic exclusion disabled. Fix the
  evaluation metrics and sufficient-evidence policy before blind evaluation.

**Deliverable:** a reusable `relevance_candidate_v1` model directory and frozen
candidate manifest. Candidate v1 is saved; another fit is unnecessary now.
The initial fit requires annotations unchanged from the
completed comparison; fit before making further label edits. Failed/interrupted
fits retain their partial output; choose a new candidate directory for another fit.

### 2. Score a pilot while preserving every comment — completed; human audit next

- Run the seeded 10,000-comment pilot drawn across videos, tentative title
  makes, dates, and lengths. Exclude the blind videos and normalized blind text,
  and separate already-labeled training comments from fresh audit examples.
  Enriched pilot proportions must not be reported as source-wide prevalence.
- Save every original `comment_id`, context/text, score, and model/source
  lineage. Summarize proposed rejections at 0.20 but retain all comments.
  Unknown language, ambiguous references, and long comments remain eligible
  for review. Exact filler rules also remain proposed skips pending their audit.
- Prepare a human audit sample from proposed rejections, retained comments, and
  boundary cases. Include cross-make comments, questions, creator praise,
  comparisons, and negation. Report each stratum separately or use sampling
  weights for a combined pilot estimate; the enriched audit is not a blind test.
- Measure rows/second, memory, input truncation, and make/video concentration.
  Check unique IDs, repeatability, and resume behavior before scaling.

**Deliverable:** `qualification_pilot_v1` scores, audit CSVs, and runtime report.
Use `relevance_candidate.py pilot`. Pilot scoring does not release an automatic
rejection rule. The source hash must match the candidate's frozen Parquet
snapshot. Interrupted pilots support `--resume` with unchanged configuration;
edited `pilot_audit_labels.csv` files are preserved.

### 3. Complete independent relevance validation — before excluding comments

- In the reviewed snapshot, finish the 193 blind annotations without seeing
  model scores. Supply the five outstanding training negative subtypes and
  collect independent annotations for the 105 second-reviewer rows. Preserve
  the independent responses before reconciling disagreements. If a second
  reviewer is unavailable, record the limitation and keep exclusion provisional.
- Check IDs, source hashes, split separation, language, and semantic duplicate
  risks. Prespecify handling of unsupported languages and long inputs. Never
  discard difficult examples after seeing their predictions.
- Evaluate the locked saved model at 0.20 once on completed blind labels. Report
  relevant recall, precision among rejections, confusion counts, rejected-row
  support, exact-rule comparison, and video-aware uncertainty. Prevalence applies
  to the eligible reserved-video population, not every source comment.
- Use the existing provisional targets of at least 98% relevant recall and
  95% rejection precision. A point estimate alone is insufficient: review its
  interval and number of rejected examples against the prespecified evidence
  policy. Add a separate hard-negative stress test when support is sparse.
- Treat unseen-make performance as a separate test with makes absent from
  fitting; training title-make segments do not establish it. If errors require
  a new threshold, label definition, or model, use development data and a fresh
  held-out test for the next version. Do not tune repeatedly on round 01 blind.

**Deliverable:** an independent validation report and explicit release decision.
If support or quality is inadequate, continue preserving comments in review
mode and add targeted training labels. Completing all 221 training rows is not
a prerequisite for fitting candidate v1. `relevance_candidate.py evaluate` tests
the fixed cutoff without a threshold sweep and writes a receipt in the relevance
round directory, preventing reuse by another candidate/output name. An incomplete
blind file does not consume the test. Missing independent review requires the
explicit `--reviewer-unavailable` flag and is recorded as a limitation.

Candidate v1 prespecifies 100 relevant examples and 50 rejections as minimum
support, plus the 98% recall/95% rejection-precision targets. It reports
approximate video-bootstrap and Wilson 95% intervals, and requires both lower
bounds to meet the targets for its statistical screen. Wilson treats comments
as independent and prevents a perfect small sample from getting a zero-width
interval; neither interval proves a population guarantee. Statistical success
does not enable exclusions: reviewer reconciliation, language/semantic-duplicate
audits, pilot audit, and unseen-make testing remain manual release requirements.

### 4. Label make/opinion/aspect templates — initial preparation complete

- Start with about 100 human-confirmed relevant comments to test the annotation
  format, using current positive labels as seed examples. The 175 Toyota
  positives may supply additional seed candidates after ID/text deduplication;
  neither pool is a representative held-out sentiment test.
- Extend the sample to roughly 300–500 development comments across makes and
  videos as labeling becomes available. Include rare aspects and unresolved
  references. Reserve fresh videos for a separate sentiment holdout before
  model development; keep the relevance blind sample reserved for its own test.
- Label canonical target `Make`, opinion presence, aspect, polarity, evidence
  quote, and reliability-claim basis. Use the existing nine-aspect guide and
  independently review a subset. A multi-make comparison can produce more than
  one target record. Title context supports clear references but never assigns
  the target automatically. Keep ambiguous/unresolved targets explicit.
- A vehicle question or factual statement can be relevant with no opinion.
  Absent aspects stay absent; they are not neutral. A missed relevance prediction
  must not keep a human-confirmed relevant comment out of this development set.

**Deliverable:** versioned target/opinion/aspect annotation templates, guide,
split manifest, and checked human labels. Template preparation needs no new
labels; supervised sentiment fitting does. `prepare_aspect_round.py prepare`
creates a development sample with one target placeholder and nine aspect rows
per selected comment. Supplying the Toyota sample and confirmed feedback permits
the starter sample to reach 100 comments from existing positive labels.
`check-labels` verifies partial annotations and exports resolved long-form
human labels. These are development labels, not a new held-out sentiment test.

### 5. Train and evaluate sentiment models — after aspect labels exist

Build separate target-resolution, opinion, aspect-detection, and target/aspect
polarity stages. Compare simple local baselines with multi-label SetFit for
aspect categories, then a classifier conditioned on `(comment, Make, aspect)`
for polarity. SetFit supports multi-label heads
([documentation](https://huggingface.co/docs/setfit/how_to/multilabel)); this
does not make the current binary relevance model a sentiment model. The
span-oriented SetFitABSA workflow is an optional later experiment
([documentation](https://huggingface.co/docs/setfit/how_to/absa)).

Keep all records and sentences from a comment/video in one partition. Evaluate
on held-out videos and separate unseen-make tests, report support and errors per
aspect/polarity, and audit sarcasm, negation, mixed opinions, and target swaps.
Publish only aspects with adequate held-out support; leave uncertain or
unsupported outcomes unresolved. Expand labels where errors identify a useful
gap, rather than treating an initial annotation target as sufficient for every
aspect. Test the combined pipeline as well as each component.

**Deliverable:** versioned sentiment models and a report showing which
make/aspect predictions are supported.

### 6. Scale and aggregate — after the relevant validation gates pass

Implement resumable batch inference with unique versioned comment keys and
separate long-form relevance, target, and aspect outputs. Keep `accept`,
`reject`, and `review_or_pass` as operational decisions; an accepted comment
still needs target/opinion assessment. Automatic rejection starts only after
independent validation and the pilot audit support it. Use `review_or_pass` for
uncertain cases, with no forced sentiment assignment.

Build the already-planned `(Make, aspect, as_of_date, window)` feature table for
cumulative and trailing 12-month windows. Include distinct comments/videos,
polarity counts, equal-video-weighted sentiment, reliability evidence basis,
and low-support indicators. Start by assessing the proposed suppression floor
of 20 opinions across five videos. Unsupported coverage stays null. Enforce
publication cutoffs and retain model/source versions.

**Deliverable:** the four Parquet tables listed below, with reproducible
validation and support reports. Current comments first support descriptive
analysis or recommendations; historical price-model integration waits for a
verified price-observation timestamp and a valid temporal join.

### Implementation order and commands

The candidate fitter, pilot scorer/audit sampler, frozen-candidate blind evaluator,
and target/aspect annotation preparer/checker are runnable now. Follow the
[next-stage commands](RELEVANCE_ROUND_RUNBOOK.md#save-the-candidate-and-start-the-pilot)
to fit, pilot, and prepare templates. The next coding work after reviewing these
outputs is independent-audit support, a fresh sentiment holdout, sentiment
benchmarks, and then batch scoring/aggregation. Collection, original preparation,
and the completed three-model comparison do not need a rerun to start.
Implementation stays on `main` and preserves inputs and prior outputs.

When ready to label, work in `outputs/relevance_round_01_reviewed`: unfinished
training rows currently start at spreadsheet row 181 and blind rows at 109
(header is row 1); always join by `comment_id`. Enter explicit `yes`, `no`, or
`unclear` for new annotations. Leave unreviewed rows blank and preserve frozen
`*_sample.csv` files. An existing check-labels command is available after the
subtype issues are fixed; choose a fresh progress filename for each check:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_relevance_round.py check-labels --round-dir YOUTUBE\outputs\relevance_round_01_reviewed --output YOUTUBE\outputs\relevance_round_01_reviewed\label_progress_next_01.json
```

It reports annotation readiness; it does not fit the candidate or evaluate
its predictions on blind labels. See the [runbook](RELEVANCE_ROUND_RUNBOOK.md)
for existing commands and the label-editing rules.

## Goal and current evidence

Turn review-video comments into reusable, auditable features for car models and
recommendations. The first published aggregate will be at `Make` level. A
comment about a vehicle may be a question, a fact, an opinion about the video’s
make, or an opinion about a competing make; these are different labels.

- `Youtube_Comments.parquet` contains 1,981,686 comments from 4,224 videos.
  All collected videos are car reviews, so there is no video eligibility gate.
- The Toyota sample has 175 confirmed vehicle-relevant and 65 confirmed
  non-vehicle comments. Forty negatives are exact filler candidates; only 25
  are non-filler negatives. The 240 labels are useful for a pilot, but are not
  representative of the full comment pool or other makes.
- Five-fold, video-grouped character TF-IDF validation kept 171/175 relevant
  comments but rejected only 2/25 non-filler negatives at a 0.5 threshold.
  The exact filler rule rejected all 40 sampled filler negatives. See
  `outputs/TOYOTA/relevance_baseline_v1/relevance_cv_metrics.json`.

## Decisions that apply to every stage

1. **Vehicle relevance** means the comment itself discusses a car, ownership,
   attributes, operation, costs, reliability, or a comparison. Vehicle questions
   and factual statements are relevant. A review title alone cannot make
   “great video” relevant.
2. **Make attribution** is separate. Keep the canonical vPIC `Make`; use the
   title only to resolve clear references such as “this car.” Permit multiple
   makes when a comment compares them. Preserve `unresolved` and `ambiguous`
   cases instead of assigning the video’s make by default.
3. **Opinion** is separate from relevance. A relevant question with no stance
   has `opinion_expressed=no`, not neutral sentiment.
4. **Aspect polarity** belongs to one `(comment_id, Make, aspect)` record.
   Multiple aspects or makes can occur in a comment. Use `positive`,
   `negative`, `neutral`, or `mixed` only when an opinion about that target and
   aspect is present. Preserve the supporting phrase.
5. **Reliability language is anecdotal evidence.** Mark claim basis as
   firsthand owner, secondhand, speculation, or unclear. Do not call a
   negative comment a measured failure rate or merge it with NHTSA complaints.
6. Keep all source text and identifiers unchanged. Every inferred label must
   carry its model version, label-schema version, source snapshot, and score.
   Join labels by `comment_id`, never by spreadsheet row number.

## Phase 1 — Build labels that test the real problem

1. Audit the source for empty text, exact/near duplicate text, language,
   comment length, publication dates, video concentration, and title-to-make
   coverage. Keep a reproducible source manifest. `author` is a display name,
   not a verified person ID.
2. Preserve the 240 Toyota labels. Draw an additional **representative blind
   test set** of roughly 200–300 comments across makes, videos, and years.
   Label it before model selection and do not use it for training, threshold
   tuning, or active learning. Record the true relevant/irrelevant prevalence.
3. Draw a separate **training pool** of roughly 200–400 non-filler comments
   across makes. Prioritize presenter praise, spam, generic reactions, ambiguous
   pronouns, short car remarks, cross-make comparisons, and cases near the
   current model’s decision boundary. Ensure at least about 100 additional
   non-filler negatives; add batches if that target is not reached.
4. Double-label 10–20% of new rows, resolve disagreements, and keep `unclear`
   as a real label. Freeze a label guide with examples. Use video groups and
   normalized-text hashes to prevent near-duplicate leakage between training
   and evaluation.

**Gate:** report a representative prevalence estimate, counts by make and
negative subtype, disagreement rate, and enough hard negatives to judge any
proposed automatic exclusion. If the blind set has too few negatives, add a
separate hard-negative stress test; do not use that enriched set to estimate
real-world prevalence.

## Phase 2 — Choose a local relevance classifier

1. Keep the exact empty/URL-only/creator-phrase prefilter as an audited rule,
   with false-skip checks. Apply it before expensive inference only after its
   precision is confirmed on fresh data.
2. Compare three local candidates on identical video-grouped splits:
   character TF-IDF + logistic regression (existing baseline), sentence
   embeddings + logistic regression, and [SetFit](https://huggingface.co/docs/setfit/main/index)
   with a small English sentence encoder. Start with CPU-capable baselines;
   measure the extra accuracy and inference cost before using a larger model.
   Start with comment text as input; test video-title context separately on
   ambiguous references, ensuring the title alone never turns creator praise
   into a vehicle comment.
3. Evaluate on unseen videos and, separately, unseen makes. Report relevant
   recall, precision of rejected comments, confusion counts for hard
   non-filler negatives, calibration, and uncertainty intervals. A model that
   only rediscovers the exact filler rule is not a useful next gate.
4. Choose the exclusion threshold from validation data, then check it once on
   the untouched blind test. Provisional release targets are at least 98%
   recall of relevant comments and at least 95% precision among auto-rejected
   comments, with error counts and confidence intervals shown. If the evidence
   is too thin to support those claims, keep the model in review-only mode.
5. Use three outcomes: `accept`, `reject`, and `review_or_pass`. Only
   high-confidence negatives are rejected. Ambiguous text stays available for
   make/opinion analysis or human review. Keep score and reason for every row.

**Gate:** the selected model must improve non-filler rejection over exact rules
without materially losing relevant vehicle comments on the blind test. Lock
its model artifact, preprocessing, threshold, and dependency versions.

## Phase 3 — Pilot and batch qualification

1. Run a 10,000-comment pilot spanning makes, dates, text lengths, and videos.
   Audit random accepted, rejected, and uncertain examples, including comments
   about a different make than the title. Benchmark rows/second and memory.
2. Read source Parquet in chunks; score each `comment_id` once per model
   version; checkpoint completed chunks and support restart without duplicate
   output. Never modify `Youtube_Comments.parquet` in place.
3. Publish a per-comment qualification table with `vehicle_relevance_score`,
   `relevance_decision`, `prefilter_reason`, and provenance. Recheck label mix
   and rejection rate by make, video, and month for drift.

**Gate:** sampled production errors and distribution shifts remain within the
validated bounds before proceeding to make-level sentiment.

## Phase 4 — Resolve the make and label opinion/aspects

1. Build a versioned make-alias dictionary from canonical vPIC `Make` values.
   Match video titles as candidate context, then inspect comment text. A
   comparison can produce two `(comment_id, Make)` targets; a relevant comment
   with no defensible make remains useful in the qualified dataset but cannot
   enter a make aggregate. Save each comment's `target_resolution_status` as
   `resolved`, `ambiguous`, or `unresolved`, including comments with no target
   row.
2. Extend the label sample with target-make, `opinion_expressed`, aspect,
   polarity, evidence phrase, and reliability basis. Start with roughly
   300–500 relevant comments across makes, oversampling rare aspects such as
   reliability and safety for training. Maintain a separate representative
   held-out set for prevalence and final evaluation. Double-label a subset.
3. Use the nine aspect categories in the existing label guide:
   `reliability`, `maintenance_cost`, `price_value`, `fuel_economy_range`,
   `performance`, `comfort`, `safety`, `technology`, and `design`. Keep an
   optional `overall` opinion separate from these aspect fields if later needed.
4. Benchmark a social-text polarity model such as
   [CardiffNLP Twitter RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
   only as an **overall-sentiment baseline**. It does not solve target or aspect
   attribution. For the working pipeline, compare a few-shot
   [multi-label SetFit](https://huggingface.co/docs/setfit/how_to/multilabel)
   aspect detector with a small local encoder, followed by a polarity
   classifier conditioned on `(comment, Make, aspect)`. The
   [SetFitABSA](https://huggingface.co/docs/setfit/how_to/absa) span method is
   an additional experiment for explicit aspect phrases; implicit comments
   still need category-level labeling. This separation follows the standard
   aspect-based sentiment task design in
   [SemEval-2014 Task 4](https://aclanthology.org/S14-2004/).
5. Treat `no_opinion`, `neutral`, and `mixed` separately. Split long comments
   into sentences for evidence, but preserve the original comment-level ID and
   avoid counting the same comment twice for one aspect. Audit negation,
   sarcasm, comparisons, and first-person reliability claims explicitly.

**Gate:** on held-out videos/makes, report target-make precision/recall,
opinion confusion, per-aspect precision/recall/F1, polarity confusion, and
reliability-basis agreement. Publish an aspect only when its audited errors
are acceptable; keep unsupported or uncertain predictions null. Validate
probability calibration before using scores as aggregate weights
([scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)).

## Phase 5 — Publish reusable make-level features

1. Save long-form comment/aspect records first. Aggregate by canonical `Make`,
   `aspect`, and monthly `as_of_date`, for both a trailing 12-month window and
   a cumulative window. Enforce `published_at <= as_of_date` and record the
   exact window boundaries.
2. Publish distinct supporting comment and video counts; positive, negative,
   neutral, and mixed opinion counts; positive/negative shares; and net
   sentiment `(positive - negative) / opinion_count`. Publish per-video means
   and an equal-video-weighted make score so one popular review cannot dominate.
   Store an approximate distinct-author count separately because source
   `author` is only a display name.
3. For reliability, publish separate firsthand-negative and other-basis
   counts/shares. Add coverage, uncertainty, and a minimum-support indicator.
   A starting suppression rule is fewer than 20 opinion comments or fewer
   than five videos for a `(Make, aspect, window)` cell; review the actual
   coverage before fixing this rule. Missing or suppressed is **null**, never
   neutral or zero sentiment.
4. Keep a long feature table for agents and analysis. Pivot selected numeric
   features to one row per `(Make, as_of_date, window)` when a conventional
   model requires wide inputs. Preserve `ModelYear`, `Make`, `Model`, and
   `Trim` as separate columns if those features are later joined to vehicle
   cohorts; the opinion remains a make-level signal.
5. Validate row counts, unique keys, time cutoffs, score ranges, support,
   video concentration, model versions, and spot-check evidence quotes against
   source text. The feature build must be reproducible from source snapshot,
   labels, models, thresholds, and aggregation configuration.

**Time rule:** the current used-car price dataset is historical, while comments
extend into 2026. Do not train a historical price model with comments written
after the relevant price observation. The present `listed_date` alone has not
been verified as that observation timestamp. Make-level features can first
serve current analysis or an as-of-date recommendation system; price-model use
waits for a defensible temporal join.

## Parallel expert-review baseline

The supplied DougScore CSV now supplies a separate expert baseline alongside these
community features. Its audit found 585 reviews with valid category scores
and consistent totals. The source has no separate Trim, review publication,
or score-version date columns, and the exported video references are
time-formatted text rather than usable URLs. The audited snapshot is imported:
585 reviews and 2,925 proposed mapped aspect records are in
`EXPERT_REVIEWS/outputs/dougscore_v1/`. Initial vPIC coverage has 308 reviews
with text candidates and 277 without; all 863 candidate rows remain unapproved.
Source identity columns are Year, Make, Model; full Model text is retained.
Video linkage and filming/country analysis are deferred at the user's request.
No community scores, relevance labels, or sentiment targets are inferred from
expert ratings.

Follow [the expert integration plan](../../EXPERT_REVIEWS/DOUGSCORE_INTEGRATION_PLAN.md)
for source preservation, a verified vehicle/video crosswalk, coverage
measurement, and a paired-review experiment. Compare design, performance,
comfort, technology (partial), and price/value where attribution and support
are adequate. Keep expert Quality, Practicality, and Cool Factor separate;
they do not supply dedicated reliability, fuel-economy, or safety labels.

The current relevance fit, qualification pilot, and structured aspect
templates remain the next executable steps. Expert scores do not replace
human polarity labels or sentiment holdouts. Pairing a reviewed vehicle with
community opinion requires validated model/variant comment targets beyond
the current make-only contract. Same-review audiences and score revisions
must be identified before treating expert/community agreement as independent
evidence or using scores for historical price prediction.

## Planned artifacts and first execution order

These are proposed names and are **not implemented yet**:

| Artifact | Grain | Purpose |
| --- | --- | --- |
| `youtube_comment_relevance.parquet` | one `comment_id` | Qualification decision, score, rule reason, target-resolution status after Phase 4, and model/source versions. |
| `youtube_comment_targets.parquet` | one `(comment_id, Make)` | Confirmed make attribution and opinion status; unresolved comments have no target row. |
| `youtube_comment_aspect_sentiment.parquet` | one `(comment_id, Make, aspect)` | Polarity, score, evidence phrase, reliability basis, versions. |
| `youtube_make_sentiment_features.parquet` | one `(Make, aspect, as_of_date, window)` | Counts, video-balanced sentiment, support, uncertainty, suppression status. |

Start with the runnable Phase 1 preparation:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_relevance_round.py prepare --output-dir YOUTUBE\outputs\relevance_round_01
```

Review the audit and manifest before labeling. Fill the relevance fields in the
training/blind CSVs and have a second reviewer label their separate subset.
Then validate the labels without exposing blind class counts during selection:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_relevance_round.py check-labels --round-dir YOUTUBE\outputs\relevance_round_01 --output YOUTUBE\outputs\relevance_round_01\label_progress_01.json
```

The new round preserves the existing
[`comment_label_sample.csv`](../outputs/TOYOTA/comment_label_sample.csv) and
[`relevance_cv_predictions.csv`](../outputs/TOYOTA/relevance_baseline_v1/relevance_cv_predictions.csv).
`toyota_error_review.csv` extracts the four missed relevant comments and 23
non-filler negatives that the baseline kept, with a blank `review_notes` field.
The new CSV label schema uses `vehicle_relevant=yes/no/unclear`; it is not
directly accepted by the older `evaluate_comment_relevance.py`, which reads
the original pilot's `comment_topic`. Keep that diagnostic scoped to Toyota
until the multi-candidate evaluation stage is implemented.
The current runnable diagnostic is:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\evaluate_comment_relevance.py --sample YOUTUBE\outputs\TOYOTA\comment_label_sample.csv --output-dir YOUTUBE\outputs\TOYOTA\relevance_baseline_next
```

Use a fresh `--output-dir` because the diagnostic preserves existing results.
No full-data scoring or feature-aggregation command exists yet; implement each
phase only after its preceding gate passes.
