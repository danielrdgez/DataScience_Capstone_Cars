# Experimental YouTube sentiment features with DougScore

Decision: October 1, 2026. The user has deferred new labeling. This is the
preserved legacy prototype route; the supervised validation route remains documented
in MAKE_SENTIMENT_PIPELINE_PLAN.md for later use. The pipeline is implemented in `../sentiment_pipeline.py`; runnable commands
and remaining work are in [the README](../../README.md#previous-youtube-sentiment-benchmark-workflow)
and [next steps](../NEXT_STEPS.md). This document retains the design rationale. Existing labels, source tables, models, and expert
snapshots remain intact. No blind test is consumed by this route.

**Current route (October 6, 2026):** new raw-source runs use
[the GPU text runbook](GPU_TEXT_RUNBOOK.md). Full-source students and make features
are complete, and the selected make export is available to the retrospective
price comparison. This plan preserves the earlier nine-aspect benchmark design.

## Objective and available inputs

Build reproducible, experimental community features from YouTube comments
without requiring additional human labels. Use DougScore to choose the first
comparable aspects and supply separate expert features. Do not infer comment
polarity from an expert rating or train the comment classifier on those ratings.

Available: 1,981,686 source comments from 4,224 videos; saved SetFit relevance
candidate v1; 10,000 scored pilot comments from 1,496 videos; 585 imported expert
reviews and 2,925 mapped expert-aspect rows. The pilot is enriched and is useful
for engineering checks, not population estimates. The 863 vPIC candidate rows
are unapproved matches; 308 reviews have candidates and 277 have none.

## 1. Build a small local inference prototype

Start with a deterministic 1,000-comment subset of the existing pilot, balanced
across videos and relevance score bands. Record sampling probabilities and
limits. Preserve all source text and scores. Do not read protected blind labels
or use protected blind comments/videos for development. Do not rerun the saved
fit or overwrite existing outputs. Expand to the 10,000-comment pilot after
measuring throughput, token lengths, memory, and output consistency.

Use the existing relevance score as an experimental eligibility signal. Keep
every input row and compare downstream aggregates for cutoff 0.20 and 0.50,
plus a view without the model cutoff. Flag empty/URL-only/exact filler text.
No source deletion occurs. Scores are uncalibrated: do not use relevance
probabilities as statistical weights or call the resulting filter validated.

Use local pretrained inference rather than fit a new sentiment model:

- Candidate topic/opinion model: facebook/bart-large-mnli, with explicit
  hypotheses for vehicle opinion versus creator praise, questions, facts,
  comparisons, and the nine existing aspects. It supports zero-shot
  classification and independent multi-label scores. Benchmark its speed
  on the small subset before committing to source-wide inference.
- Overall tone baseline: cardiffnlp/twitter-roberta-base-sentiment-latest.
  It supplies English negative/neutral/positive scores. Apply it to short
  evidence clauses, preserving sentence offsets and parent comment IDs.
  Its neutral output does not establish that the text contains an opinion.
- Experimental aspect polarity: score target/aspect-specific positive and
  negative opinion hypotheses with the NLI model on the same clause. Preserve
  raw scores, a separate no-opinion hypothesis, ambiguous outcomes, and
  sentence tone. Treat conflicts as unresolved, rather than copying the
  whole comment's tone to every aspect. NLI scores are uncalibrated and its
  neutral relation label is not neutral sentiment. Positive and negative
  evidence can yield mixed; mixed never defaults to neutral.

Pin model revisions, package versions, hypothesis text, preprocessing, and
thresholds in a manifest. English is the initial scope; record unsupported or
unknown language separately. Split long comments into clauses/windows and
record truncation rather than silently discarding their ends. Diagnostic
examples cover negation, sarcasm, creator praise, questions, and comparisons;
passing these checks demonstrates software behavior, not measured accuracy.

Sources: [BART model card](https://huggingface.co/facebook/bart-large-mnli),
[CardiffNLP model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
These models are candidates for this domain, not established car-sentiment models.

## 2. Resolve targets conservatively

Match explicit comment mentions to canonical vPIC makes using token boundaries
and an explicit alias table. Use title identity as context, retaining
title_context separately from explicit_comment attribution. Never copy an
opinion about a rival make to the title vehicle. Ambiguous, multiple-target,
or pronoun-only statements can remain unresolved. In comparisons, assign
polarity only where the clause resolves the target.

Publish explicit-target make features first. Publish title-context experimental
features separately so their impact is measurable. Resolve Year/Make/Model
from titles and the DougScore/vPIC candidates only when the text supports a
unique identity, and record the method/evidence. Retain candidate status for
automated matches. Do not infer Trim or broadcast one expert review to all trims.
An exact cohort is ModelYear, Make, Model, Trim with all dimensions supported;
year/make/model and make-only signals have different scopes.

DougScore's video link, location, and country fields are outside this initial
route. Without verified review/video linkage, do not claim comments came from
Doug's own review or represent the same reviewed variant.

## 3. Start with five comparable aspects

| Community aspect | Expert reference |
| --- | --- |
| design | Styling |
| performance | Acceleration, Handling, Fun Factor; preserve components |
| comfort | Comfort |
| technology | Features; partial proxy |
| price_value | Value; descriptive only initially |

Keep the other four community aspects in the schema: reliability,
maintenance_cost, fuel_economy_range, safety. They can be experimental outputs
but have no dedicated DougScore reference. Quality is not an isolated
reliability rating. Preserve original expert categories, Daily, Weekend,
and total scores; do not manufacture community labels from them.

## 4. Build inspectable feature tables

Use separate experimental outputs; do not silently populate the previously
planned validated tables with model predictions.

| Proposed artifact | Grain |
| --- | --- |
| youtube_experimental_predictions.parquet | One comment, target, aspect; evidence list and unresolved status retained |
| youtube_experimental_video_features.parquet | One video, target scope, aspect, as_of_date, window, inference policy |
| youtube_experimental_make_features.parquet | One Make, attribution scope, aspect, as_of_date, window, inference policy |
| dougscore_experimental_reference.parquet | One source review and aspect, with candidate/approved identity status |

Deduplicate normalized exact text within each video/target/aspect for the
feature calculation; retain duplicate counts and all original prediction rows.
Count each comment once per target/aspect. Cross-video repeated text is flagged
for a sensitivity view, not silently removed from the source. Do not use display
names as verified author IDs.

Per cell publish positive, negative, neutral, mixed counts; resolved opinion
count; distinct comments/videos; unresolved share; duplicate share; language
coverage; and truncation share. Net sentiment is
(positive_count - negative_count) / resolved_opinion_count, including resolved
neutral and mixed opinions in that denominator. No-opinion and unresolved
records are outside it. Compute each video's net then average equally across
videos with eligible opinions. This video-balanced net is the primary feature;
also retain comment-weighted net and across-video spread.

Build monthly cumulative and trailing-12-month windows using published_at <=
as_of_date. Suppress sentiment where fewer than 20 resolved opinion comments
or five videos support a cell; retain coverage counts. These are initial
coverage heuristics, not accuracy guarantees. Missing, unsupported, and
suppressed scores are null. Candidate wide features include
yt_design_video_balanced_net, yt_performance_video_balanced_net,
yt_comfort_video_balanced_net, yt_technology_video_balanced_net,
yt_price_value_video_balanced_net, plus per-aspect video/opinion counts and
missing indicators. Keep method=pretrained_experimental and validation_status
on every output.

## 5. Compare with the expert reference

Report coverage first. Pair only compatible target scopes and supported
identities; candidate matches stay in diagnostic views. Make-level DougScore
averages describe its selective reviewed-vehicle sample, not every vehicle
of that make. Do not compare make sentiment with one vehicle as if their
populations were identical.

Compare supported aspects through ranks, distributions, and disagreement
examples. If an expert display scale is needed, (score - 5.5) / 4.5 aligns
1..10 to -1..1, but does not calibrate ordinal scores to sentiment. Do not
average expert and community scores or tune sentiment thresholds to maximize
DougScore agreement. Agreement is external consistency evidence, not accuracy.
Keep Daily/Weekend/category predictors separate, and defer price-informed
Value from the first price-prediction experiment.

## 6. Scale and test feature usefulness

After engineering checks, benchmark a fresh video-balanced source sample
before the full 1.98-million-comment run. The enriched pilot must not supply
final population features. Extend protected-sample-aware relevance scoring
and pretrained inference with immutable input hashes, resumable batch shards,
and fresh output directories. Aggregate directly from the full dated snapshot.
Keep development/protected evaluation flags for any later classifier fitting.

Begin with dated descriptive analysis. For price experiments first establish
the listing price-observation cutoff and expert score availability; current
comments can be newer than historical listings. Without that evidence, do not
claim historical predictive improvement. A pretrained model's own training
dates must also be documented for strict historical simulation.

Once timing and identity are defensible, compare existing predictors, existing
plus expert features, existing plus community features, and existing plus both
on identical VIN samples and splits. Listings decode through lowercase vin;
retain four separate cohort dimensions. Join make-level features by canonical
Make and the applicable as-of date, clearly retaining make-level scope.
Choose configurations on validation dollar MAE and evaluate the held-out
price test once. Include support-only ablations to distinguish sentiment gain
from coverage/source selection. Keep missing-score handling identical.

## Implementation order and completion conditions

1. Implement a fresh 1,000-comment local pretrained inference prototype.
2. Expand to 10,000 pilot comments; test cutoffs, attribution policies, runtime,
   score ranges, unique keys, evidence spans, dates, and resume equivalence.
3. Implement experimental video/make aggregates and expert comparison views.
4. Benchmark a fresh source sample, then run resumable full-data inference.
5. Implement price feature joins and paired ablations only after time evidence.

No new manual labels are prerequisites for steps 1–4. Their outputs remain
experimental. The commands `prepare`, `source`, `infer`, `aggregate`, `expert`, and `compare`
now implement these stages. Neural runs, engineering review and price integration
remain pending; see [next steps](../NEXT_STEPS.md).
New labeling and independent sentiment evaluation can improve this route
later without blocking the requested prototype. No full inference, model
download, expert match approval, or price training is performed by this plan.
