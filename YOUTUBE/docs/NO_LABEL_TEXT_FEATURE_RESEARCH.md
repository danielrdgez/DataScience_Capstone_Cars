# YouTube text features without new manual labels

**Implementation update (October 6, 2026):** the semantic/topic and reduced
sentiment route reviewed here is implemented in `YOUTUBE/text_pipeline.py`.
Full-source scoring and aggregation are complete, and the selected make view is
available to the retrospective price comparison. See
[the GPU runbook](GPU_TEXT_RUNBOOK.md) for the measured coverage and limitations.
The review below retains its original research date and experimental rationale.

Research and local artifact review: **2026-10-05**. These are proposed experiments,
not a replacement pipeline or changes to the six model-development Parquet tables.
Searches used SciSpace and Consensus; primary papers and model cards were checked
where accessible. No annotation files were edited, no protected blind labels were
opened, no model was fitted, and no new neural inference was run for this review.

## Recommendation

Benchmark two feature families on the same eligible source comments:

1. **Semantic/topic features:** sentence embeddings or TF-IDF/SVD, plus proportions
   of comments about reliability, maintenance, performance, comfort, technology,
   design, fuel economy/range, and safety. These need no comment sentiment labels.
2. **Targeted sentiment:** pretrained or seed-guided extraction of opinions about
   a supported make and aspect. Retain unknown, no-opinion, and mixed outcomes.

Use make-level aggregation because videos do not support generalizing to every
`ModelYear`, `Make`, `Model`, `Trim` combination. Call the output a **make-level
review-audience signal**: pooling model-specific review comments does not establish
the attitudes of all owners or buyers of that make.

The first benchmark should compare a cheap embedding/topic route with a reduced
pretrained sentiment route. JASen and Snorkel are useful second-stage research
options if corpus adaptation is needed. Do not start by running the current
nine-aspect BART pipeline across all 1.98 million comments.

**Use DougScore as the professional expert-opinion baseline.** Preserve its
original ratings and compare compatible expert aspects with the separately
estimated YouTube audience features. Include expert-only and expert-plus-community
price comparisons after identity and timing are defensible. The expert baseline
does not require manually scoring or labeling YouTube comments.

There is no need to manufacture a sentiment training target for every comment.
Pretrained models supply representations or inferred scores; **listing `price` is
the supervised outcome for the downstream price model**. A separate problem is
identifying the vehicle make that a comment actually discusses.

## What the available data supports

The read-only source check confirmed the documented 12-column schema:
`video_id`, `playlist_id`, `video_title`, `source`, `text`, `extracted_at`,
`comment_id`, `author`, `like_count`, `reply_count`, `published_at`, `updated_at`.

| Observed source property | Result | Consequence |
| --- | ---: | --- |
| Comments / unique comment IDs | 1,981,686 / 1,981,686 | Preserve `comment_id` as the identity key. |
| Videos / playlists | 4,224 / 59 | Videos are the available aggregation units; playlists are not verified channel IDs. |
| Empty or whitespace-only text | 94 | Cheap deterministic exclusion is straightforward; most noise is semantic. |
| Median / 95th percentile / maximum text length | 75 / 402 / 9,964 characters | Short-text uncertainty and long-text segmentation both matter. |
| Publication dates | 2012-01-24 to 2026-08-26 | Comments cannot all be used for historical listing prices. |
| Publication dates parsing as MM-DD-YYYY | All rows | Parse explicitly; preserve original date strings. |
| Comments published through 2020-12-31 | 675,792 | A preliminary date-eligible pool exists; it is not a verified historically available dataset. |
| Later update date than publication date | 7,920 | Publication date alone does not establish availability of the saved text. |
| Extraction dates | 2026-06-04 to 2026-08-26 | Engagement values and surviving comments reflect a later extraction. |
| Comments per video: median / 95th percentile / maximum | 260 / 1,141 / 1,515 | Raw comment pooling gives heavily commented videos more influence. |

The source has **no sentiment gold labels, ratings, canonical comment targets,
verified owner status, stable author-channel IDs, video publication dates, video
transcripts, or reply text/thread structure**. `reply_count` is only a count.
`source=comment` does not imply vehicle relevance. Display names are not verified
person identities. Methods needing star ratings, purchase verification, audio,
video content, or complete conversation threads do not directly fit this schema.

The current collector defaults to relevance ordering and permits unlimited
comments. The Parquet lacks the original collection-limit/order history, so do
not assert that the imported source is exhaustive or a random audience sample.

## What has already been tried locally

The saved manifests show completed pretrained inference for **1,000 and 10,000
comments**, followed by video/make/wide feature generation. The 10,000-comment CPU
run recorded **64,032.7 inference seconds (17.8 hours)**. A simple linear projection
to all source rows is about **147 days** of equivalent CPU inference; to the
675,792 date-eligible rows it is about **50 days**. These are workload illustrations,
not runtime forecasts: source text mix, exclusion policy, batching, hardware, and
parallelism can change cost substantially.

The 10,000-comment input is an **enriched engineering pilot**, not a representative
sample for final make features. Its observed outcomes are diagnostic:

- 3,730 distinct comments have at least one resolved positive, negative, neutral,
  or mixed aspect opinion.
- Language detection records 7,931 English, 1,906 unknown, and 163 unsupported
  comments. The current detector marks text with fewer than 20 ASCII letters
  unknown; this can lose useful short opinions.
- In the cumulative 2026-10-01 view without a relevance cutoff, the current
  20-opinion/five-video support rule yields **22 explicit-attribution make/aspect
  cells across six makes**, versus **322 title-context cells across 42 makes**.
  These are different attribution policies, not independent accuracy estimates.
- The saved relevance candidate was trained on prior human labels. The `source`
  command requires that candidate and its frozen source hash. Therefore the
  current full-source route avoids *new* manual labeling but is not completely
  independent of previously labeled training data.

Keep these runs as benchmarks. A new route can read raw source comments without
requiring the saved supervised relevance candidate. Preserve its artifacts and
protected evaluation exclusions when making comparable development runs.

## Research options and fit

| Option | Supervision needed from this project | Proposed output | Main limitation | Priority |
| --- | --- | --- | --- | --- |
| Sentence embeddings or TF-IDF/SVD | None | Small semantic vector per make/date; topic similarities | Similarity is not sentiment or calibrated relevance; static make vectors are redundant with `Make` | First baseline |
| Pretrained zero-shot relevance + targeted tone | Class descriptions and rules, no annotated comments | Relevance/opinion flags and make/aspect sentiment | Domain transfer, attribution errors, and repeated inference cost | First sentiment comparator |
| BERTopic + pretrained sentiment | None for topic discovery; optional topic naming | Topic prevalence and topic-associated tone | Unstable clusters, short-comment noise, and topics that describe creators rather than cars | Exploratory comparator |
| JASen seed-guided aspect/sentiment | A small aspect/sentiment vocabulary, no labeled comments | Corpus-adapted aspect/polarity predictions | Original evidence concerns restaurant/laptop reviews; target linking remains separate | Second-stage adaptation |
| Snorkel weak supervision | Programmatic rules or pretrained signals, no per-comment labeling | Pseudo-labels and a smaller relevance/opinion classifier | Correlated weak sources can share mistakes; pseudo-labels are not ground truth | Second-stage scale option |
| Structured zero-shot LLM extraction | Instructions, supported make list, and evidence schema | Make, aspect, opinion, polarity, exact evidence spans | Hallucinated attribution, prompt sensitivity, expense, and model-dependent quality | Bounded benchmark first |
| VADER | None; uses a published lexicon | Fast clause-tone baseline | Does not solve relevance, target attribution, sarcasm, or automotive meanings | Cheap baseline only |

For example, JASen seeds could include `reliability: reliable, breakdown, failure`,
`maintenance_cost: repair, service, maintenance`, and `comfort: seat, ride, noise`.
Selecting seeds defines the task; it does not require scoring individual comments.
Words such as `sick`, `cheap`, and `unbelievable` are context-dependent, so seed
matches alone must not be treated as reliable polarity labels.

## Papers to use in the capstone

Each proposed adaptation below is a project recommendation, not a result measured
on these YouTube comments. Benchmark results from other domains do not establish
in-domain accuracy or a price-model improvement.

| Paper | Evidence and practical use | Fit boundary |
| --- | --- | --- |
| Yin, Hay & Roth (2019), [Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach](https://aclanthology.org/D19-1404/) | Formulates classification using textual entailment without task-specific training examples. Supports hypotheses for vehicle content, opinion, creator praise, facts, and questions. | Label wording and domain shift need sensitivity checks; entailment neutral is not neutral sentiment. |
| Huang et al. (2020), [Weakly-Supervised Aspect-Based Sentiment Analysis via Joint Aspect-Sentiment Topic Embedding](https://aclanthology.org/2020.emnlp-main.568/) | JASen learns from an unlabeled corpus with a few aspect/sentiment seed words. Direct methodological match to avoiding new manual labels. [Author implementation](https://github.com/teapot123/JASen). | Evaluated on restaurant/laptop reviews; it does not automatically identify automotive makes or solve comparisons. |
| Reimers & Gurevych (2019), [Sentence-BERT](https://aclanthology.org/D19-1410/) | Produces reusable sentence representations for semantic similarity and clustering. Supports topic prototypes and pooled make vectors without sentiment targets. | The proposed pooled price features are our adaptation; embeddings are not an opinion probability. |
| Grootendorst (2022), [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) | Combines document embeddings, clustering, and topic descriptions. Useful for discovering what eligible comments discuss. | Preprint methodology; clusters can capture presenter catchphrases and video-specific topics. Fit learned transformations only on the designated training/development text. |
| Ratner et al. (2020), [Snorkel: rapid training data creation with weak supervision](https://link.springer.com/article/10.1007/s00778-019-00552-1) | Combines noisy labeling functions rather than manually labeling each record. Supports a compact classifier trained from rules and pretrained signals. | Needs meaningful weak-source diversity and abstention. Several prompts to the same model are not independent evidence. |
| Meng et al. (2020), [Text Classification Using Label Names Only](https://aclanthology.org/2020.emnlp-main.724/) | LOTClass learns categories from label names and unlabeled text using self-training. Supports a possible corpus-adapted relevance/topic classifier. | Optional alternative to Snorkel; broad label names can be hard to separate in short mixed-purpose comments. |
| Hutto & Gilbert (2014), [VADER](https://ojs.aaai.org/index.php/ICWSM/article/view/14550) | Published rule-based sentiment method for social-media text; inexpensive benchmark requiring no new training labels. | Whole-comment tone can reflect a creator or another make. Preserve negation, emphasis, and emoticons for its scoring. |
| Mathew et al. (2026), [Quantifying customer sentiment for automobile brand perception analysis using machine learning on Twitter](https://www.nature.com/articles/s41598-026-35637-9) | Automotive example of pretrained sentiment and brand-level polarity indices. Supports brand aggregation and stability/event/cross-model diagnostics. | Twitter is not YouTube. Its influence weighting needs follower/account-age and engagement fields absent here; its benchmark is not validation of our comment labels. |
| Liu et al. (2025), [Innovative Mining of User Requirements Through Combined Topic Modeling and Sentiment Analysis: An Automotive Case Study](https://www.mdpi.com/1424-8220/25/6/1731) | The abstract describes combining BERTopic and RoBERTa for automotive UGC, including performance, comfort, price, and safety topics. Relevant domain precedent for topic features. | Abstract-level evidence reviewed; publisher full text was inaccessible in this session. Do not assume its model or preprocessing transfers to English YouTube comments. |
| Smid, Priban & Kral (2024), [LLaMA-Based Models for Aspect-Based Sentiment Analysis](https://aclanthology.org/2024.wassa-1.6/) | Evaluates LLM ABSA and reports weaknesses in zero/few-shot settings relative to fine-tuning. Supports benchmarking an extractor rather than assuming it is accurate. | Its strongest fine-tuned route needs labeled task data; that route does not satisfy this project's no-new-label constraint. |

## DougScore professional-opinion baseline

The imported expert snapshot already contains **585 unique reviews**, all ten
original category scores, and their Weekend, Daily, and overall totals. A
read-only recheck found zero subtotal/total mismatches. The separate aspect table
contains 2,925 rows: five mappings per review. See
[the data dictionary](../../DATA_DICTIONARY.md#dougscore-expert-opinion-baseline)
for the actual data and definitions.

The [official methodology](https://www.dougdemuro.com/dougscore) defines ten 1–10
ratings and warns that the review selection favors enthusiast vehicles and scores
can change. It is one named expert's baseline, not a consensus of professionals.

| Expert metric | Community comparison | Baseline treatment |
| --- | --- | --- |
| Styling | `design` | Direct aspect reference, while retaining scope differences. |
| Acceleration, Handling, Fun Factor | `performance` | Keep the three scores; the existing equal-weight mean is a named derived index. |
| Comfort | `comfort` | Compare supported ride/seating/noise opinions. |
| Features | `technology` | Partial equipment proxy; not a score of software usability. |
| Value | `price_value` | Descriptive value reference; exclude from the initial price predictors. |
| Quality | No clean existing aspect match | Preserve expert build/materials/perceived-reliability judgment; do not relabel it as a failure-rate or reliability score. |
| Practicality | No clean existing aspect match | Preserve the compound usability/cargo/fuel-economy judgment. |
| Cool Factor | No clean existing aspect match | Preserve desirability/enthusiast appeal independently. |
| Weekend / Daily / DougScore totals | Broad expert summaries | Keep original totals; Daily and overall include price-informed Value. |

No dedicated DougScore metric isolates safety, maintenance cost, fuel-economy/
range, or reliability. Missing expert aspect coverage stays null, even when
YouTube comments discuss that aspect.

Use the baseline in three ways:

1. **Expert-only reference now:** distributions and rankings of original ratings
   at the reviewed-vehicle grain. Preserve raw Year/Make/Model/variant text and
   snapshot identity; no YouTube labels are needed.
2. **Expert/community comparison when identity supports it:** compare compatible
   aspect ranks at supported reviewed-variant or year/make/model scope. Where
   community support only permits a make comparison, average each expert review
   once and report review counts and reviewed-model mix. Compare this selective
   expert sample with video-balanced audience opinions as distinct populations.
   Spearman rank association and disagreement panels are descriptive consistency
   checks, not sentiment accuracy or evidence that the opinions must agree.
3. **Expert price baseline after time evidence:** test the nine non-Value category
   scores, or a small predefined subset, as a separate expert feature block. If
   summaries are desired, proposed `expert_daily_without_value = daily_total -
   value` and `expert_total_without_value = dougscore - value` must be named as
   derived indices. Do not use Daily or overall DougScore unchanged while claiming
   that price-informed Value was excluded. Non-Value ratings still need dates and
   supported identity; subtraction does not resolve selection or timing issues.

The source has no separate Trim, usable review video IDs, review publication
dates, or score-availability/version dates. Current candidate matches are
unapproved. Do not broadcast one reviewed variant to all trims or claim same-video
expert/audience evidence. Deterministic unique make matches can be proposed and
audited separately; ambiguous identities remain missing. None of this calls for
manual YouTube sentiment labels or converting an expert score into comment labels.

Retain expert and community measures separately. Do not tune NLP thresholds to
maximize DougScore agreement, blend their scales into one quality score, or treat
agreement as an independent benchmark when comments may react to Doug's review.
Any future verified same-review comments should be identified as potentially
influenced by that expert rating.

Two supplementary leads should receive lower evidential weight:

- Mathew et al. (2026), [Uncovering customer sentiment and brand perception by
  leveraging large language models: A case study in the automotive industry](https://www.sciencedirect.com/science/article/pii/S2543925126000124).
  The indexed abstract describes zero-shot automotive perception and aspect
  analysis. The publisher returned an access error, and Consensus fetch supplied
  no abstract. It is a relevant follow-up lead, not verified implementation
  guidance or a basis for choosing a particular LLM.
- Yan (2025), [Research on Second-Hand Car Price Prediction Based on the
  Integration of Sentiment Analysis and Machine Learning](https://doi.org/10.1109/ICEACE67491.2025.11439882).
  The fetched Consensus abstract describes random forests plus online sentiment,
  but does not establish usable data grain, temporal controls, splits, or a
  reproducible gain. Full text was unavailable. It motivates an experiment;
  it does not prove that these make-level YouTube features improve prices.

## Proposed preprocessing and attribution

1. **Freeze source and time policy.** Preserve raw text and identifiers. Write
   derived fields to separate versioned outputs. Parse publication/update dates;
   invalid or unavailable dates have explicit exclusion reasons.
2. **Cheap quality flags.** Flag empty text, URL-only text, repeated advertising,
   and exact duplicates. Do not reject every short comment, every link-bearing
   comment, or every comment without a make name. `Great car` can be useful with
   supported title context. Preserve emojis, negation, punctuation, and casing
   for neural/VADER scoring; use a separate normalization for duplicate detection.
3. **Separate vehicle content from opinion.** Allow vehicle opinion, vehicle
   question, vehicle fact, creator/video discussion, spam, and unresolved outputs.
   These can overlap in a comment, so classify clauses or use multiple flags.
   Facts and questions can contribute to semantic/topic features; only supported
   opinions contribute to sentiment. A question mark alone does not prove that
   there is no opinion.
4. **Resolve targets before aggregating.** Explicit supported make mentions come
   first. Use a versioned vPIC make/alias vocabulary with token boundaries and
   context checks for ambiguous names such as MINI, SMART, RAM, and GENESIS.
   Unique title identity may supply context for an implicit vehicle reference;
   it must not convert creator praise or rival-brand statements into opinions
   about the title make. Keep explicit and title-context policies separate.
   Unsupported pronouns, multiple targets, and comparisons may remain unresolved.
5. **Extract evidence at clause/target level.** For `Toyota is reliable but Nissan
   CVTs worry me`, retain distinct target evidence and avoid one tone for both
   brands. For `Great review, Alex`, retain creator discussion without car
   sentiment. An LLM output must contain exact source substrings or validated
   offsets; reject invented evidence and targets outside the canonical vocabulary.
6. **Build features with abstention.** Cache embeddings once; compare semantic
   prototypes or discover topics. Infer sentiment only for eligible opinions and
   relevant aspects rather than every aspect on every clause. Preserve raw scores
   and disagreement flags; confidence thresholds are experimental operating rules,
   not validated probabilities. Compare handling of short/unknown language rather
   than forcing it to English or neutral.

No manual comment labeling is a prerequisite for these proposed stages. External
pretrained models may themselves have been trained on labeled public data; this
does not mean that project comments need manual labels.

## Candidate feature contract

Use long aggregates keyed by `(Make, attribution_scope, as_of_date, window,
feature_method, aspect_or_topic)` and a corresponding wide model-input table.
Source comments may have several target/aspect rows, but each listing must receive
one selected wide feature row per make and eligible date. Do not join all policy,
aspect, or time rows to a listing and create duplicate listing observations.

| Feature family | Suggested initial fields | Denominator / interpretation |
| --- | --- | --- |
| Sentiment | Existing `yt_<aspect>_video_balanced_net`, negative share, mixed share | Resolved target/aspect opinions; neutral and mixed opinions remain in the net denominator. |
| Topic attention | Proposed `yt_<aspect>_mention_share` | Mean of per-video fractions of eligible targeted vehicle comments discussing that aspect; facts/questions may count. |
| Semantics | Proposed `yt_semantic_01` through `yt_semantic_08` | A small fixed embedding/PCA or TF-IDF/SVD representation; coordinates are not sentiment scores. |
| Evidence coverage | Existing counts/missing flags; proposed date-span and attribution shares | Support and selection diagnostics, not sentiment or owner prevalence. |

For sentiment, compute each video's `(positive-negative)/resolved_opinions` and
then average equally across eligible videos. For topics and semantic vectors,
aggregate within video first, then across videos. Keep comment-weighted versions
as sensitivity checks. Deduplicate normalized exact text within video/target for
feature calculation and compare excluding cross-video repeats; retain source rows.

Keep cumulative and trailing-12-month windows. Publish coverage by **historical
cutoff**, not only in the 2026 view. Retain null for missing/suppressed scores with
explicit missing flags. The current 20-opinion/five-video minimum is a coverage
heuristic, not an accuracy guarantee. Video-block bootstrap or leave-one-video-out
checks can describe sampling stability, without proving correct NLP predictions.

Start with fewer aspects in the expensive branch: reliability, maintenance cost,
performance, and comfort are useful candidates to benchmark. They are perceptions,
not verified failure rates. Retain the existing nine-aspect vocabulary for
comparability. Leave `price_value` and direct vehicle purchase/listing-price
mentions out of the first price-feature experiment; assess them separately after
timing and target-proxy risks are understood. Keep DougScore Value separate too.

## Price-model evaluation and limits

**Timing is the first integration gate.** The listing dates end in 2020, but the
actual price-observation cutoff remains unverified. `2020-12-31` is a provisional
comparison cutoff, not evidence that all prices were observed then. Exclude later
comments and any known post-cutoff text updates. Even earlier comments were
harvested in 2026: deleted comments, original text versions, and earlier engagement
counts are unavailable. Do not use 2026 likes/replies as historical weights.

The current [Cardiff model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
states that its pretraining includes tweets through December 2021. Applying it to
pre-2021 comments is a retrospective extraction experiment, not an encoder that
was available in a 2020 deployment. A strict historical simulation needs
appropriate model availability/training dates as well as text availability.

**Static make features have no new information conditional on Make.** If every
Toyota listing receives the same vector and `Make` is already encoded, the vector
is a deterministic function of that predictor. It may affect finite-model
regularization or sharing across sparse makes, but it cannot establish a new
within-make signal. Dated features can add variation only if the available text
and listing observation dates support defensible temporal alignment. Interactions
with age/mileage can be tested as modeling choices; they do not create observed
cohort-level sentiment.

Once timing permits a price comparison:

1. Use identical listing/VIN samples and existing grouped splits. Link listings
   through lowercase vPIC `vin`, then join the selected make/date feature table
   using canonical `Make`. Keep `ModelYear`, `Make`, `Model`, `Trim` separate.
2. Compare the existing structured baseline, coverage-only additions,
   **expert-only additions**, semantic/topic additions, targeted sentiment
   additions, and **expert-plus-community additions**. Keep
   missingness handling identical. Evaluate title context separately from explicit
   attribution. Include removal/masking of make/model tokens in representations
   as a sensitivity check for merely re-encoding identity.
3. Fit TF-IDF vocabulary, PCA/SVD, clusters, pseudo-label classifiers, scalers, and
   imputers on training/development data only. Freeze feature rules and methods
   before the final held-out price evaluation; no price-driven threshold choices
   on the test set. Limit dimensionality because millions of listings do not
   create millions of independent make-level observations.
4. Select configurations using validation dollar MAE; report held-out MAE/RMSE,
   paired error differences, and supported/missing make segments. Use grouping
   appropriate to shared make/time features when estimating uncertainty. Support
   and creator/video selection can explain gains independently of sentiment.

With no manually labeled automotive sentiment holdout, we can assess executable
behavior, evidence fidelity, perturbation sensitivity, model agreement, support,
runtime, and downstream price usefulness. We **cannot estimate in-domain
sentiment/relevance accuracy or prove that model agreement means correctness**.
Automatically scored public sentiment benchmarks help compare generic models but
do not close that domain-validation gap. That limitation does not require starting
another annotation round.

## Next implementation decision

The recommended next change is a separate **raw-source benchmark mode** that
freezes a reproducible video-balanced development sample, bypasses the existing
label-trained relevance dependency, and benchmarks cached sentence embeddings,
cheap tone, and reduced target-aware inference. Use a fresh source sample to
estimate coverage; keep the enriched pilot for engineering comparisons. Measure
seconds per comment and usable make/date cells before choosing a full-source run.

Full-source inference, API inference, annotation/model replacement, and
price-model integration were not performed for this review. These remain
subsequent implementation choices, rather than completed research outcomes.

## Reproduce the local evidence

Run read-only from the repository root with the project Python environment. This
is the calculation behind the key source and pilot findings; it creates no files.

```python
import json
from pathlib import Path
import polars as pl

source = pl.scan_parquet("DATA/CAR_DATA_FINAL/Youtube_Comments.parquet")
dated = source.with_columns(
    pl.col(name).str.strptime(pl.Date, "%m-%d-%Y", strict=False).alias(name + "_date")
    for name in ("published_at", "updated_at", "extracted_at")
)
print(dated.select(
    pl.len().alias("comments"),
    pl.col("comment_id").n_unique().alias("unique_ids"),
    pl.col("video_id").n_unique().alias("videos"),
    pl.col("playlist_id").n_unique().alias("playlists"),
    pl.col("published_at_date").min().alias("first_comment"),
    pl.col("published_at_date").max().alias("last_comment"),
    pl.col("published_at_date").null_count().alias("invalid_publication_dates"),
    (pl.col("published_at_date") <= pl.date(2020, 12, 31)).sum().alias("through_2020"),
    (pl.col("updated_at_date") > pl.col("published_at_date")).sum().alias("later_updates"),
    (pl.col("text").fill_null("").str.strip_chars() == "").sum().alias("empty_text"),
    pl.col("text").str.len_chars().median().alias("median_chars"),
    pl.col("text").str.len_chars().quantile(.95).alias("p95_chars"),
    pl.col("text").str.len_chars().max().alias("max_chars"),
).collect())

outputs = Path("YOUTUBE/outputs")
run = outputs / "sentiment_inference_10k_v1"
manifest = json.loads((run / "inference_manifest.json").read_text())
print("state, comments, seconds:", manifest["state"],
      manifest["comments"], manifest["inference_seconds"])
predictions = pl.read_parquet(run / "youtube_experimental_predictions.parquet")
opinion = predictions.filter(
    pl.col("polarity").is_in(["positive", "negative", "neutral", "mixed"])
)
print("comments with any resolved opinion:", opinion["comment_id"].n_unique())
print(predictions.group_by("language_status").agg(pl.col("comment_id").n_unique()))
features = pl.read_parquet(outputs / "sentiment_features_10k_v1" /
                           "youtube_experimental_make_features.parquet")
supported = features.filter(
    (pl.col("window") == "cumulative") &
    (pl.col("inference_policy") == "relevance_none_cross_repeat_retained") &
    ~pl.col("support_suppressed")
)
print(supported.group_by("attribution_scope").agg(
    pl.len().alias("supported_make_aspect_cells"),
    pl.col("Make").n_unique().alias("makes"),
))
```

Controlling local evidence: `DATA_DICTIONARY.md`, `README.md`,
`YOUTUBE/sentiment/{common,inference,source,features}.py`,
`YOUTUBE/outputs/sentiment_inference_v1/inference_manifest.json`,
`YOUTUBE/outputs/sentiment_inference_10k_v1/inference_manifest.json`,
`YOUTUBE/outputs/sentiment_features_10k_v1/feature_manifest.json`, and the source,
prediction, and feature Parquets named in the code above. Existing source/input
hashes are retained in those manifests; this review does not rewrite them.

The expert checks read `EXPERT_REVIEWS/outputs/dougscore_v1/import_manifest.json`,
`expert_review_scores.parquet`, and `expert_aspect_scores.parquet`; summed the five
Weekend and five Daily components row by row; reconciled the total; and checked
review-ID uniqueness, mapped-aspect counts, and null video/score-date metadata.

```python
expert = pl.read_parquet("EXPERT_REVIEWS/outputs/dougscore_v1/expert_review_scores.parquet")
weekend = ["styling", "acceleration", "handling", "fun_factor", "cool_factor"]
daily = ["features", "comfort", "quality", "practicality", "value"]
print(expert.shape, expert["review_record_id"].n_unique())
print(expert.select(
    (pl.sum_horizontal(*weekend) != pl.col("weekend_total")).sum().alias("weekend_errors"),
    (pl.sum_horizontal(*daily) != pl.col("daily_total")).sum().alias("daily_errors"),
    ((pl.col("weekend_total") + pl.col("daily_total")) != pl.col("dougscore"))
        .sum().alias("overall_errors"),
    (pl.col("daily_total") - pl.col("value")).min().alias("daily_without_value_min"),
    (pl.col("daily_total") - pl.col("value")).max().alias("daily_without_value_max"),
    (pl.col("dougscore") - pl.col("value")).min().alias("total_without_value_min"),
    (pl.col("dougscore") - pl.col("value")).max().alias("total_without_value_max"),
))
```
