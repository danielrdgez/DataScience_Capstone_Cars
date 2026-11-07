# YouTube text features without new manual labels

Research/artifact review: October 5, 2026. The recommended semantic/topic and reduced-sentiment route is implemented in `YOUTUBE/text_pipeline.py`; the [GPU runbook](GPU_TEXT_RUNBOOK.md) owns execution and saved runs. This note retains the evidence and alternatives behind that choice. Reported literature results were not reproduced on this corpus.

## Available evidence

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

## Legacy benchmark evidence

The saved `sentiment_inference_10k_v1/inference_manifest.json` records 64,032.7 CPU inference seconds (17.8 hours). Of its enriched 10,000 comments, 3,730 had a resolved aspect opinion; language status was 7,931 English, 1,906 unknown, and 163 unsupported. At the cumulative 2026-10-01 no-cutoff view, the 20-opinion/five-video floor supported 22 explicit make/aspect cells across six makes versus 322 title-context cells across 42 makes. These engineering observations motivated a cheaper raw-source route; they are neither population estimates nor accuracy measurements.

The legacy scorer depends on a relevance candidate fitted with earlier human labels. The raw-source GPU route avoids that classifier but preserves protected-evaluation exclusions. External pretrained models may themselves use labeled public data.

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

## Evaluation implications

- Make-level review-audience signals do not establish owner prevalence or trim-level sentiment. The selected schema, attribution, dates, and support rules belong in the [dictionary](../../DATA_DICTIONARY.md#selected-youtube-predictors).
- A static vector joined by Make is a deterministic function of an already encoded Make. Any price benefit may reflect regularization or information sharing, rather than new within-make observations. Compare identity masking and coverage-only controls.
- Use identical eligible VINs/splits for structured, coverage-only, semantic/topic, sentiment, expert, and combined price ablations. Fit learned text transformations on development data; select on validation USD MAE and preserve final testing.
- The 2026 extraction lacks deleted comments and historical text/engagement versions. Cardiff pretraining includes 2021 text. Earlier publication dates alone do not establish a valid 2020 forecasting simulation.
- Software behavior, evidence fidelity, perturbation sensitivity, support, runtime, and downstream errors can be measured without new labels. Independent in-domain sentiment accuracy cannot.

[DougScore integration](../../EXPERT_REVIEWS/DOUGSCORE_INTEGRATION_PLAN.md) owns the expert snapshot, aspect mappings, and pairing design. Expert/community agreement is descriptive consistency, not sentiment truth; their scales remain separate.

## Reproduce the local evidence

Source profiles are in [DATA_DICTIONARY.md](../../DATA_DICTIONARY.md); source metadata is `DATA/CAR_DATA_FINAL/manifest.json`. Legacy inference and aggregation receipts are under `YOUTUBE/outputs/sentiment_inference_10k_v1/` and `sentiment_features_10k_v1/`. Inspect the referenced prediction/feature Parquets using their manifest hashes; filter one attribution/window/policy before counting supported make/aspect rows. The [DougScore audit notebook](../../EXPERT_REVIEWS/DOUGSCORE_AUDIT.ipynb) reproduces expert score and identity checks. Preserve the recorded source hashes and outputs.
