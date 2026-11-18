# Supervised YouTube relevance and sentiment plan

New labeling is deferred. For the optional supervised route, follow the relevance runbook and labeling guide. Use the GPU runbook for new raw-source runs. [Relevance runbook](RELEVANCE_ROUND_RUNBOOK.md) · [Label guide](MAKE_SENTIMENT_LABEL_GUIDE.md) · [GPU runbook](GPU_TEXT_RUNBOOK.md)

## Current checkpoint

Saved-run checkpoint from October 1, 2026.

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

The [artifact review](../outputs/pipeline_run_review_01/run_review.json) binds candidate hashes, source fields, score checkpoints, and sample isolation. Training CV at the selected 0.20 cutoff followed error review and is not independent performance. Scores are uncalibrated; saved bootstrap intervals apply to 0.50.

## Validation and implementation sequence

| Stage | Remaining work / completion condition |
| --- | --- |
| Relevance labels | Finish blind annotations, resolve the five subtype issues without changing binary judgments implicitly, and preserve independent reviewer answers before reconciliation. Training completion is optional for the saved candidate. |
| Pilot audit | Review proposed-rejection, boundary, and retained strata. Report strata separately or weight by their sampling support; the enriched pilot cannot estimate source prevalence. Audit language, long inputs, and semantic duplicates. |
| Locked relevance evaluation | Freeze candidate/threshold first; evaluate once. Candidate v1 requires at least 100 relevant examples, 50 proposed rejections, and both point values and lower video-bootstrap/Wilson bounds meeting 98% relevant recall and 95% rejection precision. Reviewer/pilot/unseen-make checks remain manual release requirements even if this screen passes. |
| Target/aspect development | Annotate canonical make, opinion presence, aspect, polarity, exact evidence, and reliability basis. Start with existing confirmed positives; reserve fresh videos for sentiment evaluation. |
| Sentiment models | Separate target resolution, opinion, aspect detection, and target/aspect polarity. Compare simple baselines with multi-label SetFit/conditioned polarity; span-based SetFitABSA is optional. Report per-aspect support, errors, target swaps, negation, sarcasm, and mixed opinions. |
| Batch features | Only validated source rejection may exclude comments automatically. Keep uncertain rows as `review_or_pass`; use versioned resumable outputs and retain source text/IDs. |

All records from a comment/video stay in one partition; normalized duplicates must not cross it. Test unseen makes separately. Model or threshold revisions after blind evaluation require development data and a fresh held-out test. Confidence intervals are approximate screens, not population guarantees.

## Proposed supervised outputs

These exports are proposed for the deferred supervised route. `text_pipeline.py` produces the current raw-source exports.

| Artifact | Grain / contents |
| --- | --- |
| `youtube_comment_relevance.parquet` | One `comment_id`; score, decision, rule reason, target-resolution status, versions. |
| `youtube_comment_targets.parquet` | One `(comment_id, Make)`; supported target and opinion state. Unresolved comments have no target row. |
| `youtube_comment_aspect_sentiment.parquet` | One `(comment_id, Make, aspect)`; polarity, evidence, reliability basis, versions. |
| `youtube_make_sentiment_features.parquet` | One `(Make, aspect, as_of_date, window)`; counts, video-balanced net, support, uncertainty, suppression. |

Aggregate cumulative and trailing-12-month windows after date filtering. Count each comment once per target/aspect; compute `(positive-negative)/opinion_count` within video, including neutral/mixed opinions in the denominator, then average videos equally. Proposed support is 20 opinions across five videos. Missing/suppressed measurements stay null. Source display names are not verified authors.

Expert ratings remain a separate [DougScore baseline](../../EXPERT_REVIEWS/DOUGSCORE_INTEGRATION_PLAN.md), not human sentiment labels. Historical listing prices and later text/score snapshots support retrospective comparisons until observation-time availability is established. Current selected field definitions belong in the [dictionary](../../DATA_DICTIONARY.md#selected-youtube-predictors).

References: [SetFit multi-label](https://huggingface.co/docs/setfit/how_to/multilabel), [SetFitABSA](https://huggingface.co/docs/setfit/how_to/absa), [SemEval aspect-based sentiment](https://aclanthology.org/S14-2004/).
