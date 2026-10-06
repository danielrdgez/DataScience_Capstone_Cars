# YouTube text features: current process and next steps

The current entry point is **`text_pipeline.py`**. Read
[the GPU runbook](docs/GPU_TEXT_RUNBOOK.md) for setup and PowerShell commands.
It implements raw-source preparation, cached embeddings/topics, pretrained teacher
pseudo-labels, two GPU-trained MiniLM students, training-only semantic reduction,
and disk-backed dated make features. Use `.venv-youtube` for this route.

Completed: 12,000-comment GPU teachers, three-epoch students/reducer under
`outputs/text_models_v3`, the separate 2,000-comment benchmark, and full-source
`text_scored_v1` (1,287,155 comments, 2,872,698 clauses) / `text_features_v1`.
The selected explicit-comment/cumulative full-source view contains 83 makes,
73 supported semantic/topic makes, and only one populated sentiment net (zero).
`outputs/price_features_v1` freezes its 46 price-comparison candidates, coverage,
and sentiment limitation. Validation teacher agreement is 0.681
for content and 0.722 for sentiment, with sparse fact/question support. Exact saved
paths and findings are in the GPU runbook.

1. Verify CUDA with `doctor`; review saved development/training manifests.
2. For new training, use fresh paths: `prepare`, teacher `score`, then `train`.
   Teacher agreement is distillation fidelity, not automotive accuracy.
3. Score a bounded independent sample to measure runtime, memory, attribution and
   usable make support. Training/development videos/text remain excluded.
4. Use `prepare --sample-size 0`, student `score`, then `aggregate` for all source
   rows. Score/train resume with identical settings and `--resume`.
5. The price comparison now joins the selected canonical make export with
   matched NHTSA/price subsets. Run `python -m NHTSA.text_features run-full
   --sample-size 0 --coverage-ablations --evaluate-test` in `.venv` for all eight
   price estimators and all-make/$25k/$100k+ reports. See the README for defaults,
   validation-only runs, output files, and independent-test preservation.
   Candidate sentiment is currently degenerate; diagnose confidence and aspect
   routing on independent automotive examples before regenerating those scores.
   Keep explicit/title and repeat-sensitivity policies separately.
6. These joins are retrospective. Verify price/text/expert availability before historical forecasting. Keep
   cohort dimensions separate and use lowercase vPIC `vin`.

## Preserved previous sentiment benchmark

Implemented: one pipeline CLI, nine-aspect clause inference with checkpoints,
source relevance scoring, dated video/make/wide features, separate DougScore
reference, and approved-identity descriptive panels. Labeling remains deferred.

Created: `outputs/sentiment_sample_v1/sample.parquet` (1,000 pilot comments),
`outputs/sentiment_sample_10k_v1/sample.parquet` (10,000 pilot comments), and
`outputs/dougscore_reference_v1/dougscore_experimental_reference.parquet`
(2,925 review/aspect rows). Existing sources and outputs are preserved.

Completed saved runs: `sentiment_inference_v1`, `sentiment_inference_10k_v1`,
`sentiment_features_v1`, and `sentiment_features_10k_v1`. All remain enriched
engineering outputs outside the price model. The 10,000-comment CPU inference
recorded 17.8 hours. See the [no-new-label research review](docs/NO_LABEL_TEXT_FEATURE_RESEARCH.md)
for proposed raw-source embeddings/topics and cheaper targeted sentiment
benchmarks; these alternatives are now implemented in the GPU route and its
selected make export is available to the retrospective price comparison.

1. Review saved manifests/results before another legacy run. For reproduction, use
   `YOUTUBE/requirements.txt` and [the README commands](../README.md#previous-youtube-sentiment-benchmark-workflow)
   with fresh output directories. First use downloads public weights and freezes
   commit revisions. Benchmark a reduced route before source-wide CPU inference.
2. Inspect evidence and unresolved/coverage counts, runtime and RAM, languages,
   negation, creator praise, sarcasm and comparisons. These engineering checks do
   not measure sentiment accuracy. Unknown/short-language evidence stays unresolved.
3. The whole 10,000-comment pilot already has saved inference/aggregate outputs.
   Compare 0.20/0.50/no-cutoff views and explicit/title attribution. Add monthly
   cutoffs with `aggregate --monthly-start YYYY-MM-DD`; optionally add
   `--cross-video-sensitivity`. Pilot features remain engineering-only.
4. Benchmark a fresh source sample before committing to all 1.98 million comments.
   Then use `source --output-dir <new-dir>`, pass that directory to `infer`, and
   aggregate the source-scored predictions. Source/inference batches resume with
   the identical command plus `--resume`. Full-source aggregation currently loads
   predictions into RAM; partition/stream that stage if the measured footprint
   requires it. Frozen source hashes must match the saved relevance candidate.
5. Review DougScore canonical identities and recover review/video and score dates.
   `compare` retains missing expert coverage while matches are pending. Keep expert
   ratings and community sentiment separate; do not claim same-review linkage.
6. Verify price observation dates before joining features to historical listings.
   Then compare baseline, expert-only, community-only and combined additions on
   identical VIN samples/splits, including support-only ablations. Leave Value out
   initially. Legacy sentiment/expert predictor integration and independent sentiment evaluation remain
   future work.

Files now live under `sentiment/`, `relevance/`, `tests/`, and `docs/`.
Historical manifests keep their original paths/hashes; saved candidates recognize the exact helper relocation edits. Old pilot/comparison
resumes require a fresh run after relocation. Use existing scored pilot artifacts
for the current route.
Do not edit old manifests to force a resume.
