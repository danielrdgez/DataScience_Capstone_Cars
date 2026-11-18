# Legacy pretrained YouTube sentiment benchmark

The October 1, 2026 nine-aspect CPU prototype uses `YOUTUBE/sentiment_pipeline.py` and `YOUTUBE/requirements.txt`. For new runs, see the [GPU text pipeline](GPU_TEXT_RUNBOOK.md).

## Reproduce a bounded benchmark

Use fresh output names and the saved relevance candidate. First inference downloads public BART/Cardiff weights and freezes revisions. The completed 10,000-comment CPU run took 17.8 hours; benchmark a small sample before scaling.

```powershell
.\.venv\Scripts\python.exe YOUTUBE/sentiment_pipeline.py prepare --output-dir YOUTUBE/outputs/sentiment_sample_next --sample-size 1000
.\.venv\Scripts\python.exe YOUTUBE/sentiment_pipeline.py infer --sample-dir YOUTUBE/outputs/sentiment_sample_next --output-dir YOUTUBE/outputs/sentiment_inference_next --device cpu
.\.venv\Scripts\python.exe YOUTUBE/sentiment_pipeline.py aggregate --inference-dir YOUTUBE/outputs/sentiment_inference_next --output-dir YOUTUBE/outputs/sentiment_features_next --as-of 2026-10-01
```

Run stages in order and stop on failure. `infer` and full-source `source` support identical-command `--resume`. `source --output-dir <new-dir>` scores raw comments with the frozen relevance candidate, then its output can supply `infer --sample-dir`. Aggregation offers `--monthly-start YYYY-MM-DD` and `--cross-video-sensitivity`; it loads predictions into RAM, so source-wide memory needs measurement.

## Inference and attribution

- BART MNLI supplies independent aspect/opinion hypotheses; Cardiff supplies clause tone. Their scores are uncalibrated and domain transfer remains unvalidated. NLI neutral is distinct from neutral sentiment.
- Preserve clause offsets, original text, language, truncation, and unresolved evidence. Multi-target comparisons require target-specific polarity; whole-comment tone cannot label every aspect.
- Explicit comment makes use canonical vPIC names/aliases with token boundaries. Title context is a separate policy; it cannot turn creator praise into a vehicle opinion. Year/model/trim claims require their own identity evidence.
- The engineering pilot excludes protected evaluation IDs/videos/text. It is enriched across video/score bands and cannot estimate source prevalence. Cutoff 0.20, 0.50, and no-cutoff aggregates retain all source rows.

## Outputs and aggregation

| Artifact | Grain |
| --- | --- |
| `youtube_experimental_predictions.parquet` | Comment/target/aspect with evidence and unresolved state |
| `youtube_experimental_video_features.parquet` | Video/target/aspect/date/window/policy |
| `youtube_experimental_make_features.parquet` | Make/scope/aspect/date/window/policy |
| `dougscore_experimental_reference.parquet` | Source review/mapped aspect and identity status |

The nine aspects are defined in the [label guide](MAKE_SENTIMENT_LABEL_GUIDE.md). Deduplicate exact normalized text within video/target/aspect for aggregates; retain original rows and cross-video repeat flags. Count each comment once, compute `(positive-negative)/(positive+negative+neutral+mixed)` per video, then average eligible videos equally. Unsupported cells (fewer than 20 opinions or five videos) stay null. No-opinion and unresolved records are outside the denominator.

## Expert reference and evaluation

`sentiment_pipeline.py expert --output-dir <new-dir>` exports the DougScore reference. `compare --reference-dir <reference> --feature-dir <features> --output-dir <new-dir>` uses approved identities to build descriptive panels. It also accepts GPU-generated make features. [Expert aspect mappings and pairing limits](../../EXPERT_REVIEWS/DOUGSCORE_INTEGRATION_PLAN.md)

Saved engineering outputs are `sentiment_inference_v1`, `sentiment_inference_10k_v1`, `sentiment_features_v1`, and `sentiment_features_10k_v1` under `YOUTUBE/outputs/`. Independent automotive sentiment accuracy remains unmeasured. Historical text/score availability and identical VIN populations/splits are required for forecasting claims; current source additions are retrospective experiments. Price commands and selected predictors are documented in the [README](../../README.md#listing-price-models) and [dictionary](../../DATA_DICTIONARY.md#selected-youtube-predictors).
