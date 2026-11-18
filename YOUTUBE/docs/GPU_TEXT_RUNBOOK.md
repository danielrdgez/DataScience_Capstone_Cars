# Raw-source YouTube text features on an RTX 2060 Super

Use `YOUTUBE/text_pipeline.py` for new runs. The GPU workflow was implemented on 2026-10-05. The older `sentiment_pipeline.py`, relevance tools, labels, and snapshots support historical comparisons.

## Models and why these defaults

The defaults fit an 8 GB GPU and account for the cost of processing 1.98 million comments. The pipeline pins downloaded model commits. Comparative accuracy on this automotive corpus remains unmeasured.

| Role | Default | Rationale and limits |
|---|---|---|
| Cached embeddings/topic similarities | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Compact 384-dimensional encoder, about 22.7M parameters. General semantic representations, not calibrated relevance or sentiment. |
| Content teacher | [MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33](https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33) | Binary entailment model for efficient zero-shot routing. The author's card explicitly trades quality for efficiency and does not report zero-shot evaluation for this checkpoint. |
| Sentiment teacher | [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | Negative/neutral/positive social-text baseline. Twitter-to-YouTube transfer is unvalidated; pretraining includes tweets through 2021. |
| Two trained students | [nreimers/MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) | Separate compact content and sentiment classifiers, fine-tuned locally on confident teacher predictions with mixed precision. |
| Eight semantic coordinates | PCA on eligible training-video embeddings | Saves the fitted mean/components as NumPy arrays; validation/test text never fits the reducer. CPU fitting is small compared with neural inference. |

[all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
is a stronger general embedding comparator, with 768 coordinates and higher cost;
the [Sentence Transformers model guide](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
describes the speed/quality tradeoff. DeBERTa base/large zero-shot models are future
teacher comparators. The old BART route remains a comparator. Large generative
models, QLoRA and full-corpus BART are unnecessary for this first GPU baseline.
These are engineering choices, not measured automotive model rankings.

## Setup (PowerShell, repository root)

Use an isolated environment to preserve the existing price/legacy dependencies.
CUDA 12.6 wheels support this GPU; the tested Python is 3.13. The GPU has compute
capability 7.5. Use FP16, not BF16. PyTorch bundles its CUDA runtime; a separate
CUDA toolkit is unnecessary. Installation commands follow
[PyTorch's official version instructions](https://docs.pytorch.org/get-started/previous-versions/).

```powershell
Set-Location E:\DataScience_Capstone_Cars
python -m venv .venv-youtube
.\.venv-youtube\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-youtube\Scripts\python.exe -m pip install -r YOUTUBE\requirements-gpu.txt
.\.venv-youtube\Scripts\python.exe YOUTUBE\text_pipeline.py doctor
```

The environment and public model cache are installed locally and ignored by Git.
The existing `.venv` is CPU-only; use `.venv-youtube` for these commands. No Hub
account or token is required for the public models, and no comments are uploaded.

## Development and GPU training

```powershell
.\.venv-youtube\Scripts\python.exe YOUTUBE\text_pipeline.py prepare --output-dir YOUTUBE\outputs\text_development_v4 --sample-size 20000 --as-of 2026-10-01
.\.venv-youtube\Scripts\python.exe YOUTUBE\text_pipeline.py score --input-dir YOUTUBE\outputs\text_development_v4 --output-dir YOUTUBE\outputs\text_teacher_v4 --backend teacher --device cuda --batch-size 32
.\.venv-youtube\Scripts\python.exe YOUTUBE\text_pipeline.py train --teacher-dir YOUTUBE\outputs\text_teacher_v4 --output-dir YOUTUBE\outputs\text_models_v4 --device cuda --batch-size 16 --accumulation 2 --epochs 3
```

`prepare` streams raw `Youtube_Comments.parquet`, builds its canonical make
vocabulary independently from `NHTSA_vPIC.Make`, and excludes protected IDs,
videos, normalized protected text and the original reserved-video partition.
It reads frozen reference text/identifiers only, never annotation scores. The
existing supervised relevance model is not loaded. The first cutoff excludes
both post-cutoff publication and post-cutoff text updates. The default development
pool contains a seeded 25% of eligible videos; rotating per-video quotas limit
volume dominance. The sample is a development benchmark, not a population sample.
Videos split 80/10/10 into train/validation/test; exact text cannot cross partitions.

`score --backend teacher` caches all clause evidence, content scores, semantic
embeddings, topic similarities and eligible tone scores. Content routing separates
vehicle opinions, vehicle facts/questions, creator discussion and irrelevant text.
Confident predictions are pseudo-labels, not observed truth. Short English rule
decisions remain explicitly flagged. Unsupported/unknown text abstains.

`train` fine-tunes **two separate models** using confident training-video
pseudo-labels, FP16, gradient accumulation, class balancing and gradient clipping.
Content batches resample training examples with inverse-frequency weights;
sentiment uses inverse-frequency loss weights. Validation/test are never resampled.
Validation selects each checkpoint by macro F1 agreement with its teacher; test
agreement is reported once afterwards. No manual labels or price targets are used.
The eight-component reducer fits only eligible training text. Class counts below
20 cause a clear failure: prepare a larger sample rather than inventing labels.
The optional `irrelevant` class is omitted and recorded as `unlearned_labels` if
the teacher supplies no confident examples; abstentions are never relabeled as
irrelevant. Content probabilities then cover the three supported classes.
An optional `--max-steps N` is a bounded engineering run and is recorded as such.

## Feature generation

```powershell
.\.venv-youtube\Scripts\python.exe YOUTUBE\text_pipeline.py prepare --output-dir YOUTUBE\outputs\text_source_v4 --sample-size 0 --as-of 2026-10-01
.\.venv-youtube\Scripts\python.exe YOUTUBE\text_pipeline.py score --input-dir YOUTUBE\outputs\text_source_v4 --output-dir YOUTUBE\outputs\text_scored_v4 --backend student --model-dir YOUTUBE\outputs\text_models_v4 --device cuda --batch-size 32
.\.venv-youtube\Scripts\python.exe YOUTUBE\text_pipeline.py aggregate --score-dir YOUTUBE\outputs\text_scored_v4 --model-dir YOUTUBE\outputs\text_models_v4 --output-dir YOUTUBE\outputs\text_features_v4 --as-of 2026-10-01
```

Full-source scoring excludes every development-sample video and exact development
text from features, preserving independent feature-generation videos. It retains
their rows and flags in cached outputs. For a quick bounded test before this full
run, prepare another sample using `--sample-size 2000 --seed 44` and fresh paths;
the independent seed provides videos outside the training sample. Those resulting
features still describe an engineering sample. Manifests report throughput and
peak allocated VRAM; use measured GPU throughput to estimate the source run.

Facts/questions contribute to semantic/topic features without becoming neutral
sentiment. Explicit targets and title context stay separate. Title context requires
a unique title make, no competing explicit make and a vehicle cue. Ambiguous make
words require automotive context. Clauses with multiple makes abstain. Original
text offsets are retained, long tails become bounded windows, and truncated or
windowed evidence cannot resolve sentiment. Multi-aspect clauses do not transfer
one tone to all their aspects. Multiple resolved clause polarities within one
comment can produce `mixed`; incomplete evidence remains unresolved.

Topic similarities use the eight existing non-price aspect descriptions; cosine
0.30 is an experimental threshold. Sentiment resolves only reliability,
maintenance cost, performance and comfort in this first reduced route. Direct
currency/explicit numerical price mentions are flagged and excluded from both
feature branches. Price/value and DougScore Value stay out of the initial price
experiment. Canonical make tokens/known aliases are masked for semantic encoding
by default; `--keep-make-tokens` requires separately trained reducer artifacts for
an identity sensitivity run. Model-name identity can still remain in text.

Aggregation streams one make/scope group at a time from disk-backed SQLite. It deduplicates normalized text within video/make/scope, preserving punctuation and emojis. Count each comment once per topic or aspect, average within each video, then weight videos equally. Outputs include video, long make, and wide make Parquets for cumulative and trailing-12-month windows.

By default, semantic and topic measures require 20 comments across five videos. Sentiment requires 20 resolved opinions across five opinion-bearing videos, with neutral and mixed opinions included in the net denominator. Missing or suppressed measurements stay null, with flags and coverage counts.
`--exclude-cross-video-repeats` produces a separate repeat sensitivity run.

## Resume and troubleshooting

- Repeat an interrupted **score** or **train** command with exactly the same
  arguments plus `--resume`. Score checkpoints are per comment shard; training
  checkpoints save optimizer/scaler/RNG state every 100 updates and each epoch.
- Inputs, completed shards, model artifacts, code and package versions are
  checksum/configuration guarded. Changed settings/code require fresh output
  paths. Preparation and aggregation use fresh directories; rerun those into a
  new directory after interruption. Do not edit manifests to bypass guards.
- CUDA failure is explicit; the pipeline never silently falls back to CPU.
  On out-of-memory errors, use fresh paths with scoring `--batch-size 16` or
  training `--batch-size 8 --accumulation 4`.
- `teacher_agreement_macro_f1` measures distillation fidelity, not sentiment
  accuracy. Student argmax retains uncertainty; optional threshold decisions
  abstain. Inspect exported coverage and human-reviewed accuracy separately.
- Run focused software checks with
  `.\.venv-youtube\Scripts\python.exe -m unittest YOUTUBE.tests.test_text_pipeline -v`.

## Sentiment v2 decisions and evaluation

Student scoring now defaults to `--sentiment-policy argmax`. Teacher scoring and
training pseudo-label selection retain confident labels. Content routing, target
attribution, date limits, development exclusions and 20-opinion/five-video support
remain enforced. Aspect sentiment still needs one eligible topic. Overall car
sentiment includes eligible opinions with multiple topics. Both hard-label net
and continuous `positive_score-negative_score` are averaged within comments,
within videos, then equally across videos. Uncertainty remains explicit.

Rebuild student decisions from cached `text_scored_v1` probabilities without retraining or GPU inference. Use a fresh run directory and stop on failure:

```powershell
$PythonExe = '.\.venv-youtube\Scripts\python.exe'
$RunRoot = 'YOUTUBE/outputs/sentiment_argmax_' + (Get-Date -Format 'yyyyMMddTHHmmss')
& $PythonExe -m YOUTUBE.text_pipeline relabel --score-dir YOUTUBE/outputs/text_scored_v1 --output-dir "$RunRoot/scored"
if ($LASTEXITCODE -ne 0) { throw 'Relabel failed' }
& $PythonExe -m YOUTUBE.text_pipeline aggregate --score-dir "$RunRoot/scored" --model-dir YOUTUBE/outputs/text_models_v3 --output-dir "$RunRoot/features" --as-of 2026-10-01
if ($LASTEXITCODE -ne 0) { throw 'Aggregation failed' }
& $PythonExe -m YOUTUBE.text_pipeline price-export --feature-dir "$RunRoot/features" --output-dir "$RunRoot/price"
```

`relabel --resume` requires identical inputs, policy, calibration, and code. Preserved embeddings make full-source relabel outputs large; use one writer per output directory.

Evaluate the existing student and fit an optional temperature on development
validation videos; test-video results include all-row and accepted-only reports:

```powershell
& $PythonExe -m YOUTUBE.text_pipeline evaluate-sentiment --teacher-dir YOUTUBE/outputs/text_teacher_v2 --model-dir YOUTUBE/outputs/text_models_v3 --output-dir "$RunRoot/evaluation" --device cpu --cpu-threads 4
```

The result is `sentiment_calibration.json` and `review_predictions.csv`. Default
labels are teacher pseudo-labels, not human gold labels. The sample is conditional
on confident teacher opinion routing; this does not measure full content/aspect
accuracy. Existing historical model-selection/test use remains disclosed.
Calibration changes score scale, not the argmax class. Use an optional
`--calibration "$RunRoot/evaluation/sentiment_calibration.json"` with `relabel`
or student `score` in another fresh output directory for a calibrated comparison.
The artifact must match the model manifest; label source is recorded downstream.

For reviewed calibration, fill `human_label` (`negative`, `neutral`, `positive`)
and `reviewer` in a copy of `review_predictions.csv`, then run `evaluate-sentiment`
with `--review-csv <copy.csv>` and a fresh output directory. Unlabeled rows are
omitted. Text/video keys cannot change or repeat. Both preserved validation and
test video partitions must contain all classes. Review content relevance and
aspect assignment separately before claiming independent automotive accuracy.

To rerun inference rather than reuse cached probabilities:

```powershell
& $PythonExe -m YOUTUBE.text_pipeline score --input-dir YOUTUBE/outputs/text_source_v1 --backend student --model-dir YOUTUBE/outputs/text_models_v3 --output-dir "$RunRoot/rescored" --device cuda --batch-size 32 --sentiment-policy argmax
```

Train a validation-only price comparison after the export exists (40 of its 75 candidate fields are selected):

```powershell
& .\.venv\Scripts\python.exe -m MODELS.Price_ML_Models --model-type full_youtube_sentiment --youtube-feature-dir "$RunRoot/price" --models catboost --sample-size 100000 --validation-only --device auto
```

Use identical listing samples/splits for comparisons with `enhanced_safety`; select on validation before final testing. This does not activate a dashboard predictor. Saved predictors retain the exact export/manifest and reject incompatible schemas. Selected field definitions, aspect support, and calibration status are in the [dictionary](../../DATA_DICTIONARY.md#selected-youtube-predictors).

## Saved runs

These recorded measurements exclude initial model loading/downloads; they are not accuracy or future-runtime guarantees.

| Artifact under `YOUTUBE/outputs/` | Recorded evidence |
| --- | --- |
| `text_development_v1` / `text_teacher_v2` | 12,000 comments, 29,244 clauses; teacher scoring 147.2 s, 1.09 GiB peak VRAM. |
| `text_models_v3` | Three-epoch students; 6,322 content and 1,741 sentiment training clauses; reducer fitted to 3,015 eligible clauses. Validation teacher-agreement macro F1: 0.681 content, 0.722 sentiment; fact/question support is sparse and `irrelevant` was unlearned. |
| `text_benchmark_v1` / `text_benchmark_scored_v1` | Separate 2,000-comment seed-44 sample; 7.8 s, zero development video/text overlap. |
| `text_benchmark_features_v1` | 119 wide, 952 long, 4,928 video rows; six supported semantic/topic make/scope cells, no supported aspect sentiment. |
| `text_source_v1` / `text_scored_v1` | Complete 1,287,155-comment, 2,999-video source run; 2,872,698 clauses, 4,300.3 s, 0.383 GiB peak VRAM. |
| `text_features_v1` | 273 wide, 2,184 long, 312,696 video rows across separate policies/windows. |
| `sentiment_argmax_20261027T155951/price` | Selected sentiment-v2 export: 83 makes, 75 candidate predictors; 73 makes have semantic/topic and overall sentiment support. |

Reuse trained `text_models_v3` for compatible inference. To copy the selected feature view without GPU scoring:

```powershell
.\.venv-youtube\Scripts\python.exe -m YOUTUBE.text_pipeline price-export --feature-dir YOUTUBE/outputs/sentiment_argmax_20261027T155951/features --output-dir YOUTUBE/outputs/price_sentiment_v2_copy
```

## Price experiment and expert baseline

Outputs remain separate experimental artifacts; the six source Parquets and the
existing price model are not automatically changed. The 2026 extraction,
historical text survival, actual price-observation dates and encoder availability
must be addressed before historical price claims. An earlier cutoff needs teachers,
students and the reducer prepared/trained with that earlier text cutoff; later-fit
artifacts are rejected. All dated features remain retrospective extractions.

Select one make/scope/date/window row per listing to prevent row multiplication. The opinions describe makes. Independent automotive NLP validation remains pending. [Expert comparison commands](EXPERIMENTAL_SENTIMENT_FEATURE_PLAN.md#expert-reference-and-evaluation) · [Review matching](../../EXPERT_REVIEWS/DOUGSCORE_INTEGRATION_PLAN.md) · [Price workflows and matched ablations](../../README.md#nhtsa-and-youtube-feature-comparison)
