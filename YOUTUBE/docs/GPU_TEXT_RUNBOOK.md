# Raw-source YouTube text features on an RTX 2060 Super

Implemented 2026-10-05. This is the recommended entry point for new runs:
`YOUTUBE/text_pipeline.py`. The previous `sentiment_pipeline.py`, relevance tools,
annotation files and output snapshots remain available for comparison.

## Models and why these defaults

There is no established best model for this unlabeled automotive corpus. These
defaults balance the public model evidence, the 8 GB GPU and the cost of processing
1.98 million comments. The pipeline freezes each downloaded model's commit SHA.

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

Aggregation uses a disk-backed SQLite intermediate and streams one make/scope
group at a time instead of loading all full-source predictions into RAM. It
deduplicates exact normalized text within video/make/scope, preserving punctuation
and emojis, counts each comment once per topic/aspect, averages within video and
then equally across videos. It exports video, long make and wide make Parquets for
cumulative and trailing-12-month windows. Missing/suppressed measurements stay
null, with explicit missing flags and coverage. Support defaults to 20 comments
and five videos for semantic/topics; sentiment needs 20 resolved opinions and five
opinion-bearing videos. Neutral and mixed opinions remain in the net denominator.
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
  accuracy. Low-confidence students abstain; inspect coverage before expanding.
- Run focused software checks with
  `.\.venv-youtube\Scripts\python.exe -m unittest YOUTUBE.tests.test_text_pipeline -v`.

## Completed local benchmark (2026-10-05)

The isolated CUDA environment is installed. Saved artifacts:

- `outputs/text_development_v1`: 12,000 raw comments; protected material excluded.
- `outputs/text_teacher_v2`: 29,244 clauses, 147.2 scoring seconds excluding initial
  downloads/loading; 81.5 comments/second and 1.09 GiB peak allocated VRAM.
- `outputs/text_models_v3`: both students trained for three epochs. Content uses
  6,322 confident training clauses (6,112 opinions, 91 facts/questions, 119 creator
  comments); sentiment uses 1,741 (426 negative, 479 neutral, 836 positive).
  The reducer fits 3,015 eligible training clauses. Earlier versions preserve
  class-balancing comparisons; model selection used validation.
- Validation teacher-agreement macro F1: **0.681 content**, **0.722 sentiment**.
  Fact/question F1 is 0.308 on eight validation examples; `irrelevant` has no
  confident training examples. The students remain experimental candidates.
- `outputs/text_benchmark_v1` and `text_benchmark_scored_v1`: separate seed-44
  2,000-comment sample; 7.8 scoring seconds excluding loading (256.3 comments/s).
  Included videos/text have zero overlap with all development partitions.
- `outputs/text_benchmark_features_v1`: 119 wide, 952 long and 4,928 video rows.
  Six cumulative make/scope cells pass semantic/topic support (BMW, Honda, Tesla,
  Toyota). **No aspect-sentiment cell passes support in this small benchmark**.
  Full-source coverage was not yet measured during that benchmark; the completed
  source run below records its current coverage.
- `outputs/text_benchmark_expert_v1`: 952 descriptive panel rows; zero approved
  expert identities, so expert coverage stays missing. No blended score is created.

All 37 YouTube software checks pass. Actual artifacts passed split separation,
independent feature exclusions, grain uniqueness and suppression checks; scoring
resume was verified. The new README Mermaid passed static lint/schema validation.
Commands above use fresh v4 paths; to reuse trained models, supply
`--model-dir YOUTUBE/outputs/text_models_v3`. The benchmark remains separate from
the subsequently completed full-source run below.

## Completed full-source run (verified 2026-10-06)

The saved `outputs/text_source_v1` preparation contains 1,287,155 eligible comments
across 2,999 videos, from the 1,981,686-comment source snapshot, using the
`2026-10-01` publication/update cutoff. `outputs/text_scored_v1/score_manifest.json`
is complete: 1,287,155 comments and 2,872,698 clauses scored with `text_models_v3`.
Its recorded scoring duration is 4,300.3 seconds and peak allocated VRAM is 0.383 GiB.

`outputs/text_features_v1/feature_manifest.json` is also complete: 273 wide make
rows, 2,184 long make/aspect rows, and 312,696 video feature rows across the separate
attribution/window policies. Development-video/text exclusions and suppression
rules remain applied. These are retrospective feature artifacts, not automotive
accuracy measurements.

`outputs/price_features_v1/price_feature_manifest.json` freezes the selected
explicit-comment/cumulative view: 83 unique makes, 73 supported semantic/topic
makes, and 46 candidate predictors. There are no populated comfort,
maintenance-cost, or performance sentiment nets; the only populated reliability
net is zero. The manifest records `directional_sentiment_available=false`.
Compare coverage and semantic/topic additions separately before interpreting any
sentiment-block increment.

The current [full price comparison](../../README.md#full-comparison-including-youtube-and-all-makes)
uses this selected export. Reuse completed source/score/feature outputs;
reproduction requires fresh directories. `price-export` can select a fresh export
without repeating GPU inference:

```powershell
.\.venv-youtube\Scripts\python.exe -m YOUTUBE.text_pipeline price-export --feature-dir YOUTUBE/outputs/text_features_v1 --output-dir YOUTUBE/outputs/price_features_v2
```

## Price experiment and expert baseline

Outputs remain separate experimental artifacts; the six source Parquets and the
existing price model are not automatically changed. The 2026 extraction,
historical text survival, actual price-observation dates and encoder availability
must be addressed before historical price claims. An earlier cutoff needs teachers,
students and the reducer prepared/trained with that earlier text cutoff; later-fit
artifacts are rejected. All dated features remain retrospective extractions.

Use the existing `sentiment_pipeline.py expert` and `compare` route for the separate
DougScore reference and approved-identity panels. `compare` accepts both legacy
experimental and new text make-feature Parquets with their manifest hashes:

```powershell
.\.venv-youtube\Scripts\python.exe YOUTUBE\sentiment_pipeline.py compare --reference-dir YOUTUBE\outputs\dougscore_reference_v1 --feature-dir YOUTUBE\outputs\text_features_v4 --output-dir YOUTUBE\outputs\text_expert_comparison_v4
```

Compare structured baseline,
coverage-only, expert-only, semantic/topic, sentiment, and expert-plus-community
features on identical VIN samples/grouped splits when timing permits. Select one
make/scope/date/window wide row per listing: joining all policies duplicates rows.
Keep `ModelYear`, `Make`, `Model`, `Trim` separate and use lowercase vPIC `vin`.
Static make features can merely re-encode Make; they are not observed cohort
sentiment. No in-domain NLP accuracy claim is possible without independent gold
evaluation, and no new annotation round is required to run these experiments.
