# Supervised relevance and aspect runbook

New labeling is deferred. See the supervised plan for release criteria and the label guide for target and aspect definitions. Use the GPU runbook for new raw-source runs. [Supervised plan](MAKE_SENTIMENT_PIPELINE_PLAN.md) · [Label guide](MAKE_SENTIMENT_LABEL_GUIDE.md) · [GPU runbook](GPU_TEXT_RUNBOOK.md)

Run PowerShell from the repository root with `.venv`. The commands show historical run names. Choose fresh output paths for completed stages or changed settings and preserve frozen sources, samples, and annotations.

## Prepare and label relevance

```powershell
Set-Location E:\DataScience_Capstone_Cars
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_relevance_round.py prepare --output-dir YOUTUBE\outputs\relevance_round_01
```

Defaults read the saved comment/vPIC Parquets and Toyota labels/predictions. Preparation streams two source passes but holds source-sized ID/text-hash sets in memory. Freeze the snapshot during preparation. The manifest binds inputs, dependencies, seed, and split sizes.

| Output | Use |
| --- | --- |
| `source_audit.json`, `sample_manifest.json` | Inspect source integrity and sampling before labeling. |
| `toyota_error_review.csv` | Review 27 baseline errors; model predictions are diagnostic. |
| `train_labels.csv` | 400 enriched non-filler examples for development. |
| `blind_labels.csv` | 300 comments from reserved videos; keep outcomes out of model/threshold selection. |
| `second_reviewer_labels.csv` | Independent 15% subset; preserve answers before reconciliation. |
| `train_sample.csv`, `blind_sample.csv` | Frozen originals; never edit. |

Edit only `vehicle_relevant` (`yes`, `no`, `unclear`; blank means unfinished), `negative_subtype` (`creator_video`, `spam`, `generic_reaction`, `other` for `no`), and notes. Vehicle facts, questions, and comparisons are relevant; title context alone cannot make creator praise relevant. Preserve IDs/text and UTF-8 encoding.

The blind sample is comment-uniform within seeded reserved videos, excluding prior Toyota videos/text. Training is enriched; it cannot estimate prevalence. Exact duplicate separation does not resolve semantic duplicates. Aim for 100 additional genuine non-filler negatives; sampling does not guarantee them. Missing independent review remains a limitation.

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_relevance_round.py check-labels --round-dir YOUTUBE\outputs\relevance_round_01 --output YOUTUBE\outputs\relevance_round_01\label_progress_01.json
```

`check-labels` verifies IDs/text, split separation, subtypes, and completion without revealing blind class counts. Use a new progress filename each time.

## Partial annotations and a working copy

Historical recovery interprets blanks as `no` only within the user-confirmed reviewed bounds (train through spreadsheet row 180, blind through 108). It restores line-ending-only changes from checksum-verified templates and transfers labels by `comment_id`; substantive edits/conflicts fail. New annotations use explicit labels.

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\partial_relevance.py snapshot --round-dir YOUTUBE\outputs\relevance_round_01 --output-dir YOUTUBE\outputs\relevance_round_01_working_rebuild --reviewed-train-through 180 --reviewed-blind-through 108 --reviewed-blanks no
```

`relevance_round_01_working/input_backups/` preserves originals. The partial diagnostic used 179 labels (95 yes/84 no) and does not read blind annotations:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\partial_relevance.py evaluate --round-dir YOUTUBE\outputs\relevance_round_01_working --output-dir YOUTUBE\outputs\relevance_round_01_working\baseline_partial_v2
```

User-confirmed `yes`/`no` error-review notes and subtype corrections produced `relevance_round_01_reviewed` (92 yes/87 no, 221 unfinished). Subtype exports cannot undo binary corrections. `review_changes.csv` and `review_manifest.json` bind the import:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\partial_relevance.py apply-reviews --round-dir YOUTUBE\outputs\relevance_round_01_working --review-dir YOUTUBE\outputs\relevance_round_01_working\baseline_partial_v1 --output-dir YOUTUBE\outputs\relevance_round_01_reviewed_rebuild
```

Continue labels in the reviewed snapshot, preserving earlier copies. Four negative subtypes are missing and one `question` subtype is unsupported; keep binary judgments until explicitly reviewed.

## Compare models with current labels

```powershell
Set-Location E:\DataScience_Capstone_Cars
.\.venv\Scripts\python.exe -m pip install -r YOUTUBE\requirements.txt
.\.venv\Scripts\python.exe YOUTUBE\relevance\compare_relevance_models.py --round-dir YOUTUBE\outputs\relevance_round_01_reviewed --output-dir YOUTUBE\outputs\relevance_round_01_reviewed\model_comparison_v1
```

TF-IDF/logistic, fixed MiniLM/logistic, and SetFit use identical five video-grouped training folds. Neural runs pin the public encoder revision; CPU is default, CUDA optional. Blank/unclear annotations stay out of fitting and blind files are never opened. Long text truncation is reported. Completed fold predictions checkpoint; identical interrupted commands support `--resume`.

Review `model_comparison.csv`, `comparison_metrics.json`, `threshold_diagnostics.csv`, `calibration_bins.csv`, and `cv_errors.csv`. The completed v1 comparison proposed SetFit with rejection strictly below 0.20: training CV retained 91/92 relevant comments and 41/42 flagged rejections were irrelevant. These are post-review development results; intervals saved at 0.50 do not describe 0.20.

## Save the candidate and start the pilot

These stages completed for v1. Reproduction needs fresh outputs and labels matching the comparison.

```powershell
Set-Location E:\DataScience_Capstone_Cars
.\.venv\Scripts\python.exe YOUTUBE\relevance\relevance_candidate.py fit --round-dir YOUTUBE\outputs\relevance_round_01_reviewed --output-dir YOUTUBE\outputs\relevance_candidate_v1
```

Fit saves encoder/head, source/training references, preprocessing, cutoff/evidence policy, and a reload-score check. It never opens blind annotations. Failed fits require a fresh candidate name.

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\relevance_candidate.py pilot --candidate-dir YOUTUBE\outputs\relevance_candidate_v1 --output-dir YOUTUBE\outputs\qualification_pilot_v1
```

The preserving pilot contains 10,000 comments from 1,496 videos, 2,243 proposed rejections, 97 truncated model inputs, and 79 batches. All decisions remain `review_or_pass`. Training/blind/reserved videos and protected exact text are excluded. The audit has up to 100 rows per rejection/boundary/retained stratum; report strata separately or apply sampling weights. The 22.43% pilot flag rate is not source prevalence.

Pilot `--resume` reuses compatible batches and preserves edited audit labels. Inspect sample/scores/summary/manifests and the [artifact review](../outputs/pipeline_run_review_01/run_review.json) before further scoring.

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_aspect_round.py prepare --round-dir YOUTUBE\outputs\relevance_round_01_reviewed --toyota-sample YOUTUBE\outputs\TOYOTA\comment_label_sample.csv --toyota-feedback YOUTUBE\outputs\TOYOTA\comment_relevance_feedback.csv --output-dir YOUTUBE\outputs\aspect_round_01
```

The starter has 100 confirmed-relevant comments (66 reviewed-round, 34 Toyota), 100 target placeholders, and 900 aspect rows. Frozen `comment_sample.csv`/`canonical_makes.csv` supply context; edit target/aspect labels using the [structured guide](MAKE_SENTIMENT_LABEL_GUIDE.md#structured-make-and-aspect-rounds). This is development data; reserve fresh videos for sentiment evaluation.

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_aspect_round.py check-labels --round-dir YOUTUBE\outputs\aspect_round_01 --output-dir YOUTUBE\outputs\aspect_round_01\label_check_v1
```

Partial checking verifies source text, canonical targets, unique keys, polarity/evidence, and reliability basis. It exports valid human labels and reports unfinished work without fitting sentiment models.

## Evaluate the locked candidate once

Finish blind annotations/subtypes and independent review first. `--reviewer-unavailable` explicitly records a missing reviewer. Freeze the candidate, cutoff, and sufficient-evidence policy before consulting blind outcomes.

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\relevance_candidate.py evaluate --candidate-dir YOUTUBE\outputs\relevance_candidate_v1 --round-dir YOUTUBE\outputs\relevance_round_01_reviewed --output-dir YOUTUBE\outputs\relevance_candidate_v1\blind_evaluation_v1
```

The evaluator uses the fixed 0.20 cutoff. Incomplete annotations fail before claiming the test; `blind_evaluation_receipt.json` prevents a second candidate/output from reusing it. Only an incomplete identical evaluation supports `--resume`. Model/threshold revisions need development data and a fresh held-out test.

`blind_metrics.json` reports confusion/support, exact-rule comparison, approximate video-bootstrap/Wilson intervals, unclear outcomes, and reviewer limits. The [plan](MAKE_SENTIMENT_PIPELINE_PLAN.md#validation-and-implementation-sequence) defines the statistical and manual release gates. Passing statistics alone never enables source exclusions. No blind evaluation was consumed by the recorded development workflow.
