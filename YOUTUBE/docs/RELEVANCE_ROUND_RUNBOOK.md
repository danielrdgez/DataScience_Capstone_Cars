# First execution: relevance labels before sentiment

**Current context (October 6, 2026):** this is the preserved relevance-labeling
route. New labeling is deferred; [the GPU text runbook](GPU_TEXT_RUNBOOK.md)
describes the current raw-source workflow and completed full-source artifacts.

This implements Phase 1 of [the pipeline plan](MAKE_SENTIMENT_PIPELINE_PLAN.md).
Run these commands yourself from `E:\DataScience_Capstone_Cars`, then ask Codex
to read the saved outputs. No source data is modified and no model is trained.
The code was checked on synthetic Parquet fixtures. The user completed the
real-source round 01 preparation, and its audit/manifest and split integrity
were reviewed. Partial human labeling and an exploratory diagnostic are now
available as described below.

## 1. Prepare the audit and labeling batch

The current `.venv` already has the required dependencies. If using a new
environment, follow the root README setup first. No model downloads, API key,
GPU, or new packages are required for this stage.

```powershell
Set-Location E:\DataScience_Capstone_Cars
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_relevance_round.py prepare --output-dir YOUTUBE\outputs\relevance_round_01
```

The defaults read `DATA/CAR_DATA_FINAL/Youtube_Comments.parquet`, canonical
`Make` values from `NHTSA_vPIC.parquet`, and the existing Toyota labels and
out-of-fold predictions. `--source`, `--vpic`, `--existing-labels`, and
`--predictions` can point to other **Parquet snapshots / CSV inputs explicitly**;
the script does not silently switch to SQLite. Source inputs are read-only.

It makes two streaming passes over the comments and computes file checksums.
Source-sized ID and normalized-text hash sets are held in memory during the
audit; this is not a constant-memory job. Progress prints every 250,000 audit
rows. Sampling is deterministic for a fixed seed and snapshot. The output
directory must be new: use `relevance_round_02` to preserve a prior run.
Stop collection/export while preparing so the snapshot stays stable.

| Output in `YOUTUBE/outputs/relevance_round_01/` | What to do |
| --- | --- |
| `source_audit.json` | Share first. Counts empty/duplicate text, invalid dates, text lengths, title-make contexts, filler flags, and video concentration. |
| `sample_manifest.json` | Share first. Records source hashes, dependencies, seed, sampling population, and split sizes. |
| `toyota_error_review.csv` | Review the 27 baseline errors; put explanations in `review_notes`. Predictions are diagnostic, not new gold labels. |
| `train_labels.csv` | Label 400 new non-filler comments. Candidate strata deliberately enrich difficult examples. |
| `blind_labels.csv` | Label 300 held-out comments independently of predictions. Keep these out of model/threshold selection. |
| `second_reviewer_labels.csv` | Have a second person independently label the supplied 15% subset before comparing answers. |
| `train_sample.csv`, `blind_sample.csv` | Frozen originals for integrity checks; never edit these. |

**First checkpoint:** after preparation, tell Codex “Read the relevance round 01
audit and manifest.” Codex can inspect available outputs and identify sampling
or source problems before you spend time labeling. If that session cannot read
the files, paste the two JSON outputs in chat.

## 2. Label relevance only

Edit only the last three columns in the label CSVs. Preserve every row, stable
ID, and source text, and save UTF-8 CSV. Do not add sentiment/aspect labels yet.

| Editable column | Values |
| --- | --- |
| `vehicle_relevant` | `yes`, `no`, or `unclear`; blank means unfinished. |
| `negative_subtype` | Only for `no`: `creator_video`, `spam`, `generic_reaction`, or `other`. |
| `labeler_notes` | Explain ambiguous targets, non-English text, sarcasm, mixed vehicle/creator content, or disagreements. |

Use `yes` when the comment itself discusses a vehicle, ownership, operation,
attributes, costs, reliability, or a comparison. Questions and facts count:
“Does this have a CVT?” and “CVT sucks” are relevant. “Great video” is `no` /
`creator_video`. “First!” is `no` / `generic_reaction`. Praise plus a substantive
car remark is `yes`. A different make from the title is still relevant.
Use `unclear` when you cannot confidently interpret the comment, including a
language you cannot assess. The video title alone does not establish relevance.
`candidate_makes` is tentative title context; multiple or unmatched makes are
allowed and it is never a confirmed target label.

The blind sample is comment-uniform within approximately 20% of videos reserved
by seeded hash, excluding videos/text already in the Toyota labels. It retains
filler and empty comments so the test is not enriched with handpicked negatives.
Its eventual prevalence estimate applies to that eligible reserved-video
population; report video-cluster uncertainty and the exclusions. Training is
enriched and cannot estimate prevalence. All reserved videos stay outside
training, even those with no sampled blind row. Normalized exact text duplicates
are excluded across the selected splits; semantic near duplicates/paraphrases
still need an audit before model evaluation. Canonical make title matching does
not include an alias dictionary yet; title omissions remain in both pools.

Target at least **100 additional non-filler negatives** in training. Sampling
does not guarantee that count: inspect the label report and collect another
training batch if needed while retaining the original blind set. Do not create
a new evaluation split to get better results.

Have a second reviewer annotate their CSV without seeing your labels. Preserve
their original independent answers in a separate copy before reconciliation;
use notes to record resolutions. Without a second reviewer, the code can still
report progress, but the label-quality gate remains incomplete. An unresolved
disagreement or insufficient negative support means more review, not automatic
exclusion by a model.

## 3. Validate your completed labels

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_relevance_round.py check-labels --round-dir YOUTUBE\outputs\relevance_round_01 --output YOUTUBE\outputs\relevance_round_01\label_progress_01.json
```

This reports training class counts, negative subtypes, title-make coverage,
second-reviewer disagreement, and blind completion. It checks source-text/ID
integrity and split overlap. It deliberately does not reveal blind class counts
or compare predictions against blind labels. Use a new progress filename for a
later check, e.g. `label_progress_02.json`; existing files are never overwritten.

**Second checkpoint:** tell Codex “Read relevance round 01 label progress and
training labels.” Share `label_progress_01.json` plus reviewed Toyota errors.
Keep `blind_labels.csv` out of training and threshold tuning. `unclear` rows
stay available for review and will be excluded from binary training.

## 4. Continue after the labeling gate

Next implementation compares the TF-IDF baseline, sentence embeddings plus
logistic regression, and SetFit on identical video/duplicate-grouped training
splits. We will inspect hard negatives, relevant recall, rejection precision,
uncertainty, unseen-make behavior, and runtime before locking a threshold and
checking the blind test once. No full-data scoring command is available yet.

After that gate, run a 10,000-comment qualification pilot, then implement
checkpointed scoring. Make attribution and opinion/aspect labels come next;
only validated aspect predictions feed time-bounded make features. Use the
existing plan's 98% relevant-recall / 95% rejection-precision targets as
provisional gates, with support and confidence intervals. The present Toyota
baseline is a diagnostic and is not approved for automatic rejection.

## Partial annotations and a working copy

The user reviewed training through spreadsheet row 180 and blind through row
108, and explicitly confirmed that blanks inside those ranges mean **no**.
Rows after those bounds remain unlabeled. The blind annotations were saved in
`blind_sample.csv`, and a spreadsheet save changed only line endings inside
33 training comment texts. The separate working snapshot preserves the original
input CSVs in `input_backups/`, restores canonical source fields from
checksum-verified templates, and transfers annotations by `comment_id`.
No original input file is overwritten. Recovery refuses substantive source
edits and conflicting annotations. For reproduction in a **new directory**:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\partial_relevance.py snapshot --round-dir YOUTUBE\outputs\relevance_round_01 --output-dir YOUTUBE\outputs\relevance_round_01_working_rebuild --reviewed-train-through 180 --reviewed-blind-through 108 --reviewed-blanks no
```

The ready working copy is `YOUTUBE/outputs/relevance_round_01_working/`.
Continue annotation there: training rows **181–401**, blind rows **109–301**.
Use explicit `yes`, `no`, or `unclear` going forward, so reviewed rows are
distinguishable from unfinished rows. Edit only `*_labels.csv`, never the
frozen templates. Leave the originals and input backups intact.

The first diagnostic already ran in `baseline_partial_v1/`: 179 training labels
(95 yes / 84 no), five folds grouped by video, 21 false rejections and 17 false
acceptances at the diagnostic 0.5 cutoff. It is not safe for automatic rejection
of the source pool. No blind annotations are read or scored by this evaluator.
The old Toyota labels are not merged into this new-round diagnostic.

1. Review the 38 entries in `baseline_partial_v1/cv_error_review.csv`. Add
   `review_notes`; correct a human annotation only if your review warrants it,
   by stable `comment_id` in the working `train_labels.csv`.
2. Review `baseline_partial_v1/annotation_issues.csv`. There are 45 confirmed
   negative annotations with missing subtypes plus one unsupported `question`
   subtype. Assign an allowed subtype where applicable; vehicle questions
   themselves are relevant, while a request for a reviewer to make a video
   needs context. Preserve the original binary labels until you decide.
3. Continue the 221 training rows, then the remaining 193 blind rows. The
   optional `training_review_queue.csv` prioritizes ambiguous model scores and
   retains source `csv_row`/IDs for lookup. Do not copy predictions into labels.
   Do not use training predictions to label the blind set.
4. Have a second reviewer label their subset independently, retain a copy of
   their answers, and reconcile disagreements. Aim for at least 100 additional
   genuine non-filler negative examples; current training has 84.

After more labels, rerun the diagnostic with a fresh output name:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\partial_relevance.py evaluate --round-dir YOUTUBE\outputs\relevance_round_01_working --output-dir YOUTUBE\outputs\relevance_round_01_working\baseline_partial_v2
```

Once subtypes are resolved, the original strict progress check also works:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_relevance_round.py check-labels --round-dir YOUTUBE\outputs\relevance_round_01_working --output YOUTUBE\outputs\relevance_round_01_working\label_progress_01.json
```

Use a new progress/output name for every rerun. Partial training diagnostics are
allowed before full labeling, but automatic exclusion, full model comparison,
and blind evaluation still require the planned validation gates. The partial
evaluator preserves missing/unsupported subtypes and reports them as issues
instead of inventing replacements. It never writes predicted labels back to
the annotation files or changes the six source Parquet tables.

## Compare models now with the current labels

The user asked to defer further labeling and proceed with existing annotations.
The reviewed `cv_error_review.csv` used exact `yes`/`no` entries in `review_notes`.
Those entries were imported as relevance judgments, and 45 changed negative
subtypes were applied by `comment_id`. Unchanged binary labels in subtype
exports were not allowed to undo the relevance corrections. The resulting
snapshot is `YOUTUBE/outputs/relevance_round_01_reviewed/` (92 yes, 87 no, 221
unlabeled). Five subtype issues remain; these do not block exploratory binary
training. Original and working-round input files are preserved, including the
blind annotations. `review_changes.csv` and `review_manifest.json` record the
corrections and provenance. To reproduce import, choose a new output directory:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\partial_relevance.py apply-reviews --round-dir YOUTUBE\outputs\relevance_round_01_working --review-dir YOUTUBE\outputs\relevance_round_01_working\baseline_partial_v1 --output-dir YOUTUBE\outputs\relevance_round_01_reviewed_rebuild
```

The TF-IDF candidate has already run in `relevance_round_01_reviewed/tfidf_comparison_v1/`:
76/92 relevant comments retained, 16 missed, 72/87 irrelevant comments rejected,
and 15 retained at 0.5. These are descriptive training CV counts after error
review, not independent performance estimates. No automatic rejection is enabled.

Run the three-candidate comparison without more labeling:

```powershell
Set-Location E:\DataScience_Capstone_Cars
.\.venv\Scripts\python.exe -m pip install -r YOUTUBE\requirements.txt
.\.venv\Scripts\python.exe YOUTUBE\relevance\compare_relevance_models.py --round-dir YOUTUBE\outputs\relevance_round_01_reviewed --output-dir YOUTUBE\outputs\relevance_round_01_reviewed\model_comparison_v1
```

The first neural run downloads the public
[all-MiniLM-L6-v2 encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and pins its revision. No account/key or comment upload is required. CPU is the
default; this can take several minutes or longer, depending on hardware. If a
CUDA-enabled PyTorch environment is already installed, `--device cuda` is
supported. The optional requirement file is separate from price-model packages;
exact runtime versions are saved with the comparison.

The candidates use identical five video-grouped folds:

- Character TF-IDF plus balanced logistic regression.
- Fixed, normalized sentence embeddings plus balanced logistic regression.
- [SetFit](https://huggingface.co/docs/setfit/quickstart), which starts from a
  fresh encoder per fold and fine-tunes only on that fold's training rows.
  Default limits are five pair-sampling iterations and at most 100 contrastive
  body-training steps per fold; these are an initial experiment, not tuned settings.

Training log integrations are disabled. Blank/unclear training annotations stay
out of fitting. Blind annotation files are never opened. Neural input exceeding
the encoder token limit is counted and truncated; original comment text remains
unchanged. Semantic near-duplicate/language audits and unseen-make evaluation
remain follow-up validation, not claims established by this comparison.

Completed method/fold predictions are checkpointed. If interrupted, repeat the
same comparison command with `--resume` at the end. Resuming requires unchanged
labels, code, dependency versions, and configuration. New labels/settings need
a new output name, such as `model_comparison_v2`.

After it finishes, tell Codex **“Read relevance round 01 model comparison v1.”**
Codex should read `model_comparison.csv`, `comparison_metrics.json`,
`threshold_diagnostics.csv`, `calibration_bins.csv`, and `cv_errors.csv` before
choosing the next model/threshold. Candidate title-make segments do not establish
performance on an unseen make. Video-bootstrap intervals are approximate; CV
after model-error relabeling does not replace an independent test.

The three-model run in `model_comparison_v1` completed on October 1, 2026.
The next proposed candidate is SetFit with a rejection cutoff strictly below
0.20. Training CV retained 91/92 relevant comments and correctly rejected 41
of 42 rejected rows at that cutoff. These are descriptive results, not a blind
test, and the saved bootstrap intervals apply to 0.50 rather than 0.20.

Follow the [current next-step plan](MAKE_SENTIMENT_PIPELINE_PLAN.md#next-steps-from-the-completed-candidate-and-pilot).
Fit/save a reusable candidate and run a pilot that preserves all comments with
`relevance_candidate.py`. Prepare make/aspect labeling templates in parallel
with `prepare_aspect_round.py`. Further training labeling can wait, while the remaining 193 blind
annotations, independent review, and five subtype issues need attention before
the release decision. Freeze the candidate and evidence policy before consulting
blind outcomes. Evaluate it once; if model/threshold revisions are needed, use
development data and a fresh held-out test. The completed comparison need not be
rerun, and it does not save a final model or infer make/aspect sentiment.

## Save the candidate and start the pilot

These commands are implemented and use the installed project/neural packages
and cached encoder from the completed comparison. Run from the project root in
this order. Choose fresh output names if any already exist. Fit before making
further training annotation edits: changes from the completed comparison require
a new comparison. Inputs and prior outputs are preserved.

### 1. Fit and save the current SetFit candidate

```powershell
Set-Location E:\DataScience_Capstone_Cars
.\.venv\Scripts\python.exe YOUTUBE\relevance\relevance_candidate.py fit --round-dir YOUTUBE\outputs\relevance_round_01_reviewed --output-dir YOUTUBE\outputs\relevance_candidate_v1
```

Fit uses the 179 completed labels, cached pinned encoder, five sampling
iterations, and 100 training steps. It never opens blind annotations. Outputs
include the saved body/head in `model/`, candidate manifest, source/training
references, and label-guide snapshot. Save/load scores are checked. The 0.20
cutoff and evidence policy are frozen with automatic exclusion disabled.
An interrupted/failed fit retains its incomplete directory; retry with a new
candidate name rather than `--resume`.

### 2. Score the preserving pilot

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\relevance_candidate.py pilot --candidate-dir YOUTUBE\outputs\relevance_candidate_v1 --output-dir YOUTUBE\outputs\qualification_pilot_v1
```

This samples 10,000 comments enriched across tentative title-make/year/length
strata. Training-pool/blind videos, seed-reserved videos, and protected exact
text are excluded. The source hash must match the candidate's frozen Parquet
snapshot; no silent source fallback is used. Every sampled comment is retained
with `relevance_decision=review_or_pass` and a proposed-rejection flag.

Outputs include `pilot_sample.csv`, `pilot_scores.csv`, `pilot_summary.json`,
`pilot_manifest.json`, batch checkpoints under `chunks/`, and separate
`pilot_audit_sample.csv`/`pilot_audit_labels.csv`. The audit selects up to 100
examples each from proposed rejections, retained scores below 0.50, and retained
scores at least 0.50. Assess strata separately or weight by their actual support;
neither this enriched pilot nor its audit estimates full-source prevalence.

If interrupted, repeat the same pilot command with `--resume` appended. Completed
matching batches are reused and edited audit labels remain intact. Changed
candidate/source/code/dependencies/settings need a fresh output directory.
Runtime includes inference/tokenization speed and end-of-run process RSS when
available. The user completed the real-data run on October 1: 10,000 comments
from 1,496 videos, 2,243 proposed rejections, zero exclusions, 97 truncated model
inputs, and 79 saved batches. Inference/tokenization took 117 seconds. The saved
candidate fit took 449 seconds and its reload score check matched exactly.
The [artifact review](../outputs/pipeline_run_review_01/run_review.json) verified
hashes, source fields, checkpoint scores, and protected-sample separation.

### 3. Prepare the first make/aspect annotation round

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_aspect_round.py prepare --round-dir YOUTUBE\outputs\relevance_round_01_reviewed --toyota-sample YOUTUBE\outputs\TOYOTA\comment_label_sample.csv --toyota-feedback YOUTUBE\outputs\TOYOTA\comment_relevance_feedback.csv --output-dir YOUTUBE\outputs\aspect_round_01
```

It selects up to 100 human-confirmed relevant comments from the reviewed round
and confirmed Toyota positives, deduplicating IDs/text and excluding relevance
blind sources. It never treats model scores as gold labels. Omitting both Toyota
arguments uses only the current reviewed positive pool (at most 92 before
deduplication).

Outputs are frozen `comment_sample.csv`/`canonical_makes.csv`, editable
`target_labels.csv` and `aspect_labels.csv`, and an annotation manifest. Each
comment starts with one target placeholder and nine aspect rows, so 100 comments
produce 900 aspect-template rows. This is a development sample; reserve a fresh
video-based sentiment holdout before fitting sentiment models.

These commands and the artifact review are now complete. The aspect starter
has 100 comments (66 reviewed-round and 34 Toyota), 100 target rows, and 900
aspect rows. Its template check is valid with all 100 comments unfinished.
The pilot audit has 100 proposed-rejection, 100 boundary, and 100 retained
examples; all 300 relevance annotations are unfinished. Start by checking
proposed-rejection examples for vehicle-relevant comments, then complete the
other strata. Preserve each row's stratum and support for separate or weighted
analysis. The 22.43% pilot flag rate does not estimate source-wide prevalence
or establish model accuracy. No blind evaluation has been consumed.

## Label and validate after the pilot

1. In `qualification_pilot_v1/pilot_audit_labels.csv`, edit only the human
   `vehicle_relevant`, `negative_subtype`, `language`, and `labeler_notes` fields.
   Language is an observed audit note, not an automatic exclusion rule.
2. Continue relevance labeling in **`relevance_round_01_reviewed`**. Finish the
   193 remaining blind rows (currently starting at spreadsheet row 109) without
   seeing model scores. Resolve four missing negative training subtypes and one
   unsupported `question` subtype while retaining the user's binary judgments.
   Collect independent reviewer labels and preserve their original responses
   before reconciliation. More primary training labels can wait; changes to an
   already-fitted binary label require a new candidate before evaluation.
3. Label targets and aspect opinions in `aspect_round_01` using the
   [structured label guide](MAKE_SENTIMENT_LABEL_GUIDE.md#structured-make-and-aspect-rounds).
   Preserve frozen samples/source fields; import identifiers and dates as Text
   when editing in Excel. No opinion and absent aspects create no neutral labels.

Check partial aspect annotations with a fresh output name:

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\prepare_aspect_round.py check-labels --round-dir YOUTUBE\outputs\aspect_round_01 --output-dir YOUTUBE\outputs\aspect_round_01\label_check_v1
```

This verifies canonical makes, source text/context, target/aspect uniqueness,
polarity, verbatim evidence, and reliability basis. It exports individually
valid resolved human targets/aspects and reports unfinished annotations; it
does not train sentiment models or establish held-out performance.

## Evaluate the locked candidate once

Run only after blind annotations and subtype cleanup are complete. Independent
reviewer annotations are required by default; if a reviewer is unavailable,
explicitly add `--reviewer-unavailable` and retain that limitation.

```powershell
.\.venv\Scripts\python.exe YOUTUBE\relevance\relevance_candidate.py evaluate --candidate-dir YOUTUBE\outputs\relevance_candidate_v1 --round-dir YOUTUBE\outputs\relevance_round_01_reviewed --output-dir YOUTUBE\outputs\relevance_candidate_v1\blind_evaluation_v1
```

The evaluator uses the fixed 0.20 cutoff without a threshold sweep. Every blind
row must be `yes`, `no`, or `unclear`; unclear outcomes are reported separately.
Incomplete labels fail before claiming the test. The round-level
`blind_evaluation_receipt.json` prevents reuse with another candidate/output
name. Only an incomplete identical evaluation can use `--resume`; a completed
test cannot be rerun. Model/threshold revisions need development data and a
fresh held-out test.

`blind_metrics.json` reports confusion counts, support, exact-prefilter baseline,
approximate video-bootstrap and Wilson 95% intervals, unclear outcomes, and
reviewer limitations. The prespecified screen needs at least 100 relevant
examples and 50 proposed rejections, and point values plus both lower interval
bounds meeting 98% relevant recall and 95% rejection precision. These screens do
not establish population guarantees. No automatic exclusion is enabled:
reviewer reconciliation, pilot/language/semantic-duplicate audits, and unseen-make
testing remain required release work even if statistical screens pass.

After reports and human annotations are reviewed, implement a fresh sentiment
holdout and target/opinion/aspect/polarity benchmarks. Validated models then
support resumable source-wide inference and dated, video-balanced make
aggregates. Historical price-model use still needs a verified observation
timestamp and valid temporal join.
