# Make-level YouTube comment pilot: labeling guide

**Current context (October 6, 2026):** new manual labeling is deferred. This guide
is retained for the optional supervised pilot. New raw-source processing follows
[the GPU text runbook](GPU_TEXT_RUNBOOK.md) and does not require this annotation round.

This pilot first labels whether a comment discusses a vehicle, then asks whether
it expresses an opinion about the target make.
All collected videos are car reviews; video eligibility is not filtered. It does
not measure reliability or change selected model inputs.
Keep `comment_id`, `video_id`, and the original `text` unchanged in the sample.

## Video make context

`video_make_coverage.csv` reports videos whose titles contain the requested
vPIC `Make` or an alias. These videos provide a candidate pool for sampling;
they do not receive a sentiment label as a whole. Comparisons stay in the pool.
A comment may concern the target make, another car, the reviewer, or neither.
Titles that omit the make are outside this first make-specific sample and can be
covered by later make mapping or comment target detection.

## Label the comment sample

The sample deliberately includes off-topic comments. Do not delete them: they
are needed to evaluate the relevance filter. `sample_stratum` says whether a
row came from the cross-video sample or the audit of proposed filler skips.
`prefilter_reason` is a proposed skip reason, not a correct label.

| Column | Allowed values / guidance |
| --- | --- |
| `comment_topic` | `target_make`, `other_vehicle`, `vehicle_specific`, `video_creator`, `off_topic`, `not_vehicle_specific`, or `unclear`. Use `vehicle_specific` when the user confirms the comment is about a vehicle but has not labeled which make it concerns. Use `target_make` only when the comment actually discusses the requested make; a video title alone does not establish this. Use `not_vehicle_specific` for confirmed exclusions whose more specific topic was not labeled. |
| `opinion_expressed` | `yes`, `no`, or `unclear` about the target make. Questions and factual statements can be relevant without expressing sentiment. |
| `referenced_make` | Exact make named or clearly implied by the comment; leave blank if unclear. A comment may compare makes; explain that in `labeler_notes`. |
| Aspect columns | Leave blank when absent. For an expressed opinion about the target make, use `positive`, `negative`, `neutral`, or `mixed`. Multiple aspect columns may be filled in one row. |
| `reliability_claim_basis` | For a reliability opinion: `firsthand_owner`, `secondhand`, `speculation`, or `unclear`. Otherwise leave blank. This is evidence type, not a measured defect rate. |
| `evidence_quote` | Short phrase supporting the aspect and polarity. Keep it verbatim. |
| `labeler_notes` | Explain sarcasm, ambiguity, comparisons, or a suspected filter error. |

Reliability concerns failures, dependability, durability, and recurring faults.
`maintenance_cost` concerns repair expense or service burden; `price_value`
concerns purchase price, depreciation, and value. A remark such as “CVT sucks”
is short but potentially relevant; it must not be excluded by a length rule.
“Great video!” is about the creator or video, even if the video features a car.

Toyota feedback is keyed by `comment_id`; `vehicle_specific` confirms relevance only, and `not_vehicle_specific` excludes aspect scoring. It does not establish make, opinion, or polarity. The [relevance runbook](RELEVANCE_ROUND_RUNBOOK.md) owns label recovery, review, and classifier validation.

## Structured make and aspect rounds

`prepare_aspect_round.py` creates a new development sample from confirmed
relevant comments; relevance blind annotations are not read. Use
`comment_sample.csv` and `canonical_makes.csv` as frozen references. The new
round does not require editing the older Toyota sample or its relevance feedback.

In `target_labels.csv`, one initial row has `target_index=1` for each comment.
Edit only `target_resolution_status`, `Make`, `opinion_expressed`, and notes.
Use `resolved` with an exact canonical Make when evidence supports the target;
use `ambiguous` or `unresolved` with blank Make/opinion otherwise. Leave pending
rows blank. Title context may resolve a clear reference, but title matches alone
are not target labels. A resolved target can have opinion=yes, no, or unclear.
For a multi-make comparison, copy that comment's target row with index 2 (then
3, etc.) and annotate the other target. Every index must be a positive integer,
and the same comment cannot have two resolved rows for the same Make.

`aspect_labels.csv` starts with nine rows per initial target. Match records by
`comment_id` and `target_index`, not spreadsheet position. For a resolved target
with opinion=yes, enter an `aspect_present` value for every aspect:

| Field | Meaning |
|---|---|
| `aspect_present=yes` | The comment expresses an opinion on this aspect for this target. Supply polarity and verbatim evidence. |
| `aspect_present=no` | No opinion on this aspect for this target. Leave polarity, evidence, and basis blank. |
| `aspect_present=unclear` | Human uncertainty about the aspect opinion. Leave polarity/evidence/basis blank and explain in notes. |
| Blank `aspect_present` | Unfinished when the target has opinion=yes. It does not mean no or neutral. |
| `polarity` | Only for a present opinion: `positive`, `negative`, `neutral`, or `mixed`. |
| `evidence_quote` | A verbatim phrase from the original comment supporting the target/aspect judgment. The checker verifies source containment, not semantic correctness. |
| `reliability_claim_basis` | Only for a present reliability opinion: `firsthand_owner`, `secondhand`, `speculation`, or `unclear`. |

Targets with opinion=no/unclear or unresolved targets need no aspect annotations;
leave their aspect cells blank. To annotate another make, copy all nine aspect
rows for the comment and change their target index. Preserve source/context and
the predefined aspect values. An aspect is interpreted for the Make on its
target row; the later export supplies that Make rather than requiring it in each
aspect template row.

For example, "The interior looks great but servicing it is expensive" can have
design=positive and maintenance_cost=negative for the resolved target. Reliability
is not automatically negative because maintenance costs are high. "Does it have
adaptive cruise control?" may be relevant with opinion=no and no aspect sentiment.

`check-labels` accepts valid partial annotations, reports unfinished comments,
and exports resolved human `(comment_id, Make)` targets and
`(comment_id, Make, aspect)` opinions. No absent aspect is exported as neutral.
Independent review, sufficient aspect/polarity support, and a fresh video-based
holdout remain necessary before sentiment model development.
