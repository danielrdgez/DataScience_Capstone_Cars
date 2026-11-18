# YouTube workflow guide

Use `text_pipeline.py` in `.venv-youtube` for new raw-source runs. [Setup and recovery](docs/GPU_TEXT_RUNBOOK.md) · [Sentiment-v2 fields and coverage](../DATA_DICTIONARY.md#selected-youtube-predictors) · [Price training](../README.md#listing-price-models)

Remaining work:

- Review attribution, content routing, and sparse aspect support on independent automotive examples. Teacher agreement measures distillation fidelity.
- Compare matched validation runs with coverage-only, semantic/topic, sentiment, and expert ablations before final testing.
- Establish historical text/score availability before claiming historical forecasting. Current enriched comparisons are retrospective.

## Earlier workflows

| Route | Reference | Use |
| --- | --- | --- |
| Pretrained nine-aspect sentiment | [Experimental plan](docs/EXPERIMENTAL_SENTIMENT_FEATURE_PLAN.md) | Preserve/reproduce the 1,000- and 10,000-comment CPU benchmarks. |
| Supervised relevance and aspect labels | [Runbook](docs/RELEVANCE_ROUND_RUNBOOK.md), [label guide](docs/MAKE_SENTIMENT_LABEL_GUIDE.md), [plan](docs/MAKE_SENTIMENT_PIPELINE_PLAN.md) | Optional independent validation; new labeling remains deferred. |
| Methods and literature | [Research review](docs/NO_LABEL_TEXT_FEATURE_RESEARCH.md) | Rationale and alternatives to source-wide BART inference. |

Historical manifests keep their recorded paths and hashes. Do not edit them to force resumes; changed code/inputs require fresh outputs. Retain annotated files, benchmark outputs, and run logs.
