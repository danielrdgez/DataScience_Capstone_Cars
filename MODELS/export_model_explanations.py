"""Save notebook-compatible explanations for completed price-model checkpoints.

Run from the project root with ``python -m MODELS.export_model_explanations``.
Existing exact-signature caches are reused; no model is trained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from MODELS.price_model_diagnostics import (
    FEATURE_SETS, MODELS, collect_importance, discover_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--rows", type=int, default=100, help="Frozen test rows for booster SHAP")
    parser.add_argument("--forest-method", choices=["impurity", "shap"], default="impurity",
                        help="Notebook default is impurity; exact forest SHAP can take tens of minutes")
    parser.add_argument("--forest-rows", type=int, default=20)
    args = parser.parse_args(argv)
    if args.rows < 1 or args.forest_rows < 1:
        parser.error("Explanation row counts must be positive")
    output = ROOT / "MODEL_OUTPUT/diagnostics/model_evaluations"
    selected, _, _ = discover_checkpoints(ROOT / "MODEL_OUTPUT")
    selected = {key: value for key, value in selected.items()
                if key[0] in args.feature_sets and key[1] in args.models}
    if not selected:
        parser.error("No matching completed checkpoints are available")
    output.mkdir(parents=True, exist_ok=True)
    importance, statuses = collect_importance(
        selected, output / "explanations", rows=args.rows,
        forest_rows=args.forest_rows, forest_method=args.forest_method,
    )
    # Keep exports separate from notebook summaries when a subset is requested.
    importance.to_csv(output / "exported_importance.csv", index=False)
    statuses.to_json(output / "exported_explanations.json", orient="records", indent=2)
    print(statuses[["feature_set", "model", "status"]].to_string(index=False))
    print(f"Saved caches: {output / 'explanations'}")
    failed = statuses[statuses.status.ne("completed")]
    if not failed.empty:
        for row in failed.to_dict("records"):
            print(json.dumps(row, default=str))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
