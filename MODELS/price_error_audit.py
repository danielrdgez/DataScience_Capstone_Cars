"""Audit chart extremes in selected frozen validation/test prediction receipts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from MODELS.price_model_diagnostics import discover_checkpoints

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "MODEL_OUTPUT/diagnostics/error_extremes"
    output.mkdir(parents=True, exist_ok=True)
    selected, _, _ = discover_checkpoints(ROOT / "MODEL_OUTPUT")
    summaries, families, examples, sources = [], [], [], []
    for (variant, model), (run, metadata, _) in selected.items():
        for partition in ("test", "validation"):
            path = run / f"{model}_{partition}_predictions.parquet"
            frame = pd.read_parquet(path, columns=["vin", "ModelYear", "Make", "Model", "actual_price", "prediction"])
            if frame.vin.isna().any() or frame.vin.duplicated().any():
                raise ValueError(f"Invalid VIN identity: {path}")
            if not np.isfinite(frame[["actual_price", "prediction"]].to_numpy()).all():
                raise ValueError(f"Nonfinite predictions: {path}")
            frame["signed_error"] = frame.prediction - frame.actual_price
            frame["absolute_error"] = frame.signed_error.abs()
            frame["squared_error"] = frame.signed_error ** 2
            sse = frame.squared_error.sum()
            cutoff = frame.absolute_error.quantile(.99)
            counts, edges = np.histogram(frame.signed_error, bins=40)
            context = dict(feature_set=variant, model=model, partition=partition)
            summaries.append({**context, "rows": len(frame), "mae": frame.absolute_error.mean(),
                "rmse": np.sqrt(frame.squared_error.mean()), "largest_histogram_bin_pct": counts.max()/len(frame)*100,
                "minimum_error": frame.signed_error.min(), "maximum_error": frame.signed_error.max(),
                "p99_absolute_error": cutoff,
                "top_one_pct_squared_error_share_pct": frame.loc[frame.absolute_error.ge(cutoff), "squared_error"].sum()/sse*100,
                "observed_price_over_250k_rows": int(frame.actual_price.gt(250000).sum()),
                "prediction_over_250k_rows": int(frame.prediction.gt(250000).sum())})
            grouped = frame.groupby(["ModelYear", "Make", "Model"], dropna=False).agg(
                rows=("vin", "size"), mean_observed_price=("actual_price", "mean"),
                max_observed_price=("actual_price", "max"), max_prediction=("prediction", "max"),
                total_absolute_error=("absolute_error", "sum"), squared_error=("squared_error", "sum"),
                maximum_absolute_error=("absolute_error", "max"))
            grouped["squared_error_share_pct"] = grouped.squared_error / sse * 100
            grouped["mae"] = grouped.total_absolute_error / grouped.rows
            grouped = grouped.sort_values("squared_error", ascending=False).head(20).reset_index()
            families.extend([{**context, **row} for row in grouped.to_dict("records")])
            examples.extend([{**context, **row} for row in frame.nlargest(10, "absolute_error").to_dict("records")])
            sources.append({**context, "file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                            "mtime_ns": path.stat().st_mtime_ns, "split_fingerprint": metadata["split_fingerprint"],
                            "source_file_sha256": metadata["source_file_sha256"]})
        print(f"Audited {variant}/{model}", flush=True)
    pd.DataFrame(summaries).to_csv(output / "model_summary.csv", index=False)
    family_table = pd.DataFrame(families)
    family_table.to_csv(output / "top_families.csv", index=False)
    grouped = family_table.groupby(["partition", "ModelYear", "Make", "Model"], dropna=False).agg(
        model_results=("model", "size"), minimum_sse_share_pct=("squared_error_share_pct", "min"),
        maximum_sse_share_pct=("squared_error_share_pct", "max"),
        max_observed_price=("max_observed_price", "max"), max_prediction=("max_prediction", "max"),
        max_absolute_error=("maximum_absolute_error", "max"),
    ).reset_index().sort_values(["partition", "maximum_sse_share_pct"], ascending=[True, False])
    grouped.to_csv(output / "family_summary.csv", index=False)
    pd.DataFrame(examples).to_csv(output / "extreme_predictions.csv", index=False)
    (output / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
