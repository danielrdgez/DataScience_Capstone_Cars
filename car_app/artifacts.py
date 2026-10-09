"""Read saved evaluations without loading or retraining estimators."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from car_app.data import ROOT, COHORT, Filters, snapshot

DEFAULT_BASE = "base/20260930T002248817124Z-pilot-100000"
DEFAULT_ENRICHED = "enriched/20260930T011439171946Z-pilot-100000"


@dataclass
class Run:
    path: Path
    metadata: dict
    legacy: bool = False

    @property
    def name(self):
        return self.path.relative_to(ROOT / "MODEL_OUTPUT").as_posix()

    def table(self, name):
        if name not in {"metrics.csv", "validation_metrics.csv", "segment_metrics.csv",
                        "year_make_metrics.csv", "validation_segment_metrics.csv",
                        "validation_year_make_metrics.csv", "validation_make_price_metrics.csv", "validation_focus_make_metrics.csv",
                        "validation_feature_comparison.csv", "validation_weighting_comparison.csv",
                        "youtube_join_coverage.csv"}:
            raise ValueError("Unsupported report")
        file = self.path / name
        return pd.read_csv(file) if file.is_file() and file.stat().st_size > 2 else pd.DataFrame()

    def prediction_path(self, model):
        if model not in set(self.table("metrics.csv").get("model", [])):
            return None
        file = self.path / f"{model}_test_predictions.parquet"
        return file if file.is_file() else None


@st.cache_data(max_entries=4, ttl=900, show_spinner=False)
def inventory(refresh=0):
    runs, ignored = [], []
    for file in sorted((ROOT / "MODEL_OUTPUT").rglob("run_metadata.json")):
        try:
            meta = json.loads(file.read_text(encoding="utf-8-sig"))
            status = meta.get("status")
            if status not in (None, "completed", "validation_complete", "test_complete"):
                ignored.append((str(file.parent.relative_to(ROOT / "MODEL_OUTPUT")), status))
                continue
            legacy = status is None or (meta.get("model_type") is not None
                                         and meta.get("preprocessing_version") != "advertised-price-log-v3")
            run = Run(file.parent, meta, legacy)
            if not run.table("metrics.csv").empty or not run.table("validation_metrics.csv").empty:
                runs.append(run)
        except (ValueError, OSError, pd.errors.ParserError):
            ignored.append((str(file.parent.relative_to(ROOT / "MODEL_OUTPUT")), "unreadable"))
    return runs, ignored


def errors(frame: pd.DataFrame):
    if frame.empty:
        return {"rows": 0, "family_blocks": 0, "support": "no_rows",
                **{k: None for k in ["mae", "rmse", "r2", "wape_pct", "mean_signed_error", "p95_absolute_error"]}}
    actual = frame.actual_price.to_numpy(dtype=float)
    predicted = frame.prediction.to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation contains non-finite prices")
    delta = predicted - actual
    families = frame["family_block"].nunique(dropna=False) if "family_block" in frame else None
    return {
        "rows": len(frame), "family_blocks": families,
        "support": "supported" if len(frame) >= 100 and families is not None and families >= 5 else "low_support",
        "mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "r2": float(1 - np.sum(delta ** 2) / np.sum((actual - actual.mean()) ** 2))
              if len(actual) > 1 and np.ptp(actual) > 0 else None,
        "wape_pct": float(np.abs(delta).sum() / actual.sum() * 100) if actual.sum() > 0 else None,
        "mean_signed_error": float(delta.mean()),
        "p95_absolute_error": float(np.quantile(np.abs(delta), .95)),
    }


@st.cache_data(max_entries=4, ttl=900, show_spinner=False)
def selected_vins(source_identity: str, filters: Filters):
    source = snapshot()
    if source.identity != source_identity:
        raise ValueError("Snapshot changed; refresh the app")
    return source.catalog(filters).select("vin").collect()


@st.cache_data(max_entries=24, ttl=900, show_spinner=False)
def evaluation(file: str, signature: tuple, source_identity: str, filters: Filters):
    predictions = pl.scan_parquet(file)
    if filters.active():
        # Use the frozen receipt for cohort identity/actual price; current source
        # attributes may supplement filters only after snapshot hashes agree.
        run_path = Path(file).parent
        meta = json.loads((run_path / "run_metadata.json").read_text())
        validate_sources(meta)
        selected = selected_vins(source_identity, filters)
        predictions = predictions.join(selected.lazy(), on="vin", how="semi")
    frame = predictions.collect().to_pandas()
    if frame.vin.duplicated().any():
        raise ValueError("Duplicate evaluation VINs")
    return frame


def paired_comparison(left: Run, right: Run, filters, source_identity):
    from MODELS.Price_ML_Models import compatible_run_metadata
    if not compatible_run_metadata(left.metadata, right.metadata, allow_feature_set=True):
        raise ValueError("Run fingerprints, split strategies, or training settings differ")
    models = sorted(set(left.table("metrics.csv").model) & set(right.table("metrics.csv").model))
    rows = []
    for model in models:
        lp, rp = left.prediction_path(model), right.prediction_path(model)
        if lp is None or rp is None:
            continue
        a = evaluation(str(lp), stat_key(lp), source_identity, Filters())
        b = evaluation(str(rp), stat_key(rp), source_identity, Filters())
        paired = a.merge(b, on="vin", validate="one_to_one", suffixes=("_a", "_b"))
        if len(paired) != len(a) or len(a) != len(b):
            raise ValueError("Evaluation populations differ")
        for column in ["actual_price", *COHORT, "split_group"]:
            aa, bb = paired[f"{column}_a"], paired[f"{column}_b"]
            if not (aa.eq(bb) | (aa.isna() & bb.isna())).all():
                raise ValueError(f"Evaluation {column} values differ")
        af = evaluation(str(lp), stat_key(lp), source_identity, filters)
        bf = evaluation(str(rp), stat_key(rp), source_identity, filters)
        ea, eb = errors(af), errors(bf)
        rows.append({"model": model, "rows": ea["rows"], "base_mae": ea["mae"],
                     "enriched_mae": eb["mae"],
                     "enriched_minus_base_mae": eb["mae"] - ea["mae"] if ea["rows"] else None})
    return pd.DataFrame(rows)


def stat_key(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


@st.cache_data(max_entries=16, ttl=900, show_spinner=False)
def hash_file(path: str, signature: tuple):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(metadata):
    hashes = metadata.get("source_file_sha256")
    if not hashes:
        raise ValueError("No snapshot hashes recorded; filtered evaluation/inference unavailable")
    source = snapshot()
    for name, expected in hashes.items():
        path = source.directory / name
        if path.parent != source.directory or not path.is_file():
            raise ValueError("Invalid source reference in artifact metadata")
        if hash_file(str(path), stat_key(path)) != expected:
            raise ValueError(f"Source fingerprint changed: {name}")
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if fingerprint != metadata.get("dataset_fingerprint"):
        raise ValueError("Dataset fingerprint does not match recorded source hashes")


def saved_shap(run, model):
    if model not in set(run.table("metrics.csv").get("model", [])):
        return pd.DataFrame()
    files = list(run.path.glob(f"{model}/{model}_shap_global.csv"))
    return pd.read_csv(files[0]) if files else pd.DataFrame()
