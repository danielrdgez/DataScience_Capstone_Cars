"""Exploratory specification groups without v6 eligibility or rejection gates.

Use the v6 artifact contract and evidence workflow, recording grouping_version=7.
Profile medians/modes bypass support masking. Source cleaning, text-evidence
support and price-model policies remain separate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from MODELS import vehicle_similarity_v3 as v3
from MODELS import vehicle_taxonomy_v2 as v2


def select_fields(data):
    from MODELS.vehicle_similarity_v6 import NUMERIC

    training = data.loc[data.development]
    fields, audit = [], []
    for field in NUMERIC + v2.CATEGORICAL:
        if field not in data:
            continue
        values = training[field]
        # Constant and entirely absent development fields cannot separate groups.
        selected = values.nunique() > 1
        if selected:
            fields.append(field)
        audit.append({"field": field, "development_coverage": float(values.notna().mean()),
                      "reason": "selected" if selected else "no_development_variation"})
    return fields, audit


def discover(training, fields, config):
    """Choose the best available silhouette; never reject a fitted partition."""
    constant = {"method": "constant", "fields": fields, "fit_families": sorted(training.family_block.unique())}
    if len(training) < 2 or not fields:
        return {"kmeans": constant}, [{"method": "constant", "k": 1,
                "status": "exploratory_single_group"}], []
    # Retain deterministic family weighting, development-only encoding and scaling.
    ks = [k for k in config.clusters if k < min(len(training), config.vehicle_cap)]
    ks = ks or [1]
    with threadpool_limits(limits=1):
        candidates = v3.fit_candidates(training, fields, "kmeans", config, 42, ks, balanced=True)
        records = []
        sample = training.sample(min(config.anchor_cap, len(training)), random_state=42)
        for k, model in candidates.items():
            x = model["encoder"].transform(v3.clip_frame(sample, model["metric"]).replace({None: np.nan}))
            labels = model["model"].predict(x)
            score = silhouette_score(x, labels) if 1 < len(set(labels)) < len(labels) else -1.
            records.append({"method": "kmeans", "k": k, "silhouette": float(score),
                            "status": "exploratory_candidate"})
    best = max(records, key=lambda row: (row["silhouette"], -row["k"]))
    best["status"] = "exploratory_selected"
    return {"kmeans": candidates[best["k"]]}, records, []


def assign(model, population):
    """Assign every row, including missing/conflicting profiles and distant rows."""
    if model["method"] == "constant":
        return np.zeros(len(population), dtype=int)
    if population.empty:
        return np.array([], dtype=int)
    values = v3.clip_frame(population, model["metric"]).replace({None: np.nan})
    return model["model"].predict(model["encoder"].transform(values))


def project(path, data, fields):
    """Include every grouped profile in the saved map, even without observations."""
    from MODELS import discovery_artifacts as a

    path = Path(path)
    training = data.development.to_numpy(bool)
    values = data[fields].astype(float)
    medians = values.loc[training].median().fillna(0.)
    matrix = values.fillna(medians).to_numpy(float)
    if len(fields) >= 2 and training.sum() >= 3 and np.var(matrix[training], axis=0).sum() > 1e-12:
        a.project(path, matrix, data.profile_id.to_numpy(), training, names=fields, standardize=True)
        metadata = a.read_json(path / "projection.json")
        metadata["note"] = "V7 map includes every profile using development-median imputation; coordinates are exploratory."
        a.write_json(path / "projection.json", metadata)
    else:
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"entity_id": data.profile_id, "x": 0., "y": 0.}).to_parquet(path / "points.parquet", index=False)
        pd.DataFrame(columns=["label", "x", "y", "definition"]).to_parquet(path / "arrows.parquet", index=False)
        a.write_json(path / "projection.json", {"status": "single_location", "display_only": True,
                     "fit_entities": data.loc[data.development, "profile_id"].tolist(),
                     "note": "Every profile is shown at the origin because this context cannot fit a two-axis numeric projection."})


if __name__ == "__main__":
    from MODELS.vehicle_similarity_v6 import main

    main(grouping_version=7)
