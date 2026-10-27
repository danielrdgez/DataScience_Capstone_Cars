"""Reviewed, family-held-out category prediction with per-label abstention gates."""
from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from MODELS import vehicle_profiles as r
from MODELS import vehicle_profiles_v5 as profiles

DIMENSIONS = ("size_segment", "market_positioning", "performance_character")
NUMERIC = ["ModelYear", *profiles.NUMERIC, *profiles.EXTRA]
CATEGORICAL = ["body_style", "wheel_system", "powertrain", "fuel_type", "transmission", "drive_family"]
POLICIES = {"conservative": (.95, .85), "broader": (.90, .80)}


def partition(family):
    value = int(r.digest("review-42|" + family)[:8], 16) % 10
    return "train" if value < 6 else "calibration" if value < 8 else "test"


def wilson(successes, count):
    if not count:
        return 0.
    z, p = 1.959963984540054, successes / count
    return (p + z*z/(2*count) - z*np.sqrt(p*(1-p)/count + z*z/(4*count*count))) / (1 + z*z/count)


def gate(correct, count, policy):
    point, lower = POLICIES[policy]
    return count >= 30 and correct / max(1, count) >= point and wilson(correct, count) >= lower


def export_review(data, labels, target):
    """Balanced evidence queue, one profile per family/grain, editable independent judgments."""
    rows = []
    proposed = labels.groupby(["profile_id", "dimension"]).label.agg(lambda x: ";".join(sorted(set(x))))
    evidence = labels.groupby(["profile_id", "dimension"]).source_url.agg(lambda x: " | ".join(sorted(set(x))))
    for grain, group in data[data.identity_complete].groupby("grain"):
        representatives = group.sort_values(["vin_count", "profile_id"], ascending=[False, True]).drop_duplicates("family_block")
        for p in representatives.to_dict("records"):
            for dimension in DIMENSIONS:
                key = p["profile_id"], dimension
                rows.append({"review_id": r.digest(grain + "|" + p["family_block"] + "|" + dimension)[:24],
                    "profile_id": p["profile_id"], "grain": grain, "family_block": p["family_block"],
                    "partition": partition(p["family_block"]), **{c: p[c] for c in r.COHORT},
                    "dimension": dimension, "proposed_labels": proposed.get(key, ""), "source_urls": evidence.get(key, ""),
                    "vin_count": int(p["vin_count"]), "reviewer": "", "verdict": "", "labels": "", "reviewed_all": "", "adjudicator": ""})
    queue = pd.DataFrame(rows)
    if queue.empty:
        raise ValueError("No reviewable families")
    queue["rarity"] = queue.groupby(["grain", "dimension", "proposed_labels"]).review_id.transform("size")
    queue = queue.sort_values(["rarity", "vin_count", "review_id"], ascending=[True, False, True]).drop(columns="rarity")
    queue.to_csv(target / "review_inventory.csv", index=False)
    for batch, start in enumerate(range(0, len(queue), 50), 1):
        part = queue.iloc[start:start+50]
        part.to_csv(target / f"review_{batch:03}.csv", index=False)
        cards = []
        for row in part.to_dict("records"):
            links = " ".join(f'<a href="{html.escape(url)}" target="_blank">Source</a>' for url in row["source_urls"].split(" | ") if url)
            cards.append(f'<tr><td>{html.escape(str(row["review_id"]))}</td><td>{html.escape(" / ".join(str(row[c]) for c in r.COHORT))}</td><td>{row["grain"]}</td><td>{row["dimension"]}</td><td>{html.escape(row["proposed_labels"])}</td><td>{links}</td></tr>')
        (target / f"review_{batch:03}.html").write_text('<!doctype html><meta charset="utf-8"><title>Independent category review</title><style>body{font:16px system-ui;margin:2rem}td,th{padding:.5rem;border-bottom:1px solid #ccc}</style><h1>Category review</h1><p>Record judgments in the matching CSV. Accept the proposed labels, correct with explicit labels, or mark unknown. Empty performance labels are negatives only when reviewed_all=true. Conflicting judgments require an adjudicator.</p><table><tr><th>ID</th><th>Vehicle</th><th>Grain</th><th>Dimension</th><th>Proposed</th><th>Evidence</th></tr>' + ''.join(cards) + '</table>', encoding="utf-8")
    return queue


def import_review(inventory, ratings, definitions):
    submitted = pd.read_csv(ratings, keep_default_na=False)
    expected = inventory.set_index("review_id")
    accepted = []
    for identity, group in submitted.groupby("review_id"):
        if identity not in expected.index:
            raise ValueError("Unknown review ID")
        base = expected.loc[identity]
        judgments = []
        for row in group.to_dict("records"):
            if not row.get("verdict"):
                continue
            if not str(row.get("reviewer", "")).strip():
                raise ValueError("Judgments require a reviewer")
            if row["verdict"] not in ("accept", "correct", "unknown"):
                raise ValueError("Verdict must be accept, correct, or unknown")
            value = base.proposed_labels if row["verdict"] == "accept" else row.get("labels", "") if row["verdict"] == "correct" else ""
            labels = sorted(set(str(value).split(";")) - {""})
            if set(labels) - set(definitions["dimensions"][base.dimension]):
                raise ValueError("Unknown category label")
            if base.dimension != "performance_character" and len(labels) > 1:
                raise ValueError("Single-label dimension has multiple labels")
            reviewed_all = str(row.get("reviewed_all", "")).lower() == "true" and row["verdict"] != "unknown"
            judgments.append({**base.to_dict(), "review_id": identity, "labels": ";".join(labels), "reviewer": row["reviewer"], "reviewed_all": reviewed_all, "adjudicator": row.get("adjudicator", "")})
        if not judgments:
            continue
        unique = {(x["labels"], x["reviewed_all"]) for x in judgments}
        if len(unique) > 1:
            adjudicated = [x for x in judgments if str(x["adjudicator"]).strip()]
            if len(adjudicated) != 1:
                raise ValueError("Conflicting review requires one explicit adjudicated judgment")
            accepted.append(adjudicated[0])
        else:
            accepted.append(judgments[0])
    return pd.DataFrame(accepted, columns=[*inventory.columns, "labels", "reviewed_all", "adjudicator"]).loc[:, lambda x: ~x.columns.duplicated()]


def features(frame, with_make=False):
    cats = CATEGORICAL + (["Make"] if with_make else [])
    result = frame.reindex(columns=NUMERIC + cats).copy()
    for c in NUMERIC:
        result[c] = pd.to_numeric(result[c], errors="coerce").astype(float)
    for c in cats:
        result[c] = result[c].fillna("__UNKNOWN__").astype(str)
    return result, cats


def estimator(kind, cats):
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=4, learning_rate=.05, random_seed=42,
            thread_count=1, verbose=False, allow_writing_files=False, cat_features=cats)
    encoder = ColumnTransformer([("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore"), cats)])
    return make_pipeline(encoder, LogisticRegression(C=1., max_iter=1500, random_state=42))


def fit_model(kind, x, y, cats, weights):
    model = estimator(kind, cats)
    if kind == "catboost":
        model.fit(x, y, sample_weight=weights)
    else:
        model.fit(x, y, logisticregression__sample_weight=weights)
    return model


def choose_model(population, y, training, dimension, positive, comparisons, outer):
    from MODELS.vehicle_similarity_v3 import family_weights
    best = None
    indices = np.flatnonzero(training.to_numpy())
    groups = population.iloc[indices].family_block
    if groups.nunique() < 3:
        return "logistic", False
    for with_make in (False, True):
        x, cats = features(population, with_make)
        for kind in ("logistic", "catboost"):
            scores = []
            for a, b in GroupKFold(n_splits=3).split(indices, groups=groups):
                ai, bi = indices[a], indices[b]
                if y.iloc[ai].nunique() < 2:
                    continue
                model = fit_model(kind, x.iloc[ai], y.iloc[ai], cats, family_weights(population.iloc[ai]))
                scores.append(f1_score(y.iloc[bi], np.asarray(model.predict(x.iloc[bi])).ravel(), average="macro", zero_division=0))
            score = float(np.mean(scores)) if scores else -1.
            comparisons.append({"grain": population.iloc[0].grain, "dimension": dimension, "label": positive,
                "method": kind, "make": with_make, "macro_f1": score, "outer_fold": outer})
            candidate = (score, not with_make, kind == "logistic", kind, with_make)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    return best[3], best[4]


def train(data, labels, reviewed, definitions, target):
    """Final test families never fit estimators, calibration, thresholds, or model selection."""
    from MODELS.vehicle_similarity_v3 import family_fold, family_weights
    predictions, metrics, comparisons = [], [], []
    data = data[data.identity_complete].copy()
    data["partition"] = data.family_block.map(partition)
    if reviewed.empty:
        r.write_json(target / "classifier_status.json", {"status": "pending_review", "active_labels": 0})
        return pd.DataFrame(columns=["profile_id", "dimension", "label", "policy", "probability", "status", "model_uri"]), pd.DataFrame()
    for grain, population in data.groupby("grain"):
        for dimension in DIMENSIONS:
            source = labels[labels.dimension.eq(dimension) & labels.status.eq("source_supported")]
            known = source.groupby("profile_id").label.agg(lambda x: ";".join(sorted(set(x))))
            gold = reviewed[reviewed.grain.eq(grain) & reviewed.dimension.eq(dimension)].set_index("profile_id")
            truth = population.profile_id.map(known).copy()
            truth.index = population.index
            for idx, row in population.iterrows():
                if row.profile_id in gold.index:
                    truth.loc[idx] = gold.loc[row.profile_id, "labels"] or np.nan
            tasks = [None] if dimension != "performance_character" else list(definitions["dimensions"][dimension])
            for positive in tasks:
                y = truth.copy()
                if positive:
                    y = truth.map(lambda v: "yes" if isinstance(v, str) and positive in v.split(";") else None)
                    for idx, row in population.iterrows():
                        if row.profile_id in gold.index and bool(gold.loc[row.profile_id, "reviewed_all"]):
                            y.loc[idx] = "yes" if positive in str(gold.loc[row.profile_id, "labels"]).split(";") else "no"
                training = population.partition.eq("train") & y.notna()
                if y[training].nunique() < 2 or population.loc[training].family_block.nunique() < 20:
                    metrics.append({"grain": grain, "dimension": dimension, "label": positive or "all", "status": "insufficient_training_support"})
                    continue
                # Separate models omit each retrieval fold, including that fold's calibration labels.
                for outer in [-1, *range(5)]:
                    allowed = population.family_block.map(family_fold).ne(outer) if outer >= 0 else pd.Series(True, index=population.index)
                    fit = training & allowed
                    if y[fit].nunique() < 2:
                        continue
                    calibration = population.partition.eq("calibration") & population.profile_id.isin(gold.index) & y.notna() & allowed
                    if calibration.sum() < 20 or y[calibration].nunique() < 2:
                        continue
                    kind, with_make = choose_model(population, y, fit, dimension, positive, comparisons, outer)
                    x, cats = features(population, with_make)
                    model = fit_model(kind, x.loc[fit], y[fit], cats, family_weights(population[fit]))
                    probabilities = model.predict_proba(x)
                    classes = [str(c) for c in model.classes_]
                    calibration = population.partition.eq("calibration") & population.profile_id.isin(gold.index) & y.notna() & allowed
                    if calibration.sum() < 20 or y[calibration].nunique() < 2:
                        continue
                    # Sigmoid calibration is fitted to held-out probabilities, never in-sample predictions.
                    calibrator = LogisticRegression(C=1., max_iter=1000).fit(np.log(np.clip(probabilities[calibration], 1e-6, 1)), y[calibration], sample_weight=family_weights(population[calibration]))
                    calibrated = calibrator.predict_proba(np.log(np.clip(probabilities, 1e-6, 1)))
                    classes = list(calibrator.classes_)
                    model_name = r.digest(f"{grain}|{dimension}|{positive}|{outer}")[:20]
                    joblib.dump({"model": model, "calibrator": calibrator, "with_make": with_make,
                        "training_families": sorted(population.loc[fit | calibration, "family_block"].unique()),
                        "excluded_fold": outer}, target / (model_name + ".joblib"))
                    maximum = calibrated.max(axis=1)
                    assigned = np.asarray(classes)[calibrated.argmax(axis=1)]
                    for policy, (precision, _) in POLICIES.items():
                        for label in classes:
                            if positive and label == "no":
                                continue
                            eligible_cal = calibration & (assigned == label)
                            threshold = None
                            for cutoff in np.unique(maximum[eligible_cal]):
                                ids = population.loc[eligible_cal & (maximum >= cutoff)].sort_values("profile_id").drop_duplicates("family_block").index
                                if len(ids) >= 10 and y.loc[ids].eq(label).mean() >= precision:
                                    threshold = float(cutoff)
                                    break
                            if threshold is None:
                                continue
                            test = population.partition.eq("test") & population.profile_id.isin(gold.index) & y.notna() & (assigned == label) & (maximum >= threshold)
                            ids = population.loc[test].sort_values("profile_id").drop_duplicates("family_block").index
                            correct, count = int(y.loc[ids].eq(label).sum()), len(ids)
                            enabled = gate(correct, count, policy)
                            actual = positive or label
                            metrics.append({"grain": grain, "dimension": dimension, "label": actual, "policy": policy, "outer_fold": outer,
                                "threshold": threshold, "test_families": count, "precision": correct / count if count else None,
                                "wilson_lower": wilson(correct, count), "recall": correct / max(1, int((population.partition.eq("test") & y.eq(label)).sum())),
                                "coverage": float(test.mean()), "brier": float(np.mean((calibrated[:, classes.index(label)][test] - y[test].eq(label).to_numpy())**2)) if count else None,
                                "status": "active" if enabled else "insufficient_reviewed_precision"})
                            if not enabled or outer < 0:
                                continue
                            selected = (assigned == label) & (maximum >= threshold) & truth.isna()
                            if positive:
                                selected &= ~y.eq("no")
                            for offset in np.flatnonzero(selected.to_numpy()):
                                predictions.append({"profile_id": population.iloc[offset].profile_id, "dimension": dimension, "label": actual,
                                    "policy": policy, "probability": float(maximum[offset]), "status": "predicted", "model_uri": "model://" + model_name,
                                    "outer_fold": outer})
    pd.DataFrame(comparisons).to_csv(target / "classifier_comparison.csv", index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(target / "classifier_metrics.csv", index=False)
    result = pd.DataFrame(predictions, columns=["profile_id", "dimension", "label", "policy", "probability", "status", "model_uri", "outer_fold"])
    r.write_json(target / "classifier_status.json", {"status": "evaluated", "active_predictions": len(result), "human_review_required": True})
    return result, metrics
