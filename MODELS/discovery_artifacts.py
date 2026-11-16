"""Immutable discovery receipts and display-only projections; no work at import."""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSION = "6.0.0"
TABLES = ("used_cars", "NHTSA_vPIC", "NHTSA_Complaints", "NHTSA_Recalls", "NHTSA_Safety")
FAMILY = ["ModelYear", "Make", "Model"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        return item
    payload = json.dumps(clean(value), indent=2, default=str, allow_nan=False) + "\n"
    # Use a closed, unique sibling file for atomic replacement on Windows.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        for attempt in range(9):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                # Readers or scanners can briefly lock a recently written receipt.
                if attempt == 8:
                    raise
                time.sleep(min(.05 * 2 ** attempt, 1.))
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes(directory):
    return {name + ".parquet": file_hash(Path(directory) / (name + ".parquet")) for name in TABLES}


def vector_path(directory):
    directory = Path(directory)
    store = directory / "vector_store.json"
    if store.exists():
        path = (directory / read_json(store)["relative_path"]).resolve()
        if not path.is_relative_to(directory.parent.parent.resolve()):
            raise ValueError("Vector store escapes its channel directory")
        return path
    return directory / "vectors.npy"


def normalize_families(frame):
    result = frame.copy()
    result["ModelYear"] = pd.to_numeric(result.ModelYear, errors="coerce").astype("Int64")
    for field in FAMILY[1:]:
        result[field] = result[field].astype("string").str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    result["family_id"] = (result[FAMILY].astype("string").fillna("").agg("|".join, axis=1)
                           if len(result) else pd.Series(index=result.index, dtype="string"))
    if "family_block" not in result:
        result["family_block"] = (result[FAMILY[1:]].astype("string").fillna("").agg("|".join, axis=1)
                                 if len(result) else pd.Series(index=result.index, dtype="string"))
    result["fold"] = result.family_block.map(lambda x: int(digest("42|" + x)[:16], 16) % 5)
    return result


def producer_hashes():
    paths = [Path(__file__), ROOT / "MODELS/vehicle_similarity_v6.py", ROOT / "MODELS/vehicle_similarity_v7.py", ROOT / "NHTSA/discovery_text.py",
             ROOT / "NHTSA/text_features.py", ROOT / "MODELS/vehicle_similarity_v5.py",
             ROOT / "MODELS/vehicle_similarity_v3.py", ROOT / "MODELS/vehicle_profiles_v5.py",
             ROOT / "MODELS/vehicle_taxonomy_v2.py", ROOT / "MODELS/vehicle_taxonomy.py", ROOT / "EDA/DATA_CLEANING.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def dependency(path):
    path = Path(path).resolve()
    return {"directory": str(path), "sha256": file_hash(path / "manifest.json")}


def start(path, stage, configuration, dependencies=(), resume=False):
    path = Path(path)
    # Compare the same JSON representation that is persisted (tuples become lists).
    configuration = json.loads(json.dumps(configuration))
    contract = {"schema_version": VERSION, "stage": stage, "configuration": configuration,
                "dependencies": [dependency(p) for p in dependencies], "code_hashes": producer_hashes(),
                "package_versions": {name: version(name) for name in ("numpy", "pandas", "polars", "scikit-learn", "scipy", "joblib")}}
    if resume:
        if read_json(path / "stage_state.json") != contract:
            raise ValueError("Resume requires identical inputs, code and configuration")
        if (path / "manifest.json").exists():
            raise ValueError("Completed stages are immutable; use a fresh output directory")
    else:
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "stage_state.json", contract)
    return path


def finish(path, **details):
    path = Path(path)
    contract = read_json(path / "stage_state.json")
    if contract["code_hashes"] != producer_hashes():
        raise ValueError("Producer code changed while the stage was running")
    artifacts = {p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*"))
                 if p.is_file() and p.name not in ("manifest.json", "stage_state.json") and not p.name.endswith(".tmp")}
    receipt = {**contract, **details, "status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
               "artifacts": artifacts}
    write_json(path / "manifest.json", receipt)
    return receipt


def verify(path, full=True, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        return read_json(path / "manifest.json")
    seen.add(path)
    receipt = read_json(path / "manifest.json")
    if receipt.get("schema_version") != VERSION or receipt.get("status") != "completed":
        raise ValueError("Incomplete or incompatible discovery stage: " + str(path))
    for name, expected in receipt["artifacts"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or not target.is_file():
            raise ValueError("Invalid artifact path: " + name)
        if full and file_hash(target) != expected:
            raise ValueError("Changed artifact: " + str(target))
    for dep in receipt["dependencies"]:
        if file_hash(Path(dep["directory"]) / "manifest.json") != dep["sha256"]:
            raise ValueError("Changed discovery dependency")
        if full:
            verify(dep["directory"], full=True, seen=seen)
    return receipt


def project(path, matrix, ids, training_mask, names=None, standardize=False, prototypes=None):
    """Fit a display projection only on development rows; save the actual arrow basis."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=float)
    training_mask = np.asarray(training_mask, dtype=bool)
    empty = pd.DataFrame(columns=["entity_id", "x", "y"])
    arrows = pd.DataFrame(columns=["label", "x", "y", "definition"])
    if matrix.ndim != 2 or matrix.shape[1] < 2 or training_mask.sum() < 3:
        empty.to_parquet(path / "points.parquet", index=False)
        arrows.to_parquet(path / "arrows.parquet", index=False)
        write_json(path / "projection.json", {"status": "insufficient_support"})
        return
    observed = np.isfinite(matrix).sum(axis=1) >= 2
    # pandas can expose a read-only view. Combine masks without mutating caller-owned storage.
    training_mask = training_mask & observed
    if training_mask.sum() < 3:
        return project(path, np.empty((0, 0)), [], [], names=None)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(matrix[training_mask])
    values = imputer.transform(matrix)
    scaler = StandardScaler().fit(values[training_mask]) if standardize else None
    x = scaler.transform(values) if scaler is not None else values
    if np.var(x[training_mask], axis=0).sum() <= 1e-12:
        return project(path, np.empty((0, 0)), [], [], names=None)
    reducer = PCA(n_components=2, svd_solver="full").fit(x[training_mask])
    xy = reducer.transform(x)
    points = pd.DataFrame({"entity_id": list(ids), "x": xy[:, 0], "y": xy[:, 1]})
    points = points.loc[observed]
    if names is not None:
        # Correlation loadings express direction in standardized input space.
        basis = reducer.components_.T * np.sqrt(reducer.explained_variance_)
        order = np.argsort(-np.linalg.norm(basis, axis=1))[:8]
        arrows = pd.DataFrame([{"label": names[i], "x": float(basis[i, 0]), "y": float(basis[i, 1]),
                                "definition": "PCA feature loading"} for i in order])
    elif prototypes is not None:
        entries = []
        for label, vector in prototypes.items():
            transformed = imputer.transform(np.asarray(vector).reshape(1, -1))
            transformed = scaler.transform(transformed) if scaler is not None else transformed
            value = reducer.transform(transformed)[0]
            entries.append({"label": str(label), "x": float(value[0]), "y": float(value[1]),
                            "definition": "Projected development theme prototype"})
        arrows = pd.DataFrame(entries, columns=arrows.columns)
    points.to_parquet(path / "points.parquet", index=False)
    arrows.to_parquet(path / "arrows.parquet", index=False)
    joblib.dump({"imputer": imputer, "scaler": scaler, "reducer": reducer}, path / "projection.joblib")
    write_json(path / "projection.json", {"status": "completed", "explained_variance_ratio": reducer.explained_variance_ratio_.tolist(),
        "fit_entities": [str(v) for v in np.asarray(ids)[training_mask]], "display_only": True,
        "radar_min": np.nanquantile(matrix[training_mask], .01, axis=0).tolist(),
        "radar_max": np.nanquantile(matrix[training_mask], .99, axis=0).tolist()} )
