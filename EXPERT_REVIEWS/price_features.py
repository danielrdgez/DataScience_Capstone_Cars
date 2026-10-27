"""Automatic DougScore cohort ratings and balanced make baselines for price models."""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
from EXPERT_REVIEWS.prepare_dougscore import ROOT, WEEKEND, DAILY, TOTALS, digest, norm, write_json

KEYS = ('ModelYear', 'Make', 'Model')
CATEGORIES = WEEKEND + tuple(c for c in DAILY if c != 'value')
SCORE_COLUMNS = tuple(f'doug_{c}' for c in CATEGORIES)
COVERAGE_COLUMNS = ('doug_source_level', 'doug_review_count', 'doug_model_count',
                    'doug_cohort_match_available', 'doug_make_baseline_available')
FEATURE_COLUMNS = SCORE_COLUMNS + COVERAGE_COLUMNS
DEFAULT_DIRECTORY = ROOT / 'EXPERT_REVIEWS/outputs/price_features_v2'
VERSION = 'doug-price-v2-year-make-model'
# Existing explicit YouTube make aliases; no inferred brand consolidation.
MAKE_ALIASES = {'vw': 'volkswagen', 'chevy': 'chevrolet', 'mercedes': 'mercedes-benz'}


def make_key(value):
    value = norm(value)
    return MAKE_ALIASES.get(value, value)


def resolve_reviews(reviews, cohorts):
    """Resolve year/make/model only; the source has no separate trim column."""
    index = defaultdict(set)
    for row in cohorts:
        if row['ModelYear'] is not None and norm(row['Make']) and norm(row['Model']):
            index[(int(row['ModelYear']), make_key(row['Make']))].add(norm(row['Model']))
    resolved = []
    for review in reviews:
        scores = [review[c] for c in (*WEEKEND, *DAILY)]
        if any(not isinstance(s, (int, float)) or not math.isfinite(s) or not 1 <= s <= 10 for s in scores):
            raise ValueError('Invalid imported DougScore category')
        if (sum(scores[:5]) != review['weekend_total'] or sum(scores[5:]) != review['daily_total']
                or sum(scores) != review['dougscore']):
            raise ValueError('Imported DougScore totals do not reconcile')
        year, make, variant = int(review['ModelYear']), make_key(review['make_raw']), norm(review['model_raw'])
        choices = index[(year, make)]
        models = {m for m in choices if variant == m or variant.startswith(m + ' ')}
        longest = max(map(len, models), default=0)
        models = {m for m in models if len(m) == longest}
        model = next(iter(models)) if len(models) == 1 else None
        resolved.append({**review, 'Make': make, 'Model': model,
                         'match_status': 'automatic_exact' if model and variant == model else 'automatic_model_prefix' if model else 'unmatched',
                         'match_rule': 'year_make_model_exact' if model and variant == model else 'unique_longest_model_prefix' if model else 'unresolved',
                         'candidate_model_count': len(models), 'conflict_excluded': False})
    groups = defaultdict(list)
    for i, row in enumerate(resolved):
        if row['Model']:
            groups[vehicle_key(row)].append(i)
    for indices in groups.values():
        if len({tuple(resolved[i][c] for c in (*WEEKEND, *DAILY, *TOTALS)) for i in indices}) > 1:
            for i in indices:
                resolved[i]['conflict_excluded'] = True
    return resolved


def vehicle_key(row):
    return (row['ModelYear'], row['Make'], row['Model'], norm(row['model_raw']))


def feature_tables(resolved):
    by_cohort, by_model, seen = defaultdict(list), defaultdict(list), set()
    for row in resolved:
        identity = vehicle_key(row)
        if not row['Model'] or row['conflict_excluded'] or identity in seen:
            continue
        seen.add(identity)
        by_model[(row['Make'], row['Model'])].append(row)
        by_cohort[tuple(row[k] for k in KEYS)].append(row)
    cohort_rows = [{**dict(zip(KEYS, key)), **{f'doug_{c}': sum(r[c] for r in rows) / len(rows) for c in CATEGORIES},
                    'doug_review_count': len(rows), 'doug_model_count': 1} for key, rows in sorted(by_cohort.items())]
    makes = defaultdict(list)
    for (make, model), rows in by_model.items():
        makes[make].append(({c: sum(r[c] for r in rows) / len(rows) for c in CATEGORIES}, len(rows)))
    make_rows = [{'Make': make, **{f'doug_{c}': sum(m[0][c] for m in models) / len(models) for c in CATEGORIES},
                  'doug_review_count': sum(m[1] for m in models), 'doug_model_count': len(models)}
                 for make, models in sorted(makes.items()) if len(models) >= 5]
    values = {**{c: pl.Float64 for c in SCORE_COLUMNS}, 'doug_review_count': pl.Int64, 'doug_model_count': pl.Int64}
    return (pl.DataFrame(cohort_rows, schema={'ModelYear': pl.Int64, 'Make': pl.String, 'Model': pl.String, **values}),
            pl.DataFrame(make_rows, schema={'Make': pl.String, **values}))


def build_features(snapshot, vpic, output):
    snapshot, vpic, output = map(Path, (snapshot, vpic, output))
    manifest_path, reviews_path = snapshot / 'import_manifest.json', snapshot / 'expert_review_scores.parquet'
    imported = json.loads(manifest_path.read_text(encoding='utf-8'))
    if digest(reviews_path) != imported['files_sha256'][reviews_path.name]:
        raise ValueError('Imported expert reviews changed; re-audit the snapshot')
    paths = {'reviews': reviews_path, 'vpic': vpic, 'import_manifest': manifest_path}
    hashes = {key: digest(path) for key, path in paths.items()}
    reviews = pl.read_parquet(reviews_path).to_dicts()
    if len({r['review_record_id'] for r in reviews}) != len(reviews):
        raise ValueError('Duplicate review_record_id')
    cohorts = pl.scan_parquet(vpic).select(KEYS).unique().collect().to_dicts()
    resolved = resolve_reviews(reviews, cohorts)
    cohort_features, make_features = feature_tables(resolved)
    validate_tables(cohort_features, make_features)
    if hashes != {key: digest(path) for key, path in paths.items()}:
        raise RuntimeError('Expert sources changed during construction')
    output.mkdir(parents=True, exist_ok=False)
    cohort_features.write_parquet(output / 'cohort_features.parquet')
    make_features.write_parquet(output / 'make_features.parquet')
    pl.from_dicts(resolved, infer_schema_length=None).write_parquet(output / 'review_resolution.parquet')
    coverage = {'review_rows': len(reviews), 'resolution_counts': dict(Counter(r['match_status'] for r in resolved)),
                'conflicting_reviews_excluded': sum(r['conflict_excluded'] for r in resolved),
                'cohort_rows': cohort_features.height, 'supported_makes': make_features.height,
                'make_support': make_features.select('Make', 'doug_review_count', 'doug_model_count').to_dicts()}
    result = {'state': 'complete', 'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
              'source_snapshot_sha256': imported['source_snapshot_sha256'], 'source_sha256': hashes,
              'source_paths': {key: str(path.resolve()) for key, path in paths.items()},
              'script_sha256': digest(Path(__file__)), 'minimum_models': 5, 'make_weighting': 'equal_model_means', 'cohort_keys': list(KEYS),
              'feature_columns': list(FEATURE_COLUMNS), 'interpretation': 'retrospective_association_unverified_score_availability',
              'coverage': coverage, 'files_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / 'expert_feature_manifest.json', result)
    return result


def validate_tables(cohorts, makes):
    for frame, keys in ((cohorts, KEYS), (makes, ('Make',))):
        if set(frame.columns) != {*keys, *SCORE_COLUMNS, 'doug_review_count', 'doug_model_count'} or frame.select(keys).unique().height != frame.height:
            raise ValueError('Expert feature schema or key uniqueness is invalid')
        for key in keys:
            if frame[key].null_count() or (key != 'ModelYear' and any(not norm(v) or norm(v) != v for v in frame[key])):
                raise ValueError('Expert keys must be nonempty and normalized')
        for column in SCORE_COLUMNS:
            if frame[column].null_count() or any(not math.isfinite(v) or not 1 <= v <= 10 for v in frame[column]):
                raise ValueError('Expert scores must be complete and in 1-10')
        if any(v is None or v < 1 for c in ('doug_review_count', 'doug_model_count') for v in frame[c]):
            raise ValueError('Invalid expert support count')
    if any(v < 5 for v in makes['doug_model_count']) or any(v != 1 for v in cohorts['doug_model_count']):
        raise ValueError('Invalid model support: cohort needs one model, make needs five')


def load_features(directory=DEFAULT_DIRECTORY, *, vpic_sha256=None):
    directory = Path(directory)
    try:
        path = directory / 'expert_feature_manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if (manifest['state'] != 'complete' or manifest['version'] != VERSION or manifest['feature_columns'] != list(FEATURE_COLUMNS)
                or manifest['minimum_models'] != 5 or manifest['make_weighting'] != 'equal_model_means' or manifest['cohort_keys'] != list(KEYS)):
            raise ValueError('Incompatible expert feature manifest')
        for name in ('cohort_features.parquet', 'make_features.parquet', 'review_resolution.parquet'):
            if digest(directory / name) != manifest['files_sha256'][name]:
                raise ValueError(f'Expert artifact changed: {name}')
        if vpic_sha256 is not None and manifest['source_sha256']['vpic'] != vpic_sha256:
            raise ValueError('Expert features were resolved against a different vPIC snapshot')
        cohorts, makes = (pl.read_parquet(directory / n) for n in ('cohort_features.parquet', 'make_features.parquet'))
        validate_tables(cohorts, makes)
        return cohorts, makes, {'directory': str(directory.resolve()), 'manifest_sha256': digest(path), **manifest}
    except (OSError, KeyError, ValueError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f'Cannot load DougScore features from {directory}: {exc}. Build a fresh directory with '
                         'python -m EXPERT_REVIEWS.price_features --output-dir <directory>, '
                         'then pass --expert-feature-dir <directory>.') from exc


def join_features(listings, cohorts, makes):
    validate_tables(cohorts, makes)
    if set(FEATURE_COLUMNS) & set(listings.columns):
        raise ValueError('Listing frame already contains expert predictors')
    keys = ['_doug_year', '_doug_make', '_doug_model']
    frame = listings.with_columns(pl.col('ModelYear').cast(pl.Int64, strict=False).alias(keys[0]),
        pl.col('Make').cast(pl.String).str.strip_chars().str.to_lowercase().str.replace_all(r'\s+', ' ').replace(MAKE_ALIASES).alias(keys[1]),
        pl.col('Model').cast(pl.String).str.strip_chars().str.to_lowercase().str.replace_all(r'\s+', ' ').alias(keys[2]))
    values = (*SCORE_COLUMNS, 'doug_review_count', 'doug_model_count')
    frame = frame.join(cohorts.rename({c: c + '_cohort' for c in values}), left_on=keys, right_on=list(KEYS),
                       how='left', validate='m:1', maintain_order='left')
    frame = frame.join(makes.rename({c: c + '_make' for c in values}), left_on=keys[1], right_on='Make',
                       how='left', validate='m:1', maintain_order='left')
    exact, fallback = pl.col('doug_review_count_cohort').is_not_null(), pl.col('doug_review_count_make').is_not_null()
    frame = frame.with_columns(*[pl.when(exact).then(pl.col(c + '_cohort')).otherwise(pl.col(c + '_make')).alias(c) for c in values],
        pl.when(exact).then(pl.lit('cohort')).when(fallback).then(pl.lit('make_average')).otherwise(pl.lit('missing')).alias('doug_source_level'),
        exact.cast(pl.Int8).alias('doug_cohort_match_available'), fallback.cast(pl.Int8).alias('doug_make_baseline_available'))
    return frame.drop([*keys, *[c + suffix for c in values for suffix in ('_cohort', '_make')]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path, default=ROOT / 'EXPERT_REVIEWS/outputs/dougscore_v1')
    parser.add_argument('--vpic', type=Path, default=ROOT / 'DATA/CAR_DATA_FINAL/NHTSA_vPIC.parquet')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_features(args.snapshot_dir, args.vpic, args.output_dir)['coverage'], indent=2))


if __name__ == '__main__':
    main()
