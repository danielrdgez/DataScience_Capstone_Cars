"""Freeze a balanced engineering sample from the existing scored pilot."""
from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path
import polars as pl
from .common import DEFAULTS, atomic_json, file_hash, load_json, new_dir, text_hash, utc_now


def prepare(input_path, candidate_dir, output_dir, size=1000, seed=43):
    input_path, candidate_dir, output_dir = map(Path, (input_path, candidate_dir, output_dir))
    manifest = load_json(candidate_dir/'candidate_manifest.json')
    if manifest.get('state') != 'complete':
        raise ValueError('Candidate must be complete')
    protected, protected_videos, protected_text = set(), set(), set()
    for name in ['protected_train_sample.csv', 'protected_blind_sample.csv']:
        if file_hash(candidate_dir/name) != manifest[name.replace('.csv', '_sha256')]:
            raise ValueError('Frozen protected sample changed')
        with (candidate_dir/name).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                protected.add(row['comment_id']); protected_videos.add(row['video_id'])
                protected_text.add(text_hash(row['text']))
    with input_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if size <= 0 or size > len(rows):
        raise ValueError('Sample size must be between 1 and the pilot size')
    candidate_hash = file_hash(candidate_dir/'candidate_manifest.json')
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row['comment_id'] in seen:
            raise ValueError('Duplicate comment ID')
        seen.add(row['comment_id'])
        if row['comment_id'] in protected or row['video_id'] in protected_videos or text_hash(row['text']) in protected_text:
            raise ValueError('Input overlaps protected training/evaluation material')
        if row.get('candidate_manifest_sha256') != candidate_hash:
            raise ValueError('Pilot candidate lineage differs')
        score = float(row['prob_vehicle_relevant'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid relevance score')
        row['prob_vehicle_relevant'] = score
        band = 'low' if score < .2 else 'middle' if score < .5 else 'high'
        row['score_band'] = band
        groups[(band, row['video_id'])].append(row)
    def rank(value):
        return hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()
    ordered_groups = {band: sorted([k for k in groups if k[0] == band], key=rank)
                      for band in ['low', 'middle', 'high']}
    for group in groups.values():
        group.sort(key=lambda r: rank(r['comment_id']))
    selected, quotas = [], defaultdict(int)
    # Rotate bands and videos so a high-volume video cannot dominate the prototype.
    while len(selected) < size:
        progress = False
        for band in ordered_groups:
            for key in ordered_groups[band]:
                if quotas[key] < len(groups[key]):
                    selected.append(groups[key][quotas[key]])
                    quotas[key] += 1; progress = True
                    break
            if len(selected) == size:
                break
            ordered_groups[band] = sorted(ordered_groups[band], key=lambda k: (quotas[k], rank(k)))
        if not progress:
            break
    for row in selected:
        key = (row['score_band'], row['video_id'])
        row['sample_cell_rows'] = len(groups[key])
        row['sample_cell_selected'] = quotas[key]
        row['conditional_sampling_fraction'] = quotas[key]/len(groups[key])
        row['sampling_population'] = 'enriched_pilot_engineering_only'
    new_dir(output_dir)
    pl.from_dicts(selected, infer_schema_length=None).write_parquet(output_dir/'sample.parquet')
    atomic_json(output_dir/'sample_manifest.json', {
        'created_at_utc': utc_now(), 'input_path': str(input_path.resolve()),
        'input_sha256': file_hash(input_path), 'candidate_manifest_sha256': file_hash(candidate_dir/'candidate_manifest.json'),
        'sample_sha256': file_hash(output_dir/'sample.parquet'), 'rows': len(selected),
        'videos': len({r['video_id'] for r in selected}), 'seed': seed,
        'canonical_makes': manifest['canonical_makes'], 'config': DEFAULTS,
        'sampling_population': 'enriched_pilot_engineering_only',
        'sampling': 'Seeded equal-band, rotating-video quotas; fractions conditional on allocated cells; not source inclusion probabilities or weights.',
        'zero_allocation_cells': sum(k not in quotas for k in groups),
        'protected_sample_metadata_only': True,
    })
    return {'sample': str(output_dir/'sample.parquet'), 'rows': len(selected)}
