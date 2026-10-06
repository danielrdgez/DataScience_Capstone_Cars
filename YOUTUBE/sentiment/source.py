"""Freeze and relevance-score a full source snapshot in resumable batches."""
import csv
import time
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
from .common import DEFAULTS, atomic_json, code_hashes, file_hash, new_dir, text_hash, utc_now, versions, load_json
from ..relevance.relevance_candidate import LocalSetFit, candidate_metadata, probabilities, reserved_video


def source(source_path, candidate_dir, output_dir, batch_size=1024, device='cpu', resume=False):
    source_path, candidate_dir, output_dir = map(Path, (source_path, candidate_dir, output_dir))
    manifest = candidate_metadata(candidate_dir)
    source_hash = file_hash(source_path)
    if source_hash != manifest['source']['sha256']:
        raise ValueError('Source differs from the relevance candidate frozen snapshot')
    config = {'source_sha256': source_hash, 'candidate_manifest_sha256': file_hash(candidate_dir/'candidate_manifest.json'),
              'batch_size': batch_size, 'device': device, 'dependencies': versions(), 'code_sha256': code_hashes()}
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    if resume:
        state = load_json(output_dir/'source_manifest.json')
        if state['config'] != config:
            raise ValueError('Source scoring configuration changed; choose a fresh directory')
        for name, digest in state['shards'].items():
            if file_hash(output_dir/name) != digest:
                raise ValueError('Completed source shard changed')
    else:
        new_dir(output_dir); (output_dir/'shards').mkdir()
        state = {'config': config, 'created_at_utc': utc_now(), 'state': 'running', 'shards': {}, 'rows': 0}
        atomic_json(output_dir/'source_manifest.json', state)
    ids, videos, texts = set(), set(), set()
    for name in ['protected_train_sample.csv', 'protected_blind_sample.csv']:
        with (candidate_dir/name).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                ids.add(row['comment_id']); videos.add(row['video_id']); texts.add(text_hash(row['text']))
    # Loading frozen weights does not fit a model; record this new adapter/runtime separately.
    model = None
    source_file = pq.ParquetFile(source_path)
    columns = ['comment_id', 'video_id', 'video_title', 'published_at', 'text']
    seen = set()
    for index, batch in enumerate(source_file.iter_batches(batch_size=batch_size, columns=columns)):
        name = f'shards/source_{index:06d}.parquet'
        rows = batch.to_pylist()
        for row in rows:
            if not row['comment_id'] or not row['video_id'] or row['comment_id'] in seen:
                raise ValueError('Missing or duplicated comment/video identity')
            seen.add(row['comment_id'])
        if name in state['shards']:
            continue
        if model is None:
            model = LocalSetFit.load(candidate_dir/'model', device)
            if model.max_seq_length != manifest['config']['max_seq_length']:
                raise ValueError('Saved relevance tokenizer input limit changed')
        start = time.perf_counter()
        scores = probabilities(model.score([r['text'] or '' for r in rows]), len(rows))
        for row, score in zip(rows, scores):
            row.update(prob_vehicle_relevant=float(score), development_protected=(row['comment_id'] in ids or
                row['video_id'] in videos or text_hash(row['text']) in texts or reserved_video(row['video_id'], manifest['reservation'])),
                sampling_population='full_source_snapshot_protected_excluded_from_features',
                candidate_manifest_sha256=config['candidate_manifest_sha256'])
        temp = output_dir/(name+'.tmp')
        pl.from_dicts(rows, infer_schema_length=None).write_parquet(temp); temp.replace(output_dir/name)
        state['shards'][name] = file_hash(output_dir/name)
        state['rows'] += len(rows)
        state['last_batch_seconds'] = time.perf_counter()-start
        atomic_json(output_dir/'source_manifest.json', state)
        print(f'Source relevance rows checkpointed: {state["rows"]:,}', flush=True)
    if file_hash(source_path) != source_hash:
        raise ValueError('Source changed during scoring')
    path = output_dir/'sample.parquet'
    pl.concat([pl.scan_parquet(output_dir/n) for n in state['shards']], how='diagonal_relaxed').sink_parquet(path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)
    sample = {'created_at_utc': utc_now(), 'sample_sha256': file_hash(path), 'rows': state['rows'],
              'canonical_makes': manifest['canonical_makes'], 'config': DEFAULTS,
              'sampling_population': 'full_source_snapshot_protected_excluded_from_features',
              'source_sha256': source_hash, 'candidate_manifest_sha256': config['candidate_manifest_sha256'],
              'source_scoring_config': config, 'protected_rows_retained': True}
    atomic_json(output_dir/'sample_manifest.json', sample)
    state['state'] = 'complete'; atomic_json(output_dir/'source_manifest.json', state)
    return {'sample': str(path), 'rows': state['rows']}
