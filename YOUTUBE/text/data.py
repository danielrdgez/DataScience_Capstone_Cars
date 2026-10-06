"""Freeze raw-source samples, canonical identities and development exclusions."""
from __future__ import annotations

import csv
import hashlib
import heapq
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from YOUTUBE.sentiment.common import (
    ROOT, ASPECTS, atomic_json, cutoff_date, file_hash, load_json, make_matcher,
    new_dir, normalized, parse_date, text_hash, utc_now, versions,
)

MODEL_IDS = {
    'embedding': 'sentence-transformers/all-MiniLM-L6-v2',
    'content_teacher': 'MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33',
    'sentiment_teacher': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'student': 'nreimers/MiniLM-L6-H384-uncased',
}
CONTENT = {
    'vehicle_opinion': 'The author expresses an opinion or personal experience about a car or its attributes.',
    'vehicle_fact_question': 'The author states a car fact or asks a car question without expressing a personal opinion.',
    'creator_other': 'The author discusses the presenter, filming, music or video production rather than a car.',
    'irrelevant': 'The comment is spam, unrelated conversation or contains no meaningful information about a car.',
}
TONE_LABELS = ['negative', 'neutral', 'positive']
TOPICS = {k: v for k, v in ASPECTS.items() if k != 'price_value'}
SENTIMENT_ASPECTS = {'reliability', 'maintenance_cost', 'performance', 'comfort'}
ALIASES = {'VW': 'VOLKSWAGEN', 'CHEVY': 'CHEVROLET', 'MERCEDES': 'MERCEDES-BENZ'}
AMBIGUOUS_MAKES = {'MINI', 'SMART', 'RAM', 'GENESIS', 'BOSS', 'EAGLE', 'SATURN', 'INTERNATIONAL'}
VEHICLE_CUES = re.compile(
    r'\b(car|cars|vehicle|truck|suv|engine|transmission|cvts?|seats?|ride|driv\w*|'
    r'brakes?|reliable|reliability|comfort\w*|handling|acceleration|mpg|mileage|'
    r'fuel|maintenance|repair\w*|battery|range|steering|1500|2500|3500|cooper)\b', re.I)
PRICE_CUES = re.compile(r'[$€£]\s*\d|\b\d[\d,.]*\s*(?:dollars?|usd)\b|\b(?:paid|price|costs?)\s+(?:\w+\s+)?\d', re.I)
SHORT_ENGLISH = re.compile(r'\b(this|the|is|are|not|love|hate|great|good|bad|reliable|unreliable|comfortable|slow|fast|engine|car|seats|looks|ugly|beautiful)\b', re.I)


def implementation_hashes():
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT/'YOUTUBE/text_pipeline.py', ROOT/'YOUTUBE/sentiment/common.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def runtime():
    result = versions()
    import importlib.metadata
    for name in ('sentence-transformers', 'scikit-learn', 'sentencepiece'):
        result[name] = importlib.metadata.version(name)
    return result


def rank(seed, value):
    return int(hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()[:16], 16)


def duplicate_hash(text):
    # Preserve negation, punctuation and emoji; changing them can reverse meaning.
    value = ' '.join(unicodedata.normalize('NFKC', text or '').casefold().split())
    return hashlib.sha256(value.encode()).hexdigest()


def video_split(video_id, seed):
    value = rank(seed, 'split|'+video_id)/2**64
    return 'train' if value < .8 else 'validation' if value < .9 else 'test'


def read_protection(directory):
    """Read identifiers/text only. Never read editable annotation files or scores."""
    ids, videos, texts, hashes = set(), set(), set(), {}
    directory = Path(directory)
    if not directory.exists():
        return ids, videos, texts, hashes, None
    manifest_path = directory/'candidate_manifest.json'
    metadata = load_json(manifest_path)
    hashes[str(manifest_path.resolve())] = file_hash(manifest_path)
    for name in ('protected_train_sample.csv', 'protected_blind_sample.csv'):
        path = directory/name
        digest = file_hash(path)
        if digest != metadata[name.replace('.csv', '_sha256')]:
            raise ValueError('Protected reference checksum changed: '+name)
        hashes[str(path.resolve())] = digest
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                ids.add(row['comment_id']); videos.add(row['video_id']); texts.add(text_hash(row['text']))
    return ids, videos, texts, hashes, metadata.get('reservation')


def prepare(source, vpic, output, as_of, sample_size=20000, seed=43, protection_dir=None):
    source, vpic, output = map(Path, (source, vpic, output))
    if sample_size < 0:
        raise ValueError('Sample size is zero (full source) or positive')
    cutoff = cutoff_date(as_of)
    protection_dir = protection_dir or ROOT/'YOUTUBE/outputs/relevance_candidate_v1'
    ids, videos, texts, protection_hashes, reservation = read_protection(protection_dir)
    makes = sorted(pl.scan_parquet(vpic).select('Make').unique().collect()['Make'].drop_nulls().to_list())
    makes = sorted({str(m).strip().upper() for m in makes if str(m).strip()})
    if not makes:
        raise ValueError('No canonical vPIC makes')
    source_sha, vpic_sha = file_hash(source), file_hash(vpic)
    audit = Counter()
    counts = Counter()
    required = ['comment_id', 'video_id', 'video_title', 'text', 'published_at', 'updated_at', 'extracted_at']

    def rows(record_audit=False):
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16384, columns=required):
            for row in batch.to_pylist():
                if record_audit:
                    audit['source_rows'] += 1
                cid, vid = row['comment_id'], row['video_id']
                if not cid or not vid:
                    raise ValueError('Missing comment/video identity')
                published, updated = parse_date(row['published_at']), parse_date(row['updated_at'])
                reason = None
                if published is None or updated is None:
                    reason = 'unparseable_dates'
                elif published > cutoff or updated > cutoff:
                    reason = 'post_cutoff_publication_or_update'
                else:
                    digest = text_hash(row['text'])
                    reserved = reservation and rank(reservation['seed'], 'video|'+vid)/2**64 < reservation['fraction']
                    if cid in ids or vid in videos or digest in texts or reserved:
                        reason = 'protected_id_video_text_or_reservation'
                if reason:
                    if record_audit:
                        audit[reason] += 1
                    continue
                row.update(normalized_text_sha256=duplicate_hash(row['text']), published_date=published.date().isoformat(),
                           updated_date=updated.date().isoformat(), split=video_split(vid, seed),
                           direct_price_mention=bool(PRICE_CUES.search(row['text'] or '')))
                yield row

    for row in rows(True):
        counts[row['video_id']] += 1
    if not counts:
        raise ValueError('No eligible source rows')
    development_counts = {v:n for v,n in counts.items() if rank(seed, 'development|'+v)/2**64 < .25} if sample_size else counts
    if not development_counts:
        raise ValueError('No development videos; change seed or use more source videos')
    new_dir(output); (output/'shards').mkdir()
    if sample_size:
        capacity = max(1, math.ceil(min(sample_size,sum(development_counts.values()))/len(development_counts)))
        while sum(min(n, capacity) for n in development_counts.values()) < min(sample_size,sum(development_counts.values())):
            capacity += 1
        heaps = defaultdict(list)
        for row in rows():
            if row['video_id'] not in development_counts:
                continue
            key = (-rank(seed, row['comment_id']), row['comment_id'])
            heap = heaps[row['video_id']]
            if len(heap) < capacity:
                heapq.heappush(heap, (key, row))
            elif key > heap[0][0]:
                heapq.heapreplace(heap, (key, row))
        ordered = {v: [r for _, r in sorted(h, reverse=True)] for v, h in heaps.items()}
        selected = []
        for i in range(capacity):
            for vid in sorted(ordered, key=lambda v: rank(seed, v)):
                if i < len(ordered[vid]):
                    selected.append(ordered[vid][i])
                if len(selected) >= sample_size:
                    break
            if len(selected) >= sample_size:
                break
        selected.sort(key=lambda r: (r['video_id'], rank(seed, r['comment_id'])))
        if len({r['comment_id'] for r in selected}) != len(selected):
            raise ValueError('Duplicate source comment IDs in development sample')
        owners = {}
        for row in selected:
            key = row['normalized_text_sha256']
            if key in owners and owners[key] != row['split']:
                row['split'] = 'duplicate_excluded'
            else:
                owners[key] = row['split']
        batches = (selected[i:i+4096] for i in range(0, len(selected), 4096))
    else:
        def full_batches():
            batch = []
            for row in rows():
                batch.append(row)
                if len(batch) == 4096:
                    yield batch; batch = []
            if batch:
                yield batch
        batches = full_batches()
    shards, splits, total = {}, Counter(), 0
    for index, batch in enumerate(batches):
        path = output/'shards'/f'raw_{index:06d}.parquet'
        pl.from_dicts(batch, infer_schema_length=None).write_parquet(path)
        shards[str(path.relative_to(output))] = file_hash(path)
        splits.update(r['split'] for r in batch); total += len(batch)
    if file_hash(source) != source_sha or file_hash(vpic) != vpic_sha:
        raise ValueError('Source changed during preparation')
    manifest = dict(state='complete', created_at_utc=utc_now(), source=str(source.resolve()),
                    source_sha256=source_sha, vpic_sha256=vpic_sha, canonical_makes=makes,
                    as_of=as_of, seed=seed, rows=total, eligible_source_rows=sum(counts.values()),
                    eligible_videos=len(counts), sample_size=sample_size, splits=dict(splits),
                    development_video_fraction=.25 if sample_size else None,
                    population='video_balanced_development' if sample_size else 'eligible_source_snapshot',
                    selection='seeded rotating video quotas; no population weights' if sample_size else 'all eligible rows',
                    protection_hashes=protection_hashes, audit=dict(audit), shards=shards,
                    code_sha256=implementation_hashes(), validation_status='not_independently_validated')
    atomic_json(output/'prepare_manifest.json', manifest)
    return manifest


def verified_manifest(directory, name):
    directory = Path(directory)
    manifest = load_json(directory/name)
    if manifest['state'] != 'complete':
        raise ValueError('Input stage is incomplete: '+str(directory))
    for path, digest in manifest['shards'].items():
        if file_hash(directory/path) != digest:
            raise ValueError('Input shard changed: '+path)
    return manifest


def canonical_matcher(makes):
    base = make_matcher(makes, ALIASES)
    def match(text):
        candidates = base(text)
        return [m for m in candidates if m not in AMBIGUOUS_MAKES or VEHICLE_CUES.search(text)]
    return match


def language_status(text):
    from langdetect import DetectorFactory, detect_langs
    letters = re.findall(r'[^\W\d_]', text, re.U)
    ascii_letters = sum(c.isascii() for c in letters)
    if not letters or ascii_letters/len(letters) < .9:
        return 'unsupported_or_unknown'
    if len(text.split()) < 6:
        return 'short_en_rule' if SHORT_ENGLISH.search(text) and ascii_letters >= 5 else 'unknown'
    DetectorFactory.seed = 43
    try:
        detected = detect_langs(text)[0]
        return 'en' if detected.lang == 'en' and detected.prob >= .8 else 'unsupported_or_unknown'
    except Exception:
        return 'unknown'


def evidence_rows(row, matcher):
    """All original offsets survive; contrasts split, multi-target clauses abstain."""
    text = row['text'] or ''
    explicit = matcher(text)
    title = matcher(row['video_title'] or '')
    boundaries = [0] + [m.start() for m in re.finditer(r'(?<=[.!?;\n])\s+|\b(?:but|however|whereas)\b', text, re.I)] + [len(text)]
    language = language_status(text) if text.strip() else 'unknown'
    for start, end in zip(boundaries, boundaries[1:]):
        cursor = start
        while cursor < end:
            stop = min(cursor+600, end)
            if stop < end:
                space = text.rfind(' ', cursor+300, stop)
                if space > cursor:
                    stop = space+1
            clause = text[cursor:stop]
            if clause.strip():
                targets = matcher(clause)
                make, scope = None, 'unresolved'
                if len(targets) == 1:
                    make, scope = targets[0], 'explicit_comment'
                elif not targets and not explicit and len(title) == 1 and VEHICLE_CUES.search(clause):
                    make, scope = title[0], 'title_context'
                yield dict(row, evidence_start=cursor, evidence_end=stop, evidence_text=clause,
                           Make=make, attribution_scope=scope, language_status=language,
                           windowed=(end-start > 600), multiple_targets=len(targets)>1)
            cursor = stop
    if not text.strip():
        yield dict(row, evidence_start=0, evidence_end=len(text), evidence_text=text,
                   Make=None, attribution_scope='unresolved', language_status='unknown',
                   windowed=False, multiple_targets=False)


def choose_label(scores, threshold=.75, margin=.15):
    ordered = sorted(scores, key=scores.get, reverse=True)
    if scores[ordered[0]] >= threshold and scores[ordered[0]]-scores[ordered[1]] >= margin:
        return ordered[0]
    return 'unresolved'
