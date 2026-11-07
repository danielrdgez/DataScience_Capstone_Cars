"""Sentiment decisions, uncertainty, and reproducible cached-score migration."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from YOUTUBE.sentiment.common import ROOT, atomic_json, file_hash, load_json, new_dir, utc_now
from .data import SENTIMENT_ASPECTS, TONE_LABELS, implementation_hashes

DEFAULT_PRICE_FEATURE_DIRECTORY = ROOT / 'YOUTUBE/outputs/sentiment_argmax_20261027T155951/price'

UNCERTAINTY_SUFFIXES = ('video_balanced_soft_net', 'mean_confidence', 'mean_margin',
                        'mean_entropy', 'low_confidence_share')
V2_GROUPS = ('overall', *sorted(SENTIMENT_ASPECTS))
EXTRA_FEATURE_COLUMNS = tuple(
    [f'yt_overall_{s}' for s in ('video_balanced_net', 'opinion_count', 'opinion_video_count', 'missing')]
    + [f'yt_{a}_{s}' for a in V2_GROUPS for s in UNCERTAINTY_SUFFIXES])


def decide(scores, policy='argmax', threshold=.75, margin=.15, temperature=1.):
    if policy not in ('argmax', 'threshold') or not 0 <= threshold <= 1 or not 0 <= margin <= 1:
        raise ValueError('Invalid sentiment decision policy or thresholds')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if not scores:
        return dict(label='unresolved', top_label=None, confidence=None, margin=None,
                    entropy=None, soft_net=None, confident=False)
    if set(scores) != set(TONE_LABELS):
        raise ValueError('Sentiment requires negative, neutral and positive scores')
    p = np.array([scores[k] for k in TONE_LABELS], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1., atol=1e-5):
        raise ValueError('Sentiment scores must be finite probabilities summing to one')
    p /= p.sum()
    if temperature != 1.:
        logits = np.log(np.maximum(p, 1e-12))/temperature
        p = np.exp(logits-logits.max()); p /= p.sum()
    order = np.argsort(-p, kind='stable')
    confidence, lead = float(p[order[0]]), float(p[order[0]]-p[order[1]])
    confident = confidence >= threshold and lead >= margin
    top = TONE_LABELS[order[0]]
    return dict(label=top if policy == 'argmax' or confident else 'unresolved', top_label=top,
                confidence=confidence, margin=lead,
                entropy=float(np.clip(-np.sum(p*np.log(np.maximum(p, 1e-12)))/np.log(3), 0., 1.)),
                soft_net=float(p[2]-p[0]), confident=confident)


def annotate(row, policy='argmax', threshold=.75, margin=.15, temperature=1.):
    """Preserve content/target/exclusion rules; never copy one tone to several aspects."""
    result = dict(row)
    decision = decide(json.loads(row.get('tone_scores_json') or '{}'), policy, threshold, margin, temperature)
    eligible = (row.get('feature_eligible', False) and row.get('content_label') == 'vehicle_opinion'
                and row.get('Make') is not None and not row.get('development_excluded', False)
                and not row.get('model_truncated', False) and not row.get('windowed', False)
                and not row.get('direct_price_mention', False))
    result.update({f'sentiment_{k}': v for k, v in decision.items() if k != 'label'})
    result['overall_polarity'] = decision['label'] if eligible else 'unresolved'
    topics = row.get('topics') or []
    result['polarity'] = (result['overall_polarity'] if len(topics) == 1 and topics[0] in SENTIMENT_ASPECTS
                          else 'unresolved')
    return result


def calibration_temperature(path, train_sha256):
    if path is None:
        return 1.
    calibration = load_json(path)
    if calibration.get('state') != 'complete' or calibration['train_sha256'] != train_sha256:
        raise ValueError('Calibration model lineage differs from scored students')
    return calibration['temperature']


def relabel(score_dir, output_dir, policy='argmax', threshold=.75, margin=.15,
            calibration=None, resume=False):
    """Reuse immutable student probabilities/embeddings; no neural inference required."""
    score_dir, output_dir = Path(score_dir), Path(output_dir)
    source = load_json(score_dir/'score_manifest.json')
    if source['state'] != 'complete' or source['config']['backend'] != 'student':
        raise ValueError('Relabel requires complete student scores, never teacher/development substitutes')
    temperature = calibration_temperature(calibration, source['config']['train_sha256'])
    decide({}, policy, threshold, margin, temperature)
    config = dict(source['config'], sentiment_schema_version=2, sentiment_policy=policy,
                  tone_threshold=threshold, tone_margin=margin, sentiment_temperature=temperature,
                  calibration_sha256=file_hash(calibration) if calibration else None,
                  calibration_label_source=load_json(calibration).get('label_source') if calibration else None,
                  parent_score_sha256=file_hash(score_dir/'score_manifest.json'),
                  code_sha256=implementation_hashes())
    if resume:
        state = load_json(output_dir/'score_manifest.json')
        if state['config'] != config:
            raise ValueError('Resume inputs/policy/code changed; choose a fresh output directory')
    else:
        new_dir(output_dir); (output_dir/'shards').mkdir()
        state = dict(source, state='running', created_at_utc=utc_now(), config=config,
                     shards={}, decision_counts={}, parent_score_dir=str(score_dir.resolve()))
    counts = Counter(state.get('decision_counts', {}))
    for relative, digest in sorted(source['shards'].items()):
        if relative in state['shards']:
            if file_hash(output_dir/relative) != state['shards'][relative]:
                raise ValueError('Completed relabeled shard changed: '+relative)
            continue
        if file_hash(score_dir/relative) != digest:
            raise ValueError('Input score shard changed: '+relative)
        rows = [annotate(r, policy, threshold, margin, temperature)
                for r in pl.read_parquet(score_dir/relative).to_dicts()]
        frame = pl.from_dicts(rows, infer_schema_length=None).with_columns(
            pl.col('Make').cast(pl.String), pl.col('topics').cast(pl.List(pl.String)),
            pl.col('embedding').cast(pl.List(pl.Float32)))
        path = output_dir/relative; temporary = path.with_suffix('.parquet.tmp')
        frame.write_parquet(temporary); temporary.replace(path)
        state['shards'][relative] = file_hash(path)
        counts.update(r['overall_polarity'] for r in rows)
        state['decision_counts'] = dict(counts)
        atomic_json(output_dir/'score_manifest.json', state)
        if len(state['shards']) % 100 == 0:
            print(f'Relabeled {len(state["shards"]):,}/{len(source["shards"]):,} shards', flush=True)
    state['state'] = 'complete'
    atomic_json(output_dir/'score_manifest.json', state)
    return state


def evaluate_sentiment(teacher_dir, model_dir, output_dir, device='cpu', batch_size=32, review_csv=None, cpu_threads=4):
    """Calibrate on development validation videos; report separate test-video agreement."""
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import classification_report, log_loss
    from .data import verified_manifest
    from .neural import Classifier
    from .training import training_rows

    if batch_size < 1 or cpu_threads < 1:
        raise ValueError('Batch size and CPU threads must be positive')
    if device == 'cpu':
        import torch
        torch.set_num_threads(cpu_threads)

    teacher_dir, model_dir, output_dir = map(Path, (teacher_dir, model_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError('Use a fresh evaluation output directory: '+str(output_dir))
    teacher = verified_manifest(teacher_dir, 'score_manifest.json')
    model = load_json(model_dir/'train_manifest.json')
    if model['state'] != 'complete' or model['config']['teacher_sha256'] != file_hash(teacher_dir/'score_manifest.json'):
        raise ValueError('Teacher/student lineage differs')
    for name, digest in model['artifacts'].items():
        if file_hash(model_dir/name) != digest:
            raise ValueError('Trained artifact changed: '+name)
    columns = ['split', 'model_truncated', 'windowed', 'tone_scores_json', 'content_label',
               'Make', 'evidence_text', 'video_id', 'topics']
    frame = pl.concat([pl.read_parquet(teacher_dir/p, columns=columns) for p in sorted(teacher['shards'])])
    rows = [r for r in training_rows(frame, 'sentiment') if r['split'] in ('validation', 'test')]
    by_video = {}
    for row in rows:
        if row['video_id'] in by_video and by_video[row['video_id']] != row['split']:
            raise ValueError('Calibration and test videos overlap')
        by_video[row['video_id']] = row['split']
    label_source = 'teacher_pseudo_labels'
    review_sha = None
    if review_csv:
        import csv
        with Path(review_csv).open(encoding='utf-8-sig', newline='') as stream:
            reviewed = list(csv.DictReader(stream))
        lookup = {(r['video_id'], r['text']): r for r in rows}
        selected, seen = [], set()
        for row in reviewed:
            label = row.get('human_label', '').strip().lower()
            if not label:
                continue
            key = (row['video_id'], row['text'])
            if key not in lookup or key in seen or label not in TONE_LABELS or not row.get('reviewer', '').strip():
                raise ValueError('Invalid/duplicate review key, reviewer or human label')
            seen.add(key); selected.append(dict(lookup[key], label=label))
        rows = selected; label_source = 'human_review'; review_sha = file_hash(review_csv)
    partitions = {s: [r for r in rows if r['split'] == s] for s in ('validation', 'test')}
    if any(set(r['label'] for r in part) != set(TONE_LABELS) for part in partitions.values()):
        raise ValueError('Calibration and test partitions must each contain all three sentiment classes')
    classifier = Classifier(str(model_dir/'sentiment'), device)
    scores, flags = classifier.tone([r['text'] for r in rows], batch_size)
    if any(flags):
        raise ValueError('Evaluation tokenization truncated a development clause')
    validation = [i for i, r in enumerate(rows) if r['split'] == 'validation']
    test = [i for i, r in enumerate(rows) if r['split'] == 'test']
    probabilities = np.array([[s[k] for k in TONE_LABELS] for s in scores])
    actual = np.array([TONE_LABELS.index(r['label']) for r in rows])

    def scaled(indices, temperature):
        logits = np.log(np.maximum(probabilities[indices], 1e-12))/temperature
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p/p.sum(axis=1, keepdims=True)

    objective = lambda log_t: log_loss(actual[validation], scaled(validation, np.exp(log_t)), labels=[0, 1, 2])
    fitted = minimize_scalar(objective, bounds=(np.log(.05), np.log(20)), method='bounded')
    temperature = float(np.exp(fitted.x))
    if not fitted.success or not np.isfinite(temperature):
        raise ValueError('Temperature calibration failed')
    reports = {}
    for policy, temp, name in [('argmax', 1., 'argmax'), ('threshold', 1., 'legacy_threshold'),
                               ('argmax', temperature, 'calibrated_argmax'),
                               ('threshold', temperature, 'calibrated_threshold')]:
        decisions = [decide(scores[i], policy, temperature=temp) for i in test]
        predictions = [d['label'] for d in decisions]
        accepted = [j for j, p in enumerate(predictions) if p != 'unresolved']
        reports[name] = dict(rows=len(test), accepted=len(accepted), coverage=len(accepted)/len(test),
            all_rows_report=classification_report([rows[i]['label'] for i in test], predictions,
                labels=TONE_LABELS, output_dict=True, zero_division=0),
            accepted_class_counts=dict(Counter(predictions[j] for j in accepted)),
            accepted_only_report=(classification_report([rows[test[j]]['label'] for j in accepted],
                [predictions[j] for j in accepted], labels=TONE_LABELS, output_dict=True, zero_division=0)
                if accepted else None),
            negative_log_loss=float(log_loss(actual[test], scaled(test, temp), labels=[0, 1, 2])))
    new_dir(output_dir)
    result = dict(state='complete', created_at_utc=utc_now(), train_sha256=file_hash(model_dir/'train_manifest.json'),
                  teacher_sha256=file_hash(teacher_dir/'score_manifest.json'), temperature=temperature,
                  calibration_split='validation', evaluation_split='test', label_source=label_source,
                  review_sha256=review_sha, calibration_rows=len(validation), test_rows=len(test), reports=reports,
                  review_csv=str(Path(review_csv).resolve()) if review_csv else None,
                  validation_status='reviewed_sample_only' if review_csv else 'not_independently_validated',
                  interpretation='Development-set diagnostic; existing model selection/test use remains historical. '
                  'Tone evaluation is conditional on confident teacher opinion routing, not full-pipeline automotive accuracy.')
    atomic_json(output_dir/'sentiment_calibration.json', result)
    export = []
    for row, score in zip(rows, scores):
        export.append(dict(row, **score, human_label='', reviewer=''))
    pl.from_dicts(export).write_csv(output_dir/'review_predictions.csv')
    return result


def inspect_features(price_dir, makes=None):
    price_dir = Path(price_dir)
    manifest = load_json(price_dir/'price_feature_manifest.json')
    if file_hash(price_dir/'make_features.parquet') != manifest['features_sha256']:
        raise ValueError('Selected feature file changed')
    frame = pl.read_parquet(price_dir/'make_features.parquet')
    if makes:
        frame = frame.filter(pl.col('Make').is_in([s.strip().upper() for s in makes]))
    columns = ['Make', *[f'yt_{a}_{s}' for a in V2_GROUPS for s in
               ('video_balanced_net', 'video_balanced_soft_net', 'opinion_count', 'opinion_video_count', 'missing')
               if f'yt_{a}_{s}' in frame.columns]]
    return dict(rows=manifest['rows'], sentiment_policy=manifest.get('sentiment_policy', 'threshold'),
                sentiment_profile=manifest['sentiment_profile'], values=frame.select(columns).to_dicts())
