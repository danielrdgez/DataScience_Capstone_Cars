"""Local pretrained clause inference with conservative attribution and resumable shards."""
from __future__ import annotations
import math
import re
import time
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
from .common import (ASPECTS, METHOD, MODELS, STATUS, atomic_json, code_hashes,
                     file_hash, load_json, make_matcher, new_dir, text_hash, utc_now, versions)
from ..relevance.prepare_make_sentiment_pilot import prefilter_reason

TOPICS = {
    'vehicle_opinion': 'The author expresses an opinion about a vehicle.',
    'creator_praise': 'The author praises the presenter or video rather than a vehicle.',
    'question': 'The author asks a question without expressing a vehicle opinion.',
    'fact': 'The author states a vehicle fact without expressing an opinion.',
    'comparison': 'The author compares different vehicles.',
}


class LocalModels:
    """Independent entailment/contradiction scores; tone is a separate baseline."""
    def __init__(self, revisions, device='cpu', cache_dir=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device = torch, device
        self.models, self.tokenizers = {}, {}
        torch.manual_seed(43)
        for key, name in MODELS.items():
            options = dict(revision=revisions[key], trust_remote_code=False, token=False, cache_dir=cache_dir)
            self.tokenizers[key] = AutoTokenizer.from_pretrained(name, **options)
            self.models[key] = AutoModelForSequenceClassification.from_pretrained(name, **options).to(device).eval()
        labels = {str(v).lower(): int(k) for k, v in self.models['nli'].config.id2label.items()}
        if not {'entailment', 'contradiction'} <= labels.keys():
            raise ValueError('NLI model must expose named entailment and contradiction labels')
        self.entailment, self.contradiction = labels['entailment'], labels['contradiction']

    def nli(self, text, hypotheses, max_tokens=512):
        scores, truncated = [], False
        tokenizer = self.tokenizers['nli']
        for start in range(0, len(hypotheses), 16):
            batch = hypotheses[start:start+16]
            for h in batch:
                truncated |= len(tokenizer(text, h, truncation=False)['input_ids']) > max_tokens
            encoded = tokenizer([text]*len(batch), batch, padding=True, truncation='only_first',
                                max_length=max_tokens, return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                logits = self.models['nli'](**encoded).logits[:, [self.contradiction, self.entailment]]
                scores.extend(logits.softmax(dim=-1)[:, 1].cpu().tolist())
        return scores, truncated

    def tone(self, text, max_tokens=512):
        tokenizer = self.tokenizers['tone']
        # Cardiff's documented normalization, keeping the original evidence separately.
        cleaned = re.sub(r'https?://\S+', 'http', text)
        cleaned = re.sub(r'(?<!\w)@\w+', '@user', cleaned)
        truncated = len(tokenizer(cleaned, truncation=False)['input_ids']) > max_tokens
        encoded = tokenizer(cleaned, truncation=True, max_length=max_tokens, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            scores = self.models['tone'](**encoded).logits.softmax(dim=-1)[0].cpu().tolist()
        labels = self.models['tone'].config.id2label
        result = {str(labels[i]).lower(): float(score) for i, score in enumerate(scores)}
        if set(result) != {'negative', 'neutral', 'positive'}:
            raise ValueError('Tone model must expose negative/neutral/positive labels')
        return result, truncated


def detect_language(text):
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 43
    if len(re.findall(r'[A-Za-z]', text)) < 20:
        return 'unknown'
    try:
        candidates = detect_langs(text)
        if not candidates or candidates[0].prob < .8:
            return 'unknown'
        return 'en' if candidates[0].lang == 'en' else 'unsupported'
    except Exception:
        return 'unknown'


def clauses(text, max_chars=700):
    """Keep original character offsets and split oversized sentences without dropping tails."""
    for match in re.finditer(r'[^.!?;\n]+(?:[.!?;]+|$)', text):
        start, end = match.span()
        while start < end:
            stop = min(end, start+max_chars)
            if stop < end:
                space = text.rfind(' ', start+max_chars//2, stop)
                if space > start:
                    stop = space+1
            if text[start:stop].strip():
                yield start, stop, text[start:stop], end-match.start() > max_chars
            start = stop


def checked_nli(model, text, hypotheses, max_tokens):
    values, truncated = model.nli(text, hypotheses, max_tokens)
    if len(values) != len(hypotheses) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid NLI scores')
    return dict(zip(hypotheses, values)), bool(truncated)


def polarity(scores, tone, threshold, margin):
    ranked = sorted(scores, key=scores.get, reverse=True)
    top = ranked[0]
    if scores[top] < threshold or scores[top]-scores[ranked[1]] < margin:
        return 'unresolved'
    if top == 'no_opinion':
        return 'no_opinion'
    # Same-clause simultaneous support is ambiguity, not sufficient mixed evidence.
    if scores['positive'] >= threshold and scores['negative'] >= threshold:
        return 'unresolved'
    opposite = 'negative' if top == 'positive' else 'positive' if top == 'negative' else None
    if opposite and tone.get(opposite, 0) >= threshold:
        return 'unresolved'
    return top


def infer_comment(row, matcher, model, config, language_fn=detect_language):
    text = row.get('text') or ''
    language = language_fn(text) if not prefilter_reason(text) else 'unknown'
    explicit = matcher(text)
    title = matcher(row.get('video_title') or '')
    targets = [(make, 'explicit_comment') for make in explicit]
    if not targets:
        targets = [(title[0], 'title_context')] if len(title) == 1 else [(None, 'unresolved')]
    evidence = {(make, scope, aspect): [] for make, scope in targets for aspect in ASPECTS}
    if language == 'en' and not prefilter_reason(text) and not row.get('development_protected', False):
        for start, end, clause, windowed in clauses(text, config['clause_chars']):
            clause_targets = matcher(clause)
            hypotheses = list(TOPICS.values()) + [f'The text discusses {description} of a vehicle.' for description in ASPECTS.values()]
            raw, nli_truncated = checked_nli(model, clause, hypotheses, config['max_tokens'])
            topic = {k: raw[v] for k, v in TOPICS.items()}
            tone, tone_truncated = model.tone(clause, config['max_tokens'])
            if set(tone) != {'positive', 'negative', 'neutral'} or any(not math.isfinite(v) or not 0 <= v <= 1 for v in tone.values()):
                raise ValueError('Invalid tone scores')
            for make, scope in targets:
                supported = ((scope == 'explicit_comment' and clause_targets == [make]) or
                             (scope == 'title_context' and not explicit and not clause_targets))
                for aspect, description in ASPECTS.items():
                    aspect_score = raw[f'The text discusses {description} of a vehicle.']
                    entry = {'start': start, 'end': end, 'text': clause, 'topic_scores': topic,
                             'aspect_score': aspect_score, 'tone_scores': tone,
                             'input_truncated': nli_truncated or tone_truncated, 'windowed': windowed,
                             'target_supported': supported, 'polarity_scores': None}
                    if aspect_score < config['threshold']:
                        decision = 'aspect_not_present'
                    elif not supported or windowed or nli_truncated or tone_truncated:
                        decision = 'unresolved'
                    elif topic['vehicle_opinion'] < config['threshold']:
                        decision = 'no_opinion' if max(topic['question'], topic['fact'], topic['creator_praise']) >= config['threshold'] else 'unresolved'
                    elif topic['creator_praise'] >= topic['vehicle_opinion']:
                        decision = 'unresolved'
                    else:
                        subject = f'the {make} vehicle' if scope == 'explicit_comment' else 'the vehicle discussed in the video'
                        polar_h = {
                            'positive': f'The author expresses a positive opinion about {description} of {subject}.',
                            'negative': f'The author expresses a negative opinion about {description} of {subject}.',
                            'neutral': f'The author explicitly expresses a neutral opinion about {description} of {subject}.',
                            'no_opinion': f'The author expresses no opinion about {description} of {subject}.',
                        }
                        polar_raw, truncated = checked_nli(model, clause, list(polar_h.values()), config['max_tokens'])
                        scores = {k: polar_raw[v] for k, v in polar_h.items()}
                        entry['polarity_scores'] = scores
                        entry['input_truncated'] |= truncated
                        decision = 'unresolved' if truncated else polarity(scores, tone, config['threshold'], config['margin'])
                    entry['decision'] = decision
                    evidence[(make, scope, aspect)].append(entry)
    results = []
    import json
    for (make, scope, aspect), entries in evidence.items():
        decisions = {e['decision'] for e in entries if e['decision'] != 'aspect_not_present'}
        if row.get('development_protected', False):
            decision = 'unresolved'
        elif prefilter_reason(text):
            decision = 'no_opinion'
        elif language != 'en' or scope == 'unresolved':
            decision = 'unresolved'
        elif 'unresolved' in decisions:
            decision = 'unresolved'
        elif {'positive', 'negative'} <= decisions:
            decision = 'mixed'
        elif decisions & {'positive', 'negative', 'neutral'}:
            decision = next(k for k in ['positive', 'negative', 'neutral'] if k in decisions)
        else:
            decision = 'no_opinion' if 'no_opinion' in decisions else 'aspect_not_present'
        results.append({**row, 'Make': make, 'attribution_scope': scope, 'aspect': aspect,
                        'polarity': decision, 'language_status': language,
                        'input_truncated_flag': any(e['input_truncated'] for e in entries),
                        'windowed_flag': any(e['windowed'] for e in entries),
                        'normalized_text_sha256': text_hash(text),
                        'prefilter_reason': prefilter_reason(text), 'evidence_json': json.dumps(entries),
                        'method': METHOD, 'validation_status': STATUS})
    return results


def infer(sample_dir, output_dir, batch_size=16, device='cpu', resume=False, cache_dir=None,
          model_factory=LocalModels, language_fn=detect_language, revisions=None):
    sample_dir, output_dir = Path(sample_dir), Path(output_dir)
    sample_manifest = load_json(sample_dir/'sample_manifest.json')
    if file_hash(sample_dir/'sample.parquet') != sample_manifest['sample_sha256']:
        raise ValueError('Frozen sample changed')
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    config = {'sample_sha256': sample_manifest['sample_sha256'], 'sample_manifest_sha256': file_hash(sample_dir/'sample_manifest.json'),
              'dependencies': versions(), 'code_sha256': code_hashes(), 'batch_size': batch_size,
              'device': device, 'models': MODELS, 'policy': sample_manifest['config'],
              'topic_hypotheses': TOPICS, 'aspects': ASPECTS}
    if resume:
        saved = load_json(output_dir/'inference_manifest.json')
        if saved['config'] != config:
            raise ValueError('Input, code, dependencies or settings changed; use a fresh output directory')
        revisions = saved['model_revisions']
        for name, digest in saved['shards'].items():
            if file_hash(output_dir/name) != digest:
                raise ValueError('Completed prediction shard changed')
    else:
        if revisions is None:
            from huggingface_hub import HfApi
            revisions = {key: HfApi(token=False).model_info(name).sha for key, name in MODELS.items()}
        new_dir(output_dir)
        (output_dir/'shards').mkdir()
        saved = {'created_at_utc': utc_now(), 'state': 'running', 'config': config,
                 'model_revisions': revisions, 'shards': {}, 'sampling_population': sample_manifest['sampling_population'],
                 'batch_seconds': [], 'method': METHOD, 'validation_status': STATUS}
        atomic_json(output_dir/'inference_manifest.json', saved)
    sample_file = pq.ParquetFile(sample_dir/'sample.parquet')
    row_count = sample_file.metadata.num_rows
    matcher = make_matcher(sample_manifest['canonical_makes'], sample_manifest['config']['aliases'])
    model = None
    for batch_index, batch in enumerate(sample_file.iter_batches(batch_size=batch_size)):
        rows = batch.to_pylist()
        index = batch_index * batch_size
        name = f'shards/batch_{batch_index:06d}.parquet'
        if name in saved['shards']:
            continue
        if model is None:
            model = model_factory(revisions, device, cache_dir)
        start = time.perf_counter()
        predictions = [p for r in rows
                       for p in infer_comment(r, matcher, model, sample_manifest['config'], language_fn)]
        frame = pl.from_dicts(predictions, infer_schema_length=None)
        temp = output_dir/(name+'.tmp')
        frame.write_parquet(temp); temp.replace(output_dir/name)
        saved['shards'][name] = file_hash(output_dir/name)
        saved['batch_seconds'].append(time.perf_counter()-start)
        atomic_json(output_dir/'inference_manifest.json', saved)
        print(f'Comments {min(index+batch_size,row_count)}/{row_count} checkpointed', flush=True)
    if file_hash(sample_dir/'sample.parquet') != sample_manifest['sample_sha256']:
        raise ValueError('Frozen sample changed during inference')
    scans = [pl.scan_parquet(output_dir/name) for name in saved['shards']]
    output = output_dir/'youtube_experimental_predictions.parquet'
    pl.concat(scans, how='diagonal_relaxed').sink_parquet(output.with_suffix('.tmp'))
    output.with_suffix('.tmp').replace(output)
    saved.update(state='complete', predictions_sha256=file_hash(output), comments=row_count,
                 inference_seconds=sum(saved['batch_seconds']), completed_at_utc=utc_now())
    atomic_json(output_dir/'inference_manifest.json', saved)
    return {'predictions': str(output), 'comments': row_count,
            'comments_per_second': row_count/sum(saved['batch_seconds']) if sum(saved['batch_seconds']) else None}
