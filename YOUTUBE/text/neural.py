"""Batched GPU teachers/students, cached embeddings and resumable scoring."""
from __future__ import annotations

import gc
import json
import re
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import polars as pl

from YOUTUBE.sentiment.common import atomic_json, file_hash, load_json, new_dir, utc_now
from .data import (
    ALIASES, CONTENT, MODEL_IDS, SENTIMENT_ASPECTS, TOPICS, canonical_matcher, choose_label,
    evidence_rows, implementation_hashes, runtime, verified_manifest,
)


def device_info(device='cuda'):
    import torch
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use .venv-youtube and install torch==2.7.1 from the cu126 index; run doctor.')
        # Exercise a kernel too: detection alone does not prove wheel compatibility.
        torch.ones(2, 2, device=device).matmul(torch.ones(2, 2, device=device))
        return dict(device=device, name=torch.cuda.get_device_name(0),
                    vram_gb=round(torch.cuda.get_device_properties(0).total_memory/2**30, 2),
                    capability=list(torch.cuda.get_device_capability(0)), torch=torch.__version__,
                    cuda_runtime=torch.version.cuda)
    if device != 'cpu':
        raise ValueError('Device must be cuda or cpu')
    return dict(device=device, torch=torch.__version__, cuda_runtime=torch.version.cuda)


def resolve_revisions(cache_dir):
    from huggingface_hub import HfApi
    api = HfApi(token=False)
    return {k: dict(model_id=v, revision=api.model_info(v).sha) for k, v in MODEL_IDS.items()}


def amp(device):
    import torch
    return torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else nullcontext()


class Classifier:
    def __init__(self, name, device, revision=None, cache_dir=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        options = dict(trust_remote_code=False, token=False, cache_dir=str(cache_dir), revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(name, **options)
        # These two public repositories supply original .bin weights. Avoid an
        # implicit Hub safetensors conversion request; torch>=2.6 uses safe loading.
        binary_only = name in (MODEL_IDS['sentiment_teacher'], MODEL_IDS['student'])
        self.model = AutoModelForSequenceClassification.from_pretrained(name, use_safetensors=False if binary_only else None, **options).to(device).eval()
        self.device, self.torch = device, torch
        self.labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}

    def probabilities(self, texts, batch_size, max_tokens=256, pairs=None):
        values, truncated = [], []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            pair = pairs[start:start+batch_size] if pairs is not None else None
            lengths = self.tokenizer(batch, pair, truncation=False, verbose=False)['input_ids']
            truncated.extend(len(tokens)>max_tokens for tokens in lengths)
            encoded = self.tokenizer(batch, pair, truncation='only_first' if pair else True,
                                     max_length=max_tokens, padding=True, return_tensors='pt').to(self.device)
            with self.torch.inference_mode(), amp(self.device):
                logits = self.model(**encoded).logits
            values.extend(logits.float().softmax(-1).cpu().numpy())
        return np.asarray(values), truncated

    def content(self, texts, batch_size, teacher):
        if not teacher:
            values, truncated = self.probabilities(texts, batch_size)
            scores = [{self.labels[i]: float(x) for i, x in enumerate(row)} for row in values]
            if scores and (not set(scores[0]) <= set(CONTENT) or not {'vehicle_opinion','vehicle_fact_question','creator_other'} <= set(scores[0])):
                raise ValueError('Unexpected content student labels')
            return scores, truncated
        pairs = [h for _ in texts for h in CONTENT.values()]
        premises = [text for text in texts for _ in CONTENT]
        values, flags = self.probabilities(premises, batch_size, pairs=pairs)
        entailment = [i for i, label in self.labels.items() if label=='entailment']
        if len(entailment) != 1:
            raise ValueError('Teacher must expose a named entailment label')
        scores = values[:, entailment[0]].reshape(len(texts), len(CONTENT))
        return [dict(zip(CONTENT, map(float, row))) for row in scores], np.asarray(flags).reshape(len(texts), -1).any(1).tolist()

    def tone(self, texts, batch_size):
        # Cardiff normalization also applies to the distilled students.
        cleaned = [re.sub(r'(?<!\w)@\w+', '@user', re.sub(r'https?://\S+', 'http', t)) for t in texts]
        values, flags = self.probabilities(cleaned, batch_size)
        scores = [{self.labels[i]: float(x) for i, x in enumerate(row)} for row in values]
        if scores and set(scores[0]) != {'negative', 'neutral', 'positive'}:
            raise ValueError('Unexpected sentiment model labels')
        return scores, flags


def checkpoint_state(output, name, config, resume):
    output = Path(output)
    if resume:
        state = load_json(output/name)
        if state['config'] != config:
            raise ValueError('Resume configuration/inputs/code/runtime changed; choose a fresh directory')
        for path, digest in state['shards'].items():
            if file_hash(output/path) != digest:
                raise ValueError('Checkpoint shard changed: '+path)
    else:
        new_dir(output); (output/'shards').mkdir()
        state = dict(state='running', created_at_utc=utc_now(), config=config, shards={}, comments=0, clauses=0, seconds=0.)
        atomic_json(output/name, state)
    return state


def score(input_dir, output_dir, backend='teacher', model_dir=None, device='cuda',
          batch_size=32, shard_size=256, cache_dir=None, resume=False, mask_makes=True):
    from sentence_transformers import SentenceTransformer
    import torch
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    cache_dir = Path(cache_dir or output_dir.parent/'.cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    if batch_size < 1 or shard_size < 1:
        raise ValueError('Batch/shard sizes must be positive')
    source = verified_manifest(input_dir, 'prepare_manifest.json')
    hardware = device_info(device)
    if resume:
        frozen = load_json(output_dir/'score_manifest.json')['config']['models']
    elif backend == 'teacher':
        frozen = resolve_revisions(cache_dir)
    else:
        if not model_dir:
            raise ValueError('Student scoring requires --model-dir')
        frozen = load_json(Path(model_dir)/'train_manifest.json')['config']['models']
    trained = None
    if backend == 'student':
        trained = load_json(Path(model_dir)/'train_manifest.json')
        if trained['state'] != 'complete':
            raise ValueError('Student training is incomplete')
        for path, digest in trained['artifacts'].items():
            if file_hash(Path(model_dir)/path) != digest:
                raise ValueError('Trained artifact changed: '+path)
        if trained['source_sha256'] != source['source_sha256']:
            raise ValueError('Student/source snapshots differ')
        if source['as_of'] < trained['as_of']:
            raise ValueError('Student training used later text than the requested source cutoff')
        if mask_makes != trained['mask_makes']:
            raise ValueError('Make masking must match student/reducer training')
    config = dict(input_sha256=file_hash(input_dir/'prepare_manifest.json'), backend=backend,
                  train_sha256=file_hash(Path(model_dir)/'train_manifest.json') if trained else None,
                  models=frozen, hardware=hardware, batch_size=batch_size, shard_size=shard_size,
                  mask_makes=mask_makes, max_tokens=256, topic_threshold=.30,
                  content_threshold=.75, content_margin=.15, tone_threshold=.75, tone_margin=.15,
                  cache_dir=str(cache_dir.resolve()), dependencies=runtime(), code_sha256=implementation_hashes(),
                  hypotheses=CONTENT, topics=TOPICS)
    state = checkpoint_state(output_dir, 'score_manifest.json', config, resume)
    if state['state'] == 'complete':
        return state
    matcher = canonical_matcher(source['canonical_makes'])
    embed_spec = frozen['embedding']
    encoder = SentenceTransformer(embed_spec['model_id'], revision=embed_spec['revision'],
                                  device=device, cache_folder=str(cache_dir), trust_remote_code=False)
    encoder.max_seq_length = 256
    topic_texts = [f'The car has {description}.' for description in TOPICS.values()]
    with amp(device):
        prototypes = encoder.encode(topic_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    content_spec, tone_spec = frozen['content_teacher'], frozen['sentiment_teacher']
    content = Classifier(str(Path(model_dir)/'content') if trained else content_spec['model_id'],
                         device, None if trained else content_spec['revision'], cache_dir)
    tone = Classifier(str(Path(model_dir)/'sentiment') if trained else tone_spec['model_id'],
                      device, None if trained else tone_spec['revision'], cache_dir)
    excluded_videos = set(trained['development_videos']) if trained else set()
    excluded_texts = set(trained['development_text_hashes']) if trained else set()
    part = 0
    for relative in sorted(source['shards']):
        raw = pl.read_parquet(input_dir/relative).to_dicts()
        for start in range(0, len(raw), shard_size):
            shard = f'shards/scores_{part:06d}.parquet'; part += 1
            if shard in state['shards']:
                continue
            begin = time.perf_counter()
            original = raw[start:start+shard_size]
            rows = [r for row in original for r in evidence_rows(row, matcher)]
            active = []
            for index, row in enumerate(rows):
                row.update(content_label='unresolved', content_scores_json='{}', tone_scores_json='{}',
                           topic_scores_json='{}', topics=[], polarity='unresolved', embedding=None,
                           model_truncated=False, feature_eligible=False,
                           development_excluded=row['video_id'] in excluded_videos or row['normalized_text_sha256'] in excluded_texts)
                if (row['language_status'] in ('en', 'short_en_rule') and row['evidence_text'].strip()
                        and not row['direct_price_mention'] and not row['development_excluded']
                        and row['split'] != 'duplicate_excluded'):
                    active.append(index)
            if active:
                texts = [rows[i]['evidence_text'] for i in active]
                labels, truncated = content.content(texts, batch_size, backend=='teacher')
                embedding_text = []
                embed_flags = []
                for text in texts:
                    cleaned = text
                    if mask_makes:
                        for make in matcher(text):
                            cleaned = re.sub(r'(?<!\w)'+re.escape(make)+r'(?!\w)', 'vehicle', cleaned, flags=re.I)
                        for alias, make in ALIASES.items():
                            if make in matcher(text):
                                cleaned = re.sub(r'(?<!\w)'+re.escape(alias)+r'(?!\w)', 'vehicle', cleaned, flags=re.I)
                    embedding_text.append(cleaned)
                    embed_flags.append(len(encoder.tokenizer(cleaned, truncation=False, verbose=False)['input_ids'])>256)
                with amp(device):
                    embeddings = encoder.encode(embedding_text, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
                similarities = np.asarray(embeddings, dtype=np.float32) @ np.asarray(prototypes, dtype=np.float32).T
                opinion_indices = []
                for pos, index in enumerate(active):
                    row = rows[index]
                    label = choose_label(labels[pos])
                    topic_scores = dict(zip(TOPICS, map(float, similarities[pos])))
                    topics = [k for k,v in topic_scores.items() if v>=.30]
                    row.update(content_label=label, content_scores_json=json.dumps(labels[pos]),
                               topic_scores_json=json.dumps(topic_scores), topics=topics,
                               embedding=np.asarray(embeddings[pos], dtype=np.float32).tolist(),
                               model_truncated=bool(truncated[pos] or embed_flags[pos]))
                    row['feature_eligible'] = (label in ('vehicle_opinion', 'vehicle_fact_question') and
                                               row['Make'] is not None and not row['model_truncated'])
                    if label=='vehicle_opinion' and row['Make'] is not None and not row['model_truncated'] and not row['windowed']:
                        opinion_indices.append(index)
                if opinion_indices:
                    tone_scores, tone_trunc = tone.tone([rows[i]['evidence_text'] for i in opinion_indices], batch_size)
                    for index, values, flag in zip(opinion_indices, tone_scores, tone_trunc):
                        row = rows[index]
                        row['tone_scores_json'] = json.dumps(values)
                        row['model_truncated'] |= bool(flag)
                        # A single tone cannot identify separate polarities of several aspects.
                        if len(row['topics'])==1 and row['topics'][0] in SENTIMENT_ASPECTS and not flag:
                            row['polarity'] = choose_label(values)
            temporary = output_dir/(shard+'.tmp')
            # Explicit list types allow all-abstention shards to concatenate safely.
            frame = pl.from_dicts(rows, infer_schema_length=None).with_columns(
                pl.col('Make').cast(pl.String), pl.col('embedding').cast(pl.List(pl.Float32)),
                pl.col('topics').cast(pl.List(pl.String)))
            frame.write_parquet(temporary); temporary.replace(output_dir/shard)
            state['shards'][shard] = file_hash(output_dir/shard)
            state['comments'] += len(original); state['clauses'] += len(rows)
            state['seconds'] += time.perf_counter()-begin
            state.update(source_sha256=source['source_sha256'], population=source['population'], as_of=source['as_of'],
                         canonical_makes=source['canonical_makes'], validation_status='not_independently_validated')
            atomic_json(output_dir/'score_manifest.json', state)
            print(f'{backend}: {state["comments"]:,} comments, {state["clauses"]:,} clauses checkpointed', flush=True)
    state['state'] = 'complete'
    state['comments_per_second'] = state['comments']/max(state['seconds'], .001)
    state['peak_vram_gb'] = round(torch.cuda.max_memory_allocated()/2**30, 3) if device=='cuda' else None
    atomic_json(output_dir/'score_manifest.json', state)
    del encoder, content, tone
    gc.collect()
    if device=='cuda':
        torch.cuda.empty_cache()
    return state
