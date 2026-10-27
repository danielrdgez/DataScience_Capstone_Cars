"""Train two MiniLM students on confident teachers; evaluate agreement by video."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from YOUTUBE.sentiment.common import atomic_json, file_hash, load_json, new_dir, utc_now
from .data import CONTENT, TONE_LABELS, choose_label, implementation_hashes, runtime, verified_manifest
from .neural import amp, device_info


def training_rows(frame, task):
    """Deduplicate each teacher task, retaining the prepared video split."""
    rows, seen = [], set()
    for row in frame.iter_rows(named=True):
        if row['split'] not in ('train', 'validation', 'test') or row['model_truncated'] or row['windowed']:
            continue
        scores = json.loads(row['content_scores_json'] if task=='content' else row['tone_scores_json'])
        if not scores:
            continue
        if task=='sentiment' and (row['content_label']!='vehicle_opinion' or row['Make'] is None):
            continue
        label = choose_label(scores)
        if label=='unresolved':
            continue
        text = row['evidence_text']
        key = ' '.join(text.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(text=text, label=label, split=row['split'], video_id=row['video_id']))
    return rows


def evaluate(model, tokenizer, rows, labels, device, batch_size):
    import torch
    from sklearn.metrics import classification_report, f1_score
    actual, predicted, loss_total = [], [], 0.
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        encoded = tokenizer([r['text'] for r in batch], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        target = torch.tensor([labels.index(r['label']) for r in batch], device=device)
        with torch.inference_mode(), amp(device):
            logits = model(**encoded).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), target)
        actual.extend(target.cpu().tolist()); predicted.extend(logits.argmax(-1).cpu().tolist())
        loss_total += loss.item()*len(batch)
    if not actual:
        return dict(rows=0, teacher_agreement_macro_f1=None, cross_entropy=None, report={})
    return dict(rows=len(actual), teacher_agreement_macro_f1=float(f1_score(actual, predicted, labels=list(range(len(labels))), average='macro', zero_division=0)),
                cross_entropy=loss_total/len(actual),
                report=classification_report(actual, predicted, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0))


def fit_student(task, rows, output, specification, device, batch_size, epochs, accumulation,
                learning_rate, seed, min_class_count, max_steps, resume, cache_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    labels = list(CONTENT) if task=='content' else TONE_LABELS
    train_rows = [r for r in rows if r['split']=='train']
    val_rows = [r for r in rows if r['split']=='validation']
    test_rows = [r for r in rows if r['split']=='test']
    counts = Counter(r['label'] for r in train_rows)
    unlearned = []
    if task=='content' and not counts['irrelevant']:
        # A teacher abstention is not an irrelevant label. The optional class is
        # learned only when the teacher supplies examples; record missing support.
        labels.remove('irrelevant'); unlearned.append('irrelevant')
    missing = {label: counts[label] for label in labels if counts[label]<min_class_count}
    if missing or not val_rows:
        raise ValueError(f'{task}: insufficient confident pseudo-labels ({missing}); prepare a larger sample. Validation must have separate videos.')
    path = output/task
    path.mkdir(exist_ok=True)
    progress_path = path/'progress.json'
    if resume and progress_path.exists() and load_json(progress_path)['state']=='complete':
        return load_json(progress_path)['result']
    torch.manual_seed(seed)
    if device=='cuda':
        torch.cuda.manual_seed_all(seed)
    options = dict(revision=specification['revision'], cache_dir=str(cache_dir), trust_remote_code=False, token=False)
    tokenizer = AutoTokenizer.from_pretrained(specification['model_id'], **options)
    model = AutoModelForSequenceClassification.from_pretrained(specification['model_id'], num_labels=len(labels),
               id2label=dict(enumerate(labels)), label2id={k:i for i,k in enumerate(labels)}, use_safetensors=False, **options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device=='cuda')
    weights = torch.ones(len(labels), device=device) if task=='content' else torch.tensor([1/counts[label] for label in labels], device=device)
    weights = weights/weights.mean()
    batches_per_epoch = math.ceil(len(train_rows)/batch_size)
    scheduled_steps = epochs*math.ceil(batches_per_epoch/accumulation)
    if max_steps:
        scheduled_steps = min(scheduled_steps, max_steps)
    next_epoch, next_batch, step, best_f1 = 0, 0, 0, -1.
    checkpoint = path/'last.pt'
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        next_epoch, next_batch, step, best_f1 = saved['epoch'], saved['batch'], saved['step'], saved['best_f1']
        torch.set_rng_state(saved['rng'].cpu())
        if device=='cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu())

    def save_checkpoint(epoch, batch):
        temporary = checkpoint.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, batch=batch, step=step, best_f1=best_f1, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None), temporary)
        temporary.replace(checkpoint)
        atomic_json(progress_path, dict(state='running', steps=step, epoch=epoch, next_batch=batch,
                                        checkpoint_sha256=file_hash(checkpoint)))

    for epoch in range(next_epoch, epochs):
        generator = np.random.default_rng(seed+epoch)
        if task=='content':
            probabilities = np.asarray([1/counts[row['label']] for row in train_rows])
            probabilities /= probabilities.sum()
            order = generator.choice(len(train_rows),size=len(train_rows),replace=True,p=probabilities)
        else:
            order = generator.permutation(len(train_rows))
        model.train(); optimizer.zero_grad(set_to_none=True)
        start_batch = next_batch if epoch==next_epoch else 0
        for index in range(start_batch, batches_per_epoch):
            if step>=scheduled_steps:
                break
            batch = [train_rows[i] for i in order[index*batch_size:(index+1)*batch_size]]
            encoded = tokenizer([r['text'] for r in batch], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            target = torch.tensor([labels.index(r['label']) for r in batch], device=device)
            group_length = min(accumulation, batches_per_epoch-(index//accumulation)*accumulation)
            with amp(device):
                logits = model(**encoded).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), target, weight=weights)/group_length
            scaler.scale(loss).backward()
            if (index+1)%accumulation==0 or index+1==batches_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                warmup = max(1, int(.1*scheduled_steps))
                factor = min((step+1)/warmup, max(.05, (scheduled_steps-step)/max(1,scheduled_steps-warmup)))
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate*factor
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); step += 1
                if step%50==0:
                    print(f'{task}: epoch {epoch+1}/{epochs}, optimizer step {step}/{scheduled_steps}', flush=True)
                if step%100==0:
                    save_checkpoint(epoch, index+1)
        validation = evaluate(model, tokenizer, val_rows, labels, device, batch_size)
        if validation['teacher_agreement_macro_f1']>best_f1:
            best_f1 = validation['teacher_agreement_macro_f1']
            model.save_pretrained(path, safe_serialization=True); tokenizer.save_pretrained(path)
        save_checkpoint(epoch+1, 0)
        print(f'{task}: validation teacher-agreement macro F1={validation["teacher_agreement_macro_f1"]:.3f}', flush=True)
        if step>=scheduled_steps:
            break
    best = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=False).to(device)
    result = dict(train_counts=dict(counts), train_rows=len(train_rows), validation=evaluate(best, tokenizer, val_rows, labels, device, batch_size),
                  test=evaluate(best, tokenizer, test_rows, labels, device, batch_size), optimizer_steps=step,
                  labels=labels, unlearned_labels=unlearned,
                  metric_interpretation='agreement with pseudo-label teachers; not automotive accuracy',
                  max_steps_limited=bool(max_steps))
    atomic_json(progress_path, dict(state='complete', result=result))
    del best, model, optimizer
    if device=='cuda':
        torch.cuda.empty_cache()
    return result


def train(teacher_dir, output_dir, device='cuda', batch_size=16, epochs=3, accumulation=2,
          learning_rate=2e-5, seed=43, min_class_count=20, max_steps=None, resume=False, cache_dir=None):
    teacher_dir, output_dir = Path(teacher_dir), Path(output_dir)
    if min(batch_size, epochs, accumulation, min_class_count)<1 or learning_rate<=0 or (max_steps is not None and max_steps<1):
        raise ValueError('Training settings must be positive')
    teacher = verified_manifest(teacher_dir, 'score_manifest.json')
    if teacher['config']['backend']!='teacher':
        raise ValueError('Training requires pretrained teacher outputs')
    if teacher['population']!='video_balanced_development':
        raise ValueError('Use a bounded development sample, never fit on the full feature population')
    hardware = device_info(device)
    cache_dir = Path(cache_dir or teacher['config']['cache_dir'])
    config = dict(teacher_sha256=file_hash(teacher_dir/'score_manifest.json'), models=teacher['config']['models'],
                  hardware=hardware, batch_size=batch_size, epochs=epochs, accumulation=accumulation,
                  learning_rate=learning_rate, seed=seed, min_class_count=min_class_count, max_steps=max_steps,
                  balancing={'content':'inverse_frequency_sampling','sentiment':'inverse_frequency_loss'}, code_sha256=implementation_hashes(),
                  dependencies=runtime(), cache_dir=str(cache_dir.resolve()))
    if resume:
        state = load_json(output_dir/'train_manifest.json')
        if state['config']!=config:
            raise ValueError('Training resume configuration changed')
        if state['state']=='complete':
            for path, digest in state['artifacts'].items():
                if file_hash(output_dir/path)!=digest:
                    raise ValueError('Training artifact changed')
            return state
        for task in ('content', 'sentiment'):
            progress = output_dir/task/'progress.json'
            if progress.exists() and load_json(progress)['state']=='running':
                if file_hash(output_dir/task/'last.pt')!=load_json(progress)['checkpoint_sha256']:
                    raise ValueError('Training checkpoint changed')
    else:
        new_dir(output_dir)
        state = dict(state='running', created_at_utc=utc_now(), config=config, results={})
        atomic_json(output_dir/'train_manifest.json', state)
    frame = pl.concat([pl.read_parquet(teacher_dir/p) for p in sorted(teacher['shards'])], how='vertical_relaxed')
    for task in ('content', 'sentiment'):
        state['results'][task] = fit_student(task, training_rows(frame, task), output_dir, config['models']['student'],
                   device, batch_size, epochs, accumulation, learning_rate, seed, min_class_count, max_steps, resume, cache_dir)
        atomic_json(output_dir/'train_manifest.json', state)
    # Coordinates learned ONLY from eligible training-video text; no test rows in PCA.
    from sklearn.decomposition import PCA
    training = frame.filter((pl.col('split')=='train') & pl.col('feature_eligible') & pl.col('embedding').is_not_null())
    training = training.unique(['normalized_text_sha256', 'evidence_start', 'evidence_end'])
    embeddings = np.asarray(training['embedding'].to_list(), dtype=np.float32)
    if len(embeddings)<8:
        raise ValueError('At least eight eligible training clauses are required for the reducer')
    reducer = PCA(n_components=8, svd_solver='full').fit(embeddings)
    np.savez(output_dir/'semantic_reducer.npz', mean=reducer.mean_, components=reducer.components_,
             explained_variance_ratio=reducer.explained_variance_ratio_)
    state.update(state='complete', source_sha256=teacher['source_sha256'], as_of=teacher['as_of'],
                 mask_makes=teacher['config']['mask_makes'], embedding_model=teacher['config']['models']['embedding'],
                 development_videos=sorted(frame['video_id'].unique().to_list()),
                 development_text_hashes=sorted(frame['normalized_text_sha256'].unique().to_list()),
                 reducer_fit_rows=len(embeddings), validation_status='not_independently_validated',
                 artifacts={str(p.relative_to(output_dir)): file_hash(p) for p in sorted(output_dir.rglob('*'))
                            if p.is_file() and p.name not in ('train_manifest.json', 'last.pt', 'progress.json')})
    atomic_json(output_dir/'train_manifest.json', state)
    return state
