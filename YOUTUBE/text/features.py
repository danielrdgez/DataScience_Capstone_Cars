"""Disk-backed deduplication and bounded-memory dated video/make features."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from YOUTUBE.sentiment.common import atomic_json, cutoff_date, file_hash, load_json, new_dir, utc_now
from .data import SENTIMENT_ASPECTS, TOPICS, implementation_hashes, verified_manifest
from .sentiment import EXTRA_FEATURE_COLUMNS, UNCERTAINTY_SUFFIXES, V2_GROUPS


def collapse_polarities(values):
    values = set(values)
    if not values or 'unresolved' in values:
        return 'unresolved'
    if {'positive', 'negative'} <= values:
        return 'mixed'
    return 'positive' if 'positive' in values else 'negative' if 'negative' in values else 'neutral'


def export_price_features(feature_dir, output_dir):
    """Freeze one explicit-comment/cumulative row per canonical make."""
    feature_dir, output_dir = Path(feature_dir), Path(output_dir)
    manifest = load_json(feature_dir/'feature_manifest.json')
    name = 'youtube_text_make_features_wide.parquet'
    if manifest['state'] != 'complete' or file_hash(feature_dir/name) != manifest['shards'][name]:
        raise ValueError('YouTube wide features are incomplete or changed')
    source = pl.read_parquet(feature_dir/name)
    selected = source.filter((pl.col('attribution_scope') == 'explicit_comment') &
                             (pl.col('window') == 'cumulative'))
    if not selected.height or selected['Make'].null_count():
        raise ValueError('No usable canonical make rows')
    if selected['as_of_date'].unique().to_list() != [manifest['as_of']]:
        raise ValueError('YouTube export contains inconsistent cutoffs')
    selected = selected.with_columns(pl.col('Make').str.strip_chars().str.to_uppercase().str.replace_all(r'\s+', ' '))
    if selected['Make'].n_unique() != selected.height or selected.filter(pl.col('Make') == '').height:
        raise ValueError('YouTube make keys are empty or duplicated')
    coverage = ['yt_comment_count', 'yt_eligible_count', 'yt_video_count', 'yt_support_suppressed',
                'yt_semantic_missing'] + [f'yt_{aspect}_topic_missing' for aspect in TOPICS]
    topics = [f'yt_semantic_{i:02d}' for i in range(1, 9)] + [f'yt_{aspect}_mention_share' for aspect in TOPICS]
    sentiment = [f'yt_{aspect}_{suffix}' for aspect in sorted(SENTIMENT_ASPECTS)
                 for suffix in ('video_balanced_net', 'opinion_count', 'opinion_video_count', 'missing')]
    extended = manifest.get('sentiment_schema_version') == 2
    if extended:
        sentiment += list(EXTRA_FEATURE_COLUMNS)
    missing = set(coverage + topics + sentiment) - set(selected.columns)
    if missing:
        raise ValueError('Missing YouTube features: ' + ', '.join(sorted(missing)))
    suppressed = selected.filter(pl.col('yt_support_suppressed'))
    if any(suppressed[c].drop_nulls().len() for c in topics):
        raise ValueError('Unsupported topic/semantic measurements must remain null')
    profile = {}
    for aspect in (V2_GROUPS if extended else sorted(SENTIMENT_ASPECTS)):
        column = f'yt_{aspect}_video_balanced_net'
        values = selected[column].cast(pl.Float64).drop_nulls()
        if values.filter(~values.is_finite() | (values.abs() > 1)).len():
            raise ValueError('Invalid net sentiment range')
        if selected.filter(pl.col(f'yt_{aspect}_missing'))[column].drop_nulls().len():
            raise ValueError('Unsupported sentiment must remain null')
        profile[aspect] = {'populated_makes': len(values), 'distinct_scores': values.n_unique(),
                           'nonzero_scores': int((values != 0).sum())}
        if extended:
            for suffix in UNCERTAINTY_SUFFIXES:
                c = f'yt_{aspect}_{suffix}'; v = selected[c].drop_nulls().cast(pl.Float64)
                lower = -1 if suffix == 'video_balanced_soft_net' else 0
                if v.filter(~v.is_finite() | (v < lower) | (v > 1)).len():
                    raise ValueError('Invalid sentiment measurement: '+c)
                if selected.filter(pl.col(f'yt_{aspect}_missing'))[c].drop_nulls().len():
                    raise ValueError('Unsupported sentiment measurements must remain null')
            soft = selected[f'yt_{aspect}_video_balanced_soft_net'].cast(pl.Float64).drop_nulls()
            profile[aspect]['soft_distinct_scores'] = soft.n_unique()
            profile[aspect]['soft_nonzero_scores'] = int((soft != 0).sum())
    usable_sentiment = any(v['distinct_scores'] > 1 or v.get('soft_distinct_scores', 0) > 1 for v in profile.values())
    selected = selected.select('Make', *coverage, *topics, *sentiment).with_columns(
        pl.lit(1, dtype=pl.Int8).alias('yt_feature_row_present')).sort('Make')
    coverage = ['yt_feature_row_present', *coverage]
    new_dir(output_dir)
    selected.write_parquet(output_dir/'make_features.parquet')
    result = dict(state='complete', created_at_utc=utc_now(), as_of=manifest['as_of'],
                  attribution_scope='explicit_comment', window='cumulative', rows=selected.height,
                  supported_topic_makes=selected.height-int(selected['yt_support_suppressed'].sum()),
                  source_feature_manifest_sha256=file_hash(feature_dir/'feature_manifest.json'),
                  source_wide_sha256=manifest['shards'][name],
                  features_sha256=file_hash(output_dir/'make_features.parquet'),
                  validation_status=manifest['validation_status'], interpretation='retrospective_association',
                  feature_blocks=dict(coverage=coverage, topics=topics, sentiment=sentiment),
                  sentiment_schema_version=2 if extended else 1,
                  sentiment_policy=manifest.get('sentiment_policy', 'threshold'),
                  sentiment_temperature=manifest.get('sentiment_temperature', 1.),
                  score_calibration=manifest.get('score_calibration', 'uncalibrated'),
                  calibration_label_source=manifest.get('calibration_label_source'),
                  sentiment_profile=profile, directional_sentiment_available=usable_sentiment,
                  sentiment_limitation=None if usable_sentiment else
                      'Selected sentiment is constant or missing; this snapshot cannot demonstrate a directional sentiment effect.',
                  code_sha256=implementation_hashes())
    atomic_json(output_dir/'price_feature_manifest.json', result)
    return result


def target_comments(rows, mean, components, extended=False):
    """One comment per make/scope; each topic/opinion is counted at most once."""
    grouped = defaultdict(list)
    for row in rows:
        if row['Make'] and not row['development_excluded']:
            grouped[(row['comment_id'], row['Make'], row['attribution_scope'])].append(row)
    for (cid, make, scope), clauses in grouped.items():
        first = clauses[0]
        eligible = [r for r in clauses if r['feature_eligible']]
        topics = sorted({t for r in eligible for t in r['topics']})
        polarities = defaultdict(list)
        for row in eligible:
            if row['content_label']=='vehicle_opinion':
                for aspect in set(row['topics']) & SENTIMENT_ASPECTS:
                    polarities[aspect].append(row['polarity'])
        vectors = [np.asarray(r['embedding'], dtype=np.float32) for r in eligible if r['embedding'] is not None]
        semantic = ((np.mean(vectors,axis=0)-mean) @ components.T).tolist() if vectors else None
        sentiment = {}
        if extended:
            for aspect in V2_GROUPS:
                evidence = [r for r in eligible if r['content_label'] == 'vehicle_opinion' and
                            (aspect == 'overall' or aspect in r['topics'])]
                if not evidence:
                    continue
                label = collapse_polarities([r['overall_polarity'] if aspect == 'overall' else r['polarity']
                                             for r in evidence])
                valid = label != 'unresolved' and all(r.get('sentiment_soft_net') is not None for r in evidence)
                sentiment[aspect] = dict(polarity=label, **{
                    k: float(np.mean([r['sentiment_'+k] for r in evidence])) if valid else None
                    for k in ('soft_net', 'confidence', 'margin', 'entropy')},
                    low_confidence=float(any(not r['sentiment_confident'] for r in evidence)) if valid else None)
        yield (cid, first['video_id'], make, scope, first['published_date'], first['updated_date'],
               first['normalized_text_sha256'], int(bool(eligible)), json.dumps(topics),
               json.dumps({k:collapse_polarities(v) for k,v in polarities.items()}),
               json.dumps(semantic) if semantic else None, first['language_status'],
               int(any(r['model_truncated'] or r['windowed'] for r in clauses)), json.dumps(sentiment))


def aggregate(score_dir, model_dir, output_dir, as_of, min_comments=20, min_videos=5,
              exclude_cross_video_repeats=False):
    score_dir, model_dir, output_dir = map(Path, (score_dir, model_dir, output_dir))
    if min_comments<1 or min_videos<1:
        raise ValueError('Support minimums must be positive')
    scores = verified_manifest(score_dir, 'score_manifest.json')
    extended = scores['config'].get('sentiment_schema_version') == 2
    model = load_json(model_dir/'train_manifest.json')
    if model['state']!='complete':
        raise ValueError('Reducer training is incomplete')
    for path, digest in model['artifacts'].items():
        if file_hash(model_dir/path)!=digest:
            raise ValueError('Trained artifact changed: '+path)
    if model['source_sha256']!=scores['source_sha256'] or model['embedding_model']!=scores['config']['models']['embedding']:
        raise ValueError('Reducer/source/embedding lineage differs')
    if model['mask_makes']!=scores['config']['mask_makes']:
        raise ValueError('Embedding make masking differs from the trained reducer')
    if scores['config'].get('train_sha256') and scores['config']['train_sha256'] != file_hash(model_dir/'train_manifest.json'):
        raise ValueError('Scores and reducer/student training manifest differ')
    cutoff = cutoff_date(as_of).date()
    if cutoff.isoformat()<model['as_of']:
        raise ValueError('Reducer/students were fitted on text later than this cutoff')
    new_dir(output_dir)
    reducer = np.load(model_dir/'semantic_reducer.npz', allow_pickle=False)
    mean, components = reducer['mean'], reducer['components']
    connection = sqlite3.connect(output_dir/'aggregation_work.sqlite')
    connection.execute('PRAGMA temp_store=FILE')
    connection.execute('PRAGMA cache_size=-65536')
    connection.execute('''CREATE TABLE comments (
        id TEXT, video TEXT, make TEXT, scope TEXT, published TEXT, updated TEXT,
        text_hash TEXT, eligible INTEGER, topics TEXT, polarities TEXT, semantic TEXT,
        language TEXT, truncated INTEGER, sentiment TEXT, PRIMARY KEY (id,make,scope))''')
    audit = Counter()
    for path in sorted(scores['shards']):
        frame = pl.read_parquet(score_dir/path)
        audit['scored_clauses'] += len(frame)
        audit['development_excluded_clauses'] += frame['development_excluded'].sum()
        audit['unresolved_target_clauses'] += frame['Make'].null_count()
        records = list(target_comments(frame.to_dicts(), mean, components, extended))
        # Duplicate IDs from different input shards are a source error, never silently multiplied.
        connection.executemany('INSERT INTO comments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', records)
        connection.commit()
    connection.execute('CREATE INDEX group_time ON comments(make,scope,published,updated,video,text_hash)')
    if exclude_cross_video_repeats:
        connection.execute('CREATE TABLE repeats AS SELECT make,scope,text_hash FROM comments GROUP BY make,scope,text_hash HAVING COUNT(DISTINCT video)>1')
        connection.execute('CREATE INDEX repeat_keys ON repeats(make,scope,text_hash)')
    groups = connection.execute('SELECT DISTINCT make,scope FROM comments ORDER BY make,scope').fetchall()
    video_output, long_output, wide_output = [], [], []
    method = 'minilm_topics_'+scores['config']['backend']+('_sentiment_v2' if extended else '_distillation_v1')
    try:
        trailing_start = cutoff.replace(year=cutoff.year-1)+timedelta(days=1)
    except ValueError:
        trailing_start = cutoff.replace(year=cutoff.year-1,day=28)+timedelta(days=1)
    for window, start_date in [('cumulative','0001-01-01'),('trailing_12_months',trailing_start.isoformat())]:
        for make, scope in groups:
            extra = ('AND NOT EXISTS (SELECT 1 FROM repeats r WHERE r.make=c.make AND r.scope=c.scope AND r.text_hash=c.text_hash)' if exclude_cross_video_repeats else '')
            cursor = connection.execute(f'''SELECT *, ROW_NUMBER() OVER (PARTITION BY video,text_hash ORDER BY id) AS duplicate_rank
                FROM comments c WHERE make=? AND scope=? AND published>=? AND published<=? AND updated<=? {extra}
                ORDER BY video,id''', (make,scope,start_date,cutoff.isoformat(),cutoff.isoformat()))
            per_video = {}
            for record in cursor:
                cid, vid, _, _, published, updated, digest, eligible, topics, polarities, semantic, language, truncated, sentiment, duplicate_rank = record
                stats = per_video.setdefault(vid, dict(raw=0, duplicates=0, comments=0, eligible=0, topics=Counter(),
                            polarities=defaultdict(Counter), sentiment=defaultdict(list), vector_sum=np.zeros(8), vectors=0,
                            language_unknown=0, truncations=0))
                stats['raw'] += 1
                if duplicate_rank>1:
                    stats['duplicates'] += 1
                    continue
                stats['comments'] += 1; stats['eligible'] += eligible
                stats['topics'].update(json.loads(topics))
                for aspect, polarity in json.loads(polarities).items():
                    stats['polarities'][aspect][polarity] += 1
                for aspect, values in json.loads(sentiment).items():
                    if aspect == 'overall':
                        stats['polarities'][aspect][values['polarity']] += 1
                    if values['soft_net'] is not None:
                        stats['sentiment'][aspect].append(values)
                if semantic:
                    stats['vector_sum'] += np.asarray(json.loads(semantic)); stats['vectors'] += 1
                stats['language_unknown'] += language not in ('en','short_en_rule')
                stats['truncations'] += truncated
            if not per_video:
                continue
            videos = list(per_video.values())
            eligible_videos = [v for v in videos if v['eligible']]
            eligible_count = sum(v['eligible'] for v in videos)
            suppressed = eligible_count<min_comments or len(eligible_videos)<min_videos
            base = dict(Make=make, attribution_scope=scope, as_of_date=as_of, window=window,
                        feature_method=method, sampling_population=scores['population'],
                        validation_status='not_independently_validated',
                        cross_video_repeat_policy='excluded' if exclude_cross_video_repeats else 'retained')
            wide = dict(base, yt_comment_count=sum(v['comments'] for v in videos), yt_eligible_count=eligible_count,
                        yt_video_count=len(eligible_videos), yt_support_suppressed=suppressed,
                        yt_duplicate_share=sum(v['duplicates'] for v in videos)/max(1,sum(v['raw'] for v in videos)),
                        yt_unknown_language_share=sum(v['language_unknown'] for v in videos)/max(1,sum(v['comments'] for v in videos)),
                        yt_truncated_share=sum(v['truncations'] for v in videos)/max(1,sum(v['comments'] for v in videos)))
            vectors = [v['vector_sum']/v['vectors'] for v in videos if v['vectors']]
            semantic = np.mean(vectors,axis=0) if vectors else np.zeros(8)
            for index in range(8):
                wide[f'yt_semantic_{index+1:02d}'] = None if suppressed else float(semantic[index])
            wide['yt_semantic_missing'] = suppressed
            for aspect in [*TOPICS, *(['overall'] if extended else [])]:
                mention_share = (float(np.mean([v['topics'][aspect]/v['eligible'] for v in eligible_videos]))
                                 if eligible_videos and aspect != 'overall' else None)
                opinions = Counter()
                nets, opinion_videos = [], 0
                for vid, video in per_video.items():
                    counts = video['polarities'][aspect]
                    opinions.update(counts)
                    denominator = sum(counts[k] for k in ['positive','negative','neutral','mixed'])
                    net = (counts['positive']-counts['negative'])/denominator if denominator else None
                    if net is not None:
                        nets.append(net); opinion_videos += 1
                    video_output.append(dict(base, video_id=vid, aspect=aspect, eligible_count=video['eligible'],
                                    mention_count=video['topics'][aspect] if aspect != 'overall' else None, opinion_count=denominator,
                                    unresolved_opinion_count=counts['unresolved'], net_sentiment=net,
                                    raw_comment_count=video['raw'], duplicate_count=video['duplicates']))
                denominator = sum(opinions[k] for k in ['positive','negative','neutral','mixed'])
                sentiment_suppressed = denominator<min_comments or opinion_videos<min_videos
                net = float(np.mean(nets)) if nets and not sentiment_suppressed else None
                total_mentioned = sum(v['topics'][aspect] for v in videos) if aspect != 'overall' else None
                long_output.append(dict(base, aspect=aspect, eligible_count=eligible_count, eligible_video_count=len(eligible_videos),
                            mention_count=total_mentioned, mention_share=None if suppressed else mention_share,
                            opinion_count=denominator, opinion_video_count=opinion_videos,
                            positive_count=opinions['positive'], negative_count=opinions['negative'],
                            neutral_count=opinions['neutral'], mixed_count=opinions['mixed'],
                            unresolved_opinion_count=opinions['unresolved'], video_balanced_net=net,
                            comment_weighted_net=(opinions['positive']-opinions['negative'])/denominator if denominator and not sentiment_suppressed else None,
                            negative_share=opinions['negative']/denominator if denominator and not sentiment_suppressed else None,
                            mixed_share=opinions['mixed']/denominator if denominator and not sentiment_suppressed else None,
                            topic_support_suppressed=suppressed, sentiment_support_suppressed=sentiment_suppressed))
                if aspect != 'overall':
                    wide[f'yt_{aspect}_mention_share'] = None if suppressed else mention_share
                    wide[f'yt_{aspect}_topic_missing'] = suppressed
                if aspect in SENTIMENT_ASPECTS or aspect == 'overall':
                    wide[f'yt_{aspect}_video_balanced_net'] = net
                    wide[f'yt_{aspect}_opinion_count'] = denominator
                    wide[f'yt_{aspect}_opinion_video_count'] = opinion_videos
                    wide[f'yt_{aspect}_missing'] = sentiment_suppressed
                    if extended:
                        suffix_keys = dict(video_balanced_soft_net='soft_net', mean_confidence='confidence',
                                           mean_margin='margin', mean_entropy='entropy', low_confidence_share='low_confidence')
                        supported_values = [v['sentiment'][aspect] for v in videos if v['sentiment'][aspect]]
                        for suffix, key in suffix_keys.items():
                            value = (float(np.mean([np.mean([r[key] for r in group]) for group in supported_values]))
                                     if supported_values and not sentiment_suppressed else None)
                            wide[f'yt_{aspect}_{suffix}'] = value
                            long_output[-1][suffix] = value
            wide_output.append(wide)
    connection.close()
    if not wide_output:
        raise ValueError('No attributable comments at this cutoff; inspect source/score exclusions')
    shards = {}
    for name, rows in [('youtube_text_video_features.parquet',video_output),
                       ('youtube_text_make_features.parquet',long_output),('youtube_text_make_features_wide.parquet',wide_output)]:
        path = output_dir/name
        pl.from_dicts(rows, infer_schema_length=None).write_parquet(path)
        shards[name] = file_hash(path)
    manifest = dict(state='complete', created_at_utc=utc_now(), score_sha256=file_hash(score_dir/'score_manifest.json'),
                    train_sha256=file_hash(model_dir/'train_manifest.json'), as_of=as_of,
                    sampling_population=scores['population'], wide_rows=len(wide_output), long_rows=len(long_output),
                    video_rows=len(video_output), min_comments=min_comments,min_videos=min_videos,
                    audit=dict(audit), shards=shards, code_sha256=implementation_hashes(),
                    validation_status='not_independently_validated', price_model_integration=False,
                    sentiment_schema_version=2 if extended else 1,
                    sentiment_policy=scores['config'].get('sentiment_policy', 'threshold'),
                    sentiment_temperature=scores['config'].get('sentiment_temperature', 1.),
                    score_calibration='fitted' if scores['config'].get('calibration_sha256') else 'uncalibrated',
                    calibration_label_source=scores['config'].get('calibration_label_source'),
                    support_suppressed_wide_rows=sum(r['yt_support_suppressed'] for r in wide_output))
    atomic_json(output_dir/'feature_manifest.json', manifest)
    return manifest
