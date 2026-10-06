"""Dated, deduplicated video/make features and separate expert references."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
from .common import (DEFAULTS, METHOD, STATUS, atomic_json, code_hashes, cutoff_date, file_hash,
                     load_json, new_dir, parse_date, utc_now, versions)

KEY = ['Make', 'attribution_scope', 'aspect']
RESOLVED = ['positive', 'negative', 'neutral', 'mixed']


def windows(as_of, monthly):
    end = cutoff_date(as_of)
    if not monthly:
        return [end]
    current = cutoff_date(monthly)
    if current > end:
        raise ValueError('Monthly start must not exceed as-of date')
    result = []
    while (current.year, current.month) <= (end.year, end.month):
        if current.month == 12:
            following = datetime(current.year+1, 1, 1, tzinfo=timezone.utc)
        else:
            following = datetime(current.year, current.month+1, 1, tzinfo=timezone.utc)
        from datetime import timedelta
        result.append(min(end, following-timedelta(microseconds=1)))
        current = following
    return result


def aggregate_frame(frame, as_of, window='cumulative', cutoff=None, min_opinions=20,
                    min_videos=5, exclude_cross_video_duplicates=False):
    if window not in ['cumulative', 'trailing_12m']:
        raise ValueError('Unsupported window')
    if min_opinions < 1 or min_videos < 1:
        raise ValueError('Support minima must be positive')
    end = cutoff_date(as_of)
    start = end.replace(year=end.year-1, day=min(end.day, 28)) if window == 'trailing_12m' else None
    if window == 'trailing_12m':
        # Calendar-month window: the following day after the previous year's same cutoff day.
        import calendar
        start = end.replace(year=end.year-1, day=min(end.day, calendar.monthrange(end.year-1, end.month)[1]))
    parsed = [parse_date(v) for v in frame['published_at'].to_list()]
    frame = frame.with_columns(pl.Series('_date', parsed, dtype=pl.Datetime('us', 'UTC')))
    frame = frame.filter(pl.col('_date').is_not_null() & (pl.col('_date') <= end) & pl.col('Make').is_not_null())
    if start:
        frame = frame.filter(pl.col('_date') > start)
    if cutoff is not None:
        if not 0 <= cutoff <= 1:
            raise ValueError('Cutoff must be in [0, 1]')
        frame = frame.filter(pl.col('prob_vehicle_relevant') >= cutoff)
    if 'development_protected' in frame.columns:
        frame = frame.filter(~pl.col('development_protected'))
    unique_key = ['comment_id', 'Make', 'attribution_scope', 'aspect']
    if frame.select(unique_key).is_duplicated().any():
        raise ValueError('Duplicate prediction key')
    duplicate_key = ['video_id', *KEY, 'normalized_text_sha256']
    cross = frame.group_by([*KEY, 'normalized_text_sha256']).agg(pl.col('video_id').n_unique().alias('_repeat_videos'))
    frame = frame.join(cross, on=[*KEY, 'normalized_text_sha256'])
    if exclude_cross_video_duplicates:
        frame = frame.filter(pl.col('_repeat_videos') == 1)
    raw_counts = frame.group_by(['video_id', *KEY]).agg(pl.len().alias('raw_comment_count'))
    dedup = frame.sort('comment_id').unique(subset=duplicate_key, keep='first', maintain_order=True)
    count_exprs = [(pl.col('polarity') == k).sum().alias(k+'_count') for k in RESOLVED]
    video = dedup.group_by(['video_id', *KEY]).agg(
        pl.len().alias('comment_count'), *count_exprs,
        pl.col('polarity').is_in(RESOLVED).sum().alias('opinion_count'),
        (pl.col('polarity') == 'unresolved').sum().alias('unresolved_count'),
        (pl.col('polarity') == 'no_opinion').sum().alias('no_opinion_count'),
        (pl.col('polarity') == 'aspect_not_present').sum().alias('aspect_not_present_count'),
        (pl.col('language_status') == 'en').sum().alias('english_count'),
        pl.col('input_truncated_flag').sum().alias('truncated_count'),
        pl.col('windowed_flag').sum().alias('windowed_count'),
        (pl.col('_repeat_videos') > 1).sum().alias('cross_video_repeat_count'),
    ).join(raw_counts, on=['video_id', *KEY]).with_columns(
        (pl.col('raw_comment_count')-pl.col('comment_count')).alias('duplicate_count'),
        pl.when(pl.col('opinion_count') > 0).then((pl.col('positive_count').cast(pl.Float64)-pl.col('negative_count').cast(pl.Float64))/pl.col('opinion_count'))
          .otherwise(None).alias('net_sentiment'),
    )
    totals = ['raw_comment_count', 'comment_count', 'opinion_count', 'duplicate_count', 'unresolved_count',
              'no_opinion_count', 'aspect_not_present_count', 'english_count', 'truncated_count',
              'windowed_count', 'cross_video_repeat_count', *[k+'_count' for k in RESOLVED]]
    make = video.group_by(KEY).agg(
        *[pl.col(k).sum().alias(k) for k in totals],
        pl.len().alias('observed_video_count'),
        (pl.col('opinion_count') > 0).sum().alias('video_count'),
        pl.col('net_sentiment').mean().alias('video_balanced_net'),
        pl.col('net_sentiment').std(ddof=0).alias('across_video_std'),
    ).with_columns(
        ((pl.col('opinion_count') < min_opinions) | (pl.col('video_count') < min_videos)).alias('support_suppressed'),
        pl.when(pl.col('opinion_count') > 0).then((pl.col('positive_count').cast(pl.Float64)-pl.col('negative_count').cast(pl.Float64))/pl.col('opinion_count'))
          .otherwise(None).alias('comment_weighted_net'),
        (pl.col('unresolved_count')/pl.col('comment_count')).alias('unresolved_share'),
        (pl.col('duplicate_count')/pl.col('raw_comment_count')).alias('duplicate_share'),
        (pl.col('english_count')/pl.col('comment_count')).alias('english_share'),
        (pl.col('truncated_count')/pl.col('comment_count')).alias('truncation_share'),
    )
    make = make.with_columns(*[pl.when(~pl.col('support_suppressed')).then(pl.col(k)).otherwise(None).alias(k)
                              for k in ['video_balanced_net', 'comment_weighted_net', 'across_video_std']],
                            *[pl.when(~pl.col('support_suppressed')).then(pl.col(k+'_count')/pl.col('opinion_count')).otherwise(None).alias(k+'_share') for k in RESOLVED])
    policy = f'relevance_{cutoff if cutoff is not None else "none"}_cross_repeat_{"excluded" if exclude_cross_video_duplicates else "retained"}'
    def metadata(table):
        return table.with_columns(pl.lit(as_of).alias('as_of_date'), pl.lit(window).alias('window'),
                                  pl.lit(start.isoformat() if start else None, dtype=pl.String).alias('window_start_exclusive'),
                                  pl.lit(end.isoformat()).alias('window_end'), pl.lit(policy).alias('inference_policy'),
                                  pl.lit(METHOD).alias('method'), pl.lit(STATUS).alias('validation_status'))
    return metadata(video), metadata(make)


def wide_features(make):
    index = ['Make', 'attribution_scope', 'as_of_date', 'window', 'inference_policy', 'method', 'validation_status', 'sampling_population', 'inference_manifest_sha256']
    metrics = ['video_balanced_net', 'comment_weighted_net', 'opinion_count', 'video_count', 'support_suppressed', 'unresolved_share', 'duplicate_share']
    keys = make.select(index).unique()
    for aspect in make['aspect'].unique().sort().to_list():
        subset = make.filter(pl.col('aspect') == aspect).select(index+metrics).rename({k: f'yt_{aspect}_{k}' for k in metrics})
        keys = keys.join(subset, on=index, how='left')
        keys = keys.with_columns(pl.col(f'yt_{aspect}_video_balanced_net').is_null().alias(f'yt_{aspect}_missing'))
    return keys


def aggregate(inference_dir, output_dir, as_of, monthly_start=None, cross_sensitivity=False):
    inference_dir, output_dir = Path(inference_dir), Path(output_dir)
    manifest = load_json(inference_dir/'inference_manifest.json')
    predictions = inference_dir/'youtube_experimental_predictions.parquet'
    if manifest['state'] != 'complete' or file_hash(predictions) != manifest['predictions_sha256']:
        raise ValueError('Inference must be complete and unchanged')
    frame = pl.read_parquet(predictions)
    if not len(frame):
        raise ValueError('No prediction rows')
    config = manifest['config']['policy']
    videos, makes = [], []
    for end in windows(as_of, monthly_start):
        day = end.date().isoformat()
        for window in ['cumulative', 'trailing_12m']:
            for cutoff in [None, .2, .5]:
                for exclude in ([False, True] if cross_sensitivity else [False]):
                    v, m = aggregate_frame(frame, day, window, cutoff, config['min_opinions'], config['min_videos'], exclude)
                    videos.append(v); makes.append(m)
    def tagged(tables):
        return pl.concat(tables, how='diagonal_relaxed').with_columns(
            pl.lit(manifest['sampling_population']).alias('sampling_population'),
            pl.lit(file_hash(inference_dir/'inference_manifest.json')).alias('inference_manifest_sha256'))
    video, make = tagged(videos), tagged(makes)
    new_dir(output_dir)
    video.write_parquet(output_dir/'youtube_experimental_video_features.parquet')
    make.write_parquet(output_dir/'youtube_experimental_make_features.parquet')
    wide_features(make).write_parquet(output_dir/'youtube_experimental_make_features_wide.parquet')
    atomic_json(output_dir/'feature_manifest.json', {
        'created_at_utc': utc_now(), 'code_sha256': code_hashes(), 'dependencies': versions(), 'as_of_date': as_of, 'monthly_start': monthly_start,
        'prediction_sha256': file_hash(predictions), 'inference_manifest_sha256': file_hash(inference_dir/'inference_manifest.json'),
        'sampling_population': manifest['sampling_population'], 'method': METHOD, 'validation_status': STATUS,
        'support_minima': {'opinions': config['min_opinions'], 'videos': config['min_videos']},
        'invalid_date_prediction_rows': sum(parse_date(v) is None for v in frame['published_at']),
        'video_feature_rows': len(video), 'make_feature_rows': len(make),
        'price_model_consumes_features': False,
        'files_sha256': {p.name: file_hash(p) for p in output_dir.glob('*.parquet')},
    })
    return {'video_rows': len(video), 'make_rows': len(make), 'output': str(output_dir)}


def expert_reference(snapshot, output_dir):
    snapshot, output_dir = Path(snapshot), Path(output_dir)
    manifest = load_json(snapshot/'import_manifest.json')
    for name in ['expert_review_scores.parquet', 'expert_aspect_scores.parquet']:
        if file_hash(snapshot/name) != manifest['files_sha256'][name]:
            raise ValueError('Imported expert snapshot changed')
    reviews = pl.read_parquet(snapshot/'expert_review_scores.parquet')
    aspects = pl.read_parquet(snapshot/'expert_aspect_scores.parquet')
    # Crosswalk stays separate: multiple cohort candidates must not duplicate review/aspect rows.
    crosswalk = pl.read_csv(snapshot/'vehicle_video_crosswalk.csv', infer_schema_length=0)
    matches = crosswalk.group_by('review_record_id').agg(
        (pl.col('match_status') == 'approved').sum().alias('approved_match_count'),
        pl.len().alias('identity_record_count'))
    reference = aspects.join(reviews.drop('source_snapshot_sha256'), on='review_record_id', validate='m:1').join(matches, on='review_record_id', how='left').with_columns(
        pl.when(pl.col('approved_match_count') > 0).then(pl.lit('approved_identity_available_in_crosswalk'))
          .otherwise(pl.lit('pending')).alias('identity_status'),
        pl.lit('review_snapshot').alias('target_scope'), pl.lit('expert_reference').alias('method'),
        pl.lit('proposed_mapping_linkage_pending').alias('validation_status'),
        (pl.col('expert_score')-5.5).truediv(4.5).alias('expert_display_index'),
        (pl.col('aspect') != 'price_value').alias('first_price_experiment_candidate'),
    )
    if reference.select(['review_record_id', 'aspect']).is_duplicated().any():
        raise ValueError('Duplicate review/aspect identity')
    new_dir(output_dir)
    path = output_dir/'dougscore_experimental_reference.parquet'
    reference.write_parquet(path)
    atomic_json(output_dir/'expert_reference_manifest.json', {
        'created_at_utc': utc_now(), 'code_sha256': code_hashes(), 'dependencies': versions(), 'rows': len(reference), 'reviews': reviews.height,
        'source_files_sha256': {p.name: file_hash(p) for p in [snapshot/'expert_review_scores.parquet', snapshot/'expert_aspect_scores.parquet', snapshot/'vehicle_video_crosswalk.csv']},
        'output_sha256': file_hash(path), 'comparison_generated': False, 'price_model_consumes_features': False,
    })
    return {'reference': str(path), 'rows': len(reference)}


def compare(reference_dir, feature_dir, snapshot_dir, output_dir):
    """Descriptive compatible make panels only where approved canonical identity exists."""
    reference_dir, feature_dir, snapshot_dir, output_dir = map(Path, (reference_dir, feature_dir, snapshot_dir, output_dir))
    reference_path = reference_dir/'dougscore_experimental_reference.parquet'
    ref_manifest = load_json(reference_dir/'expert_reference_manifest.json')
    if file_hash(reference_path) != ref_manifest['output_sha256']:
        raise ValueError('Expert reference changed')
    make_path = feature_dir/'youtube_experimental_make_features.parquet'
    feature_manifest = load_json(feature_dir/'feature_manifest.json')
    if not make_path.exists():
        make_path = feature_dir/'youtube_text_make_features.parquet'
    declared_hashes = feature_manifest.get('files_sha256', feature_manifest.get('shards', {}))
    if file_hash(make_path) != declared_hashes.get(make_path.name):
        raise ValueError('Community features changed')
    crosswalk_path = snapshot_dir/'vehicle_video_crosswalk.csv'
    crosswalk = pl.read_csv(crosswalk_path, infer_schema_length=0)
    approved = crosswalk.filter((pl.col('match_status') == 'approved') & pl.col('Make').is_not_null() & (pl.col('Make').str.strip_chars() != '')).select('review_record_id', 'Make').unique()
    # An approved review with contradictory makes remains ambiguous.
    ambiguous = approved.group_by('review_record_id').len().filter(pl.col('len') > 1)
    approved = approved.join(ambiguous.select('review_record_id'), on='review_record_id', how='anti')
    reference = pl.read_parquet(reference_path)
    expert = reference.join(approved, on='review_record_id', how='inner').group_by('Make', 'aspect').agg(
        pl.col('expert_score').mean().alias('expert_review_sample_mean'),
        pl.col('review_record_id').n_unique().alias('expert_review_count'))
    community = pl.read_parquet(make_path)
    panel = community.join(expert, on=['Make', 'aspect'], how='left').with_columns(
        pl.lit('selective_expert_review_sample_vs_youtube_make_opinions').alias('population_scope'),
        pl.lit('descriptive_only_unverified_expert_time').alias('comparison_status'))
    new_dir(output_dir)
    panel.write_parquet(output_dir/'expert_community_make_panel.parquet')
    atomic_json(output_dir/'comparison_coverage.json', {
        'expert_reviews': reference['review_record_id'].n_unique(),
        'approved_review_identities': approved['review_record_id'].n_unique(),
        'ambiguous_approved_review_identities': ambiguous.height,
        'community_make_aspect_cells': community.select('Make','aspect').unique().height,
        'supported_panel_rows': panel.filter(pl.col('expert_review_count').is_not_null()).height,
        'scope': 'descriptive_make_panel; no same-review/video or cohort comparison asserted',
        'expert_community_blend_defined': False,
        'source_files_sha256': {'reference':file_hash(reference_path),'features':file_hash(make_path),'crosswalk':file_hash(crosswalk_path)},
    })
    return {'panel_rows': len(panel), 'approved_review_identities': approved['review_record_id'].n_unique(), 'output': str(output_dir)}
