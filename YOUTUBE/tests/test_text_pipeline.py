"""Consequential attribution, split, polarity and feature-denominator checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from YOUTUBE.sentiment.common import atomic_json, file_hash
from YOUTUBE.text.data import canonical_matcher, duplicate_hash, evidence_rows, video_split
from YOUTUBE.text.features import aggregate, collapse_polarities, export_price_features
from YOUTUBE.text.training import training_rows


class RawTextPipelineTests(unittest.TestCase):
    def test_attribution_offsets_and_competing_makes(self):
        matcher = canonical_matcher(['TOYOTA','NISSAN','MINI','RAM'])
        def clauses(text, title='Toyota review'):
            return list(evidence_rows(dict(text=text,video_title=title),matcher))
        self.assertIsNone(clauses('Great review, Alex!')[0]['Make'])
        self.assertEqual(matcher('A mini version is smart'), [])
        self.assertEqual(matcher('Ram 1500 engine'), ['RAM'])
        self.assertIsNone(clauses('Toyota is better than Nissan')[0]['Make'])
        text = 'Toyota is reliable but Nissan engines worry me.'
        rows = clauses(text)
        self.assertEqual([r['Make'] for r in rows], ['TOYOTA','NISSAN'])
        for row in rows:
            self.assertEqual(row['evidence_text'],text[row['evidence_start']:row['evidence_end']])
        self.assertEqual(clauses('This car has comfortable seats')[0]['attribution_scope'],'title_context')
        self.assertIsNone(clauses('Nissan is reliable. This car is slow')[1]['Make'])
        long = 'Toyota engine '+('good '*450)
        self.assertEqual(''.join(r['evidence_text'] for r in clauses(long)),long)
        self.assertTrue(all(r['windowed'] for r in clauses(long)))

    def test_duplicate_meaning_and_video_split(self):
        self.assertEqual(duplicate_hash('  Toyota   is GOOD '),duplicate_hash('toyota is good'))
        self.assertNotEqual(duplicate_hash('Love Toyota 🙂'),duplicate_hash('Love Toyota 🙁'))
        self.assertNotEqual(duplicate_hash('good'),duplicate_hash('not good'))
        self.assertEqual(video_split('video-a',43),video_split('video-a',43))

    def test_teacher_rows_never_mix_splits_or_make_facts_neutral(self):
        records = []
        for split,text,content,tone in [('train','Toyota is reliable','vehicle_opinion',{'positive':.96,'negative':.02,'neutral':.02}),
                                        ('test','Toyota is reliable','vehicle_opinion',{'positive':.96,'negative':.02,'neutral':.02}),
                                        ('validation','Toyota has four doors','vehicle_fact_question',{})]:
            records.append(dict(split=split,evidence_text=text,video_id=split,model_truncated=False,windowed=False,
                content_label=content,Make='TOYOTA',content_scores_json=json.dumps({k:.98 if k==content else .01 for k in ['vehicle_opinion','vehicle_fact_question','creator_other','irrelevant']}),tone_scores_json=json.dumps(tone)))
        rows = training_rows(pl.from_dicts(records),'sentiment')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['split'],'train')
        self.assertEqual(collapse_polarities(['positive','negative']),'mixed')
        self.assertEqual(collapse_polarities(['positive','unresolved']),'unresolved')

    def test_video_balance_dedup_dates_support_and_fact_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); scored=root/'scores'; model=root/'model'
            scored.mkdir(); model.mkdir()
            np.savez(model/'semantic_reducer.npz',mean=np.zeros(8),components=np.eye(8))
            embed=dict(model_id='stub',revision='frozen')
            atomic_json(model/'train_manifest.json',dict(state='complete',source_sha256='source',as_of='2020-12-31',
                embedding_model=embed,mask_makes=True,artifacts={'semantic_reducer.npz':file_hash(model/'semantic_reducer.npz')}))
            rows=[]
            def add(cid,vid,polarity,text_hash,content='vehicle_opinion',date='2020-01-01'):
                rows.append(dict(comment_id=cid,video_id=vid,Make='TOYOTA',attribution_scope='explicit_comment',
                    published_date=date,updated_date=date,normalized_text_sha256=text_hash,feature_eligible=True,
                    content_label=content,topics=['comfort'],polarity=polarity,embedding=[1.]*8,language_status='en',
                    model_truncated=False,windowed=False,development_excluded=False))
            for index in range(4): add(f'p{index}','a','positive',f'p{index}')
            add('dup','a','positive','p0')
            add('n','b','negative','n')
            add('f','b','unresolved','f',content='vehicle_fact_question')
            add('future','b','negative','future',date='2021-01-01')
            pl.from_dicts(rows).write_parquet(scored/'predictions.parquet')
            atomic_json(scored/'score_manifest.json',dict(state='complete',source_sha256='source',population='eligible_source_snapshot',
                config=dict(models={'embedding':embed},mask_makes=True,backend='student'),shards={'predictions.parquet':file_hash(scored/'predictions.parquet')}))
            aggregate(scored,model,root/'features','2020-12-31',min_comments=2,min_videos=2)
            wide=pl.read_parquet(root/'features/youtube_text_make_features_wide.parquet').filter(pl.col('window')=='cumulative').row(0,named=True)
            self.assertEqual(wide['yt_eligible_count'],6)
            self.assertEqual(wide['yt_comfort_opinion_count'],5)
            self.assertEqual(wide['yt_comfort_video_balanced_net'],0.)
            self.assertEqual(wide['yt_comfort_mention_share'],1.)
            self.assertAlmostEqual(wide['yt_duplicate_share'],1/7)
            selected_manifest = export_price_features(root/'features', root/'price_features')
            selected = pl.read_parquet(root/'price_features/make_features.parquet')
            self.assertEqual(selected.height, 1)
            self.assertEqual(selected['Make'].to_list(), ['TOYOTA'])
            self.assertEqual(selected_manifest['supported_topic_makes'], 1)
            self.assertFalse(selected_manifest['directional_sentiment_available'])
            self.assertEqual(selected_manifest['sentiment_profile']['comfort']['nonzero_scores'], 0)
            self.assertNotIn('window', selected.columns)
            self.assertNotIn('as_of_date', selected.columns)
            self.assertEqual(selected['yt_feature_row_present'].to_list(), [1])
            aggregate(scored,model,root/'suppressed','2020-12-31',min_comments=20,min_videos=5)
            wide=pl.read_parquet(root/'suppressed/youtube_text_make_features_wide.parquet').row(0,named=True)
            self.assertIsNone(wide['yt_comfort_video_balanced_net'])
            self.assertTrue(wide['yt_semantic_missing'])
            export_price_features(root/'suppressed', root/'price_suppressed')
            selected = pl.read_parquet(root/'price_suppressed/make_features.parquet')
            self.assertIsNone(selected['yt_semantic_01'][0])
            self.assertIsNone(selected['yt_comfort_video_balanced_net'][0])
            changed = pl.read_parquet(root/'features/youtube_text_make_features_wide.parquet')
            pl.concat([changed, changed]).write_parquet(root/'features/youtube_text_make_features_wide.parquet')
            with self.assertRaisesRegex(ValueError, 'changed'):
                export_price_features(root/'features', root/'bad_price_features')


if __name__=='__main__':
    unittest.main()
