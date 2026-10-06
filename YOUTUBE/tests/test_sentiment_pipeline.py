"""Behavior checks, not a sentiment accuracy evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
import polars as pl
from YOUTUBE.sentiment.common import DEFAULTS, atomic_json, file_hash, make_matcher
from YOUTUBE.sentiment.features import aggregate_frame, aggregate
from YOUTUBE.sentiment.inference import clauses, infer, infer_comment, polarity


class FakeModels:
    calls = 0
    fail_after = None
    def __init__(self, *args):
        pass
    def nli(self, text, hypotheses, max_tokens=512):
        FakeModels.calls += 1
        if self.fail_after and self.calls >= self.fail_after:
            raise RuntimeError('simulated interruption')
        result = []
        for h in hypotheses:
            h = h.lower()
            if h == 'the author expresses an opinion about a vehicle.':
                score = .95
            elif 'styling and appearance' in h:
                if 'positive opinion' in h:
                    score = .95 if 'ugly' not in text else .05
                elif 'negative opinion' in h:
                    score = .95 if 'ugly' in text else .05
                elif 'neutral opinion' in h or 'no opinion' in h:
                    score = .05
                else:
                    score = .95
            else:
                score = .05
            result.append(score)
        return result, False
    def tone(self, text, max_tokens=512):
        return ({'negative': .95, 'positive': .02, 'neutral': .03} if 'ugly' in text else
                {'negative': .02, 'positive': .95, 'neutral': .03}), False


def row(comment='c', video='v', text='Toyota has beautiful styling.', date='2025-01-01'):
    return {'comment_id': comment, 'video_id': video, 'video_title': 'Toyota review',
            'published_at': date, 'text': text, 'prob_vehicle_relevant': .8,
            'sampling_population': 'enriched_pilot_engineering_only'}


def prediction(comment, video, polarity='positive', text_hash=None, date='2025-01-01'):
    return {**row(comment, video, date=date), 'Make': 'TOYOTA', 'attribution_scope': 'explicit_comment',
            'aspect': 'design', 'polarity': polarity, 'normalized_text_sha256': text_hash or comment,
            'language_status': 'en', 'input_truncated_flag': False, 'windowed_flag': False}


class ExperimentalFeaturesTest(unittest.TestCase):
    def setUp(self):
        FakeModels.calls = 0; FakeModels.fail_after = None
        self.matcher = make_matcher(['TOYOTA', 'HONDA', 'VOLKSWAGEN'], {'VW': 'VOLKSWAGEN'})
    def test_token_boundaries_aliases(self):
        self.assertEqual(self.matcher('VW and Honda.'), ['HONDA', 'VOLKSWAGEN'])
        self.assertEqual(self.matcher('Toyotathon'), [])
    def test_evidence_mixed_and_comparison(self):
        rows = infer_comment(row(text='Toyota is beautiful. Toyota is ugly.'), self.matcher, FakeModels(), DEFAULTS, lambda t: 'en')
        design = next(r for r in rows if r['aspect'] == 'design')
        self.assertEqual(design['polarity'], 'mixed')
        for e in json.loads(design['evidence_json']):
            self.assertEqual(design['text'][e['start']:e['end']], e['text'])
        rows = infer_comment(row(text='Toyota is beautiful but Honda is ugly.'), self.matcher, FakeModels(), DEFAULTS, lambda t: 'en')
        self.assertTrue(all(r['polarity'] == 'unresolved' for r in rows if r['aspect'] == 'design'))
    def test_title_does_not_receive_rival_opinion(self):
        rows = infer_comment(row(text='Honda is ugly.'), self.matcher, FakeModels(), DEFAULTS, lambda t: 'en')
        self.assertEqual({r['Make'] for r in rows}, {'HONDA'})
        self.assertEqual({r['attribution_scope'] for r in rows}, {'explicit_comment'})
    def test_unknown_and_filler(self):
        rows = infer_comment(row(), self.matcher, FakeModels(), DEFAULTS, lambda t: 'unknown')
        self.assertTrue(all(r['polarity'] == 'unresolved' for r in rows))
        rows = infer_comment(row(text='great video'), self.matcher, FakeModels(), DEFAULTS, lambda t: 'en')
        self.assertTrue(all(r['polarity'] == 'no_opinion' for r in rows))
        self.assertEqual(FakeModels.calls, 0)
    def test_windows_preserve_tail(self):
        text = 'Toyota '+('very '*400)+'beautiful.'
        parts = list(clauses(text, 100))
        self.assertEqual(''.join(p[2] for p in parts), text)
        self.assertTrue(all(p[3] for p in parts))
    def test_tone_conflict_and_same_clause_ambiguity(self):
        self.assertEqual(polarity({'positive':.9,'negative':.1,'neutral':.1,'no_opinion':.1}, {'negative':.9}, .65, .15), 'unresolved')
        self.assertEqual(polarity({'positive':.95,'negative':.7,'neutral':.1,'no_opinion':.1}, {}, .65, .15), 'unresolved')
    def test_video_balance_dedup_dates_and_negative(self):
        rows = [prediction(str(i),'large','positive') for i in range(10)]
        rows += [prediction('neg','small','negative'), prediction('dup','small','negative','neg'),
                 prediction('future','future', date='2026-01-01'), prediction('bad','bad',date='bad')]
        video, make = aggregate_frame(pl.from_dicts(rows), '2025-12-31', min_opinions=1, min_videos=1)
        m = make.row(0, named=True)
        self.assertEqual(m['opinion_count'], 11)
        self.assertEqual(m['duplicate_count'], 1)
        self.assertAlmostEqual(m['video_balanced_net'], 0)
        self.assertAlmostEqual(m['comment_weighted_net'], 9/11)
        self.assertEqual(video.filter(pl.col('video_id') == 'small')['net_sentiment'][0], -1)
    def test_null_suppression_neutral_denominator_and_trailing_window(self):
        rows = [prediction('p','v','positive'), prediction('n','v','negative'),
                prediction('m','v','mixed'), prediction('z','v','neutral'), prediction('q','v','no_opinion'),
                prediction('old','v','positive',date='2023-01-01')]
        _, make = aggregate_frame(pl.from_dicts(rows), '2025-12-31', 'trailing_12m')
        m = make.row(0,named=True)
        self.assertEqual(m['opinion_count'], 4)
        self.assertEqual(m['no_opinion_count'], 1)
        self.assertTrue(m['support_suppressed'])
        self.assertIsNone(m['video_balanced_net'])
    def test_duplicate_prediction_rejected(self):
        p = prediction('p','v')
        with self.assertRaises(ValueError):
            aggregate_frame(pl.from_dicts([p,p]), '2025-12-31')
    def test_source_retains_protected_rows_and_resumes(self):
        from unittest.mock import patch
        from YOUTUBE.sentiment.source import source
        class FakeRelevance:
            max_seq_length = 256
            @classmethod
            def load(cls, *args):
                return cls()
            def score(self, texts):
                return [.8]*len(texts)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); candidate = root/'candidate'; candidate.mkdir()
            path = root/'comments.parquet'
            pl.from_dicts([row('protected','protected_video'), row('other','open_video')]).write_parquet(path)
            metadata = {'source':{'sha256':file_hash(path)}, 'config':{'max_seq_length':256},
                        'canonical_makes':['TOYOTA'], 'reservation':{'seed':43,'fraction':0}}
            atomic_json(candidate/'candidate_manifest.json', metadata)
            for name in ['protected_train_sample.csv','protected_blind_sample.csv']:
                (candidate/name).write_text('comment_id,video_id,text\nprotected,protected_video,different words\n',encoding='utf-8')
            with patch('YOUTUBE.sentiment.source.candidate_metadata',return_value=metadata), patch('YOUTUBE.sentiment.source.LocalSetFit',FakeRelevance):
                source(path,candidate,root/'scored',batch_size=1)
                source(path,candidate,root/'scored',batch_size=1,resume=True)
            frame=pl.read_parquet(root/'scored/sample.parquet')
            self.assertEqual(len(frame),2)
            self.assertEqual(frame['development_protected'].to_list(),[True,False])
            self.assertEqual(frame['prob_vehicle_relevant'].to_list(),[.8,.8])

    def test_comparison_requires_unambiguous_approved_make(self):
        from YOUTUBE.sentiment.features import compare
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ref=root/'ref'; ref.mkdir(); features=root/'features'; features.mkdir(); snapshot=root/'snapshot'; snapshot.mkdir()
            rpath=ref/'dougscore_experimental_reference.parquet'
            pl.from_dicts([{'review_record_id':'r','aspect':'design','expert_score':8.0}]).write_parquet(rpath)
            atomic_json(ref/'expert_reference_manifest.json',{'output_sha256':file_hash(rpath)})
            mpath=features/'youtube_experimental_make_features.parquet'
            pl.from_dicts([{'Make':'TOYOTA','aspect':'design','video_balanced_net':.5}]).write_parquet(mpath)
            atomic_json(features/'feature_manifest.json',{'files_sha256':{mpath.name:file_hash(mpath)}})
            cross=snapshot/'vehicle_video_crosswalk.csv'
            cross.write_text('review_record_id,Make,match_status\nr,TOYOTA,pending\n',encoding='utf-8')
            result=compare(ref,features,snapshot,root/'pending')
            self.assertEqual(result['approved_review_identities'],0)
            cross.write_text('review_record_id,Make,match_status\nr,TOYOTA,approved\n',encoding='utf-8')
            result=compare(ref,features,snapshot,root/'approved')
            self.assertEqual(result['approved_review_identities'],1)
            self.assertEqual(pl.read_parquet(root/'approved/expert_community_make_panel.parquet')['expert_review_sample_mean'][0],8.0)

    def test_restart_equivalence_and_tamper_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sample = root/'sample'; sample.mkdir()
            pl.from_dicts([row('one'), row('two')]).write_parquet(sample/'sample.parquet')
            atomic_json(sample/'sample_manifest.json', {'sample_sha256': file_hash(sample/'sample.parquet'),
                        'config': DEFAULTS, 'canonical_makes': ['TOYOTA'], 'sampling_population': 'enriched_pilot_engineering_only'})
            options = dict(batch_size=1, model_factory=FakeModels, language_fn=lambda t:'en', revisions={'nli':'fake','tone':'fake'})
            uninterrupted = root/'complete'; interrupted = root/'restart'
            infer(sample, uninterrupted, **options)
            FakeModels.calls = 0; FakeModels.fail_after = 3
            with self.assertRaises(RuntimeError):
                infer(sample, interrupted, **options)
            FakeModels.calls = 0; FakeModels.fail_after = None
            infer(sample, interrupted, resume=True, **options)
            a = pl.read_parquet(uninterrupted/'youtube_experimental_predictions.parquet')
            b = pl.read_parquet(interrupted/'youtube_experimental_predictions.parquet')
            self.assertTrue(a.equals(b))
            aggregate(interrupted, root/'features', '2025-12-31')
            wide = pl.read_parquet(root/'features/youtube_experimental_make_features_wide.parquet')
            self.assertIn('yt_design_video_balanced_net', wide.columns)
            shard = next((interrupted/'shards').glob('*.parquet')); shard.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                infer(sample, interrupted, resume=True, **options)


if __name__ == '__main__':
    unittest.main()
