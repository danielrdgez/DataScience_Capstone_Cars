"""GPU raw-source YouTube pipeline; no new manual comment labeling required."""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from YOUTUBE.sentiment.common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor', help='Verify CUDA with a real kernel')
    doctor.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    prep = sub.add_parser('prepare', help='Freeze a raw video-balanced sample, or all eligible source rows')
    prep.add_argument('--source', type=Path, default=ROOT/'DATA/CAR_DATA_FINAL/Youtube_Comments.parquet')
    prep.add_argument('--vpic', type=Path, default=ROOT/'DATA/CAR_DATA_FINAL/NHTSA_vPIC.parquet')
    prep.add_argument('--protection-dir', type=Path, default=ROOT/'YOUTUBE/outputs/relevance_candidate_v1')
    prep.add_argument('--output-dir', type=Path, required=True)
    prep.add_argument('--as-of', required=True, help='Latest allowed publication AND update date (YYYY-MM-DD)')
    prep.add_argument('--sample-size', type=int, default=20000, help='0 = full source; development uses 25%% of videos')
    prep.add_argument('--seed', type=int, default=43)
    scoring = sub.add_parser('score', help='Cache embeddings/topics and teacher or trained-student clause scores')
    scoring.add_argument('--input-dir', type=Path, required=True)
    scoring.add_argument('--output-dir', type=Path, required=True)
    scoring.add_argument('--backend', choices=['teacher', 'student'], default='teacher')
    scoring.add_argument('--model-dir', type=Path)
    scoring.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    scoring.add_argument('--batch-size', type=int, default=32)
    scoring.add_argument('--shard-size', type=int, default=256)
    scoring.add_argument('--cache-dir', type=Path, default=ROOT/'YOUTUBE/.cache/huggingface')
    scoring.add_argument('--resume', action='store_true')
    scoring.add_argument('--keep-make-tokens', action='store_true', help='Identity sensitivity experiment; default masks canonical makes')
    scoring.add_argument('--sentiment-policy', choices=['argmax', 'threshold'],
                         help='Default: argmax for students; threshold for teacher training labels')
    scoring.add_argument('--threshold', type=float, default=.75, help='Diagnostic confidence cutoff; argmax retains lower scores')
    scoring.add_argument('--margin', type=float, default=.15)
    scoring.add_argument('--calibration', type=Path, help='Optional lineage-matched sentiment_calibration.json')
    relabel = sub.add_parser('relabel', help='Rebuild student sentiment decisions from cached scores without GPU inference')
    relabel.add_argument('--score-dir', type=Path, required=True)
    relabel.add_argument('--output-dir', type=Path, required=True)
    relabel.add_argument('--policy', choices=['argmax', 'threshold'], default='argmax')
    relabel.add_argument('--threshold', type=float, default=.75)
    relabel.add_argument('--margin', type=float, default=.15)
    relabel.add_argument('--calibration', type=Path)
    relabel.add_argument('--resume', action='store_true')
    evaluation = sub.add_parser('evaluate-sentiment', help='Compare decision coverage and fit validation-video temperature scaling')
    evaluation.add_argument('--teacher-dir', type=Path, required=True)
    evaluation.add_argument('--model-dir', type=Path, required=True)
    evaluation.add_argument('--output-dir', type=Path, required=True)
    evaluation.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    evaluation.add_argument('--batch-size', type=int, default=32)
    evaluation.add_argument('--cpu-threads', type=int, default=4)
    evaluation.add_argument('--review-csv', type=Path, help='Optional reviewed development predictions with human_label and reviewer')
    fitting = sub.add_parser('train', help='GPU fine-tune content and sentiment MiniLM students, fit training-only PCA')
    fitting.add_argument('--teacher-dir', type=Path, required=True)
    fitting.add_argument('--output-dir', type=Path, required=True)
    fitting.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    fitting.add_argument('--batch-size', type=int, default=16)
    fitting.add_argument('--epochs', type=int, default=3)
    fitting.add_argument('--accumulation', type=int, default=2)
    fitting.add_argument('--learning-rate', type=float, default=2e-5)
    fitting.add_argument('--seed', type=int, default=43)
    fitting.add_argument('--min-class-count', type=int, default=20)
    fitting.add_argument('--max-steps', type=int, help='Bounded engineering run per student; report records this limit')
    fitting.add_argument('--cache-dir', type=Path, default=ROOT/'YOUTUBE/.cache/huggingface')
    fitting.add_argument('--resume', action='store_true')
    agg = sub.add_parser('aggregate', help='Stream cached clauses into video-balanced, dated make features')
    agg.add_argument('--score-dir', type=Path, required=True)
    agg.add_argument('--model-dir', type=Path, required=True, help='Frozen reducer from train')
    agg.add_argument('--output-dir', type=Path, required=True)
    agg.add_argument('--as-of', required=True)
    agg.add_argument('--min-comments', type=int, default=20)
    agg.add_argument('--min-videos', type=int, default=5)
    agg.add_argument('--exclude-cross-video-repeats', action='store_true')
    export = sub.add_parser('price-export', help='Select explicit-comment/cumulative make predictors for price comparisons')
    export.add_argument('--feature-dir', type=Path, required=True)
    export.add_argument('--output-dir', type=Path, required=True)
    inspection = sub.add_parser('inspect', help='Print verified selected sentiment values and coverage')
    inspection.add_argument('--price-dir', type=Path, required=True)
    inspection.add_argument('--makes', nargs='+')
    args = vars(parser.parse_args()); command = args.pop('command')
    if command=='doctor':
        from YOUTUBE.text.neural import device_info
        result = device_info(**args)
    elif command=='prepare':
        from YOUTUBE.text.data import prepare
        args.update(source=args.pop('source'), vpic=args.pop('vpic'), output=args.pop('output_dir'))
        result = prepare(**args)
    elif command=='score':
        from YOUTUBE.text.neural import score
        args['mask_makes'] = not args.pop('keep_make_tokens')
        result = score(**args)
    elif command=='train':
        from YOUTUBE.text.training import train
        result = train(**args)
    elif command=='price-export':
        from YOUTUBE.text.features import export_price_features
        result = export_price_features(**args)
    elif command in ('relabel', 'evaluate-sentiment', 'inspect'):
        from YOUTUBE.text.sentiment import relabel, evaluate_sentiment, inspect_features
        result = {'relabel': relabel, 'evaluate-sentiment': evaluate_sentiment, 'inspect': inspect_features}[command](**args)
    else:
        from YOUTUBE.text.features import aggregate
        result = aggregate(**args)
    print(json.dumps({k:v for k,v in result.items() if k not in
                     {'config','shards','artifacts','canonical_makes','development_videos','development_text_hashes','code_sha256'}}, indent=2))


if __name__=='__main__':
    main()
