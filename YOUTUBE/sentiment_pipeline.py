"""One entry point for experimental YouTube sentiment feature creation."""
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
    prep = sub.add_parser('prepare', help='Freeze a protected, balanced engineering subset of saved pilot scores')
    prep.add_argument('--input', type=Path, default=ROOT/'YOUTUBE/outputs/qualification_pilot_v1/pilot_scores.csv')
    prep.add_argument('--candidate-dir', type=Path, default=ROOT/'YOUTUBE/outputs/relevance_candidate_v1')
    prep.add_argument('--output-dir', type=Path, required=True)
    prep.add_argument('--sample-size', type=int, default=1000)
    prep.add_argument('--seed', type=int, default=43)
    score = sub.add_parser('infer', help='Local pretrained models; first run downloads pinned public weights')
    score.add_argument('--sample-dir', type=Path, required=True)
    score.add_argument('--output-dir', type=Path, required=True)
    score.add_argument('--batch-size', type=int, default=16)
    score.add_argument('--device', default='cpu')
    score.add_argument('--cache-dir', type=Path)
    score.add_argument('--resume', action='store_true')
    agg = sub.add_parser('aggregate', help='Dated video/make and wide features at three relevance cutoffs')
    agg.add_argument('--inference-dir', type=Path, required=True)
    agg.add_argument('--output-dir', type=Path, required=True)
    agg.add_argument('--as-of', required=True)
    agg.add_argument('--monthly-start')
    agg.add_argument('--cross-video-sensitivity', action='store_true')
    expert = sub.add_parser('expert', help='Export separate five-aspect DougScore snapshot reference')
    expert.add_argument('--snapshot-dir', type=Path, default=ROOT/'EXPERT_REVIEWS/outputs/dougscore_v1')
    expert.add_argument('--output-dir', type=Path, required=True)
    src = sub.add_parser('source', help='Resumable full-source relevance scoring; preserves protected flags')
    src.add_argument('--source', type=Path, default=ROOT/'DATA/CAR_DATA_FINAL/Youtube_Comments.parquet')
    src.add_argument('--candidate-dir', type=Path, default=ROOT/'YOUTUBE/outputs/relevance_candidate_v1')
    src.add_argument('--output-dir', type=Path, required=True)
    src.add_argument('--batch-size', type=int, default=1024)
    src.add_argument('--device', default='cpu')
    src.add_argument('--resume', action='store_true')
    comparison = sub.add_parser('compare', help='Coverage and descriptive make panels with approved identities only')
    comparison.add_argument('--reference-dir', type=Path, required=True)
    comparison.add_argument('--feature-dir', type=Path, required=True)
    comparison.add_argument('--snapshot-dir', type=Path, default=ROOT/'EXPERT_REVIEWS/outputs/dougscore_v1')
    comparison.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        from YOUTUBE.sentiment.prepare import prepare
        result = prepare(args.input, args.candidate_dir, args.output_dir, args.sample_size, args.seed)
    elif args.command == 'infer':
        from YOUTUBE.sentiment.inference import infer
        result = infer(args.sample_dir, args.output_dir, args.batch_size, args.device, args.resume,
                       str(args.cache_dir) if args.cache_dir else None)
    elif args.command == 'aggregate':
        from YOUTUBE.sentiment.features import aggregate
        result = aggregate(args.inference_dir, args.output_dir, args.as_of, args.monthly_start, args.cross_video_sensitivity)
    elif args.command == 'source':
        from YOUTUBE.sentiment.source import source
        result = source(args.source, args.candidate_dir, args.output_dir, args.batch_size, args.device, args.resume)
    elif args.command == 'compare':
        from YOUTUBE.sentiment.features import compare
        result = compare(args.reference_dir, args.feature_dir, args.snapshot_dir, args.output_dir)
    else:
        from YOUTUBE.sentiment.features import expert_reference
        result = expert_reference(args.snapshot_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
