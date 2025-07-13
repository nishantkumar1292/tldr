import argparse
import os
import json
from tldr.youtube_summarizer import YouTubeSummarizer

def main():
    parser = argparse.ArgumentParser(description='Summarize YouTube videos.')
    subparsers = parser.add_subparsers(dest='command')

    yt_parser = subparsers.add_parser('youtube', help='Summarize a YouTube video')
    yt_parser.add_argument('url', type=str, help='YouTube video URL')
    yt_parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    yt_parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use for summarization')
    yt_parser.add_argument('--target-segments', type=int, default=7, help='Number of segments to create')
    yt_parser.add_argument('--min-segment-minutes', type=int, default=3, help='Minimum segment duration in minutes')
    yt_parser.add_argument('--max-segment-minutes', type=int, default=15, help='Maximum segment duration in minutes')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nExample: tldr youtube <url> --output-dir <dir>")
        return

    if args.command == 'youtube':
        summarizer = YouTubeSummarizer(
            model=args.model,
            target_segments=args.target_segments,
            min_segment_minutes=args.min_segment_minutes,
            max_segment_minutes=args.max_segment_minutes
        )
        segments = summarizer.process(args.url)
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, 'summary.json')
        with open(out_path, 'w') as f:
            json.dump([s.__dict__ for s in segments], f, indent=2)
        print(f'Summary saved to {out_path}')

if __name__ == '__main__':
    main()
