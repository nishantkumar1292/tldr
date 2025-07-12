#!/usr/bin/env python3
"""
Test script for the YouTubeSummarizer API
"""

import os
from tldr import YouTubeSummarizer

def main():
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Initialize the summarizer
    summarizer = YouTubeSummarizer(
        model="gpt-4o-mini",
        target_segments=5,
        min_segment_minutes=2,
        max_segment_minutes=10
    )

    # Example YouTube URL (replace with actual video)
    video_url = "https://youtube.com/watch?v=YOUR_VIDEO_ID"

    print("YouTubeSummarizer API Test")
    print("=" * 50)
    print(f"Processing: {video_url}")
    print()

    try:
        # Process the video
        segments = summarizer.process(video_url)

        # Display results
        print(f"\nCreated {len(segments)} segments:")
        print("-" * 50)

        for i, segment in enumerate(segments, 1):
            print(f"\nSegment {i}:")
            print(f"Title: {segment.title}")
            print(f"Duration: {segment.duration}")
            print(f"Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
            print(f"Summary: {segment.summary}")
            print("-" * 30)

    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()
