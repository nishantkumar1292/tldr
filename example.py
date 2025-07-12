#!/usr/bin/env python3
"""
Example usage of the tldr YouTubeSummarizer API
"""

import os
from tldr import YouTubeSummarizer

def main():
    # Make sure you have set your OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return

    # Create a YouTube summarizer instance
    summarizer = YouTubeSummarizer(
        model="gpt-4o-mini",      # Use GPT-4o-mini for cost efficiency
        target_segments=5,        # Create 5 segments
        min_segment_minutes=2,    # Minimum 2 minutes per segment
        max_segment_minutes=8     # Maximum 8 minutes per segment
    )

    # Replace with an actual YouTube video URL
    video_url = "https://youtube.com/watch?v=YOUR_VIDEO_ID"

    print("🎥 YouTube Video Summarizer")
    print("=" * 40)
    print(f"Processing: {video_url}")
    print()

    try:
        # Process the video (this will download, transcribe, and summarize)
        segments = summarizer.process(video_url)

        # Display the results
        print(f"\n✅ Successfully created {len(segments)} segments:")
        print("=" * 50)

        for i, segment in enumerate(segments, 1):
            print(f"\n📝 Segment {i}:")
            print(f"   Title: {segment.title}")
            print(f"   Duration: {segment.duration}")
            print(f"   Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
            print(f"   Summary: {segment.summary}")
            print("-" * 40)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   - You have a valid OpenAI API key")
        print("   - The YouTube URL is valid and accessible")
        print("   - You have sufficient disk space for downloads")

if __name__ == "__main__":
    main()
