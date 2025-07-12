import os
from pathlib import Path
from typing import List
from .core.downloader import YouTubeDownloader
from .core.extractor import AudioExtractor, VideoSplitter
from .core.transcriber import Transcriber
from .core.segmenter import IntelligentSegmenter
from . import Segment

class YouTubeSummarizer:
    def __init__(self,
                 openai_api_key: str = None,
                 model: str = "gpt-4o-mini",
                 target_segments: int = 7,
                 min_segment_minutes: int = 3,
                 max_segment_minutes: int = 15,
                 output_dir: str = "output"):
        """
        Initialize YouTube summarizer

        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI model for summarization
            target_segments: Target number of segments to create
            min_segment_minutes: Minimum segment duration
            max_segment_minutes: Maximum segment duration
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.downloader = YouTubeDownloader(output_dir="downloads")
        self.extractor = AudioExtractor(output_dir="audio")
        self.video_splitter = VideoSplitter()
        self.transcriber = Transcriber(cache_dir="transcripts")
        self.segmenter = IntelligentSegmenter(
            api_key=openai_api_key,
            model=model,
            target_segments=target_segments,
            min_segment_minutes=min_segment_minutes,
            max_segment_minutes=max_segment_minutes
        )

    def process(self, url: str) -> List[Segment]:
        """
        Process a YouTube video and return summarized segments

        Args:
            url: YouTube video URL

        Returns:
            List of Segment objects with title, summary, and duration
        """
        print(f"Processing YouTube video: {url}")

        # Step 1: Download video
        print("Downloading video...")
        video_info = self.downloader.download(url)
        video_path = video_info['filepath']
        video_title = video_info['title']

        # Step 2: Extract audio
        print("Extracting audio...")
        audio_path = self.extractor.extract(video_path)

        # Step 3: Transcribe audio
        print("Transcribing audio...")
        transcript = self.transcriber.transcribe(audio_path)

        # Step 4: Segment and summarize
        print("Segmenting and summarizing...")
        segmentation_result = self.segmenter.segment_transcript(transcript, video_path)

        # Step 5: Convert to Segment objects and create video segments
        segments = []
        segments_dir = self.output_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        for i, seg in enumerate(segmentation_result['segments']):
            duration_seconds = seg['end_time'] - seg['start_time']
            duration_str = self._format_duration(duration_seconds)

            # Create video segment
            segment_filename = f"segment_{i+1:02d}_{seg['title'].replace(' ', '_')[:30]}.mp4"
            segment_path = segments_dir / segment_filename

            print(f"Creating video segment {i+1}/{len(segmentation_result['segments'])}: {segment_filename}")
            self.video_splitter.split(
                video_path,
                seg['start_time'],
                seg['end_time'],
                segment_path
            )

            segment = Segment(
                title=seg['title'],
                summary=seg['description'],
                duration=duration_str,
                start_time=seg['start_time'],
                end_time=seg['end_time'],
                video_path=str(segment_path)
            )
            segments.append(segment)

        print(f"Created {len(segments)} video segments")
        return segments

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable string"""
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
