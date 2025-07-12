from dataclasses import dataclass
from typing import List

@dataclass
class Segment:
    """Represents a summarized segment of a video"""
    title: str
    summary: str
    duration: str
    start_time: float
    end_time: float
    video_path: str = None


__all__ = ['YouTubeSummarizer', 'Segment']

from .youtube_summarizer import YouTubeSummarizer
