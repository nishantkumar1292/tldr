from faster_whisper import WhisperModel
import torch
import json
from pathlib import Path

class Transcriber:
    def __init__(self, model_name="base", cache_dir="transcripts"):
        # Auto-detect best available device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")
        self.model = WhisperModel(model_name, device=device)

        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def transcribe(self, audio_path, force_retranscribe=False):
        """Transcribe audio file and return transcript with timestamps"""
        audio_path = Path(audio_path)

        # Create cache filename based on audio file
        cache_file = self.cache_dir / f"{audio_path.stem}.json"

        # Check if cached transcript exists
        if cache_file.exists() and not force_retranscribe:
            print(f"Loading cached transcript: {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        print("Starting transcription...")

        segments, info = self.model.transcribe(
            str(audio_path),
            log_progress=True  # Enable built-in progress logging
        )

        # Convert to list and format
        segment_list = []
        full_text = ""

        for segment in segments:
            segment_list.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
            full_text += segment.text

        transcript = {
            'text': full_text.strip(),
            'segments': segment_list
        }

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(transcript, f, indent=2)
        print(f"Transcript cached: {cache_file}")

        return transcript
