from faster_whisper import WhisperModel
import torch
import json
import os
from pathlib import Path

class Transcriber:
    def __init__(self, model_name="base", cache_dir="transcripts", device=None):
        # Fix OpenMP conflict on macOS
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        # Use CPU by default to avoid OpenMP conflicts
        # GPU can be enabled explicitly if needed
        if device is None:
            device = "cpu"  # Default to CPU to avoid conflicts
        elif device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")

        # Initialize model with compute type optimization
        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=1  # Reduce threading to avoid conflicts
        )

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

        # Use beam_size=1 for faster processing
        segments, info = self.model.transcribe(
            str(audio_path),
            log_progress=True,
            beam_size=1,  # Faster processing
            language="en",  # Assuming English, speeds up processing
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)
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
