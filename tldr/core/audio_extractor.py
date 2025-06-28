import ffmpeg
from pathlib import Path

class AudioExtractor:
    def __init__(self, output_dir="audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def extract(self, video_path, overwrite=False):
        """Extract audio from video file and return audio file path"""
        video_path = Path(video_path)
        audio_path = self.output_dir / f"{video_path.stem}.wav"

        # Check if audio file already exists
        if audio_path.exists() and not overwrite:
            print(f"Audio file already exists: {audio_path}")
            return str(audio_path)

        print(f"Extracting audio to: {audio_path}")
        ffmpeg.input(str(video_path)).output(
            str(audio_path),
            acodec='pcm_s16le',  # 16-bit PCM for best transcription quality
            ac=1,  # Mono channel
            ar=16000  # 16kHz sample rate (optimal for Whisper)
        ).overwrite_output().run(quiet=True)

        return str(audio_path)
