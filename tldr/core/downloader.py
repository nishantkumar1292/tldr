import yt_dlp
from pathlib import Path

class YouTubeDownloader:
    def __init__(self, output_dir="downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def download(self, url):
        """Download video and return filepath + metadata"""
        opts = {
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'format': 'best'  # Take the best version available
        }

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            return {
                'filepath': filepath,
                'title': info.get('title'),
                'duration': info.get('duration'),
                'url': url
            }
