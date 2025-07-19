#!/usr/bin/env python3
"""
TLDR Web App - FastAPI Backend

A minimal web interface for the TLDR YouTube video segmentation library.
"""

import sys
from pathlib import Path
# Add parent directory to path for importing tldr library
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import the TLDR library
try:
    from tldr import YouTubeSummarizer
except ImportError:
    raise ImportError("TLDR library not found. Please run: pip install -e ..")

app = FastAPI(title="TLDR Web App", version="0.1.0")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount output directory for serving video segments
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
app.mount("/videos", StaticFiles(directory="output"), name="videos")

# Global storage for processing results (in production, use a database)
processing_results = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_video(
    video_url: str = Form(...),
    target_segments: int = Form(5),
    min_segment_minutes: float = Form(0.5),
    max_segment_minutes: float = Form(2.0)
):
    """Process a YouTube video and return segments"""

    # Validate URL
    if not video_url or not ("youtube.com" in video_url or "youtu.be" in video_url):
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL")

    try:
        # Initialize the summarizer with user parameters
        summarizer = YouTubeSummarizer(
            target_segments=target_segments,
            min_segment_minutes=min_segment_minutes,
            max_segment_minutes=max_segment_minutes,
            output_dir="output"
        )

        # Process the video
        segments = summarizer.process(video_url)

        # Convert segments to JSON-serializable format
        result = {
            "success": True,
            "video_url": video_url,
            "settings": {
                "target_segments": target_segments,
                "min_segment_minutes": min_segment_minutes,
                "max_segment_minutes": max_segment_minutes
            },
            "segments": [
                {
                    "title": segment.title,
                    "summary": segment.summary,
                    "duration": segment.duration,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "video_url": f"/videos/{Path(segment.video_path).relative_to(output_dir)}" if hasattr(segment, 'video_path') and segment.video_path else None
                }
                for segment in segments
            ]
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Failed to process video. Please check the URL and try again."
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tldr-web-app"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
