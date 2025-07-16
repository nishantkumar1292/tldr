#!/usr/bin/env python3
"""
TLDR Web App - FastAPI Backend

A minimal web interface for the TLDR YouTube video segmentation library.
"""

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

# Global storage for processing results (in production, use a database)
processing_results = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_video(video_url: str = Form(...)):
    """Process a YouTube video and return segments"""

    # Validate URL
    if not video_url or not ("youtube.com" in video_url or "youtu.be" in video_url):
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL")

    try:
        # Initialize the summarizer
        summarizer = YouTubeSummarizer(
            target_segments=5,  # Smaller number for faster processing
            min_segment_minutes=0.5,
            max_segment_minutes=2
        )

        # Process the video
        segments = summarizer.process(video_url)

        # Convert segments to JSON-serializable format
        result = {
            "success": True,
            "video_url": video_url,
            "segments": [
                {
                    "title": segment.title,
                    "summary": segment.summary,
                    "duration": segment.duration,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time
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
