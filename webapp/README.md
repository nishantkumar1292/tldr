# TLDR Web App

A minimal web interface for the TLDR YouTube video segmentation library.

## Overview

This web app provides a simple interface to use the TLDR library for segmenting YouTube videos. It consists of:

- **Backend**: FastAPI server that uses the `tldr` library
- **Frontend**: Simple HTML/JavaScript interface using Tailwind CSS and Alpine.js

## Prerequisites

- Python 3.12+
- FFmpeg (for video processing)
- OpenAI API Key

## Setup

### 1. Install the core library
```bash
# From the parent directory
pip install -e ..
```

### 2. Install web app dependencies
```bash
cd webapp
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

### 4. Run the application

**Option A: Using the run script (recommended)**
```bash
python run.py
```

**Option B: Using uvicorn directly**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Option C: Using Docker**
```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Usage

1. Start the server using one of the methods above
2. Open http://localhost:8000 in your browser
3. Paste a YouTube URL and click "Process"
4. Wait for processing to complete (this may take a few minutes)
5. View the generated segments with titles, summaries, and timestamps

## Features

- **Simple Interface**: Clean, modern UI with real-time processing status
- **Error Handling**: Clear error messages for common issues
- **Responsive Design**: Works on desktop and mobile devices
- **Loading States**: Visual feedback during processing
- **Segment Display**: Shows title, summary, duration, and timestamps for each segment

## API Endpoints

- `GET /` - Main web interface
- `POST /process` - Process a YouTube video (form data: `video_url`)
- `GET /health` - Health check endpoint

## Configuration

You can customize the processing by modifying the `YouTubeSummarizer` parameters in `main.py`:

```python
summarizer = YouTubeSummarizer(
    target_segments=5,      # Number of segments to create
    min_segment_minutes=2,  # Minimum segment duration
    max_segment_minutes=10  # Maximum segment duration
)
```

## Architecture

```
webapp/
├── main.py              # FastAPI application
├── templates/
│   └── index.html       # Main web interface
├── static/
│   └── style.css        # Custom styles
├── requirements.txt     # Python dependencies
├── run.py              # Simple startup script
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose setup
└── README.md           # This file
```

## Future Enhancements

- [ ] User authentication and session management
- [ ] Video processing history and favorites
- [ ] Export options (PDF, markdown, video clips)
- [ ] Batch processing of multiple videos
- [ ] Real-time processing progress with WebSocket
- [ ] Advanced customization options
- [ ] Caching layer for improved performance

## Note

This web app is a demonstration of the TLDR library capabilities. The core functionality is provided by the `tldr` Python package. The web interface does not get installed when you run `pip install tldr`.
