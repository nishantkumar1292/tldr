# tldr: Summarize Long-form Content

Summarize long-form content from YouTube (for now, more content support coming soon..). **tldr**, modified from TL;DR ([Wikipedia](https://en.wikipedia.org/wiki/TL;DR)), is a python library for summarizing. Originally tldr is used in context of texts, but this library can be used for summarizing any kind of data.

## Installation

```bash
# Clone the repo
$ git clone https://github.com/nishantkumar1292/tldr.git
$ cd tldr

# Install dependencies (recommended in a virtualenv)
$ uv pip install -e .
```

## Usage

### Python API (YouTube Example)
```python
from tldr import YouTubeSummarizer
summarizer = YouTubeSummarizer()
segments = summarizer.process("https://youtube.com/watch?v=YOUR_VIDEO_ID")
for segment in segments:
    print(f"Title: {segment.title}")
    print(f"Summary: {segment.summary}")
    print(f"Duration: {segment.duration}")
```

### CLI
```bash
# Summarize a YouTube video
$ tldr youtube https://youtube.com/watch?v=YOUR_VIDEO_ID --output-dir ./output
```

---


## Content Types Supported
- **Long youtube videos** (LIVE) - Extract highlights from any video URL from YouTube, extracts the snippets, and also provides a summary of extracted snippets.
    - Customizable Params:
        - Number of segments to create
        - Minimum and Maximum duration of the segments
        - Model to use for the summarization (gpt-4.1, gpt-4-turbo, o4-mini etc.)
- **Long Research Papers** - Helps with extracting the key points from the paper. Creates a markdown file with screenshots of the relevant section, and creates a summary.

### Extended scope
- **Sports Highlights** - This library can be used to create sports highlights from long sports match videos.

## TODO
- [ ] Publish package to PyPI as `omni-tldr`
- [ ] Set up proper package structure with setup.py/pyproject.toml
- [ ] Create documentation and usage examples
- [ ] Implement video summarization functionality
- [ ] Implement research paper summarization functionality
- [ ] infer number of segments from the video length
