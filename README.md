tldr, modified from TL;DR ([Wikipedia](https://en.wikipedia.org/wiki/TL;DR)), is a python library for summarizing. Now originally tldr is used in context of texts, but this library can be used for summarizing any kind of data.

## What kind of data can be summarized?
- **Long youtube videos** - This library can help with extracting highlights from the video, extracts the snippets, and also provides a summary of extracted snippets.
    - Arguments for the summarization:
        - length of the summary
        - resolution of the snippets
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
