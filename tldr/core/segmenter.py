import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()

MODEL_TEMPERATURES = {
    "o4-mini": 1.0,
}


class IntelligentSegmenter:
    """
    Phase 4: Intelligent Topic Segmentation using OpenAI API

    Uses GPT-4o-mini with intelligent chunking to handle large transcripts
    while respecting API token limits and maintaining context.
    """


    def __init__(self,
                api_key: str = None,
                model: str = "gpt-4.1",
                target_segments: int = 7,
                min_segment_minutes: int = 3,
                max_segment_minutes: int = 15,
                max_tokens_per_request: int = 25000):  # Conservative limit
        """
        Initialize the OpenAI-based segmenter.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI model to use (gpt-4o-mini recommended)
            target_segments: Target number of segments to create
            min_segment_minutes: Minimum segment duration
            max_segment_minutes: Maximum segment duration
            max_tokens_per_request: Max tokens to send in one API call
        """
        self.model = model
        self.target_segments = target_segments
        self.min_segment_minutes = min_segment_minutes
        self.max_segment_minutes = max_segment_minutes
        self.max_tokens_per_request = max_tokens_per_request

        # Initialize OpenAI client
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")

        self.client = OpenAI(api_key=api_key)

    def segment_transcript(self, transcript: Dict[str, Any], video_path: str = None) -> Dict[str, Any]:
        """
        Main method to segment transcript using GPT analysis.
        Adds caching: if segmentation_results/{video_id}.json exists, load and return it.
        """
        segments = transcript['segments']
        duration_minutes = segments[-1]['end'] / 60.0 if segments else 0
        video_id = Path(video_path).stem if video_path else "default"
        cache_dir = Path("segmentation_results")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{video_id}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        print(f"Analyzing {duration_minutes:.1f}-minute transcript with GPT...")
        print(f"Input: {len(segments)} micro-segments")
        print(f"Target: {self.target_segments} intelligent segments")
        full_transcript_text = self._create_timestamped_transcript(segments)
        estimated_tokens = self._estimate_tokens(full_transcript_text)
        print(f"Estimated tokens: {estimated_tokens:,}")
        if estimated_tokens > self.max_tokens_per_request:
            print("Transcript too large, using intelligent chunking...")
            segmentation_result = self._analyze_with_chunking(segments, duration_minutes)
        else:
            print("Analyzing complete transcript...")
            segmentation_result = self._analyze_full_transcript(segments, duration_minutes)
        intelligent_segments = self._map_to_transcript_segments(
            segmentation_result, segments
        )
        result = {
            'segments': intelligent_segments,
            'total_duration_minutes': duration_minutes,
            'original_micro_segments': len(segments),
            'intelligent_segments': len(intelligent_segments),
            'processing_method': 'openai_gpt_smart_chunking',
            'estimated_tokens': estimated_tokens
        }
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return result

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of tokens (1 token ≈ 4 characters for English).
        """
        return len(text) // 4

    def _create_timestamped_transcript(self, segments: List[Dict]) -> str:
        """
        Create transcript with second-based timestamps for easier GPT analysis.
        """
        transcript_lines = []
        for segment in segments:
            # Use seconds directly - much simpler for GPT to work with
            transcript_lines.append(f"[{segment['start']:.1f}s] {segment['text']}")

        return "\n".join(transcript_lines)

    def _analyze_with_chunking(self, segments: List[Dict], duration_minutes: float) -> Dict:
        """
        Analyze transcript using intelligent chunking with overlap.
        """
        # Create overlapping chunks that respect token limits
        chunks = self._create_smart_chunks(segments)
        print(f"Created {len(chunks)} overlapping chunks for analysis")

        all_chunk_results = []

        for i, chunk in enumerate(chunks):
            print(f"Analyzing chunk {i+1}/{len(chunks)} ({chunk['duration_minutes']:.1f}m)...")

            # Calculate target segments for this chunk proportionally
            chunk_target_segments = max(1, int(self.target_segments * chunk['duration_minutes'] / duration_minutes))

            chunk_result = self._analyze_chunk(chunk, chunk_target_segments)
            if chunk_result and 'segments' in chunk_result:
                all_chunk_results.extend(chunk_result['segments'])

        # Return all segments without complex merging for now
        return {'segments': all_chunk_results}

    def _create_smart_chunks(self, segments: List[Dict]) -> List[Dict]:
        """
        Create overlapping chunks that respect token limits while maintaining context.
        """
        chunks = []
        chunk_duration_minutes = 60  # Start with 60-minute chunks
        overlap_minutes = 10  # 10-minute overlap for context

        i = 0
        while i < len(segments):
            chunk_start_time = segments[i]['start']
            chunk_end_time = chunk_start_time + (chunk_duration_minutes * 60)

            # Collect segments for this chunk
            chunk_segments = []
            j = i
            while j < len(segments) and segments[j]['start'] < chunk_end_time:
                chunk_segments.append(segments[j])
                j += 1

            if chunk_segments:
                chunk_text = self._create_timestamped_transcript(chunk_segments)

                chunks.append({
                    'start_time': chunk_start_time,
                    'end_time': chunk_segments[-1]['end'],
                    'duration_minutes': (chunk_segments[-1]['end'] - chunk_start_time) / 60,
                    'text': chunk_text,
                    'segments': chunk_segments
                })

            # Check if we've processed all segments
            if j >= len(segments):
                # We've reached the end of all segments, no need to continue
                break

            # Move to next chunk with overlap
            next_start_time = chunk_start_time + ((chunk_duration_minutes - overlap_minutes) * 60)

            # Find the segment closest to next_start_time
            while i < len(segments) and segments[i]['start'] < next_start_time:
                i += 1

        return chunks

    def _build_segmentation_prompt(self, transcript_text, duration_minutes, segment_count):
        return f"""You are analyzing a {duration_minutes:.1f}-minute podcast transcript to create intelligent topic segments.

TRANSCRIPT (with second timestamps):
{transcript_text}

TASK: Create {segment_count} - {segment_count + 2} meaningful segments that:
1. Each segment must focus on a single, non-repeating, information-dense theme or topic.
2. Do NOT include segments about jokes, banter, or off-topic conversation—ignore all joking around or filler content. These can be part of the main conversation, but not a segment itself.
3. Each segment must start with clear context, so the listener immediately understands what is being discussed and why it matters.
4. Themes must not repeat across segments; each segment should cover a unique aspect or topic.
5. Group only related discussions together, even if separated by brief tangents, but do NOT merge unrelated topics.
6. Each segment should be {self.min_segment_minutes}-{self.max_segment_minutes} minutes long.
7. The content of each segment should be highly information-dense, focusing on insights, analysis, or key facts.
8. If a segment does not have enough information-dense content, do not create it.

INSTRUCTIONS FOR TIMING:
- Analyze the timestamped transcript to identify natural topic transitions
- Use the [X.Xs] timestamps to determine actual start and end times
- Timestamps are already in seconds (e.g., [1234.5s] = 1234.5 seconds)
- Find natural breakpoints where topics shift or new discussions begin
- Ensure segments don't overlap and cover the entire section
- The start_time should be the actual timestamp of the sentence that starts the segment, and must include context for the topic
- The end_time should be the actual timestamp of the sentence that ends the segment

For each segment, provide:
- start_time: Actual start time in seconds based on transcript analysis
- end_time: Actual end time in seconds based on transcript analysis
- title: Descriptive, specific title reflecting the actual content (no generic titles)
- theme: 1-3 word topic category (unique for each segment)
- description: 2-3 sentences explaining what is actually discussed, with clear context at the start

REQUIREMENTS:
- Segments must be chronological (no overlapping times)
- Must cover the entire {duration_minutes:.1f}-minute duration
- Titles should be specific and engaging
- Each segment should have substantial, information-dense content
- Do NOT include segments about jokes, banter
- Themes must not repeat across segments

Return as JSON:
{{
  "segments": [
    {{
      "start_time": "Actual start time in seconds. Only include the number, no other text.",
      "end_time": "Actual end time in seconds. Only include the number, no other text.",
      "title": "Descriptive Title Here",
      "theme": "Topic Category",
      "description": "Clear description of what this segment covers and the key points discussed, with clear context at the start. Don't include statements like 'hosts discuss this..'. Just describe the topic for a person looking at this segment in isolation. Don't mention the timestamps in the description."
    }}
  ]
}}"""

    def _analyze_chunk(self, chunk: Dict, target_segments: int) -> Dict:
        prompt = self._build_segmentation_prompt(
            chunk['text'],
            chunk['duration_minutes'],
            target_segments
        )
        temperature = MODEL_TEMPERATURES.get(self.model, 0.1)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are an expert podcast editor creating intelligent topic segments. PAY CLOSE ATTENTION TO THE DURATION OF THE SEGMENTS CREATED AND MAKE SURE THEY ARE STRICTLY WITHIN THE RANGE OF {self.min_segment_minutes}-{self.max_segment_minutes} MINUTES."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            result = json.loads(response.choices[0].message.content)
            print(f"✓ {target_segments} segments created for chunk")
            return result
        except Exception as e:
            print(f"Chunk analysis error: {e}")
            return None

    def _analyze_full_transcript(self, segments: List[Dict], duration_minutes: float) -> Dict:
        transcript_text = self._create_timestamped_transcript(segments)
        prompt = self._build_segmentation_prompt(
            transcript_text,
            duration_minutes,
            self.target_segments
        )
        temperature = MODEL_TEMPERATURES.get(self.model, 0.1)
        try:
            print("Sending full transcript to GPT for analysis...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are an expert podcast editor creating intelligent topic segments. PAY CLOSE ATTENTION TO THE DURATION OF THE SEGMENTS CREATED AND MAKE SURE THEY ARE STRICTLY WITHIN THE RANGE OF {self.min_segment_minutes}-{self.max_segment_minutes} MINUTES."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            result = json.loads(response.choices[0].message.content)
            print(f"✓ GPT analysis complete - created {len(result.get('segments', []))} segments")
            return result
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._create_fallback_segments(segments, duration_minutes)

    def _parse_time(self, val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            return float(val.strip().replace('s', '').replace('S', ''))
        return 0.0

    def _map_to_transcript_segments(self, gpt_result: Dict, original_segments: List[Dict]) -> List[Dict]:
        """
        Map GPT's time-based segmentation back to the original transcript segments.
        """
        intelligent_segments = []

        for segment_info in gpt_result.get('segments', []):
            start_seconds = self._parse_time(segment_info.get('start_time', 0))
            end_seconds = self._parse_time(segment_info.get('end_time', 0))

            # Find transcript segments within this time range
            segment_texts = []
            for orig_seg in original_segments:
                if start_seconds <= orig_seg['start'] < end_seconds:
                    segment_texts.append(orig_seg['text'])

            # Create intelligent segment
            intelligent_segments.append({
                'start_time': start_seconds,
                'end_time': end_seconds,
                'duration_minutes': (end_seconds - start_seconds) / 60,
                'title': segment_info.get('title', 'Untitled Segment'),
                'theme': segment_info.get('theme', 'General'),
                'description': segment_info.get('description', 'Content segment'),
                'text': ' '.join(segment_texts),
                'word_count': len(' '.join(segment_texts).split())
            })

        return intelligent_segments

    def _create_fallback_segments(self, segments: List[Dict], duration_minutes: float) -> Dict:
        """
        Create basic time-based segments if GPT fails.
        """
        segment_duration = duration_minutes / self.target_segments
        fallback_segments = []

        for i in range(self.target_segments):
            start_time = i * segment_duration * 60
            end_time = min((i + 1) * segment_duration * 60, segments[-1]['end'])

            fallback_segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'title': f"Segment {i+1}",
                'theme': "General",
                'description': f"Content from minutes {start_time/60:.1f} to {end_time/60:.1f}"
            })

        return {'segments': fallback_segments}

    def get_segment_summary(self, segment_results: Dict) -> str:
        """
        Generate a summary of the segmentation results.
        """
        segments = segment_results.get('segments', [])
        total_duration = segment_results.get('total_duration_minutes', 0)
        estimated_tokens = segment_results.get('estimated_tokens', 0)

        summary = "🎯 Intelligent Segmentation Results:\n"
        summary += f"• Duration: {total_duration:.1f} minutes\n"
        summary += f"• Tokens: {estimated_tokens:,} estimated\n"
        summary += f"• Segments: {len(segments)} intelligent segments\n"
        summary += f"• Average: {total_duration/len(segments):.1f} minutes per segment\n\n"

        summary += "📚 Segments:\n"
        for i, segment in enumerate(segments, 1):
            summary += f"{i}. {segment['title']} ({segment['duration_minutes']:.1f}m)\n"
            summary += f"   Theme: {segment['theme']}\n"
            summary += f"   {segment['description']}\n\n"

        return summary

def extract_video_segments(video_path, segments, output_dir):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    manifest_path = Path(output_dir) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    from .extractor import VideoSplitter
    splitter = VideoSplitter()
    results = []
    for i, seg in enumerate(segments):
        start = seg.get('start_time', 0)
        end = seg.get('end_time', 0)
        if end - start < 2:
            continue
        out_path = Path(output_dir) / f'segment_{i+1:03d}.mp4'
        if not out_path.exists():
            splitter.split(video_path, start, end, out_path)
        meta = dict(seg)
        meta['video_path'] = str(out_path)
        results.append(meta)
    with open(manifest_path, "w") as f:
        json.dump(results, f)
    return results
