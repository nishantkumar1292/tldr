import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class IntelligentSegmenter:
    """
    Phase 4: Intelligent Topic Segmentation using OpenAI API

    Uses GPT-4o-mini with intelligent chunking to handle large transcripts
    while respecting API token limits and maintaining context.
    """

    def __init__(self,
                 api_key: str = None,
                 model: str = "gpt-4o-mini",
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

    def segment_transcript(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to segment transcript using GPT analysis.

        Args:
            transcript: Transcript dict with 'segments' list

        Returns:
            Dictionary with intelligent segments and metadata
        """
        segments = transcript['segments']
        duration_minutes = segments[-1]['end'] / 60.0 if segments else 0

        print(f"Analyzing {duration_minutes:.1f}-minute transcript with GPT...")
        print(f"Input: {len(segments)} micro-segments")
        print(f"Target: {self.target_segments} intelligent segments")

        # Check if we need to chunk the transcript
        full_transcript_text = self._create_timestamped_transcript(segments)
        estimated_tokens = self._estimate_tokens(full_transcript_text)

        print(f"Estimated tokens: {estimated_tokens:,}")

        if estimated_tokens > self.max_tokens_per_request:
            print(f"Transcript too large, using intelligent chunking...")
            segmentation_result = self._analyze_with_chunking(segments, duration_minutes)
        else:
            print("Analyzing complete transcript...")
            segmentation_result = self._analyze_full_transcript(segments, duration_minutes)

        # Map GPT results back to transcript segments
        intelligent_segments = self._map_to_transcript_segments(
            segmentation_result, segments
        )

        return {
            'segments': intelligent_segments,
            'total_duration_minutes': duration_minutes,
            'original_micro_segments': len(segments),
            'intelligent_segments': len(intelligent_segments),
            'processing_method': 'openai_gpt_smart_chunking',
            'estimated_tokens': estimated_tokens
        }

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

    def _analyze_chunk(self, chunk: Dict, target_segments: int) -> Dict:
        """
        Analyze a single chunk with GPT.
        """
        prompt = f"""You are analyzing a {chunk['duration_minutes']:.1f}-minute section of a podcast transcript.

TRANSCRIPT SECTION (with second timestamps):
{chunk['text']}

TASK: Create {target_segments} meaningful segments from this section that:
1. Group related discussions together
2. Have complete narrative arcs
3. Respect natural conversation flow
4. Each segment should be {self.min_segment_minutes}-{self.max_segment_minutes} minutes long

INSTRUCTIONS FOR TIMING:
- Analyze the timestamped transcript to identify natural topic transitions
- Use the [X.Xs] timestamps to determine actual start and end times
- Timestamps are already in seconds (e.g., [1234.5s] = 1234.5 seconds)
- Find natural breakpoints where topics shift or new discussions begin
- Ensure segments don't overlap and cover the entire section
- The start_time should be the actual timestamp of the sentence that starts the segment
- The end_time should be the actual timestamp of the sentence that ends the segment

For each segment, provide:
- start_time: Actual start time in seconds based on transcript analysis
- end_time: Actual end time in seconds based on transcript analysis
- title: Descriptive, specific title reflecting the actual content
- theme: 1-3 word topic category
- description: 2-3 sentences explaining what is actually discussed

EXAMPLE OF HOW TO DETERMINE TIMING:
If you see "[750.0s] We're moving on to discuss AI copyright..."
And the next major topic shift happens at "[1125.5s] Now let's talk about stocks..."
Then create: start_time: 750.0, end_time: 1125.5

Return as JSON:
{{
  "segments": [
    {{
      "start_time": [ACTUAL_START_TIME_IN_SECONDS],
      "end_time": [ACTUAL_END_TIME_IN_SECONDS],
      "title": "[ACTUAL_CONTENT_BASED_TITLE]",
      "theme": "[ACTUAL_THEME]",
      "description": "[ACTUAL_DESCRIPTION_OF_CONTENT]"
    }}
  ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are an expert podcast editor creating intelligent topic segments. PAY CLOSE ATTENTION TO THE DURATION OF THE SEGMENTS CREATED AND MAKE SURE THEY ARE STRICTLY WITHIN THE RANGE OF {self.min_segment_minutes}-{self.max_segment_minutes} MINUTES."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content)
            print(f"✓ Chunk {target_segments} segments created")
            return result

        except Exception as e:
            print(f"Chunk analysis error: {e}")
            return None

    def _merge_overlapping_segments(self, all_segments: List[Dict]) -> List[Dict]:
        """
        Merge segments from overlapping chunks into final coherent segmentation.
        """
        if not all_segments:
            return []

        # Sort by start time
        all_segments.sort(key=lambda x: x.get('start_time', 0))

        merged = []
        current_segment = all_segments[0].copy()

        for segment in all_segments[1:]:
            # If segments overlap significantly, merge them
            overlap = min(current_segment.get('end_time', 0), segment.get('start_time', 0))

            if overlap > 60:  # More than 1 minute overlap
                # Extend current segment
                current_segment['end_time'] = max(current_segment.get('end_time', 0), segment.get('end_time', 0))
                # Combine titles if different
                if segment.get('title') not in current_segment.get('title', ''):
                    current_segment['title'] += f" & {segment.get('title', '')}"
            else:
                # No significant overlap, add current and start new
                merged.append(current_segment)
                current_segment = segment.copy()

        # Add final segment
        merged.append(current_segment)

        # Limit to target number of segments
        if len(merged) > self.target_segments:
            # Keep the longest/most important segments
            merged.sort(key=lambda x: x.get('end_time', 0) - x.get('start_time', 0), reverse=True)
            merged = merged[:self.target_segments]
            merged.sort(key=lambda x: x.get('start_time', 0))

        return merged

    def _analyze_full_transcript(self, segments: List[Dict], duration_minutes: float) -> Dict:
        """
        Send complete transcript to GPT for intelligent segmentation analysis.
        """
        transcript_text = self._create_timestamped_transcript(segments)

        prompt = f"""You are analyzing a {duration_minutes:.0f}-minute podcast transcript to create intelligent topic segments.

COMPLETE TRANSCRIPT (with second timestamps):
{transcript_text}

TASK: Create exactly {self.target_segments} meaningful segments that:
1. Group related discussions together (even if separated by brief tangents)
2. Have complete narrative arcs with clear beginnings and endings
3. Capture the main themes/topics discussed
4. Respect natural conversation flow and transitions
5. Each segment should be {self.min_segment_minutes}-{self.max_segment_minutes} minutes long

ANALYSIS GUIDELINES:
- Look for natural topic transitions and conversation shifts
- Group related discussions that may be interrupted by tangents
- Identify when speakers move to completely new subjects
- Ensure each segment tells a complete story or covers a complete topic
- Create segments that would be useful as standalone podcast chapters

INSTRUCTIONS FOR TIMING:
- Use the [X.Xs] timestamps to determine actual start and end times
- Timestamps are already in seconds (e.g., [1234.5s] = 1234.5 seconds)
- Find natural breakpoints where topics shift or new discussions begin
- Segments must be chronological and non-overlapping

For each segment, provide:
- start_time: Start time in seconds (number)
- end_time: End time in seconds (number)
- title: Descriptive, specific title (e.g., "AI Copyright Ruling Analysis")
- theme: 1-3 word topic category
- description: 2-3 sentences explaining the key points discussed

REQUIREMENTS:
- Segments must be chronological (no overlapping times)
- Must cover the entire {duration_minutes:.0f}-minute duration
- Titles should be specific and engaging
- Each segment should have substantial content

Return as JSON:
{{
  "segments": [
    {{
      "start_time": 0.0,
      "end_time": 300.0,
      "title": "Descriptive Title Here",
      "theme": "Topic Category",
      "description": "Clear description of what this segment covers and the key points discussed."
    }}
  ]
}}"""

        try:
            print("Sending full transcript to GPT for analysis...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert podcast editor who creates intelligent topic-based segments from long-form content. You analyze complete transcripts to find natural breakpoints and create meaningful chapters."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for consistent segmentation
            )

            result = json.loads(response.choices[0].message.content)
            print(f"✓ GPT analysis complete - created {len(result.get('segments', []))} segments")
            return result

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._create_fallback_segments(segments, duration_minutes)

    def _map_to_transcript_segments(self, gpt_result: Dict, original_segments: List[Dict]) -> List[Dict]:
        """
        Map GPT's time-based segmentation back to the original transcript segments.
        """
        intelligent_segments = []

        for segment_info in gpt_result.get('segments', []):
            start_seconds = float(segment_info.get('start_time', 0))
            end_seconds = float(segment_info.get('end_time', 0))

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
