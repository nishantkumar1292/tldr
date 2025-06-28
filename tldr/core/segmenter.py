from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BasicSegmenter:
    def __init__(self, min_segment_duration=30, max_segment_duration=300):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
        self.min_duration = min_segment_duration  # 30 seconds minimum
        self.max_duration = max_segment_duration  # 5 minutes maximum

    def segment(self, transcript):
        """Segment transcript into topic-based chunks"""
        segments = transcript['segments']
        if len(segments) < 3:
            return [segments]  # Too short to segment

        # Create sliding windows of text
        window_size = 3  # Group 3 consecutive segments for comparison
        windows = []

        for i in range(len(segments) - window_size + 1):
            window_text = ' '.join([seg['text'] for seg in segments[i:i+window_size]])
            windows.append({
                'text': window_text,
                'start_idx': i,
                'start_time': segments[i]['start'],
                'end_time': segments[i+window_size-1]['end']
            })

        # Get embeddings for all windows
        texts = [w['text'] for w in windows]
        embeddings = self.model.encode(texts)

        # Find topic boundaries using similarity drops
        boundaries = [0]  # Always start with first segment

        for i in range(1, len(embeddings)):
            similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]

            # If similarity drops significantly, it's a potential boundary
            if similarity < 0.7:  # Threshold for topic change
                boundary_idx = windows[i]['start_idx']

                # Ensure minimum duration constraint
                if self._check_min_duration(segments, boundaries[-1], boundary_idx):
                    boundaries.append(boundary_idx)

        boundaries.append(len(segments))  # Always end with last segment

        # Create final segments respecting duration constraints
        final_segments = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            segment_parts = segments[start_idx:end_idx]
            final_segments.append(segment_parts)

        return final_segments

    def _check_min_duration(self, segments, start_idx, end_idx):
        """Check if segment meets minimum duration requirement"""
        duration = segments[end_idx-1]['end'] - segments[start_idx]['start']
        return duration >= self.min_duration