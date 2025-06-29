import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class IntelligentSegmenter:
    """
    Phase 4 Task 4.1: Theme Discovery and Topic Modeling

    Extracts 6-12 main themes from entire transcript using BERTopic,
    identifies recurring topics, and creates semantic fingerprints for each theme.
    """

    def __init__(self,
                 target_themes: int = 8,
                 min_themes: int = 6,
                 max_themes: int = 12,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 verbose: bool = False):
        """
        Initialize the intelligent segmenter.

        Args:
            target_themes: Target number of themes to extract
            min_themes: Minimum number of themes
            max_themes: Maximum number of themes
            embedding_model: Sentence transformer model for embeddings
        """
        self.target_themes = target_themes
        self.min_themes = min_themes
        self.max_themes = max_themes
        self.verbose = verbose

        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model)
        self.topic_model = None

        # State
        self.themes = {}
        self.theme_embeddings = None
        self.segment_theme_scores = None

    def _create_text_chunks(self, segments: List[Dict], chunk_duration: int = 30) -> Tuple[List[str], List[List[int]]]:
        """
        Combine micro-segments into larger chunks for more robust theme discovery.

        Args:
            segments: List of transcript segments
            chunk_duration: Target duration for each chunk in seconds

        Returns:
            Tuple of (chunk_texts, chunk_mapping) where chunk_mapping shows which segments belong to each chunk
        """
        chunks = []
        chunk_mapping = []
        current_chunk = []
        current_mapping = []
        current_duration = 0

        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            if not text:
                continue

            segment_duration = segment.get('end', 0) - segment.get('start', 0)

            # Add to current chunk
            current_chunk.append(text)
            current_mapping.append(i)
            current_duration += segment_duration

            # Check if chunk is large enough
            if current_duration >= chunk_duration or len(current_chunk) >= 10:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) > 20:  # Only include substantial chunks
                    chunks.append(chunk_text)
                    chunk_mapping.append(current_mapping.copy())

                # Reset for next chunk
                current_chunk = []
                current_mapping = []
                current_duration = 0

        # Handle remaining segments
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) > 20:
                chunks.append(chunk_text)
                chunk_mapping.append(current_mapping)

        if self.verbose:
            print(f"Created {len(chunks)} text chunks from {len(segments)} segments")
        return chunks, chunk_mapping

    def discover_themes(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task 4.1: Extract main themes from entire transcript using BERTopic.

        Args:
            transcript: Transcript with segments containing text and timestamps

        Returns:
            Dictionary containing discovered themes and metadata
        """
        segments = transcript['segments']

        # Preprocess: combine micro-segments into larger chunks for theme discovery
        # This prevents memory issues and gives better semantic coherence
        chunk_texts, chunk_mapping = self._create_text_chunks(segments)

        if len(chunk_texts) < self.min_themes:
            return self._create_fallback_themes(segments)

        if self.verbose:
            print(f"Discovering themes from {len(chunk_texts)} text chunks...")

        try:
            # Configure BERTopic for our use case
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                nr_topics=self.target_themes,
                min_topic_size=max(2, len(chunk_texts) // 15),  # Dynamic based on content length
                calculate_probabilities=True,
                verbose=self.verbose
            )

            # Fit the model and extract topics
            topics, probabilities = self.topic_model.fit_transform(chunk_texts)

        except Exception as e:
            if self.verbose:
                print(f"BERTopic failed with error: {e}")
                print("Falling back to simpler clustering approach...")
            return self._create_fallback_themes(segments)

        # Get topic information
        topic_info = self.topic_model.get_topic_info()

        # Filter out noise topic (-1) and ensure we have the right number of themes
        valid_topics = topic_info[topic_info['Topic'] != -1]

        if len(valid_topics) < self.min_themes:
            if self.verbose:
                print(f"Warning: Only found {len(valid_topics)} valid themes, using fallback approach")
            return self._create_fallback_themes(segments)

        # Limit to max themes if we found too many
        if len(valid_topics) > self.max_themes:
            valid_topics = valid_topics.head(self.max_themes)

        # Create theme fingerprints and metadata
        themes = self._create_theme_fingerprints(valid_topics, chunk_texts, topics, probabilities)

        # Score each segment against discovered themes
        segment_scores = self._score_segments_against_themes(segments, themes)

        return {
            'themes': themes,
            'segment_theme_scores': segment_scores,
            'topic_model': self.topic_model,
            'total_segments': len(segments),
            'total_themes': len(themes),
            'theme_coverage': self._calculate_theme_coverage(segment_scores),
            'chunk_mapping': chunk_mapping  # Include for debugging
        }

    def _create_theme_fingerprints(self,
                                 topic_info: Any,
                                 chunk_texts: List[str],
                                 topics: List[int],
                                 probabilities: np.ndarray) -> Dict[str, Dict]:
        """
        Create semantic fingerprints for each discovered theme.
        """
        themes = {}

        for _, row in topic_info.iterrows():
            topic_id = row['Topic']

            # Get top words for this topic
            topic_words = self.topic_model.get_topic(topic_id)
            if not topic_words:
                continue

            # Extract chunks belonging to this topic
            topic_chunks = [
                chunk_texts[i] for i, t in enumerate(topics)
                if t == topic_id
            ]

            # Get representative text snippets
            representative_texts = topic_chunks[:3]  # Top 3 most representative

            # Create embedding fingerprint for this theme
            theme_text = ' '.join([word for word, _ in topic_words[:10]])
            theme_embedding = self.embedding_model.encode([theme_text])[0]

            # Calculate theme statistics
            theme_probabilities = [
                probabilities[i][topic_id] if topic_id < len(probabilities[i]) else 0.0
                for i, t in enumerate(topics) if t == topic_id
            ]

            # Generate descriptive title from top words
            top_words = [word for word, _ in topic_words[:5]]
            theme_title = self._generate_theme_title(top_words, representative_texts)

            themes[f"theme_{topic_id}"] = {
                'id': topic_id,
                'title': theme_title,
                'keywords': top_words,
                'embedding': theme_embedding,
                'representative_texts': representative_texts,
                'segment_count': len(topic_chunks),
                'avg_confidence': np.mean(theme_probabilities) if theme_probabilities else 0.0,
                'topic_words': topic_words[:10]  # Top 10 words with scores
            }

        return themes

    def _score_segments_against_themes(self,
                                     segments: List[Dict],
                                     themes: Dict[str, Dict]) -> List[Dict]:
        """
        Score each transcript segment against all discovered themes.
        """
        segment_scores = []

        # Get theme embeddings
        theme_ids = list(themes.keys())
        theme_embeddings = np.array([themes[tid]['embedding'] for tid in theme_ids])

        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            if not text:
                segment_scores.append({
                    'segment_idx': i,
                    'theme_scores': {tid: 0.0 for tid in theme_ids},
                    'primary_theme': None,
                    'confidence': 0.0,
                    'is_multi_theme': False
                })
                continue

            # Get segment embedding
            segment_embedding = self.embedding_model.encode([text])[0]

            # Calculate similarity scores with all themes
            similarities = cosine_similarity([segment_embedding], theme_embeddings)[0]

            # Create theme scores dictionary
            theme_scores = {
                theme_ids[j]: float(similarities[j]) for j in range(len(theme_ids))
            }

            # Determine primary theme and confidence
            primary_theme = max(theme_scores.items(), key=lambda x: x[1])

            # Check if this is a multi-theme segment (multiple high scores)
            high_scores = [score for score in similarities if score > 0.7]
            is_multi_theme = len(high_scores) > 1

            segment_scores.append({
                'segment_idx': i,
                'theme_scores': theme_scores,
                'primary_theme': primary_theme[0],
                'confidence': primary_theme[1],
                'is_multi_theme': is_multi_theme,
                'start_time': segment['start'],
                'end_time': segment['end'],
                'text': text[:100] + '...' if len(text) > 100 else text
            })

        return segment_scores

    def _generate_theme_title(self, top_words: List[str], representative_texts: List[str]) -> str:
        """
        Generate a descriptive title for the theme based on keywords and content.
        """
        # Simple heuristic: capitalize key words and create a meaningful title
        if not top_words:
            return "Miscellaneous Discussion"

        # Look for common patterns in words to create better titles
        title_words = []
        for word in top_words[:3]:  # Use top 3 words
            if len(word) > 2:  # Skip very short words
                title_words.append(word.title())

        if not title_words:
            title_words = [top_words[0].title()]

        # Create title
        if len(title_words) == 1:
            return f"{title_words[0]} Discussion"
        elif len(title_words) == 2:
            return f"{title_words[0]} and {title_words[1]}"
        else:
            return f"{title_words[0]}, {title_words[1]} and {title_words[2]}"

    def _calculate_theme_coverage(self, segment_scores: List[Dict]) -> Dict[str, float]:
        """
        Calculate how well themes cover the entire transcript.
        """
        if not segment_scores:
            return {}

        theme_coverage = {}
        total_segments = len(segment_scores)

        # Count segments per theme
        theme_counts = {}
        for score_data in segment_scores:
            primary_theme = score_data.get('primary_theme')
            if primary_theme:
                theme_counts[primary_theme] = theme_counts.get(primary_theme, 0) + 1

        # Calculate coverage percentages
        for theme_id, count in theme_counts.items():
            theme_coverage[theme_id] = count / total_segments

        return theme_coverage

    def _create_fallback_themes(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Fallback approach when BERTopic fails or insufficient data.
        Uses simple clustering on embeddings.
        """
        if self.verbose:
            print("Using fallback theme discovery with embedding clustering...")

        texts = [seg['text'] for seg in segments if seg['text'].strip()]
        if len(texts) < 3:
            return {
                'themes': {'theme_0': {
                    'id': 0,
                    'title': 'General Discussion',
                    'keywords': ['discussion', 'topic'],
                    'embedding': np.zeros(384),  # Default embedding size
                    'representative_texts': texts,
                    'segment_count': len(texts),
                    'avg_confidence': 1.0,
                    'topic_words': [('discussion', 1.0)]
                }},
                'segment_theme_scores': [],
                'topic_model': None,
                'total_segments': len(segments),
                'total_themes': 1,
                'theme_coverage': {'theme_0': 1.0}
            }

        # Get embeddings and cluster - use smaller sample for fallback
        sample_size = min(200, len(texts))
        sample_texts = texts[:sample_size]
        embeddings = self.embedding_model.encode(sample_texts)
        n_clusters = min(self.target_themes, len(sample_texts) // 2)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Create simple themes from clusters
        themes = {}
        for i in range(n_clusters):
            cluster_texts = [sample_texts[j] for j in range(len(sample_texts)) if clusters[j] == i]
            themes[f"theme_{i}"] = {
                'id': i,
                'title': f"Topic {i+1}",
                'keywords': ['topic', 'discussion'],
                'embedding': kmeans.cluster_centers_[i],
                'representative_texts': cluster_texts[:3],
                'segment_count': len(cluster_texts),
                'avg_confidence': 0.8,
                'topic_words': [('topic', 1.0), ('discussion', 0.8)]
            }

        return {
            'themes': themes,
            'segment_theme_scores': [],
            'topic_model': None,
            'total_segments': len(segments),
            'total_themes': len(themes),
            'theme_coverage': {f"theme_{i}": 1.0/n_clusters for i in range(n_clusters)}
        }

    def get_theme_summary(self, theme_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of discovered themes.
        """
        themes = theme_results['themes']
        total_themes = theme_results['total_themes']

        summary = f"\n=== Theme Discovery Results ===\n"
        summary += f"Discovered {total_themes} main themes:\n\n"

        for i, (theme_id, theme_data) in enumerate(themes.items(), 1):
            summary += f"{i}. {theme_data['title']}\n"
            summary += f"   Keywords: {', '.join(theme_data['keywords'])}\n"
            summary += f"   Segments: {theme_data['segment_count']}\n"
            summary += f"   Confidence: {theme_data['avg_confidence']:.2f}\n\n"

        return summary