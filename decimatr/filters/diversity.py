"""
Diversity filter for selecting frames that maximize tag diversity.

This filter maintains a temporal buffer and selects frames that are most
different from frames already in the buffer, ensuring a diverse sample of
frames across various metrics (blur, entropy, color, etc.).
"""

import numpy as np

from decimatr.filters.base import StatefulFilter
from decimatr.scheme import VideoFramePacket


class DiversityFilter(StatefulFilter):
    """
    Filter that selects frames maximizing tag diversity.

    This filter maintains a temporal buffer and uses a diversity scoring
    mechanism to select frames that are most different from frames already
    in the buffer. This is useful for creating diverse frame samples that
    capture the full range of variation in a video.

    The filter computes diversity based on specified tag keys (e.g., blur_score,
    entropy, color histograms) and only passes frames that increase the overall
    diversity of the buffer.

    Attributes:
        buffer_size: Maximum number of frames to keep in temporal buffer
        diversity_tags: List of tag keys to use for diversity calculation
        min_distance: Minimum distance threshold for a frame to be considered diverse
        metric: Distance metric to use ('euclidean', 'manhattan', 'cosine')

    Example:
        >>> # Select diverse frames based on blur and entropy
        >>> filter = DiversityFilter(
        ...     buffer_size=100,
        ...     diversity_tags=['blur_score', 'entropy'],
        ...     min_distance=0.1,
        ...     metric='euclidean'
        ... )
        >>>
        >>> # Process frames
        >>> for packet in frame_stream:
        ...     if filter.should_pass(packet):
        ...         # Frame adds diversity, process it
        ...         process_frame(packet)
    """

    def __init__(
        self,
        buffer_size: int = 100,
        diversity_tags: list[str] | None = None,
        min_distance: float = 0.1,
        metric: str = "euclidean",
    ):
        """
        Initialize diversity filter.

        Args:
            buffer_size: Maximum number of diverse frames to maintain.
                        Larger values allow more diversity but use more memory.
            diversity_tags: List of tag keys to use for diversity calculation.
                           If None, uses all numeric tags found in frames.
                           Common tags: ['blur_score', 'entropy', 'edge_density']
            min_distance: Minimum distance threshold for a frame to pass.
                         Frame must be at least this distance from all frames
                         in buffer to be considered diverse enough.
                         Range depends on metric and tag scales.
            metric: Distance metric to use:
                   - 'euclidean': Euclidean distance (L2 norm)
                   - 'manhattan': Manhattan distance (L1 norm)
                   - 'cosine': Cosine distance (1 - cosine similarity)

        Raises:
            ValueError: If metric is invalid or min_distance is negative
        """
        super().__init__(buffer_size)

        valid_metrics = ["euclidean", "manhattan", "cosine"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

        if min_distance < 0:
            raise ValueError(f"min_distance must be non-negative, got {min_distance}")

        self.diversity_tags = diversity_tags or []
        self.min_distance = min_distance
        self.metric = metric

    def _extract_feature_vector(
        self, packet: VideoFramePacket, tag_keys: list[str]
    ) -> np.ndarray | None:
        """
        Extract feature vector from packet tags.

        Args:
            packet: VideoFramePacket with tags
            tag_keys: List of tag keys to extract

        Returns:
            Numpy array of feature values, or None if tags are missing
        """
        features = []

        for key in tag_keys:
            value = packet.get_tag(key)

            if value is None:
                return None  # Missing required tag

            # Handle different value types
            if isinstance(value, int | float):
                features.append(float(value))
            elif isinstance(value, np.ndarray):
                # Flatten arrays (e.g., histograms, embeddings)
                features.extend(value.flatten().tolist())
            else:
                # Skip non-numeric values
                continue

        if not features:
            return None

        return np.array(features, dtype=float)

    def _compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute distance between two feature vectors.

        Args:
            vec1: First feature vector
            vec2: Second feature vector

        Returns:
            Distance value (higher = more different)
        """
        # Ensure same length
        if len(vec1) != len(vec2):
            # Pad shorter vector with zeros
            max_len = max(len(vec1), len(vec2))
            vec1 = np.pad(vec1, (0, max_len - len(vec1)))
            vec2 = np.pad(vec2, (0, max_len - len(vec2)))

        if self.metric == "euclidean":
            # Euclidean distance (L2 norm)
            return float(np.linalg.norm(vec1 - vec2))

        elif self.metric == "manhattan":
            # Manhattan distance (L1 norm)
            return float(np.sum(np.abs(vec1 - vec2)))

        elif self.metric == "cosine":
            # Cosine distance (1 - cosine similarity)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance for zero vectors

            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(1.0 - cosine_sim)

        return 0.0

    def _get_tag_keys(self, packet: VideoFramePacket, history: list[VideoFramePacket]) -> list[str]:
        """
        Determine which tag keys to use for diversity calculation.

        Args:
            packet: Current packet
            history: History packets

        Returns:
            List of tag keys to use
        """
        if self.diversity_tags:
            # Use explicitly configured tags
            return self.diversity_tags

        # Auto-detect numeric tags from current packet
        numeric_tags = []
        for key, value in packet.tags.items():
            if isinstance(value, int | float | np.ndarray):
                numeric_tags.append(key)

        return numeric_tags

    def compare_with_history(
        self, packet: VideoFramePacket, history: list[VideoFramePacket]
    ) -> bool:
        """
        Compare current frame against historical frames for diversity.

        Extracts feature vectors from the current frame and all frames in the
        buffer, then computes the minimum distance. If the minimum distance
        exceeds the threshold, the frame is diverse enough to pass.

        Args:
            packet: Current frame to evaluate
            history: List of recent frames from the temporal buffer

        Returns:
            True if frame is diverse enough (passes), False otherwise
        """
        # If buffer is empty, always pass (first frame)
        if not history:
            return True

        # Determine which tags to use for diversity
        tag_keys = self._get_tag_keys(packet, history)

        if not tag_keys:
            # No tags available for diversity calculation, pass by default
            return True

        # Extract feature vector for current frame
        current_features = self._extract_feature_vector(packet, tag_keys)

        if current_features is None:
            # Missing required tags, skip this frame
            return False

        # Compute minimum distance to any frame in history
        min_distance = float("inf")

        for past_packet in history:
            past_features = self._extract_feature_vector(past_packet, tag_keys)

            if past_features is None:
                continue

            distance = self._compute_distance(current_features, past_features)
            min_distance = min(min_distance, distance)

        # Frame passes if it's sufficiently different from all history frames
        return min_distance >= self.min_distance

    @property
    def required_tags(self) -> list[str]:
        """
        Return list of tag keys required by this filter.

        Returns the configured diversity_tags if specified, otherwise returns
        empty list (will auto-detect numeric tags at runtime).

        Returns:
            List of required tag keys, or empty list for auto-detection
        """
        return self.diversity_tags if self.diversity_tags else []
