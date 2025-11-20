"""
Diversity filter for selecting frames that maximize tag diversity.

This filter maintains a temporal buffer and selects frames that are most
different from frames already in the buffer, ensuring a diverse sample of
frames across various metrics (blur, entropy, color, etc.).
"""

import numpy as np
from loguru import logger

from decimatr.filters.base import StatefulFilter
from decimatr.filters.comparison_strategies import (
    ComparisonStrategy,
    EmbeddingDistanceStrategy,
    HammingDistanceStrategy,
    HistogramDistanceStrategy,
)
from decimatr.filters.tag_classification import TagClassificationRegistry
from decimatr.scheme import VideoFramePacket


class DiversityFilter(StatefulFilter):
    """
    Filter that selects frames maximizing tag diversity.

    This filter maintains a temporal buffer and uses a diversity scoring
    mechanism to select frames that are most different from frames already
    in the buffer. This is useful for creating diverse frame samples that
    capture the full range of variation in a video.

    The filter has been enhanced to distinguish between diversity-suitable tags
    (hashes, embeddings, histograms) and metric-only tags (blur_score, entropy).
    It supports different comparison strategies for different tag types:
    - Perceptual hashes: Hamming distance
    - Embeddings: Euclidean, cosine, or Manhattan distance
    - Color histograms: Intersection, chi-square, or Bhattacharyya distance

    Attributes:
        buffer_size: Maximum number of frames to keep in temporal buffer
        diversity_tags: List of tag keys to use for diversity calculation (None = auto-detect)
        min_distance: Minimum distance threshold for a frame to be considered diverse
        metric: Legacy distance metric ('euclidean', 'manhattan', 'cosine')
        comparison_strategies: Dict mapping tag keys to comparison strategies
        enable_weighted_combination: Whether to combine multiple tag distances
        tag_weights: Weights for combining multiple tag distances

    Example:
        >>> # Select diverse frames using hash-based diversity
        >>> filter = DiversityFilter(
        ...     buffer_size=100,
        ...     diversity_tags=['dhash'],
        ...     min_distance=0.1
        ... )
        >>>
        >>> # Select diverse frames using CLIP embeddings with custom strategy
        >>> from decimatr.filters.comparison_strategies import EmbeddingDistanceStrategy
        >>> filter = DiversityFilter(
        ...     buffer_size=100,
        ...     diversity_tags=['clip_embedding'],
        ...     min_distance=0.1,
        ...     comparison_strategies={
        ...         'clip_embedding': EmbeddingDistanceStrategy(metric='cosine')
        ...     }
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
        comparison_strategies: dict[str, ComparisonStrategy] | None = None,
        enable_weighted_combination: bool = False,
        tag_weights: dict[str, float] | None = None,
    ):
        """
        Initialize diversity filter with enhanced capabilities.

        Args:
            buffer_size: Maximum number of diverse frames to maintain.
                        Larger values allow more diversity but use more memory.
            diversity_tags: List of tag keys to use for diversity calculation.
                           If None, auto-detects diversity-suitable tags.
                           Recommended tags: ['dhash', 'clip_embedding', 'color_hist']
            min_distance: Minimum distance threshold for a frame to pass.
                         Frame must be at least this distance from all frames
                         in buffer to be considered diverse enough.
                         Range depends on metric and tag scales.
            metric: Distance metric for backward compatibility ('euclidean', 'manhattan', 'cosine').
                   Used as default for embedding tags when no strategy specified.
            comparison_strategies: Custom comparison strategies per tag (optional).
                                  Maps tag keys to ComparisonStrategy instances.
            enable_weighted_combination: Enable weighted combination of tag distances (optional).
                                        If False (default), uses maximum distance across tags.
            tag_weights: Weights for each tag (used if weighted combination enabled, optional).
                        If not specified, uses equal weights for all tags.

        Raises:
            ValueError: If metric is invalid or min_distance is negative

        Example:
            >>> # Basic usage with auto-detection
            >>> filter = DiversityFilter(buffer_size=100, min_distance=0.1)
            >>>
            >>> # Specify diversity tags explicitly
            >>> filter = DiversityFilter(
            ...     buffer_size=100,
            ...     diversity_tags=['dhash', 'clip_embedding'],
            ...     min_distance=0.1
            ... )
            >>>
            >>> # Custom comparison strategies
            >>> from decimatr.filters.comparison_strategies import (
            ...     HammingDistanceStrategy,
            ...     EmbeddingDistanceStrategy
            ... )
            >>> filter = DiversityFilter(
            ...     buffer_size=100,
            ...     diversity_tags=['dhash', 'clip_embedding'],
            ...     min_distance=0.1,
            ...     comparison_strategies={
            ...         'dhash': HammingDistanceStrategy(),
            ...         'clip_embedding': EmbeddingDistanceStrategy(metric='cosine')
            ...     }
            ... )
            >>>
            >>> # Weighted combination mode
            >>> filter = DiversityFilter(
            ...     buffer_size=100,
            ...     diversity_tags=['dhash', 'clip_embedding'],
            ...     min_distance=0.1,
            ...     enable_weighted_combination=True,
            ...     tag_weights={'dhash': 0.3, 'clip_embedding': 0.7}
            ... )
        """
        super().__init__(buffer_size)

        # Validate metric parameter
        valid_metrics = ["euclidean", "manhattan", "cosine"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

        # Validate min_distance is non-negative
        if min_distance < 0:
            raise ValueError(f"min_distance must be non-negative, got {min_distance}")

        # Validate comparison strategies if provided
        if comparison_strategies:
            for tag_key, strategy in comparison_strategies.items():
                if not isinstance(strategy, ComparisonStrategy):
                    raise ValueError(
                        f"comparison_strategies['{tag_key}'] must be a ComparisonStrategy instance, "
                        f"got {type(strategy).__name__}"
                    )

        self.diversity_tags = diversity_tags
        self.min_distance = min_distance
        self.legacy_metric = metric
        self.enable_weighted_combination = enable_weighted_combination
        self.tag_weights = tag_weights or {}

        # Initialize comparison strategies
        self.comparison_strategies = comparison_strategies or {}
        self._initialize_default_strategies()

        # Track warnings to avoid spamming logs
        self._warned_no_tags = False

    def _initialize_default_strategies(self):
        """Initialize default comparison strategies for known tag types.

        Sets up default strategies for common tag types if not already configured:
        - Hash tags (dhash, phash, ahash): HammingDistanceStrategy
        - Embedding tags (clip_embedding, model_embedding): EmbeddingDistanceStrategy
        - Histogram tags (color_hist): HistogramDistanceStrategy

        Uses the legacy_metric parameter for embedding strategies to maintain
        backward compatibility.
        """
        # Hash tags default to Hamming distance
        for hash_tag in ["dhash", "phash", "ahash"]:
            if hash_tag not in self.comparison_strategies:
                self.comparison_strategies[hash_tag] = HammingDistanceStrategy()

        # Embedding tags default to cosine distance (or legacy metric)
        for emb_tag in ["clip_embedding", "model_embedding"]:
            if emb_tag not in self.comparison_strategies:
                self.comparison_strategies[emb_tag] = EmbeddingDistanceStrategy(
                    metric=self.legacy_metric
                )

        # Histogram tags default to intersection
        for hist_tag in ["color_hist", "color_histogram"]:
            if hist_tag not in self.comparison_strategies:
                self.comparison_strategies[hist_tag] = HistogramDistanceStrategy(
                    metric="intersection"
                )

    def _get_diversity_tags(self, packet: VideoFramePacket) -> list[str]:
        """
        Determine which tags to use for diversity calculation.

        If diversity_tags is explicitly configured, uses those tags.
        Otherwise, auto-detects diversity-suitable tags from the frame,
        filtering out metric-only tags.

        Args:
            packet: Frame packet with tags

        Returns:
            List of tag keys to use for diversity calculation

        Example:
            >>> # With explicit configuration
            >>> filter = DiversityFilter(diversity_tags=['dhash', 'clip_embedding'])
            >>> tags = filter._get_diversity_tags(packet)
            >>> # Returns: ['dhash', 'clip_embedding']
            >>>
            >>> # With auto-detection
            >>> filter = DiversityFilter()
            >>> # packet has tags: {'dhash': ..., 'blur_score': ..., 'clip_embedding': ...}
            >>> tags = filter._get_diversity_tags(packet)
            >>> # Returns: ['dhash', 'clip_embedding'] (blur_score filtered out)
        """
        if self.diversity_tags:
            # Use explicitly configured tags
            return self.diversity_tags

        # Auto-detect diversity-suitable tags
        diversity_tags = []
        for key in packet.tags.keys():
            if TagClassificationRegistry.is_diversity_suitable(key):
                diversity_tags.append(key)

        return diversity_tags

    def _compute_max_distance(
        self,
        current_packet: VideoFramePacket,
        history_packet: VideoFramePacket,
        tag_keys: list[str],
    ) -> float:
        """
        Compute maximum distance across all tags (default behavior).

        This method computes the distance for each tag independently using the
        appropriate comparison strategy, then returns the maximum distance across
        all tags. This is the default behavior when weighted combination is disabled.

        Missing tags are handled gracefully by skipping them. Distance computation
        failures are logged as warnings and the tag is skipped.

        Args:
            current_packet: Current frame packet
            history_packet: Frame from history buffer
            tag_keys: Tags to use for distance calculation

        Returns:
            Maximum distance across all tags (0.0 if no valid distances computed)

        Example:
            >>> # Frame with dhash and clip_embedding
            >>> current = VideoFramePacket(...)
            >>> history = VideoFramePacket(...)
            >>> distance = filter._compute_max_distance(
            ...     current, history, ['dhash', 'clip_embedding']
            ... )
            >>> # Returns max(dhash_distance, clip_embedding_distance)
        """
        max_distance = 0.0

        for tag_key in tag_keys:
            current_value = current_packet.get_tag(tag_key)
            history_value = history_packet.get_tag(tag_key)

            # Skip if either value is missing
            if current_value is None or history_value is None:
                continue

            # Get comparison strategy for this tag
            strategy = self.comparison_strategies.get(tag_key)
            if strategy is None:
                # Use default embedding strategy for unknown tags
                strategy = EmbeddingDistanceStrategy(metric=self.legacy_metric)

            try:
                distance = strategy.compute_distance(current_value, history_value)
                max_distance = max(max_distance, distance)
            except Exception as e:
                logger.warning(
                    f"Failed to compute distance for tag '{tag_key}' using strategy '{strategy.name}': {e}. "
                    f"Tag will be skipped for this comparison. "
                    f"Check that tag values are compatible with the comparison strategy."
                )
                continue

        return max_distance

    def _compute_weighted_distance(
        self,
        current_packet: VideoFramePacket,
        history_packet: VideoFramePacket,
        tag_keys: list[str],
    ) -> float:
        """
        Compute weighted combination of tag distances.

        This method computes the distance for each tag independently, normalizes
        all distances to the [0, 1] range, applies tag-specific weights (defaulting
        to equal weights if not specified), and returns the weighted average.

        This is used when enable_weighted_combination is True. Missing tags are
        handled gracefully by skipping them. Distance computation failures are
        logged as warnings and the tag is skipped.

        Args:
            current_packet: Current frame packet
            history_packet: Frame from history buffer
            tag_keys: Tags to use for distance calculation

        Returns:
            Weighted combined distance in range [0, 1] (0.0 if no valid distances)

        Example:
            >>> # Frame with dhash (weight 0.3) and clip_embedding (weight 0.7)
            >>> filter = DiversityFilter(
            ...     enable_weighted_combination=True,
            ...     tag_weights={'dhash': 0.3, 'clip_embedding': 0.7}
            ... )
            >>> distance = filter._compute_weighted_distance(
            ...     current, history, ['dhash', 'clip_embedding']
            ... )
            >>> # Returns 0.3 * normalized_dhash_dist + 0.7 * normalized_clip_dist
        """
        distances = []
        weights = []

        for tag_key in tag_keys:
            current_value = current_packet.get_tag(tag_key)
            history_value = history_packet.get_tag(tag_key)

            # Skip if either value is missing
            if current_value is None or history_value is None:
                continue

            # Get comparison strategy for this tag
            strategy = self.comparison_strategies.get(tag_key)
            if strategy is None:
                # Use default embedding strategy for unknown tags
                strategy = EmbeddingDistanceStrategy(metric=self.legacy_metric)

            try:
                distance = strategy.compute_distance(current_value, history_value)
                distances.append(distance)

                # Get weight for this tag (default to 1.0 if not specified)
                weight = self.tag_weights.get(tag_key, 1.0)
                weights.append(weight)
            except Exception as e:
                logger.warning(
                    f"Failed to compute distance for tag '{tag_key}' using strategy '{strategy.name}': {e}. "
                    f"Tag will be skipped for this comparison. "
                    f"Check that tag values are compatible with the comparison strategy."
                )
                continue

        # Return 0.0 if no valid distances were computed
        if not distances:
            return 0.0

        # Convert to numpy arrays for easier manipulation
        distances = np.array(distances)
        weights = np.array(weights)

        # Normalize distances to [0, 1] range before combining
        # This ensures different tag types are on comparable scales
        max_distance = distances.max()
        if max_distance > 0:
            distances = distances / max_distance

        # Normalize weights to sum to 1.0
        weights = weights / weights.sum()

        # Compute weighted average
        weighted_distance = np.dot(distances, weights)

        return float(weighted_distance)

    def _compute_frame_distance(
        self,
        current_packet: VideoFramePacket,
        history_packet: VideoFramePacket,
        tag_keys: list[str],
    ) -> float:
        """
        Compute distance between current frame and a history frame.

        This method routes to either max distance or weighted distance computation
        based on the enable_weighted_combination configuration. It selects the
        appropriate comparison strategy for each tag and falls back to a default
        strategy for unknown tags.

        Args:
            current_packet: Current frame packet
            history_packet: Frame from history buffer
            tag_keys: Tags to use for distance calculation

        Returns:
            Distance value (max distance or weighted combined distance)

        Example:
            >>> # Default mode (max distance)
            >>> filter = DiversityFilter(diversity_tags=['dhash', 'clip_embedding'])
            >>> distance = filter._compute_frame_distance(current, history, ['dhash', 'clip_embedding'])
            >>> # Returns max(dhash_distance, clip_embedding_distance)
            >>>
            >>> # Weighted mode
            >>> filter = DiversityFilter(
            ...     diversity_tags=['dhash', 'clip_embedding'],
            ...     enable_weighted_combination=True,
            ...     tag_weights={'dhash': 0.3, 'clip_embedding': 0.7}
            ... )
            >>> distance = filter._compute_frame_distance(current, history, ['dhash', 'clip_embedding'])
            >>> # Returns weighted average of normalized distances
        """
        if self.enable_weighted_combination:
            return self._compute_weighted_distance(current_packet, history_packet, tag_keys)
        else:
            return self._compute_max_distance(current_packet, history_packet, tag_keys)

    def compare_with_history(
        self, packet: VideoFramePacket, history: list[VideoFramePacket]
    ) -> bool:
        """
        Compare current frame against historical frames for diversity.

        This method implements the enhanced diversity comparison logic that:
        1. Gets diversity tags for the current frame (explicit or auto-detected)
        2. Validates that specified tags are present in the frame
        3. Computes minimum distance to all history frames using appropriate strategies
        4. Applies threshold to determine pass/fail

        The method uses tag classification to distinguish between diversity-suitable
        tags (hashes, embeddings, histograms) and metric-only tags (blur_score, entropy).
        It applies tag-specific comparison strategies (Hamming for hashes, cosine/euclidean
        for embeddings, histogram metrics for color data).

        Args:
            packet: Current frame to evaluate
            history: List of recent frames from the temporal buffer

        Returns:
            True if frame is diverse enough (passes), False otherwise

        Example:
            >>> # Frame passes if sufficiently different from all history frames
            >>> filter = DiversityFilter(diversity_tags=['dhash'], min_distance=0.1)
            >>> passes = filter.compare_with_history(current_frame, history_frames)
            >>> if passes:
            ...     # Frame adds diversity, process it
            ...     process_frame(current_frame)
        """
        # If buffer is empty, always pass (first frame)
        if not history:
            return True

        # Determine which tags to use for diversity calculation
        tag_keys = self._get_diversity_tags(packet)

        # Handle case where no diversity tags are available
        if not tag_keys:
            # Log warning once to avoid spamming logs
            if not self._warned_no_tags:
                available_tags = list(packet.tags.keys()) if packet.tags else []
                logger.warning(
                    "No diversity-suitable tags found in frame. "
                    "DiversityFilter will pass all frames without filtering. "
                    f"Available tags: {', '.join(available_tags) if available_tags else 'none'}. "
                    f"Consider adding diversity-suitable taggers (HashTagger, CLIPTagger) to your pipeline, "
                    f"or explicitly configure diversity_tags parameter."
                )
                self._warned_no_tags = True
            return True

        # Validate that specified tags are present in the frame
        missing_tags = [key for key in tag_keys if packet.get_tag(key) is None]
        if missing_tags:
            available_tags = list(packet.tags.keys()) if packet.tags else []
            logger.warning(
                f"Frame {packet.frame_number} missing required diversity tags: "
                f"{', '.join(missing_tags)}. Frame will be skipped. "
                f"Available tags in frame: {', '.join(available_tags) if available_tags else 'none'}. "
                f"Ensure the required taggers are in your pipeline before this filter."
            )
            return False

        # Compute minimum distance to any frame in history
        min_distance = float("inf")

        for past_packet in history:
            distance = self._compute_frame_distance(packet, past_packet, tag_keys)
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
