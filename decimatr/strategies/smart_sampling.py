"""
Smart sampling strategy combining multiple quality and diversity filters.

This strategy provides a comprehensive, predefined pipeline that combines blur
removal, duplicate detection, and diversity sampling to select high-quality,
diverse frames from video streams. It's useful for creating representative
frame samples that capture the full range of content in a video.
"""

from decimatr.filters.base import Filter
from decimatr.filters.blur import BlurFilter
from decimatr.filters.diversity import DiversityFilter
from decimatr.filters.duplicate import DuplicateFilter
from decimatr.strategies.base import FilterStrategy
from decimatr.taggers.base import Tagger
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.entropy import EntropyTagger
from decimatr.taggers.hash import HashTagger


class SmartSamplingStrategy(FilterStrategy):
    """
    Comprehensive strategy combining blur removal, duplicate detection, and diversity.

    This strategy creates a multi-stage pipeline that:
    1. Computes multiple quality metrics (blur, hash, entropy)
    2. Filters out blurry frames
    3. Removes duplicate/near-duplicate frames
    4. Selects diverse frames that maximize variation

    The pipeline ensures that selected frames are:
    - Sharp and clear (not blurry)
    - Unique (no duplicates)
    - Diverse (capture full range of content)

    This is ideal for creating high-quality frame samples for tasks like:
    - Video summarization
    - Thumbnail generation
    - Training data collection
    - Content analysis

    Attributes:
        blur_threshold: Minimum blur score for frames to pass (default: 100.0)
        duplicate_threshold: Hash similarity threshold for duplicate detection (default: 0.05)
        duplicate_window: Number of frames to check for duplicates (default: 50)
        diversity_window: Number of diverse frames to maintain (default: 100)
        diversity_min_distance: Minimum distance for diversity (default: 0.1)

    Example:
        >>> # Use default settings
        >>> strategy = SmartSamplingStrategy()
        >>> pipeline = strategy.build_pipeline()
        >>>
        >>> # Custom settings for stricter filtering
        >>> strategy = SmartSamplingStrategy(
        ...     blur_threshold=150.0,      # Only very sharp frames
        ...     duplicate_threshold=0.02,  # Stricter duplicate detection
        ...     diversity_min_distance=0.2 # More diverse frames
        ... )
        >>>
        >>> # Use with FrameProcessor
        >>> from decimatr.core.processor import FrameProcessor
        >>> processor = FrameProcessor(strategy=strategy)
        >>> for frame in processor.process('video.mp4'):
        ...     # Process high-quality, diverse frames
        ...     process_frame(frame)

    Requirements:
        - 4.1: Provides predefined FilterStrategy implementation
        - 4.4: Combines blur removal, duplicate detection, and diversity sampling
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        duplicate_threshold: float = 0.05,
        duplicate_window: int = 50,
        diversity_window: int = 100,
        diversity_min_distance: float = 0.1,
    ):
        """
        Initialize smart sampling strategy.

        Args:
            blur_threshold: Minimum blur score for frames to pass. Higher values
                           are more restrictive (require sharper images).
                           Default is 100.0.
            duplicate_threshold: Hash difference threshold for duplicate detection.
                                Lower values are stricter. Range: 0.0-1.0.
                                Default is 0.05.
            duplicate_window: Number of recent frames to check for duplicates.
                             Larger values detect duplicates over longer windows.
                             Default is 50.
            diversity_window: Maximum number of diverse frames to maintain.
                             Larger values allow more diversity but use more memory.
                             Default is 100.
            diversity_min_distance: Minimum distance threshold for diversity.
                                   Frames must be at least this distance from all
                                   frames in buffer to pass. Default is 0.1.

        Raises:
            ValueError: If any threshold is invalid

        Example:
            >>> # Default settings (balanced quality and diversity)
            >>> strategy = SmartSamplingStrategy()
            >>>
            >>> # High quality, strict filtering
            >>> strategy = SmartSamplingStrategy(
            ...     blur_threshold=200.0,
            ...     duplicate_threshold=0.01,
            ...     diversity_min_distance=0.3
            ... )
            >>>
            >>> # More lenient, capture more frames
            >>> strategy = SmartSamplingStrategy(
            ...     blur_threshold=50.0,
            ...     duplicate_threshold=0.1,
            ...     diversity_min_distance=0.05
            ... )
        """
        if blur_threshold < 0:
            raise ValueError(f"blur_threshold must be non-negative, got {blur_threshold}")

        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError(
                f"duplicate_threshold must be between 0.0 and 1.0, got {duplicate_threshold}"
            )

        if duplicate_window <= 0:
            raise ValueError(f"duplicate_window must be positive, got {duplicate_window}")

        if diversity_window <= 0:
            raise ValueError(f"diversity_window must be positive, got {diversity_window}")

        if diversity_min_distance < 0:
            raise ValueError(
                f"diversity_min_distance must be non-negative, got {diversity_min_distance}"
            )

        self.blur_threshold = blur_threshold
        self.duplicate_threshold = duplicate_threshold
        self.duplicate_window = duplicate_window
        self.diversity_window = diversity_window
        self.diversity_min_distance = diversity_min_distance

    def build_pipeline(self) -> list[Tagger | Filter]:
        """
        Build the smart sampling processing pipeline.

        Creates a comprehensive pipeline with:
        1. BlurTagger: Computes blur scores
        2. HashTagger: Computes perceptual hashes
        3. EntropyTagger: Computes entropy scores
        4. BlurFilter: Filters out blurry frames
        5. DuplicateFilter: Removes duplicate frames
        6. DiversityFilter: Selects diverse frames

        The order is important:
        - All taggers run first to compute metrics
        - Filters run in sequence, each narrowing the selection
        - Blur filter eliminates low-quality frames early
        - Duplicate filter removes redundant frames
        - Diversity filter ensures final selection is diverse

        Returns:
            Ordered list of taggers and filters forming the complete pipeline

        Example:
            >>> strategy = SmartSamplingStrategy()
            >>> pipeline = strategy.build_pipeline()
            >>> len(pipeline)
            6
            >>> # First 3 are taggers
            >>> all(isinstance(p, Tagger) for p in pipeline[:3])
            True
            >>> # Last 3 are filters
            >>> all(isinstance(p, Filter) for p in pipeline[3:])
            True
        """
        return [
            # Stage 1: Compute all metrics (taggers)
            BlurTagger(),
            HashTagger(hash_type="phash", hash_size=8),
            EntropyTagger(),
            # Stage 2: Apply filters in sequence
            BlurFilter(threshold=self.blur_threshold),
            DuplicateFilter(
                buffer_size=self.duplicate_window,
                threshold=self.duplicate_threshold,
                hash_type="phash",
                hash_size=8,
            ),
            DiversityFilter(
                buffer_size=self.diversity_window,
                diversity_tags=["blur_score", "entropy"],
                min_distance=self.diversity_min_distance,
                metric="euclidean",
            ),
        ]

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return (
            f"SmartSamplingStrategy("
            f"blur_threshold={self.blur_threshold}, "
            f"duplicate_threshold={self.duplicate_threshold}, "
            f"duplicate_window={self.duplicate_window}, "
            f"diversity_window={self.diversity_window}, "
            f"diversity_min_distance={self.diversity_min_distance})"
        )
