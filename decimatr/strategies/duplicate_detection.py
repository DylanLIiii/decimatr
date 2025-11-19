"""
Duplicate detection strategy for removing duplicate or near-duplicate frames.

This strategy provides a predefined pipeline that computes perceptual hashes
and filters out frames that are too similar to recently seen frames. It's useful
for removing redundant frames from video streams, such as static scenes or
repeated content.
"""

from decimatr.filters.base import Filter
from decimatr.filters.duplicate import DuplicateFilter
from decimatr.strategies.base import FilterStrategy
from decimatr.taggers.base import Tagger
from decimatr.taggers.hash import HashTagger


class DuplicateDetectionStrategy(FilterStrategy):
    """
    Strategy for detecting and removing duplicate frames from video streams.

    This strategy creates a two-stage pipeline:
    1. HashTagger: Computes perceptual hashes (phash by default)
    2. DuplicateFilter: Filters out frames with similar hashes within a temporal window

    The strategy uses perceptual hashing which is robust to minor variations in
    brightness, contrast, and compression artifacts. The temporal buffer allows
    detection of duplicates across a configurable time window.

    Attributes:
        threshold: Hash similarity threshold (0.0-1.0). Lower values are stricter.
                  Default 0.05 means ~3 bits difference allowed for 8x8 hash.
        window_size: Number of recent frames to compare against (default: 50)
        hash_type: Type of perceptual hash to use (default: 'phash')
        hash_size: Size of the hash in bits (default: 8 for 8x8 = 64 bits)

    Example:
        >>> # Use default settings
        >>> strategy = DuplicateDetectionStrategy()
        >>> pipeline = strategy.build_pipeline()
        >>>
        >>> # Stricter duplicate detection with larger window
        >>> strategy = DuplicateDetectionStrategy(
        ...     threshold=0.02,  # Only 1-2 bits difference allowed
        ...     window_size=100  # Compare against last 100 frames
        ... )
        >>>
        >>> # Use with FrameProcessor
        >>> from decimatr.core.processor import FrameProcessor
        >>> processor = FrameProcessor(strategy=strategy)
        >>> for frame in processor.process('video.mp4'):
        ...     # Process only unique frames
        ...     process_frame(frame)

    Requirements:
        - 4.1: Provides predefined FilterStrategy implementation
        - 4.3: Implements duplicate detection pattern with configurable parameters
    """

    def __init__(
        self,
        threshold: float = 0.05,
        window_size: int = 50,
        hash_type: str = "phash",
        hash_size: int = 8,
    ):
        """
        Initialize duplicate detection strategy.

        Args:
            threshold: Hash difference threshold as a fraction of max difference.
                      For 8x8 hash (64 bits), threshold=0.05 means 3.2 bits
                      difference allowed. Lower values are stricter.
                      Range: 0.0 (exact match only) to 1.0 (allow all).
                      Default is 0.05.
            window_size: Maximum number of recent frames to compare against.
                        Larger values detect duplicates over longer time windows
                        but use more memory. Default is 50.
            hash_type: Type of perceptual hash to use:
                      - 'phash': Perceptual hash (default, robust to minor changes)
                      - 'ahash': Average hash (faster, less robust)
                      - 'dhash': Difference hash (good for gradients)
                      - 'whash': Wavelet hash (more complex)
            hash_size: Size of the hash in bits (default 8 for 8x8 = 64 bits).
                      Larger values are more precise but slower.

        Raises:
            ValueError: If threshold is not in range [0.0, 1.0] or window_size <= 0

        Example:
            >>> # Default settings (good for most use cases)
            >>> strategy = DuplicateDetectionStrategy()
            >>>
            >>> # Strict duplicate detection
            >>> strategy = DuplicateDetectionStrategy(threshold=0.01)
            >>>
            >>> # Lenient duplicate detection with large window
            >>> strategy = DuplicateDetectionStrategy(
            ...     threshold=0.1,
            ...     window_size=200
            ... )
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.threshold = threshold
        self.window_size = window_size
        self.hash_type = hash_type
        self.hash_size = hash_size

    def build_pipeline(self) -> list[Tagger | Filter]:
        """
        Build the duplicate detection processing pipeline.

        Creates a pipeline with:
        1. HashTagger: Computes perceptual hash for each frame
        2. DuplicateFilter: Filters frames with similar hashes in temporal window

        Returns:
            Ordered list containing [HashTagger(...), DuplicateFilter(...)]

        Example:
            >>> strategy = DuplicateDetectionStrategy(
            ...     threshold=0.05,
            ...     window_size=50
            ... )
            >>> pipeline = strategy.build_pipeline()
            >>> len(pipeline)
            2
            >>> isinstance(pipeline[0], HashTagger)
            True
            >>> isinstance(pipeline[1], DuplicateFilter)
            True
        """
        return [
            HashTagger(hash_type=self.hash_type, hash_size=self.hash_size),
            DuplicateFilter(
                buffer_size=self.window_size,
                threshold=self.threshold,
                hash_type=self.hash_type,
                hash_size=self.hash_size,
            ),
        ]

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return (
            f"DuplicateDetectionStrategy("
            f"threshold={self.threshold}, "
            f"window_size={self.window_size}, "
            f"hash_type='{self.hash_type}', "
            f"hash_size={self.hash_size})"
        )
