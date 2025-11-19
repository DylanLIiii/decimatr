"""
Blur removal strategy for filtering out blurry frames.

This strategy provides a simple, predefined pipeline that computes blur scores
and filters out frames below a configurable threshold. It's useful for ensuring
only sharp, clear frames are selected from video streams.
"""

from decimatr.filters.base import Filter
from decimatr.filters.blur import BlurFilter
from decimatr.strategies.base import FilterStrategy
from decimatr.taggers.base import Tagger
from decimatr.taggers.blur import BlurTagger


class BlurRemovalStrategy(FilterStrategy):
    """
    Strategy for removing blurry frames from video streams.

    This strategy creates a simple two-stage pipeline:
    1. BlurTagger: Computes blur scores (variance of Laplacian)
    2. BlurFilter: Filters out frames below the blur threshold

    The blur score measures image sharpness - higher values indicate sharper
    images. The default threshold of 100.0 works well for most videos, but
    may need adjustment based on video quality and requirements.

    Attributes:
        threshold: Minimum blur score for frames to pass (default: 100.0)

    Example:
        >>> # Use default threshold (100.0)
        >>> strategy = BlurRemovalStrategy()
        >>> pipeline = strategy.build_pipeline()
        >>>
        >>> # Use custom threshold for stricter filtering
        >>> strategy = BlurRemovalStrategy(threshold=150.0)
        >>> pipeline = strategy.build_pipeline()
        >>>
        >>> # Use with FrameProcessor
        >>> from decimatr.core.processor import FrameProcessor
        >>> processor = FrameProcessor(strategy=strategy)
        >>> for frame in processor.process('video.mp4'):
        ...     # Process only sharp frames
        ...     process_frame(frame)

    Requirements:
        - 4.1: Provides predefined FilterStrategy implementation
        - 4.2: Implements blur removal pattern with configurable threshold
    """

    def __init__(self, threshold: float = 100.0):
        """
        Initialize blur removal strategy.

        Args:
            threshold: Minimum blur score for frames to pass. Frames with
                      blur_score >= threshold will pass through. Higher values
                      are more restrictive (require sharper images).
                      Default is 100.0.

        Raises:
            ValueError: If threshold is negative

        Example:
            >>> # Default threshold
            >>> strategy = BlurRemovalStrategy()
            >>>
            >>> # Stricter filtering (only very sharp frames)
            >>> strategy = BlurRemovalStrategy(threshold=200.0)
            >>>
            >>> # More lenient filtering
            >>> strategy = BlurRemovalStrategy(threshold=50.0)
        """
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        self.threshold = threshold

    def build_pipeline(self) -> list[Tagger | Filter]:
        """
        Build the blur removal processing pipeline.

        Creates a pipeline with:
        1. BlurTagger: Computes 'blur_score' tag for each frame
        2. BlurFilter: Filters frames with blur_score < threshold

        Returns:
            Ordered list containing [BlurTagger(), BlurFilter(threshold)]

        Example:
            >>> strategy = BlurRemovalStrategy(threshold=100.0)
            >>> pipeline = strategy.build_pipeline()
            >>> len(pipeline)
            2
            >>> isinstance(pipeline[0], BlurTagger)
            True
            >>> isinstance(pipeline[1], BlurFilter)
            True
        """
        return [BlurTagger(), BlurFilter(threshold=self.threshold)]

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"BlurRemovalStrategy(threshold={self.threshold})"
