"""
Blur-based filter for removing blurry frames.

This module provides a BlurFilter that filters out frames with blur scores
below a specified threshold, helping to ensure only sharp, clear frames
are selected for further processing.
"""

from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket


class BlurFilter(StatelessFilter):
    """
    Filter frames based on blur score threshold.

    This filter removes frames that are too blurry by evaluating the 'blur_score'
    tag computed by BlurTagger. Frames with blur scores below the threshold are
    filtered out, while frames with scores at or above the threshold pass through.

    Higher blur scores indicate sharper images. The default threshold of 100.0
    is a reasonable starting point for most use cases, but may need adjustment
    based on video quality and requirements.

    Attributes:
        threshold: Minimum blur score for frames to pass (default: 100.0)

    Example:
        >>> # Filter out frames with blur_score < 100.0
        >>> filter = BlurFilter(threshold=100.0)
        >>>
        >>> # Sharp frame passes
        >>> packet.tags = {'blur_score': 150.0}
        >>> filter.should_pass(packet)
        True
        >>>
        >>> # Blurry frame is filtered out
        >>> packet.tags = {'blur_score': 50.0}
        >>> filter.should_pass(packet)
        False

    Usage in pipeline:
        >>> from decimatr.taggers.blur import BlurTagger
        >>> from decimatr.filters.blur import BlurFilter
        >>>
        >>> pipeline = [
        ...     BlurTagger(),           # Compute blur scores
        ...     BlurFilter(threshold=100.0)  # Filter blurry frames
        ... ]
    """

    def __init__(self, threshold: float = 100.0):
        """
        Initialize blur filter.

        Args:
            threshold: Minimum blur score for frames to pass. Frames with
                      blur_score >= threshold will pass through. Higher values
                      are more restrictive (require sharper images).
                      Default is 100.0.

        Raises:
            ValueError: If threshold is negative
        """
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        self.threshold = threshold

    def should_pass(self, packet: VideoFramePacket) -> bool:
        """
        Determine if frame passes the blur filter.

        Evaluates the 'blur_score' tag against the threshold. Frames with
        blur scores at or above the threshold pass through. If the blur_score
        tag is missing, the frame is filtered out.

        Args:
            packet: VideoFramePacket containing frame data and tags

        Returns:
            True if blur_score >= threshold, False otherwise or if tag is missing
        """
        blur_score = packet.get_tag("blur_score")

        # Filter out frames with missing blur_score tag
        if blur_score is None:
            return False

        # Pass frames with blur score at or above threshold
        return blur_score >= self.threshold

    @property
    def required_tags(self) -> list[str]:
        """
        Return list of required tag keys.

        Returns:
            List containing 'blur_score'
        """
        return ["blur_score"]

    def __repr__(self) -> str:
        """String representation of the filter."""
        return f"BlurFilter(threshold={self.threshold})"
