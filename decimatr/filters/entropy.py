"""
Entropy-based filter for removing low-information frames.

This module provides an EntropyFilter that filters out frames with low entropy
(low information content), helping to ensure only frames with sufficient visual
complexity are selected for further processing.
"""

from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket


class EntropyFilter(StatelessFilter):
    """
    Filter frames based on entropy threshold.

    This filter removes frames with low entropy (low information content) by
    evaluating the 'entropy' tag computed by EntropyTagger. Frames with entropy
    values below the threshold are filtered out, while frames with entropy at
    or above the threshold pass through.

    Higher entropy indicates more visual complexity and information content.
    Low entropy frames are often uniform, blank, or contain little useful
    information. The default threshold of 4.0 is a reasonable starting point
    for most use cases.

    Attributes:
        threshold: Minimum entropy value for frames to pass (default: 4.0)

    Example:
        >>> # Filter out frames with entropy < 4.0
        >>> filter = EntropyFilter(threshold=4.0)
        >>>
        >>> # High-entropy frame passes
        >>> packet.tags = {'entropy': 5.2}
        >>> filter.should_pass(packet)
        True
        >>>
        >>> # Low-entropy frame is filtered out
        >>> packet.tags = {'entropy': 2.1}
        >>> filter.should_pass(packet)
        False

    Usage in pipeline:
        >>> from decimatr.taggers.entropy import EntropyTagger
        >>> from decimatr.filters.entropy import EntropyFilter
        >>>
        >>> pipeline = [
        ...     EntropyTagger(),              # Compute entropy values
        ...     EntropyFilter(threshold=4.0)  # Filter low-entropy frames
        ... ]
    """

    def __init__(self, threshold: float = 4.0):
        """
        Initialize entropy filter.

        Args:
            threshold: Minimum entropy value for frames to pass. Frames with
                      entropy >= threshold will pass through. Higher values
                      are more restrictive (require more visual complexity).
                      Default is 4.0.

        Raises:
            ValueError: If threshold is negative
        """
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        self.threshold = threshold

    def should_pass(self, packet: VideoFramePacket) -> bool:
        """
        Determine if frame passes the entropy filter.

        Evaluates the 'entropy' tag against the threshold. Frames with
        entropy values at or above the threshold pass through. If the entropy
        tag is missing, the frame is filtered out.

        Args:
            packet: VideoFramePacket containing frame data and tags

        Returns:
            True if entropy >= threshold, False otherwise or if tag is missing
        """
        entropy = packet.get_tag("entropy")

        # Filter out frames with missing entropy tag
        if entropy is None:
            return False

        # Pass frames with entropy at or above threshold
        return entropy >= self.threshold

    @property
    def required_tags(self) -> list[str]:
        """
        Return list of required tag keys.

        Returns:
            List containing 'entropy'
        """
        return ["entropy"]

    def __repr__(self) -> str:
        """String representation of the filter."""
        return f"EntropyFilter(threshold={self.threshold})"
