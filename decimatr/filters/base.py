"""
Base classes and interfaces for all filters.

Filters are components that make pass/fail decisions on frames based on their
tags and optionally temporal context. They come in two flavors:

1. StatelessFilter: Makes decisions based only on the current frame's tags
2. StatefulFilter: Maintains a temporal buffer to make decisions based on
                   historical context
"""

from abc import ABC, abstractmethod
from collections import deque

from decimatr.scheme import VideoFramePacket


class Filter(ABC):
    """
    Abstract base class for all filters.

    Filters determine whether a frame should pass through the processing pipeline
    based on its tags. Filters can be stateless (immediate decisions based on
    current frame) or stateful (decisions based on temporal context).

    Subclasses must implement:
        - should_pass(): Determine if frame should pass through
        - required_tags: Property listing the tag keys needed for filtering
    """

    @abstractmethod
    def should_pass(self, packet: VideoFramePacket) -> bool:
        """
        Determine if frame should pass through the filter.

        This method evaluates the frame's tags and returns True if the frame
        should continue through the pipeline, or False if it should be filtered out.

        Args:
            packet: VideoFramePacket containing frame data, metadata, and tags

        Returns:
            True if frame passes the filter, False if it should be filtered out

        Example:
            >>> filter = BlurFilter(threshold=100.0)
            >>> packet.tags = {"blur_score": 150.0}
            >>> filter.should_pass(packet)
            True
        """
        pass

    @property
    @abstractmethod
    def required_tags(self) -> list[str]:
        """
        Return list of tag keys required by this filter.

        This property allows the processing pipeline to validate that all
        required tags will be computed by upstream taggers before this filter
        is applied.

        Returns:
            List of required tag key strings

        Example:
            >>> filter = BlurFilter(threshold=100.0)
            >>> filter.required_tags
            ['blur_score']
        """
        pass


class StatelessFilter(Filter):
    """
    Base class for filters that make decisions based only on current frame tags.

    Stateless filters evaluate each frame independently without considering
    previous frames. They are ideal for threshold-based filtering and other
    immediate decisions.

    Examples:
        - BlurFilter: Filter frames below a blur threshold
        - EntropyFilter: Filter frames below an entropy threshold
        - ThresholdFilter: Generic threshold-based filtering

    Subclasses must implement:
        - should_pass(): Evaluate current frame's tags
        - required_tags: List required tag keys
    """

    pass


class StatefulFilter(Filter):
    """
    Base class for filters that maintain temporal context for decision-making.

    Stateful filters maintain a buffer of recent frames and make decisions by
    comparing the current frame against historical frames. This enables temporal
    pattern detection like duplicate detection, scene change detection, and
    diversity sampling.

    The base class provides buffer management utilities. Subclasses implement
    the comparison logic.

    Examples:
        - DuplicateFilter: Detect duplicate frames via hash comparison
        - MotionFilter: Detect scene changes via frame differencing
        - DiversityFilter: Select frames maximizing tag diversity

    Subclasses must implement:
        - compare_with_history(): Compare current frame with buffer
        - required_tags: List required tag keys

    Attributes:
        buffer_size: Maximum number of frames to keep in temporal buffer
        buffer: Deque containing recent frames that passed the filter
    """

    def __init__(self, buffer_size: int):
        """
        Initialize stateful filter with temporal buffer.

        Args:
            buffer_size: Maximum number of frames to maintain in buffer.
                        Older frames are automatically evicted when capacity
                        is reached.

        Raises:
            ValueError: If buffer_size is not positive
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")

        self.buffer_size = buffer_size
        self.buffer: deque[VideoFramePacket] = deque(maxlen=buffer_size)

    @abstractmethod
    def compare_with_history(
        self, packet: VideoFramePacket, history: list[VideoFramePacket]
    ) -> bool:
        """
        Compare current frame with historical frames to make pass/fail decision.

        This method receives the current frame and a list of recent frames from
        the temporal buffer. It should return True if the frame should pass
        (and be added to the buffer), or False if it should be filtered out.

        Args:
            packet: Current frame to evaluate
            history: List of recent frames from the temporal buffer (may be empty)

        Returns:
            True if frame should pass and be added to buffer, False otherwise

        Example:
            >>> def compare_with_history(self, packet, history):
            ...     # Duplicate detection: check if current frame is similar to any in history
            ...     current_hash = packet.get_tag("phash")
            ...     for past_frame in history:
            ...         past_hash = past_frame.get_tag("phash")
            ...         if hash_similarity(current_hash, past_hash) > self.threshold:
            ...             return False  # Duplicate found, filter out
            ...     return True  # No duplicate, pass through
        """
        pass

    def should_pass(self, packet: VideoFramePacket) -> bool:
        """
        Determine if frame should pass by comparing with history.

        This method implements the Filter interface by calling compare_with_history()
        with the current buffer contents. If the frame passes, it is automatically
        added to the buffer for future comparisons.

        Args:
            packet: VideoFramePacket to evaluate

        Returns:
            True if frame passes, False if filtered out
        """
        # Get current buffer contents as a list
        history = list(self.buffer)

        # Compare current frame with history
        passes = self.compare_with_history(packet, history)

        # If frame passes, add it to the buffer for future comparisons
        if passes:
            self.add_to_buffer(packet)

        return passes

    def add_to_buffer(self, packet: VideoFramePacket) -> None:
        """
        Add frame to the temporal buffer.

        Frames are added to the right side of the deque. When the buffer reaches
        capacity, the oldest frame (leftmost) is automatically evicted.

        Args:
            packet: VideoFramePacket to add to buffer
        """
        self.buffer.append(packet)

    def get_buffer_contents(self) -> list[VideoFramePacket]:
        """
        Get current buffer contents as a list.

        Returns:
            List of VideoFramePacket objects currently in buffer, ordered from
            oldest (index 0) to newest (index -1)
        """
        return list(self.buffer)

    def clear_buffer(self) -> None:
        """
        Clear all frames from the temporal buffer.

        This can be useful when processing multiple videos or when resetting
        the filter state.
        """
        self.buffer.clear()

    def buffer_count(self) -> int:
        """
        Get the current number of frames in the buffer.

        Returns:
            Number of frames currently stored in buffer (0 to buffer_size)
        """
        return len(self.buffer)

    def is_buffer_full(self) -> bool:
        """
        Check if the buffer has reached capacity.

        Returns:
            True if buffer contains buffer_size frames, False otherwise
        """
        return len(self.buffer) >= self.buffer_size
