"""
StatefulFilterActor for distributed stateful filter execution using xoscar.

This module provides an actor-based wrapper around StatefulFilter instances,
enabling execution of temporal filtering operations that maintain state across
frames. Unlike stateless filters, stateful filters require a single actor
instance to maintain consistent temporal buffer state.
"""

import xoscar as xo

from decimatr.core.temporal_buffer import TemporalBuffer
from decimatr.filters.base import StatefulFilter
from decimatr.scheme import VideoFramePacket


class StatefulFilterActor(xo.Actor):
    """
    Actor that executes stateful filter operations with temporal context.

    StatefulFilterActor wraps a StatefulFilter instance and maintains a
    TemporalBuffer to track recent frames. This enables temporal pattern
    detection like duplicate detection, scene change detection, and diversity
    sampling.

    Unlike stateless filters which can be parallelized across multiple actors,
    stateful filters require a single actor instance to maintain consistent
    state. The actor processes frames sequentially to ensure proper temporal
    ordering.

    The actor receives frame packets, compares them against the temporal buffer,
    and returns the packet if it passes or None if it's filtered out. Passing
    frames are automatically added to the buffer for future comparisons.

    Inherits from xo.Actor to enable distributed execution through xoscar.

    Attributes:
        filter: The StatefulFilter instance to execute
        buffer: TemporalBuffer maintaining recent frames for comparison

    Example:
        >>> # Create actor pool
        >>> await xo.create_actor_pool(address='127.0.0.1:13527', n_process=1)
        >>>
        >>> # Create stateful filter actor (single instance for state consistency)
        >>> duplicate_filter = DuplicateFilter(threshold=0.05, buffer_size=50)
        >>> actor_ref = await xo.create_actor(
        ...     StatefulFilterActor,
        ...     duplicate_filter,
        ...     address='127.0.0.1:13527'
        ... )
        >>>
        >>> # Process frames through actor
        >>> result = await actor_ref.process_frame(packet)
        >>> if result is not None:
        ...     print("Frame is unique, not a duplicate")
    """

    def __init__(self, filter: StatefulFilter):
        """
        Initialize StatefulFilterActor with a stateful filter instance.

        Creates a TemporalBuffer with the filter's configured buffer size
        to maintain temporal context for frame comparisons.

        Args:
            filter: StatefulFilter instance to execute for frame filtering

        Raises:
            TypeError: If filter is not a StatefulFilter instance
        """
        if not isinstance(filter, StatefulFilter):
            raise TypeError(f"filter must be a StatefulFilter instance, got {type(filter)}")

        self.filter = filter
        self.buffer = TemporalBuffer(filter.buffer_size)

    async def process_frame(self, packet: VideoFramePacket) -> VideoFramePacket | None:
        """
        Process a frame packet by comparing with temporal history.

        This method retrieves the current buffer contents, applies the
        stateful filter's comparison logic to determine if the frame should
        pass, and adds passing frames to the buffer for future comparisons.

        The operation is async to integrate with xoscar's actor model,
        but the actual filtering logic is synchronous (CPU-bound).

        Args:
            packet: VideoFramePacket to filter

        Returns:
            The VideoFramePacket if it passes the filter, None if filtered out

        Raises:
            TypeError: If packet is not a VideoFramePacket
            Exception: If filter evaluation fails (propagated from filter)

        Example:
            >>> # First frame (no history)
            >>> packet1 = VideoFramePacket(frame_data=frame1, ...)
            >>> packet1.tags = {"phash": "abc123"}
            >>> result1 = await actor_ref.process_frame(packet1)
            >>> # result1 is packet1 (passes, added to buffer)
            >>>
            >>> # Second frame (duplicate)
            >>> packet2 = VideoFramePacket(frame_data=frame2, ...)
            >>> packet2.tags = {"phash": "abc123"}  # Same hash
            >>> result2 = await actor_ref.process_frame(packet2)
            >>> # result2 is None (filtered as duplicate)
        """
        if not isinstance(packet, VideoFramePacket):
            raise TypeError(f"packet must be a VideoFramePacket, got {type(packet)}")

        # Get current buffer contents as history
        history = self.buffer.get_window()

        # Compare current frame with history using filter's logic
        passes = self.filter.compare_with_history(packet, history)

        # If frame passes, add it to buffer for future comparisons
        if passes:
            self.buffer.add(packet)
            return packet

        # Frame filtered out
        return None

    def get_required_tags(self) -> list:
        """
        Get the list of tag keys required by this actor's filter.

        This method allows pipeline configuration to validate that
        all required tags will be computed by upstream taggers.

        Returns:
            List of required tag key strings
        """
        return self.filter.required_tags

    def get_buffer_size(self) -> int:
        """
        Get the current number of frames in the temporal buffer.

        Returns:
            Number of frames currently stored in the buffer
        """
        return len(self.buffer)

    def clear_buffer(self) -> None:
        """
        Clear all frames from the temporal buffer.

        This can be useful when processing multiple videos or when
        resetting the filter state between processing sessions.
        """
        self.buffer.clear()

    def is_buffer_full(self) -> bool:
        """
        Check if the temporal buffer is at maximum capacity.

        Returns:
            True if buffer contains buffer_size frames, False otherwise
        """
        return self.buffer.is_full()
