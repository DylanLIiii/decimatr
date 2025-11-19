"""
FilterActor for distributed stateless filter execution using xoscar.

This module provides an actor-based wrapper around StatelessFilter instances,
enabling parallel execution of filtering operations across multiple CPU cores.
"""

import xoscar as xo

from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket


class FilterActor(xo.Actor):
    """
    Actor that executes stateless filter operations in a distributed manner.

    FilterActor wraps a StatelessFilter instance and provides async processing
    capabilities through the xoscar actor framework. This enables parallel
    execution of filtering operations across multiple CPU cores for improved
    throughput.

    The actor receives frame packets, applies the filter to determine if the
    frame should pass, and returns the packet if it passes or None if it's
    filtered out.

    Inherits from xo.Actor to enable distributed execution through xoscar.

    Attributes:
        filter: The StatelessFilter instance to execute

    Example:
        >>> # Create actor pool for parallel filtering
        >>> await xo.create_actor_pool(address='127.0.0.1:13527', n_process=4)
        >>>
        >>> # Create filter actor
        >>> blur_filter = BlurFilter(threshold=100.0)
        >>> actor_ref = await xo.create_actor(
        ...     FilterActor,
        ...     blur_filter,
        ...     address='127.0.0.1:13527'
        ... )
        >>>
        >>> # Process frame through actor
        >>> result = await actor_ref.process_frame(packet)
        >>> if result is not None:
        ...     print("Frame passed filter")
    """

    def __init__(self, filter: StatelessFilter):
        """
        Initialize FilterActor with a stateless filter instance.

        Args:
            filter: StatelessFilter instance to execute for frame filtering

        Raises:
            TypeError: If filter is not a StatelessFilter instance
        """
        if not isinstance(filter, StatelessFilter):
            raise TypeError(f"filter must be a StatelessFilter instance, got {type(filter)}")

        self.filter = filter

    async def process_frame(self, packet: VideoFramePacket) -> VideoFramePacket | None:
        """
        Process a frame packet by applying the filter.

        This method applies the stateless filter to determine if the frame
        should pass through the pipeline. If the frame passes, the packet
        is returned unchanged. If the frame is filtered out, None is returned.

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
            >>> packet = VideoFramePacket(frame_data=frame, ...)
            >>> packet.tags = {"blur_score": 150.0}
            >>> result = await actor_ref.process_frame(packet)
            >>> if result is not None:
            ...     print(f"Frame {result.frame_number} passed")
        """
        if not isinstance(packet, VideoFramePacket):
            raise TypeError(f"packet must be a VideoFramePacket, got {type(packet)}")

        # Apply filter to determine if frame should pass
        if self.filter.should_pass(packet):
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
