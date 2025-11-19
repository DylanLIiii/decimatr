"""
TaggerActor for distributed tagger execution using xoscar.

This module provides an actor-based wrapper around Tagger instances,
enabling parallel execution of tagging operations across multiple CPU cores.
"""

import xoscar as xo

from decimatr.scheme import VideoFramePacket
from decimatr.taggers.base import Tagger


class TaggerActor(xo.Actor):
    """
    Actor that executes tagger operations in a distributed manner.

    TaggerActor wraps a Tagger instance and provides async processing
    capabilities through the xoscar actor framework. This enables parallel
    execution of tagging operations across multiple CPU cores for improved
    throughput.

    The actor receives frame packets, applies the tagger to compute tags,
    updates the packet's tag registry, and returns the enriched packet.

    Inherits from xo.Actor to enable distributed execution through xoscar.

    Attributes:
        tagger: The Tagger instance to execute

    Example:
        >>> # Create actor pool for parallel tagging
        >>> await xo.create_actor_pool(address='127.0.0.1:13527', n_process=4)
        >>>
        >>> # Create tagger actor
        >>> blur_tagger = BlurTagger()
        >>> actor_ref = await xo.create_actor(
        ...     TaggerActor,
        ...     blur_tagger,
        ...     address='127.0.0.1:13527'
        ... )
        >>>
        >>> # Process frame through actor
        >>> enriched_packet = await actor_ref.process_frame(packet)
    """

    def __init__(self, tagger: Tagger):
        """
        Initialize TaggerActor with a tagger instance.

        Args:
            tagger: Tagger instance to execute for frame processing

        Raises:
            TypeError: If tagger is not a Tagger instance
        """
        if not isinstance(tagger, Tagger):
            raise TypeError(f"tagger must be a Tagger instance, got {type(tagger)}")

        self.tagger = tagger

    async def process_frame(self, packet: VideoFramePacket) -> VideoFramePacket:
        """
        Process a frame packet by computing and adding tags.

        This method applies the tagger to compute tags for the frame,
        updates the packet's tag registry with the computed tags, and
        returns the enriched packet.

        The operation is async to integrate with xoscar's actor model,
        but the actual tagging computation is synchronous (CPU-bound).

        Args:
            packet: VideoFramePacket to process

        Returns:
            The same VideoFramePacket with tags added to its tag registry

        Raises:
            TypeError: If packet is not a VideoFramePacket
            Exception: If tagger computation fails (propagated from tagger)

        Example:
            >>> packet = VideoFramePacket(frame_data=frame, ...)
            >>> enriched = await actor_ref.process_frame(packet)
            >>> print(enriched.tags)  # {'blur_score': 123.45}
        """
        if not isinstance(packet, VideoFramePacket):
            raise TypeError(f"packet must be a VideoFramePacket, got {type(packet)}")

        # Compute tags using the tagger
        tags = self.tagger.compute_tags(packet)

        # Update packet's tag registry
        packet.tags.update(tags)

        return packet

    def get_tag_keys(self) -> list:
        """
        Get the list of tag keys this actor's tagger produces.

        This method allows pipeline configuration to validate that
        required tags will be available.

        Returns:
            List of tag key strings produced by the tagger
        """
        return self.tagger.tag_keys

    def supports_gpu(self) -> bool:
        """
        Check if this actor's tagger supports GPU acceleration.

        Returns:
            True if GPU acceleration is supported, False otherwise
        """
        return self.tagger.supports_gpu

    def requires_gpu(self) -> bool:
        """
        Check if this actor's tagger requires GPU to function.

        Returns:
            True if GPU is required, False otherwise
        """
        return self.tagger.requires_gpu
