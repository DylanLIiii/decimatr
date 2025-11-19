"""Temporal buffer for maintaining sliding window of recent frames."""

from collections import deque
from collections.abc import Callable

from decimatr.scheme import VideoFramePacket


class TemporalBuffer:
    """
    Efficient sliding window buffer for stateful frame filtering.

    Maintains a fixed-size buffer of recent frames with O(1) insertion
    and automatic eviction of oldest frames when capacity is reached.

    This buffer is used by StatefulFilter implementations to maintain
    temporal context for comparison-based filtering (e.g., duplicate
    detection, motion analysis, diversity sampling).

    Attributes:
        max_size: Maximum number of frames to store in the buffer
        frames: Deque containing the buffered frames
    """

    def __init__(self, max_size: int):
        """
        Initialize temporal buffer with fixed capacity.

        Args:
            max_size: Maximum number of frames to store. When this limit
                     is reached, oldest frames are automatically evicted.

        Raises:
            ValueError: If max_size is less than 1
        """
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")

        self.max_size = max_size
        self.frames: deque[VideoFramePacket] = deque(maxlen=max_size)

    def add(self, packet: VideoFramePacket) -> None:
        """
        Add a frame packet to the buffer.

        If the buffer is at capacity, the oldest frame is automatically
        evicted to make room for the new frame (O(1) operation).

        Args:
            packet: VideoFramePacket to add to the buffer

        Raises:
            TypeError: If packet is not a VideoFramePacket
        """
        if not isinstance(packet, VideoFramePacket):
            raise TypeError(f"packet must be a VideoFramePacket, got {type(packet)}")

        self.frames.append(packet)

    def get_window(self) -> list[VideoFramePacket]:
        """
        Get all frames currently in the buffer.

        Returns a list copy of the buffer contents in chronological order
        (oldest to newest).

        Returns:
            List of VideoFramePacket objects in the buffer
        """
        return list(self.frames)

    def find_similar(
        self,
        packet: VideoFramePacket,
        similarity_fn: Callable[[VideoFramePacket, VideoFramePacket], bool],
    ) -> VideoFramePacket | None:
        """
        Find a frame in the buffer that matches the given packet.

        Searches through the buffer using a custom similarity function
        to find a matching frame. Returns the first match found.

        This method is useful for duplicate detection, where you want to
        check if a similar frame already exists in the temporal window.

        Args:
            packet: VideoFramePacket to compare against buffer contents
            similarity_fn: Function that takes two VideoFramePackets and
                          returns True if they are considered similar.
                          Signature: (packet1, packet2) -> bool

        Returns:
            First matching VideoFramePacket from the buffer, or None if
            no similar frame is found

        Raises:
            TypeError: If packet is not a VideoFramePacket or similarity_fn
                      is not callable

        Example:
            >>> def hash_similarity(p1, p2):
            ...     hash1 = p1.get_tag('phash')
            ...     hash2 = p2.get_tag('phash')
            ...     return hash1 == hash2
            >>>
            >>> buffer = TemporalBuffer(max_size=50)
            >>> similar = buffer.find_similar(new_packet, hash_similarity)
        """
        if not isinstance(packet, VideoFramePacket):
            raise TypeError(f"packet must be a VideoFramePacket, got {type(packet)}")

        if not callable(similarity_fn):
            raise TypeError(f"similarity_fn must be callable, got {type(similarity_fn)}")

        for buffered_packet in self.frames:
            if similarity_fn(packet, buffered_packet):
                return buffered_packet

        return None

    def __len__(self) -> int:
        """
        Get the current number of frames in the buffer.

        Returns:
            Number of frames currently stored
        """
        return len(self.frames)

    def __bool__(self) -> bool:
        """
        Check if the buffer contains any frames.

        Returns:
            True if buffer has frames, False if empty
        """
        return len(self.frames) > 0

    def clear(self) -> None:
        """
        Remove all frames from the buffer.

        Resets the buffer to empty state while maintaining the max_size.
        """
        self.frames.clear()

    def is_full(self) -> bool:
        """
        Check if the buffer is at maximum capacity.

        Returns:
            True if buffer contains max_size frames, False otherwise
        """
        return len(self.frames) == self.max_size
