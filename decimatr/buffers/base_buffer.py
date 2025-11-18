import collections
import logging
import uuid
from typing import Deque, List, Optional

from decimatr.scheme import VideoFramePacket


class BaseBuffer:
    """
    Base class for implementing sliding window buffers.

    This class provides common sliding window functionality that can be inherited
    by specific buffer implementations. A sliding window buffer maintains a fixed-size
    collection of the most recently added items, automatically dropping the oldest items
    when the buffer reaches capacity.

    BaseBuffer provides the core functionality for maintaining this sliding window,
    while specialized subclasses (like SlidingTopKBuffer) add additional logic for
    selecting or filtering frames based on specific criteria.

    Attributes:
        window_size (int): Maximum number of frames to keep in the sliding window.
        enabled (bool): Whether the buffer is active. If False, frames pass through without buffering.
        session_id (str): Identifier for the current session, used in logging.
        window (Deque[VideoFramePacket]): The deque that stores the sliding window of frame packets.
    """

    def __init__(
        self,
        window_size: int,
        enabled: bool = True,
        session_id: str = "default_session",
    ):
        """
        Initializes the BaseBuffer.

        Args:
            window_size: Maximum number of frames to keep in the sliding window.
            enabled: Whether the buffer is active. If False, frames pass through without buffering.
            session_id: Identifier for the current session, used in logging.
        """
        self.window_size: int = window_size
        self.enabled: bool = enabled
        self.session_id = session_id
        self.logger = logging.getLogger(f"Decimatr.{self.__class__.__name__}")

        # Initialize the sliding window deque with maximum length of window_size
        self.window: Deque[VideoFramePacket] = collections.deque(maxlen=window_size)

    def add(
        self, packet: VideoFramePacket, frame_id: Optional[str] = None, **kwargs
    ) -> bool:
        """
        Adds a frame packet to the sliding window.

        This is a base implementation that simply adds the frame packet to the window.
        Subclasses should override this method to implement specific buffering logic,
        such as filtering or sorting based on frame metrics.

        When the window reaches its maximum size (self.window_size), the oldest frame packet
        is automatically removed when a new one is added.

        Args:
            packet: The VideoFramePacket object to add to the buffer.
            frame_id: Optional identifier for the frame, used in logging. If None, it will be
                     derived from packet information.
            **kwargs: Additional keyword arguments for specific buffer implementations.

        Returns:
            bool: True if the frame packet was added to the buffer, False otherwise.
                 Returns False if the buffer is disabled.
        """
        # If frame_id is not provided, derive it from the packet
        if frame_id is None:
            frame_id = f"{packet.source_video_id}_{packet.frame_number}"
        if not self.enabled:
            self.logger.debug(
                f"Buffer is disabled. Frame {frame_id} passing through.",
                extra={
                    "component_name": self.__class__.__name__,
                    "operation": "add_frame",
                    "outcome": "passthrough",
                    "event_id": str(uuid.uuid4()),
                    "session_id": self.session_id,
                    "relevant_metadata": {
                        "frame_id": frame_id,
                        "buffer_enabled": self.enabled,
                    },
                },
            )
            return False

        self.window.append(packet)
        self.logger.debug(
            f"Frame {frame_id} added to buffer. Window size: {len(self.window)}/{self.window_size}",
            extra={
                "component_name": self.__class__.__name__,
                "operation": "add_frame",
                "outcome": "added",
                "event_id": str(uuid.uuid4()),
                "session_id": self.session_id,
                "relevant_metadata": {
                    "frame_id": frame_id,
                    "window_size": len(self.window),
                },
            },
        )
        return True

    def get_window(self) -> List[VideoFramePacket]:
        """
        Returns the current contents of the sliding window.

        The returned list is ordered from oldest to newest frame packets in the window.
        This method creates a new list containing all items currently in the window,
        leaving the internal window deque unchanged.

        Returns:
            List[VideoFramePacket]: A list of all frame packets currently in the window,
                                    ordered from oldest to newest.
        """
        return list(self.window)

    def clear(self) -> None:
        """
        Clears the sliding window buffer.

        After calling this method, the buffer will be empty, but will retain its
        original window_size. New frames can still be added to the empty buffer.
        """
        self.window.clear()
        self.logger.debug(
            "Buffer cleared",
            extra={
                "component_name": self.__class__.__name__,
                "operation": "clear_buffer",
                "outcome": "success",
                "event_id": str(uuid.uuid4()),
                "session_id": self.session_id,
            },
        )

    def flush(self) -> List[VideoFramePacket]:
        """
        Returns the current contents of the sliding window and clears the buffer.

        This is a convenience method that combines get_window() and clear() in a single
        operation, ensuring that you get all the frame packets before clearing the buffer.

        Returns:
            List[VideoFramePacket]: A list of all frame packets that were in the window before clearing,
                                    ordered from oldest to newest.
        """
        frames = self.get_window()
        self.clear()
        return frames
