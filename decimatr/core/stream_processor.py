"""
StreamProcessor for real-time frame processing with sliding window buffer.

This module provides a streaming API for processing frames as they arrive,
maintaining a fixed-size buffer of selected frames for real-time applications.
"""

import logging
from collections import deque
from enum import Enum
from typing import Any

from decimatr.core.processor import FrameProcessor
from decimatr.scheme import VideoFramePacket

logger = logging.getLogger(__name__)


class OverflowStrategy(Enum):
    """
    Strategy for handling buffer overflow in StreamProcessor.

    Attributes:
        DROP_OLDEST: Remove oldest frame when buffer is full (FIFO)
        REJECT: Reject new frame when buffer is full
    """

    DROP_OLDEST = "drop_oldest"
    REJECT = "reject"


class StreamProcessor(FrameProcessor):
    """
    Real-time stream processor for video frames with sliding window buffer.

    StreamProcessor processes frames immediately as they arrive via add() and
    maintains a fixed-size buffer of frames that passed all filters. This enables
    real-time applications to process continuous frame streams while keeping
    access to recent selected frames.

    The processor applies the configured pipeline (taggers + filters) to each
    frame as it's added. Frames that pass all filters are added to the stream
    buffer, which maintains a sliding window of the most recent selected frames.

    Stateful filters maintain separate internal buffers for temporal context,
    independent of the main stream buffer.

    This class extends FrameProcessor to reuse pipeline validation and processing
    logic while adding stream-specific buffer management.

    Attributes:
        buffer_size: Maximum number of selected frames to keep in buffer
        overflow_strategy: How to handle buffer overflow
        Inherited from FrameProcessor:
            pipeline, n_workers, use_gpu, gpu_batch_size, lazy_evaluation,
            release_memory

    Example:
        >>> # Create stream processor
        >>> from decimatr.taggers.blur import BlurTagger
        >>> from decimatr.filters.blur import BlurFilter
        >>> stream = StreamProcessor(
        ...     pipeline=[BlurTagger(), BlurFilter(threshold=100.0)],
        ...     buffer_size=100,
        ...     overflow_strategy=OverflowStrategy.DROP_OLDEST
        ... )
        >>>
        >>> # Process frames as they arrive
        >>> for frame in video_stream:
        ...     if stream.add(frame):
        ...         print(f"Frame {frame.frame_number} selected")
        ...
        ...     # Access recent selected frames at any time
        ...     for recent_frame in stream.get_selected_frames():
        ...         analyze_frame(recent_frame)

    Requirements:
        - Provides add() method for frame-by-frame processing
        - Maintains sliding window buffer of selected frames
        - Returns True/False from add() indicating pass/fail
        - Supports all FrameProcessor features (lazy eval, GPU, actors)
    """

    def __init__(
        self,
        pipeline: list | None = None,
        buffer_size: int = 1000,
        overflow_strategy: OverflowStrategy = OverflowStrategy.DROP_OLDEST,
        n_workers: int = 1,
        use_gpu: bool = False,
        gpu_batch_size: int = 32,
        lazy_evaluation: bool = True,
        release_memory: bool = True,
    ):
        """
        Initialize stream processor with pipeline and buffer configuration.

        Args:
            pipeline: Ordered list of taggers and filters. If None, uses pass-through.
            buffer_size: Maximum number of selected frames to keep in buffer.
                        When buffer is full, overflow_strategy determines behavior.
            overflow_strategy: How to handle buffer overflow:
                - DROP_OLDEST: Remove oldest frame, add new frame (default)
                - REJECT: Reject new frame, keep existing buffer
            n_workers: Number of workers for parallel processing. Default is 1
                      (single-threaded). Values > 1 enable actor-based processing.
            use_gpu: Enable GPU acceleration. Requires GPU dependencies.
            gpu_batch_size: Batch size for GPU processing. Default is 32.
            lazy_evaluation: Enable lazy tag computation. Default is True.
            release_memory: Release frame data from filtered frames. Default is True.

        Raises:
            ValueError: If buffer_size < 1 or n_workers < 1

        Note:
            For actor-based parallel processing (n_workers > 1), StreamProcessor
            uses FrameProcessor's actor pipeline infrastructure (Phase 4).
        """
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be at least 1, got {buffer_size}")

        # Initialize FrameProcessor (parent class) - validates pipeline
        super().__init__(
            pipeline=pipeline,
            n_workers=n_workers,
            use_gpu=use_gpu,
            gpu_batch_size=gpu_batch_size,
            lazy_evaluation=lazy_evaluation,
            release_memory=release_memory,
        )

        # Initialize stream-specific attributes
        self.buffer_size = buffer_size
        self.overflow_strategy = overflow_strategy

        # Initialize sliding window buffer (stores selected frames)
        self._buffer: deque[VideoFramePacket] = deque(maxlen=buffer_size)

        # Initialize metrics tracking
        self._total_frames_added = 0
        self._total_frames_selected = 0
        self._total_frames_rejected = 0

    def add(self, packet: VideoFramePacket) -> bool:
        """
        Add a frame to the stream for immediate processing.

        The frame is processed through the pipeline immediately. If it passes
        all filters, it's added to the stream buffer and the method returns True.
        If the frame is filtered out or buffer overflow occurs (when using REJECT
        strategy), the method returns False.

        Args:
            packet: VideoFramePacket to process and add to stream

        Returns:
            True if frame passed all filters and was added to buffer, False otherwise

        Example:
            >>> success = stream.add(frame_packet)
            >>> if success:
            ...     print(f"Frame {frame_packet.frame_number} selected")
            >>> else:
            ...     print(f"Frame {frame_packet.frame_number} filtered out or rejected")

        Note:
            Currently only single-threaded processing (n_workers=1) is implemented.
            Actor support (n_workers > 1) will be added in Phase 4.
        """
        # Update metrics
        self._total_frames_added += 1

        # Check buffer overflow with REJECT strategy
        if (
            self.overflow_strategy == OverflowStrategy.REJECT
            and len(self._buffer) >= self.buffer_size
        ):
            self._total_frames_rejected += 1
            return False

        # Warn about actor support not yet implemented
        if self.n_workers > 1:
            logger.warning(
                f"Actor-based processing (n_workers={self.n_workers}) not yet implemented. "
                "Falling back to single-threaded processing."
            )

        # Process frame using FrameProcessor's single-threaded logic
        # This applies taggers and filters according to lazy_evaluation setting
        processed_frame = self._process_frame(packet)

        if processed_frame is not None:
            # Frame passed all filters
            # Add to buffer (deque handles overflow based on maxlen)
            self._buffer.append(processed_frame)
            self._total_frames_selected += 1
            return True
        else:
            # Frame filtered out
            # Memory release is handled by _process_frame
            self._total_frames_rejected += 1
            return False

    def get_selected_frames(self) -> iter:
        """
        Get iterator over frames currently in the stream buffer.

        Returns an iterator yielding VideoFramePacket objects in chronological
        order (oldest to newest). The buffer contains only frames that passed
        all filters.

        Returns:
            Iterator of VideoFramePacket objects from the buffer

        Example:
            >>> # Process all frames in buffer
            >>> for frame in stream.get_selected_frames():
            ...     process_frame(frame)
            >>>
            >>> # Convert to list for random access
            >>> frames = list(stream.get_selected_frames())
            >>> print(f"Buffer contains {len(frames)} frames")
        """
        return iter(self._buffer)

    def get_buffer_count(self) -> int:
        """
        Get the current number of frames in the stream buffer.

        Returns:
            Number of frames currently stored in buffer (0 to buffer_size)

        Example:
            >>> count = stream.get_buffer_count()
            >>> print(f"Buffer contains {count} frames")
        """
        return len(self._buffer)

    def get_buffer_capacity(self) -> int:
        """
        Get the maximum capacity of the stream buffer.

        Returns:
            The configured buffer_size value
        """
        return self.buffer_size

    def is_buffer_full(self) -> bool:
        """
        Check if the buffer has reached maximum capacity.

        Returns:
            True if buffer contains buffer_size frames, False otherwise

        Example:
            >>> if stream.is_buffer_full():
            ...     print("Buffer is full")
        """
        return len(self._buffer) == self.buffer_size

    def clear(self) -> None:
        """
        Clear all frames from the stream buffer and reset state.

        Removes all frames from the buffer and resets metrics counters.
        Stateful filters maintain their own state and are not affected by this
        method (call their clear_buffer() method separately if needed).

        Example:
            >>> # Reset stream between videos
            >>> stream.clear()
            >>> for frame in new_video:
            ...     stream.add(frame)
        """
        self._buffer.clear()
        self._total_frames_added = 0
        self._total_frames_selected = 0
        self._total_frames_rejected = 0

    def get_metrics(self) -> dict[str, Any]:
        """
        Get metrics about stream processing.

        Returns:
            Dictionary containing stream statistics:
            - total_frames_added: Total frames added via add()
            - total_frames_selected: Frames that passed all filters
            - total_frames_rejected: Frames filtered out or rejected
            - buffer_count: Current number of frames in buffer
            - buffer_capacity: Maximum buffer size
            - selection_rate: Percentage of frames selected

        Example:
            >>> metrics = stream.get_metrics()
            >>> print(f"Selected {metrics['selection_rate']:.1f}% of frames")
        """
        total = self._total_frames_added
        selected = self._total_frames_selected

        return {
            "total_frames_added": total,
            "total_frames_selected": selected,
            "total_frames_rejected": self._total_frames_rejected,
            "buffer_count": len(self._buffer),
            "buffer_capacity": self.buffer_size,
            "selection_rate": (selected / total * 100.0) if total > 0 else 0.0,
        }
