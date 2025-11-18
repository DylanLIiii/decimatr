import logging
import uuid
from typing import Dict, List, Tuple

from decimatr.buffers.base_buffer import BaseBuffer
from decimatr.scheme import VideoFramePacket


class SlidingBuffer(BaseBuffer):
    """
    A buffer that extends BaseBuffer to support batched processing using a step_size.

    Frames are accumulated into a pending batch. When the batch size reaches
    `step_size`, the batch is processed. The processing involves an optional
    filtering step (via `_filter_batch`) before frames are committed to the
    main sliding window inherited from BaseBuffer.

    Attributes:
        step_size (int): The number of frames to accumulate in a batch before processing.
        pending_batch (List[Tuple[VideoFramePacket, Dict]]): Stores incoming frames
            and their associated keyword arguments before they are processed.
    """

    def __init__(
        self,
        window_size: int,
        step_size: int = 1,
        enabled: bool = True,
        session_id: str = "default_session",
    ):
        """
        Initializes the SlidingBuffer.

        Args:
            window_size: Maximum number of frames to keep in the main sliding window.
            step_size: Number of frames to accumulate in a batch before processing. Must be positive.
            enabled: Whether the buffer is active. If False, frames are batched but not
                     processed into the main window unless flush is called appropriately.
            session_id: Identifier for the current session, used in logging.
        """
        super().__init__(
            window_size=window_size, enabled=enabled, session_id=session_id
        )
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        self.step_size: int = step_size
        self.pending_batch: List[Tuple[VideoFramePacket, Dict]] = []
        self.logger = logging.getLogger(
            f"Decimatr.{self.__class__.__name__}"
        )  # Ensure logger uses this class's name

    def add(self, packet: VideoFramePacket, **kwargs) -> bool:
        """
        Adds a frame packet to the pending batch.

        If the buffer is enabled and the pending batch reaches `step_size`,
        the batch is processed.

        Args:
            packet: The VideoFramePacket object to add.
            **kwargs: Additional keyword arguments associated with the packet (e.g., frame_metric).

        Returns:
            bool: True, indicating the packet was accepted into the pending batch.
        """
        frame_identifier = f"{packet.source_video_id}_{packet.frame_number}"
        self.pending_batch.append((packet, kwargs))
        self.logger.debug(
            f"Frame {frame_identifier} added to pending batch. Batch size: {len(self.pending_batch)}/{self.step_size}",
            extra={
                "component_name": self.__class__.__name__,
                "operation": "add_to_pending_batch",
                "session_id": self.session_id,
                "event_id": str(uuid.uuid4()),
                "relevant_metadata": {
                    "frame_id": frame_identifier,
                    "pending_batch_size": len(self.pending_batch),
                    "step_size": self.step_size,
                },
            },
        )

        if self.enabled and len(self.pending_batch) >= self.step_size:
            self._process_pending_batch()
        elif not self.enabled:
            self.logger.debug(
                f"Buffer is disabled. Frame {frame_identifier} remains in pending_batch. Batch size: {len(self.pending_batch)}.",
                extra={
                    "component_name": self.__class__.__name__,
                    "operation": "add_to_pending_batch_disabled",
                    "session_id": self.session_id,
                    "event_id": str(uuid.uuid4()),
                    "relevant_metadata": {
                        "frame_id": frame_identifier,
                        "pending_batch_size": len(self.pending_batch),
                    },
                },
            )
        return True  # Packet always accepted into SlidingBuffer's batch

    def _process_pending_batch(self) -> None:
        """
        Processes the current pending_batch of frames.

        This involves filtering the batch (subclasses can customize this) and
        then adding the selected frames to the main sliding window via super().add().
        The pending_batch is cleared after its contents are copied for processing.
        """
        if not self.enabled:
            self.logger.debug(
                "Processing pending batch skipped as buffer is disabled.",
                extra={
                    "component_name": self.__class__.__name__,
                    "operation": "process_pending_batch_skipped",
                    "session_id": self.session_id,
                    "event_id": str(uuid.uuid4()),
                    "relevant_metadata": {
                        "pending_batch_size": len(self.pending_batch)
                    },
                },
            )
            return

        frames_to_process_from_batch = list(self.pending_batch)
        self.pending_batch.clear()

        # Pass a copy of the current window for potential contextual filtering
        filtered_batch_items = self._filter_batch(
            frames_to_process_from_batch, list(self.window)
        )

        committed_count = 0
        for packet, original_kwargs in filtered_batch_items:
            frame_id_for_base = f"{packet.source_video_id}_{packet.frame_number}"  # Used by BaseBuffer.add logging
            if super().add(packet, frame_id=frame_id_for_base, **original_kwargs):
                committed_count += 1

        self.logger.info(
            f"Processed {len(frames_to_process_from_batch)} frames from pending batch. "
            f"{committed_count} frames were eligible and attempted commitment to main window. "
            f"Main window size: {len(self.window)}.",
            extra={
                "component_name": self.__class__.__name__,
                "operation": "process_pending_batch_complete",
                "session_id": self.session_id,
                "event_id": str(uuid.uuid4()),
                "relevant_metadata": {
                    "processed_batch_size": len(frames_to_process_from_batch),
                    "filtered_batch_size": len(filtered_batch_items),
                    "committed_to_window_count": committed_count,
                    "current_window_size": len(self.window),
                },
            },
        )

    def _filter_batch(
        self,
        new_frames_batch_with_kwargs: List[Tuple[VideoFramePacket, Dict]],
        current_window_frames: List[VideoFramePacket],
    ) -> List[Tuple[VideoFramePacket, Dict]]:
        """
        Filters the frames from the pending batch before they are added to the main window.

        This base implementation performs no filtering and returns all frames from the batch.
        Subclasses can override this method to implement custom filtering logic,
        potentially using `current_window_frames` for context.

        Args:
            new_frames_batch_with_kwargs: A list of (VideoFramePacket, kwargs) tuples from the pending batch.
            current_window_frames: A list of VideoFramePacket objects currently in the main sliding window.

        Returns:
            List[Tuple[VideoFramePacket, Dict]]: The list of (VideoFramePacket, kwargs) tuples
                                                 that should be added to the main window.
        """
        self.logger.debug(
            f"Default _filter_batch: passing through all {len(new_frames_batch_with_kwargs)} frames from batch.",
            extra={
                "component_name": self.__class__.__name__,
                "operation": "_filter_batch_default_passthrough",
                "session_id": self.session_id,
                "event_id": str(uuid.uuid4()),
                "relevant_metadata": {"batch_size": len(new_frames_batch_with_kwargs)},
            },
        )
        return new_frames_batch_with_kwargs

    def flush(self) -> List[VideoFramePacket]:
        """
        Processes any remaining frames in the pending batch (if enabled)
        and then returns all frames from the main sliding window, clearing both.
        If disabled, pending frames are discarded.

        Returns:
            List[VideoFramePacket]: A list of all frame packets that were in the main
                                    sliding window before clearing.
        """
        if self.enabled and self.pending_batch:
            self.logger.debug(
                f"Flushing: Processing {len(self.pending_batch)} remaining frames in pending_batch.",
                extra={
                    "component_name": self.__class__.__name__,
                    "operation": "flush_process_pending",
                    "session_id": self.session_id,
                    "event_id": str(uuid.uuid4()),
                    "relevant_metadata": {
                        "pending_batch_size": len(self.pending_batch)
                    },
                },
            )
            self._process_pending_batch()  # This will also clear self.pending_batch
        elif not self.enabled and self.pending_batch:
            self.logger.info(
                f"Flushing: Discarding {len(self.pending_batch)} frames from pending_batch as buffer is disabled.",
                extra={
                    "component_name": self.__class__.__name__,
                    "operation": "flush_discard_pending_disabled",
                    "session_id": self.session_id,
                    "event_id": str(uuid.uuid4()),
                    "relevant_metadata": {
                        "pending_batch_size": len(self.pending_batch)
                    },
                },
            )
            self.pending_batch.clear()

        return super().flush()

    def clear(self) -> None:
        """
        Clears both the pending batch and the main sliding window.
        """
        self.pending_batch.clear()
        self.logger.debug(
            "Pending batch cleared.",
            extra={
                "component_name": self.__class__.__name__,
                "operation": "clear_pending_batch",
                "session_id": self.session_id,
                "event_id": str(uuid.uuid4()),
            },
        )
        super().clear()
