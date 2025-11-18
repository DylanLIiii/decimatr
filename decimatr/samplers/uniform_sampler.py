import uuid
from typing import Any, Dict, List, Optional, Iterator, Union

from decimatr.scheme import VideoFramePacket
from decimatr.logging_config import setup_logging
from decimatr.samplers.base_sampler import BaseSampler


class UniformSampler(BaseSampler):
    """
    A sampler that selects frames at uniform intervals from the input frame list.

    This sampler ensures even distribution of selected frames across the entire
    sequence, which is useful for summarizing video content with consistent
    temporal coverage.
    """

    DEFAULT_NUM_FRAMES = 10
    
    def __init__(self, num_frames: int = DEFAULT_NUM_FRAMES):
        """
        Initialize the uniform sampler with configuration.

        Args:
            num_frames: Number of frames to uniformly sample (default: 10)
        """
        self.num_frames = num_frames
        self.logger = setup_logging(self.__class__.__name__)

    def sample(
        self, frames: Union[List[VideoFramePacket], Iterator[VideoFramePacket]], session_id: str
    ) -> List[VideoFramePacket]:
        """
        Sample frames at uniform intervals from the input frame list.

        Args:
            frames: List or Iterator of VideoFramePacket objects
            session_id: Unique identifier for the current processing session

        Returns:
            List of uniformly sampled VideoFramePacket objects
        """
        # Convert iterator to list if needed
        if not isinstance(frames, list):
            frames = list(frames)
            
        if not frames:
            self.logger.warning(f"No frames to sample for session: {session_id}")
            return []

        total_frames = len(frames)
        self.logger.info(
            f"Uniform sampling from {total_frames} frames for session: {session_id}"
        )

        # If num_frames is greater than total frames, return all frames
        if self.num_frames >= total_frames:
            self.logger.info(
                f"Requested {self.num_frames} frames but only {total_frames} available. Returning all frames."
            )
            return frames

        # Calculate indices for uniformly distributed frames
        selected_frames = []
        if self.num_frames == 1:
            # If only one frame requested, take the middle frame
            middle_idx = total_frames // 2
            selected_frames = [frames[middle_idx]]
        else:
            # For multiple frames, calculate uniform distribution
            step = (
                (total_frames - 1) / (self.num_frames - 1) if self.num_frames > 1 else 1
            )
            indices = [int(round(i * step)) for i in range(self.num_frames)]
            # Ensure we don't exceed the array bounds
            indices = [min(i, total_frames - 1) for i in indices]
            # Ensure indices are unique (can happen with rounding)
            indices = list(dict.fromkeys(indices))
            selected_frames = [frames[i] for i in indices]

        # Log sampling operation
        log_d = {
            "component_name": "UniformSampler",
            "operation": "uniform_sampling",
            "outcome": "completed",
            "session_id": session_id,
            "event_id": str(uuid.uuid4()),
            "relevant_metadata": {
                "total_frames": total_frames,
                "selected_frames": len(selected_frames),
                "requested_frames": self.num_frames,
            },
        }
        self.logger.info(
            f"Uniform sampling completed for session: {session_id}", extra=log_d
        )

        return selected_frames
