import datetime  # Assuming timestamps will be handled with datetime objects
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# from PIL import Image # Commented out for now, can be added if explicitly needed later


@dataclass
class VideoFramePacket:
    """
    Standardized data packet for a single video frame and its metadata.
    This structure will be used for passing frame data between samplers,
    gates, buffers, and other processing modules.

    Enhanced with tags registry to support the new tagging/filtering architecture.
    """

    frame_data: np.ndarray  # The raw frame image data as a NumPy array (from decord)
    frame_number: int  # Sequential frame number within the source video (0-indexed or 1-indexed)
    timestamp: datetime.timedelta  # Timestamp of the frame relative to the video start
    source_video_id: str  # A unique identifier for the source video file or stream

    additional_metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)  # Tag registry for computed metrics

    def __post_init__(self):
        if not isinstance(self.frame_data, np.ndarray):
            raise TypeError(f"frame_data must be a NumPy array, got {type(self.frame_data)}")
        if not (self.frame_data.ndim == 3 and self.frame_data.shape[2] in [1, 3, 4]) and not (
            self.frame_data.ndim == 2
        ):  # (H, W, C) or (H, W)
            raise ValueError(
                "frame_data must be a 2D (grayscale) or 3D (e.g., RGB, RGBA) NumPy array. "
                f"Received shape: {self.frame_data.shape}"
            )
        if not isinstance(self.frame_number, int) or self.frame_number < 0:
            raise ValueError(
                f"frame_number must be a non-negative integer, got {self.frame_number}"
            )
        if not isinstance(self.timestamp, datetime.timedelta):
            raise TypeError(f"timestamp must be a datetime.timedelta, got {type(self.timestamp)}")
        if not isinstance(self.source_video_id, str) or not self.source_video_id.strip():
            raise ValueError(
                f"source_video_id must be a non-empty string, got '{self.source_video_id}'"
            )
        if not isinstance(self.additional_metadata, dict):
            raise TypeError(
                f"additional_metadata must be a dict, got {type(self.additional_metadata)}"
            )
        if not isinstance(self.tags, dict):
            raise TypeError(f"tags must be a dict, got {type(self.tags)}")

    def get_tag(self, key: str, default: Any = None) -> Any:
        """
        Get tag value with optional default.

        Args:
            key: Tag key to retrieve
            default: Default value if tag key doesn't exist

        Returns:
            Tag value or default if key not found
        """
        return self.tags.get(key, default)

    def has_tags(self, keys: list) -> bool:
        """
        Check if all required tags are present.

        Args:
            keys: List of tag keys to check

        Returns:
            True if all keys are present in tags, False otherwise
        """
        return all(key in self.tags for key in keys)

    def copy_without_frame_data(self) -> "VideoFramePacket":
        """
        Create lightweight copy without frame data (for logging/serialization).

        Returns:
            New VideoFramePacket with minimal frame_data but all other fields copied
        """
        return VideoFramePacket(
            frame_data=np.zeros((1, 1, 3), dtype=np.uint8),  # Minimal valid frame
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            source_video_id=self.source_video_id,
            additional_metadata=self.additional_metadata.copy(),
            tags=self.tags.copy(),
        )

    def __eq__(self, other):
        """
        Compare two VideoFramePacket objects for equality.

        Args:
            other: Another object to compare with

        Returns:
            bool: True if objects are equal, False otherwise
        """
        if not isinstance(other, VideoFramePacket):
            return False

        # Compare all fields except frame_data which is a numpy array
        fields_equal = (
            self.frame_number == other.frame_number
            and self.timestamp == other.timestamp
            and self.source_video_id == other.source_video_id
            and self.additional_metadata == other.additional_metadata
            and self.tags == other.tags
        )

        # Compare the numpy arrays
        arrays_equal = np.array_equal(self.frame_data, other.frame_data)

        return fields_equal and arrays_equal
