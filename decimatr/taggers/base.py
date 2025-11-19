"""
Base class and interface for all taggers.

Taggers are stateless components that analyze a single frame and compute
metadata tags without maintaining any state between frames.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from decimatr.scheme import VideoFramePacket


class Tagger(ABC):
    """
    Abstract base class for all taggers.

    Taggers analyze individual frames and compute metadata tags (e.g., blur score,
    hash value, entropy). They operate statelessly, meaning each frame is processed
    independently without reference to previous frames.

    Subclasses must implement:
        - compute_tags(): Compute tags for a single frame
        - tag_keys: Property listing the tag keys this tagger produces

    Optional methods:
        - compute_tags_batch(): Batch processing for GPU-accelerated taggers
    """

    @abstractmethod
    def compute_tags(self, packet: VideoFramePacket) -> dict[str, Any]:
        """
        Compute tags for a single frame.

        This method analyzes the frame data and returns a dictionary of computed
        metrics/tags. The frame packet is not modified; tags are returned separately
        to be added to the packet's tag registry by the processing pipeline.

        Args:
            packet: VideoFramePacket containing frame data and metadata

        Returns:
            Dictionary mapping tag keys to computed values

        Example:
            >>> tagger = BlurTagger()
            >>> tags = tagger.compute_tags(packet)
            >>> # tags = {"blur_score": 123.45}
        """
        pass

    @property
    @abstractmethod
    def tag_keys(self) -> list[str]:
        """
        Return list of tag keys this tagger produces.

        This property allows the processing pipeline to validate that required
        tags will be available for downstream filters.

        Returns:
            List of tag key strings

        Example:
            >>> tagger = BlurTagger()
            >>> tagger.tag_keys
            ['blur_score']
        """
        pass

    @property
    def supports_gpu(self) -> bool:
        """
        Whether this tagger supports GPU acceleration.

        Returns True if the tagger can utilize GPU resources for faster processing.
        Most CPU-based taggers return False.

        Returns:
            True if GPU acceleration is supported, False otherwise (default)
        """
        return False

    @property
    def requires_gpu(self) -> bool:
        """
        Whether this tagger requires GPU to function.

        Returns True if the tagger cannot operate without GPU resources.
        Most taggers are CPU-based and return False. Only specialized taggers
        like CLIP embeddings might require GPU for practical performance.

        Returns:
            True if GPU is required, False otherwise (default)
        """
        return False

    @property
    def is_cloud_based(self) -> bool:
        """
        Whether this tagger uses cloud-based models via async client.

        Returns True if the tagger sends requests to remote APIs for processing
        instead of computing locally. Cloud-based taggers use async HTTP clients
        and don't require local GPU resources.

        Returns:
            True if cloud-based, False otherwise (default)
        """
        return False

    def compute_tags_batch(self, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        """
        Batch compute tags for multiple frames (optional, for GPU processing).

        This method enables efficient batch processing on GPU for taggers that
        support GPU acceleration. The default implementation falls back to
        processing frames individually using compute_tags().

        GPU-accelerated taggers should override this method to implement
        efficient batch processing.

        Args:
            frames: List of frame data arrays (np.ndarray)

        Returns:
            List of tag dictionaries, one per input frame

        Raises:
            NotImplementedError: If batch processing is not supported and
                                 no fallback is desired

        Example:
            >>> tagger = CLIPTagger()
            >>> frames = [frame1, frame2, frame3]
            >>> tags_list = tagger.compute_tags_batch(frames)
            >>> # tags_list = [{"clip_embedding": ...}, {"clip_embedding": ...}, ...]
        """
        # Default implementation: process frames individually
        # GPU taggers should override this for efficient batch processing
        import datetime

        from decimatr.scheme import VideoFramePacket

        results = []
        for i, frame_data in enumerate(frames):
            # Create temporary packet for processing
            temp_packet = VideoFramePacket(
                frame_data=frame_data,
                frame_number=i,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="batch_processing",
            )
            tags = self.compute_tags(temp_packet)
            results.append(tags)

        return results
