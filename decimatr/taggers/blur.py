"""
BlurTagger - Stateless blur score computation for video frames.

This tagger computes the blur score (variance of Laplacian) for frames,
which can be used by filters to remove blurry frames from video processing pipelines.
"""

from typing import Any

import cv2
import numpy as np

from decimatr.scheme import VideoFramePacket
from decimatr.taggers.base import Tagger


class BlurTagger(Tagger):
    """
    Compute blur score using Laplacian variance (CPU-based).

    The blur score is calculated using the variance of the Laplacian operator
    applied to the grayscale version of the frame. Lower scores indicate more
    blurry images, while higher scores indicate sharper images.

    Reference: https://medium.com/@sagardhungel/laplacian-and-its-use-in-blur-detection-fbac689f0f88

    This tagger is stateless and processes each frame independently.

    Example:
        >>> tagger = BlurTagger()
        >>> tags = tagger.compute_tags(packet)
        >>> # tags = {"blur_score": 123.45}
        >>> blur_score = tags["blur_score"]
    """

    def compute_tags(self, packet: VideoFramePacket) -> dict[str, Any]:
        """
        Compute blur score for a single frame.

        Args:
            packet: VideoFramePacket containing frame data and metadata

        Returns:
            Dictionary with "blur_score" key containing the variance of Laplacian

        Example:
            >>> tagger = BlurTagger()
            >>> tags = tagger.compute_tags(packet)
            >>> print(tags)
            {"blur_score": 123.45}
        """
        blur_score = self._calculate_blur_score(packet.frame_data)
        return {"blur_score": float(blur_score)}

    def _calculate_blur_score(self, frame_data: np.ndarray) -> float:
        """
            Calculate the blur score (variance of Laplacian) for an image.

            This is a pure function that computes the blur metric without any state.
            The algorithm:
            1. Convert frame to grayscale if needed

        2. Apply Laplacian operator to detect edges
            3. Calculate variance of the Laplacian result

            Args:
                frame_data: The input frame data as a NumPy array

            Returns:
                The variance of the Laplacian (blur score)

            Note:
                - Returns 0.0 for empty frames or frames with zero dimensions
                - Handles both RGB and grayscale input frames
        """
        # Check if image is empty (has 0 width or height)
        if frame_data.size == 0 or 0 in frame_data.shape[:2]:
            return 0.0

        # Convert to grayscale if needed
        if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
            gray_image = cv2.cvtColor(frame_data, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = frame_data

        # Compute the Laplacian of the grayscale image
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

        # Calculate the variance of the Laplacian
        return float(laplacian.var())

    @property
    def tag_keys(self) -> list[str]:
        """
        Return list of tag keys this tagger produces.

        Returns:
            List containing "blur_score"
        """
        return ["blur_score"]
