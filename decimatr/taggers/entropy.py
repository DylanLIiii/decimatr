"""
EntropyTagger - Stateless entropy computation for video frames.

This tagger computes the Shannon entropy of frames, which measures the
information content or randomness in an image. Higher entropy indicates
more information/complexity, while lower entropy indicates less information
(e.g., blank frames, uniform regions).
"""

from typing import Any

import numpy as np

from decimatr.scheme import VideoFramePacket
from decimatr.taggers.base import Tagger


class EntropyTagger(Tagger):
    """
    Compute Shannon entropy for video frames (CPU-based).

    Shannon entropy measures the information content in an image by analyzing
    the distribution of pixel intensities. The entropy is calculated on the
    grayscale version of the frame using the formula:

        H = -Σ(p(i) * log2(p(i)))

    where p(i) is the probability of intensity value i occurring in the image.

    Higher entropy values indicate more complex/informative frames, while lower
    values indicate simpler frames with less information (e.g., blank frames,
    uniform backgrounds).

    This tagger is stateless and processes each frame independently.

    Example:
        >>> tagger = EntropyTagger()
        >>> tags = tagger.compute_tags(packet)
        >>> # tags = {"entropy": 6.234}
        >>> entropy_score = tags["entropy"]
    """

    def compute_tags(self, packet: VideoFramePacket) -> dict[str, Any]:
        """
        Compute Shannon entropy for a single frame.

        Args:
            packet: VideoFramePacket containing frame data and metadata

        Returns:
            Dictionary with "entropy" key containing the Shannon entropy value

        Example:
            >>> tagger = EntropyTagger()
            >>> tags = tagger.compute_tags(packet)
            >>> print(tags)
            {"entropy": 6.234}
        """
        entropy_score = self._calculate_entropy(packet.frame_data)
        return {"entropy": float(entropy_score)}

    def _calculate_entropy(self, frame_data: np.ndarray) -> float:
        """
        Calculate the Shannon entropy of an image.

        This is a pure function that computes the entropy metric without any state.
        The algorithm:
        1. Convert frame to grayscale if needed
        2. Calculate histogram of pixel intensities (256 bins for 0-255 range)
        3. Normalize histogram to get probability distribution
        4. Apply Shannon entropy formula: H = -Σ(p(i) * log2(p(i)))

        Args:
            frame_data: The input frame data as a NumPy array

        Returns:
            The Shannon entropy value (typically 0-8 for 8-bit images)

        Note:
            - Returns 0.0 for empty frames
            - Handles both RGB and grayscale input frames
            - Uses epsilon values to avoid log(0) errors
        """
        # Convert to grayscale if needed
        if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
            # Convert RGB to grayscale using standard RGB to grayscale conversion
            gray_image = np.dot(frame_data[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = frame_data

        # Check for empty image
        if gray_image.size == 0:
            return 0.0

        # Calculate histogram
        hist = np.histogram(gray_image.ravel(), bins=256, range=(0, 255))[0]

        # Normalize histogram to get probability distribution
        probabilities = hist / (hist.sum() + np.finfo(float).eps)

        # Calculate Shannon entropy
        # Add epsilon to avoid log(0)
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))

        return float(entropy)

    @property
    def tag_keys(self) -> list[str]:
        """
        Return list of tag keys this tagger produces.

        Returns:
            List containing "entropy"
        """
        return ["entropy"]
