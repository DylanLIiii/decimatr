"""
Motion detection filter using frame differencing.

This filter detects scene changes and significant motion by comparing frames
against recent history. It's useful for identifying key moments in videos,
detecting scene transitions, or filtering out static/low-motion sequences.
"""

import cv2
import numpy as np

from decimatr.filters.base import StatefulFilter
from decimatr.scheme import VideoFramePacket


class MotionFilter(StatefulFilter):
    """
    Filter that detects motion and scene changes via frame differencing.

    This filter maintains a temporal buffer of recent frames and compares each
    new frame against frames in the buffer using various motion metrics. Frames
    with motion above the threshold pass through, while static frames are filtered.

    The filter supports multiple motion detection methods:
    - 'mean_diff': Mean absolute difference between frames
    - 'mse': Mean squared error between frames
    - 'histogram': Histogram difference (robust to lighting changes)

    Attributes:
        buffer_size: Maximum number of frames to keep in temporal buffer
        threshold: Motion threshold. Higher values are more permissive.
        method: Motion detection method ('mean_diff', 'mse', 'histogram')
        compare_with: How to compare against history ('latest', 'mean', 'max')

    Example:
        >>> # Detect scene changes by comparing with most recent frame
        >>> filter = MotionFilter(
        ...     buffer_size=5,
        ...     threshold=30.0,
        ...     method='mean_diff',
        ...     compare_with='latest'
        ... )
        >>>
        >>> # Process frames
        >>> for packet in frame_stream:
        ...     if filter.should_pass(packet):
        ...         # Frame has significant motion, process it
        ...         process_frame(packet)
    """

    def __init__(
        self,
        buffer_size: int = 5,
        threshold: float = 30.0,
        method: str = "mean_diff",
        compare_with: str = "latest",
    ):
        """
        Initialize motion detection filter.

        Args:
            buffer_size: Maximum number of recent frames to compare against.
                        Smaller values detect rapid changes, larger values
                        detect gradual changes.
            threshold: Motion threshold value. Interpretation depends on method:
                      - 'mean_diff': Mean pixel difference (0-255 scale)
                      - 'mse': Mean squared error (0-65025 scale)
                      - 'histogram': Histogram correlation distance (0-1 scale)
            method: Motion detection method:
                   - 'mean_diff': Mean absolute difference (fast, simple)
                   - 'mse': Mean squared error (emphasizes large changes)
                   - 'histogram': Histogram comparison (robust to lighting)
            compare_with: How to compare against history:
                         - 'latest': Compare with most recent frame only
                         - 'mean': Compare with mean of all history frames
                         - 'max': Use maximum difference from any history frame

        Raises:
            ValueError: If method or compare_with is invalid
        """
        super().__init__(buffer_size)

        valid_methods = ["mean_diff", "mse", "histogram"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        valid_compare = ["latest", "mean", "max"]
        if compare_with not in valid_compare:
            raise ValueError(f"compare_with must be one of {valid_compare}, got '{compare_with}'")

        self.threshold = threshold
        self.method = method
        self.compare_with = compare_with

    def _compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute difference between two frames using configured method.

        Args:
            frame1: First frame (numpy array)
            frame2: Second frame (numpy array)

        Returns:
            Difference metric value (higher = more different)
        """
        # Convert to grayscale for comparison
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1

        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2

        # Ensure same size
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        if self.method == "mean_diff":
            # Mean absolute difference
            diff = np.abs(gray1.astype(float) - gray2.astype(float))
            return float(np.mean(diff))

        elif self.method == "mse":
            # Mean squared error
            diff = (gray1.astype(float) - gray2.astype(float)) ** 2
            return float(np.mean(diff))

        elif self.method == "histogram":
            # Histogram comparison (using correlation distance)
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

            # Normalize histograms
            hist1 = hist1 / (hist1.sum() + 1e-7)
            hist2 = hist2 / (hist2.sum() + 1e-7)

            # Compute correlation (1 = identical, 0 = completely different)
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # Return distance (0 = identical, 1 = completely different)
            return float(1.0 - correlation)

        return 0.0

    def compare_with_history(
        self, packet: VideoFramePacket, history: list[VideoFramePacket]
    ) -> bool:
        """
        Compare current frame against historical frames for motion detection.

        Computes motion metrics between the current frame and frames in the
        temporal buffer. If motion exceeds the threshold, the frame passes.

        Args:
            packet: Current frame to evaluate
            history: List of recent frames from the temporal buffer

        Returns:
            True if motion exceeds threshold (frame passes), False otherwise
        """
        # If buffer is empty, always pass (first frame or after reset)
        if not history:
            return True

        current_frame = packet.frame_data

        if self.compare_with == "latest":
            # Compare with most recent frame only
            past_frame = history[-1].frame_data
            motion = self._compute_frame_difference(current_frame, past_frame)
            return motion > self.threshold

        elif self.compare_with == "mean":
            # Compare with mean of all history frames
            # Stack all history frames and compute mean
            history_frames = [p.frame_data for p in history]

            # Convert to same dtype and compute mean
            mean_frame = np.mean([f.astype(float) for f in history_frames], axis=0).astype(np.uint8)

            motion = self._compute_frame_difference(current_frame, mean_frame)
            return motion > self.threshold

        elif self.compare_with == "max":
            # Use maximum difference from any history frame
            max_motion = 0.0
            for past_packet in history:
                past_frame = past_packet.frame_data
                motion = self._compute_frame_difference(current_frame, past_frame)
                max_motion = max(max_motion, motion)

            return max_motion > self.threshold

        return False

    @property
    def required_tags(self) -> list[str]:
        """
        Return list of tag keys required by this filter.

        Motion filter works directly with frame_data and doesn't require
        any pre-computed tags.

        Returns:
            Empty list (no tags required)
        """
        return []
