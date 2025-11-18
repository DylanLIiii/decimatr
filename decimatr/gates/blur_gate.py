import logging
import uuid
from typing import Optional

import cv2
import numpy as np

from decimatr.gates.base_gate import BaseGate
from decimatr.scheme import VideoFramePacket


class BlurGate(BaseGate[float]):
    DEFAULT_THRESHOLD = 100.0  # Class attribute for default blurriness threshold

    def __init__(
        self, threshold: Optional[float] = None, session_id: str = "default_session"
    ):
        """
        Initializes the BlurGate component.

        Args:
            threshold (Optional[float]): The variance of Laplacian threshold.
                             Images with variance below this are considered blurred.
                             If None, uses DEFAULT_THRESHOLD.
            session_id (str): Identifier for the current session, used in logging.
        """
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        self.logger = logging.getLogger(f"Decimatr.{self.__class__.__name__}")
        self.session_id = session_id

    def _calculate_blur_score(self, frame_data: np.ndarray) -> float:
        """
        Calculates the blur score (variance of Laplacian) for an image.
        check: https://medium.com/@sagardhungel/laplacian-and-its-use-in-blur-detection-fbac689f0f88


        Args:
            frame_data (np.ndarray): The input frame data as a NumPy array.

        Returns:
            float: The variance of the Laplacian.
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
        return laplacian.var()

    def process_frame(self, packet: VideoFramePacket) -> bool:
        """
        Process a single video frame packet to determine if it contains blur.

        Args:
            packet (VideoFramePacket): The frame packet containing the frame data and metadata

        Returns:
            bool: True if the frame passes the blur gate (not blurry), False otherwise
        """
        blur_score = self._calculate_blur_score(packet.frame_data)
        # lower score is more blurry
        is_blurred = blur_score < self.threshold

        # Logging
        log_metadata = {
            "frame_number": packet.frame_number,
            "timestamp": packet.timestamp,
            "source_video_id": packet.source_video_id,
            "blur_score_calculated": float(blur_score),
            "threshold_used": self.threshold,
            "decision_is_blurred": bool(is_blurred),
        }
        log_dict = {
            "component_name": self.__class__.__name__,
            "operation": "process_frame_blur_check",
            "outcome": "blurred" if is_blurred else "not_blurred",
            "event_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "relevant_metadata": log_metadata,
        }
        self.logger.info(
            f"Frame {packet.frame_number} processed for blur.", extra=log_dict
        )

        # Return True if the frame should pass (not blurred), False if it should be filtered (blurred)
        return not is_blurred
