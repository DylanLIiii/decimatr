import logging
import uuid
from typing import Any

from PIL import Image

from decimatr.gates.base_gate import BaseGate
from decimatr.gates.image_hash import ImageHasher
from decimatr.scheme import VideoFramePacket


class HashGate(BaseGate[Any]):
    DEFAULT_HASH_TYPE = "phash"
    DEFAULT_HASH_SIZE = 8
    DEFAULT_HIGHFREQ_FACTOR = 4
    DEFAULT_THRESHOLD = 0.05

    def __init__(
        self,
        hash_type: str = DEFAULT_HASH_TYPE,
        hash_size: int = DEFAULT_HASH_SIZE,
        highfreq_factor: int = DEFAULT_HIGHFREQ_FACTOR,
        threshold: float = DEFAULT_THRESHOLD,
        session_id: str = "default_session",
    ):
        """
        Initializes the HashGate component.

        Args:
            hash_type (str): Type of hash to use: 'ahash', 'phash', 'dhash', 'whash', or 'colorhash'
            hash_size (int): Size of the hash in bits.
            highfreq_factor (int): High frequency factor for wavelet hash.
            session_id (str): Identifier for the current session, used in logging.
        """
        self.logger = logging.getLogger(f"Decimatr.{self.__class__.__name__}")
        self.session_id = session_id
        self.hash_type = hash_type
        self.threshold = threshold
        # Create image hasher
        self.hasher = ImageHasher(hash_size=hash_size, highfreq_factor=highfreq_factor)

        # Store hashes to compare against
        self.stored_hashes = []

    def _get_pil_image(self, packet: VideoFramePacket) -> Image.Image:
        """
        Convert frame data to PIL Image for hashing.

        Args:
            packet: The VideoFramePacket containing the frame data

        Returns:
            PIL Image representation of the frame
        """
        return Image.fromarray(packet.frame_data)

    # TODO: Accelerate this with a calcualted hash function
    def _calculate_hash(self, packet: VideoFramePacket):
        """
        Calculates the perceptual hash for a frame.

        Args:
            packet: The VideoFramePacket containing the frame data

        Returns:
            imagehash.ImageHash: The perceptual hash object.
        """
        # Use compute_hash_from_array directly with frame_data
        return self.hasher.compute_hash_from_array(packet.frame_data, self.hash_type)

    def _is_duplicate(self, image_hash) -> bool:
        """
        Check if the given hash is similar to any stored hash.

        Args:
            image_hash: The hash to check against stored hashes.

        Returns:
            bool: True if it's a duplicate, False otherwise.
        """
        if not self.stored_hashes:
            return False

        for stored_hash in self.stored_hashes:
            # Calculate hash difference (lower is more similar)
            # Use direct hash difference instead of similarity
            hash_diff = self.hasher.hash_difference(image_hash, stored_hash)

            # Use a more appropriate threshold based on hash size
            # For 8-bit hash, max difference is 64 (8x8), consider similar if less than 5% different
            max_diff = self.hasher.hash_size * self.hasher.hash_size
            threshold = (
                max_diff * self.threshold
            )  # Reduced from 20% to 5% for stricter uniqueness test

            if hash_diff <= threshold:
                return True

        return False

    def process_frame(self, packet: VideoFramePacket) -> bool:
        """
        Process a single video frame packet to determine if it's a duplicate based on perceptual hashing.

        Args:
            packet: The VideoFramePacket containing the frame to analyze

        Returns:
            bool: True if the frame is unique (passes the gate), False if it's a duplicate (filtered)
        """
        frame_hash = self._calculate_hash(packet)
        is_duplicate = self._is_duplicate(frame_hash)

        # Logging
        log_metadata = {
            "frame_number": packet.frame_number,
            "timestamp": packet.timestamp,
            "source_video_id": packet.source_video_id,
            "frame_hash_calculated": str(frame_hash),
            "hash_type": self.hash_type,
            "decision_is_duplicate": bool(is_duplicate),
            "stored_hashes_count": len(self.stored_hashes),
        }
        log_dict = {
            "component_name": self.__class__.__name__,
            "operation": "process_frame_hash_check",
            "outcome": "duplicate" if is_duplicate else "unique",
            "event_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "relevant_metadata": log_metadata,
        }
        self.logger.info(
            f"Frame {packet.frame_number} processed for hash.", extra=log_dict
        )

        # If not a duplicate, store the hash for future comparisons
        if not is_duplicate:
            self.stored_hashes.append(frame_hash)

        # Return True if the frame is unique (passes the gate), False if it's a duplicate (filtered)
        return not is_duplicate

    def clear_hashes(self) -> None:
        """
        Clears the stored hashes.
        """
        self.stored_hashes = []

    def clear_all(self) -> None:
        """
        Clears both stored hashes.
        """
        self.clear_hashes()
