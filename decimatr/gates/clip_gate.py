# ClipGate implementation for using CLIP models to analyze frame similarity

import logging
import uuid
from typing import Optional, List

import numpy as np
from PIL import Image

from decimatr.gates.base_gate import BaseGate
from decimatr.scheme import VideoFramePacket
from decimatr.samplers.clip_utils import get_image_embeddings


class ClipGate(BaseGate[float]):
    """
    Gate that uses CLIP-like model to determine frame similarity.
    This decides if a frame is semantically similar to previously seen frames.

    Note: This implementation depends on an external CLIP package.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        session_id: str = "default_session",
        clip_model_name: str = "ViT-B/32",
        clip_pretrained_weights: str = "openai",
        device: Optional[str] = None
    ):
        """
        Initialize the ClipGate

        Args:
            similarity_threshold: The threshold for considering frames semantically similar
            session_id: Identifier for the current session, used in logging
            clip_model_name: CLIP model architecture to use
            clip_pretrained_weights: Pre-trained weights to use
            device: Device to perform inference on. Auto-detects if None.
        """
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(f"Decimatr.{self.__class__.__name__}")
        self.session_id = session_id
        self.stored_embeddings: List[np.ndarray] = []
        self.clip_model_name = clip_model_name
        self.clip_pretrained_weights = clip_pretrained_weights
        self.device = device

    def _get_pil_image(self, packet: VideoFramePacket) -> Image.Image:
        """
        Convert the frame data from a VideoFramePacket to a PIL Image.

        Args:
            packet: The VideoFramePacket containing the frame data

        Returns:
            PIL Image representation of the frame
        """
        return Image.fromarray(packet.frame_data)

    def _compute_embedding(self, packet: VideoFramePacket) -> np.ndarray:
        """
        Compute the CLIP embedding for a frame using utility function.

        Args:
            packet: The VideoFramePacket containing the frame to analyze

        Returns:
            np.ndarray: The embedding vector
        """
        pil_image = self._get_pil_image(packet)
        
        # get_image_embeddings returns a batch of embeddings (even for a single image)
        # so we take the first (and only) one.
        embedding_batch = get_image_embeddings(
            images=pil_image,
            model_name=self.clip_model_name,
            pretrained=self.clip_pretrained_weights,
            device=self.device
        )
        return embedding_batch[0]

    def _is_similar_to_stored(self, embedding: np.ndarray) -> bool:
        """
        Check if an embedding is similar to any stored embeddings.

        Args:
            embedding: The embedding to check

        Returns:
            bool: True if similar to any stored embedding, False otherwise
        """
        if not self.stored_embeddings:
            return False

        for stored_embedding in self.stored_embeddings:
            # Compute cosine similarity
            similarity = np.dot(embedding, stored_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
            )

            if similarity > self.similarity_threshold:
                return True

        return False

    def process_frame(self, packet: VideoFramePacket) -> bool:
        """
        Process a single video frame packet using CLIP embeddings to determine if
        it is semantically similar to previously seen frames.

        Args:
            packet: The VideoFramePacket containing the frame to analyze

        Returns:
            bool: True if the frame is unique (passes the gate), False if it's similar (filtered)
        """
        # Compute embedding for the current frame
        embedding = self._compute_embedding(packet)

        # Check if similar to any stored embeddings
        is_similar = self._is_similar_to_stored(embedding)

        # Logging
        log_metadata = {
            "frame_number": packet.frame_number,
            "timestamp": packet.timestamp,
            "source_video_id": packet.source_video_id,
            "decision_is_similar": bool(is_similar),
        }
        log_dict = {
            "component_name": self.__class__.__name__,
            "operation": "process_frame_clip_check",
            "outcome": "similar" if is_similar else "unique",
            "event_id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "relevant_metadata": log_metadata,
        }
        self.logger.info(
            f"Frame {packet.frame_number} processed with CLIP.", extra=log_dict
        )

        # If not similar to any stored embedding, store this one
        if not is_similar:
            self.stored_embeddings.append(embedding)

        # Return True if the frame is unique (passes the gate), False if similar (filtered)
        return not is_similar

    def clear_embeddings(self) -> None:
        """
        Clear all stored embeddings.
        """
        self.stored_embeddings = []
