# This is a sampler that will sample the video based on the clip feature distance. Following https://github.com/steven-ccq/ViLAMP

import uuid
from typing import Dict, List, Optional, Iterator, Union, Any
import numpy as np
import logging
import torch

from decimatr.scheme import VideoFramePacket
from decimatr.logging_config import setup_logging
from decimatr.samplers.base_sampler import BaseSampler
from decimatr.samplers.clip_utils import get_image_embeddings
from PIL import Image

class DSKSampler(BaseSampler):
    """
    A sampler that uses CLIP features to sample representative frames based on feature diversity.
    
    This implementation follows the approach from ViLAMP (https://github.com/steven-ccq/ViLAMP)
    to select frames that maximize the diversity of visual content.
    """
    
    DEFAULT_CLIP_MODEL = "ViT-B/32"
    DEFAULT_NUM_KEYFRAMES = 5
    
    def __init__(self,
                 clip_model_name: str = DEFAULT_CLIP_MODEL,
                 device: Optional[str] = None,
                 num_keyframes: int = DEFAULT_NUM_KEYFRAMES):
        """
        Initialize the DSK sampler with configuration.
        
        Args:
            clip_model_name: CLIP model to use (default: "ViT-B/32")
            device: Device to use for computation (default: "cuda" if available, else "cpu")
            num_keyframes: Number of keyframes to select (default: 5)
        """
        self.clip_model_name = clip_model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.num_keyframes = num_keyframes
        self.logger = setup_logging(self.__class__.__name__)
    
    def _get_pil_image(self, packet: VideoFramePacket) -> Image.Image:
        """
        Convert a VideoFramePacket's frame_data (NumPy array) to a PIL Image.
        
        Args:
            packet: VideoFramePacket containing the frame data
            
        Returns:
            PIL Image object
        """
        # Convert NumPy array to PIL Image
        # Assuming frame_data is in RGB format
        return Image.fromarray(packet.frame_data.astype(np.uint8))
    
    def _extract_features(self, packets: List[VideoFramePacket]) -> Optional[torch.Tensor]:
        """
        Extract CLIP features from a list of VideoFramePacket objects using utility function.
        
        Args:
            packets: List of VideoFramePacket objects
            
        Returns:
            Tensor of CLIP features (normalized) or None if no packets.
        """
        if not packets:
            self.logger.warning("No packets to extract features from.")
            return None
        
        # Extract embeddings using the utility function
        # get_image_embeddings handles the batching, PIL conversion,
        # preprocessing, model loading (with caching), and normalization.
        image_embeddings_np = get_image_embeddings(
            images=[self._get_pil_image(packet) for packet in packets],
            model_name=self.clip_model_name,
            pretrained="openai",  # Using "openai" pretrained weights
            device=self.device
        )
        
        # Convert numpy array back to torch.Tensor for farthest point sampling
        return torch.from_numpy(image_embeddings_np).to(self.device)
    
    def _select_keyframes(self, features: Optional[torch.Tensor], num_keyframes: int) -> List[int]:
        """
        Select keyframe indices based on feature diversity.
        
        Args:
            features: Tensor of CLIP features or None
            num_keyframes: Number of keyframes to select
            
        Returns:
            List of selected keyframe indices
        """
        if features is None:
            return []
        
        # Move features to CPU for computation
        features_np = features.cpu().numpy()
        num_frames = features_np.shape[0]
        
        if num_frames <= num_keyframes:
            return list(range(num_frames))
        
        # Select frames using farthest point sampling
        selected_indices = [0]  # Start with the first frame
        
        for _ in range(num_keyframes - 1):
            # Compute distances to already selected frames
            min_distances = np.min([
                np.sum((features_np - features_np[idx])**2, axis=1)
                for idx in selected_indices
            ], axis=0)
            
            # Select the frame with the maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(int(next_idx))
        
        # Sort indices by frame order
        selected_indices.sort()
        return selected_indices
    
    def sample(self, frames: Union[List[VideoFramePacket], Iterator[VideoFramePacket]], session_id: str) -> List[VideoFramePacket]:
        """
        Sample frames based on CLIP feature diversity.
        
        Args:
            frames: List or Iterator of VideoFramePacket objects
            session_id: Unique identifier for the current processing session
            
        Returns:
            List of selected VideoFramePacket objects
        """
        # Convert iterator to list if needed
        if not isinstance(frames, list):
            frames = list(frames)
            
        if not frames:
            self.logger.warning(f"No frames to sample for session: {session_id}")
            return []
        
        total_frames = len(frames)
        self.logger.info(f"DSK sampling from {total_frames} frames for session: {session_id}")
        
        # Extract CLIP features
        features = self._extract_features(frames)
        
        # Select keyframe indices
        selected_indices = self._select_keyframes(features, self.num_keyframes)
        
        # Select frames based on indices
        selected_frames = [frames[i] for i in selected_indices]
        
        # Log sampling operation
        log_d = {
            "component_name": "DSKSampler",
            "operation": "dsk_sampling",
            "outcome": "completed",
            "session_id": session_id,
            "event_id": str(uuid.uuid4()),
            "relevant_metadata": {
                "total_frames": total_frames,
                "selected_frames": len(selected_frames),
                "requested_keyframes": self.num_keyframes,
                "clip_model": self.clip_model_name
            }
        }
        self.logger.info(f"DSK sampling completed for session: {session_id}", extra=log_d)
        
        return selected_frames
