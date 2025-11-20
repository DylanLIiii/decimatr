"""
CLIPTagger - GPU-accelerated CLIP embedding computation for video frames.

This tagger computes CLIP (Contrastive Language-Image Pre-training) embeddings
for frames using GPU acceleration when available. CLIP embeddings can be used
for semantic similarity comparison, content-based filtering, and diversity
sampling.

This implementation uses open_clip library with automatic model selection:
- GPU: Standard CLIP models (ViT-B-32, ViT-L-14, etc.)
- CPU: MobileCLIP models optimized for CPU inference

Note: This tagger requires optional GPU dependencies (torch, open-clip-torch).
Install with: pip install decimatr[gpu]
"""

from typing import Any

import numpy as np

from decimatr.exceptions import GPUDependencyError
from decimatr.gpu_utils import GPUCapabilities
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.base import Tagger


class CLIPTagger(Tagger):
    """
    Compute CLIP embeddings using open_clip library.

    This tagger uses the open_clip library to compute semantic embeddings for
    video frames. It automatically selects the appropriate model based on device:
    - GPU: Standard CLIP models (ViT-B-32, ViT-L-14, etc.)
    - CPU: MobileCLIP models optimized for efficient CPU inference

    CLIP embeddings capture high-level semantic content and can be used for:
    - Semantic similarity comparison
    - Content-based frame selection
    - Diversity-based sampling

    The tagger supports both single-frame and batch processing. Batch processing
    is significantly more efficient on both GPU and CPU.

    GPU Requirements:
        - torch >= 2.0.0
        - torchvision >= 0.15.0
        - open-clip-torch >= 2.32.0
        - CUDA-capable GPU
        - Install with: pip install decimatr[gpu]

    Example:
        >>> # Check GPU availability first
        >>> if GPUCapabilities.is_available():
        ...     tagger = CLIPTagger(model_name="ViT-B-32", device="cuda")
        ... else:
        ...     # Automatically uses MobileCLIP for CPU
        ...     tagger = CLIPTagger(device="cpu")
        >>>
        >>> # Single frame processing
        >>> tags = tagger.compute_tags(packet)
        >>> embedding = tags["clip_embedding"]
        >>>
        >>> # Batch processing (more efficient)
        >>> frames = [packet1.frame_data, packet2.frame_data, packet3.frame_data]
        >>> tags_list = tagger.compute_tags_batch(frames)
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "auto",
        batch_size: int = 32
    ):
        """
        Initialize CLIP tagger with open_clip.

        Args:
            model_name: CLIP model variant to use (for GPU). Options include:
                - "ViT-B-32" (default): Balanced performance and accuracy
                - "ViT-B-16": Higher accuracy, slower
                - "ViT-L-14": Highest accuracy, slowest
                - "RN50": ResNet-50 based, faster but less accurate
                Note: For CPU, MobileCLIP models are automatically selected
            pretrained: Pretrained weights to use (for GPU):
                - "openai" (default): OpenAI's pretrained weights
                - "laion2b_s34b_b79k": LAION-2B trained weights
                - "laion400m_e32": LAION-400M trained weights
            device: Device to use for computation:
                - "auto" (default): Use CUDA if available, otherwise CPU
                - "cuda": Force GPU usage (raises error if unavailable)
                - "cpu": Force CPU usage (uses MobileCLIP)
            batch_size: Batch size for batch processing (default: 32)

        Raises:
            GPUDependencyError: If device="cuda" but GPU dependencies are missing
        """
        # Determine device
        if device == "auto":
            device = "cuda" if GPUCapabilities.is_available() else "cpu"

        # Validate GPU availability if CUDA requested
        if device == "cuda" and not GPUCapabilities.is_available():
            missing = GPUCapabilities.get_missing_dependencies()
            raise GPUDependencyError(
                f"GPU requested but dependencies are missing: {', '.join(missing)}. "
                f"Install with: pip install decimatr[gpu]"
            )

        self.device = device
        self.batch_size = batch_size
        
        # Select model based on device
        if device == "cpu":
            # Use MobileCLIP for CPU - smallest variant for best performance
            self.model_name = "MobileCLIP-S1"
            self.pretrained = "datacompdr"
        else:
            # Use standard CLIP for GPU
            self.model_name = model_name
            self.pretrained = pretrained
        
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self):
        """
        Lazy load CLIP model using open_clip.

        The model is loaded on first use to avoid unnecessary initialization
        overhead. This also allows the tagger to be instantiated even if
        CLIP dependencies are not available (error will be raised on first use).

        For CPU devices, MobileCLIP models are automatically loaded for
        optimized inference performance.

        Raises:
            GPUDependencyError: If open_clip dependencies are not installed
        """
        if self._model is None:
            try:
                import open_clip
            except ImportError as e:
                raise GPUDependencyError(
                    f"open_clip dependencies not available: {e}. "
                    f"Install with: pip install decimatr[gpu]"
                ) from e

            # Load model and preprocessing transforms
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            
            # Get tokenizer (for potential future text encoding)
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Set to evaluation mode
            self._model.eval()

    def compute_tags(self, packet: VideoFramePacket) -> dict[str, Any]:
        """
        Compute CLIP embedding for a single frame.

        Args:
            packet: VideoFramePacket containing frame data and metadata

        Returns:
            Dictionary with "clip_embedding" key containing the embedding
            as a numpy array of shape (embedding_dim,)

        Example:
            >>> tagger = CLIPTagger()
            >>> tags = tagger.compute_tags(packet)
            >>> embedding = tags["clip_embedding"]
            >>> print(embedding.shape)  # (512,) for ViT-B-32
        """
        self._load_model()

        try:
            import torch
            from PIL import Image
        except ImportError as e:
            raise GPUDependencyError(
                f"Required dependencies not available: {e}. "
                f"Install with: pip install decimatr[gpu]"
            ) from e

        # Convert frame to PIL Image (CLIP expects RGB)
        # frame_data is already in RGB format from decord
        image = Image.fromarray(packet.frame_data)

        # Preprocess and add batch dimension
        image_input = self._preprocess(image).unsqueeze(0).to(self.device)

        # Compute embedding
        with torch.no_grad():
            embedding = self._model.encode_image(image_input)
            # Normalize embedding (CLIP embeddings are typically normalized)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            # Convert to numpy and flatten
            embedding = embedding.cpu().numpy().flatten()

        return {"clip_embedding": embedding}

    def compute_tags_batch(self, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        """
        Batch compute CLIP embeddings (efficient for both GPU and CPU).

        This method processes multiple frames in a single batch, which is
        significantly more efficient than processing frames individually.
        
        For CPU devices, this uses MobileCLIP's optimized inference path.
        For GPU devices, this uses standard CLIP batch processing.

        Args:
            frames: List of frame data arrays (np.ndarray) in RGB format

        Returns:
            List of tag dictionaries, one per input frame, each containing
            "clip_embedding" key with the embedding as a numpy array

        Example:
            >>> tagger = CLIPTagger()
            >>> frames = [frame1, frame2, frame3]
            >>> tags_list = tagger.compute_tags_batch(frames)
            >>> for tags in tags_list:
            ...     embedding = tags["clip_embedding"]
            ...     print(embedding.shape)  # (512,) for ViT-B-32
        """
        self._load_model()

        try:
            import torch
            from PIL import Image
        except ImportError as e:
            raise GPUDependencyError(
                f"Required dependencies not available: {e}. "
                f"Install with: pip install decimatr[gpu]"
            ) from e

        # Convert all frames to PIL Images
        images = [Image.fromarray(frame) for frame in frames]

        # Preprocess all images and stack into batch
        image_inputs = torch.stack([self._preprocess(img) for img in images]).to(self.device)

        # Compute embeddings for entire batch
        with torch.no_grad():
            if self.device == "cpu":
                # Use MobileCLIP's optimized inference path for CPU
                # MobileCLIP models in open_clip use the same encode_image interface
                # but are optimized for CPU execution
                embeddings = self._model.encode_image(image_inputs)
            else:
                # Standard CLIP inference for GPU
                embeddings = self._model.encode_image(image_inputs)
            
            # Normalize embeddings (CLIP embeddings are typically normalized)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            # Convert to numpy
            embeddings = embeddings.cpu().numpy()

        # Return list of tag dictionaries
        return [{"clip_embedding": emb} for emb in embeddings]

    @property
    def tag_keys(self) -> list[str]:
        """
        Return list of tag keys this tagger produces.

        Returns:
            List containing "clip_embedding"
        """
        return ["clip_embedding"]

    @property
    def supports_gpu(self) -> bool:
        """
        Whether this tagger supports GPU acceleration.

        Returns:
            True (CLIP benefits significantly from GPU acceleration)
        """
        return True

    @property
    def requires_gpu(self) -> bool:
        """
        Whether this tagger requires GPU to function.

        CLIP can run on CPU but is significantly slower. For practical
        purposes, GPU is highly recommended but not strictly required.

        Returns:
            False (can fall back to CPU if needed, just slower)
        """
        return False
