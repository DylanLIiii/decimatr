"""
GPUBatchProcessor actor for GPU-accelerated batch processing of frames.

This module provides an actor that accumulates frames into batches and
processes them on GPU for improved throughput. It includes automatic CPU
fallback when GPU operations fail and tracks GPU failures to switch to
CPU processing after a threshold.
"""

import logging
from typing import Any

import xoscar as xo

from decimatr.scheme import VideoFramePacket
from decimatr.taggers.base import Tagger

logger = logging.getLogger(__name__)


class GPUBatchProcessor(xo.Actor):
    """
    Actor that batches frames for GPU processing with CPU fallback.

    GPUBatchProcessor accumulates frames into batches and processes them
    on GPU using the tagger's batch processing method. This provides
    significant performance improvements for GPU-accelerated taggers like
    CLIP embeddings.

    The actor includes automatic CPU fallback when GPU operations fail and
    tracks GPU failures to permanently switch to CPU processing after a
    configurable threshold.

    Key Features:
        - Batch accumulation for efficient GPU processing
        - Automatic CPU fallback on GPU failure
        - GPU failure tracking and automatic CPU switching
        - Configurable batch size and failure threshold

    Attributes:
        tagger: GPU-enabled Tagger instance
        batch_size: Number of frames to accumulate before processing
        fallback_to_cpu: Whether to fall back to CPU on GPU failure
        batch: Current batch of accumulated frames
        gpu_failures: Count of GPU processing failures
        max_gpu_failures: Threshold for switching to CPU permanently
        using_cpu: Whether currently using CPU processing

    Example:
        >>> # Create GPU batch processor actor
        >>> clip_tagger = CLIPTagger(device="cuda")
        >>> actor_ref = await xo.create_actor(
        ...     GPUBatchProcessor,
        ...     clip_tagger,
        ...     batch_size=32,
        ...     address='127.0.0.1:13527'
        ... )
        >>>
        >>> # Add frames to batch
        >>> result = await actor_ref.add_frame(packet1)
        >>> result = await actor_ref.add_frame(packet2)
        >>> # ... batch processes automatically when full
    """

    def __init__(
        self,
        tagger: Tagger,
        batch_size: int = 32,
        fallback_to_cpu: bool = True,
        max_gpu_failures: int = 3,
    ):
        """
        Initialize GPUBatchProcessor with a GPU-enabled tagger.

        Args:
            tagger: Tagger instance with GPU support (must implement compute_tags_batch)
            batch_size: Number of frames to accumulate before processing (default: 32)
            fallback_to_cpu: Whether to fall back to CPU on GPU failure (default: True)
            max_gpu_failures: Number of GPU failures before switching to CPU (default: 3)

        Raises:
            TypeError: If tagger is not a Tagger instance
            ValueError: If batch_size is less than 1
        """
        if not isinstance(tagger, Tagger):
            raise TypeError(f"tagger must be a Tagger instance, got {type(tagger)}")

        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        self.tagger = tagger
        self.batch_size = batch_size
        self.fallback_to_cpu = fallback_to_cpu
        self.max_gpu_failures = max_gpu_failures

        # Batch state
        self.batch: list[VideoFramePacket] = []

        # GPU failure tracking
        self.gpu_failures = 0
        self.using_cpu = False

        logger.info(
            f"GPUBatchProcessor initialized: batch_size={batch_size}, "
            f"fallback_to_cpu={fallback_to_cpu}, max_gpu_failures={max_gpu_failures}"
        )

    async def add_frame(self, packet: VideoFramePacket) -> list[VideoFramePacket] | None:
        """
        Add a frame to the batch and process when full.

        This method accumulates frames into a batch. When the batch reaches
        the configured batch_size, it automatically processes the batch and
        returns the processed frames. If the batch is not yet full, returns None.

        Args:
            packet: VideoFramePacket to add to the batch

        Returns:
            List of processed VideoFramePackets if batch is full, None otherwise

        Raises:
            TypeError: If packet is not a VideoFramePacket
            Exception: If batch processing fails and fallback is disabled

        Example:
            >>> # Add frames one by one
            >>> result = await actor_ref.add_frame(packet1)  # None (batch not full)
            >>> result = await actor_ref.add_frame(packet2)  # None (batch not full)
            >>> # ... add more frames
            >>> result = await actor_ref.add_frame(packet32)  # Returns processed batch
        """
        if not isinstance(packet, VideoFramePacket):
            raise TypeError(f"packet must be a VideoFramePacket, got {type(packet)}")

        # Add frame to batch
        self.batch.append(packet)

        # Process batch if full
        if len(self.batch) >= self.batch_size:
            return await self.process_batch()

        return None

    async def process_batch(self) -> list[VideoFramePacket]:
        """
        Process the current batch on GPU with CPU fallback.

        This method processes all frames in the current batch using the
        tagger's batch processing method. If GPU processing fails and
        fallback is enabled, it automatically falls back to CPU processing.

        The method tracks GPU failures and switches to CPU permanently
        after exceeding the failure threshold.

        Returns:
            List of processed VideoFramePackets with tags added

        Raises:
            Exception: If processing fails and fallback is disabled

        Example:
            >>> # Manually trigger batch processing
            >>> processed = await actor_ref.process_batch()
            >>> for packet in processed:
            ...     print(packet.tags)
        """
        if not self.batch:
            return []

        # Check if we should use CPU due to previous failures
        if self.using_cpu:
            logger.debug(f"Processing batch of {len(self.batch)} frames on CPU (permanent switch)")
            return await self._process_batch_cpu()

        # Try GPU processing
        try:
            logger.debug(f"Processing batch of {len(self.batch)} frames on GPU")
            results = self.tagger.compute_tags_batch([p.frame_data for p in self.batch])

            # Update packets with computed tags
            for packet, tags in zip(self.batch, results):
                packet.tags.update(tags)

            # Clear batch and return processed frames
            processed = self.batch
            self.batch = []

            logger.debug(f"Successfully processed batch of {len(processed)} frames on GPU")
            return processed

        except Exception as e:
            self.gpu_failures += 1
            logger.warning(
                f"GPU batch processing failed (attempt {self.gpu_failures}/{self.max_gpu_failures}): {e}"
            )

            # Check if we should switch to CPU permanently
            if self.gpu_failures >= self.max_gpu_failures:
                logger.warning(
                    f"GPU failures exceeded threshold ({self.max_gpu_failures}), "
                    f"switching to CPU processing permanently"
                )
                self.using_cpu = True

            # Fall back to CPU if enabled
            if self.fallback_to_cpu:
                logger.info("Falling back to CPU processing for current batch")
                return await self._process_batch_cpu()
            else:
                # Clear batch and re-raise if fallback disabled
                self.batch = []
                raise

    async def _process_batch_cpu(self) -> list[VideoFramePacket]:
        """
        Process batch using CPU (fallback method).

        This internal method processes frames one by one using the tagger's
        CPU compute_tags method. It's used as a fallback when GPU processing
        fails or after switching to CPU permanently.

        Returns:
            List of processed VideoFramePackets with tags added

        Raises:
            Exception: If CPU processing fails
        """
        try:
            # Process each frame individually on CPU
            for packet in self.batch:
                tags = self.tagger.compute_tags(packet)
                packet.tags.update(tags)

            # Clear batch and return processed frames
            processed = self.batch
            self.batch = []

            logger.debug(f"Successfully processed batch of {len(processed)} frames on CPU")
            return processed

        except Exception as e:
            logger.error(f"CPU batch processing failed: {e}")
            # Clear batch and re-raise
            self.batch = []
            raise

    async def flush(self) -> list[VideoFramePacket]:
        """
        Flush any remaining frames in the batch.

        This method processes any frames remaining in the batch, even if
        the batch is not full. It should be called at the end of processing
        to ensure all frames are processed.

        Returns:
            List of processed VideoFramePackets, empty list if no frames

        Example:
            >>> # At end of processing, flush remaining frames
            >>> remaining = await actor_ref.flush()
            >>> print(f"Processed {len(remaining)} remaining frames")
        """
        if not self.batch:
            return []

        logger.debug(f"Flushing batch with {len(self.batch)} remaining frames")
        return await self.process_batch()

    def get_batch_size(self) -> int:
        """
        Get the configured batch size.

        Returns:
            Batch size (number of frames per batch)
        """
        return self.batch_size

    def get_current_batch_count(self) -> int:
        """
        Get the number of frames currently in the batch.

        Returns:
            Number of frames in current batch
        """
        return len(self.batch)

    def get_gpu_failure_count(self) -> int:
        """
        Get the number of GPU processing failures.

        Returns:
            Count of GPU failures
        """
        return self.gpu_failures

    def is_using_cpu(self) -> bool:
        """
        Check if the actor has switched to CPU processing.

        Returns:
            True if using CPU, False if still attempting GPU
        """
        return self.using_cpu

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the batch processor.

        Returns:
            Dictionary with batch processor statistics including:
            - batch_size: Configured batch size
            - current_batch_count: Frames in current batch
            - gpu_failures: Number of GPU failures
            - using_cpu: Whether switched to CPU
            - fallback_enabled: Whether CPU fallback is enabled
        """
        return {
            "batch_size": self.batch_size,
            "current_batch_count": len(self.batch),
            "gpu_failures": self.gpu_failures,
            "using_cpu": self.using_cpu,
            "fallback_enabled": self.fallback_to_cpu,
            "max_gpu_failures": self.max_gpu_failures,
        }
