"""
Tests for GPUBatchProcessor actor.

This module tests the GPU batch processing functionality including batch
accumulation, GPU processing with CPU fallback, and failure tracking.
"""

import asyncio
from datetime import timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xoscar as xo
from decimatr.actors.gpu_actor import GPUBatchProcessor
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.base import Tagger

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio


class MockGPUTagger(Tagger):
    """Mock tagger for testing GPU batch processing."""

    def __init__(self, fail_gpu: bool = False):
        self.fail_gpu = fail_gpu
        self.compute_calls = 0
        self.batch_calls = 0

    def compute_tags(self, packet: VideoFramePacket) -> dict:
        """CPU fallback method."""
        self.compute_calls += 1
        return {"mock_tag": f"cpu_{packet.frame_number}"}

    def compute_tags_batch(self, frames: list[np.ndarray]) -> list[dict]:
        """GPU batch method."""
        self.batch_calls += 1
        if self.fail_gpu:
            raise RuntimeError("GPU processing failed")
        return [{"mock_tag": f"gpu_{i}"} for i in range(len(frames))]

    @property
    def tag_keys(self) -> list[str]:
        return ["mock_tag"]

    @property
    def supports_gpu(self) -> bool:
        return True


def create_test_packet(frame_number: int) -> VideoFramePacket:
    """Create a test frame packet."""
    return VideoFramePacket(
        frame_data=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        frame_number=frame_number,
        timestamp=timedelta(seconds=frame_number / 30.0),
        source_video_id="test_video",
        tags={},
        additional_metadata={},
    )


async def test_gpu_batch_processor_initialization():
    """Test GPUBatchProcessor initialization."""
    tagger = MockGPUTagger()

    # Test valid initialization
    processor = GPUBatchProcessor(tagger, batch_size=4)
    assert processor.batch_size == 4
    assert processor.fallback_to_cpu is True
    assert processor.max_gpu_failures == 3
    assert len(processor.batch) == 0
    assert processor.gpu_failures == 0
    assert processor.using_cpu is False


async def test_gpu_batch_processor_invalid_params():
    """Test GPUBatchProcessor with invalid parameters."""
    tagger = MockGPUTagger()

    # Test invalid tagger type
    with pytest.raises(TypeError):
        GPUBatchProcessor("not a tagger", batch_size=4)

    # Test invalid batch size
    with pytest.raises(ValueError):
        GPUBatchProcessor(tagger, batch_size=0)


async def test_gpu_batch_accumulation():
    """Test frame accumulation in batch."""
    # Create actor pool
    await xo.create_actor_pool(address="127.0.0.1:13527", n_process=1)

    tagger = MockGPUTagger()
    actor_ref = await xo.create_actor(
        GPUBatchProcessor, tagger, 3, address="127.0.0.1:13527", uid="test_gpu_1"
    )

    # Add frames one by one
    packet1 = create_test_packet(1)
    result1 = await actor_ref.add_frame(packet1)
    assert result1 is None  # Batch not full yet

    packet2 = create_test_packet(2)
    result2 = await actor_ref.add_frame(packet2)
    assert result2 is None  # Batch not full yet

    # Third frame should trigger batch processing
    packet3 = create_test_packet(3)
    result3 = await actor_ref.add_frame(packet3)
    assert result3 is not None
    assert len(result3) == 3

    # Check that tags were added
    for packet in result3:
        assert "mock_tag" in packet.tags
        assert packet.tags["mock_tag"].startswith("gpu_")


async def test_gpu_batch_flush():
    """Test flushing remaining frames."""
    await xo.create_actor_pool(address="127.0.0.1:13528", n_process=1)

    tagger = MockGPUTagger()
    actor_ref = await xo.create_actor(
        GPUBatchProcessor, tagger, 5, address="127.0.0.1:13528", uid="test_gpu_2"
    )

    # Add 2 frames (less than batch size)
    packet1 = create_test_packet(1)
    await actor_ref.add_frame(packet1)

    packet2 = create_test_packet(2)
    await actor_ref.add_frame(packet2)

    # Flush should process remaining frames
    result = await actor_ref.flush()
    assert len(result) == 2

    # Check that tags were added
    for packet in result:
        assert "mock_tag" in packet.tags


async def test_gpu_failure_fallback():
    """Test CPU fallback on GPU failure."""
    await xo.create_actor_pool(address="127.0.0.1:13529", n_process=1)

    # Create tagger that fails on GPU
    tagger = MockGPUTagger(fail_gpu=True)
    actor_ref = await xo.create_actor(
        GPUBatchProcessor,
        tagger,
        2,
        fallback_to_cpu=True,
        address="127.0.0.1:13529",
        uid="test_gpu_3",
    )

    # Add frames to trigger batch processing
    packet1 = create_test_packet(1)
    await actor_ref.add_frame(packet1)

    packet2 = create_test_packet(2)
    result = await actor_ref.add_frame(packet2)

    # Should fall back to CPU
    assert result is not None
    assert len(result) == 2

    # Check that CPU method was used
    for packet in result:
        assert "mock_tag" in packet.tags
        assert packet.tags["mock_tag"].startswith("cpu_")

    # Check failure count
    failures = await actor_ref.get_gpu_failure_count()
    assert failures == 1


async def test_gpu_failure_threshold():
    """Test switching to CPU after failure threshold."""
    await xo.create_actor_pool(address="127.0.0.1:13530", n_process=1)

    # Create tagger that fails on GPU
    tagger = MockGPUTagger(fail_gpu=True)
    actor_ref = await xo.create_actor(
        GPUBatchProcessor,
        tagger,
        2,
        fallback_to_cpu=True,
        max_gpu_failures=2,
        address="127.0.0.1:13530",
        uid="test_gpu_4",
    )

    # First batch - GPU fails, falls back to CPU
    await actor_ref.add_frame(create_test_packet(1))
    await actor_ref.add_frame(create_test_packet(2))

    # Second batch - GPU fails again, falls back to CPU
    await actor_ref.add_frame(create_test_packet(3))
    await actor_ref.add_frame(create_test_packet(4))

    # Check that we've switched to CPU permanently
    using_cpu = await actor_ref.is_using_cpu()
    assert using_cpu is True

    # Third batch - should use CPU directly
    await actor_ref.add_frame(create_test_packet(5))
    result = await actor_ref.add_frame(create_test_packet(6))

    assert result is not None
    assert len(result) == 2


async def test_gpu_batch_stats():
    """Test getting batch processor statistics."""
    await xo.create_actor_pool(address="127.0.0.1:13531", n_process=1)

    tagger = MockGPUTagger()
    actor_ref = await xo.create_actor(
        GPUBatchProcessor, tagger, 4, address="127.0.0.1:13531", uid="test_gpu_5"
    )

    # Get initial stats
    stats = await actor_ref.get_stats()
    assert stats["batch_size"] == 4
    assert stats["current_batch_count"] == 0
    assert stats["gpu_failures"] == 0
    assert stats["using_cpu"] is False
    assert stats["fallback_enabled"] is True

    # Add some frames
    await actor_ref.add_frame(create_test_packet(1))
    await actor_ref.add_frame(create_test_packet(2))

    # Check updated stats
    stats = await actor_ref.get_stats()
    assert stats["current_batch_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
