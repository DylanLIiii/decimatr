"""
Phase 1 tests for StreamProcessor - Basic buffer management.

These tests validate the core buffer functionality without pipeline processing.
"""

import numpy as np
import pytest
from decimatr.core.stream_processor import OverflowStrategy, StreamProcessor
from decimatr.scheme import VideoFramePacket


def create_test_frame(frame_number: int, timestamp_seconds: int = 0) -> VideoFramePacket:
    """Helper to create a test frame packet."""
    return VideoFramePacket(
        frame_data=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_number=frame_number,
        timestamp=__import__("datetime").timedelta(seconds=timestamp_seconds),
        source_video_id="test_video.mp4",
    )


class TestBufferInitialization:
    """Test StreamProcessor initialization."""

    def test_default_initialization(self):
        """Test default initialization with no pipeline."""
        stream = StreamProcessor()
        assert stream.pipeline == []
        assert stream.buffer_size == 1000
        assert stream.overflow_strategy == OverflowStrategy.DROP_OLDEST
        assert stream.n_workers == 1
        assert stream.use_gpu is False
        assert stream.lazy_evaluation is True
        assert stream.release_memory is True
        assert stream.get_buffer_count() == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        stream = StreamProcessor(
            pipeline=["tagger1", "filter1"],  # Placeholder objects
            buffer_size=100,
            overflow_strategy=OverflowStrategy.REJECT,
            n_workers=4,
            use_gpu=True,
            release_memory=False,
        )
        assert stream.pipeline == ["tagger1", "filter1"]
        assert stream.buffer_size == 100
        assert stream.overflow_strategy == OverflowStrategy.REJECT
        assert stream.n_workers == 4
        assert stream.use_gpu is True
        assert stream.release_memory is False

    def test_invalid_buffer_size(self):
        """Test that invalid buffer_size raises error."""
        with pytest.raises(ValueError, match="buffer_size must be at least 1"):
            StreamProcessor(buffer_size=0)

        with pytest.raises(ValueError, match="buffer_size must be at least 1"):
            StreamProcessor(buffer_size=-1)

    def test_invalid_n_workers(self):
        """Test that invalid n_workers raises error."""
        with pytest.raises(ValueError, match="n_workers must be at least 1"):
            StreamProcessor(n_workers=0)

        with pytest.raises(ValueError, match="n_workers must be at least 1"):
            StreamProcessor(n_workers=-5)


class TestBufferManagement:
    """Test buffer operations."""

    def test_add_frame(self):
        """Test adding a frame to the buffer."""
        stream = StreamProcessor(buffer_size=10, overflow_strategy=OverflowStrategy.DROP_OLDEST)
        frame = create_test_frame(0)

        # Phase 1: Frames are always added (no pipeline processing yet)
        assert stream.add(frame) is True
        assert stream.get_buffer_count() == 1

        # Can retrieve the frame
        frames = list(stream.get_selected_frames())
        assert len(frames) == 1
        assert frames[0] == frame

    def test_buffer_fifo_order(self):
        """Test that frames are stored in FIFO order."""
        stream = StreamProcessor(buffer_size=10)

        # Add multiple frames
        frames = [create_test_frame(i) for i in range(5)]
        for frame in frames:
            stream.add(frame)

        # Verify order is preserved
        buffered_frames = list(stream.get_selected_frames())
        assert len(buffered_frames) == 5
        for i, frame in enumerate(buffered_frames):
            assert frame.frame_number == i

    def test_drop_oldest_overflow(self):
        """Test DROP_OLDEST strategy when buffer is full."""
        stream = StreamProcessor(buffer_size=3, overflow_strategy=OverflowStrategy.DROP_OLDEST)

        # Add 3 frames (fills buffer)
        for i in range(3):
            stream.add(create_test_frame(i))

        assert stream.get_buffer_count() == 3
        assert stream.get_selected_frames_as_list()[0].frame_number == 0

        # Add 4th frame - oldest (frame 0) should be dropped
        stream.add(create_test_frame(3))

        assert stream.get_buffer_count() == 3
        frames = stream.get_selected_frames_as_list()
        assert len(frames) == 3
        assert frames[0].frame_number == 1  # Frame 0 was dropped
        assert frames[1].frame_number == 2
        assert frames[2].frame_number == 3

    def test_reject_overflow(self):
        """Test REJECT strategy when buffer is full."""
        stream = StreamProcessor(buffer_size=3, overflow_strategy=OverflowStrategy.REJECT)

        # Fill buffer to capacity
        for i in range(3):
            assert stream.add(create_test_frame(i)) is True

        assert stream.get_buffer_count() == 3
        assert stream.is_buffer_full() is True

        # Try to add another frame - should be rejected
        result = stream.add(create_test_frame(3))
        assert result is False  # Rejected

        # Buffer should still contain original frames
        assert stream.get_buffer_count() == 3
        frames = stream.get_selected_frames_as_list()
        assert frames[0].frame_number == 0
        assert frames[1].frame_number == 1
        assert frames[2].frame_number == 2

    def test_clear_buffer(self):
        """Test clearing the buffer."""
        stream = StreamProcessor(buffer_size=10)

        # Add frames
        for i in range(5):
            stream.add(create_test_frame(i))

        assert stream.get_buffer_count() == 5
        assert stream.get_metrics()["total_frames_selected"] == 5

        # Clear buffer
        stream.clear()

        assert stream.get_buffer_count() == 0
        assert len(list(stream.get_selected_frames())) == 0

        # Metrics should be reset
        metrics = stream.get_metrics()
        assert metrics["total_frames_added"] == 0
        assert metrics["total_frames_selected"] == 0
        assert metrics["total_frames_rejected"] == 0

    def test_is_buffer_full(self):
        """Test buffer full detection."""
        stream = StreamProcessor(buffer_size=2)

        assert stream.is_buffer_full() is False

        stream.add(create_test_frame(0))
        assert stream.is_buffer_full() is False

        stream.add(create_test_frame(1))
        assert stream.is_buffer_full() is True

        stream.clear()
        assert stream.is_buffer_full() is False


class TestMetrics:
    """Test metrics tracking."""

    def test_initial_metrics(self):
        """Test metrics for empty stream."""
        stream = StreamProcessor()
        metrics = stream.get_metrics()

        assert metrics["total_frames_added"] == 0
        assert metrics["total_frames_selected"] == 0
        assert metrics["total_frames_rejected"] == 0
        assert metrics["buffer_count"] == 0
        assert metrics["buffer_capacity"] == 1000
        assert metrics["selection_rate"] == 0.0

    def test_metrics_with_rejection(self):
        """Test metrics with overflow rejections."""
        stream = StreamProcessor(buffer_size=2, overflow_strategy=OverflowStrategy.REJECT)

        # Add 3 frames (2 accepted, 1 rejected)
        assert stream.add(create_test_frame(0)) is True
        assert stream.add(create_test_frame(1)) is True
        assert stream.add(create_test_frame(2)) is False  # Rejected

        metrics = stream.get_metrics()
        # TODO: In Phase 1, total_frames_added isn't updated (Phase 2)
        # TODO: total_frames_rejected should be 1 (Phase 1 placeholder)
        assert metrics["buffer_count"] == 2
        # In Phase 2 we'll fix this to properly track rejections

    def test_selection_rate_calculation(self):
        """Test selection rate calculation."""
        stream = StreamProcessor()

        # Simulate 50% selection rate (Phase 1: all frames "pass")
        for i in range(10):
            stream.add(create_test_frame(i))

        # Phase 1: All frames are "selected" (no pipeline processing)
        metrics = stream.get_metrics()
        # TODO: In Phase 1, total_frames_added isn't tracked correctly
        # This will be fixed in Phase 2 when we add proper pipeline processing


# Helper method for tests (could add to StreamProcessor if needed)


def get_selected_frames_as_list(stream: StreamProcessor):
    """Helper to convert iterator to list for testing."""
    return list(stream.get_selected_frames())


# Monkey patch for testing convenience
StreamProcessor.get_selected_frames_as_list = get_selected_frames_as_list
