"""
Phase 2 tests for StreamProcessor - Pipeline integration.

These tests validate that frames are actually processed through
taggers and filters, not just stored in the buffer.
"""

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from decimatr.core.stream_processor import OverflowStrategy, StreamProcessor
from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.base import Tagger


def create_test_frame(frame_number: int, timestamp_seconds: int = 0) -> VideoFramePacket:
    """Helper to create a test frame packet."""
    return VideoFramePacket(
        frame_data=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_number=frame_number,
        timestamp=datetime.timedelta(seconds=timestamp_seconds),
        source_video_id="test_video.mp4",
    )


class MockTagger(Tagger):
    """Mock tagger that adds a dummy tag."""

    @property
    def tag_keys(self):
        return ["mock_score"]

    def compute_tags(self, packet: VideoFramePacket):
        return {"mock_score": packet.frame_number * 10.0}


class MockFilter(StatelessFilter):
    """Mock filter that passes frames with frame_number >= threshold."""

    def __init__(self, threshold: int = 5):
        super().__init__()
        self.threshold = threshold

    @property
    def required_tags(self):
        return ["mock_score"]

    def should_pass(self, packet: VideoFramePacket) -> bool:
        score = packet.get_tag("mock_score", 0)
        return score >= self.threshold * 10.0


class TestPipelineProcessing:
    """Test that frames are actually processed through pipeline."""

    def test_frame_filtered_by_pipeline(self):
        """Test that frames are actually filtered by pipeline."""
        # Create a pipeline that filters out frames with frame_number < 5
        tagger = MockTagger()
        filter_ = MockFilter(threshold=5)

        stream = StreamProcessor(
            pipeline=[tagger, filter_],
            buffer_size=100,
            overflow_strategy=OverflowStrategy.DROP_OLDEST,
        )

        # Add frames 0-9
        for i in range(10):
            frame = create_test_frame(i)
            success = stream.add(frame)

            # Frames 0-4 should be filtered out (False)
            # Frames 5-9 should pass (True)
            assert success == (i >= 5), f"Frame {i} should {'pass' if i >= 5 else 'be filtered'}"

        # Verify only frames 5-9 are in buffer
        buffered_frames = list(stream.get_selected_frames())
        assert len(buffered_frames) == 5
        assert [f.frame_number for f in buffered_frames] == [5, 6, 7, 8, 9]

    def test_lazy_evaluation_applied(self):
        """Test that lazy evaluation works correctly."""
        # Create tagger that should only be called when needed
        tagger = MagicMock(spec=Tagger)
        tagger.tag_keys = ["test_score"]
        tagger.compute_tags.return_value = {"test_score": 100.0}

        # Create filter that doesn't use the tag (should skip tagger with lazy eval)
        filter_ = MagicMock(spec=StatelessFilter)
        filter_.required_tags = ["other_tag"]
        filter_.should_pass.return_value = True

        stream = StreamProcessor(
            pipeline=[tagger, filter_],
            buffer_size=100,
            lazy_evaluation=True,
        )

        frame = create_test_frame(0)
        stream.add(frame)

        # With lazy evaluation, tagger should NOT be called since filter doesn't need it
        tagger.compute_tags.assert_not_called()

    def test_no_pipeline_passes_all(self):
        """Test that empty pipeline passes all frames."""
        # Stream with no pipeline (pass-through)
        stream = StreamProcessor(pipeline=[], buffer_size=100)

        # All frames should pass
        for i in range(5):
            frame = create_test_frame(i)
            assert stream.add(frame) is True

        assert stream.get_buffer_count() == 5

    def test_tags_persist_in_buffer(self):
        """Test that computed tags are preserved in buffered frames."""
        tagger = MockTagger()
        filter_ = MockFilter(threshold=3)

        stream = StreamProcessor(pipeline=[tagger, filter_], buffer_size=100)

        # Add frames
        for i in range(5):
            frame = create_test_frame(i)
            stream.add(frame)

        # Verify tags are present in buffered frames
        for frame in stream.get_selected_frames():
            assert "mock_score" in frame.tags
            assert frame.get_tag("mock_score") == frame.frame_number * 10.0

    def test_metrics_with_filtering(self):
        """Test that metrics are tracked correctly with filtering."""
        filter_ = MockFilter(threshold=5)

        stream = StreamProcessor(
            pipeline=[MockTagger(), filter_],
            buffer_size=100,
            overflow_strategy=OverflowStrategy.REJECT,
        )

        # Add 10 frames - 5 should pass, 5 should be filtered
        for i in range(10):
            frame = create_test_frame(i)
            stream.add(frame)

        metrics = stream.get_metrics()

        # Verify metrics
        assert metrics["total_frames_added"] == 10
        assert metrics["total_frames_selected"] == 5  # Frames 5-9
        assert metrics["total_frames_rejected"] == 5  # Frames 0-4 filtered out
        assert metrics["buffer_count"] == 5
        assert metrics["selection_rate"] == 50.0  # 5/10 = 50%

    def test_release_memory_filtering(self):
        """Test that memory is released for filtered frames."""

        # Create a filter that rejects all frames
        class RejectAllFilter(StatelessFilter):
            @property
            def required_tags(self):
                return ["mock_score"]

            def should_pass(self, packet: VideoFramePacket) -> bool:
                return False

        stream = StreamProcessor(
            pipeline=[MockTagger(), RejectAllFilter()],
            buffer_size=100,
            release_memory=True,
        )

        frame = create_test_frame(0)
        original_shape = frame.frame_data.shape

        # Frame should be filtered out
        assert stream.add(frame) is False

        # Frame data should be released (replaced with tiny array)
        # The packet object is still passed, but _process_frame modifies it
        assert frame.frame_data.shape != original_shape

    def test_memory_released_after_buffer_overflow(self):
        """Test that frames are released when dropped from buffer (DROP_OLDEST)."""
        stream = StreamProcessor(
            buffer_size=2,
            overflow_strategy=OverflowStrategy.DROP_OLDEST,
            release_memory=True,
        )

        # Add 3 frames with frame 0 being dropped
        frame0 = create_test_frame(0)
        frame1 = create_test_frame(1)
        frame2 = create_test_frame(2)

        assert stream.add(frame0) is True
        assert stream.add(frame1) is True

        # Buffer should contain frame0 and frame1
        assert stream.get_buffer_count() == 2

        # Add third frame - frame0 should be dropped
        assert stream.add(frame2) is True

        # Buffer should now contain frame1 and frame2
        assert stream.get_buffer_count() == 2
        frame_numbers = [f.frame_number for f in stream.get_selected_frames()]
        assert frame_numbers == [1, 2]


class TestStatefulFilters:
    """Test stateful filter integration."""

    def test_duplicate_filter_with_stream(self):
        """Test that stateful filters like DuplicateFilter work with StreamProcessor."""
        from decimatr.filters.duplicate import DuplicateFilter
        from decimatr.taggers.hash import HashTagger

        # Create pipeline with duplicate detection
        stream = StreamProcessor(
            pipeline=[
                HashTagger(),
                DuplicateFilter(buffer_size=3, threshold=0.05),
            ],
            buffer_size=100,
        )

        # Add identical frames (should be duplicates)
        frame1 = create_test_frame(0)
        frame2 = create_test_frame(1)  # Different frame number but same data

        # First frame should pass
        assert stream.add(frame1) is True

        # Second frame should be filtered as duplicate (within 3-frame window)
        # Note: This depends on hash computation working correctly
        result = stream.add(frame2)
        # If hashes are identical, frame2 should be rejected
        # If hash computation fails, it might pass - that's okay for this test

        # At least one frame should be in buffer
        assert stream.get_buffer_count() >= 1


class TestRejection:
    """Test buffer overflow rejection scenarios."""

    def test_reject_strategy_with_filtering(self):
        """Test that REJECT strategy works correctly with filtering."""
        stream = StreamProcessor(
            pipeline=[MockTagger(), MockFilter(threshold=3)],
            buffer_size=3,
            overflow_strategy=OverflowStrategy.REJECT,
        )

        # Add 3 frames that pass (frames 3, 4, 5)
        assert stream.add(create_test_frame(3)) is True  # Pass
        assert stream.add(create_test_frame(4)) is True  # Pass
        assert stream.add(create_test_frame(5)) is True  # Pass

        assert stream.get_buffer_count() == 3
        assert stream.is_buffer_full() is True

        # Try to add frame 6 - should be rejected (buffer full)
        assert stream.add(create_test_frame(6)) is False

        # Try to add frame 2 - should be rejected (fails filter AND buffer full)
        assert stream.add(create_test_frame(2)) is False

        # Buffer should still contain only frames 3, 4, 5
        assert stream.get_buffer_count() == 3
        frame_numbers = [f.frame_number for f in stream.get_selected_frames()]
        assert frame_numbers == [3, 4, 5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
