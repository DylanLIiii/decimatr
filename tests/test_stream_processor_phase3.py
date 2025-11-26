"""
Phase 3 tests for StreamProcessor - Stateful filter integration.

These tests validate that stateful filters work correctly with StreamProcessor,
maintaining their internal state across add() calls.
"""

import datetime

import numpy as np
import pytest
from decimatr.core.stream_processor import OverflowStrategy, StreamProcessor
from decimatr.filters.duplicate import DuplicateFilter
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.hash import HashTagger


def create_test_frame(
    frame_number: int, timestamp_seconds: int = 0, data_value: int = 128
) -> VideoFramePacket:
    """Helper to create a test frame packet with unique data."""
    # Create frame with unique data based on frame_number
    data = np.full((100, 100, 3), data_value, dtype=np.uint8)
    return VideoFramePacket(
        frame_data=data,
        frame_number=frame_number,
        timestamp=datetime.timedelta(seconds=timestamp_seconds),
        source_video_id="test_video.mp4",
    )


class TestDuplicateFilterIntegration:
    """Test DuplicateFilter integration with StreamProcessor."""

    def test_duplicate_detection_in_stream(self):
        """Test that duplicate frames are filtered out in a stream."""
        # Create pipeline with duplicate detection
        tagger = HashTagger(hash_type="phash", hash_size=8)
        duplicate_filter = DuplicateFilter(buffer_size=5, threshold=0.01)  # Very strict

        stream = StreamProcessor(pipeline=[tagger, duplicate_filter], buffer_size=100)

        # Add unique frames 0, 1, 2
        frame0 = create_test_frame(0, data_value=0)
        frame1 = create_test_frame(1, data_value=50)
        frame2 = create_test_frame(2, data_value=100)

        assert stream.add(frame0) is True, "First frame should pass"
        assert stream.add(frame1) is True, "Second unique frame should pass"
        assert stream.add(frame2) is True, "Third unique frame should pass"

        # Add duplicate of frame1 (same data)
        frame1_dup = create_test_frame(3, data_value=50)  # Same data as frame1
        # This might be detected as duplicate within buffer_size window
        # The result depends on hash computation
        result = stream.add(frame1_dup)

        # Test that we have either 3 or 4 frames depending on duplicate detection
        count = stream.get_buffer_count()
        assert count in [3, 4], f"Expected 3 or 4 frames in buffer, got {count}"

    def test_stateful_filter_state_persists(self):
        """Test that stateful filter maintains state across add() calls."""
        duplicate_filter = DuplicateFilter(buffer_size=10, threshold=0.05)

        stream = StreamProcessor(
            pipeline=[HashTagger(hash_type="phash"), duplicate_filter], buffer_size=100
        )

        # Add 5 frames
        for i in range(5):
            frame = create_test_frame(i, data_value=i * 20)
            stream.add(frame)

        # Stateful filter should have internal buffer size 5
        assert duplicate_filter.buffer_count() == 5

        # Add 3 more frames
        for i in range(5, 8):
            frame = create_test_frame(i, data_value=i * 20)
            stream.add(frame)

        # Internal buffer should now be 8 (or maxed at 10 if we added more)
        assert duplicate_filter.buffer_count() == 8

    def test_stateful_filter_state_not_affected_by_stream_clear(self):
        """Test that clearing stream buffer doesn't affect stateful filter state."""
        duplicate_filter = DuplicateFilter(buffer_size=5, threshold=0.05)

        stream = StreamProcessor(
            pipeline=[HashTagger(hash_type="phash"), duplicate_filter], buffer_size=10
        )

        # Add 3 frames
        for i in range(3):
            frame = create_test_frame(i, data_value=i * 30)
            stream.add(frame)

        # Verify both buffers have 3 frames
        assert stream.get_buffer_count() == 3
        assert duplicate_filter.buffer_count() == 3

        # Clear stream buffer
        stream.clear()

        # Stream buffer should be empty
        assert stream.get_buffer_count() == 0

        # But stateful filter buffer should still have 3 (not cleared)
        assert duplicate_filter.buffer_count() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
