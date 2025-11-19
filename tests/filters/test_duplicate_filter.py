"""
Tests for DuplicateFilter functionality.
"""

import datetime

import numpy as np
import pytest
from decimatr.filters.duplicate import DuplicateFilter
from decimatr.scheme import VideoFramePacket


class TestDuplicateFilter:
    """Test DuplicateFilter stateful filtering."""

    def test_first_frame_always_passes(self):
        """Test that the first frame always passes (empty buffer)."""
        filter = DuplicateFilter(buffer_size=10, threshold=0.05)

        frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )

        assert filter.should_pass(packet) is True
        assert filter.buffer_count() == 1

    def test_identical_frame_filtered(self):
        """Test that identical frames are filtered out."""
        filter = DuplicateFilter(buffer_size=10, threshold=0.05)

        # Create identical frames
        frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        packet1 = VideoFramePacket(
            frame_data=frame_data.copy(),
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )

        packet2 = VideoFramePacket(
            frame_data=frame_data.copy(),
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )

        # First frame passes
        assert filter.should_pass(packet1) is True

        # Identical frame should be filtered
        assert filter.should_pass(packet2) is False

    def test_different_frames_pass(self):
        """Test that sufficiently different frames pass."""
        filter = DuplicateFilter(buffer_size=10, threshold=0.05)

        # Create different frames
        frame_data1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame_data2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        packet1 = VideoFramePacket(
            frame_data=frame_data1,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )

        packet2 = VideoFramePacket(
            frame_data=frame_data2,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )

        # Both frames should pass (very unlikely to have similar hashes)
        assert filter.should_pass(packet1) is True
        assert filter.should_pass(packet2) is True
        assert filter.buffer_count() == 2

    def test_buffer_size_limit(self):
        """Test that buffer respects size limit."""
        filter = DuplicateFilter(buffer_size=3, threshold=0.05)

        # Add 5 different frames
        for i in range(5):
            frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            packet = VideoFramePacket(
                frame_data=frame_data,
                frame_number=i,
                timestamp=datetime.timedelta(seconds=i),
                source_video_id="test",
            )
            filter.should_pass(packet)

        # Buffer should only contain last 3 frames
        assert filter.buffer_count() == 3

    def test_invalid_threshold_raises_error(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            DuplicateFilter(threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be between"):
            DuplicateFilter(threshold=-0.1)

    def test_required_tags_property(self):
        """Test that required_tags returns empty list."""
        filter = DuplicateFilter()

        # Returns empty because filter can compute hash if needed
        assert filter.required_tags == []
