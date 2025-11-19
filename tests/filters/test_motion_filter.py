"""
Tests for MotionFilter functionality.
"""

import datetime

import numpy as np
import pytest
from decimatr.filters.motion import MotionFilter
from decimatr.scheme import VideoFramePacket


class TestMotionFilter:
    """Test MotionFilter stateful filtering."""

    def test_first_frame_always_passes(self):
        """Test that the first frame always passes (empty buffer)."""
        filter = MotionFilter(buffer_size=5, threshold=30.0)

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )

        assert filter.should_pass(packet) is True
        assert filter.buffer_count() == 1

    def test_identical_frame_filtered(self):
        """Test that identical frames are filtered out (no motion)."""
        filter = MotionFilter(buffer_size=5, threshold=30.0, method="mean_diff")

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

        # Identical frame should be filtered (motion = 0)
        assert filter.should_pass(packet2) is False

    def test_different_frames_pass(self):
        """Test that frames with sufficient motion pass."""
        filter = MotionFilter(buffer_size=5, threshold=10.0, method="mean_diff")

        # Create different frames
        frame_data1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_data2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

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

        # Both frames should pass (high motion between them)
        assert filter.should_pass(packet1) is True
        assert filter.should_pass(packet2) is True

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            MotionFilter(method="invalid")

    def test_invalid_compare_with_raises_error(self):
        """Test that invalid compare_with raises ValueError."""
        with pytest.raises(ValueError, match="compare_with must be one of"):
            MotionFilter(compare_with="invalid")

    def test_required_tags_property(self):
        """Test that required_tags returns empty list."""
        filter = MotionFilter()

        # Motion filter works with frame_data directly
        assert filter.required_tags == []
