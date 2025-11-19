"""
Tests for ThresholdFilter functionality.
"""

import datetime

import numpy as np
import pytest
from decimatr.filters.threshold import ThresholdFilter
from decimatr.scheme import VideoFramePacket


class TestThresholdFilter:
    """Test ThresholdFilter stateless filtering."""

    def test_greater_than_operator(self):
        """Test '>' operator filters correctly."""
        filter = ThresholdFilter("test_value", 100.0, ">")

        # Create test packets
        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)

        # Value above threshold should pass
        packet_pass = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet_pass.tags = {"test_value": 150.0}
        assert filter.should_pass(packet_pass) is True

        # Value equal to threshold should not pass
        packet_equal = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet_equal.tags = {"test_value": 100.0}
        assert filter.should_pass(packet_equal) is False

        # Value below threshold should not pass
        packet_fail = VideoFramePacket(
            frame_data=frame_data,
            frame_number=2,
            timestamp=datetime.timedelta(seconds=2),
            source_video_id="test",
        )
        packet_fail.tags = {"test_value": 50.0}
        assert filter.should_pass(packet_fail) is False

    def test_greater_than_or_equal_operator(self):
        """Test '>=' operator filters correctly."""
        filter = ThresholdFilter("test_value", 100.0, ">=")

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)

        # Value above threshold should pass
        packet_above = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet_above.tags = {"test_value": 150.0}
        assert filter.should_pass(packet_above) is True

        # Value equal to threshold should pass
        packet_equal = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet_equal.tags = {"test_value": 100.0}
        assert filter.should_pass(packet_equal) is True

        # Value below threshold should not pass
        packet_below = VideoFramePacket(
            frame_data=frame_data,
            frame_number=2,
            timestamp=datetime.timedelta(seconds=2),
            source_video_id="test",
        )
        packet_below.tags = {"test_value": 50.0}
        assert filter.should_pass(packet_below) is False

    def test_less_than_operator(self):
        """Test '<' operator filters correctly."""
        filter = ThresholdFilter("test_value", 100.0, "<")

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)

        # Value below threshold should pass
        packet_pass = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet_pass.tags = {"test_value": 50.0}
        assert filter.should_pass(packet_pass) is True

        # Value equal to threshold should not pass
        packet_equal = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet_equal.tags = {"test_value": 100.0}
        assert filter.should_pass(packet_equal) is False

        # Value above threshold should not pass
        packet_fail = VideoFramePacket(
            frame_data=frame_data,
            frame_number=2,
            timestamp=datetime.timedelta(seconds=2),
            source_video_id="test",
        )
        packet_fail.tags = {"test_value": 150.0}
        assert filter.should_pass(packet_fail) is False

    def test_less_than_or_equal_operator(self):
        """Test '<=' operator filters correctly."""
        filter = ThresholdFilter("test_value", 100.0, "<=")

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)

        # Value below threshold should pass
        packet_below = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet_below.tags = {"test_value": 50.0}
        assert filter.should_pass(packet_below) is True

        # Value equal to threshold should pass
        packet_equal = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet_equal.tags = {"test_value": 100.0}
        assert filter.should_pass(packet_equal) is True

        # Value above threshold should not pass
        packet_above = VideoFramePacket(
            frame_data=frame_data,
            frame_number=2,
            timestamp=datetime.timedelta(seconds=2),
            source_video_id="test",
        )
        packet_above.tags = {"test_value": 150.0}
        assert filter.should_pass(packet_above) is False

    def test_equal_operator(self):
        """Test '==' operator filters correctly."""
        filter = ThresholdFilter("test_value", 100.0, "==")

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)

        # Value equal to threshold should pass
        packet_equal = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet_equal.tags = {"test_value": 100.0}
        assert filter.should_pass(packet_equal) is True

        # Value not equal should not pass
        packet_not_equal = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet_not_equal.tags = {"test_value": 99.9}
        assert filter.should_pass(packet_not_equal) is False

    def test_not_equal_operator(self):
        """Test '!=' operator filters correctly."""
        filter = ThresholdFilter("test_value", 100.0, "!=")

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)

        # Value not equal to threshold should pass
        packet_not_equal = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet_not_equal.tags = {"test_value": 99.9}
        assert filter.should_pass(packet_not_equal) is True

        # Value equal should not pass
        packet_equal = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet_equal.tags = {"test_value": 100.0}
        assert filter.should_pass(packet_equal) is False

    def test_missing_tag_returns_false(self):
        """Test that missing tag causes frame to be filtered out."""
        filter = ThresholdFilter("test_value", 100.0, ">")

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        # No tags set

        assert filter.should_pass(packet) is False

    def test_required_tags_property(self):
        """Test that required_tags returns correct tag key."""
        filter = ThresholdFilter("my_tag", 50.0, ">")

        assert filter.required_tags == ["my_tag"]

    def test_invalid_operator_raises_error(self):
        """Test that invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="operator must be one of"):
            ThresholdFilter("test_value", 100.0, "invalid")

    def test_repr(self):
        """Test string representation."""
        filter = ThresholdFilter("blur_score", 100.0, ">=")

        repr_str = repr(filter)
        assert "ThresholdFilter" in repr_str
        assert "blur_score" in repr_str
        assert "100.0" in repr_str
        assert ">=" in repr_str
