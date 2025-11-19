"""
Tests for BlurFilter functionality.
"""

import datetime

import numpy as np
import pytest
from decimatr.filters.blur import BlurFilter
from decimatr.scheme import VideoFramePacket


class TestBlurFilter:
    """Test BlurFilter stateless filtering."""

    def test_frame_above_threshold_passes(self):
        """Test that frames with blur_score >= threshold pass."""
        filter = BlurFilter(threshold=100.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"blur_score": 150.0}

        assert filter.should_pass(packet) is True

    def test_frame_at_threshold_passes(self):
        """Test that frames with blur_score == threshold pass."""
        filter = BlurFilter(threshold=100.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"blur_score": 100.0}

        assert filter.should_pass(packet) is True

    def test_frame_below_threshold_filtered(self):
        """Test that frames with blur_score < threshold are filtered out."""
        filter = BlurFilter(threshold=100.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"blur_score": 50.0}

        assert filter.should_pass(packet) is False

    def test_missing_blur_score_filtered(self):
        """Test that frames without blur_score tag are filtered out."""
        filter = BlurFilter(threshold=100.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        # No blur_score tag

        assert filter.should_pass(packet) is False

    def test_default_threshold(self):
        """Test that default threshold is 100.0."""
        filter = BlurFilter()

        assert filter.threshold == 100.0

    def test_custom_threshold(self):
        """Test that custom threshold is applied correctly."""
        filter = BlurFilter(threshold=200.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)

        # Frame with blur_score 150.0 should be filtered with threshold 200.0
        packet_fail = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet_fail.tags = {"blur_score": 150.0}
        assert filter.should_pass(packet_fail) is False

        # Frame with blur_score 250.0 should pass with threshold 200.0
        packet_pass = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet_pass.tags = {"blur_score": 250.0}
        assert filter.should_pass(packet_pass) is True

    def test_required_tags_property(self):
        """Test that required_tags returns ['blur_score']."""
        filter = BlurFilter()

        assert filter.required_tags == ["blur_score"]

    def test_negative_threshold_raises_error(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            BlurFilter(threshold=-10.0)

    def test_zero_threshold(self):
        """Test that zero threshold is valid."""
        filter = BlurFilter(threshold=0.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"blur_score": 0.0}

        assert filter.should_pass(packet) is True

    def test_repr(self):
        """Test string representation."""
        filter = BlurFilter(threshold=150.0)

        repr_str = repr(filter)
        assert "BlurFilter" in repr_str
        assert "150.0" in repr_str
