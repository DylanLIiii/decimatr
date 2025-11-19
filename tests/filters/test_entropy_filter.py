"""
Tests for EntropyFilter functionality.
"""

import datetime

import numpy as np
import pytest
from decimatr.filters.entropy import EntropyFilter
from decimatr.scheme import VideoFramePacket


class TestEntropyFilter:
    """Test EntropyFilter stateless filtering."""

    def test_frame_above_threshold_passes(self):
        """Test that frames with entropy >= threshold pass."""
        filter = EntropyFilter(threshold=4.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"entropy": 5.2}

        assert filter.should_pass(packet) is True

    def test_frame_at_threshold_passes(self):
        """Test that frames with entropy == threshold pass."""
        filter = EntropyFilter(threshold=4.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"entropy": 4.0}

        assert filter.should_pass(packet) is True

    def test_frame_below_threshold_filtered(self):
        """Test that frames with entropy < threshold are filtered out."""
        filter = EntropyFilter(threshold=4.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"entropy": 2.1}

        assert filter.should_pass(packet) is False

    def test_missing_entropy_filtered(self):
        """Test that frames without entropy tag are filtered out."""
        filter = EntropyFilter(threshold=4.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        # No entropy tag

        assert filter.should_pass(packet) is False

    def test_default_threshold(self):
        """Test that default threshold is 4.0."""
        filter = EntropyFilter()

        assert filter.threshold == 4.0

    def test_custom_threshold(self):
        """Test that custom threshold is applied correctly."""
        filter = EntropyFilter(threshold=6.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)

        # Frame with entropy 5.0 should be filtered with threshold 6.0
        packet_fail = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet_fail.tags = {"entropy": 5.0}
        assert filter.should_pass(packet_fail) is False

        # Frame with entropy 7.0 should pass with threshold 6.0
        packet_pass = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet_pass.tags = {"entropy": 7.0}
        assert filter.should_pass(packet_pass) is True

    def test_required_tags_property(self):
        """Test that required_tags returns ['entropy']."""
        filter = EntropyFilter()

        assert filter.required_tags == ["entropy"]

    def test_negative_threshold_raises_error(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            EntropyFilter(threshold=-1.0)

    def test_zero_threshold(self):
        """Test that zero threshold is valid."""
        filter = EntropyFilter(threshold=0.0)

        frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"entropy": 0.0}

        assert filter.should_pass(packet) is True

    def test_repr(self):
        """Test string representation."""
        filter = EntropyFilter(threshold=5.5)

        repr_str = repr(filter)
        assert "EntropyFilter" in repr_str
        assert "5.5" in repr_str
