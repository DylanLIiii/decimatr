"""
Tests for DiversityFilter functionality.
"""

import datetime

import numpy as np
import pytest
from decimatr.filters.diversity import DiversityFilter
from decimatr.scheme import VideoFramePacket


class TestDiversityFilter:
    """Test DiversityFilter stateful filtering."""

    def test_first_frame_always_passes(self):
        """Test that the first frame always passes (empty buffer)."""
        filter = DiversityFilter(
            buffer_size=10, diversity_tags=["blur_score", "entropy"], min_distance=0.1
        )

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"blur_score": 100.0, "entropy": 5.0}

        assert filter.should_pass(packet) is True
        assert filter.buffer_count() == 1

    def test_similar_frame_filtered(self):
        """Test that frames with similar tags are filtered out."""
        filter = DiversityFilter(
            buffer_size=10, diversity_tags=["blur_score"], min_distance=10.0, metric="euclidean"
        )

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        # First frame
        packet1 = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet1.tags = {"blur_score": 100.0}

        # Similar frame (difference = 5.0, below threshold of 10.0)
        packet2 = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet2.tags = {"blur_score": 105.0}

        assert filter.should_pass(packet1) is True
        assert filter.should_pass(packet2) is False

    def test_diverse_frame_passes(self):
        """Test that frames with diverse tags pass."""
        filter = DiversityFilter(
            buffer_size=10, diversity_tags=["blur_score"], min_distance=10.0, metric="euclidean"
        )

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        # First frame
        packet1 = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet1.tags = {"blur_score": 100.0}

        # Diverse frame (difference = 50.0, above threshold of 10.0)
        packet2 = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet2.tags = {"blur_score": 150.0}

        assert filter.should_pass(packet1) is True
        assert filter.should_pass(packet2) is True
        assert filter.buffer_count() == 2

    def test_missing_tags_filtered(self):
        """Test that frames with missing tags are filtered."""
        filter = DiversityFilter(
            buffer_size=10, diversity_tags=["blur_score", "entropy"], min_distance=0.1
        )

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        # First frame with all tags
        packet1 = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet1.tags = {"blur_score": 100.0, "entropy": 5.0}

        # Second frame missing entropy tag
        packet2 = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet2.tags = {"blur_score": 150.0}

        assert filter.should_pass(packet1) is True
        assert filter.should_pass(packet2) is False

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            DiversityFilter(metric="invalid")

    def test_negative_min_distance_raises_error(self):
        """Test that negative min_distance raises ValueError."""
        with pytest.raises(ValueError, match="min_distance must be non-negative"):
            DiversityFilter(min_distance=-1.0)

    def test_required_tags_property(self):
        """Test that required_tags returns configured tags."""
        filter = DiversityFilter(diversity_tags=["blur_score", "entropy"])

        assert filter.required_tags == ["blur_score", "entropy"]

    def test_auto_detect_tags(self):
        """Test that filter auto-detects numeric tags when none specified."""
        filter = DiversityFilter(buffer_size=10, min_distance=0.1)

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        # Frame with multiple numeric tags
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {
            "blur_score": 100.0,
            "entropy": 5.0,
            "name": "test",  # Non-numeric, should be ignored
        }

        # Should pass (first frame)
        assert filter.should_pass(packet) is True
