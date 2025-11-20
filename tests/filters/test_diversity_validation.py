"""
Tests for DiversityFilter validation and warning functionality.
"""

import datetime
import io
import sys

import numpy as np
import pytest
from decimatr.filters.comparison_strategies import (
    ComparisonStrategy,
    EmbeddingDistanceStrategy,
    HammingDistanceStrategy,
)
from decimatr.filters.diversity import DiversityFilter
from decimatr.scheme import VideoFramePacket


class InvalidStrategy:
    """Not a ComparisonStrategy subclass."""

    pass


class TestDiversityFilterValidation:
    """Test DiversityFilter validation functionality."""

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            DiversityFilter(metric="invalid_metric")

    def test_negative_min_distance_raises_error(self):
        """Test that negative min_distance raises ValueError."""
        with pytest.raises(ValueError, match="min_distance must be non-negative"):
            DiversityFilter(min_distance=-1.0)

    def test_invalid_comparison_strategy_raises_error(self):
        """Test that invalid comparison strategy raises ValueError."""
        with pytest.raises(ValueError, match="must be a ComparisonStrategy instance"):
            DiversityFilter(comparison_strategies={"dhash": InvalidStrategy()})

    def test_valid_comparison_strategy_accepted(self):
        """Test that valid comparison strategies are accepted."""
        # Should not raise any errors
        filter = DiversityFilter(
            comparison_strategies={
                "dhash": HammingDistanceStrategy(),
                "clip_embedding": EmbeddingDistanceStrategy(metric="cosine"),
            }
        )
        assert filter is not None

    def test_behavior_with_no_diversity_tags(self):
        """Test that frames pass when no diversity tags are available."""
        filter = DiversityFilter(buffer_size=10, min_distance=0.1)

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        # Frame with only metric-only tags (no diversity-suitable tags)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet.tags = {"blur_score": 100.0, "entropy": 5.0}

        # Should pass because there are no diversity tags to compare
        # (warning will be logged to stderr)
        assert filter.should_pass(packet) is True

    def test_warning_on_missing_specified_tags(self):
        """Test that frames with missing tags are filtered out."""
        filter = DiversityFilter(
            buffer_size=10, diversity_tags=["dhash", "clip_embedding"], min_distance=0.1
        )

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        # First frame with all tags
        packet1 = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )
        packet1.tags = {"dhash": "abc123", "clip_embedding": [0.1, 0.2, 0.3]}

        # Second frame missing clip_embedding tag
        packet2 = VideoFramePacket(
            frame_data=frame_data,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )
        packet2.tags = {"dhash": "def456"}

        assert filter.should_pass(packet1) is True
        # Frame with missing tag should be filtered out (warning will be logged)
        assert filter.should_pass(packet2) is False

    def test_multiple_frames_with_no_diversity_tags(self):
        """Test that multiple frames pass when no diversity tags are available."""
        filter = DiversityFilter(buffer_size=10, min_distance=0.1)

        frame_data = np.zeros((100, 100, 3), dtype=np.uint8)

        # Process multiple frames with no diversity tags
        # All should pass (warning logged only once)
        for i in range(3):
            packet = VideoFramePacket(
                frame_data=frame_data,
                frame_number=i,
                timestamp=datetime.timedelta(seconds=i),
                source_video_id="test",
            )
            packet.tags = {"blur_score": 100.0}
            assert filter.should_pass(packet) is True

    def test_all_valid_metrics_accepted(self):
        """Test that all valid metrics are accepted."""
        valid_metrics = ["euclidean", "manhattan", "cosine"]

        for metric in valid_metrics:
            # Should not raise any errors
            filter = DiversityFilter(metric=metric)
            assert filter.legacy_metric == metric

    def test_zero_min_distance_accepted(self):
        """Test that zero min_distance is accepted."""
        # Should not raise any errors
        filter = DiversityFilter(min_distance=0.0)
        assert filter.min_distance == 0.0
