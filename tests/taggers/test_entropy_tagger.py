"""
Tests for EntropyTagger functionality.
"""

import datetime

import numpy as np
import pytest
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.entropy import EntropyTagger


class TestEntropyTagger:
    """Test EntropyTagger stateless frame analysis."""

    def test_compute_tags_returns_entropy(self):
        """Test that compute_tags returns a dictionary with entropy."""
        tagger = EntropyTagger()

        # Create a simple test frame
        frame_data = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        assert "entropy" in tags
        assert isinstance(tags["entropy"], float)
        assert tags["entropy"] >= 0

    def test_tag_keys_property(self):
        """Test that tag_keys returns the correct list."""
        tagger = EntropyTagger()

        assert tagger.tag_keys == ["entropy"]

    def test_high_entropy_random_frame(self):
        """Test that random frames have high entropy."""
        tagger = EntropyTagger()

        # Create random frame (high entropy - many different pixel values)
        np.random.seed(42)
        frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        # Random frame should have high entropy (close to 8 for 8-bit images)
        assert tags["entropy"] > 6.0

    def test_low_entropy_uniform_frame(self):
        """Test that uniform frames have low entropy."""
        tagger = EntropyTagger()

        # Create solid color frame (low entropy - single pixel value)
        frame_data = np.ones((100, 100, 3), dtype=np.uint8) * 128

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        # Solid color should have very low entropy (near 0)
        assert tags["entropy"] < 0.1

    def test_empty_frame_handling(self):
        """Test that empty frames return 0 entropy."""
        tagger = EntropyTagger()

        # Create empty frame
        frame_data = np.zeros((0, 0, 3), dtype=np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        assert tags["entropy"] == 0.0

    def test_grayscale_frame_handling(self):
        """Test that grayscale frames are handled correctly."""
        tagger = EntropyTagger()

        # Create grayscale frame with random values
        np.random.seed(42)
        frame_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        assert "entropy" in tags
        assert tags["entropy"] > 0

    def test_stateless_operation(self):
        """Test that tagger is stateless (same input produces same output)."""
        tagger = EntropyTagger()

        # Create test frame
        np.random.seed(42)
        frame_data = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        # Compute tags multiple times
        tags1 = tagger.compute_tags(packet)
        tags2 = tagger.compute_tags(packet)
        tags3 = tagger.compute_tags(packet)

        # All results should be identical
        assert tags1["entropy"] == tags2["entropy"]
        assert tags2["entropy"] == tags3["entropy"]

    def test_gpu_properties(self):
        """Test that EntropyTagger correctly reports GPU capabilities."""
        tagger = EntropyTagger()

        # EntropyTagger is CPU-based
        assert tagger.supports_gpu is False
        assert tagger.requires_gpu is False
        assert tagger.is_cloud_based is False

    def test_entropy_range(self):
        """Test that entropy values are within expected range for 8-bit images."""
        tagger = EntropyTagger()

        # Create various test frames
        test_frames = [
            np.zeros((50, 50, 3), dtype=np.uint8),  # All black
            np.ones((50, 50, 3), dtype=np.uint8) * 255,  # All white
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),  # Random
        ]

        for frame_data in test_frames:
            packet = VideoFramePacket(
                frame_data=frame_data,
                frame_number=0,
                timestamp=datetime.timedelta(seconds=0),
                source_video_id="test_video",
            )

            tags = tagger.compute_tags(packet)

            # Entropy for 8-bit images should be between 0 and 8
            # Allow small negative values due to floating point precision
            assert -1e-10 <= tags["entropy"] <= 8.0
