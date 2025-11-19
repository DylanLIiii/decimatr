"""
Tests for BlurTagger functionality.
"""

import datetime

import cv2
import numpy as np
import pytest
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.blur import BlurTagger


class TestBlurTagger:
    """Test BlurTagger stateless frame analysis."""

    def test_compute_tags_returns_blur_score(self):
        """Test that compute_tags returns a dictionary with blur_score."""
        tagger = BlurTagger()

        # Create a simple test frame
        frame_data = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        assert "blur_score" in tags
        assert isinstance(tags["blur_score"], float)
        assert tags["blur_score"] >= 0

    def test_tag_keys_property(self):
        """Test that tag_keys returns the correct list."""
        tagger = BlurTagger()

        assert tagger.tag_keys == ["blur_score"]

    def test_sharp_frame_high_score(self):
        """Test that sharp frames (checkerboard) have high blur scores."""
        tagger = BlurTagger()

        # Create checkerboard pattern (high variance)
        frame_data = np.zeros((24, 24, 3), dtype=np.uint8)
        frame_data[::2, ::2] = 255

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        # Checkerboard should have high blur score
        assert tags["blur_score"] > 100

    def test_blurry_frame_low_score(self):
        """Test that blurry frames (solid color) have low blur scores."""
        tagger = BlurTagger()

        # Create solid color frame (low variance)
        frame_data = np.ones((24, 24, 3), dtype=np.uint8) * 128

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        # Solid color should have very low blur score (near 0)
        assert tags["blur_score"] < 1.0

    def test_empty_frame_handling(self):
        """Test that empty frames return 0 blur score."""
        tagger = BlurTagger()

        # Create empty frame
        frame_data = np.zeros((0, 0, 3), dtype=np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        assert tags["blur_score"] == 0.0

    def test_grayscale_frame_handling(self):
        """Test that grayscale frames are handled correctly."""
        tagger = BlurTagger()

        # Create grayscale frame with pattern
        frame_data = np.zeros((24, 24), dtype=np.uint8)
        frame_data[::2, ::2] = 255

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        assert "blur_score" in tags
        assert tags["blur_score"] > 0

    def test_stateless_operation(self):
        """Test that tagger is stateless (same input produces same output)."""
        tagger = BlurTagger()

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
        assert tags1["blur_score"] == tags2["blur_score"]
        assert tags2["blur_score"] == tags3["blur_score"]

    def test_gpu_properties(self):
        """Test that BlurTagger correctly reports GPU capabilities."""
        tagger = BlurTagger()

        # BlurTagger is CPU-based
        assert tagger.supports_gpu is False
        assert tagger.requires_gpu is False
        assert tagger.is_cloud_based is False
