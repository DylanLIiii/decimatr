"""
Tests for HashTagger functionality.
"""

import datetime

import imagehash
import numpy as np
import pytest
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.hash import HashTagger


class TestHashTagger:
    """Test HashTagger stateless frame analysis."""

    def test_initialization_default(self):
        """Test HashTagger initializes with default settings."""
        tagger = HashTagger()

        assert tagger.hash_type == "phash"
        assert tagger.hash_size == 8
        assert tagger.highfreq_factor == 4

    def test_initialization_custom(self):
        """Test HashTagger initializes with custom settings."""
        tagger = HashTagger(hash_type="dhash", hash_size=16, highfreq_factor=8)

        assert tagger.hash_type == "dhash"
        assert tagger.hash_size == 16
        assert tagger.highfreq_factor == 8

    def test_invalid_hash_type_raises_error(self):
        """Test that invalid hash type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid hash_type"):
            HashTagger(hash_type="invalid_hash")

    def test_compute_tags_returns_hash_tags(self):
        """Test that compute_tags returns hash tags."""
        tagger = HashTagger(hash_type="phash")

        # Create a simple test frame
        frame_data = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)

        # Should contain both hash type key and hash_value
        assert "phash" in tags
        assert "hash_value" in tags
        assert isinstance(tags["phash"], str)
        assert isinstance(tags["hash_value"], imagehash.ImageHash)

    def test_tag_keys_property(self):
        """Test that tag_keys returns the correct list."""
        tagger = HashTagger(hash_type="phash")

        assert tagger.tag_keys == ["phash", "hash_value"]

    def test_tag_keys_different_hash_types(self):
        """Test that tag_keys reflects the configured hash type."""
        phash_tagger = HashTagger(hash_type="phash")
        dhash_tagger = HashTagger(hash_type="dhash")
        ahash_tagger = HashTagger(hash_type="ahash")

        assert phash_tagger.tag_keys == ["phash", "hash_value"]
        assert dhash_tagger.tag_keys == ["dhash", "hash_value"]
        assert ahash_tagger.tag_keys == ["ahash", "hash_value"]

    def test_identical_frames_same_hash(self):
        """Test that identical frames produce the same hash."""
        tagger = HashTagger(hash_type="phash")

        # Create identical frames
        frame_data = np.ones((24, 24, 3), dtype=np.uint8) * 128

        packet1 = VideoFramePacket(
            frame_data=frame_data.copy(),
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        packet2 = VideoFramePacket(
            frame_data=frame_data.copy(),
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test_video",
        )

        tags1 = tagger.compute_tags(packet1)
        tags2 = tagger.compute_tags(packet2)

        # Hashes should be identical
        assert tags1["phash"] == tags2["phash"]
        assert tags1["hash_value"] == tags2["hash_value"]

    def test_different_frames_different_hash(
        self, create_video_frame_packet, very_different_frames
    ):
        """Test that very different frames produce different hashes."""
        tagger = HashTagger(hash_type="phash")

        frame1_data, frame2_data = very_different_frames

        packet1 = create_video_frame_packet(frame_data=frame1_data, frame_number=0)
        packet2 = create_video_frame_packet(frame_data=frame2_data, frame_number=1)

        tags1 = tagger.compute_tags(packet1)
        tags2 = tagger.compute_tags(packet2)

        # Hashes should be different
        assert tags1["phash"] != tags2["phash"]
        assert tags1["hash_value"] != tags2["hash_value"]

    def test_different_hash_types_produce_different_hashes(self):
        """Test that different hash types produce different hash values."""
        phash_tagger = HashTagger(hash_type="phash")
        dhash_tagger = HashTagger(hash_type="dhash")
        ahash_tagger = HashTagger(hash_type="ahash")

        # Create test frame
        frame_data = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        phash_tags = phash_tagger.compute_tags(packet)
        dhash_tags = dhash_tagger.compute_tags(packet)
        ahash_tags = ahash_tagger.compute_tags(packet)

        # Different hash types should produce different hash strings
        assert phash_tags["phash"] != dhash_tags["dhash"]
        assert phash_tags["phash"] != ahash_tags["ahash"]
        assert dhash_tags["dhash"] != ahash_tags["ahash"]

    def test_stateless_operation(self):
        """Test that tagger is stateless (same input produces same output)."""
        tagger = HashTagger(hash_type="phash")

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
        assert tags1["phash"] == tags2["phash"]
        assert tags2["phash"] == tags3["phash"]
        assert tags1["hash_value"] == tags2["hash_value"]
        assert tags2["hash_value"] == tags3["hash_value"]

    def test_hash_value_can_be_compared(self, create_video_frame_packet, slightly_different_frames):
        """Test that hash_value objects can be compared for similarity."""
        tagger = HashTagger(hash_type="phash")

        base_frame, similar_frame = slightly_different_frames

        packet1 = create_video_frame_packet(frame_data=base_frame, frame_number=0)
        packet2 = create_video_frame_packet(frame_data=similar_frame, frame_number=1)

        tags1 = tagger.compute_tags(packet1)
        tags2 = tagger.compute_tags(packet2)

        # Hash values should be comparable (difference should be a number)
        hash_diff = tags1["hash_value"] - tags2["hash_value"]
        assert isinstance(hash_diff, int | float | np.integer)

        # Slightly different frames should have small hash difference
        assert hash_diff < 20  # Reasonable threshold for slightly different frames

    def test_gpu_properties(self):
        """Test that HashTagger correctly reports GPU capabilities."""
        tagger = HashTagger()

        # HashTagger is CPU-based
        assert tagger.supports_gpu is False
        assert tagger.requires_gpu is False
        assert tagger.is_cloud_based is False

    def test_all_supported_hash_types(self):
        """Test that all supported hash types work correctly."""
        hash_types = ["ahash", "phash", "dhash", "whash", "colorhash"]

        frame_data = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        for hash_type in hash_types:
            tagger = HashTagger(hash_type=hash_type)
            tags = tagger.compute_tags(packet)

            # Should contain the hash type key and hash_value
            assert hash_type in tags
            assert "hash_value" in tags
            assert isinstance(tags[hash_type], str)
            assert isinstance(tags["hash_value"], imagehash.ImageHash)

    def test_hash_string_representation(self):
        """Test that hash string representation is valid."""
        tagger = HashTagger(hash_type="phash")

        frame_data = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test_video",
        )

        tags = tagger.compute_tags(packet)
        hash_str = tags["phash"]

        # Hash string should be non-empty and hexadecimal
        assert len(hash_str) > 0
        # Should be valid hex string (will raise ValueError if not)
        int(hash_str, 16)
