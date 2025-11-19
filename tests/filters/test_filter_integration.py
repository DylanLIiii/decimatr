"""
Integration tests for filters working with taggers.
"""

import datetime

import numpy as np
import pytest
from decimatr.filters.blur import BlurFilter
from decimatr.filters.entropy import EntropyFilter
from decimatr.filters.threshold import ThresholdFilter
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.entropy import EntropyTagger


class TestFilterTaggerIntegration:
    """Test filters working with taggers in a pipeline."""

    def test_blur_tagger_with_blur_filter(self):
        """Test BlurTagger output works with BlurFilter."""
        tagger = BlurTagger()
        filter = BlurFilter(threshold=100.0)

        # Create sharp frame (checkerboard pattern)
        sharp_frame = np.zeros((24, 24, 3), dtype=np.uint8)
        sharp_frame[::2, ::2] = 255

        sharp_packet = VideoFramePacket(
            frame_data=sharp_frame,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )

        # Tag the frame
        tags = tagger.compute_tags(sharp_packet)
        sharp_packet.tags.update(tags)

        # Sharp frame should pass
        assert filter.should_pass(sharp_packet) is True

        # Create blurry frame (solid color)
        blurry_frame = np.ones((24, 24, 3), dtype=np.uint8) * 128

        blurry_packet = VideoFramePacket(
            frame_data=blurry_frame,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )

        # Tag the frame
        tags = tagger.compute_tags(blurry_packet)
        blurry_packet.tags.update(tags)

        # Blurry frame should be filtered
        assert filter.should_pass(blurry_packet) is False

    def test_entropy_tagger_with_entropy_filter(self):
        """Test EntropyTagger output works with EntropyFilter."""
        tagger = EntropyTagger()
        filter = EntropyFilter(threshold=4.0)

        # Create high-entropy frame (random noise)
        np.random.seed(42)
        high_entropy_frame = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)

        high_entropy_packet = VideoFramePacket(
            frame_data=high_entropy_frame,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )

        # Tag the frame
        tags = tagger.compute_tags(high_entropy_packet)
        high_entropy_packet.tags.update(tags)

        # High entropy frame should pass
        assert filter.should_pass(high_entropy_packet) is True

        # Create low-entropy frame (solid color)
        low_entropy_frame = np.ones((24, 24, 3), dtype=np.uint8) * 128

        low_entropy_packet = VideoFramePacket(
            frame_data=low_entropy_frame,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )

        # Tag the frame
        tags = tagger.compute_tags(low_entropy_packet)
        low_entropy_packet.tags.update(tags)

        # Low entropy frame should be filtered
        assert filter.should_pass(low_entropy_packet) is False

    def test_multiple_taggers_and_filters(self):
        """Test pipeline with multiple taggers and filters."""
        blur_tagger = BlurTagger()
        entropy_tagger = EntropyTagger()
        blur_filter = BlurFilter(threshold=100.0)
        entropy_filter = EntropyFilter(threshold=4.0)

        # Create a good frame (sharp and high entropy)
        np.random.seed(42)
        good_frame = np.random.randint(0, 255, (24, 24, 3), dtype=np.uint8)
        # Add some structure to increase blur score
        good_frame[::2, ::2] = 255

        good_packet = VideoFramePacket(
            frame_data=good_frame,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )

        # Apply taggers
        good_packet.tags.update(blur_tagger.compute_tags(good_packet))
        good_packet.tags.update(entropy_tagger.compute_tags(good_packet))

        # Should pass both filters
        assert blur_filter.should_pass(good_packet) is True
        assert entropy_filter.should_pass(good_packet) is True

        # Create a bad frame (blurry and low entropy)
        bad_frame = np.ones((24, 24, 3), dtype=np.uint8) * 128

        bad_packet = VideoFramePacket(
            frame_data=bad_frame,
            frame_number=1,
            timestamp=datetime.timedelta(seconds=1),
            source_video_id="test",
        )

        # Apply taggers
        bad_packet.tags.update(blur_tagger.compute_tags(bad_packet))
        bad_packet.tags.update(entropy_tagger.compute_tags(bad_packet))

        # Should fail both filters
        assert blur_filter.should_pass(bad_packet) is False
        assert entropy_filter.should_pass(bad_packet) is False

    def test_threshold_filter_with_custom_tag(self):
        """Test ThresholdFilter with custom tag from tagger."""
        tagger = BlurTagger()
        # Use ThresholdFilter to filter blur_score
        filter = ThresholdFilter("blur_score", 100.0, ">=")

        # Create sharp frame
        sharp_frame = np.zeros((24, 24, 3), dtype=np.uint8)
        sharp_frame[::2, ::2] = 255

        packet = VideoFramePacket(
            frame_data=sharp_frame,
            frame_number=0,
            timestamp=datetime.timedelta(seconds=0),
            source_video_id="test",
        )

        # Tag and filter
        packet.tags.update(tagger.compute_tags(packet))

        # Should pass
        assert filter.should_pass(packet) is True
