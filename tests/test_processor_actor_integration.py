"""
Tests for FrameProcessor integration with ActorPipeline.

This module tests the integration between FrameProcessor and ActorPipeline,
verifying that parallel processing works correctly when n_workers > 1.
"""

from datetime import timedelta

import numpy as np
import pytest
from decimatr.core.processor import FrameProcessor
from decimatr.filters.blur import BlurFilter
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.blur import BlurTagger


def create_test_frame(frame_number: int, blur_level: str = "sharp") -> VideoFramePacket:
    """
    Create a test frame with specified blur characteristics.

    Args:
        frame_number: Frame number
        blur_level: "sharp" or "blurry"

    Returns:
        VideoFramePacket with appropriate blur characteristics
    """
    if blur_level == "sharp":
        # Create a sharp image with high-frequency content
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Add some edges
        frame[40:60, :] = 255
        frame[:, 40:60] = 0
    else:
        # Create a blurry image (uniform color)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

    return VideoFramePacket(
        frame_data=frame,
        frame_number=frame_number,
        timestamp=timedelta(seconds=frame_number / 30.0),
        source_video_id="test_video",
        tags={},
        additional_metadata={},
    )


def test_single_threaded_processing():
    """Test that single-threaded processing works (n_workers=1)."""
    # Create processor with n_workers=1
    pipeline = [BlurTagger(), BlurFilter(threshold=50.0)]
    processor = FrameProcessor(pipeline=pipeline, n_workers=1)

    # Create test frames
    frames = [
        create_test_frame(0, "sharp"),
        create_test_frame(1, "blurry"),
        create_test_frame(2, "sharp"),
    ]

    # Process frames
    results = list(processor.process(frames))

    # Should filter out blurry frame
    assert len(results) == 2
    assert results[0].frame_number == 0
    assert results[1].frame_number == 2


def test_actor_based_processing():
    """Test that actor-based processing works (n_workers > 1)."""
    # Create processor with n_workers=2
    pipeline = [BlurTagger(), BlurFilter(threshold=50.0)]
    processor = FrameProcessor(pipeline=pipeline, n_workers=2)

    # Create test frames
    frames = [
        create_test_frame(0, "sharp"),
        create_test_frame(1, "blurry"),
        create_test_frame(2, "sharp"),
        create_test_frame(3, "blurry"),
        create_test_frame(4, "sharp"),
    ]

    # Process frames
    results = list(processor.process(frames))

    # Should filter out blurry frames
    assert len(results) == 3
    assert results[0].frame_number == 0
    assert results[1].frame_number == 2
    assert results[2].frame_number == 4


def test_actor_processing_with_empty_pipeline():
    """Test that actor processing handles empty pipeline correctly."""
    # Create processor with empty pipeline and n_workers > 1
    processor = FrameProcessor(pipeline=[], n_workers=2)

    # Create test frames
    frames = [
        create_test_frame(0, "sharp"),
        create_test_frame(1, "blurry"),
    ]

    # Process frames - should pass all frames (no filtering)
    results = list(processor.process(frames))

    # All frames should pass
    assert len(results) == 2


def test_processing_mode_selection():
    """Test that correct processing mode is selected based on n_workers."""
    pipeline = [BlurTagger(), BlurFilter(threshold=50.0)]

    # n_workers=1 should use single-threaded
    processor_single = FrameProcessor(pipeline=pipeline, n_workers=1)
    frames = [create_test_frame(0, "sharp")]

    # Should work without actor pipeline
    results = list(processor_single.process(frames))
    assert len(results) == 1

    # n_workers>1 should use actor-based
    processor_multi = FrameProcessor(pipeline=pipeline, n_workers=2)

    # Should work with actor pipeline
    results = list(processor_multi.process(frames))
    assert len(results) == 1


def test_actor_processing_with_result():
    """Test that actor processing returns ProcessingResult correctly."""
    pipeline = [BlurTagger(), BlurFilter(threshold=50.0)]
    processor = FrameProcessor(pipeline=pipeline, n_workers=2)

    # Create test frames
    frames = [
        create_test_frame(0, "sharp"),
        create_test_frame(1, "blurry"),
        create_test_frame(2, "sharp"),
    ]

    # Process frames with result
    frame_iter, result = processor.process(frames, return_result=True)
    results = list(frame_iter)

    # Check results
    assert len(results) == 2
    assert result.total_frames == 3
    assert result.selected_frames == 2
    assert result.filtered_frames == 1


def test_builder_methods_with_parallel_processing():
    """Test that builder methods work with parallel processing."""
    # Test with_blur_removal
    processor = FrameProcessor.with_blur_removal(threshold=50.0, n_workers=2)

    frames = [
        create_test_frame(0, "sharp"),
        create_test_frame(1, "blurry"),
    ]

    results = list(processor.process(frames))
    assert len(results) == 1
    assert results[0].frame_number == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
