"""
Demonstration of the FrameProcessor API.

This example shows how to use the FrameProcessor to process video frames
through various filtering strategies.
"""

import datetime

import numpy as np
from decimatr.core.processor import FrameProcessor
from decimatr.filters.blur import BlurFilter
from decimatr.filters.duplicate import DuplicateFilter
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger


def create_sample_frames(count: int = 20) -> list:
    """Create sample frames for demonstration."""
    frames = []
    for i in range(count):
        # Alternate between sharp and blurry frames
        if i % 3 == 0:
            # Blurry frame (uniform color)
            frame_data = np.ones((100, 100, 3), dtype=np.uint8) * 128
        else:
            # Sharp frame (random noise)
            frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=i,
            timestamp=datetime.timedelta(seconds=i * 0.033),  # ~30fps
            source_video_id="demo_video",
        )
        frames.append(packet)

    return frames


def example_1_blur_removal():
    """Example 1: Simple blur removal using builder method."""
    print("\n" + "=" * 60)
    print("Example 1: Blur Removal")
    print("=" * 60)

    # Create sample frames
    frames = create_sample_frames(20)
    print(f"Created {len(frames)} sample frames")

    # Create processor with blur removal
    processor = FrameProcessor.with_blur_removal(threshold=100.0)

    # Process frames
    selected_frames = list(processor.process(frames))

    print(f"Selected {len(selected_frames)} frames after blur removal")
    print(f"Filtered out {len(frames) - len(selected_frames)} blurry frames")


def example_2_custom_pipeline():
    """Example 2: Custom pipeline with multiple filters."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Pipeline")
    print("=" * 60)

    # Create sample frames
    frames = create_sample_frames(20)
    print(f"Created {len(frames)} sample frames")

    # Create custom pipeline
    pipeline = [
        BlurTagger(),
        HashTagger(hash_type="phash"),
        BlurFilter(threshold=100.0),
        DuplicateFilter(buffer_size=10, threshold=0.05),
    ]

    processor = FrameProcessor(pipeline=pipeline)

    # Process frames with result tracking
    frame_iter, result = processor.process(frames, return_result=True)
    list(frame_iter)

    print("\nProcessing Results:")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Selected frames: {result.selected_frames}")
    print(f"  Filtered frames: {result.filtered_frames}")
    print(f"  Selection rate: {result.get_selection_rate():.1f}%")
    print(f"  Processing time: {result.processing_time:.4f}s")
    print(f"  Throughput: {result.get_throughput():.1f} fps")

    # Show stage metrics
    print("\nStage Metrics:")
    for stage_name, metrics in result.stage_metrics.items():
        print(f"  {stage_name}:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")


def example_3_duplicate_detection():
    """Example 3: Duplicate detection strategy."""
    print("\n" + "=" * 60)
    print("Example 3: Duplicate Detection")
    print("=" * 60)

    # Create frames with some duplicates
    frames = []
    for i in range(15):
        if i % 5 == 0:
            # Create identical frames every 5 frames
            frame_data = np.ones((100, 100, 3), dtype=np.uint8) * 100
        else:
            # Create unique frames
            frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=i,
            timestamp=datetime.timedelta(seconds=i * 0.033),
            source_video_id="demo_video",
        )
        frames.append(packet)

    print(f"Created {len(frames)} frames (with some duplicates)")

    # Use duplicate detection strategy
    processor = FrameProcessor.with_duplicate_detection(threshold=0.05, window_size=10)

    selected_frames = list(processor.process(frames))

    print(f"Selected {len(selected_frames)} unique frames")
    print(f"Filtered out {len(frames) - len(selected_frames)} duplicate frames")


def example_4_smart_sampling():
    """Example 4: Smart sampling strategy."""
    print("\n" + "=" * 60)
    print("Example 4: Smart Sampling")
    print("=" * 60)

    # Create diverse sample frames
    frames = create_sample_frames(30)
    print(f"Created {len(frames)} sample frames")

    # Use smart sampling strategy (combines blur removal, duplicate detection, diversity)
    processor = FrameProcessor.with_smart_sampling(
        blur_threshold=100.0, duplicate_threshold=0.05, diversity_min_distance=0.1
    )

    selected_frames = list(processor.process(frames))

    print(f"Selected {len(selected_frames)} high-quality, diverse frames")
    print(f"Filtered out {len(frames) - len(selected_frames)} frames")


def example_5_tag_inspection():
    """Example 5: Inspecting computed tags."""
    print("\n" + "=" * 60)
    print("Example 5: Tag Inspection")
    print("=" * 60)

    # Create a few sample frames
    frames = create_sample_frames(5)

    # Create processor with blur removal
    processor = FrameProcessor.with_blur_removal(threshold=50.0)

    # Process frames and inspect tags
    print("\nFrame Tags:")
    for frame in processor.process(frames):
        print(f"  Frame {frame.frame_number}:")
        for tag_key, tag_value in frame.tags.items():
            if isinstance(tag_value, float):
                print(f"    {tag_key}: {tag_value:.2f}")
            else:
                print(f"    {tag_key}: {tag_value}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FrameProcessor API Demonstration")
    print("=" * 60)

    example_1_blur_removal()
    example_2_custom_pipeline()
    example_3_duplicate_detection()
    example_4_smart_sampling()
    example_5_tag_inspection()

    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)
