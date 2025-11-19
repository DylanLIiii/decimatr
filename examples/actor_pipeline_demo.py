"""
Demo: FrameProcessor with ActorPipeline Integration

This example demonstrates how to use FrameProcessor with parallel processing
via ActorPipeline. When n_workers > 1, the processor automatically uses
actor-based distributed processing for improved throughput.
"""

from datetime import timedelta

import numpy as np
from decimatr.core.processor import FrameProcessor
from decimatr.scheme import VideoFramePacket


def create_sample_frames(count: int = 10) -> list:
    """Create sample frames for demonstration."""
    frames = []
    for i in range(count):
        # Create frames with varying blur levels
        if i % 3 == 0:
            # Sharp frame with high-frequency content
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame[40:60, :] = 255
            frame[:, 40:60] = 0
        else:
            # Blurry frame (uniform color)
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

        packet = VideoFramePacket(
            frame_data=frame,
            frame_number=i,
            timestamp=timedelta(seconds=i / 30.0),
            source_video_id="demo_video",
            tags={},
            additional_metadata={}
        )
        frames.append(packet)

    return frames


def demo_single_threaded():
    """Demonstrate single-threaded processing (n_workers=1)."""
    print("\n=== Single-Threaded Processing (n_workers=1) ===")

    # Create processor with blur removal
    processor = FrameProcessor.with_blur_removal(
        threshold=50.0,
        n_workers=1  # Single-threaded
    )

    # Create sample frames
    frames = create_sample_frames(10)
    print(f"Created {len(frames)} sample frames")

    # Process frames
    results = list(processor.process(frames))
    print(f"Selected {len(results)} frames after filtering")

    for frame in results:
        print(f"  - Frame {frame.frame_number}: blur_score={frame.tags.get('blur_score', 'N/A'):.2f}")


def demo_parallel_processing():
    """Demonstrate parallel processing with actors (n_workers > 1)."""
    print("\n=== Parallel Processing with ActorPipeline (n_workers=4) ===")

    # Create processor with parallel execution
    processor = FrameProcessor.with_blur_removal(
        threshold=50.0,
        n_workers=4  # Actor-based parallel processing
    )

    # Create sample frames
    frames = create_sample_frames(10)
    print(f"Created {len(frames)} sample frames")

    # Process frames with result summary
    frame_iter, result = processor.process(frames, return_result=True)
    results = list(frame_iter)

    print(f"Selected {len(results)} frames after filtering")
    print("\nProcessing Summary:")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Selected frames: {result.selected_frames}")
    print(f"  Filtered frames: {result.filtered_frames}")
    print(f"  Processing time: {result.processing_time:.3f}s")
    print(f"  Throughput: {result.get_throughput():.1f} frames/sec")

    for frame in results:
        print(f"  - Frame {frame.frame_number}: blur_score={frame.tags.get('blur_score', 'N/A'):.2f}")


def demo_smart_sampling():
    """Demonstrate smart sampling with parallel processing."""
    print("\n=== Smart Sampling with Parallel Processing ===")

    # Create processor with smart sampling strategy
    processor = FrameProcessor.with_smart_sampling(n_workers=4)

    # Create more frames for diversity filtering
    frames = create_sample_frames(20)
    print(f"Created {len(frames)} sample frames")

    # Process frames
    results = list(processor.process(frames))
    print(f"Selected {len(results)} frames after smart sampling")


if __name__ == "__main__":
    print("FrameProcessor with ActorPipeline Integration Demo")
    print("=" * 60)

    # Demo 1: Single-threaded processing
    demo_single_threaded()

    # Demo 2: Parallel processing with actors
    demo_parallel_processing()

    # Demo 3: Smart sampling with parallel processing
    demo_smart_sampling()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("\nKey Points:")
    print("  - n_workers=1: Uses single-threaded processing")
    print("  - n_workers>1: Uses ActorPipeline for parallel execution")
    print("  - ActorPipeline automatically distributes work across CPU cores")
    print("  - Each processor instance uses a unique port to avoid conflicts")
