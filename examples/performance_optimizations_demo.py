"""
Demo: Performance Optimizations in Decimatr

This example demonstrates the performance optimizations implemented in Phase 9:
1. Lazy evaluation - compute tags only when needed by filters
2. Memory release - free frame data from filtered frames
3. Comprehensive metrics - track performance and throughput

Requirements:
    - 6.6: Lazy evaluation, computing tags only when required
    - 6.7: Releases frame data from memory after filtering
    - 9.4: Tracks processing statistics
    - 9.5: Provides performance metrics
"""

import datetime

import numpy as np
from decimatr.core.processor import FrameProcessor
from decimatr.filters.blur import BlurFilter
from decimatr.filters.threshold import ThresholdFilter
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.entropy import EntropyTagger
from decimatr.taggers.hash import HashTagger


def create_sample_frames(count: int = 100) -> list:
    """
    Create sample video frames for demonstration.

    Args:
        count: Number of frames to create

    Returns:
        List of VideoFramePacket objects
    """
    frames = []
    for i in range(count):
        # Alternate between sharp and blurry frames
        if i % 3 == 0:
            # Blurry frame - uniform color
            frame_data = np.ones((480, 640, 3), dtype=np.uint8) * 128
        else:
            # Sharp frame - random noise (high frequency content)
            frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=i,
            timestamp=datetime.timedelta(seconds=i / 30.0),
            source_video_id="demo_video",
        )
        frames.append(packet)

    return frames


def demo_lazy_evaluation():
    """Demonstrate lazy evaluation optimization."""
    print("\n" + "=" * 70)
    print("DEMO 1: Lazy Evaluation")
    print("=" * 70)
    print("\nLazy evaluation computes tags only when needed by downstream filters.")
    print("This saves computation when taggers produce tags that aren't used.\n")

    # Create sample frames
    frames = create_sample_frames(50)

    # Pipeline with unused tagger
    print("Pipeline:")
    print("  1. BlurTagger - computes blur_score")
    print("  2. EntropyTagger - computes entropy (NOT USED by any filter)")
    print("  3. HashTagger - computes phash (NOT USED by any filter)")
    print("  4. BlurFilter - only needs blur_score\n")

    processor = FrameProcessor(
        pipeline=[
            BlurTagger(),
            EntropyTagger(),  # Not used by any filter
            HashTagger(),  # Not used by any filter
            BlurFilter(threshold=100.0),
        ],
        lazy_evaluation=True,
        release_memory=True,
    )

    # Process frames with metrics
    results, result = processor.process(frames, return_result=True)
    list(results)

    print("Results:")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Selected frames: {result.selected_frames}")
    print(f"  Filtered frames: {result.filtered_frames}")
    print(f"  Processing time: {result.processing_time:.3f}s")
    print(f"  Throughput: {result.get_throughput():.1f} fps")
    print("\nOptimizations:")
    print(f"  Lazy evaluation: {'enabled' if result.lazy_evaluation_enabled else 'disabled'}")
    print(f"  Memory release: {'enabled' if result.memory_release_enabled else 'disabled'}")

    # Show which taggers were actually executed
    print("\nStage Metrics:")
    for stage_name, metrics in result.stage_metrics.items():
        if metrics["type"] == "tagger":
            frames_processed = metrics["frames_processed"]
            lazy = metrics.get("lazy_evaluated", False)
            print(
                f"  {stage_name}: {frames_processed} frames processed "
                f"{'(lazy)' if lazy else '(eager)'}"
            )


def demo_memory_release():
    """Demonstrate memory release optimization."""
    print("\n" + "=" * 70)
    print("DEMO 2: Memory Release")
    print("=" * 70)
    print("\nMemory release frees frame data from filtered frames to reduce memory usage.")
    print("This is especially important when processing large videos.\n")

    # Create sample frames
    frames = create_sample_frames(30)

    print("Pipeline:")
    print("  1. BlurTagger")
    print("  2. BlurFilter (threshold=150.0) - strict filter\n")

    # Process with memory release enabled
    processor_with_release = FrameProcessor(
        pipeline=[BlurTagger(), BlurFilter(threshold=150.0)],
        lazy_evaluation=True,
        release_memory=True,
    )

    results, result = processor_with_release.process(frames, return_result=True)
    list(results)

    print("Results (with memory release):")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Selected frames: {result.selected_frames}")
    print(f"  Filtered frames: {result.filtered_frames}")
    print(f"  Selection rate: {result.get_selection_rate():.1f}%")
    print(f"\nNote: Frame data from {result.filtered_frames} filtered frames was released")
    print("      to reduce memory usage during processing.")


def demo_comprehensive_metrics():
    """Demonstrate comprehensive metrics collection."""
    print("\n" + "=" * 70)
    print("DEMO 3: Comprehensive Metrics")
    print("=" * 70)
    print("\nThe processor tracks detailed metrics for monitoring and optimization.\n")

    # Create sample frames
    frames = create_sample_frames(100)

    print("Pipeline:")
    print("  1. BlurTagger")
    print("  2. EntropyTagger")
    print("  3. BlurFilter (threshold=100.0)")
    print("  4. ThresholdFilter (entropy > 4.0)\n")

    processor = FrameProcessor(
        pipeline=[
            BlurTagger(),
            EntropyTagger(),
            BlurFilter(threshold=100.0),
            ThresholdFilter(tag_key="entropy", threshold=4.0, operator=">"),
        ],
        lazy_evaluation=True,
        release_memory=True,
    )

    # Process frames with metrics
    results, result = processor.process(frames, return_result=True)
    list(results)

    # Print detailed summary
    result.print_summary()

    # Also show programmatic access to metrics
    print("\nProgrammatic Access to Metrics:")
    summary = result.get_summary()
    print(f"  Session ID: {summary['session_id']}")
    print(f"  Throughput: {summary['throughput_fps']:.1f} fps")
    print(f"  Selection Rate: {summary['selection_rate_percent']:.1f}%")
    print(f"  Error Count: {summary['error_count']}")


def demo_comparison():
    """Compare lazy vs eager evaluation."""
    print("\n" + "=" * 70)
    print("DEMO 4: Lazy vs Eager Evaluation Comparison")
    print("=" * 70)
    print("\nComparing performance between lazy and eager evaluation.\n")

    # Create sample frames
    frames = create_sample_frames(100)

    print("Pipeline:")
    print("  1. BlurTagger")
    print("  2. EntropyTagger (not used by filters)")
    print("  3. HashTagger (not used by filters)")
    print("  4. BlurFilter\n")

    # Test with lazy evaluation
    print("Testing with LAZY evaluation...")
    processor_lazy = FrameProcessor(
        pipeline=[BlurTagger(), EntropyTagger(), HashTagger(), BlurFilter(threshold=100.0)],
        lazy_evaluation=True,
    )

    results_lazy, result_lazy = processor_lazy.process(frames.copy(), return_result=True)
    list(results_lazy)

    # Test with eager evaluation
    print("Testing with EAGER evaluation...")
    processor_eager = FrameProcessor(
        pipeline=[BlurTagger(), EntropyTagger(), HashTagger(), BlurFilter(threshold=100.0)],
        lazy_evaluation=False,
    )

    results_eager, result_eager = processor_eager.process(frames.copy(), return_result=True)
    list(results_eager)

    # Compare results
    print("\nComparison:")
    print(f"  {'Metric':<30} {'Lazy':<15} {'Eager':<15}")
    print(f"  {'-' * 30} {'-' * 15} {'-' * 15}")
    print(
        f"  {'Selected frames':<30} {result_lazy.selected_frames:<15} {result_eager.selected_frames:<15}"
    )
    print(
        f"  {'Processing time (s)':<30} {result_lazy.processing_time:<15.3f} {result_eager.processing_time:<15.3f}"
    )
    print(
        f"  {'Throughput (fps)':<30} {result_lazy.get_throughput():<15.1f} {result_eager.get_throughput():<15.1f}"
    )

    # Calculate speedup
    if result_eager.processing_time > 0:
        speedup = result_eager.processing_time / result_lazy.processing_time
        print(f"\n  Lazy evaluation speedup: {speedup:.2f}x")

    print("\nNote: Lazy evaluation skips computing entropy and hash tags since")
    print("      no filters use them, resulting in faster processing.")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Decimatr Performance Optimizations Demo")
    print("=" * 70)
    print("\nThis demo showcases the performance optimizations in Phase 9:")
    print("  - Lazy evaluation: compute tags only when needed")
    print("  - Memory release: free frame data from filtered frames")
    print("  - Comprehensive metrics: track performance and throughput")

    # Run demonstrations
    demo_lazy_evaluation()
    demo_memory_release()
    demo_comprehensive_metrics()
    demo_comparison()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Lazy evaluation improves performance by skipping unused computations")
    print("  2. Memory release reduces memory usage for large video processing")
    print("  3. Comprehensive metrics help monitor and optimize processing pipelines")
    print("  4. Both optimizations are enabled by default for best performance")
    print()


if __name__ == "__main__":
    main()
