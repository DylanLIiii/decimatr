"""
StreamProcessor demo - Real-time frame processing example.

This example demonstrates how to use StreamProcessor for real-time video
frame processing with a sliding window buffer.
"""

import datetime

import numpy as np
from loguru import logger

# Note: imagehash package installation required
# pip install imagehash

try:
    from decimatr.core.stream_processor import OverflowStrategy, StreamProcessor
    from decimatr.filters.blur import BlurFilter
    from decimatr.scheme import VideoFramePacket
    from decimatr.taggers.blur import BlurTagger
    from decimatr.taggers.hash import HashTagger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure decimatr is installed: pip install -e '.[dev]'")
    exit(1)


def create_synthetic_frames(num_frames: int, width: int = 640, height: int = 480):
    """Generate synthetic video frames for demonstration."""
    frames = []
    for i in range(num_frames):
        # Create frame with varying blur (simulates real video)
        # Use sine wave to create periodic sharp/blurry frames
        blur_amount = int(128 + 127 * np.sin(i * 0.5))
        frame_data = np.full((height, width, 3), blur_amount, dtype=np.uint8)

        # Add some noise to make frames unique
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame_data = np.clip(frame_data + noise, 0, 255).astype(np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=i,
            timestamp=datetime.timedelta(seconds=i / 30.0),  # 30 fps
            source_video_id="synthetic_video.mp4",
        )
        frames.append(packet)

    return frames


def demo_blur_filter_stream():
    """Demonstrate StreamProcessor with blur filtering."""
    logger.info("=== StreamProcessor Demo: Blur Filtering ===")

    # Create stream processor with blur removal
    stream = StreamProcessor(
        pipeline=[
            BlurTagger(),
            BlurFilter(threshold=100.0),  # Only keep frames with blur_score >= 100
        ],
        buffer_size=50,  # Keep last 50 selected frames
        overflow_strategy=OverflowStrategy.DROP_OLDEST,
        lazy_evaluation=True,
        release_memory=True,
    )

    logger.info(f"Created StreamProcessor with {len(stream.pipeline)} pipeline components")
    logger.info(f"Buffer capacity: {stream.get_buffer_capacity()} frames")

    # Generate synthetic video frames
    logger.info("Generating 100 synthetic frames...")
    frames = create_synthetic_frames(100, width=320, height=240)

    # Process frames in a stream
    logger.info("Processing frames through stream...")
    selected_count = 0
    for i, frame in enumerate(frames):
        # Add frame to stream - returns True if frame passes filters
        if stream.add(frame):
            selected_count += 1

        # Simulate real-time processing (e.g., analyzing last 10 frames every 10 frames)
        if i > 0 and i % 10 == 0:
            recent_frames = list(stream.get_selected_frames())
            logger.info(
                f"  After frame {i}: "
                f"{stream.get_buffer_count()} in buffer, "
                f"{selected_count} selected so far"
            )

    # Show final results
    metrics = stream.get_metrics()
    logger.info("\n=== Final Results ===")
    logger.info(f"Total frames added: {metrics['total_frames_added']}")
    logger.info(f"Frames selected: {metrics['total_frames_selected']}")
    logger.info(f"Frames rejected: {metrics['total_frames_rejected']}")
    logger.info(f"Selection rate: {metrics['selection_rate']:.1f}%")

    # Access buffered frames
    buffered = list(stream.get_selected_frames())
    logger.info(f"Buffered frames: {len(buffered)}")
    if buffered:
        logger.info(f"  First buffered frame: #{buffered[0].frame_number}")
        logger.info(f"  Last buffered frame: #{buffered[-1].frame_number}")

    # Demonstrate buffer clearing
    logger.info("\n=== Clearing Buffer ===")
    logger.info(f"Before clear: {stream.get_buffer_count()} frames")
    stream.clear()
    logger.info(f"After clear: {stream.get_buffer_count()} frames")

    logger.success("Blur filtering demo completed!\n")


def demo_buffer_overflow_strategies():
    """Demonstrate different buffer overflow strategies."""
    logger.info("=== Demo: Buffer Overflow Strategies ===")

    # Strategy 1: DROP_OLDEST (default)
    logger.info("\n1. DROP_OLDEST strategy:")
    stream1 = StreamProcessor(
        pipeline=[],
        buffer_size=3,
        overflow_strategy=OverflowStrategy.DROP_OLDEST,
    )

    for i in range(5):
        frames = create_synthetic_frames(1)
        stream1.add(frames[0])
        logger.info(f"  Added frame {i}: buffer has {stream1.get_buffer_count()} frames")

    frames1 = [f.frame_number for f in stream1.get_selected_frames()]
    logger.info(f"  Final buffer (oldest -> newest): {frames1}")

    # Strategy 2: REJECT
    logger.info("\n2. REJECT strategy:")
    stream2 = StreamProcessor(
        pipeline=[],
        buffer_size=3,
        overflow_strategy=OverflowStrategy.REJECT,
    )

    for i in range(5):
        frames = create_synthetic_frames(1)
        success = stream2.add(frames[0])
        logger.info(
            f"  Added frame {i}: {'accepted' if success else 'rejected'} "
            f"(buffer: {stream2.get_buffer_count()} frames)"
        )

    frames2 = [f.frame_number for f in stream2.get_selected_frames()]
    logger.info(f"  Final buffer (oldest -> newest): {frames2}")

    logger.success("Buffer overflow strategies demo completed!\n")


def demo_stream_buffer_access():
    """Demonstrate accessing frames from the stream buffer."""
    logger.info("=== Demo: Real-Time Buffer Access ===")

    stream = StreamProcessor(
        pipeline=[BlurTagger(), BlurFilter(threshold=80.0)],
        buffer_size=20,
    )

    # Simulate real-time stream processing
    logger.info("Processing 50 frames with real-time analysis...")
    frames = create_synthetic_frames(50)

    for i, frame in enumerate(frames):
        stream.add(frame)

        # Analyze every 10th frame
        if i % 10 == 0 and i > 0:
            logger.info(f"\n--- Analysis at frame {i} ---")

            # Get buffered frames
            buffered = list(stream.get_selected_frames())
            logger.info(f"Frames in buffer: {len(buffered)}")

            if buffered:
                # Calculate average blur score of buffered frames
                blur_scores = [f.get_tag("blur_score", 0) for f in buffered]
                avg_blur = sum(blur_scores) / len(blur_scores)
                logger.info(f"Average blur score: {avg_blur:.1f}")

                logger.info(
                    f"Frame range in buffer: #{buffered[0].frame_number} - "
                    f"#{buffered[-1].frame_number}"
                )

    logger.success("Real-time buffer access demo completed!\n")


def main():
    """Run all StreamProcessor demos."""
    print("\n" + "=" * 70)
    print("StreamProcessor - Real-Time Frame Processing Examples")
    print("=" * 70 + "\n")

    try:
        # Run demos
        demo_blur_filter_stream()
        demo_buffer_overflow_strategies()
        demo_stream_buffer_access()

        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
