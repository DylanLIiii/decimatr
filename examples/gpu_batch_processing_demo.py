"""
GPU Batch Processing Demo

This example demonstrates how to use GPU batch processing with the
GPUBatchProcessor actor for efficient GPU-accelerated frame tagging.

The GPUBatchProcessor accumulates frames into batches and processes them
on GPU for improved throughput. It includes automatic CPU fallback when
GPU operations fail.

Requirements:
    - GPU dependencies: pip install decimatr[gpu]
    - CUDA-capable GPU
"""

import asyncio
from datetime import timedelta

import numpy as np

from decimatr.actors.gpu_actor import GPUBatchProcessor
from decimatr.core.processor import FrameProcessor
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.clip import CLIPTagger


def create_sample_frames(count: int = 10) -> list[VideoFramePacket]:
    """Create sample frames for demonstration."""
    frames = []
    for i in range(count):
        frame_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=i,
            timestamp=timedelta(seconds=i / 30.0),
            source_video_id="demo_video",
            tags={},
            additional_metadata={},
        )
        frames.append(packet)
    return frames


def demo_gpu_availability():
    """Check GPU availability before using GPU features."""
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)

    # Check if GPU is available
    gpu_available = FrameProcessor.check_gpu_available()
    print(f"GPU Available: {gpu_available}")

    if gpu_available:
        # Get detailed GPU info
        gpu_info = FrameProcessor.get_gpu_info()
        print(f"CUDA Version: {gpu_info.get('cuda_version', 'N/A')}")
        print(f"Device Count: {gpu_info.get('device_count', 0)}")
        print(f"Device Name: {gpu_info.get('device_name', 'N/A')}")
    else:
        missing = FrameProcessor.get_gpu_info()["missing_dependencies"]
        print(f"Missing Dependencies: {', '.join(missing)}")
        print("Install with: pip install decimatr[gpu]")

    print()
    return gpu_available


async def demo_gpu_batch_processor():
    """Demonstrate GPU batch processing with GPUBatchProcessor actor."""
    print("=" * 60)
    print("GPU Batch Processor Demo")
    print("=" * 60)

    # Check GPU availability
    if not FrameProcessor.check_gpu_available():
        print("GPU not available. Skipping GPU batch processor demo.")
        print("Install GPU dependencies with: pip install decimatr[gpu]")
        return

    # Import xoscar for actor creation
    import xoscar as xo

    # Create CLIP tagger for GPU processing
    clip_tagger = CLIPTagger(model_name="ViT-B/32", device="cuda")

    # Create actor pool
    await xo.create_actor_pool(address="127.0.0.1:13600", n_process=1)

    # Create GPU batch processor actor
    batch_processor = await xo.create_actor(
        GPUBatchProcessor,
        clip_tagger,
        batch_size=4,  # Process 4 frames at a time
        fallback_to_cpu=True,  # Enable CPU fallback
        max_gpu_failures=3,  # Switch to CPU after 3 failures
        address="127.0.0.1:13600",
        uid="demo_gpu_processor",
    )

    print("Created GPU batch processor with batch_size=4")
    print()

    # Create sample frames
    frames = create_sample_frames(10)
    print(f"Processing {len(frames)} frames...")

    # Process frames through batch processor
    processed_count = 0
    for i, frame in enumerate(frames):
        result = await batch_processor.add_frame(frame)

        if result is not None:
            # Batch was processed
            print(f"Batch processed: {len(result)} frames")
            for packet in result:
                embedding = packet.tags.get("clip_embedding")
                print(f"  Frame {packet.frame_number}: embedding shape {embedding.shape}")
            processed_count += len(result)

    # Flush remaining frames
    remaining = await batch_processor.flush()
    if remaining:
        print(f"Flushed remaining batch: {len(remaining)} frames")
        for packet in remaining:
            embedding = packet.tags.get("clip_embedding")
            print(f"  Frame {packet.frame_number}: embedding shape {embedding.shape}")
        processed_count += len(remaining)

    print(f"\nTotal frames processed: {processed_count}")

    # Get batch processor statistics
    stats = await batch_processor.get_stats()
    print("\nBatch Processor Statistics:")
    print(f"  Batch size: {stats['batch_size']}")
    print(f"  GPU failures: {stats['gpu_failures']}")
    print(f"  Using CPU: {stats['using_cpu']}")
    print(f"  Fallback enabled: {stats['fallback_enabled']}")


def demo_frame_processor_with_gpu():
    """Demonstrate FrameProcessor with GPU acceleration."""
    print("=" * 60)
    print("FrameProcessor with GPU Demo")
    print("=" * 60)

    # Check GPU availability
    if not FrameProcessor.check_gpu_available():
        print("GPU not available. Using CPU processing.")
        use_gpu = False
    else:
        print("GPU available. Enabling GPU acceleration.")
        use_gpu = True

    # Create CLIP tagger
    clip_tagger = CLIPTagger(
        model_name="ViT-B/32", device="cuda" if use_gpu else "cpu"
    )

    # Create processor with GPU support
    processor = FrameProcessor(
        pipeline=[clip_tagger],
        n_workers=2,  # Use 2 workers for parallel processing
        use_gpu=use_gpu,  # Enable GPU if available
        gpu_batch_size=8,  # Process 8 frames per GPU batch
    )

    print(f"Created FrameProcessor with use_gpu={use_gpu}")
    print()

    # Create sample frames
    frames = create_sample_frames(20)
    print(f"Processing {len(frames)} frames...")

    # Process frames
    processed_frames = list(processor.process(frames))

    print(f"Processed {len(processed_frames)} frames")
    for packet in processed_frames[:3]:  # Show first 3
        embedding = packet.tags.get("clip_embedding")
        print(f"  Frame {packet.frame_number}: embedding shape {embedding.shape}")

    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("GPU Batch Processing Demo")
    print("=" * 60 + "\n")

    # Demo 1: Check GPU availability
    gpu_available = demo_gpu_availability()

    # Demo 2: GPU batch processor (async)
    if gpu_available:
        asyncio.run(demo_gpu_batch_processor())
        print()

    # Demo 3: FrameProcessor with GPU
    demo_frame_processor_with_gpu()

    print("=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
