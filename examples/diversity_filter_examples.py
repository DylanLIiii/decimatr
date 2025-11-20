"""
Comprehensive examples for the enhanced DiversityFilter.

This example demonstrates all the new features of the enhanced DiversityFilter:
1. Hash-based diversity (perceptual hashes)
2. CLIP embedding diversity (semantic similarity)
3. Color histogram diversity
4. Combining multiple diversity tags
5. Custom comparison strategies
6. Weighted combination mode

The enhanced DiversityFilter distinguishes between diversity-suitable tags
(hashes, embeddings, histograms) and metric-only tags (blur_score, entropy),
and supports tag-specific comparison strategies for accurate diversity measurement.
"""

import datetime

import cv2
import numpy as np
from decimatr.core.processor import FrameProcessor
from decimatr.filters.comparison_strategies import (
    EmbeddingDistanceStrategy,
    HammingDistanceStrategy,
    HistogramDistanceStrategy,
)
from decimatr.filters.diversity import DiversityFilter
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger


def create_diverse_frames(count: int = 20) -> list:
    """
    Create sample frames with varying visual characteristics.

    Creates frames with different colors, patterns, and blur levels to
    demonstrate diversity filtering.
    """
    frames = []
    for i in range(count):
        # Create frames with different characteristics
        if i % 4 == 0:
            # Red frames
            frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
            frame_data[:, :, 2] = 200  # Red channel
        elif i % 4 == 1:
            # Blue frames
            frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
            frame_data[:, :, 0] = 200  # Blue channel
        elif i % 4 == 2:
            # Green frames
            frame_data = np.zeros((100, 100, 3), dtype=np.uint8)
            frame_data[:, :, 1] = 200  # Green channel
        else:
            # Random noise frames
            frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=i,
            timestamp=datetime.timedelta(seconds=i * 0.033),  # ~30fps
            source_video_id="demo_video",
        )
        frames.append(packet)

    return frames


def example_1_hash_based_diversity():
    """
    Example 1: Using hash-based diversity with perceptual hashes.

    Perceptual hashes (dhash, phash, ahash) are efficient for detecting
    visually similar frames. They use Hamming distance for comparison.

    This is the fastest diversity method and works well for detecting
    near-duplicate frames or frames with similar visual structure.
    """
    print("\n" + "=" * 70)
    print("Example 1: Hash-Based Diversity (Perceptual Hashes)")
    print("=" * 70)

    # Create sample frames
    frames = create_diverse_frames(20)
    print(f"Created {len(frames)} sample frames with varying colors and patterns")

    # Create pipeline with hash tagger and diversity filter
    pipeline = [
        HashTagger(hash_type="dhash"),  # Compute perceptual hash
        DiversityFilter(
            buffer_size=50,
            diversity_tags=["dhash"],  # Use dhash for diversity
            min_distance=0.1,  # Minimum Hamming distance threshold
        ),
    ]

    processor = FrameProcessor(pipeline=pipeline)

    # Process frames
    frame_iter, result = processor.process(frames, return_result=True)
    selected_frames = list(frame_iter)

    print("\nResults:")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Selected frames: {result.selected_frames}")
    print(f"  Filtered frames: {result.filtered_frames}")
    print(f"  Selection rate: {result.get_selection_rate():.1f}%")

    print(f"\nSelected frame numbers: {[f.frame_number for f in selected_frames]}")

    print("\nKey Points:")
    print("  - dhash uses Hamming distance (bit-level comparison)")
    print("  - Very fast and memory efficient")
    print("  - Good for detecting near-duplicate frames")
    print("  - Normalized to [0, 1] range automatically")


def example_2_clip_embedding_diversity():
    """
    Example 2: Using CLIP embedding diversity for semantic similarity.

    CLIP embeddings capture semantic content of frames. This is useful for
    selecting frames with different semantic content (e.g., different scenes,
    objects, or activities).

    Note: This example requires GPU dependencies. Install with:
        pip install decimatr[gpu]

    If GPU is not available, the example will be skipped.
    """
    print("\n" + "=" * 70)
    print("Example 2: CLIP Embedding Diversity (Semantic Similarity)")
    print("=" * 70)

    # Check if GPU dependencies are available
    try:
        from decimatr.taggers.clip import CLIPTagger

        gpu_available = FrameProcessor.check_gpu_available()
        print(f"GPU available: {gpu_available}")

        # Create sample frames
        frames = create_diverse_frames(15)
        print(f"Created {len(frames)} sample frames")

        # Create pipeline with CLIP tagger and diversity filter
        # CLIPTagger automatically selects MobileCLIP for CPU, standard CLIP for GPU
        pipeline = [
            CLIPTagger(device="auto", batch_size=8),  # Compute CLIP embeddings
            DiversityFilter(
                buffer_size=50,
                diversity_tags=["clip_embedding"],  # Use CLIP embeddings
                min_distance=0.15,  # Cosine distance threshold
                comparison_strategies={
                    "clip_embedding": EmbeddingDistanceStrategy(metric="cosine")
                },
            ),
        ]

        processor = FrameProcessor(pipeline=pipeline, use_gpu=gpu_available)

        # Process frames
        frame_iter, result = processor.process(frames, return_result=True)
        selected_frames = list(frame_iter)

        print("\nResults:")
        print(f"  Total frames: {result.total_frames}")
        print(f"  Selected frames: {result.selected_frames}")
        print(f"  Filtered frames: {result.filtered_frames}")
        print(f"  Selection rate: {result.get_selection_rate():.1f}%")

        print(f"\nSelected frame numbers: {[f.frame_number for f in selected_frames]}")

        print("\nKey Points:")
        print("  - CLIP embeddings capture semantic content")
        print("  - Cosine distance measures semantic similarity")
        print("  - Automatically uses MobileCLIP on CPU for efficiency")
        print("  - Best for selecting semantically diverse frames")

    except ImportError:
        print("\nSkipping CLIP example - GPU dependencies not installed")
        print("Install with: pip install decimatr[gpu]")
        print("Or: uv pip install -e '.[gpu]'")


def example_3_color_histogram_diversity():
    """
    Example 3: Using color histogram diversity.

    Color histograms capture the distribution of colors in a frame. This is
    useful for selecting frames with different color palettes or lighting
    conditions.

    This example shows how to compute color histograms and use them for
    diversity filtering.
    """
    print("\n" + "=" * 70)
    print("Example 3: Color Histogram Diversity")
    print("=" * 70)

    # Create sample frames
    frames = create_diverse_frames(20)
    print(f"Created {len(frames)} sample frames with varying colors")

    # Create a custom tagger for color histograms
    from decimatr.taggers.base import Tagger

    class ColorHistogramTagger(Tagger):
        """Compute color histogram for each frame."""

        def compute_tags(self, packet: VideoFramePacket) -> dict:
            """Compute color histogram."""
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(packet.frame_data, cv2.COLOR_BGR2HSV)

            # Compute histogram for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

            # Concatenate histograms
            hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()

            return {"color_hist": hist}

        @property
        def tag_keys(self) -> list[str]:
            return ["color_hist"]

    # Create pipeline with color histogram tagger and diversity filter
    pipeline = [
        ColorHistogramTagger(),
        DiversityFilter(
            buffer_size=50,
            diversity_tags=["color_hist"],  # Use color histogram
            min_distance=0.2,  # Histogram intersection distance threshold
            comparison_strategies={"color_hist": HistogramDistanceStrategy(metric="intersection")},
        ),
    ]

    processor = FrameProcessor(pipeline=pipeline)

    # Process frames
    frame_iter, result = processor.process(frames, return_result=True)
    selected_frames = list(frame_iter)

    print("\nResults:")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Selected frames: {result.selected_frames}")
    print(f"  Filtered frames: {result.filtered_frames}")
    print(f"  Selection rate: {result.get_selection_rate():.1f}%")

    print(f"\nSelected frame numbers: {[f.frame_number for f in selected_frames]}")

    print("\nKey Points:")
    print("  - Color histograms capture color distribution")
    print("  - Histogram intersection measures color similarity")
    print("  - Good for detecting lighting changes or color shifts")
    print("  - Alternative metrics: chi_square, bhattacharyya")


def example_4_combining_multiple_tags():
    """
    Example 4: Combining multiple diversity tags.

    This example shows how to use multiple diversity tags simultaneously.
    By default, the filter uses the maximum distance across all tags,
    ensuring frames are diverse in at least one dimension.

    This is useful when you want frames that are diverse in any aspect
    (visual structure, color, or semantic content).
    """
    print("\n" + "=" * 70)
    print("Example 4: Combining Multiple Diversity Tags")
    print("=" * 70)

    # Create sample frames
    frames = create_diverse_frames(20)
    print(f"Created {len(frames)} sample frames")

    # Create a custom color histogram tagger
    from decimatr.taggers.base import Tagger

    class ColorHistogramTagger(Tagger):
        """Compute color histogram for each frame."""

        def compute_tags(self, packet: VideoFramePacket) -> dict:
            hsv = cv2.cvtColor(packet.frame_data, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
            return {"color_hist": hist}

        @property
        def tag_keys(self) -> list[str]:
            return ["color_hist"]

    # Create pipeline with multiple taggers and diversity filter
    pipeline = [
        HashTagger(hash_type="dhash"),  # Visual structure
        ColorHistogramTagger(),  # Color distribution
        BlurTagger(),  # Blur score (will be ignored as metric-only)
        DiversityFilter(
            buffer_size=50,
            diversity_tags=["dhash", "color_hist"],  # Use both tags
            min_distance=0.15,
            comparison_strategies={
                "dhash": HammingDistanceStrategy(),
                "color_hist": HistogramDistanceStrategy(metric="intersection"),
            },
        ),
    ]

    processor = FrameProcessor(pipeline=pipeline)

    # Process frames
    frame_iter, result = processor.process(frames, return_result=True)
    selected_frames = list(frame_iter)

    print("\nResults:")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Selected frames: {result.selected_frames}")
    print(f"  Filtered frames: {result.filtered_frames}")
    print(f"  Selection rate: {result.get_selection_rate():.1f}%")

    print(f"\nSelected frame numbers: {[f.frame_number for f in selected_frames]}")

    print("\nKey Points:")
    print("  - Uses maximum distance across all tags (default)")
    print("  - Frame passes if diverse in ANY dimension")
    print("  - blur_score is automatically excluded (metric-only tag)")
    print("  - Each tag uses its appropriate comparison strategy")


def example_5_custom_comparison_strategies():
    """
    Example 5: Custom comparison strategies for different metrics.

    This example demonstrates how to configure custom comparison strategies
    for different tag types. You can choose different distance metrics
    (Euclidean, cosine, Manhattan) for embeddings, or different histogram
    metrics (intersection, chi-square, Bhattacharyya).
    """
    print("\n" + "=" * 70)
    print("Example 5: Custom Comparison Strategies")
    print("=" * 70)

    # Create sample frames
    frames = create_diverse_frames(15)
    print(f"Created {len(frames)} sample frames")

    # Create a custom embedding tagger (simulated)
    from decimatr.taggers.base import Tagger

    class MockEmbeddingTagger(Tagger):
        """Create mock embeddings for demonstration."""

        def compute_tags(self, packet: VideoFramePacket) -> dict:
            # Create a simple embedding based on frame statistics
            mean_color = packet.frame_data.mean(axis=(0, 1))
            std_color = packet.frame_data.std(axis=(0, 1))
            embedding = np.concatenate([mean_color, std_color])
            return {"model_embedding": embedding}

        @property
        def tag_keys(self) -> list[str]:
            return ["model_embedding"]

    # Create pipeline with custom comparison strategies
    pipeline = [
        HashTagger(hash_type="phash"),  # Perceptual hash
        MockEmbeddingTagger(),  # Custom embeddings
        DiversityFilter(
            buffer_size=50,
            diversity_tags=["phash", "model_embedding"],
            min_distance=0.15,
            comparison_strategies={
                # Use Hamming distance for hashes (default)
                "phash": HammingDistanceStrategy(),
                # Use Manhattan distance for embeddings (instead of default cosine)
                "model_embedding": EmbeddingDistanceStrategy(metric="manhattan"),
            },
        ),
    ]

    processor = FrameProcessor(pipeline=pipeline)

    # Process frames
    frame_iter, result = processor.process(frames, return_result=True)
    selected_frames = list(frame_iter)

    print("\nResults:")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Selected frames: {result.selected_frames}")
    print(f"  Filtered frames: {result.filtered_frames}")
    print(f"  Selection rate: {result.get_selection_rate():.1f}%")

    print(f"\nSelected frame numbers: {[f.frame_number for f in selected_frames]}")

    print("\nAvailable Comparison Strategies:")
    print("  Hash tags:")
    print("    - HammingDistanceStrategy() [default]")
    print("  Embedding tags:")
    print("    - EmbeddingDistanceStrategy(metric='euclidean')")
    print("    - EmbeddingDistanceStrategy(metric='cosine') [default]")
    print("    - EmbeddingDistanceStrategy(metric='manhattan')")
    print("  Histogram tags:")
    print("    - HistogramDistanceStrategy(metric='intersection') [default]")
    print("    - HistogramDistanceStrategy(metric='chi_square')")
    print("    - HistogramDistanceStrategy(metric='bhattacharyya')")


def example_6_weighted_combination_mode():
    """
    Example 6: Weighted combination mode for multi-tag diversity.

    This example demonstrates the weighted combination mode, where distances
    from multiple tags are combined into a single diversity score using
    configurable weights.

    This is useful when you want to balance multiple diversity dimensions
    with specific importance weights (e.g., prioritize semantic diversity
    over visual structure).
    """
    print("\n" + "=" * 70)
    print("Example 6: Weighted Combination Mode")
    print("=" * 70)

    # Create sample frames
    frames = create_diverse_frames(20)
    print(f"Created {len(frames)} sample frames")

    # Create a custom color histogram tagger
    from decimatr.taggers.base import Tagger

    class ColorHistogramTagger(Tagger):
        """Compute color histogram for each frame."""

        def compute_tags(self, packet: VideoFramePacket) -> dict:
            hsv = cv2.cvtColor(packet.frame_data, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
            return {"color_hist": hist}

        @property
        def tag_keys(self) -> list[str]:
            return ["color_hist"]

    # Create pipeline with weighted combination
    pipeline = [
        HashTagger(hash_type="dhash"),  # Visual structure
        ColorHistogramTagger(),  # Color distribution
        DiversityFilter(
            buffer_size=50,
            diversity_tags=["dhash", "color_hist"],
            min_distance=0.15,
            enable_weighted_combination=True,  # Enable weighted mode
            tag_weights={
                "dhash": 0.3,  # 30% weight on visual structure
                "color_hist": 0.7,  # 70% weight on color distribution
            },
            comparison_strategies={
                "dhash": HammingDistanceStrategy(),
                "color_hist": HistogramDistanceStrategy(metric="intersection"),
            },
        ),
    ]

    processor = FrameProcessor(pipeline=pipeline)

    # Process frames
    frame_iter, result = processor.process(frames, return_result=True)
    selected_frames = list(frame_iter)

    print("\nResults:")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Selected frames: {result.selected_frames}")
    print(f"  Filtered frames: {result.filtered_frames}")
    print(f"  Selection rate: {result.get_selection_rate():.1f}%")

    print(f"\nSelected frame numbers: {[f.frame_number for f in selected_frames]}")

    print("\nKey Points:")
    print("  - Weighted combination mode enabled")
    print("  - Distances are normalized to [0, 1] before combining")
    print("  - Weights: dhash=0.3, color_hist=0.7")
    print("  - Final distance = 0.3 * norm(dhash_dist) + 0.7 * norm(color_dist)")
    print("  - If weights not specified, uses equal weights")

    print("\nComparison with Default Mode:")
    print("  Default (max distance):")
    print("    - Frame passes if diverse in ANY dimension")
    print("    - distance = max(dhash_dist, color_dist)")
    print("  Weighted combination:")
    print("    - Frame passes based on weighted average")
    print("    - distance = 0.3 * norm(dhash_dist) + 0.7 * norm(color_dist)")
    print("    - Allows fine-tuning importance of each dimension")


def example_7_auto_detection():
    """
    Example 7: Auto-detection of diversity tags.

    This example demonstrates the auto-detection feature, where the filter
    automatically identifies diversity-suitable tags from the frame and
    excludes metric-only tags (blur_score, entropy).

    This is useful for quick prototyping or when you want the filter to
    adapt to available tags automatically.
    """
    print("\n" + "=" * 70)
    print("Example 7: Auto-Detection of Diversity Tags")
    print("=" * 70)

    # Create sample frames
    frames = create_diverse_frames(15)
    print(f"Created {len(frames)} sample frames")

    # Manually apply taggers to demonstrate auto-detection
    # (In real usage, FrameProcessor handles this automatically)
    hash_tagger = HashTagger(hash_type="dhash")
    blur_tagger = BlurTagger()

    for frame in frames:
        # Apply taggers
        frame.tags.update(hash_tagger.compute_tags(frame))
        frame.tags.update(blur_tagger.compute_tags(frame))

    # Create filter with auto-detection (no diversity_tags specified)
    diversity_filter = DiversityFilter(
        buffer_size=50,
        # diversity_tags not specified - will auto-detect
        min_distance=0.15,
    )

    # Apply filter manually to show auto-detection
    selected_frames = []
    for frame in frames:
        if diversity_filter.should_pass(frame):
            selected_frames.append(frame)

    print("\nResults:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Selected frames: {len(selected_frames)}")
    print(f"  Filtered frames: {len(frames) - len(selected_frames)}")
    print(f"  Selection rate: {len(selected_frames) / len(frames) * 100:.1f}%")

    print(f"\nSelected frame numbers: {[f.frame_number for f in selected_frames]}")

    # Show which tags were detected
    if selected_frames:
        print(f"\nTags in first selected frame: {list(selected_frames[0].tags.keys())}")
        print("  - 'dhash' is diversity-suitable (used for filtering)")
        print("  - 'blur_score' is metric-only (excluded from filtering)")

    print("\nKey Points:")
    print("  - diversity_tags not specified (auto-detection enabled)")
    print("  - Filter automatically uses 'dhash' (diversity-suitable)")
    print("  - Filter automatically excludes 'blur_score' (metric-only)")
    print("  - Diversity-suitable tags: dhash, phash, ahash, clip_embedding, color_hist")
    print("  - Metric-only tags: blur_score, entropy, edge_density")


def example_8_backward_compatibility():
    """
    Example 8: Backward compatibility with legacy API.

    This example demonstrates that the enhanced DiversityFilter maintains
    backward compatibility with the previous API. Old code continues to work
    without modifications.
    """
    print("\n" + "=" * 70)
    print("Example 8: Backward Compatibility")
    print("=" * 70)

    # Create sample frames
    frames = create_diverse_frames(15)
    print(f"Created {len(frames)} sample frames")

    # Manually apply tagger (simulating old-style usage)
    hash_tagger = HashTagger(hash_type="dhash")
    for frame in frames:
        frame.tags.update(hash_tagger.compute_tags(frame))

    # Old-style filter usage (still works)
    diversity_filter = DiversityFilter(
        buffer_size=50,
        min_distance=0.1,
        metric="euclidean",  # Legacy parameter
    )

    # Apply filter
    selected_frames = []
    for frame in frames:
        if diversity_filter.should_pass(frame):
            selected_frames.append(frame)

    print("\nResults:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Selected frames: {len(selected_frames)}")
    print(f"  Filtered frames: {len(frames) - len(selected_frames)}")
    print(f"  Selection rate: {len(selected_frames) / len(frames) * 100:.1f}%")

    print(f"\nSelected frame numbers: {[f.frame_number for f in selected_frames]}")

    # Show which tags were used
    if selected_frames:
        print(f"\nTags in first selected frame: {list(selected_frames[0].tags.keys())}")

    print("\nKey Points:")
    print("  - Old API still works (backward compatible)")
    print("  - Legacy 'metric' parameter supported")
    print("  - Auto-detection works as before")
    print("  - New features are opt-in (comparison_strategies, weighted mode)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Enhanced DiversityFilter - Comprehensive Examples")
    print("=" * 70)
    print("\nThese examples demonstrate all features of the enhanced DiversityFilter:")
    print("  1. Hash-based diversity (fast, efficient)")
    print("  2. CLIP embedding diversity (semantic similarity)")
    print("  3. Color histogram diversity (color distribution)")
    print("  4. Combining multiple diversity tags")
    print("  5. Custom comparison strategies")
    print("  6. Weighted combination mode")
    print("  7. Auto-detection of diversity tags")
    print("  8. Backward compatibility")

    # Run all examples
    example_1_hash_based_diversity()
    example_2_clip_embedding_diversity()
    example_3_color_histogram_diversity()
    example_4_combining_multiple_tags()
    example_5_custom_comparison_strategies()
    example_6_weighted_combination_mode()
    example_7_auto_detection()
    example_8_backward_compatibility()

    print("\n" + "=" * 70)
    print("All Examples Complete!")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - docs/DIVERSITY_FILTER_EXAMPLES.md")
    print("  - docs/DIVERSITY_FILTER_MIGRATION.md")
    print("  - docs/API.md")
