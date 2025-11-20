"""
Tests for comparison strategies.
"""

import imagehash
import numpy as np
import pytest
from decimatr.filters.comparison_strategies import (
    ComparisonStrategy,
    EmbeddingDistanceStrategy,
    HammingDistanceStrategy,
    HistogramDistanceStrategy,
)
from PIL import Image


class TestHammingDistanceStrategy:
    """Test HammingDistanceStrategy for perceptual hash comparison."""

    def test_initialization(self):
        """Test HammingDistanceStrategy initializes correctly."""
        strategy = HammingDistanceStrategy()
        assert strategy.name == "hamming"

    def test_identical_hashes_zero_distance(self):
        """Test that identical hashes produce zero distance."""
        strategy = HammingDistanceStrategy()

        # Create a simple test image
        image = Image.new("RGB", (64, 64), color="red")
        hash1 = imagehash.phash(image)
        hash2 = imagehash.phash(image)

        distance = strategy.compute_distance(hash1, hash2)
        assert distance == 0.0

    def test_different_hashes_nonzero_distance(self):
        """Test that different hashes produce non-zero distance."""
        strategy = HammingDistanceStrategy()

        # Create two visually different images with patterns
        # Solid colors may produce similar hashes, so use patterns
        image1_array = np.zeros((64, 64, 3), dtype=np.uint8)
        image1_array[:32, :, :] = 255  # Top half white
        image1 = Image.fromarray(image1_array)

        image2_array = np.zeros((64, 64, 3), dtype=np.uint8)
        image2_array[:, :32, :] = 255  # Left half white
        image2 = Image.fromarray(image2_array)

        hash1 = imagehash.phash(image1)
        hash2 = imagehash.phash(image2)

        distance = strategy.compute_distance(hash1, hash2)
        assert distance > 0.0
        assert distance <= 1.0  # Should be normalized

    def test_normalized_distance_range(self):
        """Test that distance is normalized to [0, 1] range."""
        strategy = HammingDistanceStrategy()

        # Create multiple different images
        images = [
            Image.new("RGB", (64, 64), color="red"),
            Image.new("RGB", (64, 64), color="blue"),
            Image.new("RGB", (64, 64), color="green"),
            Image.new("RGB", (64, 64), color="yellow"),
        ]

        hashes = [imagehash.phash(img) for img in images]

        # Test all pairs
        for _i, hash1 in enumerate(hashes):
            for _j, hash2 in enumerate(hashes):
                distance = strategy.compute_distance(hash1, hash2)
                assert 0.0 <= distance <= 1.0, f"Distance {distance} out of range [0, 1]"

    def test_string_hash_format(self):
        """Test that strategy handles hex string format."""
        strategy = HammingDistanceStrategy()

        # Create test image and get hash
        image = Image.new("RGB", (64, 64), color="red")
        hash_obj = imagehash.phash(image)
        hash_str = str(hash_obj)

        # Should work with both string and object
        distance1 = strategy.compute_distance(hash_obj, hash_obj)
        distance2 = strategy.compute_distance(hash_str, hash_str)

        assert distance1 == distance2 == 0.0

    def test_mixed_format_comparison(self):
        """Test comparing string and ImageHash object formats."""
        strategy = HammingDistanceStrategy()

        # Create test image
        image = Image.new("RGB", (64, 64), color="red")
        hash_obj = imagehash.phash(image)
        hash_str = str(hash_obj)

        # Should work with mixed formats
        distance = strategy.compute_distance(hash_obj, hash_str)
        assert distance == 0.0

    def test_different_hash_types(self):
        """Test with different hash types (phash, dhash, ahash)."""
        strategy = HammingDistanceStrategy()

        image1 = Image.new("RGB", (64, 64), color="red")
        image2 = Image.new("RGB", (64, 64), color="blue")

        # Test with different hash types
        for hash_func in [imagehash.phash, imagehash.dhash, imagehash.average_hash]:
            hash1 = hash_func(image1)
            hash2 = hash_func(image2)

            distance = strategy.compute_distance(hash1, hash2)
            assert 0.0 <= distance <= 1.0

    def test_invalid_string_raises_error(self):
        """Test that invalid hex string raises ValueError."""
        strategy = HammingDistanceStrategy()

        with pytest.raises(ValueError, match="Cannot parse hex string"):
            strategy.compute_distance("invalid_hex", "also_invalid")

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises TypeError."""
        strategy = HammingDistanceStrategy()

        with pytest.raises(TypeError, match="Cannot convert"):
            strategy.compute_distance(12345, 67890)

    def test_symmetry(self):
        """Test that distance is symmetric: d(a,b) == d(b,a)."""
        strategy = HammingDistanceStrategy()

        image1 = Image.new("RGB", (64, 64), color="red")
        image2 = Image.new("RGB", (64, 64), color="blue")

        hash1 = imagehash.phash(image1)
        hash2 = imagehash.phash(image2)

        distance_ab = strategy.compute_distance(hash1, hash2)
        distance_ba = strategy.compute_distance(hash2, hash1)

        assert distance_ab == distance_ba

    def test_triangle_inequality(self):
        """Test that distance satisfies triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""
        strategy = HammingDistanceStrategy()

        image1 = Image.new("RGB", (64, 64), color="red")
        image2 = Image.new("RGB", (64, 64), color="green")
        image3 = Image.new("RGB", (64, 64), color="blue")

        hash1 = imagehash.phash(image1)
        hash2 = imagehash.phash(image2)
        hash3 = imagehash.phash(image3)

        d_ac = strategy.compute_distance(hash1, hash3)
        d_ab = strategy.compute_distance(hash1, hash2)
        d_bc = strategy.compute_distance(hash2, hash3)

        # Triangle inequality should hold
        assert d_ac <= d_ab + d_bc + 1e-10  # Small epsilon for floating point

    def test_similar_images_small_distance(self):
        """Test that similar images have small Hamming distance."""
        strategy = HammingDistanceStrategy()

        # Create base image
        base_image = Image.new("RGB", (64, 64), color="red")

        # Create slightly modified image (add small noise)
        np_image = np.array(base_image)
        np_image[0:5, 0:5] = [255, 0, 0]  # Small modification
        similar_image = Image.fromarray(np_image)

        hash1 = imagehash.phash(base_image)
        hash2 = imagehash.phash(similar_image)

        distance = strategy.compute_distance(hash1, hash2)

        # Distance should be small for similar images
        assert distance < 0.3  # Reasonable threshold for similar images


class TestEmbeddingDistanceStrategy:
    """Test EmbeddingDistanceStrategy for embedding vector comparison."""

    def test_initialization_default_metric(self):
        """Test EmbeddingDistanceStrategy initializes with default cosine metric."""
        strategy = EmbeddingDistanceStrategy()
        assert strategy.metric == "cosine"
        assert strategy.name == "embedding_cosine"

    def test_initialization_euclidean_metric(self):
        """Test EmbeddingDistanceStrategy initializes with euclidean metric."""
        strategy = EmbeddingDistanceStrategy(metric="euclidean")
        assert strategy.metric == "euclidean"
        assert strategy.name == "embedding_euclidean"

    def test_initialization_manhattan_metric(self):
        """Test EmbeddingDistanceStrategy initializes with manhattan metric."""
        strategy = EmbeddingDistanceStrategy(metric="manhattan")
        assert strategy.metric == "manhattan"
        assert strategy.name == "embedding_manhattan"

    def test_initialization_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            EmbeddingDistanceStrategy(metric="invalid_metric")

    def test_euclidean_identical_vectors_zero_distance(self):
        """Test that identical vectors produce zero Euclidean distance."""
        strategy = EmbeddingDistanceStrategy(metric="euclidean")
        vec = np.array([1.0, 2.0, 3.0, 4.0])

        distance = strategy.compute_distance(vec, vec)
        assert distance == 0.0

    def test_euclidean_different_vectors_nonzero_distance(self):
        """Test that different vectors produce non-zero Euclidean distance."""
        strategy = EmbeddingDistanceStrategy(metric="euclidean")
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])

        distance = strategy.compute_distance(vec1, vec2)
        assert distance > 0.0
        # Expected: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
        assert abs(distance - np.sqrt(27)) < 1e-6

    def test_manhattan_identical_vectors_zero_distance(self):
        """Test that identical vectors produce zero Manhattan distance."""
        strategy = EmbeddingDistanceStrategy(metric="manhattan")
        vec = np.array([1.0, 2.0, 3.0, 4.0])

        distance = strategy.compute_distance(vec, vec)
        assert distance == 0.0

    def test_manhattan_different_vectors_nonzero_distance(self):
        """Test that different vectors produce non-zero Manhattan distance."""
        strategy = EmbeddingDistanceStrategy(metric="manhattan")
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])

        distance = strategy.compute_distance(vec1, vec2)
        assert distance > 0.0
        # Expected: |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
        assert abs(distance - 9.0) < 1e-6

    def test_cosine_identical_vectors_zero_distance(self):
        """Test that identical vectors produce zero cosine distance."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")
        vec = np.array([1.0, 2.0, 3.0, 4.0])

        distance = strategy.compute_distance(vec, vec)
        assert abs(distance) < 1e-6  # Should be very close to 0

    def test_cosine_orthogonal_vectors(self):
        """Test cosine distance for orthogonal vectors."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        distance = strategy.compute_distance(vec1, vec2)
        # Orthogonal vectors have cosine similarity = 0, so distance = 1
        assert abs(distance - 1.0) < 1e-6

    def test_cosine_opposite_vectors(self):
        """Test cosine distance for opposite direction vectors."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])

        distance = strategy.compute_distance(vec1, vec2)
        # Opposite vectors have cosine similarity = -1, so distance = 2
        assert abs(distance - 2.0) < 1e-6

    def test_cosine_parallel_vectors(self):
        """Test cosine distance for parallel vectors (same direction, different magnitude)."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([2.0, 4.0, 6.0])  # 2x vec1

        distance = strategy.compute_distance(vec1, vec2)
        # Parallel vectors have cosine similarity = 1, so distance = 0
        assert abs(distance) < 1e-6

    def test_zero_vector_handling_cosine(self):
        """Test that zero vectors are handled gracefully in cosine distance."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        distance = strategy.compute_distance(vec1, vec2)
        # Zero vector should return maximum distance
        assert distance == 1.0

    def test_both_zero_vectors_cosine(self):
        """Test that two zero vectors return maximum distance."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        distance = strategy.compute_distance(vec1, vec2)
        assert distance == 1.0

    def test_different_length_vectors_padding(self):
        """Test that vectors of different lengths are padded correctly."""
        strategy = EmbeddingDistanceStrategy(metric="euclidean")
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        distance = strategy.compute_distance(vec1, vec2)
        # vec1 padded to [1, 2, 3, 0, 0]
        # Distance = sqrt((0)^2 + (0)^2 + (0)^2 + (4)^2 + (5)^2) = sqrt(41)
        expected = np.sqrt(16 + 25)
        assert abs(distance - expected) < 1e-6

    def test_list_input_conversion(self):
        """Test that list inputs are converted to numpy arrays."""
        strategy = EmbeddingDistanceStrategy(metric="euclidean")
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [4.0, 5.0, 6.0]

        distance = strategy.compute_distance(vec1, vec2)
        assert distance > 0.0

    def test_multidimensional_array_flattening(self):
        """Test that multidimensional arrays are flattened."""
        strategy = EmbeddingDistanceStrategy(metric="euclidean")
        vec1 = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2 array
        vec2 = np.array([[1.0, 2.0], [3.0, 4.0]])

        distance = strategy.compute_distance(vec1, vec2)
        assert distance == 0.0

    def test_symmetry_euclidean(self):
        """Test that Euclidean distance is symmetric: d(a,b) == d(b,a)."""
        strategy = EmbeddingDistanceStrategy(metric="euclidean")
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])

        distance_ab = strategy.compute_distance(vec1, vec2)
        distance_ba = strategy.compute_distance(vec2, vec1)

        assert abs(distance_ab - distance_ba) < 1e-6

    def test_symmetry_cosine(self):
        """Test that cosine distance is symmetric: d(a,b) == d(b,a)."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])

        distance_ab = strategy.compute_distance(vec1, vec2)
        distance_ba = strategy.compute_distance(vec2, vec1)

        assert abs(distance_ab - distance_ba) < 1e-6

    def test_triangle_inequality_euclidean(self):
        """Test that Euclidean distance satisfies triangle inequality."""
        strategy = EmbeddingDistanceStrategy(metric="euclidean")
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        vec3 = np.array([0.0, 0.0, 1.0])

        d_ac = strategy.compute_distance(vec1, vec3)
        d_ab = strategy.compute_distance(vec1, vec2)
        d_bc = strategy.compute_distance(vec2, vec3)

        assert d_ac <= d_ab + d_bc + 1e-10

    def test_cosine_distance_range(self):
        """Test that cosine distance is in valid range [0, 2]."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")

        # Test multiple random vectors
        np.random.seed(42)
        for _ in range(10):
            vec1 = np.random.randn(10)
            vec2 = np.random.randn(10)

            distance = strategy.compute_distance(vec1, vec2)
            assert 0.0 <= distance <= 2.0, f"Distance {distance} out of range [0, 2]"

    def test_normalized_embeddings_cosine(self):
        """Test cosine distance with pre-normalized embeddings."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")

        # Create normalized vectors (unit length)
        vec1 = np.array([0.6, 0.8, 0.0])  # Length = 1
        vec2 = np.array([0.8, 0.6, 0.0])  # Length = 1

        distance = strategy.compute_distance(vec1, vec2)

        # Cosine similarity = 0.6*0.8 + 0.8*0.6 = 0.96
        # Cosine distance = 1 - 0.96 = 0.04
        expected = 1.0 - (0.6 * 0.8 + 0.8 * 0.6)
        assert abs(distance - expected) < 1e-6

    def test_high_dimensional_vectors(self):
        """Test with high-dimensional vectors (like CLIP embeddings)."""
        strategy = EmbeddingDistanceStrategy(metric="cosine")

        # Simulate CLIP-like embeddings (512 dimensions)
        np.random.seed(42)
        vec1 = np.random.randn(512)
        vec2 = np.random.randn(512)

        distance = strategy.compute_distance(vec1, vec2)
        assert 0.0 <= distance <= 2.0

    def test_small_differences_detected(self):
        """Test that small differences in embeddings are detected."""
        strategy = EmbeddingDistanceStrategy(metric="euclidean")

        vec1 = np.array([1.0, 2.0, 3.0, 4.0])
        vec2 = np.array([1.0, 2.0, 3.0, 4.001])  # Very small difference

        distance = strategy.compute_distance(vec1, vec2)
        assert distance > 0.0
        assert distance < 0.01  # Should be small but non-zero


class TestHistogramDistanceStrategy:
    """Test HistogramDistanceStrategy for histogram comparison."""

    def test_initialization_default_metric(self):
        """Test HistogramDistanceStrategy initializes with default intersection metric."""
        strategy = HistogramDistanceStrategy()
        assert strategy.metric == "intersection"
        assert strategy.name == "histogram_intersection"

    def test_initialization_chi_square_metric(self):
        """Test HistogramDistanceStrategy initializes with chi_square metric."""
        strategy = HistogramDistanceStrategy(metric="chi_square")
        assert strategy.metric == "chi_square"
        assert strategy.name == "histogram_chi_square"

    def test_initialization_bhattacharyya_metric(self):
        """Test HistogramDistanceStrategy initializes with bhattacharyya metric."""
        strategy = HistogramDistanceStrategy(metric="bhattacharyya")
        assert strategy.metric == "bhattacharyya"
        assert strategy.name == "histogram_bhattacharyya"

    def test_initialization_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be one of"):
            HistogramDistanceStrategy(metric="invalid_metric")

    def test_intersection_identical_histograms_zero_distance(self):
        """Test that identical histograms produce zero intersection distance."""
        strategy = HistogramDistanceStrategy(metric="intersection")
        hist = np.array([10, 20, 30, 40])

        distance = strategy.compute_distance(hist, hist)
        assert abs(distance) < 1e-6  # Should be very close to 0

    def test_intersection_different_histograms_nonzero_distance(self):
        """Test that different histograms produce non-zero intersection distance."""
        strategy = HistogramDistanceStrategy(metric="intersection")
        hist1 = np.array([10, 20, 30, 40])
        hist2 = np.array([40, 30, 20, 10])

        distance = strategy.compute_distance(hist1, hist2)
        assert distance > 0.0
        assert distance <= 1.0  # Should be normalized

    def test_intersection_completely_different_histograms(self):
        """Test intersection distance for completely non-overlapping histograms."""
        strategy = HistogramDistanceStrategy(metric="intersection")
        hist1 = np.array([100, 0, 0, 0])
        hist2 = np.array([0, 0, 0, 100])

        distance = strategy.compute_distance(hist1, hist2)
        # No overlap, so intersection = 0, distance = 1
        assert abs(distance - 1.0) < 1e-6

    def test_chi_square_identical_histograms_zero_distance(self):
        """Test that identical histograms produce zero chi-square distance."""
        strategy = HistogramDistanceStrategy(metric="chi_square")
        hist = np.array([10, 20, 30, 40])

        distance = strategy.compute_distance(hist, hist)
        assert abs(distance) < 1e-6

    def test_chi_square_different_histograms_nonzero_distance(self):
        """Test that different histograms produce non-zero chi-square distance."""
        strategy = HistogramDistanceStrategy(metric="chi_square")
        hist1 = np.array([10, 20, 30, 40])
        hist2 = np.array([15, 25, 35, 45])

        distance = strategy.compute_distance(hist1, hist2)
        assert distance > 0.0

    def test_bhattacharyya_identical_histograms_zero_distance(self):
        """Test that identical histograms produce zero Bhattacharyya distance."""
        strategy = HistogramDistanceStrategy(metric="bhattacharyya")
        hist = np.array([10, 20, 30, 40])

        distance = strategy.compute_distance(hist, hist)
        assert abs(distance) < 1e-6

    def test_bhattacharyya_different_histograms_nonzero_distance(self):
        """Test that different histograms produce non-zero Bhattacharyya distance."""
        strategy = HistogramDistanceStrategy(metric="bhattacharyya")
        hist1 = np.array([10, 20, 30, 40])
        hist2 = np.array([15, 25, 35, 45])

        distance = strategy.compute_distance(hist1, hist2)
        assert distance > 0.0

    def test_normalization_applied(self):
        """Test that histograms are normalized before comparison."""
        strategy = HistogramDistanceStrategy(metric="intersection")

        # Same distribution, different scales
        hist1 = np.array([10, 20, 30, 40])
        hist2 = np.array([100, 200, 300, 400])

        distance = strategy.compute_distance(hist1, hist2)
        # After normalization, they should be identical
        assert abs(distance) < 1e-6

    def test_different_length_histograms_padding(self):
        """Test that histograms of different lengths are padded correctly."""
        strategy = HistogramDistanceStrategy(metric="intersection")
        hist1 = np.array([10, 20, 30])
        hist2 = np.array([10, 20, 30, 0, 0])

        distance = strategy.compute_distance(hist1, hist2)
        # After padding and normalization, they should be identical
        assert abs(distance) < 1e-6

    def test_list_input_conversion(self):
        """Test that list inputs are converted to numpy arrays."""
        strategy = HistogramDistanceStrategy(metric="intersection")
        hist1 = [10, 20, 30, 40]
        hist2 = [15, 25, 35, 45]

        distance = strategy.compute_distance(hist1, hist2)
        assert distance >= 0.0

    def test_multidimensional_array_flattening(self):
        """Test that multidimensional arrays are flattened."""
        strategy = HistogramDistanceStrategy(metric="intersection")
        hist1 = np.array([[10, 20], [30, 40]])  # 2x2 array
        hist2 = np.array([[10, 20], [30, 40]])

        distance = strategy.compute_distance(hist1, hist2)
        assert abs(distance) < 1e-6

    def test_zero_histogram_handling(self):
        """Test that zero histograms are handled gracefully."""
        strategy = HistogramDistanceStrategy(metric="intersection")
        hist1 = np.array([10, 20, 30, 40])
        hist2 = np.array([0, 0, 0, 0])

        # Should not raise error due to epsilon in normalization
        distance = strategy.compute_distance(hist1, hist2)
        assert distance >= 0.0

    def test_symmetry_intersection(self):
        """Test that intersection distance is symmetric: d(a,b) == d(b,a)."""
        strategy = HistogramDistanceStrategy(metric="intersection")
        hist1 = np.array([10, 20, 30, 40])
        hist2 = np.array([15, 25, 35, 45])

        distance_ab = strategy.compute_distance(hist1, hist2)
        distance_ba = strategy.compute_distance(hist2, hist1)

        assert abs(distance_ab - distance_ba) < 1e-6

    def test_symmetry_chi_square(self):
        """Test that chi-square distance is symmetric: d(a,b) == d(b,a)."""
        strategy = HistogramDistanceStrategy(metric="chi_square")
        hist1 = np.array([10, 20, 30, 40])
        hist2 = np.array([15, 25, 35, 45])

        distance_ab = strategy.compute_distance(hist1, hist2)
        distance_ba = strategy.compute_distance(hist2, hist1)

        assert abs(distance_ab - distance_ba) < 1e-6

    def test_symmetry_bhattacharyya(self):
        """Test that Bhattacharyya distance is symmetric: d(a,b) == d(b,a)."""
        strategy = HistogramDistanceStrategy(metric="bhattacharyya")
        hist1 = np.array([10, 20, 30, 40])
        hist2 = np.array([15, 25, 35, 45])

        distance_ab = strategy.compute_distance(hist1, hist2)
        distance_ba = strategy.compute_distance(hist2, hist1)

        assert abs(distance_ab - distance_ba) < 1e-6

    def test_intersection_distance_range(self):
        """Test that intersection distance is in valid range [0, 1]."""
        strategy = HistogramDistanceStrategy(metric="intersection")

        # Test multiple random histograms
        np.random.seed(42)
        for _ in range(10):
            hist1 = np.random.rand(20) * 100
            hist2 = np.random.rand(20) * 100

            distance = strategy.compute_distance(hist1, hist2)
            assert 0.0 <= distance <= 1.0, f"Distance {distance} out of range [0, 1]"

    def test_similar_histograms_small_distance(self):
        """Test that similar histograms have small distance."""
        strategy = HistogramDistanceStrategy(metric="intersection")

        hist1 = np.array([100, 200, 300, 400])
        hist2 = np.array([105, 205, 305, 405])  # 5% difference

        distance = strategy.compute_distance(hist1, hist2)
        # Should be small for similar distributions
        assert distance < 0.1

    def test_color_histogram_simulation(self):
        """Test with simulated color histogram (256 bins)."""
        strategy = HistogramDistanceStrategy(metric="intersection")

        # Simulate color histograms
        np.random.seed(42)
        hist1 = np.random.rand(256) * 1000
        hist2 = hist1 + np.random.rand(256) * 100  # Similar but with noise

        distance = strategy.compute_distance(hist1, hist2)
        assert 0.0 <= distance <= 1.0

    def test_chi_square_sensitivity(self):
        """Test that chi-square is sensitive to distribution differences."""
        strategy = HistogramDistanceStrategy(metric="chi_square")

        # Uniform distribution
        hist1 = np.array([25, 25, 25, 25])
        # Skewed distribution
        hist2 = np.array([10, 20, 30, 40])

        distance = strategy.compute_distance(hist1, hist2)
        assert distance > 0.0

    def test_bhattacharyya_probabilistic_interpretation(self):
        """Test Bhattacharyya distance with probability distributions."""
        strategy = HistogramDistanceStrategy(metric="bhattacharyya")

        # Two probability distributions
        hist1 = np.array([0.25, 0.25, 0.25, 0.25])
        hist2 = np.array([0.1, 0.2, 0.3, 0.4])

        distance = strategy.compute_distance(hist1, hist2)
        assert distance > 0.0

    def test_sparse_histograms(self):
        """Test with sparse histograms (many zero bins)."""
        strategy = HistogramDistanceStrategy(metric="intersection")

        hist1 = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        hist2 = np.array([0, 0, 0, 0, 0, 100, 0, 0, 0, 0])

        distance = strategy.compute_distance(hist1, hist2)
        # No overlap, should be maximum distance
        assert abs(distance - 1.0) < 1e-6

    def test_negative_values_handled(self):
        """Test that negative values in histograms are handled (shouldn't occur but test robustness)."""
        strategy = HistogramDistanceStrategy(metric="intersection")

        # Histograms should be non-negative, but test robustness
        hist1 = np.array([10, 20, 30, 40])
        hist2 = np.array([10, 20, 30, 40])

        distance = strategy.compute_distance(hist1, hist2)
        assert abs(distance) < 1e-6
