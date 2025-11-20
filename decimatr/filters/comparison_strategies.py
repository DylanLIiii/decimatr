"""
Comparison strategies for computing distances between tag values.

This module provides the base interface and implementations for comparing
different types of tag values (hashes, embeddings, histograms) in the
DiversityFilter. Each strategy implements a specific distance metric
appropriate for its tag type.
"""

from abc import ABC, abstractmethod
from typing import Any


class ComparisonStrategy(ABC):
    """
    Abstract base class for tag comparison strategies.

    Comparison strategies compute distance or dissimilarity between two tag
    values. Different tag types require different comparison methods:
    - Perceptual hashes: Hamming distance
    - Embeddings: Euclidean, cosine, or Manhattan distance
    - Histograms: Intersection, chi-square, or Bhattacharyya distance

    Subclasses must implement:
        - compute_distance(): Calculate distance between two values
        - name: Property returning the strategy name
    """

    @abstractmethod
    def compute_distance(self, value1: Any, value2: Any) -> float:
        """
        Compute distance between two tag values.

        The distance should be a non-negative value where:
        - 0 indicates identical values
        - Higher values indicate greater dissimilarity

        The specific range and interpretation depends on the strategy:
        - Normalized strategies return values in [0, 1]
        - Unnormalized strategies may return unbounded values

        Args:
            value1: First tag value
            value2: Second tag value

        Returns:
            Distance value (higher = more different)

        Raises:
            ValueError: If values are incompatible with this strategy
            TypeError: If values are of incorrect type

        Example:
            >>> strategy = HammingDistanceStrategy()
            >>> hash1 = "a1b2c3d4"
            >>> hash2 = "a1b2c3d5"
            >>> distance = strategy.compute_distance(hash1, hash2)
            >>> print(f"Distance: {distance:.3f}")
            Distance: 0.125
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of this comparison strategy.

        The name should be a short, descriptive identifier for the strategy,
        typically including the metric type (e.g., "hamming", "cosine",
        "euclidean").

        Returns:
            Strategy name as a string

        Example:
            >>> strategy = HammingDistanceStrategy()
            >>> strategy.name
            'hamming'
        """
        pass


import imagehash


class HammingDistanceStrategy(ComparisonStrategy):
    """
    Compute Hamming distance between perceptual hashes.

    This strategy is designed for comparing perceptual hash values (dhash, phash,
    ahash) by computing the Hamming distance - the number of bit positions where
    the two hashes differ. The distance is normalized by the hash bit length to
    produce a value in the range [0, 1].

    The strategy supports both string (hexadecimal) and ImageHash object formats,
    automatically converting between them as needed.

    Attributes:
        None

    Example:
        >>> strategy = HammingDistanceStrategy()
        >>> hash1 = imagehash.phash(image1)
        >>> hash2 = imagehash.phash(image2)
        >>> distance = strategy.compute_distance(hash1, hash2)
        >>> print(f"Normalized distance: {distance:.3f}")
        Normalized distance: 0.125
    """

    def compute_distance(self, value1: Any, value2: Any) -> float:
        """
        Compute normalized Hamming distance between perceptual hashes.

        The Hamming distance is the number of bit positions where the two hashes
        differ. This is normalized by the total number of bits in the hash to
        produce a value in [0, 1], where:
        - 0.0 indicates identical hashes
        - 1.0 indicates completely different hashes

        Args:
            value1: Hash value (string hex representation or ImageHash object)
            value2: Hash value (string hex representation or ImageHash object)

        Returns:
            Normalized distance in range [0, 1]

        Raises:
            ValueError: If values cannot be converted to ImageHash objects
            TypeError: If values are of incompatible types

        Example:
            >>> strategy = HammingDistanceStrategy()
            >>> # Using ImageHash objects
            >>> hash1 = imagehash.phash(image1)
            >>> hash2 = imagehash.phash(image2)
            >>> distance = strategy.compute_distance(hash1, hash2)

            >>> # Using hex strings
            >>> hash_str1 = "a1b2c3d4e5f6a7b8"
            >>> hash_str2 = "a1b2c3d4e5f6a7b9"
            >>> distance = strategy.compute_distance(hash_str1, hash_str2)
        """
        # Convert to ImageHash objects if needed
        hash1 = self._to_imagehash(value1)
        hash2 = self._to_imagehash(value2)

        # Compute Hamming distance using imagehash's built-in subtraction
        # The - operator returns the number of differing bits
        hamming_dist = hash1 - hash2

        # Normalize by hash bit length
        # hash.hash is a numpy array, flatten and count bits
        bit_length = len(hash1.hash.flatten()) * 8
        normalized_dist = hamming_dist / bit_length

        return float(normalized_dist)

    def _to_imagehash(self, value: Any) -> imagehash.ImageHash:
        """
        Convert value to ImageHash object.

        Handles conversion from:
        - ImageHash objects (returned as-is)
        - Hex strings (parsed to ImageHash)

        Args:
            value: Value to convert (ImageHash or hex string)

        Returns:
            ImageHash object

        Raises:
            ValueError: If value cannot be converted to ImageHash
            TypeError: If value is of unsupported type
        """
        if isinstance(value, imagehash.ImageHash):
            return value
        elif isinstance(value, str):
            # Parse hex string to ImageHash
            try:
                return imagehash.hex_to_hash(value)
            except Exception as e:
                raise ValueError(f"Cannot parse hex string '{value}' to ImageHash: {e}") from e
        else:
            raise TypeError(
                f"Cannot convert {type(value).__name__} to ImageHash. "
                f"Expected ImageHash object or hex string."
            )

    @property
    def name(self) -> str:
        """
        Return the name of this comparison strategy.

        Returns:
            "hamming" - identifier for Hamming distance strategy
        """
        return "hamming"


import numpy as np


class EmbeddingDistanceStrategy(ComparisonStrategy):
    """
    Compute distance between embedding vectors.

    This strategy is designed for comparing high-dimensional embedding vectors
    (e.g., CLIP embeddings, model embeddings) using various distance metrics.
    Supports euclidean, cosine, and manhattan distance metrics.

    For cosine distance, vectors are automatically normalized before comparison.
    Vectors of different lengths are handled by padding the shorter vector with
    zeros. Zero vectors are handled gracefully by returning maximum distance.

    Attributes:
        metric: Distance metric to use ('euclidean', 'cosine', 'manhattan')

    Example:
        >>> strategy = EmbeddingDistanceStrategy(metric='cosine')
        >>> emb1 = np.array([0.1, 0.2, 0.3, 0.4])
        >>> emb2 = np.array([0.15, 0.25, 0.35, 0.45])
        >>> distance = strategy.compute_distance(emb1, emb2)
        >>> print(f"Cosine distance: {distance:.3f}")
        Cosine distance: 0.001
    """

    def __init__(self, metric: str = "cosine"):
        """
        Initialize embedding distance strategy.

        Args:
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')

        Raises:
            ValueError: If metric is not one of the valid options

        Example:
            >>> # Cosine distance (default, good for normalized embeddings)
            >>> strategy = EmbeddingDistanceStrategy(metric='cosine')

            >>> # Euclidean distance (L2 norm)
            >>> strategy = EmbeddingDistanceStrategy(metric='euclidean')

            >>> # Manhattan distance (L1 norm)
            >>> strategy = EmbeddingDistanceStrategy(metric='manhattan')
        """
        valid_metrics = ["euclidean", "cosine", "manhattan"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")
        self.metric = metric

    def compute_distance(self, value1: Any, value2: Any) -> float:
        """
        Compute distance between embedding vectors.

        Converts input values to numpy arrays and computes the specified distance
        metric. Handles vectors of different lengths by padding with zeros.
        For cosine distance, vectors are normalized before comparison.

        Distance interpretations:
        - Euclidean: Unbounded, 0 = identical, higher = more different
        - Manhattan: Unbounded, 0 = identical, higher = more different
        - Cosine: [0, 2], 0 = identical direction, 2 = opposite direction

        Args:
            value1: Embedding vector (numpy array, list, or array-like)
            value2: Embedding vector (numpy array, list, or array-like)

        Returns:
            Distance value based on the configured metric

        Raises:
            ValueError: If values cannot be converted to numeric arrays

        Example:
            >>> strategy = EmbeddingDistanceStrategy(metric='euclidean')
            >>> vec1 = [1.0, 2.0, 3.0]
            >>> vec2 = [1.5, 2.5, 3.5]
            >>> distance = strategy.compute_distance(vec1, vec2)
            >>> print(f"Euclidean distance: {distance:.3f}")
            Euclidean distance: 0.866
        """
        # Convert to numpy arrays and flatten
        vec1 = np.array(value1, dtype=float).flatten()
        vec2 = np.array(value2, dtype=float).flatten()

        # Ensure same length by padding with zeros
        if len(vec1) != len(vec2):
            max_len = max(len(vec1), len(vec2))
            vec1 = np.pad(vec1, (0, max_len - len(vec1)))
            vec2 = np.pad(vec2, (0, max_len - len(vec2)))

        if self.metric == "euclidean":
            # Euclidean distance (L2 norm)
            return float(np.linalg.norm(vec1 - vec2))

        elif self.metric == "manhattan":
            # Manhattan distance (L1 norm)
            return float(np.sum(np.abs(vec1 - vec2)))

        elif self.metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            # Normalize vectors first
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            # Handle zero vectors gracefully
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance for zero vectors

            vec1_normalized = vec1 / norm1
            vec2_normalized = vec2 / norm2

            # Cosine similarity = dot product of normalized vectors
            cosine_sim = np.dot(vec1_normalized, vec2_normalized)

            # Cosine distance = 1 - cosine similarity
            # Clamp to [0, 2] range to handle numerical errors
            cosine_dist = 1.0 - cosine_sim
            return float(np.clip(cosine_dist, 0.0, 2.0))

        # Should never reach here due to validation in __init__
        return 0.0

    @property
    def name(self) -> str:
        """
        Return the name of this comparison strategy.

        Returns:
            Strategy name in format "embedding_{metric}"

        Example:
            >>> strategy = EmbeddingDistanceStrategy(metric='cosine')
            >>> strategy.name
            'embedding_cosine'
        """
        return f"embedding_{self.metric}"


class HistogramDistanceStrategy(ComparisonStrategy):
    """
    Compute distance between color histograms.

    This strategy is designed for comparing color histogram distributions using
    various histogram comparison metrics. Supports intersection, chi-square, and
    Bhattacharyya distance metrics.

    Histograms are automatically normalized before comparison to ensure they
    represent probability distributions. Histograms of different lengths are
    handled by padding the shorter histogram with zeros.

    Attributes:
        metric: Distance metric to use ('intersection', 'chi_square', 'bhattacharyya')

    Example:
        >>> strategy = HistogramDistanceStrategy(metric='intersection')
        >>> hist1 = np.array([10, 20, 30, 40])
        >>> hist2 = np.array([15, 25, 35, 45])
        >>> distance = strategy.compute_distance(hist1, hist2)
        >>> print(f"Histogram distance: {distance:.3f}")
        Histogram distance: 0.067
    """

    def __init__(self, metric: str = "intersection"):
        """
        Initialize histogram distance strategy.

        Args:
            metric: Distance metric ('intersection', 'chi_square', 'bhattacharyya')

        Raises:
            ValueError: If metric is not one of the valid options

        Example:
            >>> # Histogram intersection (default, similarity-based)
            >>> strategy = HistogramDistanceStrategy(metric='intersection')

            >>> # Chi-square distance (statistical measure)
            >>> strategy = HistogramDistanceStrategy(metric='chi_square')

            >>> # Bhattacharyya distance (probabilistic measure)
            >>> strategy = HistogramDistanceStrategy(metric='bhattacharyya')
        """
        valid_metrics = ["intersection", "chi_square", "bhattacharyya"]
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")
        self.metric = metric

    def compute_distance(self, value1: Any, value2: Any) -> float:
        """
        Compute distance between color histograms.

        Converts input values to numpy arrays, normalizes them to probability
        distributions, and computes the specified distance metric. Handles
        histograms of different lengths by padding with zeros.

        Distance interpretations:
        - Intersection: [0, 1], 0 = identical, 1 = completely different
        - Chi-square: [0, ∞), 0 = identical, higher = more different
        - Bhattacharyya: [0, ∞), 0 = identical, higher = more different

        Args:
            value1: Histogram (numpy array, list, or array-like)
            value2: Histogram (numpy array, list, or array-like)

        Returns:
            Distance value based on the configured metric

        Raises:
            ValueError: If values cannot be converted to numeric arrays

        Example:
            >>> strategy = HistogramDistanceStrategy(metric='intersection')
            >>> hist1 = [100, 200, 300, 400]
            >>> hist2 = [120, 220, 320, 420]
            >>> distance = strategy.compute_distance(hist1, hist2)
            >>> print(f"Intersection distance: {distance:.3f}")
            Intersection distance: 0.020
        """
        # Convert to numpy arrays and flatten
        hist1 = np.array(value1, dtype=float).flatten()
        hist2 = np.array(value2, dtype=float).flatten()

        # Normalize histograms to probability distributions
        # Add small epsilon to avoid division by zero
        hist1 = hist1 / (np.sum(hist1) + 1e-10)
        hist2 = hist2 / (np.sum(hist2) + 1e-10)

        # Ensure same length by padding with zeros
        if len(hist1) != len(hist2):
            max_len = max(len(hist1), len(hist2))
            hist1 = np.pad(hist1, (0, max_len - len(hist1)))
            hist2 = np.pad(hist2, (0, max_len - len(hist2)))

        if self.metric == "intersection":
            # Histogram intersection measures similarity
            # intersection = sum of minimum values at each bin
            # We convert to distance: distance = 1 - intersection
            intersection = np.sum(np.minimum(hist1, hist2))
            return float(1.0 - intersection)

        elif self.metric == "chi_square":
            # Chi-square distance
            # chi_square = sum((h1 - h2)^2 / (h1 + h2))
            # Add epsilon to denominator to avoid division by zero
            chi_square = np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))
            return float(chi_square)

        elif self.metric == "bhattacharyya":
            # Bhattacharyya distance
            # BC = sum(sqrt(h1 * h2))  (Bhattacharyya coefficient)
            # distance = -ln(BC)
            bc_coeff = np.sum(np.sqrt(hist1 * hist2))
            # Add epsilon to avoid log(0)
            # Clamp bc_coeff to [0, 1] to handle numerical errors
            bc_coeff = np.clip(bc_coeff, 1e-10, 1.0)
            return float(-np.log(bc_coeff))

        # Should never reach here due to validation in __init__
        return 0.0

    @property
    def name(self) -> str:
        """
        Return the name of this comparison strategy.

        Returns:
            Strategy name in format "histogram_{metric}"

        Example:
            >>> strategy = HistogramDistanceStrategy(metric='intersection')
            >>> strategy.name
            'histogram_intersection'
        """
        return f"histogram_{self.metric}"
