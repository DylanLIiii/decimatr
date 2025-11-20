"""Tag classification system for diversity filtering.

This module provides a registry for classifying tags as either diversity-suitable
(appropriate for measuring frame-to-frame diversity) or metric-only (quality metrics
not suitable for diversity comparison).
"""

from enum import Enum


class TagCategory(Enum):
    """Classification of tag types for diversity filtering.

    Attributes:
        DIVERSITY_SUITABLE: Tags appropriate for diversity analysis (hashes, embeddings, histograms)
        METRIC_ONLY: Tags representing quality metrics not suitable for diversity (blur_score, entropy)
    """

    DIVERSITY_SUITABLE = "diversity_suitable"
    METRIC_ONLY = "metric_only"


class TagClassificationRegistry:
    """Registry for classifying tags as diversity-suitable or metric-only.

    This registry maintains default classifications for common tag types used in
    video frame processing. Tags classified as diversity-suitable are appropriate
    for measuring frame-to-frame diversity, while metric-only tags represent
    quality or property metrics that should not be used for diversity comparison.

    Default Classifications:
        Diversity-suitable:
            - Perceptual hashes: dhash, phash, ahash
            - Embeddings: clip_embedding, model_embedding
            - Color histograms: color_hist

        Metric-only:
            - Quality metrics: blur_score, entropy, edge_density

    Unknown tags default to DIVERSITY_SUITABLE to allow custom diversity tags.
    """

    # Default diversity-suitable tags
    DIVERSITY_SUITABLE_TAGS: set[str] = {
        # Perceptual hashes
        "dhash",
        "phash",
        "ahash",
        # Embeddings
        "clip_embedding",
        "model_embedding",
        # Color histograms
        "color_hist",
    }

    # Default metric-only tags
    METRIC_ONLY_TAGS: set[str] = {
        "blur_score",
        "entropy",
        "edge_density",
    }

    @classmethod
    def is_diversity_suitable(cls, tag_key: str) -> bool:
        """Check if a tag is suitable for diversity analysis.

        Args:
            tag_key: The tag key to check

        Returns:
            True if the tag is classified as diversity-suitable, False otherwise

        Note:
            Unknown tags (not in either registry) default to True to allow
            custom diversity tags without requiring registry updates.
        """
        if tag_key in cls.DIVERSITY_SUITABLE_TAGS:
            return True
        if tag_key in cls.METRIC_ONLY_TAGS:
            return False
        # Unknown tags default to diversity-suitable
        return True

    @classmethod
    def is_metric_only(cls, tag_key: str) -> bool:
        """Check if a tag is metric-only (not suitable for diversity).

        Args:
            tag_key: The tag key to check

        Returns:
            True if the tag is classified as metric-only, False otherwise
        """
        return tag_key in cls.METRIC_ONLY_TAGS

    @classmethod
    def get_category(cls, tag_key: str) -> TagCategory:
        """Get the category of a tag.

        Args:
            tag_key: The tag key to classify

        Returns:
            TagCategory.DIVERSITY_SUITABLE or TagCategory.METRIC_ONLY

        Note:
            Unknown tags (not in either registry) default to DIVERSITY_SUITABLE
            to allow custom diversity tags without requiring registry updates.
        """
        if cls.is_metric_only(tag_key):
            return TagCategory.METRIC_ONLY
        else:
            # Both explicitly diversity-suitable and unknown tags
            return TagCategory.DIVERSITY_SUITABLE
