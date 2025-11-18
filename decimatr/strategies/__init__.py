"""
Filter strategies for video frame processing.

This module provides predefined filtering strategies that combine taggers
and filters into complete processing pipelines for common use cases.
"""

from decimatr.strategies.base import FilterStrategy
from decimatr.strategies.blur_removal import BlurRemovalStrategy
from decimatr.strategies.duplicate_detection import DuplicateDetectionStrategy
from decimatr.strategies.smart_sampling import SmartSamplingStrategy

__all__ = [
    "FilterStrategy",
    "BlurRemovalStrategy",
    "DuplicateDetectionStrategy",
    "SmartSamplingStrategy",
]
