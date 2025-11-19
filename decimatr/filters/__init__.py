"""
Filter components for frame selection and decision-making.

Filters determine whether frames should pass through the processing pipeline
based on their tags and temporal context.
"""

from decimatr.filters.base import Filter, StatefulFilter, StatelessFilter
from decimatr.filters.blur import BlurFilter
from decimatr.filters.diversity import DiversityFilter
from decimatr.filters.duplicate import DuplicateFilter
from decimatr.filters.entropy import EntropyFilter
from decimatr.filters.motion import MotionFilter
from decimatr.filters.threshold import ThresholdFilter

__all__ = [
    "Filter",
    "StatelessFilter",
    "StatefulFilter",
    "ThresholdFilter",
    "BlurFilter",
    "EntropyFilter",
    "DuplicateFilter",
    "MotionFilter",
    "DiversityFilter",
]
