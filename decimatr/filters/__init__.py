"""
Filter components for frame selection and decision-making.

Filters determine whether frames should pass through the processing pipeline
based on their tags and temporal context.
"""

from decimatr.filters.base import Filter, StatelessFilter, StatefulFilter

__all__ = [
    "Filter",
    "StatelessFilter",
    "StatefulFilter",
]
