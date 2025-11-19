"""
Actor-based distributed processing components.

This module provides xoscar actor implementations for distributed frame
processing across multiple CPU cores and optional GPU resources.
"""

from decimatr.actors.filter_actor import FilterActor
from decimatr.actors.pipeline import ActorPipeline
from decimatr.actors.stateful_actor import StatefulFilterActor
from decimatr.actors.tagger_actor import TaggerActor

__all__ = [
    "TaggerActor",
    "FilterActor",
    "StatefulFilterActor",
    "ActorPipeline",
]
