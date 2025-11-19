"""
Decimatr: High-Performance Video Frame Processing Library

Decimatr is a modern, actor-based video frame processing library that provides
a clean separation between frame analysis (tagging) and decision-making (filtering).

Quick Start:
    >>> from decimatr.core.processor import FrameProcessor
    >>> processor = FrameProcessor.with_blur_removal(threshold=100.0)
    >>> for frame in processor.process('video.mp4'):
    ...     process_frame(frame)

For more information, see the documentation at:
https://github.com/DylanLIiii/decimatr
"""

__version__ = "0.1.0"
__author__ = "Dylan Li"
__license__ = "MIT"

# Core API
from .core.processor import FrameProcessor, ProcessingResult

# Utilities
from .core.temporal_buffer import TemporalBuffer

# Exceptions
from .exceptions import (
    ActorError,
    ConfigurationError,
    DecimatrError,
    GPUDependencyError,
    ProcessingError,
    TagMissingError,
)
from .filters.base import Filter, StatefulFilter, StatelessFilter

# Common filters
from .filters.blur import BlurFilter
from .filters.diversity import DiversityFilter
from .filters.duplicate import DuplicateFilter
from .filters.entropy import EntropyFilter
from .filters.motion import MotionFilter
from .filters.threshold import ThresholdFilter
from .gpu_utils import GPUCapabilities
from .scheme import VideoFramePacket
from .strategies.base import FilterStrategy

# Predefined strategies
from .strategies.blur_removal import BlurRemovalStrategy
from .strategies.duplicate_detection import DuplicateDetectionStrategy
from .strategies.smart_sampling import SmartSamplingStrategy

# Base classes for custom components
from .taggers.base import Tagger

# Common taggers
from .taggers.blur import BlurTagger
from .taggers.entropy import EntropyTagger
from .taggers.hash import HashTagger
from .video_loader import load_video_frames

__all__ = [
    # Core API
    "FrameProcessor",
    "ProcessingResult",
    "VideoFramePacket",
    "load_video_frames",
    # Base classes
    "Tagger",
    "Filter",
    "StatelessFilter",
    "StatefulFilter",
    "FilterStrategy",
    # Taggers
    "BlurTagger",
    "HashTagger",
    "EntropyTagger",
    # Filters
    "BlurFilter",
    "EntropyFilter",
    "ThresholdFilter",
    "DuplicateFilter",
    "MotionFilter",
    "DiversityFilter",
    # Strategies
    "BlurRemovalStrategy",
    "DuplicateDetectionStrategy",
    "SmartSamplingStrategy",
    # Utilities
    "TemporalBuffer",
    "GPUCapabilities",
    # Exceptions
    "DecimatrError",
    "ConfigurationError",
    "TagMissingError",
    "ProcessingError",
    "ActorError",
    "GPUDependencyError",
]
