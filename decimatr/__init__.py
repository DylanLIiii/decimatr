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
https://github.com/yourusername/decimatr
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Core API
from .core.processor import FrameProcessor, ProcessingResult
from .scheme import VideoFramePacket
from .video_loader import load_video_frames

# Base classes for custom components
from .taggers.base import Tagger
from .filters.base import Filter, StatelessFilter, StatefulFilter
from .strategies.base import FilterStrategy

# Common taggers
from .taggers.blur import BlurTagger
from .taggers.hash import HashTagger
from .taggers.entropy import EntropyTagger

# Common filters
from .filters.blur import BlurFilter
from .filters.entropy import EntropyFilter
from .filters.threshold import ThresholdFilter
from .filters.duplicate import DuplicateFilter
from .filters.motion import MotionFilter
from .filters.diversity import DiversityFilter

# Predefined strategies
from .strategies.blur_removal import BlurRemovalStrategy
from .strategies.duplicate_detection import DuplicateDetectionStrategy
from .strategies.smart_sampling import SmartSamplingStrategy

# Utilities
from .core.temporal_buffer import TemporalBuffer
from .gpu_utils import GPUCapabilities

# Exceptions
from .exceptions import (
    DecimatrError,
    ConfigurationError,
    TagMissingError,
    ProcessingError,
    ActorError,
    GPUDependencyError
)

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