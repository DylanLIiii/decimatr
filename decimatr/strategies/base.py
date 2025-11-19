"""
Base class for filter strategies.

FilterStrategy provides a high-level interface for defining complete processing
pipelines. Strategies encapsulate common video processing patterns by combining
taggers and filters into predefined configurations.
"""

from abc import ABC, abstractmethod

from decimatr.filters.base import Filter
from decimatr.taggers.base import Tagger


class FilterStrategy(ABC):
    """
    Abstract base class for filter strategies.

    A FilterStrategy defines a complete processing pipeline by specifying which
    taggers and filters should be applied, and in what order. Strategies provide
    a convenient way to implement common video processing patterns without
    manually configuring individual components.

    Strategies are used by the FrameProcessor to automatically configure
    processing pipelines for common use cases like blur removal, duplicate
    detection, or smart sampling.

    Subclasses must implement:
        - build_pipeline(): Return ordered list of taggers and filters

    Examples:
        >>> class BlurRemovalStrategy(FilterStrategy):
        ...     def __init__(self, threshold: float = 100.0):
        ...         self.threshold = threshold
        ...
        ...     def build_pipeline(self):
        ...         return [
        ...             BlurTagger(),
        ...             BlurFilter(threshold=self.threshold)
        ...         ]

        >>> strategy = BlurRemovalStrategy(threshold=150.0)
        >>> pipeline = strategy.build_pipeline()
        >>> processor = FrameProcessor(strategy=strategy)
    """

    @abstractmethod
    def build_pipeline(self) -> list[Tagger | Filter]:
        """
        Build the complete processing pipeline.

        This method returns an ordered list of taggers and filters that define
        the processing pipeline. The order matters: taggers should come before
        filters that depend on their tags, and filters are applied in sequence.

        The pipeline is executed left-to-right:
        1. All taggers compute their tags and add them to the frame packet
        2. Filters are applied in sequence; if any filter fails, the frame
           is filtered out and subsequent filters are not evaluated

        Returns:
            Ordered list of Tagger and Filter instances that form the pipeline

        Example:
            >>> def build_pipeline(self):
            ...     return [
            ...         # Taggers first: compute all needed tags
            ...         BlurTagger(),
            ...         HashTagger(),
            ...         EntropyTagger(),
            ...         # Filters second: make decisions based on tags
            ...         BlurFilter(threshold=100.0),
            ...         DuplicateFilter(threshold=0.05, buffer_size=50),
            ...         DiversityFilter(window_size=100, k=10)
            ...     ]
        """
        pass
