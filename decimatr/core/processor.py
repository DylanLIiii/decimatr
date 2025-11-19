"""
Core FrameProcessor API for video frame processing.

This module provides the main entry point for processing video frames through
configurable pipelines of taggers and filters. It supports both custom pipelines
and predefined strategies for common use cases.
"""

import asyncio
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from decimatr.exceptions import ConfigurationError
from decimatr.filters.base import Filter
from decimatr.scheme import VideoFramePacket
from decimatr.strategies.base import FilterStrategy
from decimatr.taggers.base import Tagger

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """
    Summary of a frame processing session.

    This class captures metrics and statistics from a processing session,
    including frame counts, processing time, stage-level metrics, actor
    health metrics, and any errors encountered.

    Attributes:
        session_id: Unique identifier for the processing session
        total_frames: Total number of frames processed
        processed_frames: Number of frames that completed processing
        filtered_frames: Number of frames filtered out by filters
        selected_frames: Number of frames that passed all filters
        processing_time: Total processing time in seconds
        stage_metrics: Dictionary of per-stage metrics (tagger/filter stats)
        actor_metrics: Dictionary of actor-level metrics (for parallel processing)
        errors: List of error messages encountered during processing
        lazy_evaluation_enabled: Whether lazy evaluation was used
        memory_release_enabled: Whether memory release was enabled

    Example:
        >>> result = ProcessingResult(
        ...     session_id="abc123",
        ...     total_frames=1000,
        ...     processed_frames=1000,
        ...     filtered_frames=750,
        ...     selected_frames=250,
        ...     processing_time=12.5,
        ...     stage_metrics={},
        ...     actor_metrics={},
        ...     errors=[],
        ...     lazy_evaluation_enabled=True,
        ...     memory_release_enabled=True
        ... )
        >>> print(f"Selected {result.selected_frames}/{result.total_frames} frames")
        Selected 250/1000 frames

    Requirements:
        - 9.4: Logs processing statistics
        - 9.5: Returns summary with performance metrics
        - 10.6: Provides actor health and throughput monitoring
    """

    session_id: str
    total_frames: int
    processed_frames: int
    filtered_frames: int
    selected_frames: int
    processing_time: float
    stage_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    actor_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    lazy_evaluation_enabled: bool = False
    memory_release_enabled: bool = False

    def __str__(self) -> str:
        """Human-readable summary of processing results."""
        return (
            f"ProcessingResult(session={self.session_id}, "
            f"frames={self.selected_frames}/{self.total_frames} selected, "
            f"time={self.processing_time:.2f}s, "
            f"throughput={self.get_throughput():.1f} fps, "
            f"errors={len(self.errors)})"
        )

    def get_selection_rate(self) -> float:
        """
        Get the frame selection rate as a percentage.

        Returns:
            Percentage of frames that passed all filters (0.0-100.0)
        """
        if self.total_frames == 0:
            return 0.0
        return (self.selected_frames / self.total_frames) * 100.0

    def get_throughput(self) -> float:
        """
        Get processing throughput in frames per second.

        Returns:
            Number of frames processed per second
        """
        if self.processing_time == 0:
            return 0.0
        return self.total_frames / self.processing_time

    def get_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive summary dictionary of all metrics.

        Returns:
            Dictionary containing all metrics in a serializable format
        """
        return {
            "session_id": self.session_id,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "filtered_frames": self.filtered_frames,
            "selected_frames": self.selected_frames,
            "processing_time": self.processing_time,
            "throughput_fps": self.get_throughput(),
            "selection_rate_percent": self.get_selection_rate(),
            "error_count": len(self.errors),
            "stage_metrics": self.stage_metrics,
            "actor_metrics": self.actor_metrics,
            "optimizations": {
                "lazy_evaluation": self.lazy_evaluation_enabled,
                "memory_release": self.memory_release_enabled,
            },
        }

    def print_summary(self) -> None:
        """Print a detailed summary of processing results to stdout."""
        print(f"\n{'=' * 60}")
        print(f"Processing Session: {self.session_id}")
        print(f"{'=' * 60}")
        print(
            f"Frames: {self.selected_frames}/{self.total_frames} selected "
            f"({self.get_selection_rate():.1f}%)"
        )
        print(f"Filtered: {self.filtered_frames} frames")
        print(f"Processing Time: {self.processing_time:.2f}s")
        print(f"Throughput: {self.get_throughput():.1f} fps")
        print(f"Errors: {len(self.errors)}")

        if self.lazy_evaluation_enabled or self.memory_release_enabled:
            print("\nOptimizations:")
            if self.lazy_evaluation_enabled:
                print("  - Lazy evaluation: enabled")
            if self.memory_release_enabled:
                print("  - Memory release: enabled")

        if self.stage_metrics:
            print("\nStage Metrics:")
            for stage_name, metrics in self.stage_metrics.items():
                stage_type = metrics.get("type", "unknown")
                frames_processed = metrics.get("frames_processed", 0)
                errors = metrics.get("errors", 0)

                print(f"  {stage_name} ({stage_type}):")
                print(f"    Frames processed: {frames_processed}")

                if stage_type == "filter":
                    frames_passed = metrics.get("frames_passed", 0)
                    frames_filtered = metrics.get("frames_filtered", 0)
                    print(f"    Frames passed: {frames_passed}")
                    print(f"    Frames filtered: {frames_filtered}")

                if errors > 0:
                    print(f"    Errors: {errors}")

                if metrics.get("lazy_evaluated", False):
                    print("    Lazy evaluated: yes")

        if self.actor_metrics:
            print("\nActor Metrics:")
            for actor_id, metrics in self.actor_metrics.items():
                print(f"  {actor_id}:")
                for key, value in metrics.items():
                    print(f"    {key}: {value}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:5], 1):  # Show first 5 errors
                print(f"  {i}. {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")

        print(f"{'=' * 60}\n")


class FrameProcessor:
    """
    Main API for processing video frames through tagging and filtering pipelines.

    FrameProcessor orchestrates the execution of taggers (stateless frame analysis)
    and filters (decision-making) to process video frames. It supports both custom
    pipelines and predefined strategies for common use cases.

    The processor validates pipeline configuration at initialization and provides
    a simple interface for processing video files, frame iterators, or frame lists.

    Attributes:
        pipeline: Ordered list of taggers and filters to apply
        n_workers: Number of worker threads/actors for parallel processing
        use_gpu: Whether to enable GPU acceleration (requires GPU dependencies)
        gpu_batch_size: Batch size for GPU processing

    Example:
        >>> # Using a predefined strategy
        >>> processor = FrameProcessor.with_blur_removal(threshold=100.0)
        >>> for frame in processor.process('video.mp4'):
        ...     process_frame(frame)
        >>>
        >>> # Using a custom pipeline
        >>> from decimatr.taggers.blur import BlurTagger
        >>> from decimatr.filters.blur import BlurFilter
        >>> pipeline = [BlurTagger(), BlurFilter(threshold=100.0)]
        >>> processor = FrameProcessor(pipeline=pipeline)
        >>> for frame in processor.process('video.mp4'):
        ...     process_frame(frame)

    Requirements:
        - 5.1: Accepts pipeline configuration specifying ordered sequence
        - 5.2: Supports mixing stateless and stateful filters
        - 5.3: Validates pipeline configuration at initialization
        - 7.1: Provides single-method interface for processing
    """

    def __init__(
        self,
        pipeline: list[Tagger | Filter] | None = None,
        strategy: FilterStrategy | None = None,
        n_workers: int = 1,
        use_gpu: bool = False,
        gpu_batch_size: int = 32,
        lazy_evaluation: bool = True,
        release_memory: bool = True,
    ):
        """
        Initialize frame processor with pipeline or strategy.

        Args:
            pipeline: Custom pipeline of taggers and filters. If None and no
                     strategy provided, uses pass-through (no filtering).
            strategy: Predefined FilterStrategy. If provided, overrides pipeline.
            n_workers: Number of worker threads/actors for parallel processing.
                      Default is 1 (single-threaded). Values > 1 enable
                      actor-based distributed processing.
            use_gpu: Enable GPU acceleration for supported taggers. Requires
                    GPU dependencies to be installed. Default is False.
            gpu_batch_size: Batch size for GPU processing. Larger values improve
                           throughput but use more GPU memory. Default is 32.
            lazy_evaluation: Enable lazy tag computation (compute only when required
                           by filters). Default is True for better performance.
            release_memory: Release frame_data from memory after filtering out frames.
                          Default is True to reduce memory usage.

        Raises:
            ConfigurationError: If pipeline configuration is invalid
            ValueError: If n_workers or gpu_batch_size are invalid

        Example:
            >>> # Custom pipeline
            >>> pipeline = [BlurTagger(), BlurFilter(threshold=100.0)]
            >>> processor = FrameProcessor(pipeline=pipeline)
            >>>
            >>> # Using strategy
            >>> from decimatr.strategies.blur_removal import BlurRemovalStrategy
            >>> strategy = BlurRemovalStrategy(threshold=100.0)
            >>> processor = FrameProcessor(strategy=strategy)
            >>>
            >>> # Parallel processing with memory optimization
            >>> processor = FrameProcessor(
            ...     strategy=strategy,
            ...     n_workers=4,
            ...     lazy_evaluation=True,
            ...     release_memory=True
            ... )
        """
        # Validate parameters
        if n_workers < 1:
            raise ValueError(f"n_workers must be at least 1, got {n_workers}")

        if gpu_batch_size < 1:
            raise ValueError(f"gpu_batch_size must be at least 1, got {gpu_batch_size}")

        # Store configuration
        self.n_workers = n_workers
        self.use_gpu = use_gpu
        self.gpu_batch_size = gpu_batch_size
        self.lazy_evaluation = lazy_evaluation
        self.release_memory = release_memory

        # Build pipeline from strategy or use provided pipeline
        if strategy is not None:
            self.pipeline = strategy.build_pipeline()
            logger.info(f"Built pipeline from strategy: {strategy}")
        elif pipeline is not None:
            self.pipeline = pipeline
            logger.info(f"Using custom pipeline with {len(pipeline)} components")
        else:
            # Default: pass-through (no filtering)
            self.pipeline = []
            logger.info("Using pass-through pipeline (no filtering)")

        # Validate pipeline configuration
        self._validate_pipeline()

        # Build lazy evaluation plan if enabled
        if self.lazy_evaluation:
            self._build_lazy_evaluation_plan()

        # Initialize actor pipeline reference (created lazily when needed)
        self._actor_pipeline = None
        self._actor_pipeline_initialized = False

        logger.info(
            f"FrameProcessor initialized: {len(self.pipeline)} components, "
            f"{n_workers} workers, GPU={'enabled' if use_gpu else 'disabled'}, "
            f"lazy_evaluation={'enabled' if lazy_evaluation else 'disabled'}, "
            f"release_memory={'enabled' if release_memory else 'disabled'}"
        )

    def _validate_pipeline(self) -> None:
        """
        Validate pipeline configuration.

        Checks that:
        1. All components are valid Tagger or Filter instances
        2. All filters have their required tags provided by upstream taggers
        3. Component order is valid (taggers before filters that need their tags)

        Raises:
            ConfigurationError: If pipeline configuration is invalid
        """
        if not self.pipeline:
            # Empty pipeline is valid (pass-through)
            return

        # Track which tags will be available at each stage
        available_tags = set()

        for i, component in enumerate(self.pipeline):
            # Validate component type
            if not isinstance(component, Tagger | Filter):
                raise ConfigurationError(
                    f"Pipeline component {i} must be a Tagger or Filter, "
                    f"got {type(component).__name__}"
                )

            # If it's a tagger, add its tags to available set
            if isinstance(component, Tagger):
                tagger_tags = component.tag_keys
                available_tags.update(tagger_tags)
                logger.debug(f"Tagger {type(component).__name__} will provide tags: {tagger_tags}")

            # If it's a filter, check that required tags are available
            elif isinstance(component, Filter):
                required_tags = component.required_tags
                missing_tags = set(required_tags) - available_tags

                if missing_tags:
                    raise ConfigurationError(
                        f"Filter {type(component).__name__} at position {i} "
                        f"requires tags {missing_tags} but no upstream tagger "
                        f"provides them. Available tags: {available_tags}"
                    )

                logger.debug(f"Filter {type(component).__name__} requires tags: {required_tags}")

        logger.info(f"Pipeline validation successful: {len(self.pipeline)} components")

    def _build_lazy_evaluation_plan(self) -> None:
        """
        Build lazy evaluation plan for optimized tag computation.

        Analyzes the pipeline to determine which taggers are actually needed
        by downstream filters. This allows us to skip computing tags that
        are never used, improving performance.

        Creates a mapping of which taggers need to be executed before each filter.

        Requirements:
            - 6.6: Supports lazy evaluation, computing tags only when required
        """
        if not self.pipeline:
            self._lazy_plan = {}
            return

        # Build reverse dependency map: which filters need which tags
        filter_dependencies = {}

        for i, component in enumerate(self.pipeline):
            if isinstance(component, Filter):
                required_tags = set(component.required_tags)
                filter_dependencies[i] = required_tags

        # Build forward map: which taggers provide which tags
        tagger_provides = {}
        for i, component in enumerate(self.pipeline):
            if isinstance(component, Tagger):
                for tag_key in component.tag_keys:
                    tagger_provides[tag_key] = i

        # Build lazy execution plan: for each filter, which taggers must run first
        self._lazy_plan = {}
        for filter_idx, required_tags in filter_dependencies.items():
            needed_taggers = set()
            for tag in required_tags:
                if tag in tagger_provides:
                    needed_taggers.add(tagger_provides[tag])
            self._lazy_plan[filter_idx] = sorted(needed_taggers)

        logger.debug(f"Built lazy evaluation plan: {self._lazy_plan}")

    @classmethod
    def with_blur_removal(cls, threshold: float = 100.0, **kwargs) -> "FrameProcessor":
        """
        Create processor with blur removal strategy.

        Convenience method for creating a processor that filters out blurry frames.

        Args:
            threshold: Minimum blur score for frames to pass. Higher values are
                      more restrictive. Default is 100.0.
            **kwargs: Additional arguments passed to FrameProcessor constructor
                     (n_workers, use_gpu, etc.)

        Returns:
            FrameProcessor configured with BlurRemovalStrategy

        Example:
            >>> processor = FrameProcessor.with_blur_removal(threshold=150.0)
            >>> for frame in processor.process('video.mp4'):
            ...     # Process only sharp frames
            ...     process_frame(frame)

        Requirements:
            - 7.4: Provides builder method for common configuration
        """
        from decimatr.strategies.blur_removal import BlurRemovalStrategy

        strategy = BlurRemovalStrategy(threshold=threshold)
        return cls(strategy=strategy, **kwargs)

    @classmethod
    def with_duplicate_detection(
        cls, threshold: float = 0.05, window_size: int = 50, **kwargs
    ) -> "FrameProcessor":
        """
        Create processor with duplicate detection strategy.

        Convenience method for creating a processor that removes duplicate or
        near-duplicate frames.

        Args:
            threshold: Hash similarity threshold (0.0-1.0). Lower values are
                      stricter. Default is 0.05.
            window_size: Number of recent frames to compare against. Default is 50.
            **kwargs: Additional arguments passed to FrameProcessor constructor
                     (n_workers, use_gpu, etc.)

        Returns:
            FrameProcessor configured with DuplicateDetectionStrategy

        Example:
            >>> processor = FrameProcessor.with_duplicate_detection(
            ...     threshold=0.02,
            ...     window_size=100
            ... )
            >>> for frame in processor.process('video.mp4'):
            ...     # Process only unique frames
            ...     process_frame(frame)

        Requirements:
            - 7.4: Provides builder method for common configuration
        """
        from decimatr.strategies.duplicate_detection import DuplicateDetectionStrategy

        strategy = DuplicateDetectionStrategy(threshold=threshold, window_size=window_size)
        return cls(strategy=strategy, **kwargs)

    @classmethod
    def with_smart_sampling(cls, **kwargs) -> "FrameProcessor":
        """
        Create processor with smart sampling strategy.

        Convenience method for creating a processor that combines blur removal,
        duplicate detection, and diversity sampling for comprehensive frame selection.

        Args:
            **kwargs: Additional arguments passed to FrameProcessor constructor
                     (n_workers, use_gpu, etc.) or SmartSamplingStrategy
                     (blur_threshold, duplicate_threshold, etc.)

        Returns:
            FrameProcessor configured with SmartSamplingStrategy

        Example:
            >>> processor = FrameProcessor.with_smart_sampling()
            >>> for frame in processor.process('video.mp4'):
            ...     # Process high-quality, diverse frames
            ...     process_frame(frame)
            >>>
            >>> # With custom parameters
            >>> processor = FrameProcessor.with_smart_sampling(
            ...     blur_threshold=150.0,
            ...     duplicate_threshold=0.02,
            ...     n_workers=4
            ... )

        Requirements:
            - 7.4: Provides builder method for common configuration
        """
        from decimatr.strategies.smart_sampling import SmartSamplingStrategy

        # Separate strategy kwargs from processor kwargs
        strategy_kwargs = {}
        processor_kwargs = {}

        strategy_params = {
            "blur_threshold",
            "duplicate_threshold",
            "duplicate_window",
            "diversity_window",
            "diversity_min_distance",
        }

        for key, value in kwargs.items():
            if key in strategy_params:
                strategy_kwargs[key] = value
            else:
                processor_kwargs[key] = value

        strategy = SmartSamplingStrategy(**strategy_kwargs)
        return cls(strategy=strategy, **processor_kwargs)

    def process(
        self,
        source: str | Iterator[VideoFramePacket] | list[VideoFramePacket],
        session_id: str | None = None,
        return_result: bool = False,
    ) -> Iterator[VideoFramePacket] | tuple[Iterator[VideoFramePacket], ProcessingResult]:
        """
        Process video frames through the pipeline.

        This method accepts video files, frame iterators, or frame lists and
        processes them through the configured pipeline. Frames are processed
        in a streaming fashion - taggers compute tags sequentially, then filters
        are applied in sequence with short-circuit evaluation (if any filter fails,
        subsequent filters are not evaluated).

        When n_workers > 1, uses actor-based distributed processing for parallel
        execution. When n_workers = 1, uses single-threaded processing.

        Args:
            source: Input source, one of:
                   - str: Path to video file (uses load_video_frames)
                   - Iterator[VideoFramePacket]: Frame iterator
                   - List[VideoFramePacket]: List of frames
            session_id: Optional session identifier for logging and metrics.
                       If None, a session ID is generated automatically.
            return_result: If True, returns tuple of (iterator, ProcessingResult).
                          If False (default), returns only the iterator.

        Yields:
            VideoFramePacket objects that passed all filters in the pipeline

        Returns:
            If return_result=False: Iterator of VideoFramePacket
            If return_result=True: Tuple of (Iterator, ProcessingResult)

        Example:
            >>> # Process video file (single-threaded)
            >>> processor = FrameProcessor.with_blur_removal()
            >>> for frame in processor.process('video.mp4'):
            ...     save_frame(frame)
            >>>
            >>> # Process with parallel execution
            >>> processor = FrameProcessor.with_blur_removal(n_workers=4)
            >>> for frame in processor.process('video.mp4'):
            ...     save_frame(frame)
            >>>
            >>> # Process with result summary
            >>> frames, result = processor.process('video.mp4', return_result=True)
            >>> for frame in frames:
            ...     save_frame(frame)
            >>> print(f"Selected {result.selected_frames} frames")

        Requirements:
            - 5.4: Allows filters to access tags from any previously executed tagger
            - 6.1: Processes frames in streaming fashion without loading entire video
            - 6.2: Uses xoscar Actor Model for distributed processing
            - 6.3: Provides configurable parallelism settings
            - 6.4: Falls back to single-threaded when n_workers=1
            - 7.2: Accepts video file paths, frame iterators, or frame lists
            - 7.3: Returns iterator of FramePackets that passed all filters
            - 9.4: Logs processing statistics
            - 9.5: Returns summary with performance metrics
        """
        # Determine processing mode based on n_workers
        if self.n_workers > 1 and self.pipeline:
            # Use actor-based parallel processing
            return self._process_with_actors(source, session_id, return_result)
        else:
            # Use single-threaded processing
            return self._process_single_threaded(source, session_id, return_result)

    def _process_single_threaded(
        self,
        source: str | Iterator[VideoFramePacket] | list[VideoFramePacket],
        session_id: str | None = None,
        return_result: bool = False,
    ) -> Iterator[VideoFramePacket] | tuple[Iterator[VideoFramePacket], ProcessingResult]:
        """
        Process frames using single-threaded execution.

        This is the fallback processing mode when n_workers=1 or when the
        pipeline is empty.

        Args:
            source: Input source (video file, iterator, or list)
            session_id: Optional session identifier
            return_result: Whether to return ProcessingResult

        Returns:
            Iterator of processed frames, optionally with ProcessingResult
        """
        # Generate session ID if not provided
        if session_id is None:
            import uuid

            session_id = str(uuid.uuid4())[:8]

        logger.info(f"Starting single-threaded processing session {session_id}")

        # Initialize metrics tracking
        metrics = {
            "session_id": session_id,
            "total_frames": 0,
            "processed_frames": 0,
            "filtered_frames": 0,
            "selected_frames": 0,
            "errors": [],
            "stage_metrics": {},
            "start_time": time.time(),
        }

        # Convert source to frame iterator
        frame_iterator = self._get_frame_iterator(source)

        # Create processing iterator with metrics tracking
        def process_with_metrics():
            for packet in frame_iterator:
                metrics["total_frames"] += 1
                metrics["processed_frames"] += 1

                # Process frame through pipeline
                result = self._process_frame_with_metrics(packet, metrics)

                if result is not None:
                    metrics["selected_frames"] += 1
                    yield result
                else:
                    metrics["filtered_frames"] += 1

            # Log final statistics
            processing_time = time.time() - metrics["start_time"]
            logger.info(
                f"Session {session_id} complete: "
                f"{metrics['total_frames']} frames processed, "
                f"{metrics['selected_frames']} frames passed "
                f"({metrics['selected_frames'] / metrics['total_frames'] * 100:.1f}%), "
                f"time={processing_time:.2f}s"
                if metrics["total_frames"] > 0
                else f"Session {session_id} complete: 0 frames processed"
            )

        iterator = process_with_metrics()

        if return_result:
            # Create ProcessingResult that will be populated after iteration completes
            result = ProcessingResult(
                session_id=session_id,
                total_frames=0,
                processed_frames=0,
                filtered_frames=0,
                selected_frames=0,
                processing_time=0.0,
                stage_metrics={},
                actor_metrics={},
                errors=[],
                lazy_evaluation_enabled=self.lazy_evaluation,
                memory_release_enabled=self.release_memory,
            )

            # Wrap iterator to update result after completion
            def iterator_with_result():
                yield from iterator

                # Update result after iteration completes
                result.total_frames = metrics["total_frames"]
                result.processed_frames = metrics["processed_frames"]
                result.filtered_frames = metrics["filtered_frames"]
                result.selected_frames = metrics["selected_frames"]
                result.processing_time = time.time() - metrics["start_time"]
                result.stage_metrics = metrics["stage_metrics"]
                result.errors = metrics["errors"]

            return iterator_with_result(), result
        else:
            return iterator

    def _process_with_actors(
        self,
        source: str | Iterator[VideoFramePacket] | list[VideoFramePacket],
        session_id: str | None = None,
        return_result: bool = False,
    ) -> Iterator[VideoFramePacket] | tuple[Iterator[VideoFramePacket], ProcessingResult]:
        """
        Process frames using actor-based parallel execution.

        This method uses ActorPipeline for distributed processing across
        multiple CPU cores when n_workers > 1.

        Args:
            source: Input source (video file, iterator, or list)
            session_id: Optional session identifier
            return_result: Whether to return ProcessingResult

        Returns:
            Iterator of processed frames, optionally with ProcessingResult
        """
        # Generate session ID if not provided
        if session_id is None:
            import uuid

            session_id = str(uuid.uuid4())[:8]

        logger.info(
            f"Starting actor-based processing session {session_id} with {self.n_workers} workers"
        )

        # Initialize metrics tracking
        metrics = {
            "session_id": session_id,
            "total_frames": 0,
            "processed_frames": 0,
            "filtered_frames": 0,
            "selected_frames": 0,
            "errors": [],
            "stage_metrics": {},
            "start_time": time.time(),
        }

        # Convert source to frame iterator
        frame_iterator = self._get_frame_iterator(source)

        # Create processing iterator with actor pipeline
        def process_with_actors():
            # Initialize actor pipeline
            actor_pipeline = self._get_or_create_actor_pipeline()

            try:
                # Initialize actors if not already initialized
                if not actor_pipeline.is_initialized():
                    asyncio.run(actor_pipeline.initialize())

                # Process frames through actor pipeline
                for packet in frame_iterator:
                    metrics["total_frames"] += 1
                    metrics["processed_frames"] += 1

                    try:
                        # Process frame through actor pipeline
                        result = asyncio.run(actor_pipeline.process_frame(packet))

                        if result is not None:
                            metrics["selected_frames"] += 1
                            yield result
                        else:
                            metrics["filtered_frames"] += 1

                    except Exception as e:
                        error_msg = f"Frame {packet.frame_number}: Actor processing error: {e}"
                        logger.error(error_msg)
                        metrics["errors"].append(error_msg)
                        metrics["filtered_frames"] += 1

                # Log final statistics
                processing_time = time.time() - metrics["start_time"]
                logger.info(
                    f"Session {session_id} complete: "
                    f"{metrics['total_frames']} frames processed, "
                    f"{metrics['selected_frames']} frames passed "
                    f"({metrics['selected_frames'] / metrics['total_frames'] * 100:.1f}%), "
                    f"time={processing_time:.2f}s"
                    if metrics["total_frames"] > 0
                    else f"Session {session_id} complete: 0 frames processed"
                )

            finally:
                # Shutdown actor pipeline after processing
                if actor_pipeline.is_initialized():
                    asyncio.run(actor_pipeline.shutdown())

        iterator = process_with_actors()

        if return_result:
            # Create ProcessingResult that will be populated after iteration completes
            result = ProcessingResult(
                session_id=session_id,
                total_frames=0,
                processed_frames=0,
                filtered_frames=0,
                selected_frames=0,
                processing_time=0.0,
                stage_metrics={},
                actor_metrics={},
                errors=[],
                lazy_evaluation_enabled=self.lazy_evaluation,
                memory_release_enabled=self.release_memory,
            )

            # Wrap iterator to update result after completion
            def iterator_with_result():
                yield from iterator

                # Update result after iteration completes
                result.total_frames = metrics["total_frames"]
                result.processed_frames = metrics["processed_frames"]
                result.filtered_frames = metrics["filtered_frames"]
                result.selected_frames = metrics["selected_frames"]
                result.processing_time = time.time() - metrics["start_time"]
                result.stage_metrics = metrics["stage_metrics"]
                result.actor_metrics = metrics.get("actor_metrics", {})
                result.errors = metrics["errors"]

            return iterator_with_result(), result
        else:
            return iterator

    def _get_or_create_actor_pipeline(self):
        """
        Get or create the ActorPipeline instance.

        Lazily creates the ActorPipeline when needed for parallel processing.
        Uses a unique address for each instance to avoid port conflicts.

        Returns:
            ActorPipeline instance configured with current pipeline
        """
        if self._actor_pipeline is None:
            import random

            from decimatr.actors.pipeline import ActorPipeline

            # Use a random port to avoid conflicts between multiple processors
            port = random.randint(20000, 30000)
            address = f"127.0.0.1:{port}"

            self._actor_pipeline = ActorPipeline(
                pipeline=self.pipeline,
                n_workers=self.n_workers,
                use_gpu=self.use_gpu,
                address=address,
            )
            logger.info(f"Created ActorPipeline with {self.n_workers} workers at {address}")

        return self._actor_pipeline

    def _get_frame_iterator(
        self, source: str | Iterator[VideoFramePacket] | list[VideoFramePacket]
    ) -> Iterator[VideoFramePacket]:
        """
        Convert source to frame iterator.

        Args:
            source: Video file path, frame iterator, or frame list

        Returns:
            Iterator yielding VideoFramePacket objects

        Raises:
            TypeError: If source type is not supported
            FileNotFoundError: If video file path doesn't exist
        """
        # If source is a string, treat it as a video file path
        if isinstance(source, str):
            from decimatr.video_loader import load_video_frames

            logger.info(f"Loading frames from video file: {source}")
            return load_video_frames(source)

        # If source is a list, convert to iterator
        elif isinstance(source, list):
            logger.info(f"Processing {len(source)} frames from list")
            return iter(source)

        # If source is already an iterator, use it directly
        elif hasattr(source, "__iter__") and hasattr(source, "__next__"):
            logger.info("Processing frames from iterator")
            return source

        # If source has __iter__ but not __next__, it's an iterable (convert to iterator)
        elif hasattr(source, "__iter__"):
            logger.info("Processing frames from iterable")
            return iter(source)

        else:
            raise TypeError(
                f"source must be a video file path (str), frame iterator, or frame list. "
                f"Got {type(source).__name__}"
            )

    def _process_frame(self, packet: VideoFramePacket) -> VideoFramePacket | None:
        """
        Process a single frame through the pipeline.

        Applies taggers sequentially to compute tags, then applies filters
        sequentially with short-circuit evaluation. If any filter fails,
        the frame is filtered out and subsequent filters are not evaluated.

        With lazy evaluation enabled, only computes tags that are needed by
        downstream filters. With memory release enabled, clears frame_data
        from filtered frames to reduce memory usage.

        Args:
            packet: VideoFramePacket to process

        Returns:
            VideoFramePacket if it passes all filters, None if filtered out

        Requirements:
            - 6.6: Supports lazy evaluation, computing tags only when required
            - 6.7: Releases frame data from memory after filtering
        """
        if self.lazy_evaluation and self._lazy_plan:
            # Lazy evaluation: compute tags on-demand as filters need them
            return self._process_frame_lazy(packet)
        else:
            # Eager evaluation: compute all tags upfront
            return self._process_frame_eager(packet)

    def _process_frame_eager(self, packet: VideoFramePacket) -> VideoFramePacket | None:
        """
        Process frame with eager tag computation (all tags computed upfront).

        Args:
            packet: VideoFramePacket to process

        Returns:
            VideoFramePacket if it passes all filters, None if filtered out
        """
        # Apply taggers sequentially to compute tags
        for component in self.pipeline:
            if isinstance(component, Tagger):
                try:
                    tags = component.compute_tags(packet)
                    packet.tags.update(tags)
                    logger.debug(
                        f"Frame {packet.frame_number}: "
                        f"{type(component).__name__} computed tags {list(tags.keys())}"
                    )
                except Exception as e:
                    logger.error(
                        f"Frame {packet.frame_number}: {type(component).__name__} failed: {e}"
                    )
                    # Continue processing - tagger failure doesn't filter the frame
                    # but downstream filters may fail if they need the missing tags

        # Apply filters sequentially with short-circuit evaluation
        for component in self.pipeline:
            if isinstance(component, Filter):
                try:
                    if not component.should_pass(packet):
                        logger.debug(
                            f"Frame {packet.frame_number}: filtered by {type(component).__name__}"
                        )
                        # Release frame data if memory release is enabled
                        if self.release_memory:
                            packet.frame_data = np.zeros((1, 1, 3), dtype=np.uint8)
                        return None  # Frame filtered out, stop processing

                    logger.debug(f"Frame {packet.frame_number}: passed {type(component).__name__}")
                except Exception as e:
                    logger.error(
                        f"Frame {packet.frame_number}: {type(component).__name__} error: {e}"
                    )
                    # Release frame data if memory release is enabled
                    if self.release_memory:
                        packet.frame_data = np.zeros((1, 1, 3), dtype=np.uint8)
                    # Filter error means frame doesn't pass
                    return None

        # Frame passed all filters
        return packet

    def _process_frame_lazy(self, packet: VideoFramePacket) -> VideoFramePacket | None:
        """
        Process frame with lazy tag computation (tags computed on-demand).

        Only computes tags that are actually needed by downstream filters,
        skipping unnecessary computation for better performance.

        Args:
            packet: VideoFramePacket to process

        Returns:
            VideoFramePacket if it passes all filters, None if filtered out
        """
        # Track which taggers have been executed
        executed_taggers = set()

        # Process filters in order, computing tags on-demand
        for i, component in enumerate(self.pipeline):
            if isinstance(component, Filter):
                # Check if we need to compute any tags for this filter
                if i in self._lazy_plan:
                    needed_taggers = self._lazy_plan[i]

                    # Execute taggers that haven't been executed yet
                    for tagger_idx in needed_taggers:
                        if tagger_idx not in executed_taggers:
                            tagger = self.pipeline[tagger_idx]
                            try:
                                tags = tagger.compute_tags(packet)
                                packet.tags.update(tags)
                                executed_taggers.add(tagger_idx)
                                logger.debug(
                                    f"Frame {packet.frame_number}: "
                                    f"{type(tagger).__name__} computed tags {list(tags.keys())} "
                                    f"(lazy evaluation)"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Frame {packet.frame_number}: "
                                    f"{type(tagger).__name__} failed: {e}"
                                )
                                # Mark as executed even if failed to avoid retry
                                executed_taggers.add(tagger_idx)

                # Now apply the filter
                try:
                    if not component.should_pass(packet):
                        logger.debug(
                            f"Frame {packet.frame_number}: filtered by {type(component).__name__}"
                        )
                        # Release frame data if memory release is enabled
                        if self.release_memory:
                            packet.frame_data = np.zeros((1, 1, 3), dtype=np.uint8)
                        return None  # Frame filtered out, stop processing

                    logger.debug(f"Frame {packet.frame_number}: passed {type(component).__name__}")
                except Exception as e:
                    logger.error(
                        f"Frame {packet.frame_number}: {type(component).__name__} error: {e}"
                    )
                    # Release frame data if memory release is enabled
                    if self.release_memory:
                        packet.frame_data = np.zeros((1, 1, 3), dtype=np.uint8)
                    # Filter error means frame doesn't pass
                    return None

        # Frame passed all filters
        return packet

    def _process_frame_with_metrics(
        self, packet: VideoFramePacket, metrics: dict[str, Any]
    ) -> VideoFramePacket | None:
        """
        Process a single frame through the pipeline with metrics tracking.

        This method wraps _process_frame to add detailed metrics tracking
        for each stage of the pipeline. Supports both lazy and eager evaluation.

        Args:
            packet: VideoFramePacket to process
            metrics: Dictionary to store metrics (modified in place)

        Returns:
            VideoFramePacket if it passes all filters, None if filtered out

        Requirements:
            - 6.6: Supports lazy evaluation, computing tags only when required
            - 6.7: Releases frame data from memory after filtering
            - 9.2: Logs frame-level errors and continues processing
            - 9.3: Supports configurable logging levels
        """
        if self.lazy_evaluation and self._lazy_plan:
            return self._process_frame_lazy_with_metrics(packet, metrics)
        else:
            return self._process_frame_eager_with_metrics(packet, metrics)

    def _process_frame_eager_with_metrics(
        self, packet: VideoFramePacket, metrics: dict[str, Any]
    ) -> VideoFramePacket | None:
        """Process frame with eager evaluation and metrics tracking."""
        # Apply taggers sequentially to compute tags
        for component in self.pipeline:
            if isinstance(component, Tagger):
                component_name = type(component).__name__

                try:
                    tags = component.compute_tags(packet)
                    packet.tags.update(tags)
                    logger.debug(
                        f"Frame {packet.frame_number}: "
                        f"{component_name} computed tags {list(tags.keys())}"
                    )

                    # Track tagger metrics
                    if component_name not in metrics["stage_metrics"]:
                        metrics["stage_metrics"][component_name] = {
                            "type": "tagger",
                            "frames_processed": 0,
                            "errors": 0,
                        }
                    metrics["stage_metrics"][component_name]["frames_processed"] += 1

                except Exception as e:
                    error_msg = f"Frame {packet.frame_number}: {component_name} failed: {e}"
                    logger.error(error_msg)
                    metrics["errors"].append(error_msg)

                    # Track error in metrics
                    if component_name not in metrics["stage_metrics"]:
                        metrics["stage_metrics"][component_name] = {
                            "type": "tagger",
                            "frames_processed": 0,
                            "errors": 0,
                        }
                    metrics["stage_metrics"][component_name]["errors"] += 1

                    # Continue processing - tagger failure doesn't filter the frame

        # Apply filters sequentially with short-circuit evaluation
        for component in self.pipeline:
            if isinstance(component, Filter):
                component_name = type(component).__name__

                try:
                    if not component.should_pass(packet):
                        logger.debug(f"Frame {packet.frame_number}: filtered by {component_name}")

                        # Track filter metrics
                        if component_name not in metrics["stage_metrics"]:
                            metrics["stage_metrics"][component_name] = {
                                "type": "filter",
                                "frames_evaluated": 0,
                                "frames_passed": 0,
                                "frames_filtered": 0,
                                "errors": 0,
                            }
                        metrics["stage_metrics"][component_name]["frames_evaluated"] += 1
                        metrics["stage_metrics"][component_name]["frames_filtered"] += 1

                        # Release frame data if memory release is enabled
                        if self.release_memory:
                            packet.frame_data = np.zeros((1, 1, 3), dtype=np.uint8)

                        return None  # Frame filtered out, stop processing

                    logger.debug(f"Frame {packet.frame_number}: passed {component_name}")

                    # Track filter metrics
                    if component_name not in metrics["stage_metrics"]:
                        metrics["stage_metrics"][component_name] = {
                            "type": "filter",
                            "frames_evaluated": 0,
                            "frames_passed": 0,
                            "frames_filtered": 0,
                            "errors": 0,
                        }
                    metrics["stage_metrics"][component_name]["frames_evaluated"] += 1
                    metrics["stage_metrics"][component_name]["frames_passed"] += 1

                except Exception as e:
                    error_msg = f"Frame {packet.frame_number}: {component_name} error: {e}"
                    logger.error(error_msg)
                    metrics["errors"].append(error_msg)

                    # Track error in metrics
                    if component_name not in metrics["stage_metrics"]:
                        metrics["stage_metrics"][component_name] = {
                            "type": "filter",
                            "frames_evaluated": 0,
                            "frames_passed": 0,
                            "frames_filtered": 0,
                            "errors": 0,
                        }
                    metrics["stage_metrics"][component_name]["errors"] += 1

                    # Release frame data if memory release is enabled
                    if self.release_memory:
                        packet.frame_data = np.zeros((1, 1, 3), dtype=np.uint8)

                    # Filter error means frame doesn't pass
                    return None

        # Frame passed all filters
        return packet

    def _process_frame_lazy_with_metrics(
        self, packet: VideoFramePacket, metrics: dict[str, Any]
    ) -> VideoFramePacket | None:
        """Process frame with lazy evaluation and metrics tracking."""
        # Track which taggers have been executed
        executed_taggers = set()

        # Process filters in order, computing tags on-demand
        for i, component in enumerate(self.pipeline):
            if isinstance(component, Filter):
                # Check if we need to compute any tags for this filter
                if i in self._lazy_plan:
                    needed_taggers = self._lazy_plan[i]

                    # Execute taggers that haven't been executed yet
                    for tagger_idx in needed_taggers:
                        if tagger_idx not in executed_taggers:
                            tagger = self.pipeline[tagger_idx]
                            component_name = type(tagger).__name__

                            try:
                                tags = tagger.compute_tags(packet)
                                packet.tags.update(tags)
                                executed_taggers.add(tagger_idx)
                                logger.debug(
                                    f"Frame {packet.frame_number}: "
                                    f"{component_name} computed tags {list(tags.keys())} "
                                    f"(lazy evaluation)"
                                )

                                # Track tagger metrics
                                if component_name not in metrics["stage_metrics"]:
                                    metrics["stage_metrics"][component_name] = {
                                        "type": "tagger",
                                        "frames_processed": 0,
                                        "errors": 0,
                                        "lazy_evaluated": True,
                                    }
                                metrics["stage_metrics"][component_name]["frames_processed"] += 1

                            except Exception as e:
                                error_msg = (
                                    f"Frame {packet.frame_number}: {component_name} failed: {e}"
                                )
                                logger.error(error_msg)
                                metrics["errors"].append(error_msg)

                                # Track error in metrics
                                if component_name not in metrics["stage_metrics"]:
                                    metrics["stage_metrics"][component_name] = {
                                        "type": "tagger",
                                        "frames_processed": 0,
                                        "errors": 0,
                                        "lazy_evaluated": True,
                                    }
                                metrics["stage_metrics"][component_name]["errors"] += 1

                                # Mark as executed even if failed to avoid retry
                                executed_taggers.add(tagger_idx)

                # Now apply the filter
                component_name = type(component).__name__

                try:
                    if not component.should_pass(packet):
                        logger.debug(f"Frame {packet.frame_number}: filtered by {component_name}")

                        # Track filter metrics
                        if component_name not in metrics["stage_metrics"]:
                            metrics["stage_metrics"][component_name] = {
                                "type": "filter",
                                "frames_evaluated": 0,
                                "frames_passed": 0,
                                "frames_filtered": 0,
                                "errors": 0,
                            }
                        metrics["stage_metrics"][component_name]["frames_evaluated"] += 1
                        metrics["stage_metrics"][component_name]["frames_filtered"] += 1

                        # Release frame data if memory release is enabled
                        if self.release_memory:
                            packet.frame_data = np.zeros((1, 1, 3), dtype=np.uint8)

                        return None  # Frame filtered out, stop processing

                    logger.debug(f"Frame {packet.frame_number}: passed {component_name}")

                    # Track filter metrics
                    if component_name not in metrics["stage_metrics"]:
                        metrics["stage_metrics"][component_name] = {
                            "type": "filter",
                            "frames_evaluated": 0,
                            "frames_passed": 0,
                            "frames_filtered": 0,
                            "errors": 0,
                        }
                    metrics["stage_metrics"][component_name]["frames_evaluated"] += 1
                    metrics["stage_metrics"][component_name]["frames_passed"] += 1

                except Exception as e:
                    error_msg = f"Frame {packet.frame_number}: {component_name} error: {e}"
                    logger.error(error_msg)
                    metrics["errors"].append(error_msg)

                    # Track error in metrics
                    if component_name not in metrics["stage_metrics"]:
                        metrics["stage_metrics"][component_name] = {
                            "type": "filter",
                            "frames_evaluated": 0,
                            "frames_passed": 0,
                            "frames_filtered": 0,
                            "errors": 0,
                        }
                    metrics["stage_metrics"][component_name]["errors"] += 1

                    # Release frame data if memory release is enabled
                    if self.release_memory:
                        packet.frame_data = np.zeros((1, 1, 3), dtype=np.uint8)

                    # Filter error means frame doesn't pass
                    return None

        # Frame passed all filters
        return packet
