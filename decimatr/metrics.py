"""
Performance metrics and monitoring utilities for frame processing.

This module provides utilities for tracking and reporting performance metrics
during video frame processing, including throughput, latency, and stage-level
statistics.
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageMetrics:
    """
    Metrics for a single pipeline stage (tagger or filter).

    Tracks performance and behavior statistics for individual components
    in the processing pipeline.

    Attributes:
        stage_name: Name of the pipeline stage (component class name)
        stage_type: Type of stage ('tagger' or 'filter')
        frames_processed: Number of frames processed by this stage
        frames_passed: Number of frames that passed (filters only)
        frames_filtered: Number of frames filtered out (filters only)
        errors: Number of errors encountered
        total_time: Total processing time in seconds
        lazy_evaluated: Whether this stage used lazy evaluation
    """

    stage_name: str
    stage_type: str  # 'tagger' or 'filter'
    frames_processed: int = 0
    frames_passed: int = 0
    frames_filtered: int = 0
    errors: int = 0
    total_time: float = 0.0
    lazy_evaluated: bool = False

    def get_pass_rate(self) -> float:
        """
        Get the pass rate for filters as a percentage.

        Returns:
            Percentage of frames that passed (0.0-100.0), or 0.0 for taggers
        """
        if self.stage_type != "filter" or self.frames_processed == 0:
            return 0.0
        return (self.frames_passed / self.frames_processed) * 100.0

    def get_error_rate(self) -> float:
        """
        Get the error rate as a percentage.

        Returns:
            Percentage of frames that encountered errors (0.0-100.0)
        """
        if self.frames_processed == 0:
            return 0.0
        return (self.errors / self.frames_processed) * 100.0

    def get_avg_time_per_frame(self) -> float:
        """
        Get average processing time per frame.

        Returns:
            Average time in seconds per frame, or 0.0 if no frames processed
        """
        if self.frames_processed == 0:
            return 0.0
        return self.total_time / self.frames_processed


@dataclass
class ProcessingMetrics:
    """
    Comprehensive metrics for a frame processing session.

    Tracks overall performance and per-stage statistics for monitoring
    and optimization purposes.

    Attributes:
        session_id: Unique identifier for the processing session
        total_frames: Total number of frames in the input
        processed_frames: Number of frames that completed processing
        selected_frames: Number of frames that passed all filters
        filtered_frames: Number of frames filtered out
        error_count: Total number of errors encountered
        start_time: Session start timestamp
        end_time: Session end timestamp (None if still running)
        stage_metrics: Dictionary of per-stage metrics
        actor_metrics: Dictionary of actor-level metrics (for parallel processing)

    Requirements:
        - 9.4: Tracks processing statistics
        - 9.5: Provides performance metrics
        - 10.6: Monitors actor health and throughput
    """

    session_id: str
    total_frames: int = 0
    processed_frames: int = 0
    selected_frames: int = 0
    filtered_frames: int = 0
    error_count: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    stage_metrics: dict[str, StageMetrics] = field(default_factory=dict)
    actor_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def mark_complete(self) -> None:
        """Mark the processing session as complete."""
        if self.end_time is None:
            self.end_time = time.time()

    def get_duration(self) -> float:
        """
        Get the total processing duration in seconds.

        Returns:
            Duration in seconds, or time elapsed so far if not complete
        """
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def get_throughput(self) -> float:
        """
        Get processing throughput in frames per second.

        Returns:
            Frames per second, or 0.0 if no time elapsed
        """
        duration = self.get_duration()
        if duration == 0:
            return 0.0
        return self.processed_frames / duration

    def get_selection_rate(self) -> float:
        """
        Get the frame selection rate as a percentage.

        Returns:
            Percentage of frames that passed all filters (0.0-100.0)
        """
        if self.total_frames == 0:
            return 0.0
        return (self.selected_frames / self.total_frames) * 100.0

    def get_error_rate(self) -> float:
        """
        Get the overall error rate as a percentage.

        Returns:
            Percentage of frames that encountered errors (0.0-100.0)
        """
        if self.processed_frames == 0:
            return 0.0
        return (self.error_count / self.processed_frames) * 100.0

    def add_stage_metric(self, stage_name: str, stage_type: str) -> StageMetrics:
        """
        Add or get a stage metric tracker.

        Args:
            stage_name: Name of the pipeline stage
            stage_type: Type of stage ('tagger' or 'filter')

        Returns:
            StageMetrics instance for the stage
        """
        if stage_name not in self.stage_metrics:
            self.stage_metrics[stage_name] = StageMetrics(
                stage_name=stage_name, stage_type=stage_type
            )
        return self.stage_metrics[stage_name]

    def record_frame_processed(self) -> None:
        """Record that a frame was processed."""
        self.processed_frames += 1

    def record_frame_selected(self) -> None:
        """Record that a frame passed all filters."""
        self.selected_frames += 1

    def record_frame_filtered(self) -> None:
        """Record that a frame was filtered out."""
        self.filtered_frames += 1

    def record_error(self) -> None:
        """Record that an error occurred."""
        self.error_count += 1

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary dictionary of all metrics.

        Returns:
            Dictionary containing all metrics in a serializable format
        """
        return {
            "session_id": self.session_id,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "selected_frames": self.selected_frames,
            "filtered_frames": self.filtered_frames,
            "error_count": self.error_count,
            "duration_seconds": self.get_duration(),
            "throughput_fps": self.get_throughput(),
            "selection_rate_percent": self.get_selection_rate(),
            "error_rate_percent": self.get_error_rate(),
            "stage_metrics": {
                name: {
                    "type": metrics.stage_type,
                    "frames_processed": metrics.frames_processed,
                    "frames_passed": metrics.frames_passed,
                    "frames_filtered": metrics.frames_filtered,
                    "errors": metrics.errors,
                    "total_time": metrics.total_time,
                    "avg_time_per_frame": metrics.get_avg_time_per_frame(),
                    "pass_rate_percent": metrics.get_pass_rate(),
                    "error_rate_percent": metrics.get_error_rate(),
                    "lazy_evaluated": metrics.lazy_evaluated,
                }
                for name, metrics in self.stage_metrics.items()
            },
            "actor_metrics": self.actor_metrics,
        }

    def __str__(self) -> str:
        """Human-readable summary of metrics."""
        duration = self.get_duration()
        throughput = self.get_throughput()
        selection_rate = self.get_selection_rate()

        return (
            f"ProcessingMetrics(session={self.session_id}, "
            f"frames={self.selected_frames}/{self.total_frames} selected "
            f"({selection_rate:.1f}%), "
            f"throughput={throughput:.1f} fps, "
            f"duration={duration:.2f}s, "
            f"errors={self.error_count})"
        )


class MetricsCollector:
    """
    Collector for aggregating metrics during processing.

    This class provides a convenient interface for collecting metrics
    during frame processing, with support for both single-threaded and
    actor-based parallel processing.

    Requirements:
        - 9.4: Tracks processing statistics
        - 10.6: Monitors actor health and throughput
    """

    def __init__(self, session_id: str):
        """
        Initialize metrics collector.

        Args:
            session_id: Unique identifier for the processing session
        """
        self.metrics = ProcessingMetrics(session_id=session_id)
        self._stage_timers: dict[str, float] = {}

    def start_stage(self, stage_name: str) -> None:
        """
        Start timing a pipeline stage.

        Args:
            stage_name: Name of the stage to time
        """
        self._stage_timers[stage_name] = time.time()

    def end_stage(self, stage_name: str) -> None:
        """
        End timing a pipeline stage and record the duration.

        Args:
            stage_name: Name of the stage to stop timing
        """
        if stage_name in self._stage_timers:
            duration = time.time() - self._stage_timers[stage_name]
            if stage_name in self.metrics.stage_metrics:
                self.metrics.stage_metrics[stage_name].total_time += duration
            del self._stage_timers[stage_name]

    def record_tagger_execution(
        self, tagger_name: str, success: bool = True, lazy: bool = False
    ) -> None:
        """
        Record execution of a tagger.

        Args:
            tagger_name: Name of the tagger
            success: Whether execution was successful
            lazy: Whether this was lazy evaluation
        """
        stage = self.metrics.add_stage_metric(tagger_name, "tagger")
        stage.frames_processed += 1
        stage.lazy_evaluated = lazy

        if not success:
            stage.errors += 1
            self.metrics.record_error()

    def record_filter_execution(self, filter_name: str, passed: bool, success: bool = True) -> None:
        """
        Record execution of a filter.

        Args:
            filter_name: Name of the filter
            passed: Whether the frame passed the filter
            success: Whether execution was successful
        """
        stage = self.metrics.add_stage_metric(filter_name, "filter")
        stage.frames_processed += 1

        if success:
            if passed:
                stage.frames_passed += 1
            else:
                stage.frames_filtered += 1
        else:
            stage.errors += 1
            self.metrics.record_error()

    def record_actor_metric(self, actor_id: str, metric_name: str, value: Any) -> None:
        """
        Record an actor-specific metric.

        Args:
            actor_id: Identifier for the actor
            metric_name: Name of the metric
            value: Metric value
        """
        if actor_id not in self.metrics.actor_metrics:
            self.metrics.actor_metrics[actor_id] = {}
        self.metrics.actor_metrics[actor_id][metric_name] = value

    def get_metrics(self) -> ProcessingMetrics:
        """
        Get the current metrics.

        Returns:
            ProcessingMetrics instance
        """
        return self.metrics

    def finalize(self) -> ProcessingMetrics:
        """
        Finalize metrics collection and return final metrics.

        Returns:
            Final ProcessingMetrics instance
        """
        self.metrics.mark_complete()
        return self.metrics
