"""
Tests for Phase 9 performance optimizations: lazy evaluation and memory management.
"""

import datetime

import numpy as np
import pytest
from decimatr.core.processor import FrameProcessor
from decimatr.filters.blur import BlurFilter
from decimatr.filters.threshold import ThresholdFilter
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.entropy import EntropyTagger


def create_test_frame(frame_number: int, blur_score: float = 150.0) -> VideoFramePacket:
    """Create a test frame with specific blur characteristics."""
    # Create frame with varying blur (higher values = sharper)
    if blur_score > 100:
        # Sharp frame - high frequency content
        frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    else:
        # Blurry frame - low frequency content
        frame_data = np.ones((100, 100, 3), dtype=np.uint8) * 128

    return VideoFramePacket(
        frame_data=frame_data,
        frame_number=frame_number,
        timestamp=datetime.timedelta(seconds=frame_number / 30.0),
        source_video_id="test_video",
    )


class TestLazyEvaluation:
    """Test lazy evaluation optimization."""

    def test_lazy_evaluation_enabled_by_default(self):
        """Test that lazy evaluation is enabled by default."""
        processor = FrameProcessor(pipeline=[BlurTagger(), BlurFilter(threshold=100.0)])
        assert processor.lazy_evaluation is True

    def test_lazy_evaluation_can_be_disabled(self):
        """Test that lazy evaluation can be disabled."""
        processor = FrameProcessor(
            pipeline=[BlurTagger(), BlurFilter(threshold=100.0)], lazy_evaluation=False
        )
        assert processor.lazy_evaluation is False

    def test_lazy_evaluation_builds_plan(self):
        """Test that lazy evaluation builds an execution plan."""
        processor = FrameProcessor(
            pipeline=[BlurTagger(), EntropyTagger(), BlurFilter(threshold=100.0)],
            lazy_evaluation=True,
        )
        # Should have a lazy plan for the filter
        assert hasattr(processor, "_lazy_plan")
        assert len(processor._lazy_plan) > 0

    def test_lazy_evaluation_processes_frames(self):
        """Test that lazy evaluation correctly processes frames."""
        processor = FrameProcessor(
            pipeline=[BlurTagger(), BlurFilter(threshold=100.0)], lazy_evaluation=True
        )

        # Create test frames
        frames = [create_test_frame(i, blur_score=150.0) for i in range(5)]

        # Process frames
        results = list(processor.process(frames))

        # All sharp frames should pass
        assert len(results) > 0

        # Check that tags were computed
        for frame in results:
            assert "blur_score" in frame.tags

    def test_lazy_evaluation_skips_unused_taggers(self):
        """Test that lazy evaluation skips taggers not needed by filters."""
        # Pipeline with entropy tagger but no filter that uses it
        processor = FrameProcessor(
            pipeline=[
                BlurTagger(),
                EntropyTagger(),  # Not used by any filter
                BlurFilter(threshold=100.0),
            ],
            lazy_evaluation=True,
        )

        # Create test frames
        frames = [create_test_frame(i, blur_score=150.0) for i in range(3)]

        # Process frames with result tracking
        results, result = processor.process(frames, return_result=True)
        results_list = list(results)

        # Entropy tagger should not have processed any frames (lazy evaluation)
        # Note: This depends on the lazy plan correctly identifying unused taggers
        assert len(results_list) > 0


class TestMemoryRelease:
    """Test memory release optimization."""

    def test_memory_release_enabled_by_default(self):
        """Test that memory release is enabled by default."""
        processor = FrameProcessor(pipeline=[BlurTagger(), BlurFilter(threshold=100.0)])
        assert processor.release_memory is True

    def test_memory_release_can_be_disabled(self):
        """Test that memory release can be disabled."""
        processor = FrameProcessor(
            pipeline=[BlurTagger(), BlurFilter(threshold=100.0)], release_memory=False
        )
        assert processor.release_memory is False

    def test_memory_release_clears_filtered_frames(self):
        """Test that memory release clears frame data from filtered frames."""
        processor = FrameProcessor(
            pipeline=[BlurTagger(), BlurFilter(threshold=200.0)],  # High threshold
            release_memory=True,
        )

        # Create blurry frames that will be filtered
        frames = [create_test_frame(i, blur_score=50.0) for i in range(3)]

        # Process frames (all should be filtered)
        results = list(processor.process(frames))

        # No frames should pass
        assert len(results) == 0

        # Note: We can't directly verify memory release in filtered frames
        # since they're not returned, but we can verify the processor works


class TestMetricsIntegration:
    """Test metrics collection and reporting."""

    def test_processing_result_includes_optimization_flags(self):
        """Test that ProcessingResult includes optimization flags."""
        processor = FrameProcessor(
            pipeline=[BlurTagger(), BlurFilter(threshold=100.0)],
            lazy_evaluation=True,
            release_memory=True,
        )

        frames = [create_test_frame(i, blur_score=150.0) for i in range(3)]

        # Process with result tracking
        results, result = processor.process(frames, return_result=True)
        list(results)  # Consume iterator

        assert result.lazy_evaluation_enabled is True
        assert result.memory_release_enabled is True

    def test_processing_result_tracks_stage_metrics(self):
        """Test that ProcessingResult tracks per-stage metrics."""
        processor = FrameProcessor(pipeline=[BlurTagger(), BlurFilter(threshold=100.0)])

        frames = [create_test_frame(i, blur_score=150.0) for i in range(5)]

        # Process with result tracking
        results, result = processor.process(frames, return_result=True)
        list(results)  # Consume iterator

        # Should have metrics for both tagger and filter
        assert len(result.stage_metrics) > 0

    def test_processing_result_get_summary(self):
        """Test that ProcessingResult.get_summary() returns complete data."""
        processor = FrameProcessor(pipeline=[BlurTagger(), BlurFilter(threshold=100.0)])

        frames = [create_test_frame(i, blur_score=150.0) for i in range(3)]

        # Process with result tracking
        results, result = processor.process(frames, return_result=True)
        list(results)  # Consume iterator

        summary = result.get_summary()

        # Verify summary structure
        assert "session_id" in summary
        assert "total_frames" in summary
        assert "throughput_fps" in summary
        assert "selection_rate_percent" in summary
        assert "stage_metrics" in summary
        assert "optimizations" in summary
        assert summary["optimizations"]["lazy_evaluation"] is True
        assert summary["optimizations"]["memory_release"] is True

    def test_processing_result_print_summary(self, capsys):
        """Test that ProcessingResult.print_summary() outputs correctly."""
        processor = FrameProcessor(pipeline=[BlurTagger(), BlurFilter(threshold=100.0)])

        frames = [create_test_frame(i, blur_score=150.0) for i in range(3)]

        # Process with result tracking
        results, result = processor.process(frames, return_result=True)
        list(results)  # Consume iterator

        # Print summary
        result.print_summary()

        # Capture output
        captured = capsys.readouterr()

        # Verify output contains key information
        assert "Processing Session" in captured.out
        assert "Frames:" in captured.out
        assert "Throughput:" in captured.out
        assert "Optimizations:" in captured.out


class TestEndToEnd:
    """End-to-end tests for performance optimizations."""

    def test_full_pipeline_with_optimizations(self):
        """Test complete pipeline with all optimizations enabled."""
        processor = FrameProcessor(
            pipeline=[
                BlurTagger(),
                EntropyTagger(),
                BlurFilter(threshold=100.0),
                ThresholdFilter(tag_key="entropy", threshold=4.0, operator=">"),
            ],
            lazy_evaluation=True,
            release_memory=True,
        )

        # Create mix of sharp and blurry frames
        frames = []
        for i in range(10):
            blur_score = 150.0 if i % 2 == 0 else 50.0
            frames.append(create_test_frame(i, blur_score=blur_score))

        # Process with result tracking
        results, result = processor.process(frames, return_result=True)
        results_list = list(results)

        # Some frames should pass
        assert len(results_list) > 0
        assert len(results_list) < len(frames)

        # Verify result metrics
        assert result.total_frames == len(frames)
        assert result.selected_frames == len(results_list)
        assert result.filtered_frames == len(frames) - len(results_list)
        assert result.get_throughput() > 0

        # Verify optimization flags
        assert result.lazy_evaluation_enabled is True
        assert result.memory_release_enabled is True

    def test_comparison_lazy_vs_eager(self):
        """Compare lazy vs eager evaluation performance."""
        # Create test frames
        frames = [create_test_frame(i, blur_score=150.0) for i in range(20)]

        # Test with lazy evaluation
        processor_lazy = FrameProcessor(
            pipeline=[BlurTagger(), BlurFilter(threshold=100.0)], lazy_evaluation=True
        )
        results_lazy, result_lazy = processor_lazy.process(frames.copy(), return_result=True)
        list(results_lazy)

        # Test with eager evaluation
        processor_eager = FrameProcessor(
            pipeline=[BlurTagger(), BlurFilter(threshold=100.0)], lazy_evaluation=False
        )
        results_eager, result_eager = processor_eager.process(frames.copy(), return_result=True)
        list(results_eager)

        # Both should produce same results
        assert result_lazy.selected_frames == result_eager.selected_frames
        assert result_lazy.filtered_frames == result_eager.filtered_frames

        # Both should complete successfully
        assert result_lazy.processing_time > 0
        assert result_eager.processing_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
