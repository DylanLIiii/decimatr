import time
import pytest
from decimatr.metrics import ProcessingMetrics, StageMetrics, MetricsCollector


class TestStageMetrics:
    def test_initialization(self):
        metrics = StageMetrics(stage_name="test_stage", stage_type="filter")
        assert metrics.stage_name == "test_stage"
        assert metrics.stage_type == "filter"
        assert metrics.frames_processed == 0
        assert metrics.frames_passed == 0
        assert metrics.frames_filtered == 0
        assert metrics.errors == 0
        assert metrics.total_time == 0.0
        assert metrics.lazy_evaluated is False

    def test_pass_rate_filter(self):
        metrics = StageMetrics(stage_name="test", stage_type="filter")
        assert metrics.get_pass_rate() == 0.0

        metrics.frames_processed = 10
        metrics.frames_passed = 6
        assert metrics.get_pass_rate() == 60.0

        metrics.frames_passed = 10
        assert metrics.get_pass_rate() == 100.0

    def test_pass_rate_tagger(self):
        metrics = StageMetrics(stage_name="test", stage_type="tagger")
        metrics.frames_processed = 10
        metrics.frames_passed = 10
        assert metrics.get_pass_rate() == 0.0

    def test_error_rate(self):
        metrics = StageMetrics(stage_name="test", stage_type="filter")
        assert metrics.get_error_rate() == 0.0

        metrics.frames_processed = 10
        metrics.errors = 1
        assert metrics.get_error_rate() == 10.0

    def test_avg_time(self):
        metrics = StageMetrics(stage_name="test", stage_type="filter")
        assert metrics.get_avg_time_per_frame() == 0.0

        metrics.frames_processed = 10
        metrics.total_time = 1.0
        assert metrics.get_avg_time_per_frame() == 0.1


class TestProcessingMetrics:
    def test_initialization(self):
        metrics = ProcessingMetrics(session_id="test_session")
        assert metrics.session_id == "test_session"
        assert metrics.start_time > 0
        assert metrics.end_time is None

    def test_mark_complete(self):
        metrics = ProcessingMetrics(session_id="test")
        assert metrics.end_time is None
        metrics.mark_complete()
        assert metrics.end_time is not None
        assert metrics.end_time >= metrics.start_time

    def test_duration(self):
        metrics = ProcessingMetrics(session_id="test")
        time.sleep(0.01)
        duration = metrics.get_duration()
        assert duration > 0

        metrics.mark_complete()
        final_duration = metrics.get_duration()
        assert final_duration >= duration

    def test_throughput(self):
        metrics = ProcessingMetrics(session_id="test")
        assert metrics.get_throughput() == 0.0

        metrics.processed_frames = 100
        time.sleep(0.01)  # Ensure some time passes
        assert metrics.get_throughput() > 0

    def test_rates(self):
        metrics = ProcessingMetrics(session_id="test")

        # Selection rate
        assert metrics.get_selection_rate() == 0.0
        metrics.total_frames = 100
        metrics.selected_frames = 50
        assert metrics.get_selection_rate() == 50.0

        # Error rate
        assert metrics.get_error_rate() == 0.0
        metrics.processed_frames = 100
        metrics.error_count = 5
        assert metrics.get_error_rate() == 5.0

    def test_add_stage_metric(self):
        metrics = ProcessingMetrics(session_id="test")
        stage = metrics.add_stage_metric("filter1", "filter")
        assert isinstance(stage, StageMetrics)
        assert stage.stage_name == "filter1"
        assert "filter1" in metrics.stage_metrics

        # Retrieve existing
        stage2 = metrics.add_stage_metric("filter1", "filter")
        assert stage2 is stage

    def test_record_methods(self):
        metrics = ProcessingMetrics(session_id="test")

        metrics.record_frame_processed()
        assert metrics.processed_frames == 1

        metrics.record_frame_selected()
        assert metrics.selected_frames == 1

        metrics.record_frame_filtered()
        assert metrics.filtered_frames == 1

        metrics.record_error()
        assert metrics.error_count == 1

    def test_get_summary(self):
        metrics = ProcessingMetrics(session_id="test_session")
        metrics.total_frames = 10
        metrics.record_frame_processed()
        metrics.add_stage_metric("tagger1", "tagger")

        summary = metrics.get_summary()
        assert summary["session_id"] == "test_session"
        assert summary["total_frames"] == 10
        assert summary["processed_frames"] == 1
        assert "tagger1" in summary["stage_metrics"]

    def test_str_representation(self):
        metrics = ProcessingMetrics(session_id="test")
        s = str(metrics)
        assert "ProcessingMetrics" in s
        assert "session=test" in s


class TestMetricsCollector:
    def test_flow(self):
        collector = MetricsCollector(session_id="test_collector")

        # Start stage
        collector.start_stage("tagger1")
        time.sleep(0.01)
        collector.end_stage("tagger1")

        # Verify stage created implicitly if recording execution?
        # Wait, end_stage only updates time if stage exists in metrics.
        # Let's create stage first by recording execution or manual add.
        # Looking at code: start_stage/end_stage update existing metrics if present.
        # But typically one might add the metric first.
        # Actually record_tagger_execution adds the stage.

        collector.record_tagger_execution("tagger1", success=True)

        # Now try timing
        collector.start_stage("tagger1")
        time.sleep(0.01)
        collector.end_stage("tagger1")

        metrics = collector.get_metrics()
        stage = metrics.stage_metrics["tagger1"]
        assert stage.frames_processed == 1
        assert stage.total_time > 0

    def test_record_filter(self):
        collector = MetricsCollector("test")

        collector.record_filter_execution("filter1", passed=True)
        collector.record_filter_execution("filter1", passed=False)
        collector.record_filter_execution("filter1", passed=True, success=False)

        metrics = collector.get_metrics()
        stage = metrics.stage_metrics["filter1"]
        assert stage.frames_processed == 3
        assert stage.frames_passed == 1
        assert stage.frames_filtered == 1
        assert stage.errors == 1

    def test_actor_metrics(self):
        collector = MetricsCollector("test")
        collector.record_actor_metric("actor1", "cpu", 50.0)

        metrics = collector.get_metrics()
        assert metrics.actor_metrics["actor1"]["cpu"] == 50.0

    def test_finalize(self):
        collector = MetricsCollector("test")
        metrics = collector.finalize()
        assert metrics.end_time is not None
