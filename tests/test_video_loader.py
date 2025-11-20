import datetime
from unittest.mock import MagicMock, patch

import decord
import numpy as np
import pytest
from decimatr.video_loader import load_video_frames


class TestVideoLoader:
    @patch("decimatr.video_loader.os.path.exists")
    @patch("decimatr.video_loader.decord.VideoReader")
    def test_load_video_frames_success(self, mock_video_reader, mock_exists):
        # Setup mock
        mock_exists.return_value = True

        # Mock VideoReader instance
        vr_instance = MagicMock()
        vr_instance.__len__.return_value = 2
        vr_instance.get_avg_fps.return_value = 30.0

        # Mock frame data
        frame1 = MagicMock()
        frame1.asnumpy.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = MagicMock()
        frame2.asnumpy.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        vr_instance.__getitem__.side_effect = [frame1, frame2]
        mock_video_reader.return_value = vr_instance

        # Run function
        frames = list(load_video_frames("test.mp4"))

        # Assertions
        assert len(frames) == 2
        assert frames[0].frame_number == 0
        assert frames[1].frame_number == 1
        assert frames[0].source_video_id == "test.mp4"

        # Check timestamps (0 and 1/30)
        assert frames[0].timestamp == datetime.timedelta(seconds=0.0)
        assert abs(frames[1].timestamp.total_seconds() - 1 / 30.0) < 1e-6

    @patch("decimatr.video_loader.os.path.exists")
    def test_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            list(load_video_frames("nonexistent.mp4"))

    @patch("decimatr.video_loader.os.path.exists")
    @patch("decimatr.video_loader.decord.VideoReader")
    def test_decord_runtime_error(self, mock_video_reader, mock_exists):
        mock_exists.return_value = True
        mock_video_reader.side_effect = RuntimeError("Decord error")

        with pytest.raises(RuntimeError) as excinfo:
            list(load_video_frames("corrupt.mp4"))
        assert "Failed to open video" in str(excinfo.value)

    @patch("decimatr.video_loader.os.path.exists")
    @patch("decimatr.video_loader.decord.VideoReader")
    def test_custom_source_id(self, mock_video_reader, mock_exists):
        mock_exists.return_value = True
        vr_instance = MagicMock()
        vr_instance.__len__.return_value = 1
        vr_instance.get_avg_fps.return_value = 30.0
        frame = MagicMock()
        frame.asnumpy.return_value = np.zeros((10, 10, 3))
        vr_instance.__getitem__.return_value = frame
        mock_video_reader.return_value = vr_instance

        frames = list(load_video_frames("test.mp4", source_video_id="custom_id"))
        assert frames[0].source_video_id == "custom_id"
