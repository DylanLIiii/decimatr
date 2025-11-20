import datetime
from unittest.mock import MagicMock, patch

import decimatr.utils
import numpy as np
import pytest
from decimatr.scheme import VideoFramePacket
from decimatr.utils import ImageHasher, write_packets_to_video

# Check if imagehash is available
try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False


@pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
class TestImageHasher:
    def test_compute_hash_phash(self):
        hasher = ImageHasher()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        # Add some pattern
        img[10:20, 10:20] = 255

        h1 = hasher.compute_hash_from_array(img, "phash")
        assert h1 is not None

        # Same image should have same hash (diff = 0)
        h2 = hasher.compute_hash_from_array(img, "phash")
        assert hasher.hash_difference(h1, h2) == 0

    def test_compute_hash_types(self):
        hasher = ImageHasher()
        img = np.zeros((64, 64, 3), dtype=np.uint8)

        for htype in ["ahash", "dhash", "whash", "colorhash"]:
            h = hasher.compute_hash_from_array(img, htype)
            assert h is not None

    def test_invalid_hash_type(self):
        hasher = ImageHasher()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            hasher.compute_hash_from_array(img, "invalid_type")

    def test_hash_difference(self):
        hasher = ImageHasher()
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.ones((64, 64, 3), dtype=np.uint8) * 255  # Completely different

        h1 = hasher.compute_hash_from_array(img1, "phash")
        h2 = hasher.compute_hash_from_array(img2, "phash")

        diff = hasher.hash_difference(h1, h2)
        assert diff > 0


class TestWritePacketsToVideo:
    @patch("decimatr.utils.cv2.VideoWriter")
    @patch("decimatr.utils.cv2.VideoWriter_fourcc")
    def test_write_success(self, mock_fourcc, mock_writer):
        # Setup mocks
        mock_out = MagicMock()
        mock_out.isOpened.return_value = True
        mock_writer.return_value = mock_out

        logger = MagicMock()

        packets = [
            VideoFramePacket(
                frame_data=np.zeros((100, 100, 3), dtype=np.uint8),
                frame_number=i,
                timestamp=datetime.timedelta(seconds=i),
                source_video_id="test",
            )
            for i in range(3)
        ]

        write_packets_to_video(packets, "out.mp4", logger)

        assert mock_writer.call_count == 1
        assert mock_out.write.call_count == 3
        assert mock_out.release.call_count == 1

    def test_empty_packets(self):
        logger = MagicMock()
        with pytest.raises(ValueError, match="No frame packets"):
            write_packets_to_video([], "out.mp4", logger)

    def test_invalid_frame_data(self):
        logger = MagicMock()
        # Create a mock packet that passes isinstance check for the list but fails inside the function
        # Actually the function iterates.
        # We can create a dummy class or just use MagicMock
        packet = MagicMock()
        packet.frame_data = None  # Invalid

        # The function checks hasattr(packet, "frame_data") and isinstance(..., np.ndarray)
        # So if we pass this mock, it should trigger the error.

        with pytest.raises(TypeError, match="Frame data in VideoFramePacket must be a NumPy array"):
            write_packets_to_video([packet], "out.mp4", logger)

    def test_invalid_shape(self):
        logger = MagicMock()
        packet = VideoFramePacket(
            frame_data=np.zeros((100, 100), dtype=np.uint8),  # Missing channels
            frame_number=0,
            timestamp=datetime.timedelta(0),
            source_video_id="test",
        )
        with pytest.raises(ValueError, match="HWC format"):
            write_packets_to_video([packet], "out.mp4", logger)

    @patch("decimatr.utils.cv2.VideoWriter")
    @patch("decimatr.utils.cv2.VideoWriter_fourcc")
    def test_writer_initialization_failure(self, mock_fourcc, mock_writer):
        mock_writer.side_effect = Exception("Init failed")
        logger = MagicMock()

        packet = VideoFramePacket(
            frame_data=np.zeros((100, 100, 3), dtype=np.uint8),
            frame_number=0,
            timestamp=datetime.timedelta(0),
            source_video_id="test",
        )

        with pytest.raises(RuntimeError, match="Failed to initialize VideoWriter"):
            write_packets_to_video([packet], "out.mp4", logger)


class TestExtractFrames:
    @patch("decimatr.utils.cv2.VideoCapture")
    def test_extract_frames_success(self, mock_cap):
        # Setup mock
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True

        # Mock reading two frames then stopping
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap_instance.read.side_effect = [(True, frame1), (True, frame2), (False, None)]

        mock_cap.return_value = mock_cap_instance
        logger = MagicMock()

        frames = list(decimatr.utils.extract_frames("test.mp4", logger))

        assert len(frames) == 2
        assert frames[0][0] == 0
        assert frames[1][0] == 1
        assert frames[0][1] is frame1
        assert mock_cap_instance.release.call_count == 1

    @patch("decimatr.utils.cv2.VideoCapture")
    def test_extract_frames_open_fail(self, mock_cap):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_cap.return_value = mock_cap_instance
        logger = MagicMock()

        frames = list(decimatr.utils.extract_frames("test.mp4", logger))
        assert len(frames) == 0
        assert logger.error.call_count == 1
