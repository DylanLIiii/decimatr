import pytest
from collections import deque
from decimatr.core.temporal_buffer import TemporalBuffer
from decimatr.scheme import VideoFramePacket
import datetime
import numpy as np


class TestTemporalBuffer:
    @pytest.fixture
    def buffer(self):
        return TemporalBuffer(max_size=3)

    @pytest.fixture
    def sample_packet(self):
        return VideoFramePacket(
            frame_data=np.zeros((10, 10, 3), dtype=np.uint8),
            frame_number=1,
            timestamp=datetime.timedelta(seconds=0.1),
            source_video_id="test_video",
        )

    def test_initialization(self):
        buf = TemporalBuffer(max_size=5)
        assert buf.max_size == 5
        assert len(buf) == 0
        assert not buf  # Test __bool__

        with pytest.raises(ValueError):
            TemporalBuffer(max_size=0)

    def test_add_and_len(self, buffer, sample_packet):
        buffer.add(sample_packet)
        assert len(buffer) == 1
        assert buffer

        # Add same packet multiple times
        buffer.add(sample_packet)
        buffer.add(sample_packet)
        assert len(buffer) == 3
        assert buffer.is_full()

        # Add one more, should evict oldest
        buffer.add(sample_packet)
        assert len(buffer) == 3

    def test_add_invalid_type(self, buffer):
        with pytest.raises(TypeError):
            buffer.add("not a packet")

    def test_get_window(self, buffer, sample_packet):
        buffer.add(sample_packet)
        window = buffer.get_window()
        assert isinstance(window, list)
        assert len(window) == 1
        assert window[0] == sample_packet

        # Ensure it's a copy
        window.clear()
        assert len(buffer) == 1

    def test_clear(self, buffer, sample_packet):
        buffer.add(sample_packet)
        buffer.clear()
        assert len(buffer) == 0
        assert not buffer

    def test_find_similar(self, buffer, sample_packet):
        buffer.add(sample_packet)

        # Define a similarity function
        def similarity_true(p1, p2):
            return True

        def similarity_false(p1, p2):
            return False

        # Create a new packet
        new_packet = VideoFramePacket(
            frame_data=np.zeros((10, 10, 3), dtype=np.uint8),
            frame_number=2,
            timestamp=datetime.timedelta(seconds=0.2),
            source_video_id="test_video",
        )

        found = buffer.find_similar(new_packet, similarity_true)
        assert found == sample_packet

        not_found = buffer.find_similar(new_packet, similarity_false)
        assert not_found is None

    def test_find_similar_invalid_args(self, buffer, sample_packet):
        with pytest.raises(TypeError):
            buffer.find_similar("not a packet", lambda x, y: True)

        with pytest.raises(TypeError):
            buffer.find_similar(sample_packet, "not a function")
