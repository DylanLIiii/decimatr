import pytest
from typing import List, Iterator
import numpy as np
import datetime

from decimatr.gates.base_gate import BaseGate
from decimatr.scheme import VideoFramePacket

class SimpleTestGate(BaseGate[bool]):
    """
    A simple implementation of BaseGate for testing purposes.
    """
    def __init__(self, return_value=True):
        self.return_value = return_value
        self.processed_frames = []
    
    def process_frame(self, packet: VideoFramePacket) -> bool:
        """Process a single frame, simply returning the configured return value."""
        self.processed_frames.append(packet)
        return self.return_value

class TestBaseGate:
    """
    Tests for BaseGate functionality.
    """
    
    def test_process_list(self, create_frame_sequence):
        """Test that process() correctly handles a list of packets."""
        # Create a list of frames
        frames = create_frame_sequence(num_frames=5)
        
        # Create gate that always returns True
        gate = SimpleTestGate(return_value=True)
        
        # Process the list
        results = gate.process(frames)
        
        # Check that all frames were processed
        assert len(results) == 5
        assert all(result is True for result in results)
        assert gate.processed_frames == frames
    
    def test_process_iterator(self, create_frame_sequence):
        """Test that process_iter() correctly handles an iterator of packets."""
        # Create a list of frames
        frames = create_frame_sequence(num_frames=5)
        
        # Create gate that alternates between True and False
        gate = SimpleTestGate(return_value=False)
        
        # Process the iterator
        results_iter = gate.process_iter(iter(frames))
        
        # Check that it returns an iterator
        assert isinstance(results_iter, Iterator)
        
        # Collect results
        results = list(results_iter)
        
        # Check that all frames were processed
        assert len(results) == 5
        assert all(result is False for result in results)
        assert gate.processed_frames == frames
    
    def test_call_method(self, create_video_frame_packet, create_solid_color_frame):
        """Test that the gate is callable, delegating to process_frame."""
        # Create a frame
        frame = create_video_frame_packet(
            frame_data=create_solid_color_frame((255, 0, 0)),
            frame_number=0
        )
        
        # Create gate
        gate = SimpleTestGate(return_value=True)
        
        # Call the gate directly
        result = gate(frame)
        
        # Check the result
        assert result is True
        assert gate.processed_frames == [frame]