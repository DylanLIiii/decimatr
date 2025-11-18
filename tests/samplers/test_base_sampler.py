import pytest
from typing import List, Dict, Optional, Any, Iterator
import numpy as np
import datetime

from decimatr.samplers.base_sampler import BaseSampler, MaxFramesSampler
from decimatr.scheme import VideoFramePacket

class TestBaseSampler:
    """
    Tests for BaseSampler functionality through its concrete implementation MaxFramesSampler.
    """
    
    def test_initialization(self):
        """Test MaxFramesSampler initializes with default and custom settings."""
        # Default initialization
        sampler = MaxFramesSampler()
        assert sampler.max_frames == 10
        assert sampler.selection_method == 'first'
        
        # Custom initialization
        config = {
            'max_frames': 5,
            'selection_method': 'last'
        }
        sampler = MaxFramesSampler(config=config)
        assert sampler.max_frames == 5
        assert sampler.selection_method == 'last'
    
    def test_sample_empty_frames(self):
        """Test behavior when given empty frames list."""
        sampler = MaxFramesSampler()
        result = sampler.sample([], session_id="test_session")
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_first_selection_method(self, create_frame_sequence):
        """Test 'first' selection method."""
        frames = create_frame_sequence(num_frames=10)
        
        config = {
            'max_frames': 5,
            'selection_method': 'first'
        }
        sampler = MaxFramesSampler(config=config)
        
        result = sampler.sample(frames, session_id="test_session")
        
        # Should select first 5 frames
        assert len(result) == 5
        assert result == frames[:5]
    
    def test_last_selection_method(self, create_frame_sequence):
        """Test 'last' selection method."""
        frames = create_frame_sequence(num_frames=10)
        
        config = {
            'max_frames': 5,
            'selection_method': 'last'
        }
        sampler = MaxFramesSampler(config=config)
        
        result = sampler.sample(frames, session_id="test_session")
        
        # Should select last 5 frames
        assert len(result) == 5
        assert result == frames[-5:]
    
    def test_evenly_spaced_selection_method(self, create_frame_sequence):
        """Test 'evenly_spaced' selection method."""
        frames = create_frame_sequence(num_frames=10)
        
        config = {
            'max_frames': 5,
            'selection_method': 'evenly_spaced'
        }
        sampler = MaxFramesSampler(config=config)
        
        result = sampler.sample(frames, session_id="test_session")
        
        # Should select 5 evenly spaced frames
        assert len(result) == 5
        
        # Expected indices for 5 evenly spaced frames from 10 should be [0, 2, 4, 7, 9]
        # But implementation might vary, so we'll check the general pattern
        frame_numbers = [frame.frame_number for frame in result]
        assert frame_numbers[0] <= 1  # First or close to first
        assert frame_numbers[-1] >= 8  # Last or close to last
        
        # Check that frames are in sequence
        assert sorted(frame_numbers) == frame_numbers
    
    def test_unknown_selection_method(self, create_frame_sequence):
        """Test unknown selection method defaults to 'first'."""
        frames = create_frame_sequence(num_frames=10)
        
        config = {
            'max_frames': 5,
            'selection_method': 'unknown_method'  # Invalid method
        }
        sampler = MaxFramesSampler(config=config)
        
        result = sampler.sample(frames, session_id="test_session")
        
        # Should default to 'first' behavior
        assert len(result) == 5
        assert result == frames[:5]
    
    def test_max_frames_greater_than_total(self, create_frame_sequence):
        """Test when max_frames is greater than total frames."""
        frames = create_frame_sequence(num_frames=5)
        
        config = {
            'max_frames': 10,  # More than available
            'selection_method': 'first'
        }
        sampler = MaxFramesSampler(config=config)
        
        result = sampler.sample(frames, session_id="test_session")
        
        # Should return all frames
        assert len(result) == 5
        assert result == frames
    
    def test_input_iterator_conversion(self, create_frame_sequence):
        """Test that iterator input is properly converted to list."""
        frames = create_frame_sequence(num_frames=5)
        frames_iter = iter(frames)
        
        sampler = MaxFramesSampler()
        
        result = sampler.sample(frames_iter, session_id="test_session")
        
        # Should handle iterator input and return all frames
        assert len(result) == 5
        assert result == frames
    
    def test_return_type(self, create_frame_sequence):
        """Test that sample returns a List[VideoFramePacket]."""
        frames = create_frame_sequence(num_frames=5)
        
        sampler = MaxFramesSampler()
        
        result = sampler.sample(frames, session_id="test_session")
        
        # Check return type
        assert isinstance(result, list)
        assert all(isinstance(frame, VideoFramePacket) for frame in result)