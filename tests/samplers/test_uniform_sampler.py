import pytest
import numpy as np
import datetime
from typing import List

from decimatr.samplers.uniform_sampler import UniformSampler
from decimatr.scheme import VideoFramePacket

class TestUniformSampler:
    """
    Tests for UniformSampler functionality.
    """
    
    def test_initialization(self):
        """Test UniformSampler initializes with default and custom settings."""
        # Default initialization
        sampler = UniformSampler()
        assert sampler.num_frames == 10
        
        # Custom initialization
        custom_num_frames = 5
        config = {"num_frames": custom_num_frames}
        sampler = UniformSampler(config=config)
        assert sampler.num_frames == custom_num_frames
    
    def test_sample_less_than_total(self, create_frame_sequence):
        """
        Test that when configured to select fewer frames than available,
        the correct number and distribution of frames are returned.
        """
        # Create 10 frames
        frames = create_frame_sequence(num_frames=10)
        
        # Configure sampler to select 5 frames
        config = {"num_frames": 5}
        sampler = UniformSampler(config=config)
        
        # Sample frames
        result = sampler.sample(frames, session_id="test_session")
        
        # Check the result has the correct number of frames
        assert len(result) == 5
        
        # Check that the frames are distributed uniformly
        # For 5 frames from 10, we should get indices approximately at 0, 2, 5, 7, 9
        # but actual indices can vary slightly due to rounding
        frame_numbers = [frame.frame_number for frame in result]
        
        # The frame numbers should be in ascending order
        assert sorted(frame_numbers) == frame_numbers
        
        # Check that the first and last frames are included 
        # (or close to first/last for uniform distribution)
        assert frame_numbers[0] <= 1  # First or close to first
        assert frame_numbers[-1] >= 8  # Last or close to last
        
        # Check that the frames are approximately evenly spaced
        # Calculate average spacing
        spacings = [frame_numbers[i+1] - frame_numbers[i] for i in range(len(frame_numbers)-1)]
        avg_spacing = sum(spacings) / len(spacings)
        
        # Check that all spacings are similar (within 1 frame of average)
        for spacing in spacings:
            assert abs(spacing - avg_spacing) <= 1
    
    def test_sample_more_than_total(self, create_frame_sequence):
        """
        Test that when configured to select more frames than available,
        all original frames are returned.
        """
        # Create 5 frames
        frames = create_frame_sequence(num_frames=5)
        
        # Configure sampler to select 10 frames
        config = {"num_frames": 10}
        sampler = UniformSampler(config=config)
        
        # Sample frames
        result = sampler.sample(frames, session_id="test_session")
        
        # Check that all original frames are returned
        assert len(result) == 5
        assert all(f in frames for f in result)
        assert all(f in result for f in frames)
    
    def test_sample_equal_to_total(self, create_frame_sequence):
        """
        Test that when configured to select the same number of frames as available,
        all original frames are returned.
        """
        # Create 5 frames
        frames = create_frame_sequence(num_frames=5)
        
        # Configure sampler to select 5 frames
        config = {"num_frames": 5}
        sampler = UniformSampler(config=config)
        
        # Sample frames
        result = sampler.sample(frames, session_id="test_session")
        
        # Check that all original frames are returned
        assert len(result) == 5
        assert all(f in frames for f in result)
        assert all(f in result for f in frames)
    
    def test_sample_one_frame(self, create_frame_sequence):
        """
        Test that when configured to select one frame from multiple,
        the middle frame is selected.
        """
        # Create 5 frames
        frames = create_frame_sequence(num_frames=5)
        
        # Configure sampler to select 1 frame
        config = {"num_frames": 1}
        sampler = UniformSampler(config=config)
        
        # Sample frames
        result = sampler.sample(frames, session_id="test_session")
        
        # Check that only one frame is returned
        assert len(result) == 1
        
        # Check that it's the middle frame (index 2 in a 0-4 range)
        assert result[0].frame_number == 2
    
    def test_sample_zero_frames(self, create_frame_sequence):
        """
        Test that when configured to select zero frames,
        an empty list is returned.
        """
        # Create 5 frames
        frames = create_frame_sequence(num_frames=5)
        
        # Configure sampler to select 0 frames
        config = {"num_frames": 0}
        sampler = UniformSampler(config=config)
        
        # Sample frames
        result = sampler.sample(frames, session_id="test_session")
        
        # Check that an empty list is returned
        assert len(result) == 0
        assert isinstance(result, list)
    
    def test_sample_empty_input_list(self):
        """
        Test that when an empty list of frames is provided as input,
        an empty list is returned.
        """
        # Create empty list
        frames = []
        
        # Use default configuration
        sampler = UniformSampler()
        
        # Sample frames
        result = sampler.sample(frames, session_id="test_session")
        
        # Check that an empty list is returned
        assert len(result) == 0
        assert isinstance(result, list)
    
    def test_sample_iterator_input(self, create_frame_sequence):
        """
        Test that the sampler can handle iterator input as well as list input.
        """
        # Create 10 frames
        frames = create_frame_sequence(num_frames=10)
        
        # Convert to iterator
        frames_iter = iter(frames)
        
        # Configure sampler to select 5 frames
        config = {"num_frames": 5}
        sampler = UniformSampler(config=config)
        
        # Sample frames
        result = sampler.sample(frames_iter, session_id="test_session")
        
        # Check the result has the correct number of frames
        assert len(result) == 5
        
        # Check that the frames are in ascending order by frame number
        frame_numbers = [frame.frame_number for frame in result]
        assert sorted(frame_numbers) == frame_numbers
    
    def test_return_type(self, create_frame_sequence):
        """Test that sample returns a List[VideoFramePacket]."""
        # Create frames
        frames = create_frame_sequence(num_frames=5)
        
        # Use default configuration
        sampler = UniformSampler()
        
        # Sample frames
        result = sampler.sample(frames, session_id="test_session")
        
        # Check return type
        assert isinstance(result, list)
        assert all(isinstance(frame, VideoFramePacket) for frame in result)