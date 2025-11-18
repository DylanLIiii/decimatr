import pytest
import numpy as np
import datetime

from decimatr.gates.entropy_gate import EntropyGate
from decimatr.scheme import VideoFramePacket

class TestEntropyGate:
    """
    Tests for EntropyGate functionality.
    """
    
    def test_initialization(self):
        """Test EntropyGate initializes with default and custom threshold settings."""
        # Default initialization
        gate = EntropyGate()
        assert gate.threshold == EntropyGate.DEFAULT_THRESHOLD
        
        # Custom initialization
        custom_threshold = 5.0
        gate = EntropyGate(threshold=custom_threshold, session_id="test_session")
        assert gate.threshold == custom_threshold
        assert gate.session_id == "test_session"
    
    def test_low_entropy_frame_filtered(self, create_video_frame_packet, low_entropy_frame):
        """Test that frames with low entropy are filtered out."""
        gate = EntropyGate()  # Using default threshold
        
        frame = create_video_frame_packet(
            frame_data=low_entropy_frame,
            frame_number=0
        )
        
        # Low entropy frame should be filtered (return False)
        result = gate.process_frame(frame)
        assert result is False
    
    def test_high_entropy_frame_passes(self, create_video_frame_packet, high_entropy_frame):
        """Test that frames with high entropy pass through the gate."""
        gate = EntropyGate()  # Using default threshold
        
        frame = create_video_frame_packet(
            frame_data=high_entropy_frame,
            frame_number=0
        )
        
        # High entropy frame should pass (return True)
        result = gate.process_frame(frame)
        assert result is True
    
    def test_threshold_effect(self, create_video_frame_packet, create_random_noise_frame):
        """Test that adjusting the threshold changes the gate's behavior for the same frame."""
        # Create a frame with known intermediate entropy
        np.random.seed(42)  # For reproducibility
        medium_entropy_frame = create_random_noise_frame(size=(24, 24))
        
        frame = create_video_frame_packet(
            frame_data=medium_entropy_frame,
            frame_number=0
        )
        
        # Process with default threshold first
        default_gate = EntropyGate()
        default_result = default_gate.process_frame(frame)
        
        # Calculate the entropy score directly to set a very low threshold
        entropy_score = default_gate._calculate_entropy(frame.frame_data)
        
        # Test with threshold below the calculated entropy
        low_threshold_gate = EntropyGate(threshold=entropy_score - 1.0)
        assert low_threshold_gate.process_frame(frame) is True
        
        # Test with threshold above the calculated entropy
        high_threshold_gate = EntropyGate(threshold=entropy_score + 1.0)
        assert high_threshold_gate.process_frame(frame) is False
    
    def test_empty_image_handling(self, create_video_frame_packet):
        """Test handling of empty images."""
        gate = EntropyGate()
        
        # Create an empty image (0x0)
        empty_frame_data = np.zeros((0, 0, 3), dtype=np.uint8)
        
        frame = create_video_frame_packet(
            frame_data=empty_frame_data,
            frame_number=0
        )
        
        # Implementation should handle this gracefully
        entropy = gate._calculate_entropy(frame.frame_data)
        assert entropy == 0.0
        
        # Empty image should be filtered (return False)
        result = gate.process_frame(frame)
        assert result is False
    
    def test_grayscale_and_rgb_handling(self, create_video_frame_packet, create_solid_color_frame):
        """Test that both grayscale and RGB images are handled correctly."""
        gate = EntropyGate()
        
        # Create RGB frame
        rgb_frame_data = create_solid_color_frame((100, 100, 100))
        rgb_frame = create_video_frame_packet(
            frame_data=rgb_frame_data,
            frame_number=0
        )
        
        # Create grayscale frame
        gray_frame_data = np.zeros((24, 24), dtype=np.uint8)
        gray_frame_data.fill(100)
        gray_frame = create_video_frame_packet(
            frame_data=gray_frame_data,
            frame_number=1
        )
        
        # Both should be handled, even if they're filtered
        rgb_result = gate.process_frame(rgb_frame)
        gray_result = gate.process_frame(gray_frame)
        
        # Solid color frames should have similar entropy
        # Both should be filtered due to low entropy
        assert rgb_result is False
        assert gray_result is False
    
    def test_return_type(self, create_video_frame_packet, create_random_noise_frame):
        """Test that process_frame returns a boolean."""
        gate = EntropyGate()
        
        frame = create_video_frame_packet(
            frame_data=create_random_noise_frame(),
            frame_number=0
        )
        
        result = gate.process_frame(frame)
        assert isinstance(result, bool)