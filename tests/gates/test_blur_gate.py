# tests/gates/test_blur_gate.py
import pytest
import numpy as np
import datetime
import cv2
from unittest.mock import patch, MagicMock

from decimatr.gates.blur_gate import BlurGate
from decimatr.scheme import VideoFramePacket


class TestBlurGate:
    """
    Tests for BlurGate functionality.
    """
    
    def test_initialization(self):
        """Test BlurGate initializes with default and custom threshold settings."""
        # Default initialization
        gate = BlurGate()
        assert gate.threshold == BlurGate.DEFAULT_THRESHOLD
        
        # Custom initialization
        custom_threshold = 50.0
        gate = BlurGate(threshold=custom_threshold, session_id="test_session")
        assert gate.threshold == custom_threshold
        assert gate.session_id == "test_session"
    
    def test_calculate_blur_score_rgb(self, create_random_noise_frame):
        """Test blur score calculation on RGB images."""
        gate = BlurGate()
        
        # Create a frame with high variance (sharp)
        np.random.seed(42)  # For reproducibility
        rgb_frame = create_random_noise_frame(size=(24, 24))
        
        # Calculate blur score
        blur_score = gate._calculate_blur_score(rgb_frame)
        
        # Score should be greater than 0 and be a float
        assert blur_score > 0
        assert isinstance(blur_score, float)
    
    def test_calculate_blur_score_grayscale(self):
        """Test blur score calculation on grayscale images."""
        gate = BlurGate()
        
        # Create a grayscale image with alternating pixel values
        gray_frame = np.zeros((24, 24), dtype=np.uint8)
        gray_frame[::2, ::2] = 255  # Create a pattern for high variance
        
        # Calculate blur score
        blur_score = gate._calculate_blur_score(gray_frame)
        
        # Score should be greater than 0 and be a float
        assert blur_score > 0
        assert isinstance(blur_score, float)
    
    def test_blurry_frame_filtered(self, create_video_frame_packet, create_solid_color_frame):
        """Test that blurry frames (low variance) are filtered out."""
        # Solid color frames have very low Laplacian variance (blurry)
        gate = BlurGate()
        
        frame_data = create_solid_color_frame(color=(100, 100, 100))
        frame = create_video_frame_packet(
            frame_data=frame_data,
            frame_number=0
        )
        
        # Mock the blur calculation to ensure it's below threshold
        with patch.object(gate, '_calculate_blur_score', return_value=gate.threshold / 2):
            # Blurry frame should be filtered (return False)
            result = gate.process_frame(frame)
            assert result is False
    
    def test_sharp_frame_passes(self, create_video_frame_packet, create_checkerboard_frame):
        """Test that sharp frames (high variance) pass through the gate."""
        gate = BlurGate()
        
        # Checkerboard pattern should have high Laplacian variance (sharp)
        frame_data = create_checkerboard_frame(size=(24, 24), square_size=2)
        frame = create_video_frame_packet(
            frame_data=frame_data,
            frame_number=0
        )
        
        # Mock the blur calculation to ensure it's above threshold
        with patch.object(gate, '_calculate_blur_score', return_value=gate.threshold * 2):
            # Sharp frame should pass (return True)
            result = gate.process_frame(frame)
            assert result is True
    
    def test_threshold_effect(self, create_video_frame_packet, create_gradient_frame):
        """Test that adjusting the threshold changes the gate's behavior for the same frame."""
        # Gradient frames have intermediate Laplacian variance
        frame_data = create_gradient_frame(size=(24, 24))
        frame = create_video_frame_packet(
            frame_data=frame_data,
            frame_number=0
        )
        
        # Calculate the actual blur score
        gate = BlurGate()
        blur_score = gate._calculate_blur_score(frame_data)
        
        # Test with threshold below the calculated blur score
        low_threshold_gate = BlurGate(threshold=blur_score / 2)
        assert low_threshold_gate.process_frame(frame) is True
        
        # Test with threshold above the calculated blur score
        high_threshold_gate = BlurGate(threshold=blur_score * 2)
        assert high_threshold_gate.process_frame(frame) is False
    
    def test_empty_image_handling(self, create_video_frame_packet):
        """Test handling of empty images."""
        gate = BlurGate()
        
        # Create an empty image (0x0)
        empty_frame_data = np.zeros((0, 0, 3), dtype=np.uint8)
        
        frame = create_video_frame_packet(
            frame_data=empty_frame_data,
            frame_number=0
        )
        
        # Implementation should handle this gracefully
        # Empty image should return 0 blur score and be filtered (return False)
        result = gate.process_frame(frame)
        assert result is False
    
    def test_grayscale_and_rgb_handling(self, create_video_frame_packet, create_solid_color_frame):
        """Test that both grayscale and RGB images are handled correctly in process_frame."""
        gate = BlurGate()
        
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
        
        # Mock the blur calculation to ensure consistent behavior for both
        with patch.object(gate, '_calculate_blur_score', return_value=gate.threshold / 2):
            # Both should be handled and filtered (return False) due to low blur score
            rgb_result = gate.process_frame(rgb_frame)
            gray_result = gate.process_frame(gray_frame)
            
            assert rgb_result is False
            assert gray_result is False
    
    def test_return_type(self, create_video_frame_packet, create_random_noise_frame):
        """Test that process_frame returns a boolean."""
        gate = BlurGate()
        
        frame = create_video_frame_packet(
            frame_data=create_random_noise_frame(),
            frame_number=0
        )
        
        result = gate.process_frame(frame)
        assert isinstance(result, bool)
    
    def test_laplacian_calculation(self):
        """Test that the Laplacian calculation works correctly."""
        gate = BlurGate()
        
        # Create a simple test image with a pattern that should have high variance
        test_frame = np.zeros((24, 24), dtype=np.uint8)
        test_frame[::2, ::2] = 255  # Checkered pattern
        
        # Check if Laplacian is calculated correctly
        score = gate._calculate_blur_score(test_frame)
        assert score > 0
        
        # Create a solid color image that should have zero variance
        solid_frame = np.ones((24, 24), dtype=np.uint8) * 128
        score = gate._calculate_blur_score(solid_frame)
        assert score == 0.0
    
    def test_logging(self, create_video_frame_packet, create_random_noise_frame):
        """Test that logging happens correctly in process_frame."""
        gate = BlurGate(session_id="test_logging_session")
        
        frame = create_video_frame_packet(
            frame_data=create_random_noise_frame(),
            frame_number=42,
            source_video_id="test_video_123"
        )
        
        # Mock the logger to capture the log call
        mock_logger = MagicMock()
        gate.logger = mock_logger
        
        # Process the frame
        gate.process_frame(frame)
        
        # Verify that a log was recorded
        mock_logger.info.assert_called_once()
        
        # Get the log message and extra data
        log_msg, kwargs = mock_logger.info.call_args.args[0], mock_logger.info.call_args.kwargs
        
        # Check log content
        assert "Frame 42 processed for blur" in log_msg
        assert 'extra' in kwargs
        assert kwargs['extra']['component_name'] == "BlurGate"
        assert kwargs['extra']['session_id'] == "test_logging_session"
        assert kwargs['extra']['relevant_metadata']['frame_number'] == 42
        assert kwargs['extra']['relevant_metadata']['source_video_id'] == "test_video_123"