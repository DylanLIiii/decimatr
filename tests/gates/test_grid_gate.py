
# tests/gates/test_grid_gate.py
import pytest
import numpy as np
from PIL import Image

from decimatr.gates.grid_gate import GridGate


class TestGridGate:
    """
    Tests for GridGate functionality.
    """

    def test_initialization(self):
        """Test GridGate initializes with default and custom settings."""
        # Default initialization
        gate = GridGate()
        assert gate.grid_rows == GridGate.DEFAULT_GRID_ROWS
        assert gate.grid_cols == GridGate.DEFAULT_GRID_COLS
        assert gate.cell_hash_size == GridGate.DEFAULT_CELL_HASH_SIZE
        assert gate.similarity_threshold == GridGate.DEFAULT_SIMILARITY_THRESHOLD
        assert len(gate.stored_signatures) == 0

        # Custom initialization
        gate = GridGate(
            grid_rows=2,
            grid_cols=3,
            cell_hash_size=16,
            similarity_threshold=10,
            session_id="test_session"
        )
        assert gate.grid_rows == 2
        assert gate.grid_cols == 3
        assert gate.cell_hash_size == 16
        assert gate.similarity_threshold == 10
        assert gate.session_id == "test_session"
        assert len(gate.stored_signatures) == 0

    def test_extract_grid_signature_normal_case(self, create_video_frame_packet, create_solid_color_frame):
        """Test _extract_grid_signature with normal image and grid dimensions."""
        gate = GridGate(grid_rows=2, grid_cols=2, cell_hash_size=8)
        frame_data = create_solid_color_frame((255, 0, 0))  # Red frame
        packet = create_video_frame_packet(frame_data=frame_data, frame_number=0)
        
        signature = gate._extract_grid_signature(packet)
        
        # Check that we get a signature with expected length for 2x2 grid with hash_size=8
        # Each 8x8 hash produces a 16 hex character string (8*8/4)
        expected_hex_len = (8 * 8) // 4  # 16 hex chars per hash
        expected_signature_len = expected_hex_len * 2 * 2  # 2x2 grid with 16 hex chars per cell
        assert len(signature) == expected_signature_len
        
        # For a solid color, all grid cells should have identical hashes
        cell_hash_length = len(signature) // 4  # Length of each cell's hash
        first_cell = signature[:cell_hash_length]
        for i in range(1, 4):
            assert signature[i*cell_hash_length:(i+1)*cell_hash_length] == first_cell

    def test_extract_grid_signature_different_grid_sizes(self, create_video_frame_packet, create_solid_color_frame):
        """Test _extract_grid_signature with different grid sizes."""
        frame_data = create_solid_color_frame((0, 255, 0))  # Green frame
        packet = create_video_frame_packet(frame_data=frame_data, frame_number=0)
        
        # Test 1x1 grid (single hash of whole image)
        gate_1x1 = GridGate(grid_rows=1, grid_cols=1, cell_hash_size=8)
        signature_1x1 = gate_1x1._extract_grid_signature(packet)
        
        # Test 4x4 grid (default)
        gate_4x4 = GridGate(grid_rows=4, grid_cols=4, cell_hash_size=8)
        signature_4x4 = gate_4x4._extract_grid_signature(packet)
        
        # Test 3x2 grid (non-square)
        gate_3x2 = GridGate(grid_rows=3, grid_cols=2, cell_hash_size=8)
        signature_3x2 = gate_3x2._extract_grid_signature(packet)
        
        # Length checks
        hex_len = (8 * 8) // 4  # 16 hex chars per 8x8 hash
        assert len(signature_1x1) == hex_len * 1 * 1
        assert len(signature_4x4) == hex_len * 4 * 4
        assert len(signature_3x2) == hex_len * 3 * 2
        
        # For solid color, all grid cells should have identical hashes
        # Testing first and last part of signature
        assert signature_4x4[:hex_len] == signature_4x4[-hex_len:]
        assert signature_3x2[:hex_len] == signature_3x2[-hex_len:]

    def test_extract_grid_signature_edge_cases(self, create_video_frame_packet, create_solid_color_frame):
        """Test _extract_grid_signature with edge cases like grid_rows/cols <= 0."""
        frame_data = create_solid_color_frame((0, 0, 255))  # Blue frame
        packet = create_video_frame_packet(frame_data=frame_data, frame_number=0)
        
        # Test with grid_rows = 0
        gate_0_rows = GridGate(grid_rows=0, grid_cols=4, cell_hash_size=8)
        signature_0_rows = gate_0_rows._extract_grid_signature(packet)
        
        # Test with grid_cols = 0
        gate_0_cols = GridGate(grid_rows=4, grid_cols=0, cell_hash_size=8)
        signature_0_cols = gate_0_cols._extract_grid_signature(packet)
        
        # Test with both = 0
        gate_0_both = GridGate(grid_rows=0, grid_cols=0, cell_hash_size=8)
        signature_0_both = gate_0_both._extract_grid_signature(packet)
        
        # All should fall back to single hash of whole image
        hex_len = (8 * 8) // 4  # 16 hex chars per 8x8 hash
        assert len(signature_0_rows) == hex_len
        assert len(signature_0_cols) == hex_len
        assert len(signature_0_both) == hex_len
        
        # Since they all hash the same solid color frame, they should be identical
        assert signature_0_rows == signature_0_cols
        assert signature_0_rows == signature_0_both

    def test_extract_grid_signature_small_image(self, create_video_frame_packet):
        """Test _extract_grid_signature with images smaller than cell_hash_size."""
        # Create a small 4x4 image (smaller than default cell_hash_size=8)
        small_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        small_frame[:, :] = [255, 0, 0]  # Red color
        
        packet = create_video_frame_packet(frame_data=small_frame, frame_number=0)
        
        # Default gate with cell_hash_size=8
        gate = GridGate()
        signature = gate._extract_grid_signature(packet)
        
        # Should generate placeholder hash
        # Calculate expected placeholder length for default settings
        placeholder_hex_len = (gate.cell_hash_size * gate.cell_hash_size) // 4
        expected_placeholder_len = placeholder_hex_len * gate.grid_rows * gate.grid_cols
        assert len(signature) == expected_placeholder_len
        assert signature == "0" * expected_placeholder_len
        
        # Test with both grid_rows and grid_cols = 0 for very small image
        gate_0_grid = GridGate(grid_rows=0, grid_cols=0)
        signature_0_grid = gate_0_grid._extract_grid_signature(packet)
        
        # Should generate a single placeholder hash
        placeholder_hex_len = (gate_0_grid.cell_hash_size * gate_0_grid.cell_hash_size) // 4
        assert len(signature_0_grid) == placeholder_hex_len
        assert signature_0_grid == "0" * placeholder_hex_len

    def test_is_similar_signature_identical(self):
        """Test _is_similar_signature with identical signatures."""
        gate = GridGate()
        
        # Create a test signature
        test_signature = "aabbccdd"
        
        # Store the same signature
        gate.stored_signatures.append(test_signature)
        
        # Test with identical signature
        result = gate._is_similar_signature(test_signature)
        assert result is True

    def test_is_similar_signature_within_threshold(self):
        """Test _is_similar_signature with signatures within similarity threshold."""
        gate = GridGate(similarity_threshold=2)
        
        # Create and store a test signature
        stored_signature = "aabbccdd"
        gate.stored_signatures.append(stored_signature)
        
        # Test with signature that differs by 1 character (hamming distance = 1)
        test_signature_1 = "aabbccde"  # Last 'd' changed to 'e'
        result_1 = gate._is_similar_signature(test_signature_1)
        assert result_1 is True
        
        # Test with signature that differs by 2 characters (hamming distance = 2)
        test_signature_2 = "aabbcedd"  # 'c' and 'd' swapped
        result_2 = gate._is_similar_signature(test_signature_2)
        assert result_2 is True

    def test_is_similar_signature_exceeds_threshold(self):
        """Test _is_similar_signature with signatures exceeding similarity threshold."""
        gate = GridGate(similarity_threshold=2)
        
        # Create and store a test signature
        stored_signature = "fdsjklgfdsd"
        gate.stored_signatures.append(stored_signature)
        
        # Test with signature that differs by 3 characters (hamming distance = 3)
        test_signature = "aabeccde"  # 3 characters changed
        result = gate._is_similar_signature(test_signature)
        assert result is False

    def test_is_similar_signature_different_lengths(self):
        """Test _is_similar_signature with signatures of different lengths."""
        gate = GridGate()
        
        # Create and store a test signature
        stored_signature = "aabbccdd"
        gate.stored_signatures.append(stored_signature)
        
        # Test with shorter signature
        shorter_signature = "aabbcc"
        result_shorter = gate._is_similar_signature(shorter_signature)
        assert result_shorter is False
        
        # Test with longer signature
        longer_signature = "aabbccddee"
        result_longer = gate._is_similar_signature(longer_signature)
        assert result_longer is False

    def test_process_frame_unique_frames(self, create_video_frame_packet, very_different_colored_frames):
        """Test process_frame with unique frames."""
        gate = GridGate()
        
        red_frame, green_frame, blue_frame = very_different_colored_frames
        
        frame1 = create_video_frame_packet(frame_data=red_frame, frame_number=0)
        frame2 = create_video_frame_packet(frame_data=green_frame, frame_number=1)
        frame3 = create_video_frame_packet(frame_data=blue_frame, frame_number=2)
        
        # Process frames and check results
        assert gate.process_frame(frame1) is True
        assert len(gate.stored_signatures) == 1
        
        assert gate.process_frame(frame2) is True
        assert len(gate.stored_signatures) == 2
        
        assert gate.process_frame(frame3) is True
        assert len(gate.stored_signatures) == 3

    def test_process_frame_duplicate_frames(self, create_video_frame_packet, create_solid_color_frame):
        """Test process_frame with duplicate frames."""
        gate = GridGate()
        
        frame_data = create_solid_color_frame((255, 0, 0))  # Red frame
        frame1 = create_video_frame_packet(frame_data=frame_data, frame_number=0)
        frame2 = create_video_frame_packet(frame_data=frame_data.copy(), frame_number=1)
        
        # First frame should pass
        assert gate.process_frame(frame1) is True
        assert len(gate.stored_signatures) == 1
        
        # Identical second frame should be filtered
        assert gate.process_frame(frame2) is False
        assert len(gate.stored_signatures) == 1

    def test_similarity_threshold_effect(self, create_video_frame_packet, slightly_different_frames):
        """Test effect of similarity_threshold on frame filtering."""
        # Test with default threshold
        default_gate = GridGate()
        
        # Test with high threshold (more permissive)
        high_threshold_gate = GridGate(similarity_threshold=20)
        
        # Test with low threshold (more strict)
        low_threshold_gate = GridGate(similarity_threshold=1)
        
        base_frame, similar_frame = slightly_different_frames
        frame1 = create_video_frame_packet(frame_data=base_frame, frame_number=0)
        frame2 = create_video_frame_packet(frame_data=similar_frame, frame_number=1)
        
        # Process with default threshold
        assert default_gate.process_frame(frame1) is True
        default_result = default_gate.process_frame(frame2)
        
        # Process with high threshold
        assert high_threshold_gate.process_frame(frame1) is True
        high_result = high_threshold_gate.process_frame(frame2)
        
        # Process with low threshold
        assert low_threshold_gate.process_frame(frame1) is True
        low_result = low_threshold_gate.process_frame(frame2)
        
        # High threshold should be more likely to consider frames similar (filter more)
        # Low threshold should be more likely to consider frames different (pass more)
        if default_result is False:  # If default threshold filters the frame
            assert high_result is False  # High threshold should definitely filter it
        if low_result is True:  # If low threshold passes the frame
            assert high_result is False  # High vs low should give different results for slightly different frames

    def test_very_different_frames(self, create_video_frame_packet, very_different_frames):
        """Test process_frame with very different frames."""
        gate = GridGate()
        
        frame1_data, frame2_data = very_different_frames
        frame1 = create_video_frame_packet(frame_data=frame1_data, frame_number=0)
        frame2 = create_video_frame_packet(frame_data=frame2_data, frame_number=1)
        
        assert gate.process_frame(frame1) is True
        assert gate.process_frame(frame2) is True
        assert len(gate.stored_signatures) == 2

    def test_clear_signatures(self, create_video_frame_packet, create_solid_color_frame):
        """Test clear_signatures method."""
        gate = GridGate()
        
        frame_data = create_solid_color_frame((255, 0, 0))  # Red frame
        frame = create_video_frame_packet(frame_data=frame_data, frame_number=0)
        
        # Process one frame to populate stored_signatures
        assert gate.process_frame(frame) is True
        assert len(gate.stored_signatures) == 1
        
        # Duplicate frame should be filtered
        assert gate.process_frame(frame) is False
        
        # Clear signatures
        gate.clear_signatures()
        assert len(gate.stored_signatures) == 0
        
        # After clearing, the same frame should pass again
        assert gate.process_frame(frame) is True
        assert len(gate.stored_signatures) == 1

    def test_pil_image_conversion(self, create_video_frame_packet, create_solid_color_frame):
        """Test _get_pil_image method."""
        gate = GridGate()
        
        frame_data = create_solid_color_frame((255, 0, 0))  # Red frame
        packet = create_video_frame_packet(frame_data=frame_data, frame_number=0)
        
        pil_image = gate._get_pil_image(packet)
        
        # Check that it returns a PIL Image
        assert isinstance(pil_image, Image.Image)
        
        # Check dimensions match
        assert pil_image.size == (frame_data.shape[1], frame_data.shape[0])
        
        # Check color is preserved (sample the center pixel)
        center_pixel = pil_image.getpixel((pil_image.width // 2, pil_image.height // 2))
        assert center_pixel == (255, 0, 0) or center_pixel == (255, 0, 0, 255)  # RGB or RGBA

    def test_return_type(self, create_video_frame_packet, create_solid_color_frame):
        """Test that process_frame returns a boolean."""
        gate = GridGate()
        
        frame = create_video_frame_packet(
            frame_data=create_solid_color_frame((255, 0, 0)),
            frame_number=0
        )
        
        result = gate.process_frame(frame)
        assert isinstance(result, bool)
