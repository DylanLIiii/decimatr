# tests/gates/test_clip_gate.py
import pytest
import numpy as np
from unittest.mock import patch, Mock
from PIL import Image

from decimatr.gates.clip_gate import ClipGate
from decimatr.scheme import VideoFramePacket


class TestClipGate:
    """
    Tests for ClipGate functionality.
    """

    def test_initialization(self):
        """Test ClipGate initializes with default and custom settings."""
        # Default initialization
        gate = ClipGate()
        assert gate.similarity_threshold == 0.8
        assert gate.session_id == "default_session"
        assert len(gate.stored_embeddings) == 0

        # Custom initialization
        gate = ClipGate(similarity_threshold=0.5, session_id="test_session")
        assert gate.similarity_threshold == 0.5
        assert gate.session_id == "test_session"
        assert len(gate.stored_embeddings) == 0

    def test_get_pil_image(self, create_video_frame_packet, create_solid_color_frame):
        """Test that _get_pil_image returns a PIL Image from a frame packet."""
        # Create a frame packet with a solid color frame
        frame_data = create_solid_color_frame(color=(255, 0, 0), size=(100, 100))
        packet = create_video_frame_packet(frame_data=frame_data)
        
        gate = ClipGate()
        pil_image = gate._get_pil_image(packet)
        
        # Assert that the result is a PIL Image
        assert isinstance(pil_image, Image.Image)
        
        # Assert that the image dimensions match the input frame
        assert pil_image.width == frame_data.shape[1]
        assert pil_image.height == frame_data.shape[0]

    def test_compute_embedding(self, create_video_frame_packet, create_solid_color_frame):
        """Test that _compute_embedding returns an embedding of expected shape and type."""
        # Create a frame packet with a solid color frame
        frame_data = create_solid_color_frame(color=(255, 0, 0), size=(100, 100))
        packet = create_video_frame_packet(frame_data=frame_data)
        
        gate = ClipGate()
        
        # Mock _get_pil_image to verify it's called
        with patch.object(gate, '_get_pil_image', return_value=Image.fromarray(frame_data)) as mock_get_pil:
            embedding = gate._compute_embedding(packet)
            
            # Verify _get_pil_image was called
            mock_get_pil.assert_called_once_with(packet)
        
        # Assert that the embedding is a numpy array
        assert isinstance(embedding, np.ndarray)
        
        # Assert that the embedding has the expected shape (512 dimensions according to placeholder)
        assert embedding.shape == (512,)
        
        # Assert that the embedding has the expected type
        assert embedding.dtype == np.float32

    def test_is_similar_to_stored_empty(self):
        """Test that _is_similar_to_stored returns False when there are no stored embeddings."""
        gate = ClipGate()
        
        # Create a test embedding
        test_embedding = np.ones((512,), dtype=np.float32)
        
        # Should return False if no stored embeddings
        assert gate._is_similar_to_stored(test_embedding) is False

    def test_is_similar_to_stored_similar(self):
        """Test that _is_similar_to_stored returns True when a similar embedding is found."""
        gate = ClipGate(similarity_threshold=0.9)
        
        # Create a test embedding
        base_embedding = np.ones((512,), dtype=np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)  # Normalize
        
        # Create a slightly different embedding (cosine similarity ~ 0.95)
        similar_embedding = base_embedding.copy()
        similar_embedding[:50] = similar_embedding[:50] * 0.8  # Change some values
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)  # Normalize again
        
        # Add the base embedding to stored embeddings
        gate.stored_embeddings.append(base_embedding)
        
        # The similar embedding should be detected as similar
        assert gate._is_similar_to_stored(similar_embedding) is True

    def test_is_similar_to_stored_not_similar(self):
        """Test that _is_similar_to_stored returns False when no similar embeddings are found."""
        gate = ClipGate(similarity_threshold=0.9)
        
        # Create orthogonal embeddings (cosine similarity = 0)
        embedding1 = np.zeros((512,), dtype=np.float32)
        embedding1[0] = 1.0  # Only first dimension is non-zero
        
        embedding2 = np.zeros((512,), dtype=np.float32)
        embedding2[1] = 1.0  # Only second dimension is non-zero
        
        # Add the first embedding to stored embeddings
        gate.stored_embeddings.append(embedding1)
        
        # The second embedding should not be detected as similar
        assert gate._is_similar_to_stored(embedding2) is False

    def test_is_similar_to_stored_threshold(self):
        """Test that similarity_threshold affects the behavior of _is_similar_to_stored."""
        # Create embeddings with cosine similarity of exactly 0.7
        embedding1 = np.zeros((512,), dtype=np.float32)
        embedding1[0] = 1.0
        embedding1[1] = 0.0
        
        embedding2 = np.zeros((512,), dtype=np.float32)
        embedding2[0] = 0.7  # 0.7 similarity with embedding1
        embedding2[1] = np.sqrt(1 - 0.7**2)  # Make it a unit vector
        
        # Gate with threshold 0.6 (should detect as similar)
        gate_low = ClipGate(similarity_threshold=0.6)
        gate_low.stored_embeddings.append(embedding1)
        assert gate_low._is_similar_to_stored(embedding2) is True
        
        # Gate with threshold 0.8 (should detect as not similar)
        gate_high = ClipGate(similarity_threshold=0.8)
        gate_high.stored_embeddings.append(embedding1)
        assert gate_high._is_similar_to_stored(embedding2) is False

    def test_is_similar_to_stored_zero_vectors(self):
        """Test that _is_similar_to_stored handles zero vectors appropriately."""
        gate = ClipGate()
        
        # Create zero embeddings
        zero_embedding = np.zeros((512,), dtype=np.float32)
        
        # Add one zero embedding to stored embeddings
        gate.stored_embeddings.append(zero_embedding.copy())
        
        # Testing similarity with another zero embedding
        # This should not crash, and the result depends on how the code handles
        # division by zero in the cosine similarity calculation
        try:
            result = gate._is_similar_to_stored(zero_embedding)
            # Just verify that it returned a boolean value
            assert isinstance(result, bool)
        except Exception as e:
            pytest.fail(f"_is_similar_to_stored raised an exception with zero vectors: {e}")

    def test_process_frame_unique(self, create_video_frame_packet, create_solid_color_frame):
        """Test that process_frame returns True and stores embedding for unique frames."""
        frame_data = create_solid_color_frame(color=(255, 0, 0), size=(100, 100))
        packet = create_video_frame_packet(frame_data=frame_data)
        
        gate = ClipGate()
        
        # Mock _compute_embedding to return a specific embedding
        test_embedding = np.ones((512,), dtype=np.float32)
        with patch.object(gate, '_compute_embedding', return_value=test_embedding) as mock_compute:
            # First frame should be unique (no stored embeddings yet)
            result = gate.process_frame(packet)
            
            # Verify _compute_embedding was called
            mock_compute.assert_called_with(packet)
            
            # Process_frame should return True for unique frames
            assert result is True
            
            # The embedding should be stored
            assert len(gate.stored_embeddings) == 1
            assert np.array_equal(gate.stored_embeddings[0], test_embedding)

    def test_process_frame_similar(self, create_video_frame_packet, create_solid_color_frame):
        """Test that process_frame returns False and doesn't store embedding for similar frames."""
        frame_data = create_solid_color_frame(color=(255, 0, 0), size=(100, 100))
        packet1 = create_video_frame_packet(frame_data=frame_data, frame_number=1)
        packet2 = create_video_frame_packet(frame_data=frame_data, frame_number=2)
        
        gate = ClipGate()
        
        # Create two similar embeddings
        embedding1 = np.ones((512,), dtype=np.float32)
        embedding2 = np.ones((512,), dtype=np.float32) * 0.95  # Very similar to embedding1
        
        # Use side_effect to return different embeddings for different calls
        with patch.object(gate, '_compute_embedding', side_effect=[embedding1, embedding2]):
            # First frame should pass (unique)
            assert gate.process_frame(packet1) is True
            assert len(gate.stored_embeddings) == 1
            
            # Second frame should be filtered (similar)
            assert gate.process_frame(packet2) is False
            # Embedding count should still be 1 (no new embedding stored)
            assert len(gate.stored_embeddings) == 1

    def test_process_frame_similarity_threshold(self, create_video_frame_packet, create_solid_color_frame):
        """Test that similarity_threshold affects the behavior of process_frame."""
        frame_data = create_solid_color_frame(color=(255, 0, 0), size=(100, 100))
        packet1 = create_video_frame_packet(frame_data=frame_data, frame_number=1)
        packet2 = create_video_frame_packet(frame_data=frame_data, frame_number=2)
        
        # Create embeddings with cosine similarity of exactly 0.7
        embedding1 = np.zeros((512,), dtype=np.float32)
        embedding1[0] = 1.0
        
        embedding2 = np.zeros((512,), dtype=np.float32)
        embedding2[0] = 0.7
        embedding2[1] = np.sqrt(1 - 0.7**2)  # Make it a unit vector
        
        # Gate with threshold 0.6 (should filter second frame)
        gate_low = ClipGate(similarity_threshold=0.6)
        with patch.object(gate_low, '_compute_embedding', side_effect=[embedding1, embedding2]):
            assert gate_low.process_frame(packet1) is True  # First frame passes
            assert gate_low.process_frame(packet2) is False  # Second frame filtered
        
        # Gate with threshold 0.8 (should pass both frames)
        gate_high = ClipGate(similarity_threshold=0.8)
        with patch.object(gate_high, '_compute_embedding', side_effect=[embedding1, embedding2]):
            assert gate_high.process_frame(packet1) is True  # First frame passes
            assert gate_high.process_frame(packet2) is True  # Second frame passes

    def test_clear_embeddings(self, create_video_frame_packet, create_solid_color_frame):
        """Test that clear_embeddings correctly resets stored_embeddings."""
        frame_data = create_solid_color_frame(color=(255, 0, 0), size=(100, 100))
        packet = create_video_frame_packet(frame_data=frame_data)
        
        gate = ClipGate()
        
        # Add some embeddings
        with patch.object(gate, '_compute_embedding', return_value=np.ones((512,), dtype=np.float32)):
            gate.process_frame(packet)
            assert len(gate.stored_embeddings) == 1
        
        # Clear embeddings
        gate.clear_embeddings()
        
        # Verify embeddings were cleared
        assert len(gate.stored_embeddings) == 0
        
        # Process a frame again to verify functionality still works
        with patch.object(gate, '_compute_embedding', return_value=np.ones((512,), dtype=np.float32)):
            assert gate.process_frame(packet) is True
            assert len(gate.stored_embeddings) == 1

    def test_process_frame_different_colors(
        self, create_video_frame_packet, very_different_colored_frames
    ):
        """Test with frames that have very different colors."""
        # Extract the two different frames from the fixture
        frame1_data, frame2_data, frame3_data = very_different_colored_frames
        
        # Create packet objects
        packet1 = create_video_frame_packet(frame_data=frame1_data)
        packet2 = create_video_frame_packet(frame_data=frame2_data)
        packet3 = create_video_frame_packet(frame_data=frame3_data)
        
        gate = ClipGate()
        
        # Mock embeddings that are orthogonal (completely different)
        embedding1 = np.zeros((512,), dtype=np.float32)
        embedding1[0] = 1.0
        
        embedding2 = np.zeros((512,), dtype=np.float32)
        embedding2[1] = 1.0

        embedding3 = np.zeros((512,), dtype=np.float32)
        embedding3[2] = 1.0 
        
        with patch.object(gate, '_compute_embedding', side_effect=[embedding1, embedding2, embedding3]):
            # Both frames should pass since they're very different
            assert gate.process_frame(packet1) is True
            assert gate.process_frame(packet2) is True
            assert gate.process_frame(packet3) is True
            # Both embeddings should be stored
            assert len(gate.stored_embeddings) == 3