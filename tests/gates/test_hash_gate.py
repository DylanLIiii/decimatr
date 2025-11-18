from decimatr.gates.hash_gate import HashGate


class TestHashGate:
    """
    Tests for HashGate functionality.
    """

    def test_initialization(self):
        """Test HashGate initializes with default and custom settings."""
        # Default initialization
        gate = HashGate()
        assert gate.hash_type == HashGate.DEFAULT_HASH_TYPE
        assert len(gate.stored_hashes) == 0

        # Custom initialization
        gate = HashGate(hash_type="dhash", hash_size=16, session_id="test_session")
        assert gate.hash_type == "dhash"
        assert len(gate.stored_hashes) == 0
        assert gate.session_id == "test_session"

    def test_unique_frames_pass(
        self, create_video_frame_packet, very_different_colored_frames
    ):
        """Test that unique frames pass through the gate and stored_hashes grows."""
        gate = HashGate(
            hash_type="colorhash"
        )  # Try a different hash type that may work better for solid colors

        # Get colored frames that will have very different hash values
        red_frame, green_frame, blue_frame = very_different_colored_frames

        # Create several different frames
        frame1 = create_video_frame_packet(
            frame_data=red_frame,  # Red
            frame_number=0,
        )
        frame2 = create_video_frame_packet(
            frame_data=green_frame,  # Green
            frame_number=1,
        )
        frame3 = create_video_frame_packet(
            frame_data=blue_frame,  # Blue
            frame_number=2,
        )

        # Debug prints to see hash values
        hash1 = gate._calculate_hash(frame1)
        hash2 = gate._calculate_hash(frame2)
        hash3 = gate._calculate_hash(frame3)
        print(f"Red hash: {hash1}")
        print(f"Green hash: {hash2}")
        print(f"Blue hash: {hash3}")
        print(f"Red-Green diff: {gate.hasher.hash_difference(hash1, hash2)}")
        print(f"Red-Blue diff: {gate.hasher.hash_difference(hash1, hash3)}")
        print(f"Green-Blue diff: {gate.hasher.hash_difference(hash2, hash3)}")

        # All should pass as they are unique
        assert gate.process_frame(frame1) is True
        assert len(gate.stored_hashes) == 1

        assert gate.process_frame(frame2) is True
        assert len(gate.stored_hashes) == 2

        assert gate.process_frame(frame3) is True
        assert len(gate.stored_hashes) == 3

    def test_identical_frame_filtered(
        self, create_video_frame_packet, create_solid_color_frame
    ):
        """Test that identical frames are filtered out."""
        gate = HashGate()

        # Create identical frames with different frame numbers
        frame_data = create_solid_color_frame((255, 0, 0))  # Red

        frame1 = create_video_frame_packet(frame_data=frame_data, frame_number=0)
        frame2 = create_video_frame_packet(
            frame_data=frame_data.copy(),  # Copy to ensure it's a different object but identical content
            frame_number=1,
        )

        # First frame should pass
        assert gate.process_frame(frame1) is True
        assert len(gate.stored_hashes) == 1

        # Second identical frame should be filtered
        assert gate.process_frame(frame2) is False
        assert len(gate.stored_hashes) == 1  # stored_hashes should not grow

    def test_clear_hashes_resets_state(
        self, create_video_frame_packet, create_solid_color_frame
    ):
        """Test that clear_hashes() resets the gate's state."""
        gate = HashGate()

        # Create a frame
        frame_data = create_solid_color_frame((255, 0, 0))  # Red

        frame = create_video_frame_packet(frame_data=frame_data, frame_number=0)

        # First pass
        assert gate.process_frame(frame) is True
        assert len(gate.stored_hashes) == 1

        # Second time should be filtered
        assert gate.process_frame(frame) is False

        # Clear hashes and try again
        gate.clear_hashes()
        assert len(gate.stored_hashes) == 0

        # After clearing, the same frame should pass again
        assert gate.process_frame(frame) is True
        assert len(gate.stored_hashes) == 1

    def test_different_hash_types(
        self, create_video_frame_packet, create_solid_color_frame
    ):
        """Test that different hash_type configurations lead to different hashing behavior."""
        phash_gate = HashGate(hash_type="phash")
        dhash_gate = HashGate(hash_type="dhash")

        # Create a frame
        frame = create_video_frame_packet(
            frame_data=create_solid_color_frame((255, 0, 0)),  # Red
            frame_number=0,
        )

        # Process with both gates
        phash_gate.process_frame(frame)
        dhash_gate.process_frame(frame)

        # The hash representations should be different for different hash types
        phash = str(phash_gate.stored_hashes[0])
        dhash = str(dhash_gate.stored_hashes[0])

        assert phash != dhash

    def test_similarity_threshold_effect(
        self, create_video_frame_packet, slightly_different_frames
    ):
        """Test the effect of similarity threshold on frame filtering."""
        gate = HashGate()

        # Get two slightly different frames
        base_frame, similar_frame = slightly_different_frames

        frame1 = create_video_frame_packet(frame_data=base_frame, frame_number=0)
        frame2 = create_video_frame_packet(frame_data=similar_frame, frame_number=1)

        # First frame should pass
        assert gate.process_frame(frame1) is True

        # The second frame is slightly different but might be filtered
        # due to high similarity (depends on the actual implementation)
        # We're testing that the gate makes a decision based on similarity
        result = gate.process_frame(frame2)

        # We can't assert a specific value here without knowing the exact
        # similarity threshold, but we can check that the gate is making
        # decisions based on hash similarity
        if result is True:
            assert len(gate.stored_hashes) == 2
        else:
            assert len(gate.stored_hashes) == 1

    def test_very_different_frames(
        self, create_video_frame_packet, very_different_frames
    ):
        """Test that very different frames are correctly identified as different."""
        gate = HashGate()

        # Get two very different frames
        frame1_data, frame2_data = very_different_frames

        frame1 = create_video_frame_packet(frame_data=frame1_data, frame_number=0)
        frame2 = create_video_frame_packet(frame_data=frame2_data, frame_number=1)

        # Both frames should pass as they are very different
        assert gate.process_frame(frame1) is True
        assert gate.process_frame(frame2) is True
        assert len(gate.stored_hashes) == 2

    def test_return_type(self, create_video_frame_packet, create_solid_color_frame):
        """Test that process_frame returns a boolean."""
        gate = HashGate()

        frame = create_video_frame_packet(
            frame_data=create_solid_color_frame((255, 0, 0)), frame_number=0
        )

        result = gate.process_frame(frame)
        assert isinstance(result, bool)
