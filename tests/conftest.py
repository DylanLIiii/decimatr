import datetime

import numpy as np
import pytest
from decimatr.scheme import VideoFramePacket


@pytest.fixture
def create_video_frame_packet():
    """
    Factory fixture to create VideoFramePacket instances with customizable parameters.
    """

    def _create_packet(
        frame_data=None,
        frame_number=0,
        timestamp=None,
        source_video_id="test_video",
        additional_metadata=None,
    ):
        # Default frame data is a small 24x24 black frame (RGB)
        if frame_data is None:
            frame_data = np.zeros((24, 24, 3), dtype=np.uint8)

        # Default timestamp is 0 seconds
        if timestamp is None:
            timestamp = datetime.timedelta(seconds=frame_number / 30.0)  # Assume 30 fps

        # Default empty metadata dict
        if additional_metadata is None:
            additional_metadata = {}

        return VideoFramePacket(
            frame_data=frame_data,
            frame_number=frame_number,
            timestamp=timestamp,
            source_video_id=source_video_id,
            additional_metadata=additional_metadata,
        )

    return _create_packet


@pytest.fixture
def create_solid_color_frame():
    """
    Factory fixture to create a solid color frame with the specified color and size.
    """

    def _create_solid_frame(color=(0, 0, 0), size=(24, 24)):
        frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        frame[:, :] = color
        return frame

    return _create_solid_frame


@pytest.fixture
def create_gradient_frame():
    """
    Factory fixture to create a horizontal gradient frame.
    """

    def _create_gradient(size=(24, 24)):
        gradient = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        for i in range(size[1]):
            gradient[:, i] = [int(255 * i / size[1])] * 3
        return gradient

    return _create_gradient


@pytest.fixture
def create_random_noise_frame():
    """
    Factory fixture to create a random noise frame.
    """

    def _create_noise(size=(24, 24), seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)

    return _create_noise


@pytest.fixture
def create_checkerboard_frame():
    """
    Factory fixture to create a checkerboard pattern frame.
    """

    def _create_checkerboard(size=(24, 24), square_size=4):
        checkerboard = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        for i in range(0, size[0], square_size * 2):
            for j in range(0, size[1], square_size * 2):
                # White squares
                if i + square_size <= size[0]:
                    if j + square_size <= size[1]:
                        checkerboard[i : i + square_size, j : j + square_size] = 255

                # Alternate white squares
                if i + square_size <= size[0] and j + square_size * 2 <= size[1]:
                    checkerboard[i : i + square_size, j + square_size : j + square_size * 2] = 255

                if i + square_size * 2 <= size[0] and j + square_size <= size[1]:
                    checkerboard[i + square_size : i + square_size * 2, j : j + square_size] = 255

        return checkerboard

    return _create_checkerboard


@pytest.fixture
def create_frame_sequence():
    """
    Factory fixture to create a sequence of VideoFramePacket objects.
    """

    def _create_sequence(num_frames=10, frame_generator=None, source_video_id="test_video"):
        packets = []

        for i in range(num_frames):
            # Use the provided frame generator or default to black frames
            if frame_generator:
                frame_data = frame_generator(i)
            else:
                frame_data = np.zeros((24, 24, 3), dtype=np.uint8)

            # Create packet
            packet = VideoFramePacket(
                frame_data=frame_data,
                frame_number=i,
                timestamp=datetime.timedelta(seconds=i / 30.0),  # Assume 30 fps
                source_video_id=source_video_id,
                additional_metadata={},
            )

            packets.append(packet)

        return packets

    return _create_sequence


@pytest.fixture
def slightly_different_frames():
    """
    Create a pair of slightly different frames (for hash similarity testing).
    """
    base_frame = np.zeros((24, 24, 3), dtype=np.uint8)
    base_frame[5:15, 5:15] = 255  # White square

    similar_frame = base_frame.copy()
    similar_frame[5:15, 6:16] = 255  # White square shifted by 1 pixel

    return base_frame, similar_frame


@pytest.fixture
def very_different_frames():
    """
    Create a pair of very different frames (for hash similarity testing).
    """
    frame1 = np.zeros((24, 24, 3), dtype=np.uint8)
    frame1[2:22, 2:22] = 255  # Large white square

    frame2 = np.zeros((24, 24, 3), dtype=np.uint8)
    frame2[:, :] = 255  # All white
    frame2[5:10, 5:10] = 0  # Small black square
    frame2[15:20, 15:20] = 0  # Small black square

    return frame1, frame2


@pytest.fixture
def low_entropy_frame():
    """
    Create a frame with very low entropy (solid color).
    """
    return np.ones((24, 24, 3), dtype=np.uint8) * 128  # Solid gray


@pytest.fixture
def high_entropy_frame():
    """
    Create a frame with high entropy (random noise).
    """
    np.random.seed(42)  # For reproducibility
    return np.random.randint(0, 256, (24, 24, 3), dtype=np.uint8)


@pytest.fixture
def very_different_colored_frames():
    """
    Create a set of three frames with different dominant colors and unique patterns
    to ensure they have distinct perceptual hash values with average_hash.
    """
    frame_size = (24, 24)
    # Red frame with a black square
    red_frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
    red_frame[:, :, 0] = 255  # All red
    red_frame[2:6, 2:6] = [0, 0, 0]  # Small black square

    # Green frame with a white horizontal line
    green_frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
    green_frame[:, :, 1] = 255  # All green
    green_frame[frame_size[0] // 2 - 1 : frame_size[0] // 2 + 1, 5 : frame_size[1] - 5] = [
        255,
        255,
        255,
    ]  # White horizontal line

    # Blue frame with a yellow cross
    blue_frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
    blue_frame[:, :, 2] = 255  # All blue
    # Yellow color
    yellow = [255, 255, 0]
    center_x, center_y = frame_size[1] // 2, frame_size[0] // 2
    # Draw a small 'X'
    blue_frame[center_y - 2 : center_y + 2, center_x - 2 : center_x + 2] = yellow  # Center dot
    blue_frame[center_y - 3 : center_y - 1, center_x - 1 : center_x + 1] = yellow
    blue_frame[center_y + 1 : center_y + 3, center_x - 1 : center_x + 1] = yellow
    blue_frame[center_y - 1 : center_y + 1, center_x - 3 : center_x - 1] = yellow
    blue_frame[center_y - 1 : center_y + 1, center_x + 1 : center_x + 3] = yellow

    return red_frame, green_frame, blue_frame
