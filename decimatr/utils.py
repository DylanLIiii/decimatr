import logging

import cv2
import numpy as np
from PIL import Image

try:
    import imagehash
except ImportError:
    raise ImportError(
        "imagehash package is required. Install it with: pip install imagehash"
    ) from None

from decimatr.scheme import VideoFramePacket


class ImageHasher:
    """
    Wrapper class for image hashing operations using the imagehash library.

    This class provides a simplified interface for computing various types of
    perceptual hashes (phash, ahash, dhash, whash, colorhash) and comparing
    them for similarity.

    Args:
        hash_size: Size of the hash in bits (default: 8, for 8x8 = 64 bits)
        highfreq_factor: High frequency factor for wavelet hash (default: 4)
    """

    def __init__(self, hash_size: int = 8, highfreq_factor: int = 4):
        """
        Initialize the ImageHasher.

        Args:
            hash_size: Size of the hash in bits (controls hash precision)
            highfreq_factor: High frequency factor for wavelet-based hashes
        """
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor

    def compute_hash_from_array(
        self, image_array: np.ndarray, hash_type: str = "phash"
    ) -> imagehash.ImageHash:
        """
        Compute perceptual hash from a numpy image array.

        Args:
            image_array: numpy array representing the image (BGR or RGB format)
            hash_type: Type of hash to compute ('phash', 'ahash', 'dhash',
                      'whash', 'colorhash')

        Returns:
            imagehash.ImageHash object representing the computed hash

        Raises:
            ValueError: If hash_type is not supported
        """
        # Convert BGR to RGB if needed (OpenCV uses BGR by default)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Assume BGR format from OpenCV, convert to RGB
            rgb_array = image_array[:, :, ::-1]
        else:
            rgb_array = image_array

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(rgb_array)

        # Compute hash based on type
        if hash_type == "phash":
            hash_value = imagehash.phash(pil_image, hash_size=self.hash_size)
        elif hash_type == "ahash":
            hash_value = imagehash.average_hash(pil_image, hash_size=self.hash_size)
        elif hash_type == "dhash":
            hash_value = imagehash.dhash(pil_image, hash_size=self.hash_size)
        elif hash_type == "whash":
            hash_value = imagehash.whash(pil_image, hash_size=self.hash_size, mode="haar")
        elif hash_type == "colorhash":
            hash_value = imagehash.colorhash(pil_image, binbits=self.hash_size)
        else:
            raise ValueError(
                f"Unsupported hash_type: {hash_type}. "
                f"Supported types: 'phash', 'ahash', 'dhash', 'whash', 'colorhash'"
            )

        return hash_value

    def hash_difference(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> int:
        """
        Calculate the difference between two image hashes.

        The difference is computed as the number of bits that differ between
        the two hashes (Hamming distance).

        Args:
            hash1: First imagehash.ImageHash object
            hash2: Second imagehash.ImageHash object

        Returns:
            Integer representing the number of differing bits (Hamming distance)
        """
        # imagehash library provides __sub__ operator to calculate difference
        diff = hash1 - hash2
        return int(diff)


# def rgb_image_to_pil_image(
#     pil_image: Image.Image, metadata: Dict[str, Any]
# ) -> FrameData:
#     """
#     Converts a PIL.Image.Image to an np.ndarray and incorporates metadata to create a FrameData object.

#     Args:
#         pil_image: The PIL.Image.Image object to convert.
#         metadata: A dictionary containing metadata, must include 'timestamp'.

#     Returns:
#         A FrameData object with the converted rgb_image and metadata.

#     Raises:
#         ValueError: if 'timestamp' is not in metadata.
#     """
#     if "timestamp" not in metadata:
#         raise ValueError("Metadata must contain a 'timestamp' field.")

#     rgb_array = np.array(pil_image)

#     # Create FrameData, ensuring all required fields are present
#     # Other fields from metadata can be assigned if they exist in FrameData model
#     frame_data_params = {
#         "timestamp": metadata["timestamp"],
#         "rgb_image": rgb_array,
#         "rgb_path": metadata.get("rgb_path"),
#         "depth_image": metadata.get("depth_image"),
#         "point_cloud": metadata.get("point_cloud"),
#     }
#     # Filter out None values if they are not Optional in FrameData or have defaults
#     # For this specific FrameData model, rgb_path, depth_image, point_cloud are Optional.

#     return FrameData(**frame_data_params)


def extract_frames(video_path: str, logger: logging.Logger):
    """Extract frames from a video file using OpenCV.

    Args:
        video_path: Path to the video file
        logger: Logger instance for logging messages

    Yields:
        tuple: (frame_index, frame_data) where frame_data is a numpy array
    """
    logger.info(f"Attempting to extract frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file: {video_path}")
        return

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video or error

        logger.debug(f"Extracted frame {frame_index} from {video_path}")
        yield frame_index, frame
        frame_index += 1

    cap.release()
    logger.info(f"Finished extracting frames from {video_path}. Total frames: {frame_index}")


def write_packets_to_video(
    frame_packets: list[VideoFramePacket],
    output_path: str,
    logger: logging.Logger,
    fps: int = 30,
):
    """
    Writes a list of VideoFramePacket objects to a video file using OpenCV.

    Args:
        frame_packets: A list of VideoFramePacket objects.
                       Each packet's frame_data is expected to be a NumPy array (H, W, 3) in RGB order.
        output_path: Path to save the output video file.
        logger: Logger instance for logging messages.
        fps: Frames per second for the output video.

    Raises:
        TypeError: If frame_data in any packet is not a NumPy array.
        ValueError: If frame_data has an unexpected shape or if no packets are provided.
        RuntimeError: If OpenCV VideoWriter fails to initialize or write frames.
    """
    if not frame_packets:
        logger.warning(
            f"No frame packets provided for output: {output_path}. No video will be created."
        )
        # Depending on desired behavior, could raise ValueError or return.
        # Raising ValueError to be explicit that an output was expected.
        raise ValueError("No frame packets provided to write to video.")

    first_packet = frame_packets[0]
    if not hasattr(first_packet, "frame_data") or not isinstance(
        first_packet.frame_data, np.ndarray
    ):
        logger.error(
            f"Frame data in the first VideoFramePacket is missing or not a NumPy array for output: {output_path}."
        )
        raise TypeError("Frame data in VideoFramePacket must be a NumPy array.")

    first_frame_data = first_packet.frame_data
    if first_frame_data.ndim != 3 or first_frame_data.shape[2] != 3:
        logger.error(
            f"Frame data in the first packet has unexpected shape: {first_frame_data.shape}. "
            f"Expected (H, W, 3) for output: {output_path}."
        )
        raise ValueError("Frame data must be in HWC format with 3 channels (e.g., RGB).")

    # Assuming frame_data is np.uint8. If not, conversion might be needed: e.g., .astype(np.uint8)
    if first_frame_data.dtype != np.uint8:
        logger.warning(
            f"First frame_data dtype is {first_frame_data.dtype}, not np.uint8. "
            f"OpenCV VideoWriter typically expects uint8. Attempting to proceed."
        )

    frame_height, frame_width, _ = first_frame_data.shape

    try:
        # Initialize VideoWriter. OpenCV expects (width, height).
        # Define the codec and create VideoWriter object
        # Common codecs: 'mp4v' for .mp4, 'XVID' for .avi
        # Adjust output_path extension or codec if needed.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, float(fps), (frame_width, frame_height))
        logger.info(
            f"Initialized VideoWriter for {output_path} with {frame_width}x{frame_height} @ {fps}fps using {fourcc=}."
        )
    except Exception as e:
        logger.error(f"Failed to initialize VideoWriter for {output_path}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize VideoWriter for {output_path}") from e

    frames_written_count = 0
    try:
        for i, packet in enumerate(frame_packets):
            if not hasattr(packet, "frame_data") or not isinstance(packet.frame_data, np.ndarray):
                logger.warning(
                    f"Skipping packet index {i} (frame_number {getattr(packet, 'frame_number', 'N/A')}) for {output_path}: "
                    f"frame_data is missing or not a NumPy array."
                )
                continue

            frame_data = packet.frame_data

            if (
                frame_data.shape[0] != frame_height
                or frame_data.shape[1] != frame_width
                or frame_data.ndim != 3
                or frame_data.shape[2] != 3
            ):
                logger.warning(
                    f"Skipping packet index {i} (frame_number {getattr(packet, 'frame_number', 'N/A')}) for {output_path}: "
                    f"unexpected shape {frame_data.shape}. Expected ({frame_height}, {frame_width}, 3)."
                )
                continue

            if frame_data.dtype != np.uint8:
                logger.debug(  # Log as debug, as warning was given once for the first frame
                    f"Packet index {i} (frame_number {getattr(packet, 'frame_number', 'N/A')}) has dtype {frame_data.dtype}. "
                    f"Converting to np.uint8 for OpenCV VideoWriter."
                )
                try:
                    frame_data = frame_data.astype(np.uint8)
                except Exception as cast_e:
                    logger.warning(
                        f"Could not cast frame data to np.uint8 for packet index {i}. Skipping. Error: {cast_e}"
                    )
                    continue

            # Ensure frame_data is C-contiguous for decord
            contiguous_frame_data = np.ascontiguousarray(frame_data)
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(contiguous_frame_data, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)
            frames_written_count += 1
            logger.debug(
                f"Written frame for packet index {i} (frame_number {getattr(packet, 'frame_number', 'N/A')}) to {output_path}"
            )

        logger.info(
            f"Successfully written {frames_written_count} frames out of {len(frame_packets)} packets to {output_path}."
        )
    except Exception as e:
        logger.error(
            f"Error writing frames to video {output_path} after writing {frames_written_count} frames: {e}",
            exc_info=True,
        )
        # reraise as RuntimeError to indicate partial success/failure
        raise RuntimeError(f"Error during frame writing to {output_path}") from e
    finally:
        if "video_writer" in locals() and video_writer.isOpened():
            video_writer.release()
            logger.info(f"VideoWriter for {output_path} released.")
        elif "video_writer" in locals():
            logger.warning(f"VideoWriter for {output_path} was not opened or already closed.")
