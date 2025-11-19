import datetime
import os
from collections.abc import Iterator

import decord

from decimatr.scheme import VideoFramePacket


def load_video_frames(
    video_path: str, source_video_id: str | None = None
) -> Iterator[VideoFramePacket]:
    """
    Loads frames from a video file using decord and yields VideoFramePacket objects.

    Args:
        video_path: Path to the video file.
        source_video_id: Optional unique identifier for the source video.
                         If None, the basename of the video_path will be used.

    Yields:
        VideoFramePacket: An object containing the frame data and metadata.

    Raises:
        RuntimeError: If the video cannot be opened by decord.
        FileNotFoundError: If the video_path does not exist.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        # Use a specific context if needed, e.g., decord.cpu(0) or decord.gpu(0)
        # For now, let decord decide or use default (often CPU)
        vr = decord.VideoReader(video_path)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to open video '{video_path}' with decord: {e}") from e

    if source_video_id is None:
        source_video_id = os.path.basename(video_path)

    num_frames = len(vr)
    # decord's get_batch can be efficient. Let's read frame by frame for simplicity
    # and to easily get individual timestamps if they vary per frame.
    # Some decord versions might not have direct per-frame timestamps easily accessible
    # in all scenarios or might average them. We will use frame index / fps as a fallback
    # if direct timestamps are not straightforward.
    # For now, let's assume we can get timestamps, or calculate them.

    # Example: If timestamps are directly available via vr.get_frame_timestamp(idx)
    # This method exists in some versions/builds of decord.
    # If not, an alternative is to calculate based on FPS.
    fps = vr.get_avg_fps()

    for i in range(num_frames):
        frame_data = vr[i].asnumpy()  # Get frame as NumPy array

        # Timestamp calculation:
        # Try to get precise timestamp if available, otherwise calculate from FPS.
        # Note: vr.get_frame_timestamp(i) might return a tuple (pts, ts_seconds)
        # or just ts_seconds depending on the decord version and video.
        # For simplicity, we'll calculate from FPS for now.
        # A more robust solution might inspect vr capabilities.
        current_time_seconds = i / fps if fps > 0 else 0.0
        timestamp = datetime.timedelta(seconds=current_time_seconds)

        packet = VideoFramePacket(
            frame_data=frame_data,
            frame_number=i,
            timestamp=timestamp,
            source_video_id=source_video_id,
            additional_metadata={"raw_frame_index_from_loader": i},  # Example metadata
        )
        yield packet


# Example usage (can be removed or kept for testing)
if __name__ == "__main__":
    # Create a dummy mp4 file for testing if one doesn't exist
    # This part is for local testing and might not run in the execution environment.
    sample_video_path = "sample_data/tracking_rgb.mp4"  # Adjust if your sample is elsewhere

    if not os.path.exists(sample_video_path):
        print(f"Sample video {sample_video_path} not found. Skipping example usage.")
    else:
        print(f"Attempting to load frames from: {sample_video_path}")
        try:
            for frame_packet in load_video_frames(sample_video_path):
                print(
                    f"Frame {frame_packet.frame_number}: "
                    f"Timestamp: {frame_packet.timestamp}, "
                    f"Shape: {frame_packet.frame_data.shape}, "
                    f"Source: {frame_packet.source_video_id}"
                )
                if frame_packet.frame_number >= 10:  # Print first 5 frames
                    break
        except Exception as e:
            print(f"Error during example usage: {e}")
