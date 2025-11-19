"""
Duplicate detection filter using perceptual hash comparison.

This filter maintains a temporal buffer of recent frame hashes and filters out
frames that are too similar to frames already in the buffer, effectively
removing duplicate or near-duplicate frames from the video stream.
"""

from decimatr.filters.base import StatefulFilter
from decimatr.scheme import VideoFramePacket
from decimatr.utils import ImageHasher


class DuplicateFilter(StatefulFilter):
    """
    Filter that detects and removes duplicate frames using perceptual hashing.

    This filter maintains a temporal buffer of recent frames and compares each
    new frame's hash against the hashes in the buffer. If a similar hash is found
    (within the configured threshold), the frame is considered a duplicate and
    filtered out.

    The filter uses perceptual hashing (phash by default) which is robust to
    minor variations in brightness, contrast, and compression artifacts.

    Attributes:
        buffer_size: Maximum number of frames to keep in temporal buffer
        threshold: Hash similarity threshold (0.0-1.0). Lower values are stricter.
                  For 8x8 hash, default 0.05 means ~3 bits difference allowed.
        hash_type: Type of perceptual hash to use ('phash', 'ahash', 'dhash', etc.)
        hash_size: Size of the hash in bits (default 8 for 8x8 = 64 bits)
        hasher: ImageHasher instance for computing and comparing hashes

    Example:
        >>> # Filter out frames with >95% hash similarity within last 50 frames
        >>> filter = DuplicateFilter(buffer_size=50, threshold=0.05)
        >>>
        >>> # Process frames
        >>> for packet in frame_stream:
        ...     if filter.should_pass(packet):
        ...         # Frame is unique, process it
        ...         process_frame(packet)
    """

    def __init__(
        self,
        buffer_size: int = 50,
        threshold: float = 0.05,
        hash_type: str = "phash",
        hash_size: int = 8,
        highfreq_factor: int = 4,
    ):
        """
        Initialize duplicate detection filter.

        Args:
            buffer_size: Maximum number of recent frames to compare against.
                        Larger values detect duplicates over longer time windows
                        but use more memory.
            threshold: Hash difference threshold as a fraction of max difference.
                      For 8x8 hash (64 bits), threshold=0.05 means 3.2 bits
                      difference allowed. Lower values are stricter.
                      Range: 0.0 (exact match only) to 1.0 (allow all).
            hash_type: Type of perceptual hash to use:
                      - 'phash': Perceptual hash (default, robust to minor changes)
                      - 'ahash': Average hash (faster, less robust)
                      - 'dhash': Difference hash (good for gradients)
                      - 'whash': Wavelet hash (more complex)
            hash_size: Size of the hash in bits (default 8 for 8x8 = 64 bits).
                      Larger values are more precise but slower.
            highfreq_factor: High frequency factor for wavelet hash (default 4).

        Raises:
            ValueError: If threshold is not in range [0.0, 1.0]
        """
        super().__init__(buffer_size)

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")

        self.threshold = threshold
        self.hash_type = hash_type
        self.hash_size = hash_size
        self.hasher = ImageHasher(hash_size=hash_size, highfreq_factor=highfreq_factor)

    def compare_with_history(
        self, packet: VideoFramePacket, history: list[VideoFramePacket]
    ) -> bool:
        """
        Compare current frame's hash against historical frame hashes.

        Computes the hash for the current frame and compares it against all
        hashes in the temporal buffer. If any hash is within the similarity
        threshold, the frame is considered a duplicate and filtered out.

        Args:
            packet: Current frame to evaluate. Must have 'phash' or 'hash_value'
                   tag computed by HashTagger.
            history: List of recent frames from the temporal buffer.

        Returns:
            True if frame is unique (no similar hash found), False if duplicate.
        """
        # Get current frame's hash from tags
        current_hash = packet.get_tag("hash_value")

        if current_hash is None:
            # If hash not computed yet, compute it now
            current_hash = self.hasher.compute_hash_from_array(packet.frame_data, self.hash_type)
            # Store it in tags for future use
            packet.tags["hash_value"] = current_hash
            packet.tags["phash"] = str(current_hash)

        # If buffer is empty, frame is unique
        if not history:
            return True

        # Calculate maximum allowed hash difference based on threshold
        max_diff = self.hash_size * self.hash_size
        threshold_diff = max_diff * self.threshold

        # Compare against all frames in history
        for past_packet in history:
            past_hash = past_packet.get_tag("hash_value")

            if past_hash is None:
                # Skip frames without hash (shouldn't happen in normal flow)
                continue

            # Calculate hash difference
            hash_diff = self.hasher.hash_difference(current_hash, past_hash)

            # If difference is within threshold, it's a duplicate
            if hash_diff <= threshold_diff:
                return False  # Duplicate found, filter out

        # No similar hash found, frame is unique
        return True

    @property
    def required_tags(self) -> list[str]:
        """
        Return list of tag keys required by this filter.

        The filter can work with either 'hash_value' (ImageHash object) or
        will compute it from frame_data if not present. Ideally, HashTagger
        should run before this filter to pre-compute hashes.

        Returns:
            Empty list since filter can compute hash if needed, but HashTagger
            is recommended for efficiency.
        """
        # Return empty list since we can compute hash if needed
        # But in practice, HashTagger should run first
        return []
