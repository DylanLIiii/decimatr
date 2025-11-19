"""
Hash-based tagger for computing perceptual hashes of video frames.

This tagger computes various types of perceptual hashes (phash, ahash, dhash)
for frames, which can be used by filters for duplicate detection and similarity
comparison. Unlike the original HashGate, this tagger is stateless and only
computes hash values without performing duplicate detection logic.
"""

from typing import Any

from decimatr.scheme import VideoFramePacket
from decimatr.taggers.base import Tagger
from decimatr.utils import ImageHasher


class HashTagger(Tagger):
    """
    Compute perceptual hashes for video frames.

    This tagger computes image hashes using various algorithms (phash, ahash, dhash)
    that can be used for duplicate detection and similarity comparison. The tagger
    is stateless and only computes hash values; duplicate detection logic is handled
    by separate filters.

    Supported hash types:
        - phash: Perceptual hash (default) - robust to minor modifications
        - ahash: Average hash - fast but less robust
        - dhash: Difference hash - good for detecting gradients
        - whash: Wavelet hash - more complex, better for some use cases
        - colorhash: Color-based hash - considers color information

    Args:
        hash_type: Type of hash to compute ('phash', 'ahash', 'dhash', 'whash', 'colorhash')
        hash_size: Size of the hash in bits (default: 8)
        highfreq_factor: High frequency factor for wavelet hash (default: 4)

    Example:
        >>> tagger = HashTagger(hash_type='phash', hash_size=8)
        >>> tags = tagger.compute_tags(packet)
        >>> # tags = {"phash": "a1b2c3d4e5f6g7h8", "hash_value": <ImageHash object>}
    """

    DEFAULT_HASH_TYPE = "phash"
    DEFAULT_HASH_SIZE = 8
    DEFAULT_HIGHFREQ_FACTOR = 4

    def __init__(
        self,
        hash_type: str = DEFAULT_HASH_TYPE,
        hash_size: int = DEFAULT_HASH_SIZE,
        highfreq_factor: int = DEFAULT_HIGHFREQ_FACTOR,
    ):
        """
        Initialize the HashTagger.

        Args:
            hash_type: Type of hash to use ('ahash', 'phash', 'dhash', 'whash', 'colorhash')
            hash_size: Size of the hash in bits
            highfreq_factor: High frequency factor for wavelet hash
        """
        self.hash_type = hash_type
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor

        # Validate hash type
        valid_hash_types = ["ahash", "phash", "dhash", "whash", "colorhash"]
        if hash_type not in valid_hash_types:
            raise ValueError(f"Invalid hash_type '{hash_type}'. Must be one of: {valid_hash_types}")

        # Create image hasher instance
        self.hasher = ImageHasher(hash_size=hash_size, highfreq_factor=highfreq_factor)

    def compute_tags(self, packet: VideoFramePacket) -> dict[str, Any]:
        """
        Compute perceptual hash for a single frame.

        This method computes the specified hash type for the frame and returns
        both the string representation and the ImageHash object. The ImageHash
        object can be used by filters for efficient hash comparison.

        Args:
            packet: VideoFramePacket containing frame data

        Returns:
            Dictionary with hash tags:
                - "{hash_type}": String representation of the hash
                - "hash_value": ImageHash object for comparison

        Example:
            >>> tagger = HashTagger(hash_type='phash')
            >>> tags = tagger.compute_tags(packet)
            >>> tags['phash']  # "a1b2c3d4e5f6g7h8"
            >>> tags['hash_value']  # <ImageHash object>
        """
        # Compute hash using the ImageHasher
        hash_value = self.hasher.compute_hash_from_array(packet.frame_data, self.hash_type)

        # Return both string representation and hash object
        return {self.hash_type: str(hash_value), "hash_value": hash_value}

    @property
    def tag_keys(self) -> list[str]:
        """
        Return list of tag keys this tagger produces.

        Returns:
            List containing the hash type key and "hash_value"

        Example:
            >>> tagger = HashTagger(hash_type='phash')
            >>> tagger.tag_keys
            ['phash', 'hash_value']
        """
        return [self.hash_type, "hash_value"]

    @property
    def supports_gpu(self) -> bool:
        """
        Hash computation is CPU-based and does not support GPU acceleration.

        Returns:
            False (hash computation is CPU-only)
        """
        return False

    @property
    def requires_gpu(self) -> bool:
        """
        Hash computation does not require GPU.

        Returns:
            False (hash computation is CPU-only)
        """
        return False
