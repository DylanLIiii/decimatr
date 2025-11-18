"""
Image hashing utility module for the data pipeline.

This module provides functions for generating various types of image hashes,
comparing images, and finding similar images based on hash distances.
"""

import imagehash
from PIL import Image
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Constants for hash size
DEFAULT_HASH_SIZE = 8
DEFAULT_HIGHFREQ_FACTOR = 4

class ImageHasher:
    """Utility class for image hashing operations."""
    
    def __init__(self, hash_size: int = DEFAULT_HASH_SIZE, highfreq_factor: int = DEFAULT_HIGHFREQ_FACTOR):
        """
        Initialize the ImageHasher with specified parameters.
        
        Args:
            hash_size: Size of the hash in bits. Default is 8.
            highfreq_factor: High frequency factor for wavelet hash. Default is 4.
        """
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor
    
    def compute_hashes(self, image_path: Union[str, Path]) -> Dict[str, imagehash.ImageHash]:
        """
        Compute multiple hash types for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing different hash types
        """
        try:
            img = Image.open(image_path)
            
            # Compute different hash types
            return {
                'ahash': imagehash.average_hash(img, hash_size=self.hash_size),
                'phash': imagehash.phash(img, hash_size=self.hash_size, highfreq_factor=self.highfreq_factor),
                'dhash': imagehash.dhash(img, hash_size=self.hash_size),
                'whash': imagehash.whash(img, hash_size=self.hash_size),
                'colorhash': imagehash.colorhash(img, binbits=3)
            }
        except Exception as e:
            logger.error(f"Error computing hash for {image_path}: {e}")
            raise
    
    def compute_hash(self, image_path: Union[str, Path], hash_type: str = 'phash') -> imagehash.ImageHash:
        """
        Compute a specific hash type for an image.
        
        Args:
            image_path: Path to the image file
            hash_type: Type of hash to compute ('ahash', 'phash', 'dhash', 'whash', 'colorhash')
            
        Returns:
            ImageHash object
        """
        try:
            img = Image.open(image_path)
            
            if hash_type == 'ahash':
                return imagehash.average_hash(img, hash_size=self.hash_size)
            elif hash_type == 'phash':
                return imagehash.phash(img, hash_size=self.hash_size, highfreq_factor=self.highfreq_factor)
            elif hash_type == 'dhash':
                return imagehash.dhash(img, hash_size=self.hash_size)
            elif hash_type == 'whash':
                return imagehash.whash(img, hash_size=self.hash_size)
            elif hash_type == 'colorhash':
                return imagehash.colorhash(img, binbits=3)
            else:
                raise ValueError(f"Unsupported hash type: {hash_type}")
        except Exception as e:
            logger.error(f"Error computing {hash_type} for {image_path}: {e}")
            raise
    
    def compute_hash_from_array(self, image_array: np.ndarray, hash_type: str = 'phash') -> imagehash.ImageHash:
        """
        Compute hash from a numpy array image representation.
        
        Args:
            image_array: Numpy array containing image data
            hash_type: Type of hash to compute
            
        Returns:
            ImageHash object
        """
        try:
            img = Image.fromarray(image_array)
            return self.compute_hash_from_pil(img, hash_type)
        except Exception as e:
            logger.error(f"Error computing hash from array: {e}")
            raise
    
    def compute_hash_from_pil(self, img: Image.Image, hash_type: str = 'phash') -> imagehash.ImageHash:
        """
        Compute hash from a PIL Image object.
        
        Args:
            img: PIL Image object
            hash_type: Type of hash to compute
            
        Returns:
            ImageHash object
        """
        try:
            if hash_type == 'ahash':
                return imagehash.average_hash(img, hash_size=self.hash_size)
            elif hash_type == 'phash':
                return imagehash.phash(img, hash_size=self.hash_size, highfreq_factor=self.highfreq_factor)
            elif hash_type == 'dhash':
                return imagehash.dhash(img, hash_size=self.hash_size)
            elif hash_type == 'whash':
                return imagehash.whash(img, hash_size=self.hash_size)
            elif hash_type == 'colorhash':
                return imagehash.colorhash(img, binbits=3)
            else:
                raise ValueError(f"Unsupported hash type: {hash_type}")
        except Exception as e:
            logger.error(f"Error computing hash from PIL image: {e}")
            raise
    
    @staticmethod
    def hash_difference(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> int:
        """
        Calculate the difference between two hashes.
        
        Args:
            hash1: First image hash
            hash2: Second image hash
            
        Returns:
            Integer representing the bit difference between hashes
        """
        return hash1 - hash2
    
    @staticmethod
    def hash_similarity(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
        """
        Calculate normalized similarity (0-1) between two hashes.
        
        Args:
            hash1: First image hash
            hash2: Second image hash
            
        Returns:
            Float between 0 and 1 representing similarity (1 = identical)
        """
        difference = hash1 - hash2
        max_bits = max(len(hash1.hash.flatten()), len(hash2.hash.flatten()))
        return 1.0 - (difference / max_bits)
    
    def find_similar_images(self, 
                           query_image: Union[str, Path, Image.Image, np.ndarray],
                           target_images: List[Union[str, Path]],
                           hash_type: str = 'phash',
                           threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find similar images from a list based on hash similarity.
        
        Args:
            query_image: Query image (path, PIL Image, or numpy array)
            target_images: List of paths to potential match images
            hash_type: Hash type to use for comparison
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of tuples containing (image_path, similarity_score) sorted by similarity
        """
        # Compute hash for query image
        if isinstance(query_image, (str, Path)):
            query_hash = self.compute_hash(query_image, hash_type)
        elif isinstance(query_image, Image.Image):
            query_hash = self.compute_hash_from_pil(query_image, hash_type)
        elif isinstance(query_image, np.ndarray):
            query_hash = self.compute_hash_from_array(query_image, hash_type)
        else:
            raise TypeError("Query image must be a path, PIL Image, or numpy array")
        
        # Compute similarities for all target images
        results = []
        for img_path in target_images:
            try:
                target_hash = self.compute_hash(img_path, hash_type)
                similarity = self.hash_similarity(query_hash, target_hash)
                
                if similarity >= threshold:
                    results.append((str(img_path), similarity))
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue
        
        # Sort by similarity (highest first)
        return sorted(results, key=lambda x: x[1], reverse=True)


def get_default_hasher() -> ImageHasher:
    """
    Get a default configured ImageHasher instance.
    
    Returns:
        ImageHasher with default settings
    """
    return ImageHasher() 