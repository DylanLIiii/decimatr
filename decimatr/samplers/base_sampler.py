import abc
from typing import Dict, List, Optional, Any, Iterator, Union
import logging
import uuid

from decimatr.scheme import VideoFramePacket
from decimatr.logging_config import setup_logging

class BaseSampler(abc.ABC):
    """
    Abstract base class defining the protocol for video frame samplers.
    
    Samplers are responsible for selecting frames from a filtered list based on specific criteria.
    They operate as the final step in the video processing pipeline after gates and VRA.
    """
    
    @abc.abstractmethod
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the sampler with optional configuration.
        
        Args:
            config: Optional dictionary containing configuration parameters
        """
        pass
    
    @abc.abstractmethod
    def sample(self, frames: Union[List[VideoFramePacket], Iterator[VideoFramePacket]], session_id: str) -> List[VideoFramePacket]:
        """
        Sample frames based on specific criteria.
        
        Args:
            frames: List or Iterator of VideoFramePacket objects for frames that have
                  passed previous processing stages
            session_id: Unique identifier for the current processing session
            
        Returns:
            List of selected VideoFramePacket objects
        """
        pass

class MaxFramesSampler(BaseSampler):
    """
    Base implementation of a frame sampler.
    
    This implementation can be configured to select a specified number of frames
    from the input list, defaulting to returning all frames if not specified.
    """
    
    DEFAULT_MAX_FRAMES = 10
    DEFAULT_SELECTION_METHOD = 'first'
    
    def __init__(self,
                 max_frames: int = DEFAULT_MAX_FRAMES,
                 selection_method: str = DEFAULT_SELECTION_METHOD):
        """
        Initialize the base sampler with direct parameters.
        
        Args:
            max_frames: Maximum number of frames to select (default: 10)
            selection_method: How to select frames if max_frames is set
                             (options: 'first', 'last', 'evenly_spaced', default: 'first')
        """
        self.max_frames = max_frames
        self.selection_method = selection_method
        self.logger = setup_logging(self.__class__.__name__)
        
    def sample(self, frames: Union[List[VideoFramePacket], Iterator[VideoFramePacket]], session_id: str) -> List[VideoFramePacket]:
        """
        Sample frames based on configuration.
        
        Args:
            frames: List or Iterator of VideoFramePacket objects
            session_id: Unique identifier for the current processing session
            
        Returns:
            List of selected VideoFramePacket objects
        """
        # Convert iterator to list if needed
        if not isinstance(frames, list):
            frames = list(frames)
            
        if not frames:
            self.logger.warning(f"No frames to sample for session: {session_id}")
            return []
        
        total_frames = len(frames)
        self.logger.info(f"Sampling from {total_frames} frames for session: {session_id}")
        
        # If max_frames is not set or greater than total frames, return all frames
        if self.max_frames is None or self.max_frames >= total_frames:
            self.logger.info(f"Returning all {total_frames} frames")
            return frames
        
        selected_frames = []
        
        # Select frames based on the configured method
        if self.selection_method == 'first':
            selected_frames = frames[:self.max_frames]
            self.logger.info(f"Selected first {len(selected_frames)} frames")
        
        elif self.selection_method == 'last':
            selected_frames = frames[-self.max_frames:]
            self.logger.info(f"Selected last {len(selected_frames)} frames")
        
        elif self.selection_method == 'evenly_spaced':
            if total_frames <= self.max_frames:
                selected_frames = frames
            else:
                # Calculate indices for evenly spaced frames
                step = total_frames / self.max_frames
                indices = [int(i * step) for i in range(self.max_frames)]
                selected_frames = [frames[i] for i in indices]
            self.logger.info(f"Selected {len(selected_frames)} evenly spaced frames")
        
        else:
            self.logger.warning(f"Unknown selection method: {self.selection_method}, using 'first'")
            selected_frames = frames[:self.max_frames]
        
        # Log sampling operation
        log_d = {
            "component_name": "BaseSampler",
            "operation": "frame_sampling",
            "outcome": "completed",
            "session_id": session_id,
            "event_id": str(uuid.uuid4()),
            "relevant_metadata": {
                "total_frames": total_frames,
                "selected_frames": len(selected_frames),
                "max_frames": self.max_frames,
                "selection_method": self.selection_method
            }
        }
        self.logger.info(f"Frame sampling completed for session: {session_id}", extra=log_d)
        
        return selected_frames


