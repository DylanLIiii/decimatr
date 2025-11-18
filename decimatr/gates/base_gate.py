from abc import ABC, abstractmethod
from typing import List, Iterator, Optional, TypeVar, Generic, Union

from decimatr.scheme import VideoFramePacket

# Type variable for return type of process_frame
T = TypeVar('T')

class BaseGate(Generic[T], ABC):
    """
    Abstract base class for all gate implementations.
    
    Gates are responsible for deciding whether frames should pass through
    or be filtered out, typically by analyzing frame content and returning a boolean.
    """
    
    @abstractmethod
    def process_frame(self, packet: VideoFramePacket) -> Union[bool, T]:
        """
        Process a single video frame packet to determine if it should pass through the gate.
        
        Args:
            packet (VideoFramePacket): The frame packet containing the frame data and metadata
            
        Returns:
            Union[bool, T]: True if the frame passes the gate criteria, False if it should be filtered.
                           Implementations may return additional data of type T.
        """
        pass
    
    def process(self, packets: List[VideoFramePacket]) -> List[Union[bool, T]]:
        """
        Process a list of video frame packets.
        
        Args:
            packets (List[VideoFramePacket]): List of frame packets to process
            
        Returns:
            List[Union[bool, T]]: List of results from processing each frame
        """
        return [self.process_frame(packet) for packet in packets]
    
    def process_iter(self, packets: Iterator[VideoFramePacket]) -> Iterator[Union[bool, T]]:
        """
        Process an iterator of video frame packets.
        
        Args:
            packets (Iterator[VideoFramePacket]): Iterator of frame packets to process
            
        Returns:
            Iterator[Union[bool, T]]: Iterator of results from processing each frame
        """
        for packet in packets:
            yield self.process_frame(packet)
            
    def __call__(self, packet: VideoFramePacket) -> Union[bool, T]:
        """
        Makes the gate callable, delegating to process_frame.
        This ensures backward compatibility with existing code that expects gates to be callable.
        
        Args:
            packet (VideoFramePacket): The frame packet to process
            
        Returns:
            Union[bool, T]: The result from process_frame
        """
        return self.process_frame(packet)