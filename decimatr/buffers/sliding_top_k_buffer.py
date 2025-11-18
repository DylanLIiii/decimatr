import collections
import heapq
from typing import Any, Deque, List, Tuple, Optional, Union, Dict
import logging
import uuid

from .base_buffer import BaseBuffer
from decimatr.scheme import VideoFramePacket

class SlidingTopKBuffer(BaseBuffer):
    """
    A buffer that maintains a sliding window of frames and tracks the
    top K frames based on a given metric.
    """
    def __init__(self, window_size: int, k: int, similarity_threshold: float, session_id: str = "default_session"):
        """
        Initializes the SlidingTopKBuffer.

        Args:
            window_size: Maximum number of frames to keep in the sliding window.
            k: The number of "top" frames to keep track of.
            similarity_threshold: A threshold used to decide if a frame is
                                  too similar to existing frames.
            session_id (str): Identifier for the current session, used in logging.
        """
        super().__init__(window_size, True, session_id)
        self.k: int = k
        self.similarity_threshold: float = similarity_threshold
        
        # Min-heap storing (actual_metric, VideoFramePacket)
        self.top_k_heap: List[Tuple[float, VideoFramePacket]] = []

    def add(self, packet: VideoFramePacket, frame_metric: float, frame_id: Optional[str] = None) -> bool:
        """
        Adds a frame packet to the sliding window and updates the Top-K heap.

        The method first adds the `(frame_metric, packet)` to the internal sliding
        window (`self.window`).

        It then determines if the frame should be added to or update `self.top_k_heap`:
        1. Similarity Check: The new `frame_metric` is compared against all
           metrics of frames currently in `self.top_k_heap`. If the absolute
           difference is less than `self.similarity_threshold` for any existing
           frame, the new frame is considered "similar" to an existing Top-K item.
        2. Heap Update:
           - If `self.top_k_heap` has fewer than `self.k` items:
             - If the new frame is NOT "similar" to any existing Top-K item,
               it is added to `self.top_k_heap`.
             - If it IS "similar", it is not added at this stage (it may still be
               added if the heap is full and it's better than the worst item).
           - If `self.top_k_heap` is full (contains `self.k` items):
             - If the new `frame_metric` is greater than the metric of the "worst"
               item currently in `self.top_k_heap` (the smallest metric, as
               it's a min-heap storing actual metrics), the "worst" item is
               replaced with the new frame. This happens regardless of the
               similarity check outcome if the frame is better.
        
        Args:
            packet: The VideoFramePacket containing the frame data and metadata.
            frame_metric: A numerical value representing the frame's importance or
                          uniqueness. Higher values are considered better.
            frame_id: Optional identifier for the frame, used in logging. If None,
                      it will be derived from the packet.

        Returns:
            True if the frame was added to `self.top_k_heap` or replaced an
            existing item in it. False otherwise.
        """
        # If frame_id is not provided, derive it from the packet
        if frame_id is None:
            frame_id = f"{packet.source_video_id}_{packet.frame_number}"
        # Call the parent class add method to add the packet to the base window
        if not super().add(packet, frame_id):
            return False

        operation_name = "add_frame_top_k_check"
        event_id = str(uuid.uuid4())
        log_base = {
            "component_name": self.__class__.__name__,
            "operation": operation_name,
            "event_id": event_id,
            "session_id": self.session_id,
        }

        if self.k == 0:
            self.logger.debug(f"Frame {frame_id} (metric: {frame_metric}) not considered for Top-K as k=0.", extra={
                **log_base,
                "outcome": "skipped_k_is_zero",
                "relevant_metadata": {"frame_id": frame_id, "frame_metric": frame_metric, "k": self.k}
            })
            return False # No top-K tracking

        # 2. Top-K Heap Update Logic
        is_similar_to_an_existing_top_k_item = False
        similar_to_metric = None
        if self.top_k_heap:
            for existing_metric_in_heap, _ in self.top_k_heap:
                if abs(frame_metric - existing_metric_in_heap) < self.similarity_threshold:
                    is_similar_to_an_existing_top_k_item = True
                    similar_to_metric = existing_metric_in_heap
                    break
        
        # Case 1: Heap is not full
        if len(self.top_k_heap) < self.k:
            if not is_similar_to_an_existing_top_k_item:
                heapq.heappush(self.top_k_heap, (frame_metric, packet))
                self.logger.info(f"Frame {frame_id} (metric: {frame_metric}) added to Top-K (heap not full).", extra={
                    **log_base,
                    "outcome": "added_to_top_k_not_full",
                    "relevant_metadata": {
                        "frame_id": frame_id, "frame_metric": frame_metric,
                        "heap_size_before": len(self.top_k_heap) -1, "heap_size_after": len(self.top_k_heap),
                        "is_similar": False
                    }
                })
                return True
            else:
                self.logger.info(f"Frame {frame_id} (metric: {frame_metric}) not added to Top-K (heap not full, but similar to existing metric {similar_to_metric}).", extra={
                     **log_base,
                    "outcome": "skipped_similar_heap_not_full",
                    "relevant_metadata": {
                        "frame_id": frame_id, "frame_metric": frame_metric,
                        "similar_to_metric": similar_to_metric,
                        "similarity_threshold": self.similarity_threshold,
                        "heap_size": len(self.top_k_heap)
                    }
                })
                return False

        # Case 2: Heap is full
        elif len(self.top_k_heap) == self.k:
            # The new frame must be better than the "worst" frame in the Top-K (smallest metric).
            if frame_metric > self.top_k_heap[0][0]: # Smallest item is at index 0 for min-heap
                worst_metric_before_replace = self.top_k_heap[0][0]
                heapq.heapreplace(self.top_k_heap, (frame_metric, packet))
                self.logger.info(f"Frame {frame_id} (metric: {frame_metric}) replaced worst in Top-K (heap full). Worst was {worst_metric_before_replace}", extra={
                    **log_base,
                    "outcome": "replaced_worst_in_top_k",
                    "relevant_metadata": {
                        "frame_id": frame_id, "frame_metric": frame_metric,
                        "worst_metric_replaced": worst_metric_before_replace,
                        "heap_size": len(self.top_k_heap),
                        "is_similar_if_checked_earlier": is_similar_to_an_existing_top_k_item # Informational
                    }
                })
                return True
            else:
                self.logger.info(f"Frame {frame_id} (metric: {frame_metric}) not added to Top-K (heap full, not better than worst: {self.top_k_heap[0][0]}).", extra={
                    **log_base,
                    "outcome": "skipped_not_better_than_worst_heap_full",
                    "relevant_metadata": {
                        "frame_id": frame_id, "frame_metric": frame_metric,
                        "worst_metric_in_heap": self.top_k_heap[0][0],
                        "heap_size": len(self.top_k_heap)
                    }
                })
                return False
        
        # Should not be reached if k > 0, but as a fallback
        self.logger.warning(f"Frame {frame_id} (metric: {frame_metric}) evaluation fell through without action.", extra={
            **log_base,
            "outcome": "no_action_fallback",
            "relevant_metadata": {"frame_id": frame_id, "frame_metric": frame_metric, "heap_size": len(self.top_k_heap), "k": self.k}
        })
        return False

    def get_top_k(self) -> List[VideoFramePacket]:
        """
        Returns a list of the VideoFramePacket objects from the k items
        currently in self.top_k_heap, sorted by their frame_metric
        in descending order (best frame first).

        Uses heapq.nlargest to efficiently retrieve the top K items.
        `self.top_k_heap` stores tuples of (frame_metric, VideoFramePacket).

        Returns:
            List[VideoFramePacket]: A list of VideoFramePacket objects from the top K frames,
                                    sorted by frame_metric in descending order.
        """
        if not self.top_k_heap:
            return []
        # Get the K items with the largest metrics.
        # heapq.nlargest returns them sorted from largest to smallest by the key.
        top_k_items_with_metrics = heapq.nlargest(self.k, self.top_k_heap, key=lambda x: x[0])
        # Extract just the frame_data (the second element of each tuple).
        return [item[1] for item in top_k_items_with_metrics]

    def clear(self) -> None:
        """
        Clears both the sliding window and the top-k heap.
        """
        super().clear()  # Clear the base window
        self.top_k_heap = []  # Clear the top-k heap

    def flush(self) -> List[VideoFramePacket]:
        """
        Returns the current Top-K frames and then clears the internal
        state of the buffer (both self.window and self.top_k_heap).

        This prepares the buffer for new data if it is to be reused.

        Returns:
            List[VideoFramePacket]: A list of VideoFramePacket objects from the top K frames,
                                    sorted by frame_metric in descending order,
                                    before the buffer was cleared.
        """
        top_k_frames = self.get_top_k()
        self.clear()
        return top_k_frames