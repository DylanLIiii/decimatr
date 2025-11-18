import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple, Union

from decimatr.gates.blur_gate import BlurGate
from decimatr.gates.entropy_gate import EntropyGate
from decimatr.gates.grid_gate import GridGate
from decimatr.gates.hash_gate import HashGate
from decimatr.logging_config import setup_logging
from decimatr.samplers.base_sampler import MaxFramesSampler
from decimatr.samplers.uniform_sampler import UniformSampler
from decimatr.scheme import VideoFramePacket
from decimatr.video_loader import load_video_frames


# Define a minimal placeholder for VideoContent since it's no longer in scheme.py
class VideoContent:
    """
    Placeholder for backward compatibility with the original VideoContent class.
    This will be used as a type hint and for isinstance checks.
    """

    class MetaData:
        def dict(self):
            return {}

    def __init__(self, frames=None, meta_data=None, stacked_frames=None):
        self.frames = frames or {}
        self.meta_data = meta_data or self.MetaData()
        self.stacked_frames = stacked_frames


class Orchestrator:
    """
    Orchestrates the preprocessing of video frames through various components.

    This class initializes and manages different preprocessing components such as
    Samplers, Gates (BlurGate, EntropyGate, GridGate, HashGate), and VideoRepetitionAnalyzer
    based on the provided configurations. It processes video frames in a specified pipeline
    order, applying each component's logic to the frames. The results of the processing,
    including metrics and any errors encountered, are logged and summarized.

    Attributes:
        logger: Logger instance for logging information and errors.
        components: Dictionary storing component instances keyed by their names.
        pipeline_order_active: List of active component names in the order they should be applied.
        sampler: Optional sampler instance for frame selection.
        buffer: Optional buffer instance for storing selected frames.

    Methods:
        __init__: Initializes the orchestrator with optional configurations for each component.
        process_video: Processes a video file, applying the active components in the specified order.
        _process_video_content: Processes a VideoContent object, applying the active components.
    """

    def __init__(
        self,
        gates_config: Optional[Dict] = None,
        sampler_config: Optional[Dict] = None,
        buffer_config: Optional[Dict] = None,
        pipeline_order: Optional[List[str]] = None,
        head_frames_to_keep: int = 0,
        tail_frames_to_keep: int = 0,
        logger_name: str = "PreprocessingOrchestrator",
        sampler_gate_order: str = "sampler_first",  # "sampler_first" or "gates_first"
    ):
        self.logger = setup_logging(logger_name)
        self.components = {}  # Stores component_name -> instance
        self.pipeline_order_active = []  # List of active component names in order
        self.sampler = None
        self.buffer = None
        # Configuration for head and tail frame retention
        self.head_frames_to_keep = head_frames_to_keep
        self.tail_frames_to_keep = tail_frames_to_keep
        self.sampler_gate_order = sampler_gate_order
        if self.sampler_gate_order not in ["sampler_first", "gates_first"]:
            raise ValueError(
                "sampler_gate_order must be 'sampler_first' or 'gates_first'"
            )

        self.logger.info(
            f"Orchestrator initialized with head_frames_to_keep={head_frames_to_keep}, "
            f"tail_frames_to_keep={tail_frames_to_keep}, sampler_gate_order='{self.sampler_gate_order}'"
        )
        # Initialize gate components based on configs
        if gates_config:
            for gate_name, gate_config in gates_config.items():
                if not gate_config.get("enabled", False):
                    continue

                gate_type = gate_config.get("type")
                gate_settings = gate_config.get("settings", {})

                if gate_type == "BlurGate":
                    self.components[gate_name] = BlurGate(**gate_settings)
                elif gate_type == "EntropyGate":
                    self.components[gate_name] = EntropyGate(**gate_settings)
                elif gate_type == "GridGate":
                    self.components[gate_name] = GridGate(**gate_settings)
                elif gate_type == "HashGate":
                    self.components[gate_name] = HashGate(**gate_settings)
                else:
                    self.logger.warning(
                        f"Unknown gate type: {gate_type} for {gate_name}"
                    )

        # Set pipeline order (default if not specified)
        self.pipeline_order_active = pipeline_order or list(self.components.keys())

        # Initialize sampler if configured
        if sampler_config and sampler_config.get("enabled", False):
            sampler_type = sampler_config.get("type", "MaxFramesSampler")
            sampler_settings = sampler_config.get("settings", {})

            # Default to MaxFramesSampler, but can be extended for other samplers
            if sampler_type == "MaxFramesSampler":
                self.sampler = MaxFramesSampler(**sampler_settings)
            elif sampler_type == "UniformSampler":
                self.sampler = UniformSampler(**sampler_settings)
            else:
                self.logger.warning(f"Unknown sampler type: {sampler_type}")

        # Initialize buffer if configured
        if buffer_config and buffer_config.get("enabled", False):
            buffer_type = buffer_config.get("type")
            buffer_settings = buffer_config.get("settings", {})

            if buffer_type == "SlidingTopKBuffer":
                from decimatr.buffers.sliding_top_k_buffer import SlidingTopKBuffer

                self.buffer = SlidingTopKBuffer(
                    window_size=buffer_settings.get("window_size", 50),
                    k=buffer_settings.get("k", 10),
                    similarity_threshold=buffer_settings.get(
                        "similarity_threshold", 0.1
                    ),
                    session_id=logger_name,
                )
                self.logger.info(
                    f"SlidingTopKBuffer initialized with settings: {buffer_settings}"
                )
            elif buffer_type == "SlidingBuffer":
                from decimatr.buffers.sliding_buffer import SlidingBuffer

                self.buffer = SlidingBuffer(
                    window_size=buffer_settings.get("window_size", 50),
                    step_size=buffer_settings.get("step_size", 1),
                )
            # Can add future buffer types here with elif statements
            else:
                self.logger.warning(f"Unknown buffer type: {buffer_type}")

    def process_video(
        self,
        video_path: Union[str, VideoContent, Iterator[VideoFramePacket]],
        session_id: str,
    ) -> Tuple[List[VideoFramePacket], Dict]:
        """
        Process a video through the orchestration pipeline.

        Args:
            video_path: Path to video file, VideoContent object, or Iterator of VideoFramePacket objects
            session_id: Unique identifier for this processing session

        Returns:
            Tuple containing:
                - List of final selected VideoFramePacket objects.
                - Dictionary with processing results and metrics.
        """
        if isinstance(video_path, str) and os.path.exists(video_path):
            return self._process_video_file(video_path, session_id)
        else:
            self.logger.error(
                f"Invalid video_path type or path does not exist: {video_path} for session_id: {session_id}"
            )
            error_summary = {
                "session_id": session_id,
                "video_path": str(video_path),
                "orchestrator_errors": [f"Invalid video_path: {video_path}"],
                "overall_status": "failed",
                "total_frames_input": 0,
                "frames_processed": 0,
                "frames_dropped": 0,
                "final_selected_frames": 0,
                "components": {name: {} for name in self.components},
                "errors": [],
                "processing_start_time_utc": datetime.now(timezone.utc).isoformat(),
                "processing_end_time_utc": datetime.now(timezone.utc).isoformat(),
                "total_duration_seconds": 0,
            }
            return [], error_summary

    def _keep_head_and_tail_frames(
        self, frames: List[VideoFramePacket]
    ) -> Dict[str, List[VideoFramePacket]]:
        """
        Keep the specified number of frames from the beginning and end of the list.

        Args:
            frames: List of VideoFramePacket objects

        Returns:
            Dict containing 'head_frames' and 'tail_frames' lists
        """
        head_frames = []
        tail_frames = []

        if frames:
            # Get head frames (if there are enough frames)
            head_frames = frames[: min(self.head_frames_to_keep, len(frames))]

            # Get tail frames (if there are enough frames after taking head frames)
            remaining_frames = len(frames) - len(head_frames)
            if remaining_frames > 0 and self.tail_frames_to_keep > 0:
                tail_frames = frames[-min(self.tail_frames_to_keep, remaining_frames) :]

        self.logger.info(
            f"Preserved {len(head_frames)} head frames and {len(tail_frames)} tail frames"
        )

        return {"head_frames": head_frames, "tail_frames": tail_frames}

    def _process_video_file(
        self, video_path: str, session_id: str
    ) -> Tuple[List[VideoFramePacket], Dict]:
        start_time = datetime.now(timezone.utc)
        self.logger.info(
            f"Starting video processing for session_id: {session_id}, video_path: {video_path}"
        )

        results_summary = {
            "session_id": session_id,
            "video_path": video_path,
            "processing_start_time_utc": start_time.isoformat(),
            "total_frames_input": 0,
            "frames_processed": 0,  # Frames entering the gating stage
            "frames_dropped": 0,
            "components": {name: {} for name in self.components},
            "orchestrator_errors": [],
            "errors": [],  # For errors from individual components like gates
            "final_selected_frames": 0,
        }
        final_selected_packets: List[VideoFramePacket] = []

        try:
            # 1. Load frames, separate head/tail, get core frames for processing
            core_frames_to_process, preserved_frames_dict, total_initial_frames = (
                self._load_and_prepare_frames_for_processing(video_path, session_id)
            )
            results_summary["total_frames_input"] = total_initial_frames

            frames_after_processing_logic: List[VideoFramePacket] = []
            all_gate_metrics: Dict[int, float] = {}

            if self.sampler_gate_order == "sampler_first":
                self.logger.info(
                    f"Order: Sampler first. Processing {len(core_frames_to_process)} core frames for session: {session_id}"
                )
                # 2a. Apply sampler to core frames
                frames_ready_for_gates = self._apply_sampler_if_configured(
                    core_frames_to_process, session_id
                )
                self.logger.info(
                    f"Sampler selected {len(frames_ready_for_gates)} frames. Proceeding to gates for session: {session_id}"
                )

                # 3a. Apply gates to sampled frames
                frames_passed_gates, all_gate_metrics = (
                    self._apply_gates_and_collect_metrics(
                        frames_ready_for_gates, session_id, results_summary
                    )
                )

                # 4a. Apply buffer to gate-passed frames
                frames_after_processing_logic = self._apply_buffer_if_configured(
                    frames_passed_gates, all_gate_metrics, session_id
                )

            elif self.sampler_gate_order == "gates_first":
                self.logger.info(
                    f"Order: Gates first. Processing {len(core_frames_to_process)} core frames for session: {session_id}"
                )
                # 2b. Apply gates to core frames
                frames_passed_gates, all_gate_metrics = (
                    self._apply_gates_and_collect_metrics(
                        core_frames_to_process, session_id, results_summary
                    )
                )

                # 3b. Apply buffer to gate-passed frames
                frames_after_buffer = self._apply_buffer_if_configured(
                    frames_passed_gates, all_gate_metrics, session_id
                )
                self.logger.info(
                    f"{len(frames_after_buffer)} frames obtained from buffer. Proceeding to sampler for session: {session_id}"
                )

                # 4b. Apply sampler to buffered frames
                frames_after_processing_logic = self._apply_sampler_if_configured(
                    frames_after_buffer, session_id
                )
            else:
                # This should be caught by __init__, but as a safeguard
                err_msg = f"Internal error: Invalid sampler_gate_order '{self.sampler_gate_order}' for session {session_id}"
                self.logger.error(err_msg)
                results_summary["orchestrator_errors"].append(err_msg)
                # `frames_after_processing_logic` remains empty

            # 5. Assemble final list of VideoFramePacket objects
            final_selected_packets.extend(preserved_frames_dict["head_frames"])
            final_selected_packets.extend(frames_after_processing_logic)
            final_selected_packets.extend(preserved_frames_dict["tail_frames"])

            results_summary["final_selected_frames"] = len(final_selected_packets)
            self.logger.info(
                f"Collected {len(final_selected_packets)} final frames after processing pipeline for session {session_id}."
            )

        except Exception as e:
            error_msg = f"Unhandled error during video processing for session {session_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            results_summary["orchestrator_errors"].append(error_msg)
            final_selected_packets = []  # Ensure empty list on major error

        # Finalize processing metrics
        end_time = datetime.now(timezone.utc)
        results_summary["processing_end_time_utc"] = end_time.isoformat()
        results_summary["total_duration_seconds"] = (
            end_time - start_time
        ).total_seconds()

        results_summary["frames_dropped"] = results_summary["total_frames_input"] - len(
            final_selected_packets
        )

        # Determine overall status based on any errors recorded
        if results_summary["orchestrator_errors"] or results_summary.get(
            "errors", []
        ):  # .get for 'errors' as it's populated by a helper
            results_summary["overall_status"] = "failed"
        else:
            results_summary["overall_status"] = "completed"

        # Final summary log
        summary_log_d = {
            "component_name": "PreprocessingOrchestrator",
            "operation": "video_processing_summary",
            "outcome": results_summary["overall_status"],
            "session_id": session_id,
            "event_id": str(uuid.uuid4()),
            "relevant_metadata": {
                "total_frames_input": results_summary["total_frames_input"],
                "frames_processed_by_gates": results_summary[
                    "frames_processed"
                ],  # Clarify this metric
                "final_selected_frames": results_summary["final_selected_frames"],
                "frames_dropped": results_summary["frames_dropped"],
                "duration_seconds": results_summary["total_duration_seconds"],
                "error_count": len(results_summary["orchestrator_errors"])
                + len(results_summary.get("errors", [])),
            },
        }
        self.logger.info(
            f"Video processing finished for session: {session_id}. Status: {results_summary['overall_status']}",
            extra=summary_log_d,
        )
        return final_selected_packets, results_summary

    def _load_and_prepare_frames_for_processing(self, video_path: str, session_id: str):
        """Loads all frames, separates head/tail, returns core frames and preserved."""
        self.logger.info(
            f"Loading and preparing frames for video: {video_path}, session_id: {session_id}"
        )
        video_packets_iterator = load_video_frames(video_path, session_id)
        all_video_packets = list(video_packets_iterator)
        total_initial_frames = len(all_video_packets)

        core_frames = all_video_packets
        preserved_frames_dict = {"head_frames": [], "tail_frames": []}

        if self.head_frames_to_keep > 0 or self.tail_frames_to_keep > 0:
            preserved_frames_dict = self._keep_head_and_tail_frames(all_video_packets)

            preserved_frame_numbers = set()
            for frame in (
                preserved_frames_dict["head_frames"]
                + preserved_frames_dict["tail_frames"]
            ):
                preserved_frame_numbers.add(frame.frame_number)

            core_frames = [
                packet
                for packet in all_video_packets
                if packet.frame_number not in preserved_frame_numbers
            ]
            self.logger.info(
                f"After preserving head/tail, {len(core_frames)} core frames remain from {total_initial_frames} total."
            )
        else:
            self.logger.info(
                f"No head/tail preservation configured. {len(core_frames)} core frames to process from {total_initial_frames} total."
            )

        return core_frames, preserved_frames_dict, total_initial_frames

    def _apply_gates_and_collect_metrics(
        self,
        frames_to_gate: List[VideoFramePacket],
        session_id: str,
        results_summary: Dict,
    ):
        """Applies gates to frames, updates summary, returns passed frames and metrics."""
        self.logger.info(
            f"Applying gates to {len(frames_to_gate)} frames for session: {session_id}"
        )
        passed_gate_frames = []
        gate_passage_metrics = {}  # Metrics for all frames that entered gating
        current_metric_counter = 1.0

        for packet in frames_to_gate:
            results_summary["frames_processed"] += 1
            current_frame_is_valid = True

            # Assign metric before gate processing, as per original logic
            gate_passage_metrics[packet.frame_number] = current_metric_counter
            current_metric_counter += 1.0

            for component_name in self.pipeline_order_active:
                component = self.components.get(component_name)
                if not component or not isinstance(
                    component, (BlurGate, EntropyGate, GridGate, HashGate)
                ):
                    continue  # Skip if not a gate or not found

                try:
                    counter_key = f"frames_processed_{component_name}"
                    results_summary["components"][component_name][counter_key] = (
                        results_summary["components"][component_name].get(
                            counter_key, 0
                        )
                        + 1
                    )

                    frame_passed = component(packet)

                    if not frame_passed:
                        gated_key = f"frames_gated_out_{component_name}"
                        results_summary["components"][component_name][gated_key] = (
                            results_summary["components"][component_name].get(
                                gated_key, 0
                            )
                            + 1
                        )
                        current_frame_is_valid = False
                        self.logger.info(
                            f"Frame {packet.frame_number} gated out by {component_name} for session {session_id}"
                        )
                        break  # Move to next packet
                except Exception as e:
                    error_msg = f"Error processing frame {packet.frame_number} with {component_name} in session {session_id}: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    results_summary["errors"].append(error_msg)
                    current_frame_is_valid = False
                    break  # Move to next packet

            if current_frame_is_valid:
                passed_gate_frames.append(packet)

        self.logger.info(
            f"{len(passed_gate_frames)} frames passed all gates for session: {session_id}"
        )
        return passed_gate_frames, gate_passage_metrics

    def _apply_sampler_if_configured(
        self, frames_to_sample: List[VideoFramePacket], session_id: str
    ):
        """Applies sampler if configured, otherwise returns original frames."""
        if self.sampler and frames_to_sample:  # Ensure there are frames to sample
            self.logger.info(
                f"Applying sampler to {len(frames_to_sample)} frames for session: {session_id}"
            )
            sampled_frames = self.sampler.sample(frames_to_sample, session_id)
            self.logger.info(
                f"Sampler selected {len(sampled_frames)} frames for session: {session_id}"
            )
            return sampled_frames
        elif not frames_to_sample:
            self.logger.info(f"No frames to sample for session: {session_id}")
            return []
        else:
            self.logger.info(
                f"No sampler configured. Passing through {len(frames_to_sample)} frames for session: {session_id}"
            )
            return frames_to_sample

    def _apply_buffer_if_configured(
        self,
        frames_to_buffer: List[VideoFramePacket],
        all_gate_metrics: Dict[int, float],
        session_id: str,
    ):
        """Applies buffer if configured, using provided metrics."""
        if self.buffer and frames_to_buffer:  # Ensure there are frames to buffer
            self.logger.info(
                f"Buffering {len(frames_to_buffer)} frames for session: {session_id}"
            )
            for packet in frames_to_buffer:
                frame_metric = all_gate_metrics.get(
                    packet.frame_number, float(packet.frame_number)
                )  # Fallback to frame_number
                self.logger.debug(
                    f"Adding frame {packet.frame_number} to buffer with metric {frame_metric} for session {session_id}"
                )
                self.buffer.add(packet, frame_metric=frame_metric)

            buffered_frames = self.buffer.flush()
            self.logger.info(
                f"Retrieved {len(buffered_frames)} frames from buffer for session: {session_id}"
            )
            return buffered_frames
        elif not frames_to_buffer:
            self.logger.info(f"No frames to buffer for session: {session_id}")
            return []
        else:
            self.logger.info(
                f"No buffer configured or no frames to buffer. Passing through {len(frames_to_buffer)} frames for session: {session_id}"
            )
            return frames_to_buffer
