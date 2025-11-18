import argparse
import logging

from decimatr.orchestrator import Orchestrator
from decimatr.utils import write_packets_to_video

# No need to directly import gate classes, samplers, or buffers now


def main():
    parser = argparse.ArgumentParser(description="Decimatr video processing PoC.")
    parser.add_argument(
        "--video-file",
        type=str,
        default="sample_data/dynamic.mp4",
        help="Path to the video file to process.",
    )
    # Add more arguments for configuration as needed
    parser.add_argument(
        "--blur-threshold", type=float, default=50.0, help="Blur detection threshold."
    )
    parser.add_argument(
        "--sample_k",
        type=int,
        default=10,
        help="Number of frames to sample if UniformSampler is used.",
    )

    args = parser.parse_args()

    # Define gates configuration
    gates_config = {
        "blur_gate": {
            "enabled": True,
            "type": "BlurGate",
            "settings": {"threshold": args.blur_threshold},
        },
        "hash_gate": {
            "enabled": True,
            "type": "HashGate",
            "settings": {"threshold": 0.05},
        },
        "entropy_gate": {
            "enabled": True,
            "type": "EntropyGate",
            "settings": {"threshold": 4.0},
        },
    }

    # Define sampler configuration
    sampler_config = {
        "enabled": True,
        "type": "UniformSampler",
        "settings": {"num_frames": args.sample_k},
    }

    # Define buffer configuration
    buffer_config = {
        "enabled": True,
        "type": "SlidingBuffer",
        "settings": {"window_size": 50, "step_size": 1},
    }

    # Define pipeline order
    pipeline_order = ["blur_gate", "hash_gate", "entropy_gate"]

    # Instantiate Orchestrator with configurations
    orchestrator = Orchestrator(
        gates_config=gates_config,
        # vra_config=vra_config,
        sampler_config=sampler_config,
        buffer_config=buffer_config,
        pipeline_order=pipeline_order,
        head_frames_to_keep=1,
        tail_frames_to_keep=1,
        sampler_gate_order="gates_first",
    )

    print(f"Processing video: {args.video_file}")

    # Call process_video with video path and session ID
    packets, results = orchestrator.process_video(
        video_path=args.video_file, session_id="example_session_main_py"
    )

    write_packets_to_video(packets, "output.mp4", logger=logging.getLogger(__name__))

    # Print results
    print("\n--- Processing Results ---")
    import json

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
