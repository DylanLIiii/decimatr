# Decimatr: High-Performance Video Frame Processing Library

[![Tests](https://github.com/DylanLIiii/decimatr/actions/workflows/test.yml/badge.svg)](https://github.com/DylanLIiii/decimatr/actions/workflows/test.yml)
[![Code Quality](https://github.com/DylanLIiii/decimatr/actions/workflows/quality.yml/badge.svg)](https://github.com/DylanLIiii/decimatr/actions/workflows/quality.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/decimatr.svg)](https://badge.fury.io/py/decimatr)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Decimatr is a modern, actor-based video frame processing library that provides a clean separation between frame analysis (tagging) and decision-making (filtering). Built on the xoscar Actor Model, it enables efficient distributed processing across CPU cores with optional GPU acceleration.

## Features

- **🎯 Simple API**: Process videos with just a few lines of code
- **⚡ High Performance**: Actor-based parallel processing across CPU cores
- **🧠 Smart Filtering**: Predefined strategies for common use cases (blur removal, duplicate detection, diversity sampling)
- **🔧 Extensible**: Easy to create custom taggers and filters
- **💾 Memory Efficient**: Lazy evaluation and automatic memory release
- **📊 Comprehensive Metrics**: Detailed performance tracking and monitoring
- **🎮 Optional GPU**: GPU acceleration for compute-intensive operations (CLIP embeddings)

## Installation

### CPU-Only (Default)

```bash
pip install decimatr
```

This installs all dependencies needed for CPU-based processing, including:
- Frame analysis (blur, hash, entropy detection)
- Distributed processing across CPU cores
- All filtering capabilities

### With GPU Support (Optional)

```bash
pip install decimatr[gpu]
```

This adds GPU dependencies for:
- CLIP embeddings
- GPU-accelerated batch processing

**Note**: GPU support requires CUDA-capable hardware and drivers.

## Quick Start

### Basic Usage

```python
from decimatr.core.processor import FrameProcessor

# Create processor with blur removal
processor = FrameProcessor.with_blur_removal(threshold=100.0)

# Process video
for frame in processor.process('video.mp4'):
    # Do something with sharp frames
    save_frame(frame)
```

### Parallel Processing

```python
# Use 4 CPU cores for faster processing
processor = FrameProcessor.with_blur_removal(
    threshold=100.0,
    n_workers=4
)

for frame in processor.process('video.mp4'):
    save_frame(frame)
```

### With Performance Metrics

```python
# Get detailed performance metrics
processor = FrameProcessor.with_blur_removal(threshold=100.0)
frames, result = processor.process('video.mp4', return_result=True)

for frame in frames:
    save_frame(frame)

# Print summary
result.print_summary()
# Output:
# ============================================================
# Processing Session: abc123
# ============================================================
# Frames: 250/1000 selected (25.0%)
# Filtered: 750 frames
# Processing Time: 12.50s
# Throughput: 80.0 fps
# Errors: 0
```

## Core Concepts

### Taggers (Stateless Analysis)

Taggers analyze individual frames and compute metadata tags without maintaining state:

```python
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger
from decimatr.taggers.entropy import EntropyTagger

# Taggers compute metrics for each frame
blur_tagger = BlurTagger()        # Computes blur score
hash_tagger = HashTagger()        # Computes perceptual hash
entropy_tagger = EntropyTagger()  # Computes Shannon entropy
```

**Available Taggers:**
- `BlurTagger`: Laplacian variance for blur detection
- `HashTagger`: Perceptual hashing for duplicate detection
- `EntropyTagger`: Shannon entropy for information content
- `CLIPTagger`: CLIP embeddings (requires GPU dependencies)

### Filters (Decision Making)

Filters make pass/fail decisions based on frame tags:

#### Stateless Filters

Make decisions based only on current frame:

```python
from decimatr.filters.blur import BlurFilter
from decimatr.filters.entropy import EntropyFilter
from decimatr.filters.threshold import ThresholdFilter

# Filter frames below threshold
blur_filter = BlurFilter(threshold=100.0)
entropy_filter = EntropyFilter(threshold=4.0)

# Generic threshold filter
threshold_filter = ThresholdFilter(
    tag_key='blur_score',
    threshold=100.0,
    operator='>'
)
```

#### Stateful Filters

Make decisions based on temporal context:

```python
from decimatr.filters.duplicate import DuplicateFilter
from decimatr.filters.motion import MotionFilter
from decimatr.filters.diversity import DiversityFilter

# Detect duplicates within sliding window
duplicate_filter = DuplicateFilter(
    threshold=0.05,
    buffer_size=50
)

# Detect scene changes
motion_filter = MotionFilter(
    threshold=0.3,
    buffer_size=10
)

# Maximize diversity
diversity_filter = DiversityFilter(
    window_size=100,
    min_distance=0.1
)
```

### Strategies (Predefined Pipelines)

Strategies combine taggers and filters for common use cases:

```python
# Blur removal
processor = FrameProcessor.with_blur_removal(threshold=100.0)

# Duplicate detection
processor = FrameProcessor.with_duplicate_detection(
    threshold=0.05,
    window_size=50
)

# Smart sampling (blur + duplicates + diversity)
processor = FrameProcessor.with_smart_sampling()
```

## Custom Pipelines

Create custom processing pipelines by combining taggers and filters:

```python
from decimatr.core.processor import FrameProcessor
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger
from decimatr.filters.blur import BlurFilter
from decimatr.filters.duplicate import DuplicateFilter

# Define custom pipeline
pipeline = [
    # Taggers first: compute metrics
    BlurTagger(),
    HashTagger(),
    # Filters second: make decisions
    BlurFilter(threshold=100.0),
    DuplicateFilter(threshold=0.05, buffer_size=50)
]

# Create processor
processor = FrameProcessor(pipeline=pipeline, n_workers=4)

# Process video
for frame in processor.process('video.mp4'):
    process_frame(frame)
```

## Performance Optimizations

### Lazy Evaluation

Decimatr automatically skips computing tags that aren't used by any filter:

```python
# Only BlurTagger will execute (HashTagger is unused)
pipeline = [
    BlurTagger(),
    HashTagger(),  # Not used by any filter - automatically skipped!
    BlurFilter(threshold=100.0)
]

processor = FrameProcessor(
    pipeline=pipeline,
    lazy_evaluation=True  # Default
)
```

**Performance Impact**: Up to 8x faster when taggers produce unused tags.

### Memory Release

Automatically frees frame data from filtered frames:

```python
processor = FrameProcessor(
    pipeline=pipeline,
    release_memory=True  # Default
)
```

**Memory Savings**: Up to 70% reduction in peak memory usage.

### Parallel Processing

Distribute processing across CPU cores:

```python
processor = FrameProcessor(
    pipeline=pipeline,
    n_workers=4  # Use 4 CPU cores
)
```

**Scaling Guidelines:**
- `n_workers=1`: Single-threaded (default)
- `n_workers=4`: Good for most workloads
- `n_workers=CPU_COUNT`: Maximum parallelism

## Creating Custom Components

### Custom Tagger

```python
from decimatr.taggers.base import Tagger
from decimatr.scheme import VideoFramePacket
import cv2

class CustomTagger(Tagger):
    """Compute custom metric for frames."""
    
    def compute_tags(self, packet: VideoFramePacket) -> dict:
        # Analyze frame
        gray = cv2.cvtColor(packet.frame_data, cv2.COLOR_BGR2GRAY)
        metric = compute_custom_metric(gray)
        
        return {"custom_metric": metric}
    
    @property
    def tag_keys(self) -> list:
        return ["custom_metric"]
```

### Custom Filter

```python
from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket

class CustomFilter(StatelessFilter):
    """Filter frames based on custom metric."""
    
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def should_pass(self, packet: VideoFramePacket) -> bool:
        metric = packet.get_tag("custom_metric")
        return metric > self.threshold
    
    @property
    def required_tags(self) -> list:
        return ["custom_metric"]
```

### Custom Strategy

```python
from decimatr.strategies.base import FilterStrategy

class CustomStrategy(FilterStrategy):
    """Custom processing strategy."""
    
    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold
    
    def build_pipeline(self):
        return [
            CustomTagger(),
            CustomFilter(threshold=self.threshold)
        ]

# Use custom strategy
processor = FrameProcessor(strategy=CustomStrategy(threshold=150.0))
```

## GPU Acceleration (Optional)

### Checking GPU Availability

```python
from decimatr.core.processor import FrameProcessor

# Check if GPU is available
if FrameProcessor.check_gpu_available():
    print("GPU acceleration available")
    info = FrameProcessor.get_gpu_info()
    print(f"CUDA version: {info['cuda_version']}")
    print(f"Device: {info['device_name']}")
else:
    print("GPU not available - using CPU")
```

### Using GPU Taggers

```python
from decimatr.taggers.clip import CLIPTagger

# Create GPU-accelerated tagger
clip_tagger = CLIPTagger(
    model_name="ViT-B/32",
    device="cuda"  # or "auto" to auto-detect
)

# Use in pipeline
pipeline = [
    clip_tagger,
    # ... filters ...
]

processor = FrameProcessor(
    pipeline=pipeline,
    use_gpu=True,
    gpu_batch_size=32
)
```

### GPU Installation

If GPU dependencies are missing:

```bash
# Install GPU support
pip install decimatr[gpu]

# Or install manually
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## API Reference

### FrameProcessor

Main API for processing video frames.

```python
FrameProcessor(
    pipeline: Optional[List[Union[Tagger, Filter]]] = None,
    strategy: Optional[FilterStrategy] = None,
    n_workers: int = 1,
    use_gpu: bool = False,
    gpu_batch_size: int = 32,
    lazy_evaluation: bool = True,
    release_memory: bool = True
)
```

**Parameters:**
- `pipeline`: Custom pipeline of taggers and filters
- `strategy`: Predefined strategy (overrides pipeline)
- `n_workers`: Number of worker actors (1=single-threaded)
- `use_gpu`: Enable GPU acceleration
- `gpu_batch_size`: Batch size for GPU processing
- `lazy_evaluation`: Compute tags only when needed
- `release_memory`: Free frame data from filtered frames

**Methods:**
- `process(source, session_id=None, return_result=False)`: Process frames
- `with_blur_removal(threshold, **kwargs)`: Create with blur removal strategy
- `with_duplicate_detection(threshold, window_size, **kwargs)`: Create with duplicate detection
- `with_smart_sampling(**kwargs)`: Create with smart sampling strategy
- `check_gpu_available()`: Check if GPU is available (static)
- `get_gpu_info()`: Get GPU information (static)

### ProcessingResult

Summary of processing session.

```python
result = ProcessingResult(...)
```

**Attributes:**
- `session_id`: Session identifier
- `total_frames`: Total frames processed
- `selected_frames`: Frames that passed filters
- `filtered_frames`: Frames filtered out
- `processing_time`: Total time in seconds
- `stage_metrics`: Per-stage metrics
- `actor_metrics`: Actor-level metrics
- `errors`: List of errors

**Methods:**
- `get_throughput()`: Get frames per second
- `get_selection_rate()`: Get selection percentage
- `get_summary()`: Get complete metrics dictionary
- `print_summary()`: Print detailed summary

## Examples

See the `examples/` directory for complete working examples:

- `actor_pipeline_demo.py`: Parallel processing with actors
- `frame_processor_demo.py`: Basic frame processing
- `performance_optimizations_demo.py`: Performance optimization techniques

## Documentation

- [API Documentation](docs/API.md) - Complete API reference
- [Parallel Processing Guide](docs/PARALLEL_PROCESSING.md) - Actor-based processing
- [Performance Optimizations](docs/PERFORMANCE_OPTIMIZATIONS.md) - Optimization techniques
- [GPU Setup Guide](docs/GPU_SETUP.md) - GPU installation and configuration
- [Custom Components Guide](docs/CUSTOM_COMPONENTS.md) - Creating custom taggers and filters

## Architecture

Decimatr uses a three-layer architecture:

1. **Tagging Layer**: Stateless frame analysis (blur, hash, entropy, etc.)
2. **Filtering Layer**: Decision-making (stateless and stateful)
3. **Orchestration Layer**: Actor-based distributed processing (xoscar)

```
┌─────────────────────────────────────────────────────────────────┐
│                        FrameProcessor                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Pipeline Configuration                        │  │
│  │  [Tagger1] → [Tagger2] → [Filter1] → [Filter2] → ...     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           Actor Pool (xoscar)                             │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │  │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker N │  ...          │  │
│  │  │ (CPU)    │  │ (CPU)    │  │ (GPU)    │               │  │
│  │  └──────────┘  └──────────┘  └──────────┘               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         Frame Stream (Input/Output)                       │  │
│  │  VideoFile → FrameIterator → ProcessedFrames             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.10+
- NumPy >= 2.2.5
- OpenCV >= 4.11.0
- imagehash >= 4.3.2
- xoscar >= 0.3.0
- decord >= 0.6.0
- loguru >= 0.7.3

**Optional (GPU):**
- torch >= 2.0.0
- torchvision >= 0.15.0

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/DylanLIiii/decimatr/issues)
- **Documentation**: [GitHub Repository](https://github.com/DylanLIiii/decimatr)
