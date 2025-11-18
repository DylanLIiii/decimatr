# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Decimatr** is a high-performance video frame processing library using the xoscar Actor Model for distributed parallel processing. It provides a clean separation between frame analysis (tagging) and decision-making (filtering) with optional GPU acceleration.

### Key Features
- Actor-based parallel processing across CPU cores (xoscar framework)
- Stateless frame analysis (taggers) + decision-making (filters)
- Predefined strategies for common use cases (blur removal, duplicate detection, diversity sampling)
- Lazy evaluation and memory release optimizations
- Optional GPU acceleration for compute-intensive operations (CLIP embeddings)

## Common Development Commands

### Setup & Dependencies

```bash
# Install CPU-only dependencies
pip install -e .

# Install GPU dependencies (requires CUDA-capable hardware)
pip install -e ".[gpu]"

# Or using the cpu-only group
pip install -e ".[cpu-only]"

# Install with uv (if using uv as package manager)
uv pip install -e ".[cpu-only]"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_processor_actor_integration.py

# Run tests with verbose output
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=decimatr

# Run specific test pattern
pytest tests/ -k "test_blur"
```

**Note**: Tests require `imagehash` package. Install with `pip install imagehash` if not already installed.

### Examples

```bash
# Run actor pipeline demo
python examples/actor_pipeline_demo.py

# Run frame processor demo
python examples/frame_processor_demo.py

# Run performance optimizations demo
python examples/performance_optimizations_demo.py
```

### Project Structure

```
decimatr/
├── actors/              # xoscar-based actor implementations
│   ├── filter_actor.py  # Actor for filter operations
│   ├── tagger_actor.py  # Actor for tagger operations
│   ├── stateful_actor.py # Base for stateful operations
│   └── pipeline.py      # ActorPipeline orchestration
├── core/                # Core processing engine
│   ├── processor.py     # FrameProcessor API
│   └── temporal_buffer.py # Temporal buffering for stateful filters
├── filters/             # Filter implementations
│   ├── base.py          # Filter base classes
│   ├── blur.py          # Blur-based filtering
│   ├── duplicate.py     # Duplicate detection
│   ├── diversity.py     # Diversity sampling
│   ├── entropy.py       # Entropy-based filtering
│   ├── motion.py        # Motion/scene change detection
│   └── threshold.py     # Generic threshold filtering
├── taggers/             # Tagger implementations
│   ├── base.py          # Tagger base class
│   ├── blur.py          # Blur score computation
│   ├── clip.py          # CLIP embeddings (GPU)
│   ├── entropy.py       # Entropy computation
│   └── hash.py          # Perceptual hashing
├── strategies/          # Predefined processing pipelines
│   ├── base.py          # Strategy base class
│   ├── blur_removal.py  # Blur removal strategy
│   ├── duplicate_detection.py # Duplicate detection
│   └── smart_sampling.py # Multi-criteria sampling
└── scheme.py            # Data models (VideoFramePacket, etc.)
```

## Architecture

### Three-Layer Design

1. **Tagging Layer** (`decimatr/taggers/`): Stateless frame analysis
   - Analyze individual frames
   - Compute metrics: blur scores, entropy, perceptual hashes, CLIP embeddings
   - No state maintained between frames

2. **Filtering Layer** (`decimatr/filters/`): Decision-making components
   - **StatelessFilter**: Decisions based only on current frame's tags
   - **StatefulFilter**: Decisions based on temporal context (buffer-based)
   - Make pass/fail decisions for each frame

3. **Orchestration Layer** (`decimatr/core/processor.py`, `decimatr/actors/`): Actor-based distributed processing
   - Single-threaded: `n_workers=1`
   - Actor-based parallel: `n_workers>1`
   - Uses xoscar Actor Model for distributed execution
   - Automatic port allocation (20000-30000) to prevent conflicts

### Key Components

**FrameProcessor** (`decimatr/core/processor.py:27`)
- Main API for frame processing
- Routes between single-threaded and actor-based modes based on `n_workers`
- Implements lazy evaluation and memory release optimizations
- Returns `ProcessingResult` with metrics and statistics

**ActorPipeline** (`decimatr/actors/pipeline.py`)
- Manages actor lifecycle for parallel processing
- Creates unique ports per processor instance
- Handles actor pool creation and cleanup
- Distributes frame processing across worker actors

**VideoFramePacket** (`decimatr/scheme.py`)
- Data model for frame data + metadata + tags
- Contains: frame_data, frame_number, timestamp, tags dict
- Passed through entire processing pipeline

## Usage Patterns

### Quick Start
```python
from decimatr.core.processor import FrameProcessor

# Blur removal with 4-way parallelism
processor = FrameProcessor.with_blur_removal(
    threshold=100.0,
    n_workers=4
)

# Process video
for frame in processor.process('video.mp4'):
    save_frame(frame)
```

### Custom Pipeline
```python
from decimatr.core.processor import FrameProcessor
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger
from decimatr.filters.blur import BlurFilter
from decimatr.filters.duplicate import DuplicateFilter

pipeline = [
    BlurTagger(),
    HashTagger(),
    BlurFilter(threshold=100.0),
    DuplicateFilter(threshold=0.05, buffer_size=50)
]

processor = FrameProcessor(pipeline=pipeline, n_workers=4)
```

### With GPU Acceleration
```python
from decimatr.taggers.clip import CLIPTagger
from decimatr.core.processor import FrameProcessor

# Check GPU availability
if FrameProcessor.check_gpu_available():
    clip_tagger = CLIPTagger(model_name="ViT-B/32", device="cuda")

    processor = FrameProcessor(
        pipeline=[clip_tagger, /* filters */],
        use_gpu=True,
        gpu_batch_size=32
    )
```

## Development Notes

### Performance Features

**Lazy Evaluation** (`lazy_evaluation=True`)
- Only computes tags that are used by filters
- Can provide up to 8x speedup when taggers produce unused tags
- Enabled by default in FrameProcessor

**Memory Release** (`release_memory=True`)
- Frees frame_data from filtered frames
- Up to 70% reduction in peak memory usage
- Enabled by default in FrameProcessor

**Parallel Processing**
- `n_workers=1`: Single-threaded (no actor overhead)
- `n_workers>1`: Actor-based distributed processing
- Uses xoscar for true parallel execution across CPU cores

### Testing
- Tests use synthetic frame data generated in `conftest.py`
- Main test files:
  - `test_processor_actor_integration.py`: Actor integration tests
  - `test_performance_optimizations.py`: Performance feature tests
  - `tests/filters/`: Individual filter tests
  - `tests/taggers/`: Individual tagger tests

### Dependencies
- **Required**: numpy, opencv-python, xoscar, loguru
- **Optional GPU**: torch, torchvision, open-clip-torch (install with `.[gpu]`)
- **Testing**: pytest, imagehash (install with `.[cpu-only]`)

### Known Issues / Work in Progress

1. **Missing Module**: `decimatr.gates.image_hash.ImageHasher` is referenced but doesn't exist
   - Files affected: `decimatr/filters/duplicate.py`, `decimatr/taggers/hash.py`
   - Should use `imagehash` package directly or implement wrapper
   - Workaround: Install `imagehash` package and replace imports

2. **Import Errors**: Tests may fail if dependencies not properly installed
   - Ensure `pip install imagehash` before running tests
   - Consider adding to main dependencies in `pyproject.toml`

### Configuration
- Python >= 3.10 required
- Uses `pyproject.toml` for configuration
- uv.lock present (using uv as package manager)
- .gitignore configured for Python development

### Recent Changes
- **Actor Integration**: FrameProcessor now supports both single-threaded and actor-based processing
- **Metrics**: Added comprehensive ProcessingResult with throughput, actor health, and error tracking
- **GPU Support**: CLIP tagger for GPU-accelerated embeddings

### Documentation
- `README.md`: Main documentation and quick start
- `docs/API.md`: Complete API reference
- `docs/PARALLEL_PROCESSING.md`: Actor-based processing guide
- `docs/PERFORMANCE_OPTIMIZATIONS.md`: Performance tuning guide
- `docs/GPU_SETUP.md`: GPU installation and configuration
- `docs/CUSTOM_COMPONENTS.md`: Creating custom taggers and filters
- `ACTOR_INTEGRATION_SUMMARY.md`: Notes on actor integration implementation

## Tips for Development

1. **Adding New Tagger**: Extend `decimatr.taggers.base.Tagger`
   - Implement `compute_tags()` method
   - Define `tag_keys` property
   - Optional: implement `compute_tags_batch()` for GPU acceleration

2. **Adding New Filter**: Extend `decimatr.filters.base.StatelessFilter` or `StatefulFilter`
   - Implement `should_pass()` method
   - Define `required_tags` property
   - Use `TemporalBuffer` for stateful operations

3. **Adding New Strategy**: Extend `decimatr.strategies.base.FilterStrategy`
   - Implement `build_pipeline()` method
   - Return list of taggers and filters

4. **GPU Development**:
   - Check `FrameProcessor.check_gpu_available()` before using GPU features
   - Use `FrameProcessor.get_gpu_info()` for device details
   - Batch processing: set appropriate `gpu_batch_size`

5. **Debugging**:
   - Enable logging: `import logging; logging.basicConfig(level=logging.DEBUG)`
   - Use `ProcessingResult` metrics for performance analysis
   - Actor metrics include stage-level timing and actor health stats
