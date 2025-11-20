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
# Install package in development mode
pip install -e ".[dev]"

# Install with GPU dependencies (requires CUDA-capable hardware)
pip install -e ".[gpu,dev]"

# Install with uv (recommended package manager)
uv pip install -e ".[dev]"

# Install GPU dependencies with uv
uv pip install -e ".[gpu,dev]"

# Setup pre-commit hooks (run after installation)
pre-commit install
```

### Code Quality & Formatting

```bash
# Lint code with ruff
ruff check decimatr tests

# Format code with ruff
ruff format decimatr tests

# Run both linting and formatting
ruff check --fix decimatr tests && ruff format decimatr tests

# Type checking with mypy
mypy decimatr --ignore-missing-imports

# Run pre-commit hooks on all files
pre-commit run --all-files
```

### Building

```bash
# Build wheel and source distribution
python -m build

# Clean build artifacts
rm -rf dist/ build/ *.egg-info/
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=decimatr --cov-report=term-missing

# Run tests in parallel
pytest tests/ -n auto

# Run specific test file
pytest tests/test_processor_actor_integration.py

# Run tests with verbose output
pytest tests/ -v

# Run specific test pattern
pytest tests/ -k "test_blur"

# Run tests for specific Python version
pytest tests/ --python-version 3.11
```

**Note**: Tests require `imagehash` package. Install with `pip install imagehash` or use the dev installation.

### Examples

```bash
# Run actor pipeline demo
python examples/actor_pipeline_demo.py

# Run frame processor demo
python examples/frame_processor_demo.py

# Run performance optimizations demo
python examples/performance_optimizations_demo.py

# Run GPU batch processing demo
python examples/gpu_batch_processing_demo.py
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
- Provides GPU batch processing via `use_gpu` parameter
- Static methods: `check_gpu_available()`, `get_gpu_info()`

**ActorPipeline** (`decimatr/actors/pipeline.py`)
- Manages actor lifecycle for parallel processing
- Creates unique ports per processor instance
- Handles actor pool creation and cleanup
- Distributes frame processing across worker actors
- Manages different actor types: TaggingActor, FilterActor, StatefulActor

**GPUBatchProcessor** (`decimatr/actors/gpu_actor.py`)
- Specialized actor for GPU-accelerated batch processing
- Accumulates frames for batch operations
- Falls back to CPU processing when GPU unavailable
- Integrates with CLIP and other GPU taggers

**VideoFramePacket** (`decimatr/scheme.py`)
- Data model for frame data + metadata + tags
- Contains: frame_data, frame_number, timestamp, tags dict
- Passed through entire processing pipeline
- Supports lazy tag evaluation

**Core Utilities**
- `gpu_utils.py`: GPU capability detection and batch processing utilities
- `metrics.py`: Comprehensive processing metrics and PerformanceResult
- `video_loader.py`: Frame loading and iteration utilities
- `utils.py`: Common utilities for image processing and frame manipulation
- `temporal_buffer.py`: Buffer for stateful filter operations
- `exceptions.py`: Custom exception hierarchy (DecimatrError, ActorError, etc.)

### Documentation Structure

Available documentation in `/home/heng.li/repo/decimatr/docs/`:
- **API.md**: Complete API reference with all classes and methods
- **PARALLEL_PROCESSING.md**: Actor-based processing architecture and usage
- **PERFORMANCE_OPTIMIZATIONS.md**: Performance tuning and optimization techniques
- **GPU_SETUP.md**: GPU installation, configuration, and troubleshooting
- **CUSTOM_COMPONENTS.md**: Creating custom taggers, filters, and strategies
- **QUICK_REFERENCE.md**: Quick reference guide for common tasks

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
- Automatically skipped taggers don't consume compute resources

**Memory Release** (`release_memory=True`)
- Frees frame_data from filtered frames
- Up to 70% reduction in peak memory usage
- Enabled by default in FrameProcessor
- Critical for processing long videos or high-resolution frames

**GPU Batch Processing** (`use_gpu=True`, `gpu_batch_size=32`)
- Accumulates frames for batch GPU operations
- Dramatic speedup for CLIP and other ML-based taggers
- Automatic CPU fallback when GPU unavailable
- Configurable batch size for memory vs. throughput tradeoffs

**Parallel Processing**
- `n_workers=1`: Single-threaded (default, no actor overhead)
- `n_workers>1`: Actor-based distributed processing using xoscar
- Uses xoscar for true parallel execution across CPU cores
- Automatic actor lifecycle management
- Port allocation: 20000-30000 range (automatic)

**Actor-Based Scaling**
- Separate actor types for different operations
- Efficient message passing between actors
- Actor health monitoring and metrics collection
- Stage-level timing and error tracking

### Testing
- Tests use synthetic frame data generated in `conftest.py`
- Main test files:
  - `test_processor_actor_integration.py`: Actor integration tests
  - `test_performance_optimizations.py`: Performance feature tests
  - `tests/filters/`: Individual filter tests
  - `tests/taggers/`: Individual tagger tests

### Dependencies
- **Core Required**: numpy>=2.2.5, opencv-python>=4.11.0, xoscar>=0.3.0, loguru>=0.7.3, decord>=0.6.0, imagehash>=4.3.2
- **Optional GPU**: torch>=2.0.0, torchvision>=0.15.0, open-clip-torch>=2.32.0 (install with `.[gpu]`)
  - Uses OpenCLIP library for CLIP embeddings with automatic GPU/CPU model selection
  - MobileCLIP models automatically selected for CPU inference
- **Development**: pytest>=8.3.5, pytest-cov>=6.0.0, ruff>=0.8.0, mypy>=1.0.0, pre-commit>=4.4.0
- **Build System**: hatchling (Python build backend)

**System Dependencies** (for video processing):
- ffmpeg, libsm6, libxext6 (Linux)
- Automatically installed in CI, may need manual installation locally

### Known Issues / Work in Progress

1. **Missing Module**: `decimatr.gates.image_hash.ImageHasher` is referenced but doesn't exist
   - Files affected: `decimatr/filters/duplicate.py`, `decimatr/taggers/hash.py`
   - Should use `imagehash` package directly or implement wrapper
   - Workaround: Install `imagehash` package and replace imports

2. **Import Errors**: Tests may fail if dependencies not properly installed
   - Ensure `pip install imagehash` before running tests
   - Consider adding to main dependencies in `pyproject.toml`

### Configuration
- Python >= 3.10 required (tested on 3.10, 3.11, 3.12)
- Uses `pyproject.toml` for configuration with hatchling build backend
- `uv.lock` present (uv is recommended package manager)
- `.pre-commit-config.yaml` configured for code quality
- `.gitignore` configured for Python development
- GitHub Actions for CI/CD (quality checks, tests, coverage)
- Codecov integration for coverage tracking

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
   - Check `decimatr.metrics` for detailed performance tracking

6. **Pre-commit Hooks**:
   - Run `pre-commit install` after installation
   - Automatically runs ruff check/format on git commits
   - Manual run: `pre-commit run --all-files`
   - Can skip hooks temporarily with `git commit --no-verify`

7. **Using uv** (recommended):
   - Faster dependency resolution: `uv pip install -e ".[dev]"`
   - Better workspace management: `uv sync`
   - Run scripts in isolated env: `uv run pytest tests/`
   - Lock file updates: `uv lock`
