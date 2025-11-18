# Decimatr Quick Reference

Quick reference guide for common Decimatr operations.

## Installation

```bash
# CPU-only (default)
pip install decimatr

# With GPU support
pip install decimatr[gpu]
```

## Basic Usage

### Process Video with Predefined Strategy

```python
from decimatr.core.processor import FrameProcessor

# Blur removal
processor = FrameProcessor.with_blur_removal(threshold=100.0)
for frame in processor.process('video.mp4'):
    save_frame(frame)

# Duplicate detection
processor = FrameProcessor.with_duplicate_detection(threshold=0.05)
for frame in processor.process('video.mp4'):
    save_frame(frame)

# Smart sampling
processor = FrameProcessor.with_smart_sampling()
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

processor = FrameProcessor(pipeline=pipeline)
for frame in processor.process('video.mp4'):
    save_frame(frame)
```

### Parallel Processing

```python
# Use 4 CPU cores
processor = FrameProcessor.with_blur_removal(
    threshold=100.0,
    n_workers=4
)

for frame in processor.process('video.mp4'):
    save_frame(frame)
```

### Get Performance Metrics

```python
processor = FrameProcessor.with_blur_removal()
frames, result = processor.process('video.mp4', return_result=True)

for frame in frames:
    save_frame(frame)

# Print summary
result.print_summary()

# Access metrics
print(f"Throughput: {result.get_throughput():.1f} fps")
print(f"Selection rate: {result.get_selection_rate():.1f}%")
```

## Available Taggers

```python
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger
from decimatr.taggers.entropy import EntropyTagger
from decimatr.taggers.clip import CLIPTagger  # Requires GPU dependencies

# Blur detection
BlurTagger()

# Perceptual hashing
HashTagger(hash_type='phash', hash_size=8)

# Entropy calculation
EntropyTagger()

# CLIP embeddings (GPU)
CLIPTagger(model_name="ViT-B/32", device="cuda")
```

## Available Filters

### Stateless Filters

```python
from decimatr.filters.blur import BlurFilter
from decimatr.filters.entropy import EntropyFilter
from decimatr.filters.threshold import ThresholdFilter

# Blur filter
BlurFilter(threshold=100.0)

# Entropy filter
EntropyFilter(threshold=4.0)

# Generic threshold filter
ThresholdFilter(tag_key='blur_score', threshold=100.0, operator='>')
```

### Stateful Filters

```python
from decimatr.filters.duplicate import DuplicateFilter
from decimatr.filters.motion import MotionFilter
from decimatr.filters.diversity import DiversityFilter

# Duplicate detection
DuplicateFilter(threshold=0.05, buffer_size=50)

# Motion/scene change detection
MotionFilter(threshold=0.3, buffer_size=10)

# Diversity sampling
DiversityFilter(window_size=100, min_distance=0.1)
```

## Available Strategies

```python
from decimatr.strategies.blur_removal import BlurRemovalStrategy
from decimatr.strategies.duplicate_detection import DuplicateDetectionStrategy
from decimatr.strategies.smart_sampling import SmartSamplingStrategy

# Blur removal
BlurRemovalStrategy(threshold=100.0)

# Duplicate detection
DuplicateDetectionStrategy(threshold=0.05, window_size=50)

# Smart sampling (blur + duplicates + diversity)
SmartSamplingStrategy(
    blur_threshold=100.0,
    duplicate_threshold=0.05,
    duplicate_window=50,
    diversity_window=100,
    diversity_min_distance=0.1
)
```

## FrameProcessor Options

```python
FrameProcessor(
    pipeline=None,              # Custom pipeline
    strategy=None,              # Predefined strategy
    n_workers=1,                # Number of CPU workers (1=single-threaded)
    use_gpu=False,              # Enable GPU acceleration
    gpu_batch_size=32,          # GPU batch size
    lazy_evaluation=True,       # Compute tags only when needed
    release_memory=True         # Free memory from filtered frames
)
```

## Processing Options

```python
processor.process(
    source='video.mp4',         # Video file, iterator, or list
    session_id=None,            # Optional session ID
    return_result=False         # Return ProcessingResult
)
```

## GPU Operations

```python
from decimatr.core.processor import FrameProcessor
from decimatr.gpu_utils import GPUCapabilities

# Check GPU availability
if FrameProcessor.check_gpu_available():
    print("GPU available")

# Get GPU info
info = FrameProcessor.get_gpu_info()
print(f"CUDA version: {info['cuda_version']}")
print(f"Device: {info['device_name']}")

# Check missing dependencies
missing = GPUCapabilities.get_missing_dependencies()
if missing:
    print(f"Missing: {missing}")
```

## Custom Components

### Custom Tagger

```python
from decimatr.taggers.base import Tagger
from decimatr.scheme import VideoFramePacket

class MyTagger(Tagger):
    def compute_tags(self, packet: VideoFramePacket) -> dict:
        # Analyze frame
        metric = analyze(packet.frame_data)
        return {"my_metric": metric}
    
    @property
    def tag_keys(self) -> list:
        return ["my_metric"]
```

### Custom Stateless Filter

```python
from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket

class MyFilter(StatelessFilter):
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def should_pass(self, packet: VideoFramePacket) -> bool:
        return packet.get_tag("my_metric") > self.threshold
    
    @property
    def required_tags(self) -> list:
        return ["my_metric"]
```

### Custom Stateful Filter

```python
from decimatr.filters.base import StatefulFilter
from decimatr.scheme import VideoFramePacket

class MyStatefulFilter(StatefulFilter):
    def __init__(self, buffer_size: int, threshold: float):
        super().__init__(buffer_size)
        self.threshold = threshold
    
    def compare_with_history(self, packet, history) -> bool:
        if not history:
            return True
        
        # Compare with history
        for past_frame in history:
            if is_similar(packet, past_frame, self.threshold):
                return False
        return True
    
    @property
    def required_tags(self) -> list:
        return ["my_metric"]
```

### Custom Strategy

```python
from decimatr.strategies.base import FilterStrategy

class MyStrategy(FilterStrategy):
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def build_pipeline(self):
        return [
            MyTagger(),
            MyFilter(threshold=self.threshold)
        ]
```

## Common Patterns

### Process Multiple Videos

```python
processor = FrameProcessor.with_blur_removal()

for video_path in video_paths:
    for frame in processor.process(video_path):
        save_frame(frame)
```

### Process Frame Iterator

```python
def frame_generator():
    for i in range(100):
        yield create_frame(i)

processor = FrameProcessor.with_blur_removal()
for frame in processor.process(frame_generator()):
    save_frame(frame)
```

### Process Frame List

```python
frames = [frame1, frame2, frame3]

processor = FrameProcessor.with_blur_removal()
for frame in processor.process(frames):
    save_frame(frame)
```

### Access Frame Tags

```python
processor = FrameProcessor(pipeline=[BlurTagger(), BlurFilter(100.0)])

for frame in processor.process('video.mp4'):
    blur_score = frame.get_tag("blur_score")
    print(f"Frame {frame.frame_number}: blur={blur_score:.2f}")
```

### Combine Multiple Filters

```python
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.entropy import EntropyTagger
from decimatr.filters.blur import BlurFilter
from decimatr.filters.entropy import EntropyFilter

pipeline = [
    BlurTagger(),
    EntropyTagger(),
    BlurFilter(threshold=100.0),      # Remove blurry frames
    EntropyFilter(threshold=4.0)      # Remove low-entropy frames
]

processor = FrameProcessor(pipeline=pipeline)
```

### Optimize for Performance

```python
processor = FrameProcessor(
    pipeline=pipeline,
    n_workers=4,              # Parallel processing
    lazy_evaluation=True,     # Skip unused computations
    release_memory=True       # Free filtered frame data
)
```

## Error Handling

```python
from decimatr.exceptions import (
    ConfigurationError,
    GPUDependencyError,
    ProcessingError
)

try:
    processor = FrameProcessor(pipeline=invalid_pipeline)
except ConfigurationError as e:
    print(f"Invalid pipeline: {e}")

try:
    processor = FrameProcessor(use_gpu=True)
except GPUDependencyError as e:
    print(f"GPU not available: {e}")
    print("Install with: pip install decimatr[gpu]")

try:
    for frame in processor.process('video.mp4'):
        process_frame(frame)
except ProcessingError as e:
    print(f"Processing failed: {e}")
```

## Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure specific logger
logger = logging.getLogger('decimatr')
logger.setLevel(logging.INFO)
```

## Performance Tips

1. **Use parallel processing for large videos**:
   ```python
   processor = FrameProcessor(pipeline=pipeline, n_workers=4)
   ```

2. **Enable lazy evaluation** (default):
   ```python
   processor = FrameProcessor(pipeline=pipeline, lazy_evaluation=True)
   ```

3. **Enable memory release** (default):
   ```python
   processor = FrameProcessor(pipeline=pipeline, release_memory=True)
   ```

4. **Order filters by selectivity** (most restrictive first):
   ```python
   pipeline = [
       BlurTagger(),
       HashTagger(),
       BlurFilter(threshold=150.0),  # Filters ~50% - put first
       DuplicateFilter(threshold=0.05)  # Filters ~20% - put second
   ]
   ```

5. **Remove unused taggers**:
   ```python
   # Bad: HashTagger is never used
   pipeline = [BlurTagger(), HashTagger(), BlurFilter(100.0)]
   
   # Good: Only necessary taggers
   pipeline = [BlurTagger(), BlurFilter(100.0)]
   ```

6. **Use GPU for CLIP embeddings**:
   ```python
   processor = FrameProcessor(
       pipeline=[CLIPTagger(device="cuda")],
       use_gpu=True,
       gpu_batch_size=32
   )
   ```

## See Also

- [README](../README.md) - Getting started guide
- [API Reference](API.md) - Complete API documentation
- [Parallel Processing](PARALLEL_PROCESSING.md) - Actor-based processing
- [Performance Optimizations](PERFORMANCE_OPTIMIZATIONS.md) - Optimization techniques
- [GPU Setup](GPU_SETUP.md) - GPU installation and configuration
- [Custom Components](CUSTOM_COMPONENTS.md) - Creating custom taggers and filters
