# Decimatr API Reference

Complete API reference for the Decimatr video frame processing library.

## Table of Contents

- [Core API](#core-api)
  - [FrameProcessor](#frameprocessor)
  - [ProcessingResult](#processingresult)
- [Taggers](#taggers)
  - [Tagger (Base Class)](#tagger-base-class)
  - [BlurTagger](#blurtagger)
  - [HashTagger](#hashtagger)
  - [EntropyTagger](#entropytagger)
  - [CLIPTagger](#cliptagger)
- [Filters](#filters)
  - [Filter (Base Class)](#filter-base-class)
  - [StatelessFilter](#statelessfilter)
  - [StatefulFilter](#statefulfilter)
  - [ThresholdFilter](#thresholdfilter)
  - [BlurFilter](#blurfilter)
  - [EntropyFilter](#entropyfilter)
  - [DuplicateFilter](#duplicatefilter)
  - [MotionFilter](#motionfilter)
  - [DiversityFilter](#diversityfilter)
- [Strategies](#strategies)
  - [FilterStrategy (Base Class)](#filterstrategy-base-class)
  - [BlurRemovalStrategy](#blurremovalstrategy)
  - [DuplicateDetectionStrategy](#duplicatedetectionstrategy)
  - [SmartSamplingStrategy](#smartsamplingstrategy)
- [Data Models](#data-models)
  - [VideoFramePacket](#videoframepacket)
- [Utilities](#utilities)
  - [TemporalBuffer](#temporalbuffer)
  - [GPUCapabilities](#gpucapabilities)
- [Exceptions](#exceptions)

---

## Core API

### FrameProcessor

Main API for processing video frames through tagging and filtering pipelines.

```python
from decimatr.core.processor import FrameProcessor
```

#### Constructor

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

- **pipeline** (`Optional[List[Union[Tagger, Filter]]]`): Custom pipeline of taggers and filters. If `None` and no strategy provided, uses pass-through (no filtering).

- **strategy** (`Optional[FilterStrategy]`): Predefined FilterStrategy. If provided, overrides `pipeline`.

- **n_workers** (`int`, default=1): Number of worker threads/actors for parallel processing. Values > 1 enable actor-based distributed processing.

- **use_gpu** (`bool`, default=False): Enable GPU acceleration for supported taggers. Requires GPU dependencies to be installed.

- **gpu_batch_size** (`int`, default=32): Batch size for GPU processing. Larger values improve throughput but use more GPU memory.

- **lazy_evaluation** (`bool`, default=True): Enable lazy tag computation (compute only when required by filters).

- **release_memory** (`bool`, default=True): Release frame_data from memory after filtering out frames.

**Raises:**

- `ConfigurationError`: If pipeline configuration is invalid
- `ValueError`: If `n_workers` or `gpu_batch_size` are invalid

**Example:**

```python
# Custom pipeline
pipeline = [BlurTagger(), BlurFilter(threshold=100.0)]
processor = FrameProcessor(pipeline=pipeline)

# Using strategy
from decimatr.strategies.blur_removal import BlurRemovalStrategy
strategy = BlurRemovalStrategy(threshold=100.0)
processor = FrameProcessor(strategy=strategy)

# Parallel processing with optimizations
processor = FrameProcessor(
    strategy=strategy,
    n_workers=4,
    lazy_evaluation=True,
    release_memory=True
)
```

#### Methods

##### process()

```python
process(
    source: Union[str, Iterator[VideoFramePacket], List[VideoFramePacket]],
    session_id: Optional[str] = None,
    return_result: bool = False
) -> Union[Iterator[VideoFramePacket], Tuple[Iterator[VideoFramePacket], ProcessingResult]]
```

Process video frames through the pipeline.

**Parameters:**

- **source**: Input source, one of:
  - `str`: Path to video file (uses `load_video_frames`)
  - `Iterator[VideoFramePacket]`: Frame iterator
  - `List[VideoFramePacket]`: List of frames

- **session_id** (`Optional[str]`): Optional session identifier for logging and metrics. If `None`, a session ID is generated automatically.

- **return_result** (`bool`, default=False): If `True`, returns tuple of `(iterator, ProcessingResult)`. If `False`, returns only the iterator.

**Returns:**

- If `return_result=False`: `Iterator[VideoFramePacket]`
- If `return_result=True`: `Tuple[Iterator[VideoFramePacket], ProcessingResult]`

**Example:**

```python
# Process video file
processor = FrameProcessor.with_blur_removal()
for frame in processor.process('video.mp4'):
    save_frame(frame)

# Process with result summary
frames, result = processor.process('video.mp4', return_result=True)
for frame in frames:
    save_frame(frame)
print(f"Selected {result.selected_frames} frames")
```

##### with_blur_removal() (class method)

```python
@classmethod
with_blur_removal(
    cls,
    threshold: float = 100.0,
    **kwargs
) -> FrameProcessor
```

Create processor with blur removal strategy.

**Parameters:**

- **threshold** (`float`, default=100.0): Minimum blur score for frames to pass. Higher values are more restrictive.
- **kwargs**: Additional arguments passed to FrameProcessor constructor (`n_workers`, `use_gpu`, etc.)

**Returns:** `FrameProcessor` configured with BlurRemovalStrategy

**Example:**

```python
processor = FrameProcessor.with_blur_removal(threshold=150.0, n_workers=4)
```

##### with_duplicate_detection() (class method)

```python
@classmethod
with_duplicate_detection(
    cls,
    threshold: float = 0.05,
    window_size: int = 50,
    **kwargs
) -> FrameProcessor
```

Create processor with duplicate detection strategy.

**Parameters:**

- **threshold** (`float`, default=0.05): Hash similarity threshold (0.0-1.0). Lower values are stricter.
- **window_size** (`int`, default=50): Number of recent frames to compare against.
- **kwargs**: Additional arguments passed to FrameProcessor constructor

**Returns:** `FrameProcessor` configured with DuplicateDetectionStrategy

**Example:**

```python
processor = FrameProcessor.with_duplicate_detection(
    threshold=0.02,
    window_size=100,
    n_workers=4
)
```

##### with_smart_sampling() (class method)

```python
@classmethod
with_smart_sampling(cls, **kwargs) -> FrameProcessor
```

Create processor with smart sampling strategy.

Combines blur removal, duplicate detection, and diversity sampling for comprehensive frame selection.

**Parameters:**

- **kwargs**: Arguments for SmartSamplingStrategy or FrameProcessor constructor

**Returns:** `FrameProcessor` configured with SmartSamplingStrategy

**Example:**

```python
processor = FrameProcessor.with_smart_sampling(n_workers=4)
```

##### check_gpu_available() (static method)

```python
@staticmethod
check_gpu_available() -> bool
```

Check if GPU acceleration is available.

**Returns:** `True` if GPU dependencies are installed and CUDA is available, `False` otherwise

**Example:**

```python
if FrameProcessor.check_gpu_available():
    print("GPU acceleration available")
```

##### get_gpu_info() (static method)

```python
@staticmethod
get_gpu_info() -> Dict[str, Any]
```

Get detailed GPU information.

**Returns:** Dictionary containing:
- `gpu_available` (bool): Whether GPU is available
- `missing_dependencies` (List[str]): List of missing GPU dependencies
- `cuda_version` (str): CUDA version (if available)
- `device_count` (int): Number of GPU devices (if available)
- `device_name` (str): GPU device name (if available)

**Example:**

```python
info = FrameProcessor.get_gpu_info()
print(f"GPU available: {info['gpu_available']}")
if info['gpu_available']:
    print(f"Device: {info['device_name']}")
```

---

### ProcessingResult

Summary of a frame processing session.

```python
from decimatr.core.processor import ProcessingResult
```

#### Attributes

- **session_id** (`str`): Unique identifier for the processing session
- **total_frames** (`int`): Total number of frames processed
- **processed_frames** (`int`): Number of frames that completed processing
- **filtered_frames** (`int`): Number of frames filtered out by filters
- **selected_frames** (`int`): Number of frames that passed all filters
- **processing_time** (`float`): Total processing time in seconds
- **stage_metrics** (`Dict[str, Dict[str, Any]]`): Dictionary of per-stage metrics
- **actor_metrics** (`Dict[str, Dict[str, Any]]`): Dictionary of actor-level metrics
- **errors** (`List[str]`): List of error messages encountered
- **lazy_evaluation_enabled** (`bool`): Whether lazy evaluation was used
- **memory_release_enabled** (`bool`): Whether memory release was enabled

#### Methods

##### get_throughput()

```python
get_throughput() -> float
```

Get processing throughput in frames per second.

**Returns:** Number of frames processed per second

##### get_selection_rate()

```python
get_selection_rate() -> float
```

Get the frame selection rate as a percentage.

**Returns:** Percentage of frames that passed all filters (0.0-100.0)

##### get_summary()

```python
get_summary() -> Dict[str, Any]
```

Get a comprehensive summary dictionary of all metrics.

**Returns:** Dictionary containing all metrics in a serializable format

##### print_summary()

```python
print_summary() -> None
```

Print a detailed summary of processing results to stdout.

**Example:**

```python
frames, result = processor.process('video.mp4', return_result=True)
for frame in frames:
    process_frame(frame)

# Print detailed summary
result.print_summary()

# Access specific metrics
print(f"Throughput: {result.get_throughput():.1f} fps")
print(f"Selection rate: {result.get_selection_rate():.1f}%")
```

---

## Taggers

### Tagger (Base Class)

Abstract base class for all taggers.

```python
from decimatr.taggers.base import Tagger
```

#### Abstract Methods

##### compute_tags()

```python
@abstractmethod
def compute_tags(self, packet: VideoFramePacket) -> Dict[str, Any]
```

Compute tags for a single frame.

**Parameters:**
- **packet** (`VideoFramePacket`): Frame packet containing frame data and metadata

**Returns:** Dictionary mapping tag keys to computed values

**Example:**

```python
class CustomTagger(Tagger):
    def compute_tags(self, packet):
        # Analyze frame
        metric = analyze_frame(packet.frame_data)
        return {"custom_metric": metric}
```

##### tag_keys (property)

```python
@property
@abstractmethod
def tag_keys(self) -> List[str]
```

Return list of tag keys this tagger produces.

**Returns:** List of tag key strings

#### Optional Methods

##### compute_tags_batch()

```python
def compute_tags_batch(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]
```

Batch compute tags for multiple frames (for GPU processing).

**Parameters:**
- **frames** (`List[np.ndarray]`): List of frame data arrays

**Returns:** List of tag dictionaries, one per input frame

#### Properties

##### supports_gpu

```python
@property
def supports_gpu(self) -> bool
```

Whether this tagger supports GPU acceleration. Default: `False`

##### requires_gpu

```python
@property
def requires_gpu(self) -> bool
```

Whether this tagger requires GPU to function. Default: `False`

##### is_cloud_based

```python
@property
def is_cloud_based(self) -> bool
```

Whether this tagger uses cloud-based models via async client. Default: `False`

---

### BlurTagger

Compute blur score using Laplacian variance.

```python
from decimatr.taggers.blur import BlurTagger
```

#### Constructor

```python
BlurTagger()
```

No parameters required.

#### Tags Produced

- **blur_score** (`float`): Laplacian variance. Higher values indicate sharper images.

#### Example

```python
tagger = BlurTagger()
tags = tagger.compute_tags(packet)
# tags = {"blur_score": 123.45}
```

---

### HashTagger

Compute perceptual hash for duplicate detection.

```python
from decimatr.taggers.hash import HashTagger
```

#### Constructor

```python
HashTagger(hash_type: str = 'phash', hash_size: int = 8)
```

**Parameters:**
- **hash_type** (`str`, default='phash'): Hash algorithm ('phash', 'ahash', 'dhash', 'whash')
- **hash_size** (`int`, default=8): Hash size in bits

#### Tags Produced

- **phash** (`str`): Hexadecimal hash string
- **hash_value** (`ImageHash`): ImageHash object for comparison

#### Example

```python
tagger = HashTagger(hash_type='phash', hash_size=8)
tags = tagger.compute_tags(packet)
# tags = {"phash": "a1b2c3d4...", "hash_value": <ImageHash>}
```

---

### EntropyTagger

Compute Shannon entropy for information content.

```python
from decimatr.taggers.entropy import EntropyTagger
```

#### Constructor

```python
EntropyTagger()
```

No parameters required.

#### Tags Produced

- **entropy** (`float`): Shannon entropy value. Higher values indicate more information/complexity.

#### Example

```python
tagger = EntropyTagger()
tags = tagger.compute_tags(packet)
# tags = {"entropy": 6.234}
```

---

### CLIPTagger

Compute CLIP embeddings using GPU (requires GPU dependencies).

```python
from decimatr.taggers.clip import CLIPTagger
```

#### Constructor

```python
CLIPTagger(model_name: str = "ViT-B/32", device: str = "auto")
```

**Parameters:**
- **model_name** (`str`, default="ViT-B/32"): CLIP model variant
- **device** (`str`, default="auto"): Device ('auto', 'cuda', or 'cpu')

**Raises:**
- `GPUDependencyError`: If GPU requested but dependencies are missing

#### Tags Produced

- **clip_embedding** (`np.ndarray`): CLIP embedding vector

#### Properties

- `supports_gpu`: `True`
- `requires_gpu`: `False` (can fall back to CPU)

#### Example

```python
# Requires: pip install decimatr[gpu]
tagger = CLIPTagger(model_name="ViT-B/32", device="cuda")
tags = tagger.compute_tags(packet)
# tags = {"clip_embedding": array([...])}
```

---

## Filters

### Filter (Base Class)

Abstract base class for all filters.

```python
from decimatr.filters.base import Filter
```

#### Abstract Methods

##### should_pass()

```python
@abstractmethod
def should_pass(self, packet: VideoFramePacket) -> bool
```

Determine if frame should pass through the filter.

**Parameters:**
- **packet** (`VideoFramePacket`): Frame packet with tags

**Returns:** `True` if frame passes, `False` if filtered out

##### required_tags (property)

```python
@property
@abstractmethod
def required_tags(self) -> List[str]
```

Return list of tag keys required by this filter.

**Returns:** List of required tag key strings

---

### StatelessFilter

Base class for filters that make decisions based only on current frame tags.

```python
from decimatr.filters.base import StatelessFilter
```

Stateless filters evaluate each frame independently without temporal context.

**Example:**

```python
class CustomFilter(StatelessFilter):
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def should_pass(self, packet):
        return packet.get_tag("metric") > self.threshold
    
    @property
    def required_tags(self):
        return ["metric"]
```

---

### StatefulFilter

Base class for filters that maintain temporal context for decision-making.

```python
from decimatr.filters.base import StatefulFilter
```

#### Constructor

```python
StatefulFilter(buffer_size: int)
```

**Parameters:**
- **buffer_size** (`int`): Maximum number of frames to maintain in buffer

**Raises:**
- `ValueError`: If `buffer_size` is not positive

#### Abstract Methods

##### compare_with_history()

```python
@abstractmethod
def compare_with_history(
    self,
    packet: VideoFramePacket,
    history: List[VideoFramePacket]
) -> bool
```

Compare current frame with historical frames to make pass/fail decision.

**Parameters:**
- **packet** (`VideoFramePacket`): Current frame to evaluate
- **history** (`List[VideoFramePacket]`): List of recent frames from buffer

**Returns:** `True` if frame should pass and be added to buffer, `False` otherwise

#### Methods

##### add_to_buffer()

```python
add_to_buffer(packet: VideoFramePacket) -> None
```

Add frame to the temporal buffer.

##### get_buffer_contents()

```python
get_buffer_contents() -> List[VideoFramePacket]
```

Get current buffer contents as a list.

##### clear_buffer()

```python
clear_buffer() -> None
```

Clear all frames from the temporal buffer.

##### buffer_count()

```python
buffer_count() -> int
```

Get the current number of frames in the buffer.

##### is_buffer_full()

```python
is_buffer_full() -> bool
```

Check if the buffer has reached capacity.

**Example:**

```python
class CustomStatefulFilter(StatefulFilter):
    def __init__(self, threshold: float, buffer_size: int):
        super().__init__(buffer_size)
        self.threshold = threshold
    
    def compare_with_history(self, packet, history):
        # Compare with history
        for past_frame in history:
            if is_similar(packet, past_frame, self.threshold):
                return False  # Duplicate found
        return True  # No duplicate
    
    @property
    def required_tags(self):
        return ["hash_value"]
```

---

### ThresholdFilter

Generic threshold-based filtering.

```python
from decimatr.filters.threshold import ThresholdFilter
```

#### Constructor

```python
ThresholdFilter(
    tag_key: str,
    threshold: float,
    operator: str = '>'
)
```

**Parameters:**
- **tag_key** (`str`): Tag key to evaluate
- **threshold** (`float`): Threshold value
- **operator** (`str`, default='>'): Comparison operator ('>', '<', '>=', '<=', '==', '!=')

**Raises:**
- `ValueError`: If operator is not supported

#### Example

```python
# Filter frames with blur_score > 100.0
filter = ThresholdFilter(
    tag_key='blur_score',
    threshold=100.0,
    operator='>'
)
```

---

### BlurFilter

Filter frames below a blur threshold.

```python
from decimatr.filters.blur import BlurFilter
```

#### Constructor

```python
BlurFilter(threshold: float = 100.0)
```

**Parameters:**
- **threshold** (`float`, default=100.0): Minimum blur score for frames to pass

#### Required Tags

- `blur_score`

#### Example

```python
filter = BlurFilter(threshold=150.0)
```

---

### EntropyFilter

Filter frames below an entropy threshold.

```python
from decimatr.filters.entropy import EntropyFilter
```

#### Constructor

```python
EntropyFilter(threshold: float = 4.0)
```

**Parameters:**
- **threshold** (`float`, default=4.0): Minimum entropy for frames to pass

#### Required Tags

- `entropy`

#### Example

```python
filter = EntropyFilter(threshold=5.0)
```

---

### DuplicateFilter

Detect and filter duplicate frames via hash comparison.

```python
from decimatr.filters.duplicate import DuplicateFilter
```

#### Constructor

```python
DuplicateFilter(
    threshold: float = 0.05,
    buffer_size: int = 50
)
```

**Parameters:**
- **threshold** (`float`, default=0.05): Hash similarity threshold (0.0-1.0). Lower values are stricter.
- **buffer_size** (`int`, default=50): Number of recent frames to compare against

#### Required Tags

- `hash_value`

#### Example

```python
filter = DuplicateFilter(threshold=0.02, buffer_size=100)
```

---

### MotionFilter

Detect scene changes via frame differencing.

```python
from decimatr.filters.motion import MotionFilter
```

#### Constructor

```python
MotionFilter(
    threshold: float = 0.3,
    buffer_size: int = 10
)
```

**Parameters:**
- **threshold** (`float`, default=0.3): Motion threshold (0.0-1.0)
- **buffer_size** (`int`, default=10): Number of recent frames to compare against

#### Required Tags

- None (uses frame_data directly)

#### Example

```python
filter = MotionFilter(threshold=0.4, buffer_size=5)
```

---

### DiversityFilter

Select frames maximizing tag diversity.

```python
from decimatr.filters.diversity import DiversityFilter
```

#### Constructor

```python
DiversityFilter(
    window_size: int = 100,
    min_distance: float = 0.1,
    diversity_metric: str = 'euclidean'
)
```

**Parameters:**
- **window_size** (`int`, default=100): Size of diversity window
- **min_distance** (`float`, default=0.1): Minimum distance for diversity
- **diversity_metric** (`str`, default='euclidean'): Distance metric

#### Required Tags

- Depends on diversity metric (typically `clip_embedding` or similar)

#### Example

```python
filter = DiversityFilter(
    window_size=200,
    min_distance=0.15,
    diversity_metric='cosine'
)
```

---

## Strategies

### FilterStrategy (Base Class)

Abstract base class for filter strategies.

```python
from decimatr.strategies.base import FilterStrategy
```

#### Abstract Methods

##### build_pipeline()

```python
@abstractmethod
def build_pipeline(self) -> List[Union[Tagger, Filter]]
```

Build the complete processing pipeline.

**Returns:** Ordered list of Tagger and Filter instances

**Example:**

```python
class CustomStrategy(FilterStrategy):
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def build_pipeline(self):
        return [
            BlurTagger(),
            BlurFilter(threshold=self.threshold)
        ]
```

---

### BlurRemovalStrategy

Strategy for filtering out blurry frames.

```python
from decimatr.strategies.blur_removal import BlurRemovalStrategy
```

#### Constructor

```python
BlurRemovalStrategy(threshold: float = 100.0)
```

**Parameters:**
- **threshold** (`float`, default=100.0): Minimum blur score

#### Pipeline

- `BlurTagger()`
- `BlurFilter(threshold)`

#### Example

```python
strategy = BlurRemovalStrategy(threshold=150.0)
processor = FrameProcessor(strategy=strategy)
```

---

### DuplicateDetectionStrategy

Strategy for detecting and removing duplicate frames.

```python
from decimatr.strategies.duplicate_detection import DuplicateDetectionStrategy
```

#### Constructor

```python
DuplicateDetectionStrategy(
    threshold: float = 0.05,
    window_size: int = 50
)
```

**Parameters:**
- **threshold** (`float`, default=0.05): Hash similarity threshold
- **window_size** (`int`, default=50): Comparison window size

#### Pipeline

- `HashTagger()`
- `DuplicateFilter(threshold, window_size)`

#### Example

```python
strategy = DuplicateDetectionStrategy(threshold=0.02, window_size=100)
processor = FrameProcessor(strategy=strategy)
```

---

### SmartSamplingStrategy

Comprehensive strategy combining blur removal, duplicate detection, and diversity.

```python
from decimatr.strategies.smart_sampling import SmartSamplingStrategy
```

#### Constructor

```python
SmartSamplingStrategy(
    blur_threshold: float = 100.0,
    duplicate_threshold: float = 0.05,
    duplicate_window: int = 50,
    diversity_window: int = 100,
    diversity_min_distance: float = 0.1
)
```

**Parameters:**
- **blur_threshold** (`float`, default=100.0): Blur threshold
- **duplicate_threshold** (`float`, default=0.05): Duplicate threshold
- **duplicate_window** (`int`, default=50): Duplicate window size
- **diversity_window** (`int`, default=100): Diversity window size
- **diversity_min_distance** (`float`, default=0.1): Minimum diversity distance

#### Pipeline

- `BlurTagger()`
- `HashTagger()`
- `EntropyTagger()`
- `BlurFilter(blur_threshold)`
- `DuplicateFilter(duplicate_threshold, duplicate_window)`
- `DiversityFilter(diversity_window, diversity_min_distance)`

#### Example

```python
strategy = SmartSamplingStrategy(
    blur_threshold=150.0,
    duplicate_threshold=0.02
)
processor = FrameProcessor(strategy=strategy)
```

---

## Data Models

### VideoFramePacket

Standardized data structure for frame data and metadata.

```python
from decimatr.scheme import VideoFramePacket
```

#### Attributes

- **frame_data** (`np.ndarray`): Frame image data (H x W x C)
- **frame_number** (`int`): Frame index in video
- **timestamp** (`timedelta`): Frame timestamp
- **source_video_id** (`str`): Source video identifier
- **tags** (`Dict[str, Any]`): Tag registry (computed by taggers)
- **metadata** (`Dict[str, Any]`): Additional metadata

#### Methods

##### get_tag()

```python
get_tag(key: str, default: Any = None) -> Any
```

Get tag value with optional default.

##### has_tags()

```python
has_tags(keys: List[str]) -> bool
```

Check if all required tags are present.

##### copy_without_frame_data()

```python
copy_without_frame_data() -> VideoFramePacket
```

Create lightweight copy without frame data (for logging).

#### Example

```python
packet = VideoFramePacket(
    frame_data=frame,
    frame_number=42,
    timestamp=timedelta(seconds=1.4),
    source_video_id="video.mp4"
)

# Add tags
packet.tags["blur_score"] = 123.45

# Get tag
blur = packet.get_tag("blur_score", default=0.0)

# Check tags
if packet.has_tags(["blur_score", "entropy"]):
    process(packet)
```

---

## Utilities

### TemporalBuffer

Efficient sliding window for stateful filters.

```python
from decimatr.core.temporal_buffer import TemporalBuffer
```

#### Constructor

```python
TemporalBuffer(max_size: int)
```

**Parameters:**
- **max_size** (`int`): Maximum buffer capacity

#### Methods

##### add()

```python
add(packet: VideoFramePacket) -> None
```

Add frame to buffer (O(1) operation).

##### get_window()

```python
get_window() -> List[VideoFramePacket]
```

Get all frames in buffer as a list.

##### find_similar()

```python
find_similar(
    packet: VideoFramePacket,
    similarity_fn: Callable
) -> Optional[VideoFramePacket]
```

Find similar frame in buffer using custom similarity function.

#### Example

```python
buffer = TemporalBuffer(max_size=50)

# Add frames
buffer.add(packet1)
buffer.add(packet2)

# Get window
frames = buffer.get_window()

# Find similar
similar = buffer.find_similar(
    packet,
    lambda p1, p2: hash_distance(p1, p2) < 0.05
)
```

---

### GPUCapabilities

Detect and report GPU capabilities.

```python
from decimatr.gpu_utils import GPUCapabilities
```

#### Class Methods

##### is_available()

```python
@classmethod
is_available(cls) -> bool
```

Check if GPU acceleration is available.

##### get_missing_dependencies()

```python
@classmethod
get_missing_dependencies(cls) -> List[str]
```

Return list of missing GPU dependencies.

##### get_info()

```python
@classmethod
get_info(cls) -> Dict[str, Any]
```

Get detailed GPU information.

#### Example

```python
if GPUCapabilities.is_available():
    info = GPUCapabilities.get_info()
    print(f"CUDA version: {info['cuda_version']}")
else:
    missing = GPUCapabilities.get_missing_dependencies()
    print(f"Missing: {missing}")
```

---

## Exceptions

### DecimatrError

Base exception for all Decimatr errors.

```python
from decimatr.exceptions import DecimatrError
```

### ConfigurationError

Raised when pipeline configuration is invalid.

```python
from decimatr.exceptions import ConfigurationError
```

### TagMissingError

Raised when required tag is missing.

```python
from decimatr.exceptions import TagMissingError
```

### ProcessingError

Raised when frame processing fails.

```python
from decimatr.exceptions import ProcessingError
```

### ActorError

Raised when actor operations fail.

```python
from decimatr.exceptions import ActorError
```

### GPUDependencyError

Raised when GPU acceleration is requested but GPU dependencies are missing.

```python
from decimatr.exceptions import GPUDependencyError
```

#### Example

```python
try:
    processor = FrameProcessor(use_gpu=True)
except GPUDependencyError as e:
    print(f"GPU not available: {e}")
    print("Install with: pip install decimatr[gpu]")
```

---

## See Also

- [README](../README.md) - Getting started guide
- [Parallel Processing](PARALLEL_PROCESSING.md) - Actor-based processing
- [Performance Optimizations](PERFORMANCE_OPTIMIZATIONS.md) - Optimization techniques
- [GPU Setup](GPU_SETUP.md) - GPU installation and configuration
- [Custom Components](CUSTOM_COMPONENTS.md) - Creating custom taggers and filters
