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

Compute CLIP embeddings using OpenCLIP library with GPU/CPU support.

Uses standard CLIP models on GPU and MobileCLIP models on CPU for efficient processing.

```python
from decimatr.taggers.clip import CLIPTagger
```

#### Constructor

```python
CLIPTagger(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "auto",
    batch_size: int = 32
)
```

**Parameters:**

- **model_name** (`str`, default="ViT-B-32"): CLIP model architecture. Common values:
  - GPU: `'ViT-B-32'`, `'ViT-L-14'`, `'ViT-H-14'`
  - CPU: Automatically uses `'MobileCLIP-S0'`, `'MobileCLIP-S1'`, or `'MobileCLIP-S2'`

- **pretrained** (`str`, default="openai"): Pretrained weights. Common values:
  - `'openai'` - Original OpenAI weights
  - `'laion2b_s34b_b79k'` - LAION-2B trained weights
  - `'datacompdr'` - DataComp trained weights (for MobileCLIP)

- **device** (`str`, default="auto"): Device for computation:
  - `'auto'` - Automatically selects CUDA if available, otherwise CPU
  - `'cuda'` - Force GPU (uses standard CLIP models)
  - `'cpu'` - Force CPU (automatically uses MobileCLIP for efficiency)

- **batch_size** (`int`, default=32): Batch size for processing multiple frames:
  - GPU: 32-64 recommended for optimal throughput
  - CPU: 8-16 recommended for MobileCLIP

**Raises:**
- `GPUDependencyError`: If GPU requested but dependencies are missing

**Model Selection:**

The tagger automatically selects the appropriate model based on device:

- **GPU mode** (`device='cuda'`): Uses standard CLIP models (ViT-B-32, ViT-L-14, etc.)
  - Higher quality embeddings
  - Faster with GPU acceleration
  - Requires: `pip install decimatr[gpu]`

- **CPU mode** (`device='cpu'`): Automatically uses MobileCLIP models
  - Optimized for CPU inference
  - Smaller model size and faster CPU processing
  - Still produces high-quality embeddings
  - Requires: `pip install decimatr[gpu]` (includes open-clip-torch)

#### Tags Produced

- **clip_embedding** (`np.ndarray`): CLIP embedding vector (typically 512 or 768 dimensions)

#### Methods

##### compute_tags()

```python
def compute_tags(self, packet: VideoFramePacket) -> Dict[str, Any]
```

Compute CLIP embedding for a single frame.

**Parameters:**
- **packet** (`VideoFramePacket`): Frame packet

**Returns:** Dict with `'clip_embedding'` key containing numpy array

##### compute_tags_batch()

```python
def compute_tags_batch(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]
```

Batch compute CLIP embeddings (efficient for both GPU and CPU).

**Parameters:**
- **frames** (`List[np.ndarray]`): List of frame arrays

**Returns:** List of dicts with `'clip_embedding'` keys

**Note:** Batch processing significantly improves throughput, especially on GPU.

#### Properties

- `supports_gpu`: `True`
- `requires_gpu`: `False` (can run on CPU with MobileCLIP)
- `tag_keys`: `['clip_embedding']`

#### Examples

**GPU mode (standard CLIP):**

```python
# Requires: pip install decimatr[gpu]
tagger = CLIPTagger(
    model_name='ViT-B-32',
    pretrained='openai',
    device='cuda',
    batch_size=32
)

# Single frame
tags = tagger.compute_tags(packet)
# tags = {"clip_embedding": array([...], shape=(512,))}

# Batch processing (recommended for GPU)
frames = [frame1, frame2, frame3, ...]
batch_tags = tagger.compute_tags_batch(frames)
```

**CPU mode (MobileCLIP):**

```python
# Automatically uses MobileCLIP for efficient CPU processing
tagger = CLIPTagger(
    device='cpu',
    batch_size=8
)

tags = tagger.compute_tags(packet)
# Uses MobileCLIP under the hood
```

**Auto mode (recommended):**

```python
# Automatically selects GPU if available, otherwise CPU
tagger = CLIPTagger(device='auto', batch_size=32)
```

**Different CLIP models:**

```python
# Larger model for better quality (GPU)
tagger = CLIPTagger(
    model_name='ViT-L-14',
    pretrained='openai',
    device='cuda'
)

# LAION-trained model (GPU)
tagger = CLIPTagger(
    model_name='ViT-B-32',
    pretrained='laion2b_s34b_b79k',
    device='cuda'
)
```

**Performance:**
- GPU (ViT-B-32, batch=32): ~100-200 frames/second
- CPU (MobileCLIP, batch=8): ~20-30 frames/second
- Single frame (no batching): ~5-10 frames/second

**See Also:**
- [GPU Setup Guide](GPU_SETUP.md) - Installing GPU dependencies
- [Diversity Filter Examples](DIVERSITY_FILTER_EXAMPLES.md) - Using CLIP for diversity

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

Select frames maximizing tag diversity using tag classification and custom comparison strategies.

The enhanced DiversityFilter distinguishes between diversity-suitable tags (hashes, embeddings, histograms) and metric-only tags (blur_score, entropy). It supports different comparison strategies for different tag types and can combine multiple diversity metrics.

```python
from decimatr.filters.diversity import DiversityFilter
```

#### Constructor

```python
DiversityFilter(
    buffer_size: int = 100,
    diversity_tags: Optional[List[str]] = None,
    min_distance: float = 0.1,
    metric: str = "euclidean",
    comparison_strategies: Optional[Dict[str, ComparisonStrategy]] = None,
    enable_weighted_combination: bool = False,
    tag_weights: Optional[Dict[str, float]] = None
)
```

**Parameters:**

- **buffer_size** (`int`, default=100): Maximum number of diverse frames to maintain in temporal buffer. Larger values allow more diversity but use more memory.

- **diversity_tags** (`Optional[List[str]]`, default=None): List of tag keys to use for diversity calculation. If `None`, auto-detects diversity-suitable tags from available frame tags. Recommended tags: `['dhash', 'clip_embedding', 'color_hist']`.

- **min_distance** (`float`, default=0.1): Minimum distance threshold for a frame to pass. Frame must be at least this distance from all frames in buffer to be considered diverse enough. Range depends on metric and tag scales.

- **metric** (`str`, default='euclidean'): Distance metric for backward compatibility. Valid values: `'euclidean'`, `'manhattan'`, `'cosine'`. Used as default for embedding tags when no strategy specified.

- **comparison_strategies** (`Optional[Dict[str, ComparisonStrategy]]`, default=None): Custom comparison strategies per tag. Maps tag keys to `ComparisonStrategy` instances. If not specified, uses sensible defaults:
  - Hash tags (`dhash`, `phash`, `ahash`): `HammingDistanceStrategy`
  - Embedding tags (`clip_embedding`, `model_embedding`): `EmbeddingDistanceStrategy`
  - Histogram tags (`color_hist`): `HistogramDistanceStrategy`

- **enable_weighted_combination** (`bool`, default=False): Enable weighted combination of tag distances. If `False` (default), uses maximum distance across tags. If `True`, combines normalized distances using weights.

- **tag_weights** (`Optional[Dict[str, float]]`, default=None): Weights for each tag when weighted combination is enabled. If not specified, uses equal weights for all tags. Weights are automatically normalized to sum to 1.0.

**Raises:**

- `ValueError`: If `metric` is invalid, `min_distance` is negative, or comparison strategies are invalid

**Tag Classification:**

The filter automatically classifies tags into two categories:

- **Diversity-Suitable Tags** (included in diversity calculations):
  - Perceptual hashes: `dhash`, `phash`, `ahash`
  - Embeddings: `clip_embedding`, `model_embedding`
  - Color histograms: `color_hist`, `color_histogram`

- **Metric-Only Tags** (excluded from diversity calculations):
  - `blur_score` - measures blur, not visual diversity
  - `entropy` - measures information content, not visual diversity
  - `edge_density` - measures edge content, not visual diversity

**Comparison Strategies:**

Available comparison strategies from `decimatr.filters.comparison_strategies`:

- **HammingDistanceStrategy**: For perceptual hashes (dhash, phash, ahash)
  - Computes normalized Hamming distance in range [0, 1]
  - Recommended threshold: 0.05-0.15

- **EmbeddingDistanceStrategy(metric)**: For embeddings (CLIP, custom models)
  - Supports `'euclidean'`, `'cosine'`, `'manhattan'` metrics
  - Cosine distance range: [0, 2], recommended threshold: 0.1-0.3
  - Euclidean distance range: [0, ~10+], recommended threshold: 1.0-5.0

- **HistogramDistanceStrategy(metric)**: For color histograms
  - Supports `'intersection'`, `'chi_square'`, `'bhattacharyya'` metrics
  - Intersection range: [0, 1], recommended threshold: 0.2-0.4

#### Required Tags

Returns configured `diversity_tags` if specified, otherwise empty list (auto-detection at runtime).

#### Methods

##### compare_with_history()

```python
def compare_with_history(
    packet: VideoFramePacket,
    history: List[VideoFramePacket]
) -> bool
```

Compare current frame against historical frames for diversity.

**Parameters:**
- **packet**: Current frame to evaluate
- **history**: List of recent frames from temporal buffer

**Returns:** `True` if frame is diverse enough (passes), `False` otherwise

**Behavior:**
1. Gets diversity tags for current frame (explicit or auto-detected)
2. Validates that specified tags are present in frame
3. Computes minimum distance to all history frames using appropriate strategies
4. Applies threshold to determine pass/fail

#### Examples

**Basic usage with auto-detection:**

```python
from decimatr.taggers.hash import HashTagger
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    HashTagger(hash_type='dhash'),
    DiversityFilter(
        buffer_size=100,
        min_distance=0.1
    )
]
# Automatically detects and uses 'dhash' tag
```

**Explicit tag specification:**

```python
filter = DiversityFilter(
    buffer_size=100,
    diversity_tags=['dhash', 'clip_embedding'],
    min_distance=0.1
)
```

**Custom comparison strategies:**

```python
from decimatr.filters.comparison_strategies import (
    HammingDistanceStrategy,
    EmbeddingDistanceStrategy
)

filter = DiversityFilter(
    buffer_size=100,
    diversity_tags=['dhash', 'clip_embedding'],
    min_distance=0.1,
    comparison_strategies={
        'dhash': HammingDistanceStrategy(),
        'clip_embedding': EmbeddingDistanceStrategy(metric='cosine')
    }
)
```

**Weighted combination mode:**

```python
filter = DiversityFilter(
    buffer_size=100,
    diversity_tags=['dhash', 'clip_embedding'],
    min_distance=0.1,
    enable_weighted_combination=True,
    tag_weights={
        'dhash': 0.3,           # 30% weight on visual hash
        'clip_embedding': 0.7   # 70% weight on semantic similarity
    }
)
```

**Complete pipeline with CLIP:**

```python
from decimatr.taggers.clip import CLIPTagger
from decimatr.filters.diversity import DiversityFilter
from decimatr.filters.comparison_strategies import EmbeddingDistanceStrategy

pipeline = [
    CLIPTagger(device='cuda', batch_size=32),
    DiversityFilter(
        buffer_size=100,
        diversity_tags=['clip_embedding'],
        min_distance=0.2,
        comparison_strategies={
            'clip_embedding': EmbeddingDistanceStrategy(metric='cosine')
        }
    )
]
```

**See Also:**
- [Diversity Filter Examples](DIVERSITY_FILTER_EXAMPLES.md) - Comprehensive usage examples
- [Migration Guide](DIVERSITY_FILTER_MIGRATION.md) - Upgrading from old DiversityFilter
- [Custom Components](CUSTOM_COMPONENTS.md) - Creating custom comparison strategies

---

## Comparison Strategies

Comparison strategies define how distances are computed between tag values for diversity filtering.

### ComparisonStrategy (Base Class)

Abstract base class for tag comparison strategies.

```python
from decimatr.filters.comparison_strategies import ComparisonStrategy
```

#### Abstract Methods

##### compute_distance()

```python
@abstractmethod
def compute_distance(self, value1: Any, value2: Any) -> float
```

Compute distance between two tag values.

**Parameters:**
- **value1**: First tag value
- **value2**: Second tag value

**Returns:** Distance value (higher = more different)

##### name

```python
@property
@abstractmethod
def name(self) -> str
```

Return strategy name for logging and debugging.

---

### HammingDistanceStrategy

Compute normalized Hamming distance between perceptual hashes.

```python
from decimatr.filters.comparison_strategies import HammingDistanceStrategy
```

#### Constructor

```python
HammingDistanceStrategy()
```

No parameters required.

#### Behavior

- Computes Hamming distance (number of differing bits) between hash values
- Normalizes by hash bit length to produce values in range [0, 1]
- Supports both string (hex) and ImageHash object formats
- 0.0 = identical hashes, 1.0 = completely different hashes

#### Supported Tags

- `dhash` (difference hash)
- `phash` (perceptual hash)
- `ahash` (average hash)

#### Recommended Thresholds

- 0.05-0.10: Very similar frames only
- 0.10-0.15: Moderately similar frames (recommended)
- 0.15-0.25: Quite different frames

#### Example

```python
from decimatr.filters.comparison_strategies import HammingDistanceStrategy

strategy = HammingDistanceStrategy()
filter = DiversityFilter(
    diversity_tags=['dhash'],
    min_distance=0.1,
    comparison_strategies={'dhash': strategy}
)
```

---

### EmbeddingDistanceStrategy

Compute distance between embedding vectors (CLIP, custom models).

```python
from decimatr.filters.comparison_strategies import EmbeddingDistanceStrategy
```

#### Constructor

```python
EmbeddingDistanceStrategy(metric: str = "cosine")
```

**Parameters:**
- **metric** (`str`, default='cosine'): Distance metric. Valid values: `'euclidean'`, `'cosine'`, `'manhattan'`

**Raises:**
- `ValueError`: If metric is not one of the valid values

#### Behavior

- **Euclidean**: L2 distance, measures absolute distance in embedding space
- **Cosine**: 1 - cosine similarity, measures angular distance (direction-based)
- **Manhattan**: L1 distance, sum of absolute differences

- Handles vectors of different lengths by padding with zeros
- Normalizes vectors before computing cosine distance
- Handles zero vectors gracefully (returns maximum distance)

#### Supported Tags

- `clip_embedding` (CLIP embeddings)
- `model_embedding` (custom model embeddings)
- Any vector-based tag

#### Recommended Thresholds

**Cosine distance** (recommended for semantic similarity):
- 0.1-0.2: Very similar content
- 0.2-0.3: Moderately similar content (recommended)
- 0.3-0.5: Quite different content

**Euclidean distance**:
- 1.0-2.0: Very similar content
- 2.0-5.0: Moderately similar content (recommended)
- 5.0-10.0: Quite different content

**Manhattan distance**:
- 3.0-5.0: Very similar content
- 5.0-10.0: Moderately similar content (recommended)
- 10.0-20.0: Quite different content

#### Examples

**Cosine distance (recommended for CLIP):**

```python
from decimatr.filters.comparison_strategies import EmbeddingDistanceStrategy

strategy = EmbeddingDistanceStrategy(metric='cosine')
filter = DiversityFilter(
    diversity_tags=['clip_embedding'],
    min_distance=0.2,
    comparison_strategies={'clip_embedding': strategy}
)
```

**Euclidean distance:**

```python
strategy = EmbeddingDistanceStrategy(metric='euclidean')
filter = DiversityFilter(
    diversity_tags=['clip_embedding'],
    min_distance=2.0,
    comparison_strategies={'clip_embedding': strategy}
)
```

---

### HistogramDistanceStrategy

Compute distance between color histograms.

```python
from decimatr.filters.comparison_strategies import HistogramDistanceStrategy
```

#### Constructor

```python
HistogramDistanceStrategy(metric: str = "intersection")
```

**Parameters:**
- **metric** (`str`, default='intersection'): Distance metric. Valid values: `'intersection'`, `'chi_square'`, `'bhattacharyya'`

**Raises:**
- `ValueError`: If metric is not one of the valid values

#### Behavior

- **Intersection**: 1 - histogram intersection, measures overlap (0 = identical, 1 = no overlap)
- **Chi-square**: Chi-square distance, statistical measure of difference
- **Bhattacharyya**: -log(Bhattacharyya coefficient), probabilistic measure

- Normalizes histograms before comparison (sums to 1.0)
- Handles histograms of different lengths by padding with zeros
- Adds small epsilon (1e-10) to avoid division by zero

#### Supported Tags

- `color_hist` (color histogram)
- `color_histogram` (alternative name)
- Any histogram-based tag

#### Recommended Thresholds

**Intersection** (recommended):
- 0.2-0.3: Similar color distributions
- 0.3-0.4: Moderately different colors (recommended)
- 0.4-0.6: Quite different colors

**Chi-square**:
- 0.3-0.5: Similar distributions
- 0.5-1.0: Moderately different (recommended)
- 1.0-2.0: Quite different

**Bhattacharyya**:
- 0.3-0.5: Similar distributions
- 0.5-1.0: Moderately different (recommended)
- 1.0-2.0: Quite different

#### Examples

**Histogram intersection (recommended):**

```python
from decimatr.filters.comparison_strategies import HistogramDistanceStrategy

strategy = HistogramDistanceStrategy(metric='intersection')
filter = DiversityFilter(
    diversity_tags=['color_hist'],
    min_distance=0.3,
    comparison_strategies={'color_hist': strategy}
)
```

**Chi-square distance:**

```python
strategy = HistogramDistanceStrategy(metric='chi_square')
filter = DiversityFilter(
    diversity_tags=['color_hist'],
    min_distance=0.5,
    comparison_strategies={'color_hist': strategy}
)
```

---

## Tag Classification

The tag classification system distinguishes between diversity-suitable and metric-only tags.

### TagClassificationRegistry

Registry for classifying tags as diversity-suitable or metric-only.

```python
from decimatr.filters.tag_classification import TagClassificationRegistry
```

#### Class Attributes

##### DIVERSITY_SUITABLE_TAGS

```python
DIVERSITY_SUITABLE_TAGS: Set[str] = {
    "dhash", "phash", "ahash",  # Perceptual hashes
    "clip_embedding", "model_embedding",  # Embeddings
    "color_hist", "color_histogram"  # Color histograms
}
```

Tags suitable for diversity analysis.

##### METRIC_ONLY_TAGS

```python
METRIC_ONLY_TAGS: Set[str] = {
    "blur_score", "entropy", "edge_density"
}
```

Tags representing quality metrics, not suitable for diversity.

#### Class Methods

##### is_diversity_suitable()

```python
@classmethod
def is_diversity_suitable(cls, tag_key: str) -> bool
```

Check if a tag is suitable for diversity analysis.

**Parameters:**
- **tag_key** (`str`): Tag key to check

**Returns:** `True` if tag is diversity-suitable, `False` otherwise

**Note:** Unknown tags default to diversity-suitable.

##### is_metric_only()

```python
@classmethod
def is_metric_only(cls, tag_key: str) -> bool
```

Check if a tag is metric-only (not suitable for diversity).

**Parameters:**
- **tag_key** (`str`): Tag key to check

**Returns:** `True` if tag is metric-only, `False` otherwise

##### get_category()

```python
@classmethod
def get_category(cls, tag_key: str) -> TagCategory
```

Get the category of a tag.

**Parameters:**
- **tag_key** (`str`): Tag key to classify

**Returns:** `TagCategory.DIVERSITY_SUITABLE` or `TagCategory.METRIC_ONLY`

**Note:** Unknown tags default to `TagCategory.DIVERSITY_SUITABLE`.

#### Example

```python
from decimatr.filters.tag_classification import TagClassificationRegistry

# Check if tags are diversity-suitable
is_suitable = TagClassificationRegistry.is_diversity_suitable('dhash')  # True
is_suitable = TagClassificationRegistry.is_diversity_suitable('blur_score')  # False

# Check if tags are metric-only
is_metric = TagClassificationRegistry.is_metric_only('blur_score')  # True
is_metric = TagClassificationRegistry.is_metric_only('dhash')  # False

# Get category
from decimatr.filters.tag_classification import TagCategory
category = TagClassificationRegistry.get_category('clip_embedding')
# Returns TagCategory.DIVERSITY_SUITABLE
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
