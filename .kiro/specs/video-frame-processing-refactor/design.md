# Design Document: Video Frame Processing Library Refactor

## Overview

This document outlines the architectural design for refactoring Decimatr into a high-performance, actor-based video frame processing library. The new architecture separates concerns into three main layers:

1. **Tagging Layer**: Stateless frame analysis that computes metrics and assigns tags
2. **Filtering Layer**: Decision-making components (stateless and stateful) that determine frame selection
3. **Orchestration Layer**: Actor-based distributed processing using xoscar for parallel execution

The design emphasizes:
- Clear separation between stateless (immediate) and stateful (temporal) processing
- **Distributed processing on a single machine using CPU-based actors by default**
- High performance through actor-based parallelism
- Simple, intuitive API for common use cases
- Extensibility for custom processing logic
- CPU-first architecture with optional GPU acceleration (no forced GPU dependencies)
- **Selective GPU usage**: Only specific taggers (e.g., CLIP embeddings) use GPU when available
- **Cloud model integration**: Asynchronous client for remote model inference (no local GPU required)

## Architecture

### High-Level Architecture Diagram

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

### Core Components

#### 1. FramePacket (Data Model)

Enhanced version of VideoFramePacket with tag support:

```python
@dataclass
class FramePacket:
    frame_data: np.ndarray
    frame_number: int
    timestamp: timedelta
    source_id: str
    tags: Dict[str, Any]  # NEW: Tag registry
    metadata: Dict[str, Any]
```

#### 2. Tagger (Stateless Analysis)

Base class for frame analysis components:

```python
class Tagger(ABC):
    @abstractmethod
    def compute_tags(self, packet: FramePacket) -> Dict[str, Any]:
        """Compute tags for a single frame without state."""
        pass
    
    @property
    def tag_keys(self) -> List[str]:
        """Return list of tag keys this tagger produces."""
        pass
```

**Concrete Implementations:**

CPU-based (default):
- `BlurTagger`: Computes Laplacian variance → `{"blur_score": float}`
- `HashTagger`: Computes perceptual hash → `{"phash": str, "hash_value": ImageHash}`
- `EntropyTagger`: Computes Shannon entropy → `{"entropy": float}`
- `ColorTagger`: Computes color histograms → `{"color_hist": np.ndarray}`
- `EdgeTagger`: Computes edge density → `{"edge_density": float}`

GPU-based (optional, when GPU available):
- `CLIPTagger`: Computes CLIP embeddings → `{"clip_embedding": np.ndarray}`

Cloud-based (async client):
- `CloudModelTagger`: Requests embeddings from remote API → `{"model_embedding": np.ndarray, "model_metadata": dict}`

#### 3. Filter (Decision Making)

Base classes for filtering logic:

```python
class Filter(ABC):
    @abstractmethod
    def should_pass(self, packet: FramePacket) -> bool:
        """Determine if frame should pass through."""
        pass

class StatelessFilter(Filter):
    """Makes decisions based only on current frame tags."""
    pass

class StatefulFilter(Filter):
    """Makes decisions using temporal context."""
    def __init__(self, buffer_size: int):
        self.buffer = TemporalBuffer(buffer_size)
    
    @abstractmethod
    def compare_with_history(self, packet: FramePacket, 
                            history: List[FramePacket]) -> bool:
        pass
```

**Concrete Implementations:**

Stateless:
- `ThresholdFilter`: Generic threshold-based filtering
- `BlurFilter`: Filters frames below blur threshold
- `EntropyFilter`: Filters low-entropy frames

Stateful:
- `DuplicateFilter`: Detects duplicate frames via hash comparison
- `MotionFilter`: Detects scene changes via frame differencing
- `DiversityFilter`: Maintains diverse frame selection

#### 4. FilterStrategy (Predefined Pipelines)

High-level strategies that configure complete pipelines:

```python
class FilterStrategy(ABC):
    @abstractmethod
    def build_pipeline(self) -> List[Union[Tagger, Filter]]:
        """Build the complete processing pipeline."""
        pass

class BlurRemovalStrategy(FilterStrategy):
    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold
    
    def build_pipeline(self):
        return [
            BlurTagger(),
            BlurFilter(threshold=self.threshold)
        ]

class DuplicateDetectionStrategy(FilterStrategy):
    def __init__(self, hash_threshold: float = 0.05, window_size: int = 50):
        self.hash_threshold = hash_threshold
        self.window_size = window_size
    
    def build_pipeline(self):
        return [
            HashTagger(hash_type='phash'),
            DuplicateFilter(threshold=self.hash_threshold, 
                          buffer_size=self.window_size)
        ]

class SmartSamplingStrategy(FilterStrategy):
    """Combines blur removal, duplicate detection, and diversity."""
    def build_pipeline(self):
        return [
            BlurTagger(),
            HashTagger(),
            EntropyTagger(),
            BlurFilter(threshold=100.0),
            DuplicateFilter(threshold=0.05, buffer_size=50),
            DiversityFilter(window_size=100, k=10)
        ]
```

#### 5. TemporalBuffer

Efficient sliding window for stateful filters:

```python
class TemporalBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.frames: Deque[FramePacket] = deque(maxlen=max_size)
    
    def add(self, packet: FramePacket) -> None:
        self.frames.append(packet)
    
    def get_window(self) -> List[FramePacket]:
        return list(self.frames)
    
    def find_similar(self, packet: FramePacket, 
                    similarity_fn: Callable) -> Optional[FramePacket]:
        """Find similar frame in buffer using custom similarity function."""
        pass
```

#### 6. FrameProcessor (Main API)

The primary interface for video processing:

```python
class FrameProcessor:
    def __init__(self, 
                 pipeline: Optional[List[Union[Tagger, Filter]]] = None,
                 strategy: Optional[FilterStrategy] = None,
                 n_workers: int = 4,
                 use_gpu: bool = False,
                 gpu_batch_size: int = 32):
        """
        Initialize frame processor.
        
        Args:
            pipeline: Custom pipeline of taggers and filters
            strategy: Predefined strategy (overrides pipeline)
            n_workers: Number of CPU worker actors
            use_gpu: Enable GPU acceleration
            gpu_batch_size: Batch size for GPU processing
        """
        pass
    
    def process(self, 
                source: Union[str, Iterator[FramePacket], List[FramePacket]],
                session_id: Optional[str] = None) -> Iterator[FramePacket]:
        """
        Process video frames through the pipeline.
        
        Args:
            source: Video file path, frame iterator, or frame list
            session_id: Optional session identifier for logging
            
        Yields:
            FramePacket objects that passed all filters
        """
        pass
    
    # Builder methods for common configurations
    @classmethod
    def with_blur_removal(cls, threshold: float = 100.0, **kwargs):
        return cls(strategy=BlurRemovalStrategy(threshold), **kwargs)
    
    @classmethod
    def with_duplicate_detection(cls, threshold: float = 0.05, **kwargs):
        return cls(strategy=DuplicateDetectionStrategy(threshold), **kwargs)
    
    @classmethod
    def with_smart_sampling(cls, **kwargs):
        return cls(strategy=SmartSamplingStrategy(), **kwargs)
```

### Actor-Based Parallel Processing

#### Actor Architecture

**Default Processing Model**: Distributed processing on a single machine using CPU-based actors. The xoscar framework enables parallel processing across multiple CPU cores without requiring GPU hardware or cloud infrastructure.

Using xoscar for distributed processing:

```python
import xoscar as xo

@xo.actor
class TaggerActor:
    """Actor that executes tagger operations."""
    def __init__(self, tagger: Tagger):
        self.tagger = tagger
    
    async def process_frame(self, packet: FramePacket) -> FramePacket:
        tags = self.tagger.compute_tags(packet)
        packet.tags.update(tags)
        return packet

@xo.actor
class FilterActor:
    """Actor that executes filter operations."""
    def __init__(self, filter: Filter):
        self.filter = filter
    
    async def process_frame(self, packet: FramePacket) -> Optional[FramePacket]:
        if self.filter.should_pass(packet):
            return packet
        return None

@xo.actor
class StatefulFilterActor:
    """Actor that maintains state for temporal filtering."""
    def __init__(self, filter: StatefulFilter):
        self.filter = filter
        self.buffer = TemporalBuffer(filter.buffer_size)
    
    async def process_frame(self, packet: FramePacket) -> Optional[FramePacket]:
        history = self.buffer.get_window()
        if self.filter.compare_with_history(packet, history):
            self.buffer.add(packet)
            return packet
        return None

@xo.actor
class GPUBatchProcessor:
    """Actor that batches frames for GPU processing (used only for CLIP and similar GPU taggers)."""
    def __init__(self, tagger: Tagger, batch_size: int = 32):
        self.tagger = tagger
        self.batch_size = batch_size
        self.batch: List[FramePacket] = []
    
    async def add_frame(self, packet: FramePacket) -> Optional[List[FramePacket]]:
        self.batch.append(packet)
        if len(self.batch) >= self.batch_size:
            return await self.process_batch()
        return None
    
    async def process_batch(self) -> List[FramePacket]:
        # Batch process on GPU (e.g., CLIP embeddings)
        results = self.tagger.compute_tags_batch([p.frame_data for p in self.batch])
        for packet, tags in zip(self.batch, results):
            packet.tags.update(tags)
        processed = self.batch
        self.batch = []
        return processed

@xo.actor
class CloudModelActor:
    """Actor that requests embeddings from cloud-based models via async client."""
    def __init__(self, tagger: 'CloudModelTagger', batch_size: int = 16):
        self.tagger = tagger
        self.batch_size = batch_size
        self.batch: List[FramePacket] = []
    
    async def add_frame(self, packet: FramePacket) -> Optional[List[FramePacket]]:
        self.batch.append(packet)
        if len(self.batch) >= self.batch_size:
            return await self.process_batch()
        return None
    
    async def process_batch(self) -> List[FramePacket]:
        # Send batch to cloud API asynchronously
        results = await self.tagger.compute_tags_batch_async([p.frame_data for p in self.batch])
        for packet, tags in zip(self.batch, results):
            packet.tags.update(tags)
        processed = self.batch
        self.batch = []
        return processed
```

#### Pipeline Execution Flow

```python
class ActorPipeline:
    """Manages actor-based pipeline execution on a single machine."""
    
    def __init__(self, pipeline: List[Union[Tagger, Filter]], 
                 n_workers: int, use_gpu: bool):
        self.pipeline = pipeline
        self.n_workers = n_workers  # Number of CPU workers for distributed processing
        self.use_gpu = use_gpu
        self.actor_pools = {}
    
    async def initialize(self):
        """Create actor pools for each pipeline stage.
        
        By default, creates CPU-based actor pools distributed across available cores.
        GPU actors are only created for specific taggers (e.g., CLIP) when GPU is available.
        Cloud model actors use async HTTP clients instead of GPU.
        """
        await xo.actor_pool.create_actor_pool(
            address='127.0.0.1:13527',
            n_process=self.n_workers  # Distribute across CPU cores
        )
        
        for i, component in enumerate(self.pipeline):
            if isinstance(component, Tagger):
                # Check if this is a cloud-based tagger
                if hasattr(component, 'is_cloud_based') and component.is_cloud_based:
                    # Create cloud model actor (uses async HTTP client)
                    actor_ref = await xo.create_actor(
                        CloudModelActor,
                        component,
                        address='127.0.0.1:13527'
                    )
                # Check if this specific tagger needs GPU (e.g., CLIP)
                elif self.use_gpu and component.supports_gpu and component.requires_gpu:
                    # Create GPU batch processor (only for CLIP and similar)
                    actor_ref = await xo.create_actor(
                        GPUBatchProcessor,
                        component,
                        address='127.0.0.1:13527'
                    )
                else:
                    # Create CPU tagger pool (default for most taggers)
                    actor_ref = await xo.create_actor_pool(
                        TaggerActor,
                        component,
                        n_process=self.n_workers,
                        address='127.0.0.1:13527'
                    )
                self.actor_pools[f'stage_{i}'] = actor_ref
            
            elif isinstance(component, StatefulFilter):
                # Single actor for stateful filters (maintains state)
                actor_ref = await xo.create_actor(
                    StatefulFilterActor,
                    component,
                    address='127.0.0.1:13527'
                )
                self.actor_pools[f'stage_{i}'] = actor_ref
            
            elif isinstance(component, StatelessFilter):
                # Pool of CPU actors for stateless filters
                actor_ref = await xo.create_actor_pool(
                    FilterActor,
                    component,
                    n_process=self.n_workers,
                    address='127.0.0.1:13527'
                )
                self.actor_pools[f'stage_{i}'] = actor_ref
    
    async def process_frame(self, packet: FramePacket) -> Optional[FramePacket]:
        """Process a single frame through the actor pipeline."""
        current_packet = packet
        
        for i, component in enumerate(self.pipeline):
            if current_packet is None:
                break
            
            actor_ref = self.actor_pools[f'stage_{i}']
            
            if isinstance(component, Tagger):
                current_packet = await actor_ref.process_frame(current_packet)
            elif isinstance(component, Filter):
                current_packet = await actor_ref.process_frame(current_packet)
        
        return current_packet
    
    async def shutdown(self):
        """Gracefully shutdown all actors."""
        for actor_ref in self.actor_pools.values():
            await xo.destroy_actor(actor_ref)
```

## Components and Interfaces

### Tagger Interface

```python
class Tagger(ABC):
    """Base class for all taggers."""
    
    @abstractmethod
    def compute_tags(self, packet: FramePacket) -> Dict[str, Any]:
        """Compute tags for a single frame (CPU implementation by default)."""
        pass
    
    @property
    @abstractmethod
    def tag_keys(self) -> List[str]:
        """Return list of tag keys produced."""
        pass
    
    @property
    def supports_gpu(self) -> bool:
        """Whether this tagger supports GPU acceleration."""
        return False
    
    @property
    def requires_gpu(self) -> bool:
        """Whether this tagger requires GPU (e.g., CLIP embeddings).
        Most taggers are CPU-based and return False."""
        return False
    
    @property
    def is_cloud_based(self) -> bool:
        """Whether this tagger uses cloud-based models via async client."""
        return False
    
    def compute_tags_batch(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Batch compute tags (for GPU processing, e.g., CLIP)."""
        raise NotImplementedError("Batch processing not supported")
    
    async def compute_tags_batch_async(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Async batch compute tags (for cloud-based models)."""
        raise NotImplementedError("Async batch processing not supported")
```

### Filter Interface

```python
class Filter(ABC):
    """Base class for all filters."""
    
    @abstractmethod
    def should_pass(self, packet: FramePacket) -> bool:
        """Determine if frame should pass."""
        pass
    
    @property
    @abstractmethod
    def required_tags(self) -> List[str]:
        """Return list of required tag keys."""
        pass

class StatelessFilter(Filter):
    """Filter that makes decisions based only on current frame."""
    pass

class StatefulFilter(Filter):
    """Filter that maintains temporal context."""
    
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
    
    @abstractmethod
    def compare_with_history(self, packet: FramePacket, 
                            history: List[FramePacket]) -> bool:
        """Compare current frame with historical frames."""
        pass
    
    def should_pass(self, packet: FramePacket) -> bool:
        """Implemented by base class using compare_with_history."""
        pass
```

## Data Models

### FramePacket

```python
@dataclass
class FramePacket:
    """Enhanced frame packet with tag support."""
    
    frame_data: np.ndarray
    frame_number: int
    timestamp: timedelta
    source_id: str
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_tag(self, key: str, default: Any = None) -> Any:
        """Get tag value with optional default."""
        return self.tags.get(key, default)
    
    def has_tags(self, keys: List[str]) -> bool:
        """Check if all required tags are present."""
        return all(key in self.tags for key in keys)
    
    def copy_without_frame_data(self) -> 'FramePacket':
        """Create lightweight copy without frame data (for logging)."""
        return FramePacket(
            frame_data=np.array([]),  # Empty array
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            source_id=self.source_id,
            tags=self.tags.copy(),
            metadata=self.metadata.copy()
        )
```

### ProcessingResult

```python
@dataclass
class ProcessingResult:
    """Summary of processing session."""
    
    session_id: str
    total_frames: int
    processed_frames: int
    filtered_frames: int
    selected_frames: int
    processing_time: float
    stage_metrics: Dict[str, Dict[str, Any]]
    errors: List[str]
```

## GPU Dependency Management

### Design Rationale

**Decision**: Implement CPU-first architecture with optional GPU acceleration

**Rationale**: Many deployment environments (CI/CD, edge devices, development machines) don't have GPU capabilities or GPU libraries installed. Forcing GPU dependencies would make the library harder to install and use in these common scenarios. By isolating GPU code and making it optional, we maximize compatibility while still enabling high-performance GPU acceleration when available.

### Processing Model Summary

1. **Default (CPU-based)**: Distributed processing across CPU cores using xoscar actors
   - Taggers: Blur, Hash, Entropy, Color, Edge (all CPU-based)
   - No GPU dependencies required
   - Works on any machine

2. **GPU-accelerated (Optional)**: Only for specific taggers like CLIP
   - Requires `pip install decimatr[gpu]`
   - Uses GPU batch processing for efficiency
   - Falls back to CPU if GPU unavailable

3. **Cloud-based (Async)**: Remote model inference via HTTP client
   - No local GPU required
   - Uses async HTTP client (aiohttp)
   - Batches requests for efficiency
   - Works with CPU-only installation

### GPU Isolation Strategy

#### 1. Dependency Structure

```toml
[project]
dependencies = [
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "imagehash>=4.3.1",
    "xoscar>=0.3.0",  # Actor framework for distributed CPU processing
    "decord>=0.6.0",  # Video loading
    "aiohttp>=3.8.0"  # Async HTTP client for cloud models
]

[project.optional-dependencies]
gpu = [
    "torch>=2.0.0",  # For CLIP and GPU-accelerated operations
    "torchvision>=0.15.0",
    "ftfy>=6.1.0",  # Required by CLIP
    "regex>=2023.0.0"  # Required by CLIP
]
```

**Installation Options:**
- **CPU-only (default)**: `pip install decimatr` - Supports all CPU-based taggers and cloud models
- **With GPU support**: `pip install decimatr[gpu]` - Adds CLIP embeddings and GPU acceleration
- **Note**: Cloud-based models work with CPU-only installation (no GPU required)

#### 2. GPU Availability Detection

```python
class GPUCapabilities:
    """Detect and report GPU capabilities."""
    
    _gpu_available: Optional[bool] = None
    _missing_dependencies: List[str] = []
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if GPU acceleration is available."""
        if cls._gpu_available is None:
            cls._check_gpu_dependencies()
        return cls._gpu_available
    
    @classmethod
    def _check_gpu_dependencies(cls) -> None:
        """Check for GPU dependencies and CUDA availability."""
        cls._missing_dependencies = []
        
        try:
            import torch
            if not torch.cuda.is_available():
                cls._missing_dependencies.append("CUDA runtime")
        except ImportError:
            cls._missing_dependencies.append("torch")
        
        try:
            import torchvision
        except ImportError:
            cls._missing_dependencies.append("torchvision")
        
        cls._gpu_available = len(cls._missing_dependencies) == 0
    
    @classmethod
    def get_missing_dependencies(cls) -> List[str]:
        """Return list of missing GPU dependencies."""
        if cls._gpu_available is None:
            cls._check_gpu_dependencies()
        return cls._missing_dependencies.copy()
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get detailed GPU information."""
        info = {
            "gpu_available": cls.is_available(),
            "missing_dependencies": cls.get_missing_dependencies()
        }
        
        if cls.is_available():
            import torch
            info.update({
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            })
        
        return info
```

#### 3. Tagger Implementation Patterns

##### CPU-Based Taggers (Default)

Most taggers are CPU-based and work out of the box:

```python
class BlurTagger(Tagger):
    """Compute blur score using Laplacian variance (CPU-based)."""
    
    def compute_tags(self, packet: FramePacket) -> Dict[str, Any]:
        """Compute blur score using CPU."""
        gray = cv2.cvtColor(packet.frame_data, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {"blur_score": float(laplacian_var)}
    
    @property
    def tag_keys(self) -> List[str]:
        return ["blur_score"]
```

##### GPU-Based Taggers (CLIP Embeddings)

Only specific taggers like CLIP require GPU:

```python
class CLIPTagger(Tagger):
    """Compute CLIP embeddings using GPU (when available)."""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto"):
        """
        Initialize CLIP tagger.
        
        Args:
            model_name: CLIP model variant
            device: 'auto', 'cuda', or 'cpu'
        """
        if device == "auto":
            device = "cuda" if GPUCapabilities.is_available() else "cpu"
        
        if device == "cuda" and not GPUCapabilities.is_available():
            raise GPUDependencyError(
                f"GPU requested but dependencies are missing: "
                f"{', '.join(GPUCapabilities.get_missing_dependencies())}. "
                f"Install with: pip install decimatr[gpu]"
            )
        
        self.device = device
        self.model_name = model_name
        self._model = None
        self._preprocess = None
    
    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model is None:
            import torch
            import clip
            self._model, self._preprocess = clip.load(self.model_name, device=self.device)
    
    def compute_tags(self, packet: FramePacket) -> Dict[str, Any]:
        """Compute CLIP embedding for single frame."""
        self._load_model()
        import torch
        from PIL import Image
        
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(packet.frame_data, cv2.COLOR_BGR2RGB))
        image_input = self._preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self._model.encode_image(image_input)
            embedding = embedding.cpu().numpy().flatten()
        
        return {"clip_embedding": embedding}
    
    def compute_tags_batch(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Batch compute CLIP embeddings (efficient GPU processing)."""
        self._load_model()
        import torch
        from PIL import Image
        
        # Prepare batch
        images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        image_inputs = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            embeddings = self._model.encode_image(image_inputs)
            embeddings = embeddings.cpu().numpy()
        
        return [{"clip_embedding": emb} for emb in embeddings]
    
    @property
    def tag_keys(self) -> List[str]:
        return ["clip_embedding"]
    
    @property
    def supports_gpu(self) -> bool:
        return True
    
    @property
    def requires_gpu(self) -> bool:
        """CLIP benefits significantly from GPU but can fall back to CPU."""
        return False  # Can run on CPU if needed, just slower
```

##### Cloud-Based Taggers (Async Client)

Cloud model taggers use async HTTP clients instead of local GPU:

```python
class CloudModelTagger(Tagger):
    """Request embeddings from cloud-based models via async HTTP client."""
    
    def __init__(self, api_url: str, api_key: str, model_name: str, timeout: float = 30.0):
        """
        Initialize cloud model tagger.
        
        Args:
            api_url: Cloud API endpoint
            api_key: Authentication key
            model_name: Model identifier
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        """Lazy initialize async HTTP client."""
        if self._client is None:
            import aiohttp
            self._client = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._client
    
    def compute_tags(self, packet: FramePacket) -> Dict[str, Any]:
        """Synchronous wrapper (not recommended, use async version)."""
        import asyncio
        return asyncio.run(self.compute_tags_async(packet))
    
    async def compute_tags_async(self, packet: FramePacket) -> Dict[str, Any]:
        """Compute tags by requesting cloud model."""
        client = self._get_client()
        
        # Encode frame as base64
        import base64
        _, buffer = cv2.imencode('.jpg', packet.frame_data)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send request
        async with client.post(
            self.api_url,
            json={
                "model": self.model_name,
                "image": image_b64
            },
            timeout=self.timeout
        ) as response:
            result = await response.json()
            return {
                "model_embedding": np.array(result["embedding"]),
                "model_metadata": result.get("metadata", {})
            }
    
    async def compute_tags_batch_async(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Batch compute tags via cloud API."""
        client = self._get_client()
        import base64
        
        # Encode all frames
        images_b64 = []
        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame)
            images_b64.append(base64.b64encode(buffer).decode('utf-8'))
        
        # Send batch request
        async with client.post(
            f"{self.api_url}/batch",
            json={
                "model": self.model_name,
                "images": images_b64
            },
            timeout=self.timeout * 2  # Longer timeout for batch
        ) as response:
            results = await response.json()
            return [
                {
                    "model_embedding": np.array(r["embedding"]),
                    "model_metadata": r.get("metadata", {})
                }
                for r in results["embeddings"]
            ]
    
    @property
    def tag_keys(self) -> List[str]:
        return ["model_embedding", "model_metadata"]
    
    @property
    def is_cloud_based(self) -> bool:
        return True
    
    async def close(self):
        """Close async HTTP client."""
        if self._client:
            await self._client.close()
```

#### 4. GPU Module Structure

GPU implementations are isolated in separate modules:

```
decimatr/
├── taggers/
│   ├── __init__.py
│   ├── base.py
│   ├── blur.py          # CPU implementation (always available)
│   ├── hash.py
│   ├── entropy.py
│   └── gpu/             # GPU implementations (optional)
│       ├── __init__.py
│       ├── blur_gpu.py  # GPU-accelerated blur
│       ├── hash_gpu.py
│       └── entropy_gpu.py
```

#### 5. FrameProcessor GPU Configuration

```python
class FrameProcessor:
    def __init__(self, 
                 pipeline: Optional[List[Union[Tagger, Filter]]] = None,
                 strategy: Optional[FilterStrategy] = None,
                 n_workers: int = 4,
                 use_gpu: bool = False,
                 gpu_batch_size: int = 32,
                 gpu_fallback: bool = True):
        """
        Initialize frame processor.
        
        Args:
            pipeline: Custom pipeline of taggers and filters
            strategy: Predefined strategy (overrides pipeline)
            n_workers: Number of CPU worker actors
            use_gpu: Enable GPU acceleration (requires GPU dependencies)
            gpu_batch_size: Batch size for GPU processing
            gpu_fallback: Automatically fall back to CPU if GPU fails
        
        Raises:
            GPUDependencyError: If use_gpu=True but GPU dependencies missing
        """
        self.use_gpu = use_gpu
        self.gpu_fallback = gpu_fallback
        
        # Validate GPU configuration
        if self.use_gpu and not GPUCapabilities.is_available():
            missing = GPUCapabilities.get_missing_dependencies()
            raise GPUDependencyError(
                f"GPU acceleration requested but dependencies are missing: {', '.join(missing)}. "
                f"Install with: pip install decimatr[gpu]"
            )
        
        # ... rest of initialization
    
    @staticmethod
    def check_gpu_available() -> bool:
        """Check if GPU acceleration is available."""
        return GPUCapabilities.is_available()
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get detailed GPU information."""
        return GPUCapabilities.get_info()
```

### GPU Fallback Mechanism

When GPU operations fail at runtime, the system automatically falls back to CPU:

```python
@xo.actor
class GPUBatchProcessor:
    """Actor that batches frames for GPU processing with fallback."""
    
    def __init__(self, tagger: Tagger, batch_size: int = 32, fallback_to_cpu: bool = True):
        self.tagger = tagger
        self.batch_size = batch_size
        self.fallback_to_cpu = fallback_to_cpu
        self.batch: List[FramePacket] = []
        self.gpu_failures = 0
        self.max_gpu_failures = 3  # Switch to CPU after 3 failures
    
    async def process_batch(self) -> List[FramePacket]:
        """Process batch on GPU with CPU fallback."""
        if self.gpu_failures >= self.max_gpu_failures:
            logger.warning("Too many GPU failures, switching to CPU processing")
            return await self._process_batch_cpu()
        
        try:
            # Try GPU processing
            results = self.tagger.compute_tags_batch([p.frame_data for p in self.batch])
            for packet, tags in zip(self.batch, results):
                packet.tags.update(tags)
            processed = self.batch
            self.batch = []
            return processed
        
        except Exception as e:
            self.gpu_failures += 1
            logger.warning(f"GPU batch processing failed (attempt {self.gpu_failures}): {e}")
            
            if self.fallback_to_cpu:
                logger.info("Falling back to CPU processing")
                return await self._process_batch_cpu()
            else:
                raise
    
    async def _process_batch_cpu(self) -> List[FramePacket]:
        """Process batch using CPU."""
        for packet in self.batch:
            tags = self.tagger.compute_tags(packet)  # CPU method
            packet.tags.update(tags)
        processed = self.batch
        self.batch = []
        return processed
```

## Error Handling

### Exception Hierarchy

```python
class DecimatrError(Exception):
    """Base exception for all Decimatr errors."""
    pass

class ConfigurationError(DecimatrError):
    """Raised when pipeline configuration is invalid."""
    pass

class TagMissingError(DecimatrError):
    """Raised when required tag is missing."""
    pass

class ProcessingError(DecimatrError):
    """Raised when frame processing fails."""
    pass

class ActorError(DecimatrError):
    """Raised when actor operations fail."""
    pass

class GPUDependencyError(DecimatrError):
    """Raised when GPU acceleration is requested but GPU dependencies are missing."""
    pass
```

### Error Handling Strategy

1. **Configuration Validation**: Validate pipeline at initialization, fail fast
2. **GPU Dependency Errors**: Raise descriptive error with installation instructions when GPU requested but unavailable
3. **Frame-Level Errors**: Log error, skip frame, continue processing
4. **GPU Runtime Errors**: Log warning and fall back to CPU if fallback enabled
5. **Actor Failures**: Restart failed actors, reprocess frame
6. **Critical Errors**: Shutdown gracefully, return partial results

```python
class FrameProcessor:
    async def process_frame_safe(self, packet: FramePacket) -> Optional[FramePacket]:
        """Process frame with error handling."""
        try:
            return await self.pipeline.process_frame(packet)
        except TagMissingError as e:
            self.logger.error(f"Missing tag for frame {packet.frame_number}: {e}")
            self.metrics.tag_errors += 1
            return None
        except ProcessingError as e:
            self.logger.error(f"Processing error for frame {packet.frame_number}: {e}")
            self.metrics.processing_errors += 1
            return None
        except Exception as e:
            self.logger.exception(f"Unexpected error for frame {packet.frame_number}")
            self.metrics.unexpected_errors += 1
            return None
```

## Testing Strategy

### Unit Tests

1. **Tagger Tests**: Verify tag computation correctness
   - Test each tagger with known inputs
   - Verify tag keys and value types
   - Test edge cases (empty frames, single-color frames)

2. **Filter Tests**: Verify filtering logic
   - Test threshold-based filtering
   - Test stateful filter buffer management
   - Test filter combinations

3. **TemporalBuffer Tests**: Verify buffer operations
   - Test sliding window behavior
   - Test capacity limits
   - Test similarity search

### Integration Tests

1. **Pipeline Tests**: End-to-end pipeline execution
   - Test various pipeline configurations
   - Verify frame flow through stages
   - Test error propagation

2. **Actor Tests**: Verify actor-based processing
   - Test actor pool creation and shutdown
   - Test frame distribution across workers
   - Test stateful actor state management

3. **Strategy Tests**: Verify predefined strategies
   - Test each FilterStrategy
   - Verify expected frame selection
   - Test performance characteristics

### Performance Tests

1. **Throughput Tests**: Measure frames per second
2. **Scalability Tests**: Verify performance with varying worker counts
3. **Memory Tests**: Verify memory usage stays bounded
4. **GPU Tests**: Verify GPU acceleration benefits

## Implementation Notes

### Phase 1: Core Architecture (No Actors)
- Implement FramePacket, Tagger, Filter base classes
- Implement concrete CPU taggers (Blur, Hash, Entropy)
- Implement concrete filters (Threshold, Duplicate)
- Implement TemporalBuffer
- Implement basic FrameProcessor (single-threaded)
- Implement FilterStrategy patterns
- Implement GPU capability detection (GPUCapabilities class)

### Phase 2: Actor Integration
- Integrate xoscar actor system
- Implement TaggerActor, FilterActor, StatefulFilterActor
- Implement ActorPipeline
- Implement actor pool management
- Add graceful shutdown

### Phase 3: GPU Acceleration (Optional)
- Implement GPU capability detection and validation
- Create GPU module structure (taggers/gpu/)
- Implement GPU-accelerated taggers with CPU fallback
- Implement GPUBatchProcessor actor with fallback logic
- Add GPU dependency error handling
- Optimize batch sizes and memory management

### Phase 4: Polish & Optimization
- Add comprehensive logging
- Optimize memory usage
- Add monitoring and metrics
- Performance tuning
- Documentation and examples
- Add GPU installation and troubleshooting guide

### Dependencies

```toml
[project]
name = "decimatr"
dependencies = [
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "imagehash>=4.3.1",
    "xoscar>=0.3.0",  # Actor framework
    "decord>=0.6.0"   # Video loading
]

[project.optional-dependencies]
gpu = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "ftfy>=6.1.0",
    "regex>=2023.0.0"
]

dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0"
]
```

**Installation Options:**
- **CPU-only (default)**: `pip install decimatr` - Distributed CPU processing + cloud models
- **With GPU support**: `pip install decimatr[gpu]` - Adds CLIP embeddings
- **Development**: `pip install decimatr[dev]`

### File Structure

```
decimatr/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── frame_packet.py      # FramePacket data model
│   ├── processor.py          # FrameProcessor main API
│   └── temporal_buffer.py    # TemporalBuffer implementation
├── taggers/
│   ├── __init__.py
│   ├── base.py              # Tagger base class
│   ├── blur.py              # BlurTagger (CPU)
│   ├── hash.py              # HashTagger (CPU)
│   ├── entropy.py           # EntropyTagger (CPU)
│   ├── color.py             # ColorTagger (CPU)
│   ├── edge.py              # EdgeTagger (CPU)
│   ├── clip.py              # CLIPTagger (GPU, optional dependency)
│   └── cloud_model.py       # CloudModelTagger (async HTTP client)
├── filters/
│   ├── __init__.py
│   ├── base.py              # Filter base classes
│   ├── threshold.py         # ThresholdFilter
│   ├── blur.py              # BlurFilter
│   ├── duplicate.py         # DuplicateFilter
│   ├── motion.py            # MotionFilter
│   └── diversity.py         # DiversityFilter
├── strategies/
│   ├── __init__.py
│   ├── base.py              # FilterStrategy base
│   ├── blur_removal.py      # BlurRemovalStrategy
│   ├── duplicate_detection.py  # DuplicateDetectionStrategy
│   └── smart_sampling.py    # SmartSamplingStrategy
├── actors/
│   ├── __init__.py
│   ├── tagger_actor.py      # TaggerActor (CPU-based)
│   ├── filter_actor.py      # FilterActor (CPU-based)
│   ├── stateful_actor.py    # StatefulFilterActor (CPU-based)
│   ├── gpu_actor.py         # GPUBatchProcessor (for CLIP)
│   ├── cloud_actor.py       # CloudModelActor (async HTTP)
│   └── pipeline.py          # ActorPipeline
├── utils/
│   ├── __init__.py
│   ├── video_loader.py      # Video loading utilities
│   ├── logging.py           # Logging configuration
│   ├── metrics.py           # Performance metrics
│   └── gpu_utils.py         # GPU detection and capabilities
└── exceptions.py            # Exception classes
```
