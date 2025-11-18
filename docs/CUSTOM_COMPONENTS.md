# Creating Custom Components

This guide shows you how to create custom taggers, filters, and strategies for Decimatr.

## Table of Contents

- [Custom Taggers](#custom-taggers)
- [Custom Stateless Filters](#custom-stateless-filters)
- [Custom Stateful Filters](#custom-stateful-filters)
- [Custom Strategies](#custom-strategies)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Custom Taggers

Taggers analyze frames and compute metadata tags. They are stateless - each frame is processed independently.

### Basic Tagger Template

```python
from decimatr.taggers.base import Tagger
from decimatr.scheme import VideoFramePacket
from typing import Dict, Any, List
import numpy as np

class MyTagger(Tagger):
    """
    Brief description of what this tagger computes.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
    
    Tags Produced:
        - tag_name (type): Description of tag
    
    Example:
        >>> tagger = MyTagger(param1=value)
        >>> tags = tagger.compute_tags(packet)
        >>> print(tags)
        {'tag_name': value}
    """
    
    def __init__(self, param1: float = 1.0, param2: str = "default"):
        """Initialize tagger with parameters."""
        self.param1 = param1
        self.param2 = param2
    
    def compute_tags(self, packet: VideoFramePacket) -> Dict[str, Any]:
        """
        Compute tags for a single frame.
        
        Args:
            packet: VideoFramePacket containing frame data
        
        Returns:
            Dictionary of computed tags
        """
        # Access frame data
        frame = packet.frame_data  # numpy array (H, W, C)
        
        # Compute your metric
        metric_value = self._compute_metric(frame)
        
        # Return tags
        return {
            "my_metric": metric_value
        }
    
    @property
    def tag_keys(self) -> List[str]:
        """Return list of tag keys this tagger produces."""
        return ["my_metric"]
    
    def _compute_metric(self, frame: np.ndarray) -> float:
        """Helper method to compute metric."""
        # Your computation logic here
        return 0.0
```

### Example: Color Histogram Tagger

```python
import cv2
import numpy as np
from decimatr.taggers.base import Tagger
from decimatr.scheme import VideoFramePacket

class ColorHistogramTagger(Tagger):
    """
    Compute color histogram for frame.
    
    Args:
        bins: Number of bins per channel (default: 32)
        normalize: Whether to normalize histogram (default: True)
    
    Tags Produced:
        - color_histogram (np.ndarray): Flattened color histogram
        - dominant_color (tuple): RGB values of dominant color
    """
    
    def __init__(self, bins: int = 32, normalize: bool = True):
        self.bins = bins
        self.normalize = normalize
    
    def compute_tags(self, packet: VideoFramePacket) -> Dict[str, Any]:
        frame = packet.frame_data
        
        # Compute histogram for each channel
        hist_b = cv2.calcHist([frame], [0], None, [self.bins], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [self.bins], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [self.bins], [0, 256])
        
        # Concatenate histograms
        histogram = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        
        # Normalize if requested
        if self.normalize:
            histogram = histogram / histogram.sum()
        
        # Find dominant color
        mean_color = frame.mean(axis=(0, 1))
        dominant_color = tuple(int(c) for c in mean_color)
        
        return {
            "color_histogram": histogram,
            "dominant_color": dominant_color
        }
    
    @property
    def tag_keys(self) -> List[str]:
        return ["color_histogram", "dominant_color"]
```

### Example: Edge Density Tagger

```python
import cv2
from decimatr.taggers.base import Tagger
from decimatr.scheme import VideoFramePacket

class EdgeDensityTagger(Tagger):
    """
    Compute edge density using Canny edge detection.
    
    Args:
        low_threshold: Lower threshold for Canny (default: 50)
        high_threshold: Upper threshold for Canny (default: 150)
    
    Tags Produced:
        - edge_density (float): Percentage of edge pixels (0.0-1.0)
        - edge_count (int): Number of edge pixels
    """
    
    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def compute_tags(self, packet: VideoFramePacket) -> Dict[str, Any]:
        frame = packet.frame_data
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # Count edge pixels
        edge_count = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_density = edge_count / total_pixels
        
        return {
            "edge_density": float(edge_density),
            "edge_count": int(edge_count)
        }
    
    @property
    def tag_keys(self) -> List[str]:
        return ["edge_density", "edge_count"]
```

### GPU-Accelerated Tagger

```python
from decimatr.taggers.base import Tagger
from decimatr.scheme import VideoFramePacket
from decimatr.gpu_utils import GPUCapabilities
from decimatr.exceptions import GPUDependencyError
import numpy as np

class GPUTagger(Tagger):
    """
    GPU-accelerated tagger example.
    
    Args:
        device: Device to use ('auto', 'cuda', or 'cpu')
    """
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if GPUCapabilities.is_available() else "cpu"
        
        if device == "cuda" and not GPUCapabilities.is_available():
            raise GPUDependencyError(
                f"GPU requested but dependencies are missing: "
                f"{', '.join(GPUCapabilities.get_missing_dependencies())}. "
                f"Install with: pip install decimatr[gpu]"
            )
        
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Lazy load model."""
        if self._model is None:
            import torch
            # Load your model here
            self._model = YourModel().to(self.device)
    
    def compute_tags(self, packet: VideoFramePacket) -> Dict[str, Any]:
        """Compute tags using GPU."""
        self._load_model()
        import torch
        
        # Convert frame to tensor
        frame_tensor = torch.from_numpy(packet.frame_data).to(self.device)
        
        # Process on GPU
        with torch.no_grad():
            result = self._model(frame_tensor)
        
        # Convert back to numpy
        result_np = result.cpu().numpy()
        
        return {"gpu_metric": result_np}
    
    def compute_tags_batch(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Batch process on GPU for efficiency."""
        self._load_model()
        import torch
        
        # Stack frames into batch
        batch = torch.stack([
            torch.from_numpy(f) for f in frames
        ]).to(self.device)
        
        # Process batch on GPU
        with torch.no_grad():
            results = self._model(batch)
        
        # Convert to list of dicts
        results_np = results.cpu().numpy()
        return [{"gpu_metric": r} for r in results_np]
    
    @property
    def tag_keys(self) -> List[str]:
        return ["gpu_metric"]
    
    @property
    def supports_gpu(self) -> bool:
        return True
    
    @property
    def requires_gpu(self) -> bool:
        return False  # Can fall back to CPU
```

## Custom Stateless Filters

Stateless filters make decisions based only on the current frame's tags.

### Basic Stateless Filter Template

```python
from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket
from typing import List

class MyFilter(StatelessFilter):
    """
    Brief description of what this filter does.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
    
    Required Tags:
        - tag_name: Description of required tag
    
    Example:
        >>> filter = MyFilter(param1=value)
        >>> packet.tags = {"tag_name": value}
        >>> filter.should_pass(packet)
        True
    """
    
    def __init__(self, param1: float = 1.0, param2: str = "default"):
        """Initialize filter with parameters."""
        self.param1 = param1
        self.param2 = param2
    
    def should_pass(self, packet: VideoFramePacket) -> bool:
        """
        Determine if frame should pass through filter.
        
        Args:
            packet: VideoFramePacket with tags
        
        Returns:
            True if frame passes, False if filtered out
        """
        # Get required tag
        tag_value = packet.get_tag("tag_name")
        
        # Make decision
        return tag_value > self.param1
    
    @property
    def required_tags(self) -> List[str]:
        """Return list of required tag keys."""
        return ["tag_name"]
```

### Example: Color Saturation Filter

```python
from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket
import numpy as np

class ColorSaturationFilter(StatelessFilter):
    """
    Filter frames based on color saturation.
    
    Args:
        min_saturation: Minimum average saturation (0.0-1.0)
        max_saturation: Maximum average saturation (0.0-1.0)
    
    Required Tags:
        - None (computes saturation directly from frame_data)
    """
    
    def __init__(self, min_saturation: float = 0.1, max_saturation: float = 0.9):
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
    
    def should_pass(self, packet: VideoFramePacket) -> bool:
        # Convert BGR to HSV
        import cv2
        hsv = cv2.cvtColor(packet.frame_data, cv2.COLOR_BGR2HSV)
        
        # Get saturation channel (0-255)
        saturation = hsv[:, :, 1]
        
        # Compute average saturation (normalize to 0-1)
        avg_saturation = saturation.mean() / 255.0
        
        # Check if within range
        return self.min_saturation <= avg_saturation <= self.max_saturation
    
    @property
    def required_tags(self) -> List[str]:
        return []  # No tags required
```

### Example: Multi-Condition Filter

```python
from decimatr.filters.base import StatelessFilter
from decimatr.scheme import VideoFramePacket

class QualityFilter(StatelessFilter):
    """
    Filter frames based on multiple quality metrics.
    
    Args:
        min_blur: Minimum blur score
        min_entropy: Minimum entropy
        min_edge_density: Minimum edge density
    
    Required Tags:
        - blur_score
        - entropy
        - edge_density
    """
    
    def __init__(
        self,
        min_blur: float = 100.0,
        min_entropy: float = 4.0,
        min_edge_density: float = 0.1
    ):
        self.min_blur = min_blur
        self.min_entropy = min_entropy
        self.min_edge_density = min_edge_density
    
    def should_pass(self, packet: VideoFramePacket) -> bool:
        # Get all required tags
        blur = packet.get_tag("blur_score", 0.0)
        entropy = packet.get_tag("entropy", 0.0)
        edge_density = packet.get_tag("edge_density", 0.0)
        
        # All conditions must be met
        return (
            blur >= self.min_blur and
            entropy >= self.min_entropy and
            edge_density >= self.min_edge_density
        )
    
    @property
    def required_tags(self) -> List[str]:
        return ["blur_score", "entropy", "edge_density"]
```

## Custom Stateful Filters

Stateful filters maintain a temporal buffer and make decisions based on historical context.

### Basic Stateful Filter Template

```python
from decimatr.filters.base import StatefulFilter
from decimatr.scheme import VideoFramePacket
from typing import List

class MyStatefulFilter(StatefulFilter):
    """
    Brief description of what this filter does.
    
    Args:
        buffer_size: Size of temporal buffer
        param1: Description of parameter
    
    Required Tags:
        - tag_name: Description of required tag
    
    Example:
        >>> filter = MyStatefulFilter(buffer_size=50, param1=value)
        >>> filter.should_pass(packet)
        True
    """
    
    def __init__(self, buffer_size: int = 50, param1: float = 1.0):
        """Initialize filter with buffer and parameters."""
        super().__init__(buffer_size)
        self.param1 = param1
    
    def compare_with_history(
        self,
        packet: VideoFramePacket,
        history: List[VideoFramePacket]
    ) -> bool:
        """
        Compare current frame with historical frames.
        
        Args:
            packet: Current frame to evaluate
            history: List of recent frames from buffer
        
        Returns:
            True if frame should pass and be added to buffer
        """
        # If buffer is empty, pass the frame
        if not history:
            return True
        
        # Get current frame's tag
        current_value = packet.get_tag("tag_name")
        
        # Compare with history
        for past_frame in history:
            past_value = past_frame.get_tag("tag_name")
            
            # Your comparison logic here
            if abs(current_value - past_value) < self.param1:
                return False  # Too similar, filter out
        
        return True  # Different enough, pass through
    
    @property
    def required_tags(self) -> List[str]:
        """Return list of required tag keys."""
        return ["tag_name"]
```

### Example: Scene Change Detector

```python
from decimatr.filters.base import StatefulFilter
from decimatr.scheme import VideoFramePacket
import cv2
import numpy as np

class SceneChangeFilter(StatefulFilter):
    """
    Detect scene changes and only pass frames at scene boundaries.
    
    Args:
        buffer_size: Number of recent frames to keep
        threshold: Difference threshold for scene change (0.0-1.0)
    
    Required Tags:
        - None (uses frame_data directly)
    """
    
    def __init__(self, buffer_size: int = 5, threshold: float = 0.3):
        super().__init__(buffer_size)
        self.threshold = threshold
    
    def compare_with_history(
        self,
        packet: VideoFramePacket,
        history: List[VideoFramePacket]
    ) -> bool:
        # Always pass first frame
        if not history:
            return True
        
        # Compare with most recent frame
        prev_frame = history[-1]
        
        # Compute frame difference
        diff = self._compute_difference(
            packet.frame_data,
            prev_frame.frame_data
        )
        
        # Pass if difference exceeds threshold (scene change detected)
        return diff > self.threshold
    
    def _compute_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute normalized difference between frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Normalize to 0-1
        return diff.mean() / 255.0
    
    @property
    def required_tags(self) -> List[str]:
        return []
```

### Example: Temporal Smoothing Filter

```python
from decimatr.filters.base import StatefulFilter
from decimatr.scheme import VideoFramePacket
import numpy as np

class TemporalSmoothingFilter(StatefulFilter):
    """
    Only pass frames when metric is consistently above threshold.
    
    Args:
        buffer_size: Number of frames to consider
        threshold: Metric threshold
        min_consecutive: Minimum consecutive frames above threshold
    
    Required Tags:
        - blur_score (or any numeric metric)
    """
    
    def __init__(
        self,
        buffer_size: int = 10,
        threshold: float = 100.0,
        min_consecutive: int = 3
    ):
        super().__init__(buffer_size)
        self.threshold = threshold
        self.min_consecutive = min_consecutive
    
    def compare_with_history(
        self,
        packet: VideoFramePacket,
        history: List[VideoFramePacket]
    ) -> bool:
        # Get current metric
        current_metric = packet.get_tag("blur_score", 0.0)
        
        # Check if current frame meets threshold
        if current_metric < self.threshold:
            return False
        
        # Count consecutive frames above threshold
        consecutive = 1  # Current frame
        
        for past_frame in reversed(history):
            past_metric = past_frame.get_tag("blur_score", 0.0)
            if past_metric >= self.threshold:
                consecutive += 1
            else:
                break
        
        # Pass if we have enough consecutive frames
        return consecutive >= self.min_consecutive
    
    @property
    def required_tags(self) -> List[str]:
        return ["blur_score"]
```

## Custom Strategies

Strategies combine taggers and filters into complete pipelines.

### Basic Strategy Template

```python
from decimatr.strategies.base import FilterStrategy
from typing import List, Union
from decimatr.taggers.base import Tagger
from decimatr.filters.base import Filter

class MyStrategy(FilterStrategy):
    """
    Brief description of what this strategy does.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
    
    Pipeline:
        - Tagger1: Description
        - Tagger2: Description
        - Filter1: Description
        - Filter2: Description
    
    Example:
        >>> strategy = MyStrategy(param1=value)
        >>> processor = FrameProcessor(strategy=strategy)
    """
    
    def __init__(self, param1: float = 1.0, param2: int = 50):
        """Initialize strategy with parameters."""
        self.param1 = param1
        self.param2 = param2
    
    def build_pipeline(self) -> List[Union[Tagger, Filter]]:
        """
        Build the complete processing pipeline.
        
        Returns:
            Ordered list of taggers and filters
        """
        from decimatr.taggers.blur import BlurTagger
        from decimatr.filters.blur import BlurFilter
        # Import other components as needed
        
        return [
            # Taggers first
            BlurTagger(),
            # Add more taggers...
            
            # Filters second
            BlurFilter(threshold=self.param1),
            # Add more filters...
        ]
```

### Example: High-Quality Frame Strategy

```python
from decimatr.strategies.base import FilterStrategy
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.entropy import EntropyTagger
from decimatr.filters.blur import BlurFilter
from decimatr.filters.entropy import EntropyFilter

class HighQualityStrategy(FilterStrategy):
    """
    Select only high-quality frames based on multiple metrics.
    
    Args:
        blur_threshold: Minimum blur score (default: 150.0)
        entropy_threshold: Minimum entropy (default: 5.0)
        edge_threshold: Minimum edge density (default: 0.15)
    
    Pipeline:
        - BlurTagger: Compute blur score
        - EntropyTagger: Compute entropy
        - EdgeDensityTagger: Compute edge density
        - QualityFilter: Filter based on all metrics
    """
    
    def __init__(
        self,
        blur_threshold: float = 150.0,
        entropy_threshold: float = 5.0,
        edge_threshold: float = 0.15
    ):
        self.blur_threshold = blur_threshold
        self.entropy_threshold = entropy_threshold
        self.edge_threshold = edge_threshold
    
    def build_pipeline(self):
        from decimatr.taggers.blur import BlurTagger
        from decimatr.taggers.entropy import EntropyTagger
        from your_module import EdgeDensityTagger, QualityFilter
        
        return [
            # Compute all quality metrics
            BlurTagger(),
            EntropyTagger(),
            EdgeDensityTagger(),
            
            # Filter based on combined metrics
            QualityFilter(
                min_blur=self.blur_threshold,
                min_entropy=self.entropy_threshold,
                min_edge_density=self.edge_threshold
            )
        ]
```

### Example: Adaptive Sampling Strategy

```python
from decimatr.strategies.base import FilterStrategy

class AdaptiveSamplingStrategy(FilterStrategy):
    """
    Adaptive sampling that adjusts based on content.
    
    Args:
        base_threshold: Base blur threshold
        duplicate_threshold: Duplicate detection threshold
        scene_change_threshold: Scene change threshold
    
    Pipeline:
        - BlurTagger
        - HashTagger
        - BlurFilter (removes very blurry frames)
        - SceneChangeFilter (detects scene changes)
        - DuplicateFilter (removes duplicates within scenes)
    """
    
    def __init__(
        self,
        base_threshold: float = 100.0,
        duplicate_threshold: float = 0.05,
        scene_change_threshold: float = 0.3
    ):
        self.base_threshold = base_threshold
        self.duplicate_threshold = duplicate_threshold
        self.scene_change_threshold = scene_change_threshold
    
    def build_pipeline(self):
        from decimatr.taggers.blur import BlurTagger
        from decimatr.taggers.hash import HashTagger
        from decimatr.filters.blur import BlurFilter
        from decimatr.filters.duplicate import DuplicateFilter
        from your_module import SceneChangeFilter
        
        return [
            # Taggers
            BlurTagger(),
            HashTagger(),
            
            # Filters
            BlurFilter(threshold=self.base_threshold),
            SceneChangeFilter(threshold=self.scene_change_threshold),
            DuplicateFilter(
                threshold=self.duplicate_threshold,
                buffer_size=50
            )
        ]
```

## Best Practices

### 1. Clear Documentation

Always document your components thoroughly:

```python
class MyComponent:
    """
    One-line summary.
    
    Detailed description of what this component does,
    when to use it, and any important considerations.
    
    Args:
        param1: Description with type and default
        param2: Description with type and default
    
    Tags Produced/Required:
        - tag_name (type): Description
    
    Example:
        >>> component = MyComponent(param1=value)
        >>> result = component.process(data)
    
    Requirements:
        - Requirement 1.2: Description
    """
```

### 2. Type Hints

Use type hints for better IDE support:

```python
from typing import Dict, Any, List
import numpy as np

def compute_tags(self, packet: VideoFramePacket) -> Dict[str, Any]:
    pass

def should_pass(self, packet: VideoFramePacket) -> bool:
    pass
```

### 3. Error Handling

Handle errors gracefully:

```python
def compute_tags(self, packet: VideoFramePacket) -> Dict[str, Any]:
    try:
        # Your computation
        result = compute_metric(packet.frame_data)
        return {"metric": result}
    except Exception as e:
        # Log error and return default
        logger.error(f"Error computing metric: {e}")
        return {"metric": 0.0}
```

### 4. Efficient Computation

Optimize for performance:

```python
# Good: Compute once, use multiple times
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = compute_blur(gray)
entropy = compute_entropy(gray)

# Bad: Convert multiple times
blur = compute_blur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
entropy = compute_entropy(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
```

### 5. Configurable Parameters

Make components configurable:

```python
class MyTagger(Tagger):
    def __init__(
        self,
        threshold: float = 1.0,
        method: str = "default",
        normalize: bool = True
    ):
        self.threshold = threshold
        self.method = method
        self.normalize = normalize
```

### 6. Unit Tests

Write tests for your components:

```python
def test_my_tagger():
    tagger = MyTagger(threshold=1.0)
    
    # Create test packet
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    packet = VideoFramePacket(
        frame_data=frame,
        frame_number=0,
        timestamp=timedelta(0),
        source_video_id="test"
    )
    
    # Test computation
    tags = tagger.compute_tags(packet)
    assert "my_metric" in tags
    assert isinstance(tags["my_metric"], float)
```

## Examples

### Complete Example: Face Detection Pipeline

```python
# face_detection.py
from decimatr.taggers.base import Tagger
from decimatr.filters.base import StatelessFilter
from decimatr.strategies.base import FilterStrategy
from decimatr.scheme import VideoFramePacket
import cv2

class FaceDetectionTagger(Tagger):
    """Detect faces in frames using Haar cascades."""
    
    def __init__(self, min_face_size: tuple = (30, 30)):
        self.min_face_size = min_face_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def compute_tags(self, packet: VideoFramePacket) -> dict:
        gray = cv2.cvtColor(packet.frame_data, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size
        )
        
        return {
            "face_count": len(faces),
            "face_boxes": faces.tolist() if len(faces) > 0 else [],
            "has_faces": len(faces) > 0
        }
    
    @property
    def tag_keys(self) -> list:
        return ["face_count", "face_boxes", "has_faces"]


class FaceFilter(StatelessFilter):
    """Filter frames based on face detection."""
    
    def __init__(self, min_faces: int = 1, max_faces: int = 10):
        self.min_faces = min_faces
        self.max_faces = max_faces
    
    def should_pass(self, packet: VideoFramePacket) -> bool:
        face_count = packet.get_tag("face_count", 0)
        return self.min_faces <= face_count <= self.max_faces
    
    @property
    def required_tags(self) -> list:
        return ["face_count"]


class FaceDetectionStrategy(FilterStrategy):
    """Strategy for selecting frames with faces."""
    
    def __init__(
        self,
        min_faces: int = 1,
        blur_threshold: float = 100.0
    ):
        self.min_faces = min_faces
        self.blur_threshold = blur_threshold
    
    def build_pipeline(self):
        from decimatr.taggers.blur import BlurTagger
        from decimatr.filters.blur import BlurFilter
        
        return [
            BlurTagger(),
            FaceDetectionTagger(),
            BlurFilter(threshold=self.blur_threshold),
            FaceFilter(min_faces=self.min_faces)
        ]


# Usage
from decimatr.core.processor import FrameProcessor

processor = FrameProcessor(strategy=FaceDetectionStrategy())
for frame in processor.process('video.mp4'):
    # Process frames with faces
    face_count = frame.get_tag("face_count")
    print(f"Frame {frame.frame_number}: {face_count} faces")
```

## See Also

- [API Reference](API.md) - Complete API documentation
- [Examples](../examples/) - More code examples
- [README](../README.md) - Getting started guide
