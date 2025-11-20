# DiversityFilter Usage Examples

This document provides comprehensive examples for using the enhanced DiversityFilter component in Decimatr. The DiversityFilter has been enhanced with tag classification, tag-specific comparison strategies, and improved validation.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Example 1: Hash-Based Diversity](#example-1-hash-based-diversity)
4. [Example 2: CLIP Embedding Diversity](#example-2-clip-embedding-diversity)
5. [Example 3: Color Histogram Diversity](#example-3-color-histogram-diversity)
6. [Example 4: Combining Multiple Tags](#example-4-combining-multiple-tags)
7. [Example 5: Custom Comparison Strategies](#example-5-custom-comparison-strategies)
8. [Example 6: Weighted Combination Mode](#example-6-weighted-combination-mode)
9. [Example 7: Auto-Detection](#example-7-auto-detection)
10. [Example 8: Backward Compatibility](#example-8-backward-compatibility)
11. [Running the Examples](#running-the-examples)

## Overview

The enhanced DiversityFilter introduces several key features:

- **Tag Classification**: Distinguishes between diversity-suitable tags (hashes, embeddings, histograms) and metric-only tags (blur_score, entropy)
- **Tag-Specific Strategies**: Different comparison methods for different tag types (Hamming for hashes, cosine/Euclidean for embeddings, histogram metrics for color data)
- **Flexible Configuration**: Explicit tag specification or automatic detection
- **Weighted Combination**: Optional weighted averaging of multiple tag distances
- **Backward Compatibility**: Maintains existing API while adding new capabilities

## Quick Start

```python
from decimatr.core.processor import FrameProcessor
from decimatr.filters.diversity import DiversityFilter
from decimatr.taggers.hash import HashTagger

# Simple hash-based diversity filtering
pipeline = [
    HashTagger(hash_type="dhash"),
    DiversityFilter(
        buffer_size=100,
        diversity_tags=["dhash"],
        min_distance=0.1
    )
]

processor = FrameProcessor(pipeline=pipeline)
for frame in processor.process("video.mp4"):
    # Process diverse frames
    save_frame(frame)
```

## Example 1: Hash-Based Diversity

**Use Case**: Fast, efficient diversity filtering using perceptual hashes

**Best For**: Detecting near-duplicate frames or frames with similar visual structure

```python
from decimatr.filters.diversity import DiversityFilter
from decimatr.taggers.hash import HashTagger

pipeline = [
    HashTagger(hash_type="dhash"),  # Compute perceptual hash
    DiversityFilter(
        buffer_size=50,
        diversity_tags=["dhash"],  # Use dhash for diversity
        min_distance=0.1,  # Minimum Hamming distance threshold
    ),
]
```

**Key Points**:
- dhash uses Hamming distance (bit-level comparison)
- Very fast and memory efficient
- Good for detecting near-duplicate frames
- Normalized to [0, 1] range automatically
- Alternative hash types: `phash`, `ahash`

**Performance**: ~1000 frames/sec on typical hardware

## Example 2: CLIP Embedding Diversity

**Use Case**: Semantic diversity filtering using CLIP embeddings

**Best For**: Selecting frames with different semantic content (scenes, objects, activities)

**Requirements**: GPU dependencies (`pip install decimatr[gpu]`)

```python
from decimatr.filters.comparison_strategies import EmbeddingDistanceStrategy
from decimatr.filters.diversity import DiversityFilter
from decimatr.taggers.clip import CLIPTagger

pipeline = [
    CLIPTagger(device="auto", batch_size=8),  # Auto-selects MobileCLIP for CPU
    DiversityFilter(
        buffer_size=50,
        diversity_tags=["clip_embedding"],
        min_distance=0.15,  # Cosine distance threshold
        comparison_strategies={
            "clip_embedding": EmbeddingDistanceStrategy(metric="cosine")
        },
    ),
]
```

**Key Points**:
- CLIP embeddings capture semantic content
- Cosine distance measures semantic similarity
- Automatically uses MobileCLIP on CPU for efficiency
- Best for selecting semantically diverse frames
- GPU acceleration available for faster processing

**Performance**: 
- GPU: ~100-200 frames/sec (batch processing)
- CPU (MobileCLIP): ~20-50 frames/sec

## Example 3: Color Histogram Diversity

**Use Case**: Diversity based on color distribution

**Best For**: Detecting lighting changes, color shifts, or different color palettes

```python
import cv2
import numpy as np
from decimatr.filters.comparison_strategies import HistogramDistanceStrategy
from decimatr.filters.diversity import DiversityFilter
from decimatr.taggers.base import Tagger

class ColorHistogramTagger(Tagger):
    """Compute color histogram for each frame."""
    
    def compute_tags(self, packet):
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(packet.frame_data, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Concatenate histograms
        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        
        return {"color_hist": hist}
    
    @property
    def tag_keys(self):
        return ["color_hist"]

pipeline = [
    ColorHistogramTagger(),
    DiversityFilter(
        buffer_size=50,
        diversity_tags=["color_hist"],
        min_distance=0.2,  # Histogram intersection distance threshold
        comparison_strategies={
            "color_hist": HistogramDistanceStrategy(metric="intersection")
        },
    ),
]
```

**Key Points**:
- Color histograms capture color distribution
- Histogram intersection measures color similarity
- Good for detecting lighting changes or color shifts
- Alternative metrics: `chi_square`, `bhattacharyya`

**Performance**: ~500 frames/sec on typical hardware

## Example 4: Combining Multiple Tags

**Use Case**: Multi-dimensional diversity (visual structure + color)

**Best For**: Ensuring frames are diverse in at least one aspect

```python
from decimatr.filters.comparison_strategies import (
    HammingDistanceStrategy,
    HistogramDistanceStrategy,
)
from decimatr.filters.diversity import DiversityFilter
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger

pipeline = [
    HashTagger(hash_type="dhash"),  # Visual structure
    ColorHistogramTagger(),  # Color distribution
    BlurTagger(),  # Blur score (will be ignored as metric-only)
    DiversityFilter(
        buffer_size=50,
        diversity_tags=["dhash", "color_hist"],  # Use both tags
        min_distance=0.15,
        comparison_strategies={
            "dhash": HammingDistanceStrategy(),
            "color_hist": HistogramDistanceStrategy(metric="intersection"),
        },
    ),
]
```

**Key Points**:
- Uses maximum distance across all tags (default)
- Frame passes if diverse in ANY dimension
- `blur_score` is automatically excluded (metric-only tag)
- Each tag uses its appropriate comparison strategy

**Behavior**: `distance = max(dhash_distance, color_hist_distance)`

## Example 5: Custom Comparison Strategies

**Use Case**: Fine-tuning distance metrics for specific use cases

**Best For**: Optimizing diversity measurement for your specific content

```python
from decimatr.filters.comparison_strategies import (
    EmbeddingDistanceStrategy,
    HammingDistanceStrategy,
)
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    HashTagger(hash_type="phash"),
    MockEmbeddingTagger(),  # Your custom embedding tagger
    DiversityFilter(
        buffer_size=50,
        diversity_tags=["phash", "model_embedding"],
        min_distance=0.15,
        comparison_strategies={
            # Use Hamming distance for hashes (default)
            "phash": HammingDistanceStrategy(),
            # Use Manhattan distance for embeddings (instead of default cosine)
            "model_embedding": EmbeddingDistanceStrategy(metric="manhattan"),
        },
    ),
]
```

**Available Strategies**:

### Hash Tags
- `HammingDistanceStrategy()` [default]

### Embedding Tags
- `EmbeddingDistanceStrategy(metric='euclidean')`
- `EmbeddingDistanceStrategy(metric='cosine')` [default]
- `EmbeddingDistanceStrategy(metric='manhattan')`

### Histogram Tags
- `HistogramDistanceStrategy(metric='intersection')` [default]
- `HistogramDistanceStrategy(metric='chi_square')`
- `HistogramDistanceStrategy(metric='bhattacharyya')`

## Example 6: Weighted Combination Mode

**Use Case**: Balancing multiple diversity dimensions with specific importance weights

**Best For**: When you want to prioritize certain aspects of diversity over others

```python
from decimatr.filters.comparison_strategies import (
    HammingDistanceStrategy,
    HistogramDistanceStrategy,
)
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    HashTagger(hash_type="dhash"),  # Visual structure
    ColorHistogramTagger(),  # Color distribution
    DiversityFilter(
        buffer_size=50,
        diversity_tags=["dhash", "color_hist"],
        min_distance=0.15,
        enable_weighted_combination=True,  # Enable weighted mode
        tag_weights={
            "dhash": 0.3,  # 30% weight on visual structure
            "color_hist": 0.7,  # 70% weight on color distribution
        },
        comparison_strategies={
            "dhash": HammingDistanceStrategy(),
            "color_hist": HistogramDistanceStrategy(metric="intersection"),
        },
    ),
]
```

**Key Points**:
- Weighted combination mode enabled
- Distances are normalized to [0, 1] before combining
- Weights: dhash=0.3, color_hist=0.7
- Final distance = 0.3 × norm(dhash_dist) + 0.7 × norm(color_dist)
- If weights not specified, uses equal weights

**Comparison with Default Mode**:

| Mode | Behavior | Formula |
|------|----------|---------|
| Default (max) | Frame passes if diverse in ANY dimension | `max(dhash_dist, color_dist)` |
| Weighted | Frame passes based on weighted average | `0.3 × norm(dhash_dist) + 0.7 × norm(color_dist)` |

## Example 7: Auto-Detection

**Use Case**: Quick prototyping or adaptive filtering

**Best For**: When you want the filter to adapt to available tags automatically

```python
from decimatr.filters.diversity import DiversityFilter
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger

pipeline = [
    HashTagger(hash_type="dhash"),  # Diversity-suitable
    BlurTagger(),  # Metric-only (will be auto-excluded)
    DiversityFilter(
        buffer_size=50,
        # diversity_tags not specified - will auto-detect
        min_distance=0.15,
    ),
]
```

**Key Points**:
- `diversity_tags` not specified (auto-detection enabled)
- Filter automatically uses 'dhash' (diversity-suitable)
- Filter automatically excludes 'blur_score' (metric-only)

**Tag Classifications**:
- **Diversity-suitable**: dhash, phash, ahash, clip_embedding, model_embedding, color_hist
- **Metric-only**: blur_score, entropy, edge_density

## Example 8: Backward Compatibility

**Use Case**: Upgrading existing code without breaking changes

**Best For**: Maintaining existing pipelines while gaining access to new features

```python
from decimatr.filters.diversity import DiversityFilter
from decimatr.taggers.hash import HashTagger

# Old-style usage (still works)
pipeline = [
    HashTagger(hash_type="dhash"),
    DiversityFilter(
        buffer_size=50,
        min_distance=0.1,
        metric="euclidean",  # Legacy parameter
    ),
]
```

**Key Points**:
- Old API still works (backward compatible)
- Legacy 'metric' parameter supported
- Auto-detection works as before
- New features are opt-in (comparison_strategies, weighted mode)

## Running the Examples

All examples are available in a single runnable file:

```bash
# Run all examples
python examples/diversity_filter_examples.py

# Or run with uv
uv run python examples/diversity_filter_examples.py
```

**Note**: Example 2 (CLIP) requires GPU dependencies:
```bash
pip install decimatr[gpu]
# or
uv pip install -e '.[gpu]'
```

## Performance Comparison

| Method | Speed (frames/sec) | Memory | Best For |
|--------|-------------------|--------|----------|
| Hash-based | ~1000 | Low | Near-duplicates, visual structure |
| CLIP (GPU) | ~100-200 | High | Semantic diversity |
| CLIP (CPU/MobileCLIP) | ~20-50 | Medium | Semantic diversity (no GPU) |
| Color Histogram | ~500 | Low | Color distribution, lighting |
| Combined (Hash + Color) | ~400 | Low | Multi-dimensional diversity |

## Best Practices

1. **Start Simple**: Begin with hash-based diversity for fast prototyping
2. **Use Auto-Detection**: Let the filter detect suitable tags automatically
3. **Explicit Configuration**: Specify `diversity_tags` for production pipelines
4. **Tune Thresholds**: Adjust `min_distance` based on your content and needs
5. **Monitor Performance**: Use `ProcessingResult` metrics to track throughput
6. **GPU Acceleration**: Use CLIP with GPU for semantic diversity at scale
7. **Weighted Mode**: Use weighted combination when you need fine control over multiple dimensions

## Troubleshooting

### No frames selected
- **Cause**: `min_distance` threshold too high
- **Solution**: Lower the threshold (try 0.05-0.15)

### All frames selected
- **Cause**: `min_distance` threshold too low or no diversity tags available
- **Solution**: Increase threshold or add diversity taggers to pipeline

### Missing tags warning
- **Cause**: Specified diversity tags not computed by taggers
- **Solution**: Ensure required taggers are in pipeline before the filter

### Slow processing
- **Cause**: CLIP on CPU or large buffer size
- **Solution**: Use GPU for CLIP, or switch to hash-based diversity

## See Also

- [Migration Guide](DIVERSITY_FILTER_MIGRATION.md) - Upgrading from previous version
- [API Reference](API.md) - Complete API documentation
- [Custom Components](CUSTOM_COMPONENTS.md) - Creating custom taggers and strategies
