# Decimatr Examples

This directory contains comprehensive examples demonstrating the features of Decimatr.

## Available Examples

### Core Examples

- **`frame_processor_demo.py`** - Basic FrameProcessor usage with various filtering strategies
- **`actor_pipeline_demo.py`** - Parallel processing with ActorPipeline (multi-worker)
- **`gpu_batch_processing_demo.py`** - GPU-accelerated batch processing
- **`performance_optimizations_demo.py`** - Performance optimization features

### Enhanced DiversityFilter Examples

- **`diversity_filter_examples.py`** - Comprehensive examples for the enhanced DiversityFilter

## DiversityFilter Examples

The `diversity_filter_examples.py` file contains 8 comprehensive examples:

### 1. Hash-Based Diversity
Fast, efficient diversity using perceptual hashes (dhash, phash, ahash).
- **Speed**: ~1000 frames/sec
- **Best for**: Near-duplicate detection, visual structure

```python
from decimatr.filters.diversity import DiversityFilter
from decimatr.taggers.hash import HashTagger

pipeline = [
    HashTagger(hash_type="dhash"),
    DiversityFilter(
        buffer_size=50,
        diversity_tags=["dhash"],
        min_distance=0.1
    )
]
```

### 2. CLIP Embedding Diversity
Semantic diversity using CLIP embeddings.
- **Speed**: ~100-200 frames/sec (GPU), ~20-50 frames/sec (CPU)
- **Best for**: Semantic content diversity
- **Requires**: `pip install decimatr[gpu]`

```python
from decimatr.filters.comparison_strategies import EmbeddingDistanceStrategy
from decimatr.filters.diversity import DiversityFilter
from decimatr.taggers.clip import CLIPTagger

pipeline = [
    CLIPTagger(device="auto", batch_size=8),
    DiversityFilter(
        buffer_size=50,
        diversity_tags=["clip_embedding"],
        min_distance=0.15,
        comparison_strategies={
            "clip_embedding": EmbeddingDistanceStrategy(metric="cosine")
        }
    )
]
```

### 3. Color Histogram Diversity
Diversity based on color distribution.
- **Speed**: ~500 frames/sec
- **Best for**: Lighting changes, color shifts

### 4. Combining Multiple Tags
Multi-dimensional diversity (visual + color).
- Uses maximum distance across all tags
- Frame passes if diverse in ANY dimension

### 5. Custom Comparison Strategies
Fine-tuning distance metrics for specific use cases.
- Hamming distance for hashes
- Euclidean/Cosine/Manhattan for embeddings
- Intersection/Chi-square/Bhattacharyya for histograms

### 6. Weighted Combination Mode
Balancing multiple diversity dimensions with weights.
- Prioritize certain aspects over others
- Weighted average of normalized distances

### 7. Auto-Detection
Automatic detection of diversity-suitable tags.
- No explicit configuration needed
- Excludes metric-only tags automatically

### 8. Backward Compatibility
Legacy API still works without modifications.
- Maintains existing behavior
- New features are opt-in

## Running the Examples

### Run All DiversityFilter Examples
```bash
python examples/diversity_filter_examples.py
```

### Run Individual Examples
```bash
# Frame processor demo
python examples/frame_processor_demo.py

# Actor pipeline demo
python examples/actor_pipeline_demo.py

# GPU batch processing (requires GPU dependencies)
python examples/gpu_batch_processing_demo.py

# Performance optimizations
python examples/performance_optimizations_demo.py
```

### Using uv
```bash
# Run with uv
uv run python examples/diversity_filter_examples.py
```

## Requirements

### Core Examples
- Python >= 3.10
- decimatr with dev dependencies: `pip install -e ".[dev]"`

### GPU Examples (CLIP, GPU batch processing)
- GPU dependencies: `pip install -e ".[gpu]"`
- CUDA-capable GPU (optional, falls back to CPU)

## Quick Reference

### DiversityFilter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer_size` | int | 100 | Maximum frames in temporal buffer |
| `diversity_tags` | List[str] | None | Tags to use (None = auto-detect) |
| `min_distance` | float | 0.1 | Minimum distance threshold |
| `metric` | str | "euclidean" | Legacy distance metric |
| `comparison_strategies` | Dict | None | Custom strategies per tag |
| `enable_weighted_combination` | bool | False | Enable weighted mode |
| `tag_weights` | Dict | None | Weights for each tag |

### Comparison Strategies

**Hash Tags**:
- `HammingDistanceStrategy()` - Bit-level comparison

**Embedding Tags**:
- `EmbeddingDistanceStrategy(metric='euclidean')` - L2 distance
- `EmbeddingDistanceStrategy(metric='cosine')` - Cosine distance (default)
- `EmbeddingDistanceStrategy(metric='manhattan')` - L1 distance

**Histogram Tags**:
- `HistogramDistanceStrategy(metric='intersection')` - Histogram intersection (default)
- `HistogramDistanceStrategy(metric='chi_square')` - Chi-square distance
- `HistogramDistanceStrategy(metric='bhattacharyya')` - Bhattacharyya distance

### Tag Classifications

**Diversity-Suitable** (used for filtering):
- `dhash`, `phash`, `ahash` - Perceptual hashes
- `clip_embedding`, `model_embedding` - Embeddings
- `color_hist`, `color_histogram` - Color histograms

**Metric-Only** (excluded from filtering):
- `blur_score` - Blur quality metric
- `entropy` - Image entropy
- `edge_density` - Edge detection metric

## Performance Tips

1. **Hash-based diversity** is fastest (~1000 fps)
2. **GPU acceleration** for CLIP provides 5-10x speedup
3. **MobileCLIP** on CPU is 2-3x faster than standard CLIP
4. **Batch processing** improves GPU utilization
5. **Smaller buffer_size** reduces memory usage
6. **Higher min_distance** selects fewer frames (faster)

## Documentation

For detailed documentation, see:
- [DiversityFilter Examples](../docs/DIVERSITY_FILTER_EXAMPLES.md) - Comprehensive examples guide
- [Migration Guide](../docs/DIVERSITY_FILTER_MIGRATION.md) - Upgrading from previous version
- [API Reference](../docs/API.md) - Complete API documentation
- [Custom Components](../docs/CUSTOM_COMPONENTS.md) - Creating custom taggers and strategies

## Troubleshooting

### No frames selected
- Lower `min_distance` threshold (try 0.05-0.15)

### All frames selected
- Increase `min_distance` threshold
- Ensure diversity taggers are in pipeline

### Missing tags warning
- Add required taggers before the filter in pipeline

### Slow processing
- Use hash-based diversity instead of CLIP
- Enable GPU for CLIP
- Reduce buffer_size
- Use batch processing

## Contributing

To add new examples:
1. Create a new Python file in this directory
2. Follow the existing example structure
3. Add documentation to this README
4. Test with `python examples/your_example.py`
