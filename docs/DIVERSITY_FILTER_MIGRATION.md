# DiversityFilter Enhancement Migration Guide

This guide helps existing Decimatr users migrate to the enhanced DiversityFilter with tag classification, comparison strategies, and improved CLIP support.

## What's New

The enhanced DiversityFilter introduces several powerful features:

1. **Tag Classification System**: Automatically distinguishes between diversity-suitable tags (hashes, embeddings, histograms) and metric-only tags (blur_score, entropy)
2. **Tag-Specific Comparison Strategies**: Different comparison methods for different tag types (Hamming for hashes, cosine/euclidean for embeddings, histogram metrics for color data)
3. **Improved CLIP Support**: Integration with OpenCLIP library and MobileCLIP for efficient CPU processing
4. **Better Validation**: Clear warnings when diversity tags are unavailable or misconfigured
5. **Weighted Combination Mode**: Optional weighted combination of multiple tag distances

## Backward Compatibility

**Good news**: The enhanced DiversityFilter maintains full backward compatibility. Your existing code will continue to work without changes.

```python
# Old code - still works!
filter = DiversityFilter(
    buffer_size=100,
    min_distance=0.1,
    metric='euclidean'
)
```

However, you may want to update your code to take advantage of the new features.

## Migration Steps

### Step 1: Review Your Diversity Tags

The filter now distinguishes between diversity-suitable and metric-only tags:

**Diversity-Suitable Tags** (good for diversity):
- Perceptual hashes: `dhash`, `phash`, `ahash`
- Embeddings: `clip_embedding`, `model_embedding`
- Color histograms: `color_hist`, `color_histogram`

**Metric-Only Tags** (not suitable for diversity):
- `blur_score` - measures blur, not visual diversity
- `entropy` - measures information content, not visual diversity
- `edge_density` - measures edge content, not visual diversity

**Before (may include metric-only tags):**
```python
filter = DiversityFilter(
    buffer_size=100,
    diversity_tags=['blur_score', 'entropy'],  # These are metric-only!
    min_distance=0.1
)
```

**After (use diversity-suitable tags):**
```python
filter = DiversityFilter(
    buffer_size=100,
    diversity_tags=['dhash', 'clip_embedding'],  # Diversity-suitable
    min_distance=0.1
)
```

### Step 2: Add Appropriate Taggers

Ensure your pipeline includes taggers for the diversity tags you want to use:

**For hash-based diversity:**
```python
from decimatr.taggers.hash import HashTagger
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    HashTagger(hash_type='dhash'),  # Add hash tagger
    DiversityFilter(
        buffer_size=100,
        diversity_tags=['dhash'],
        min_distance=0.1
    )
]
```

**For CLIP embedding diversity:**
```python
from decimatr.taggers.clip import CLIPTagger
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    CLIPTagger(device='cuda'),  # Add CLIP tagger
    DiversityFilter(
        buffer_size=100,
        diversity_tags=['clip_embedding'],
        min_distance=0.1
    )
]
```

### Step 3: Configure Comparison Strategies (Optional)

You can now specify custom comparison strategies for each tag type:

**Basic usage (uses sensible defaults):**
```python
filter = DiversityFilter(
    buffer_size=100,
    diversity_tags=['dhash', 'clip_embedding'],
    min_distance=0.1
)
# Automatically uses:
# - HammingDistanceStrategy for dhash
# - EmbeddingDistanceStrategy(metric='euclidean') for clip_embedding
```

**Advanced usage (custom strategies):**
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

### Step 4: Consider Weighted Combination (Optional)

For multi-tag diversity, you can now use weighted combination:

**Default mode (maximum distance):**
```python
filter = DiversityFilter(
    buffer_size=100,
    diversity_tags=['dhash', 'clip_embedding'],
    min_distance=0.1
)
# Uses max(dhash_distance, clip_embedding_distance)
```

**Weighted mode (combined distance):**
```python
filter = DiversityFilter(
    buffer_size=100,
    diversity_tags=['dhash', 'clip_embedding'],
    min_distance=0.1,
    enable_weighted_combination=True,
    tag_weights={'dhash': 0.3, 'clip_embedding': 0.7}
)
# Uses 0.3 * normalized_dhash + 0.7 * normalized_clip
```

## Common Migration Scenarios

### Scenario 1: Using Blur Score for Diversity

**Problem**: You were using `blur_score` for diversity, but it's now classified as metric-only.

**Before:**
```python
from decimatr.taggers.blur import BlurTagger
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    BlurTagger(),
    DiversityFilter(diversity_tags=['blur_score'], min_distance=10.0)
]
```

**After (use hash-based diversity instead):**
```python
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger
from decimatr.filters.blur import BlurFilter
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    BlurTagger(),
    HashTagger(hash_type='dhash'),
    BlurFilter(threshold=100.0),  # Filter blurry frames
    DiversityFilter(diversity_tags=['dhash'], min_distance=0.1)  # Then select diverse
]
```

**Explanation**: Use `BlurFilter` to remove blurry frames, then use `DiversityFilter` with perceptual hashes to select visually diverse frames.

### Scenario 2: Auto-Detection

**Problem**: You weren't specifying diversity_tags and relied on auto-detection.

**Before:**
```python
filter = DiversityFilter(buffer_size=100, min_distance=0.1)
# Auto-detected all numeric tags (including metric-only tags)
```

**After (same code, better behavior):**
```python
filter = DiversityFilter(buffer_size=100, min_distance=0.1)
# Now auto-detects only diversity-suitable tags (filters out metric-only)
```

**Action**: Test your pipeline to ensure it still selects diverse frames as expected. The filter will now ignore metric-only tags during auto-detection.

### Scenario 3: CLIP Embeddings

**Problem**: You want to use CLIP embeddings for semantic diversity.

**Before (if you had custom CLIP integration):**
```python
# Custom CLIP tagger implementation
```

**After (use built-in CLIPTagger with OpenCLIP):**
```python
from decimatr.taggers.clip import CLIPTagger
from decimatr.filters.diversity import DiversityFilter
from decimatr.filters.comparison_strategies import EmbeddingDistanceStrategy

# GPU mode (uses standard CLIP)
pipeline = [
    CLIPTagger(
        model_name='ViT-B-32',
        pretrained='openai',
        device='cuda'
    ),
    DiversityFilter(
        buffer_size=100,
        diversity_tags=['clip_embedding'],
        min_distance=0.1,
        comparison_strategies={
            'clip_embedding': EmbeddingDistanceStrategy(metric='cosine')
        }
    )
]

# CPU mode (automatically uses MobileCLIP)
pipeline = [
    CLIPTagger(device='cpu'),  # Automatically selects MobileCLIP
    DiversityFilter(
        buffer_size=100,
        diversity_tags=['clip_embedding'],
        min_distance=0.1,
        comparison_strategies={
            'clip_embedding': EmbeddingDistanceStrategy(metric='cosine')
        }
    )
]
```

### Scenario 4: Multiple Diversity Metrics

**Problem**: You want to combine multiple diversity metrics.

**Before (used single metric):**
```python
filter = DiversityFilter(
    buffer_size=100,
    diversity_tags=['dhash'],
    min_distance=0.1
)
```

**After (combine multiple metrics):**
```python
from decimatr.taggers.hash import HashTagger
from decimatr.taggers.clip import CLIPTagger
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    HashTagger(hash_type='dhash'),
    CLIPTagger(device='cuda'),
    DiversityFilter(
        buffer_size=100,
        diversity_tags=['dhash', 'clip_embedding'],
        min_distance=0.1,
        enable_weighted_combination=True,
        tag_weights={'dhash': 0.4, 'clip_embedding': 0.6}
    )
]
```

## Troubleshooting

### Warning: "No diversity-suitable tags found"

**Cause**: The filter couldn't find any diversity-suitable tags in the frame.

**Solution**: Add appropriate taggers to your pipeline before the DiversityFilter:
```python
from decimatr.taggers.hash import HashTagger

pipeline = [
    HashTagger(hash_type='dhash'),  # Add this
    DiversityFilter(...)
]
```

### Warning: "Frame missing required diversity tags"

**Cause**: A frame is missing one or more of the specified diversity_tags.

**Solution**: Ensure all required taggers are in the pipeline and running successfully:
```python
# Make sure taggers come before filter
pipeline = [
    HashTagger(hash_type='dhash'),
    CLIPTagger(device='cuda'),
    DiversityFilter(diversity_tags=['dhash', 'clip_embedding'], ...)
]
```

### ValueError: "metric must be one of..."

**Cause**: Invalid metric parameter.

**Solution**: Use valid metrics: 'euclidean', 'manhattan', or 'cosine':
```python
filter = DiversityFilter(metric='cosine')  # Valid
```

Or use comparison strategies for more control:
```python
from decimatr.filters.comparison_strategies import EmbeddingDistanceStrategy

filter = DiversityFilter(
    comparison_strategies={
        'clip_embedding': EmbeddingDistanceStrategy(metric='cosine')
    }
)
```

### Distance values seem wrong

**Cause**: Different tag types have different distance scales.

**Solution**: Use weighted combination mode to normalize distances:
```python
filter = DiversityFilter(
    diversity_tags=['dhash', 'clip_embedding'],
    min_distance=0.1,
    enable_weighted_combination=True  # Normalizes distances before combining
)
```

## Performance Considerations

### Hash-Based Diversity (Fastest)

```python
from decimatr.taggers.hash import HashTagger
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    HashTagger(hash_type='dhash'),  # Very fast
    DiversityFilter(diversity_tags=['dhash'], min_distance=0.1)
]
```

**Performance**: ~1000 frames/second on CPU
**Use case**: Fast diversity selection based on visual similarity

### CLIP Embedding Diversity (GPU)

```python
from decimatr.taggers.clip import CLIPTagger
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    CLIPTagger(device='cuda', batch_size=32),  # GPU accelerated
    DiversityFilter(diversity_tags=['clip_embedding'], min_distance=0.1)
]
```

**Performance**: ~100-200 frames/second on GPU (batch processing)
**Use case**: Semantic diversity selection (understands content)

### CLIP Embedding Diversity (CPU)

```python
from decimatr.taggers.clip import CLIPTagger
from decimatr.filters.diversity import DiversityFilter

pipeline = [
    CLIPTagger(device='cpu', batch_size=8),  # Uses MobileCLIP
    DiversityFilter(diversity_tags=['clip_embedding'], min_distance=0.1)
]
```

**Performance**: ~20-30 frames/second on CPU (MobileCLIP)
**Use case**: Semantic diversity without GPU

## Best Practices

1. **Choose the right diversity metric**:
   - Use hashes for fast visual similarity
   - Use CLIP embeddings for semantic understanding
   - Use color histograms for color-based diversity

2. **Set appropriate thresholds**:
   - Hash distances: 0.05-0.15 (normalized to [0, 1])
   - CLIP cosine distances: 0.1-0.3
   - CLIP euclidean distances: 1.0-5.0

3. **Use auto-detection for simple cases**:
   ```python
   filter = DiversityFilter(buffer_size=100, min_distance=0.1)
   # Automatically uses available diversity-suitable tags
   ```

4. **Specify tags explicitly for production**:
   ```python
   filter = DiversityFilter(
       buffer_size=100,
       diversity_tags=['dhash'],  # Explicit
       min_distance=0.1
   )
   ```

5. **Combine metrics for robust diversity**:
   ```python
   filter = DiversityFilter(
       diversity_tags=['dhash', 'clip_embedding'],
       min_distance=0.1,
       enable_weighted_combination=True,
       tag_weights={'dhash': 0.3, 'clip_embedding': 0.7}
   )
   ```

## Getting Help

If you encounter issues during migration:

1. Check the warning messages - they provide helpful context
2. Review the [API documentation](API.md) for detailed parameter descriptions
3. See [CUSTOM_COMPONENTS.md](CUSTOM_COMPONENTS.md) for creating custom comparison strategies
4. Open an issue on GitHub with your pipeline configuration

## Summary

The enhanced DiversityFilter provides:
- ✅ Backward compatibility with existing code
- ✅ Automatic tag classification (diversity-suitable vs metric-only)
- ✅ Tag-specific comparison strategies
- ✅ Improved CLIP support with OpenCLIP and MobileCLIP
- ✅ Better validation and error messages
- ✅ Optional weighted combination mode

Most users can migrate by simply updating their diversity_tags to use diversity-suitable tags (hashes, embeddings, histograms) instead of metric-only tags (blur_score, entropy).
