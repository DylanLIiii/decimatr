# Parallel Processing with ActorPipeline

## Overview

FrameProcessor supports both single-threaded and parallel processing modes. When `n_workers > 1`, the processor automatically uses ActorPipeline for distributed processing across multiple CPU cores using the xoscar Actor Model.

## Quick Start

### Single-Threaded Processing (Default)

```python
from decimatr.core.processor import FrameProcessor

# Create processor with single-threaded execution
processor = FrameProcessor.with_blur_removal(
    threshold=100.0,
    n_workers=1  # Default: single-threaded
)

# Process video
for frame in processor.process('video.mp4'):
    # Process each frame
    save_frame(frame)
```

### Parallel Processing with Actors

```python
from decimatr.core.processor import FrameProcessor

# Create processor with parallel execution
processor = FrameProcessor.with_blur_removal(
    threshold=100.0,
    n_workers=4  # Use 4 worker actors
)

# Process video (automatically uses ActorPipeline)
for frame in processor.process('video.mp4'):
    # Process each frame
    save_frame(frame)
```

## When to Use Parallel Processing

### Use Single-Threaded (n_workers=1) When:
- Processing small videos (< 100 frames)
- Running on resource-constrained systems
- Debugging or development
- Actor overhead exceeds processing time

### Use Parallel Processing (n_workers>1) When:
- Processing large videos (> 1000 frames)
- CPU-bound operations (blur detection, hash computation)
- Multiple CPU cores available
- Throughput is critical

## Performance Considerations

### Actor Initialization Overhead
- First-time initialization: ~0.5-1.0 seconds
- Each processor instance uses a unique port
- Actors are created lazily when needed
- Automatic cleanup after processing

### Scaling Guidelines
- **n_workers = CPU cores**: Good starting point
- **n_workers = 2-4**: Typical for most workloads
- **n_workers > 8**: May have diminishing returns
- Test with your specific workload

### Memory Usage
- Each actor maintains its own state
- Stateful filters use single actor (shared state)
- Stateless components parallelized across workers
- Frame data released after filtering

## Advanced Usage

### Custom Pipeline with Parallel Processing

```python
from decimatr.core.processor import FrameProcessor
from decimatr.taggers.blur import BlurTagger
from decimatr.taggers.hash import HashTagger
from decimatr.filters.blur import BlurFilter
from decimatr.filters.duplicate import DuplicateFilter

# Create custom pipeline
pipeline = [
    BlurTagger(),
    HashTagger(),
    BlurFilter(threshold=100.0),
    DuplicateFilter(threshold=0.05, buffer_size=50)
]

# Use with parallel processing
processor = FrameProcessor(
    pipeline=pipeline,
    n_workers=4
)

# Process with result summary
frames, result = processor.process('video.mp4', return_result=True)

for frame in frames:
    process_frame(frame)

# Print performance metrics
print(f"Throughput: {result.get_throughput():.1f} fps")
print(f"Selection rate: {result.get_selection_rate():.1f}%")
```

### Using Strategies with Parallel Processing

```python
# Blur removal with parallel processing
processor = FrameProcessor.with_blur_removal(
    threshold=150.0,
    n_workers=4
)

# Duplicate detection with parallel processing
processor = FrameProcessor.with_duplicate_detection(
    threshold=0.02,
    window_size=100,
    n_workers=4
)

# Smart sampling with parallel processing
processor = FrameProcessor.with_smart_sampling(
    n_workers=4
)
```

## Architecture Details

### Processing Flow

```
Single-Threaded (n_workers=1):
Frame → Tagger1 → Tagger2 → Filter1 → Filter2 → Output

Parallel (n_workers>1):
Frame → ActorPipeline → [Worker Pool] → Output
         ↓
    TaggerActor1 (pool of n_workers)
         ↓
    TaggerActor2 (pool of n_workers)
         ↓
    FilterActor1 (pool of n_workers)
         ↓
    StatefulFilterActor2 (single actor)
```

### Actor Types

1. **TaggerActor**: Parallelized across n_workers
   - Stateless computation
   - Each worker processes frames independently
   - Used for: BlurTagger, HashTagger, EntropyTagger

2. **FilterActor**: Parallelized across n_workers
   - Stateless filtering
   - Each worker evaluates frames independently
   - Used for: BlurFilter, EntropyFilter, ThresholdFilter

3. **StatefulFilterActor**: Single actor (maintains state)
   - Stateful filtering with temporal context
   - Single actor ensures consistent state
   - Used for: DuplicateFilter, MotionFilter, DiversityFilter

### Port Management

Each FrameProcessor instance automatically:
- Generates a unique random port (20000-30000)
- Avoids conflicts with other processors
- Cleans up resources after processing
- No manual port management required

## Troubleshooting

### Port Already in Use Error

If you see `OSError: [Errno 98] address already in use`:
- This should not happen with the current implementation
- Each processor uses a unique random port
- If it persists, restart your Python process

### Actor Initialization Fails

If actor initialization fails:
- Check that xoscar is installed: `pip install xoscar`
- Verify sufficient system resources
- Try reducing n_workers
- Check system logs for details

### Performance Not Improving

If parallel processing isn't faster:
- Verify CPU-bound workload (not I/O bound)
- Check CPU utilization during processing
- Try different n_workers values
- Consider single-threaded for small videos

### Memory Issues

If running out of memory:
- Reduce n_workers
- Process video in chunks
- Use streaming mode (default)
- Ensure frames are released after filtering

## Best Practices

1. **Start with n_workers=4** for most workloads
2. **Profile your workload** to find optimal n_workers
3. **Use single-threaded** for debugging
4. **Monitor resource usage** during processing
5. **Test with representative data** before production

## Examples

See `examples/actor_pipeline_demo.py` for complete working examples.

## API Reference

### FrameProcessor Constructor

```python
FrameProcessor(
    pipeline: Optional[List[Union[Tagger, Filter]]] = None,
    strategy: Optional[FilterStrategy] = None,
    n_workers: int = 1,  # Number of worker actors
    use_gpu: bool = False,
    gpu_batch_size: int = 32
)
```

### Builder Methods

All builder methods support the `n_workers` parameter:

```python
FrameProcessor.with_blur_removal(threshold=100.0, n_workers=4)
FrameProcessor.with_duplicate_detection(threshold=0.05, n_workers=4)
FrameProcessor.with_smart_sampling(n_workers=4)
```

## Related Documentation

- [ActorPipeline API](../decimatr/actors/pipeline.py)
- [FrameProcessor API](../decimatr/core/processor.py)
- [xoscar Documentation](https://github.com/xorbitsai/xoscar)
