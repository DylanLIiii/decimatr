# GPU Setup Guide

This guide covers GPU installation, configuration, and troubleshooting for Decimatr.

## Overview

Decimatr supports optional GPU acceleration for compute-intensive operations like CLIP embeddings. GPU support is **completely optional** - the library works perfectly with CPU-only dependencies for most use cases.

## When to Use GPU

### Use GPU When:
- Computing CLIP embeddings for semantic similarity
- Processing very large videos (>10,000 frames)
- Batch processing multiple videos
- Real-time processing requirements

### Use CPU When:
- Basic frame analysis (blur, hash, entropy)
- Small to medium videos (<1,000 frames)
- Development and testing
- Deployment environments without GPU

## Installation

### Step 1: Check GPU Availability

First, verify that your system has a CUDA-capable GPU:

```bash
# Check NVIDIA GPU
nvidia-smi
```

If this command works, you have an NVIDIA GPU with drivers installed.

### Step 2: Install Decimatr with GPU Support

```bash
# Install with GPU dependencies
pip install decimatr[gpu]
```

This installs:
- `torch` (PyTorch with CUDA support)
- `torchvision`
- `ftfy` (text processing for CLIP)
- `regex` (required by CLIP)

### Step 3: Verify Installation

```python
from decimatr.core.processor import FrameProcessor

# Check GPU availability
if FrameProcessor.check_gpu_available():
    print("✓ GPU acceleration available")
    
    # Get detailed info
    info = FrameProcessor.get_gpu_info()
    print(f"CUDA version: {info['cuda_version']}")
    print(f"Device count: {info['device_count']}")
    print(f"Device name: {info['device_name']}")
else:
    print("✗ GPU not available")
    
    # Check what's missing
    from decimatr.gpu_utils import GPUCapabilities
    missing = GPUCapabilities.get_missing_dependencies()
    print(f"Missing dependencies: {missing}")
```

## Manual Installation

If automatic installation doesn't work, install dependencies manually:

### For CUDA 11.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ftfy regex
```

### For CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ftfy regex
```

### For CPU-Only PyTorch

If you want PyTorch but without GPU support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ftfy regex
```

## Using GPU Acceleration

### Basic Usage

```python
from decimatr.core.processor import FrameProcessor
from decimatr.taggers.clip import CLIPTagger
from decimatr.filters.diversity import DiversityFilter

# Create GPU-accelerated pipeline
pipeline = [
    CLIPTagger(device="cuda"),  # Use GPU
    DiversityFilter(window_size=100)
]

processor = FrameProcessor(
    pipeline=pipeline,
    use_gpu=True,
    gpu_batch_size=32
)

# Process video
for frame in processor.process('video.mp4'):
    process_frame(frame)
```

### Auto-Detection

Let Decimatr automatically detect GPU availability:

```python
# Auto-detect GPU (falls back to CPU if unavailable)
clip_tagger = CLIPTagger(device="auto")

# This will use GPU if available, CPU otherwise
processor = FrameProcessor(
    pipeline=[clip_tagger],
    use_gpu=True  # Automatically falls back to CPU if GPU unavailable
)
```

### Batch Size Tuning

Adjust batch size based on GPU memory:

```python
# Small GPU (4GB)
processor = FrameProcessor(
    pipeline=pipeline,
    use_gpu=True,
    gpu_batch_size=16
)

# Medium GPU (8GB)
processor = FrameProcessor(
    pipeline=pipeline,
    use_gpu=True,
    gpu_batch_size=32
)

# Large GPU (16GB+)
processor = FrameProcessor(
    pipeline=pipeline,
    use_gpu=True,
    gpu_batch_size=64
)
```

## GPU vs CPU Performance

### Typical Performance Comparison

For CLIP embeddings on 1000 frames (640x480):

| Configuration | Throughput | Speedup |
|--------------|-----------|---------|
| CPU (single-threaded) | ~5 fps | 1x |
| CPU (4 workers) | ~15 fps | 3x |
| GPU (batch_size=16) | ~80 fps | 16x |
| GPU (batch_size=32) | ~120 fps | 24x |

### When GPU Doesn't Help

GPU acceleration provides minimal benefit for:
- Blur detection (CPU is fast enough)
- Hash computation (CPU is fast enough)
- Entropy calculation (CPU is fast enough)
- Small videos (<100 frames)

## Troubleshooting

### GPU Not Detected

**Problem**: `check_gpu_available()` returns `False`

**Solutions**:

1. **Check NVIDIA drivers**:
   ```bash
   nvidia-smi
   ```
   If this fails, install/update NVIDIA drivers.

2. **Check CUDA installation**:
   ```bash
   nvcc --version
   ```
   If this fails, install CUDA toolkit.

3. **Check PyTorch CUDA support**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

4. **Reinstall PyTorch with CUDA**:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Out of Memory Errors

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
   ```python
   processor = FrameProcessor(
       pipeline=pipeline,
       use_gpu=True,
       gpu_batch_size=8  # Reduce from 32
   )
   ```

2. **Use smaller CLIP model**:
   ```python
   # Instead of ViT-L/14 (large)
   clip_tagger = CLIPTagger(model_name="ViT-B/32")  # Smaller model
   ```

3. **Process in chunks**:
   ```python
   # Process video in chunks
   for chunk_start in range(0, total_frames, 1000):
       chunk_frames = frames[chunk_start:chunk_start+1000]
       for frame in processor.process(chunk_frames):
           process_frame(frame)
   ```

4. **Clear GPU cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Slow GPU Performance

**Problem**: GPU is slower than expected

**Solutions**:

1. **Increase batch size**:
   ```python
   # GPU efficiency improves with larger batches
   processor = FrameProcessor(
       pipeline=pipeline,
       use_gpu=True,
       gpu_batch_size=64  # Increase if memory allows
   )
   ```

2. **Check GPU utilization**:
   ```bash
   # Monitor GPU usage while processing
   watch -n 1 nvidia-smi
   ```
   GPU utilization should be >80% during processing.

3. **Verify CUDA version match**:
   ```python
   import torch
   print(f"PyTorch CUDA: {torch.version.cuda}")
   # Should match your CUDA installation
   ```

4. **Use mixed precision** (advanced):
   ```python
   # Enable automatic mixed precision for faster processing
   import torch
   with torch.cuda.amp.autocast():
       for frame in processor.process('video.mp4'):
           process_frame(frame)
   ```

### Import Errors

**Problem**: `ImportError: cannot import name 'CLIPTagger'`

**Solution**: Install GPU dependencies:
```bash
pip install decimatr[gpu]
```

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install PyTorch:
```bash
pip install torch torchvision
```

### Version Conflicts

**Problem**: Conflicting CUDA versions

**Solution**: Match PyTorch CUDA version to system CUDA:

```bash
# Check system CUDA version
nvcc --version

# Install matching PyTorch
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Best Practices

### 1. Start with CPU

Always develop and test with CPU before enabling GPU:

```python
# Development: CPU-only
processor = FrameProcessor(
    pipeline=pipeline,
    use_gpu=False
)

# Production: GPU-accelerated
processor = FrameProcessor(
    pipeline=pipeline,
    use_gpu=True,
    gpu_batch_size=32
)
```

### 2. Use Auto-Detection

Let Decimatr handle GPU availability:

```python
# Automatically uses GPU if available, CPU otherwise
clip_tagger = CLIPTagger(device="auto")
```

### 3. Monitor GPU Usage

Track GPU utilization and memory:

```bash
# Terminal 1: Run processing
python process_video.py

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

### 4. Batch Size Guidelines

Choose batch size based on GPU memory:

| GPU Memory | Recommended Batch Size |
|-----------|----------------------|
| 4GB | 8-16 |
| 6GB | 16-24 |
| 8GB | 24-32 |
| 12GB | 32-48 |
| 16GB+ | 48-64 |

### 5. Profile Performance

Measure actual performance gains:

```python
import time

# CPU baseline
processor_cpu = FrameProcessor(pipeline=pipeline, use_gpu=False)
start = time.time()
list(processor_cpu.process('video.mp4'))
cpu_time = time.time() - start

# GPU comparison
processor_gpu = FrameProcessor(pipeline=pipeline, use_gpu=True)
start = time.time()
list(processor_gpu.process('video.mp4'))
gpu_time = time.time() - start

print(f"CPU: {cpu_time:.2f}s")
print(f"GPU: {gpu_time:.2f}s")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

## Environment-Specific Setup

### Docker

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install Decimatr with GPU support
RUN pip3 install decimatr[gpu]

# Your application code
COPY . /app
WORKDIR /app

CMD ["python3", "process_video.py"]
```

Run with GPU access:
```bash
docker run --gpus all your-image
```

### Cloud Platforms

#### AWS EC2

1. Launch GPU instance (p3.2xlarge or similar)
2. Use Deep Learning AMI (includes CUDA)
3. Install Decimatr:
   ```bash
   pip install decimatr[gpu]
   ```

#### Google Cloud

1. Create instance with GPU (NVIDIA T4 or similar)
2. Install CUDA drivers:
   ```bash
   curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
   sudo python3 install_gpu_driver.py
   ```
3. Install Decimatr:
   ```bash
   pip install decimatr[gpu]
   ```

#### Azure

1. Create NC-series VM
2. Install NVIDIA drivers
3. Install Decimatr:
   ```bash
   pip install decimatr[gpu]
   ```

### Jupyter Notebooks

```python
# Check GPU in notebook
!nvidia-smi

# Install Decimatr
!pip install decimatr[gpu]

# Use GPU
from decimatr.core.processor import FrameProcessor
processor = FrameProcessor.with_smart_sampling(use_gpu=True)
```

## Advanced Configuration

### Multiple GPUs

```python
# Specify GPU device
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

# Or in code
clip_tagger = CLIPTagger(device="cuda:0")  # First GPU
clip_tagger = CLIPTagger(device="cuda:1")  # Second GPU
```

### Mixed CPU/GPU Pipeline

```python
# Some taggers on CPU, some on GPU
pipeline = [
    BlurTagger(),  # CPU
    HashTagger(),  # CPU
    CLIPTagger(device="cuda"),  # GPU
    BlurFilter(threshold=100.0),
    DiversityFilter(window_size=100)
]

processor = FrameProcessor(pipeline=pipeline, use_gpu=True)
```

### GPU Memory Management

```python
import torch

# Clear cache between videos
torch.cuda.empty_cache()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## FAQ

**Q: Do I need a GPU to use Decimatr?**

A: No! Decimatr works perfectly with CPU-only dependencies. GPU is optional for specific use cases like CLIP embeddings.

**Q: Which GPU do I need?**

A: Any NVIDIA GPU with CUDA support (GTX 1060 or newer). More memory = larger batch sizes = better performance.

**Q: Can I use AMD GPUs?**

A: Currently, only NVIDIA GPUs with CUDA are supported. AMD ROCm support may be added in the future.

**Q: How much faster is GPU?**

A: For CLIP embeddings: 10-30x faster. For basic operations (blur, hash): minimal benefit.

**Q: Can I use CPU and GPU together?**

A: Yes! CPU taggers run on CPU, GPU taggers run on GPU automatically.

**Q: What if GPU runs out of memory?**

A: Reduce `gpu_batch_size` or use a smaller model (e.g., ViT-B/32 instead of ViT-L/14).

## See Also

- [API Reference](API.md) - Complete API documentation
- [Performance Optimizations](PERFORMANCE_OPTIMIZATIONS.md) - Optimization techniques
- [Parallel Processing](PARALLEL_PROCESSING.md) - Actor-based processing
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/) - Official PyTorch guide
