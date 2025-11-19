# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-19

### Added

#### Core Processing Engine
- **FrameProcessor**: Main API for video frame processing with support for both single-threaded and actor-based parallel processing
- **ActorPipeline**: Distributed processing orchestration using xoscar Actor Model
- **ProcessingResult**: Comprehensive metrics including throughput, actor health, and error tracking
- **VideoFramePacket**: Data model for frame data, metadata, and tags
- **TemporalBuffer**: Temporal buffering for stateful filter operations

#### Tagging Layer (Frame Analysis)
- **BlurTagger**: Compute blur scores using Laplacian variance
- **HashTagger**: Perceptual hashing for duplicate detection
- **EntropyTagger**: Entropy computation for information content analysis
- **CLIPTagger**: GPU-accelerated CLIP embeddings for semantic analysis

#### Filtering Layer (Decision Making)
- **StatelessFilter**: Base for filters that decide based on current frame only
- **StatefulFilter**: Base for filters that use temporal context
- **BlurFilter**: Remove blurry frames based on blur score threshold
- **DuplicateFilter**: Detect and filter duplicate/similar frames using perceptual hashing
- **DiversityFilter**: Ensure diverse frame selection using CLIP embeddings
- **EntropyFilter**: Filter frames based on information entropy
- **MotionFilter**: Detect scene changes and significant motion
- **ThresholdFilter**: Generic threshold-based filtering

#### Predefined Strategies
- **BlurRemovalStrategy**: Remove blurry frames from video sequences
- **DuplicateDetectionStrategy**: Intelligent duplicate frame detection
- **SmartSamplingStrategy**: Multi-criteria sampling (blur + duplicates + diversity)

#### Performance Features
- **Lazy Evaluation**: Only compute tags required by filters (up to 8x speedup)
- **Memory Release**: Automatic frame data cleanup for filtered frames (70% memory reduction)
- **Parallel Processing**: Actor-based distributed processing with configurable worker count
- **GPU Acceleration**: Optional GPU support for CLIP embeddings and batch processing

#### Infrastructure
- **GPU Capabilities**: GPU detection and management utilities
- **Exception Hierarchy**: Custom exceptions for error handling (DecimatrError, TagMissingError, ProcessingError, ActorError, GPUDependencyError)
- **Video Loader**: Video frame loading utilities using decord

#### Documentation
- Comprehensive README with architecture overview and quick start
- API reference documentation
- Custom components guide
- GPU setup guide
- Parallel processing guide
- Performance optimization guide
- Working examples and demos

#### Examples
- `actor_pipeline_demo.py`: Demonstrates actor-based parallel processing
- `frame_processor_demo.py`: Shows basic FrameProcessor usage
- `performance_optimizations_demo.py`: Performance tuning examples

### Technical Details
- Python 3.10+ support
- Actor-based parallelism using xoscar framework
- Clean separation of concerns: tagging (analysis) vs filtering (decision-making)
- Automatic port allocation (20000-30000) for actor communication
- Comprehensive test suite with 15+ test files

[unreleased]: https://github.com/DylanLIiii/decimatr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DylanLIiii/decimatr/releases/tag/v0.1.0
