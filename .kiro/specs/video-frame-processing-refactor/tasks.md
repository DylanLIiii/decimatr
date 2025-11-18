# Implementation Plan: Video Frame Processing Library Refactor

## Overview
This plan refactors Decimatr from its current gate-based architecture into a modern, actor-based video frame processing library with clear separation between tagging (stateless analysis), filtering (decision-making), and orchestration (distributed processing).

**Current State**: The codebase has gates (BlurGate, HashGate, EntropyGate, GridGate), samplers (UniformSampler), buffers (SlidingBuffer, SlidingTopKBuffer), and an Orchestrator that processes videos through a configurable pipeline.

**Target State**: A high-performance, actor-based library with Taggers (stateless frame analysis), Filters (stateless and stateful decision-making), FilterStrategies (predefined pipelines), and ActorPipeline (distributed processing using xoscar). The old architecture will be replaced entirely.

---

## Phase 1: Core Data Models and Base Classes

- [x] 1. Enhance FramePacket data model
  - Extend existing `VideoFramePacket` to support tags registry
  - Add `tags: Dict[str, Any]` field to store computed metrics
  - Add helper methods: `get_tag()`, `has_tags()`, `copy_without_frame_data()`
  - Maintain backward compatibility with existing code
  - _Requirements: 1.1, 1.4_

- [x] 2. Create Tagger base class and interface
  - Create `decimatr/taggers/base.py` with abstract `Tagger` class
  - Define `compute_tags()` abstract method that returns `Dict[str, Any]`
  - Define `tag_keys` property to list produced tag keys
  - Add `supports_gpu`, `requires_gpu`, `is_cloud_based` properties (all default False)
  - Add optional `compute_tags_batch()` method for GPU processing
  - _Requirements: 1.2, 1.3_

- [x] 3. Create Filter base classes
  - Create `decimatr/filters/base.py` with abstract `Filter` class
  - Define `should_pass()` abstract method returning boolean
  - Define `required_tags` property listing needed tag keys
  - Create `StatelessFilter` subclass for immediate decisions
  - Create `StatefulFilter` subclass with buffer management
  - Add `compare_with_history()` abstract method to `StatefulFilter`
  - _Requirements: 2.1, 2.2, 3.1, 3.2_

- [x] 4. Implement TemporalBuffer utility
  - Create `decimatr/core/temporal_buffer.py`
  - Implement sliding window with `deque` (O(1) insertion)
  - Add `add()`, `get_window()`, `find_similar()` methods
  - Support configurable max_size with automatic eviction
  - _Requirements: 3.1, 3.5_

---

## Phase 2: CPU-Based Taggers (Migrate from Gates)

- [x] 5. Implement BlurTagger (migrate from BlurGate)
  - Create `decimatr/taggers/blur.py`
  - Migrate blur score calculation from `BlurGate._calculate_blur_score()`
  - Return `{"blur_score": float}` tag
  - Remove state, make purely functional
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 6. Implement HashTagger (migrate from HashGate)
  - Create `decimatr/taggers/hash.py`
  - Migrate hash calculation from `HashGate._calculate_hash()`
  - Return `{"phash": str, "hash_value": ImageHash}` tags
  - Remove duplicate detection logic (moves to filter)
  - Support configurable hash types (phash, ahash, dhash)
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 7. Implement EntropyTagger (migrate from EntropyGate)
  - Create `decimatr/taggers/entropy.py`
  - Migrate entropy calculation from `EntropyGate`
  - Return `{"entropy": float}` tag
  - Use Shannon entropy formula
  - _Requirements: 1.1, 1.2, 1.3_

---

## Phase 3: Stateless and Stateful Filters

- [x] 9. Implement stateless filters
- [x] 9.1 Create ThresholdFilter (generic)
  - Create `decimatr/filters/threshold.py`
  - Accept tag_key and threshold parameters
  - Support comparison operators (>, <, >=, <=, ==)
  - _Requirements: 2.1, 2.2, 2.4_

- [x] 9.2 Create BlurFilter
  - Create `decimatr/filters/blur.py`
  - Use ThresholdFilter pattern for blur_score
  - Default threshold: 100.0
  - _Requirements: 2.1, 2.2, 2.4_

- [x] 9.3 Create EntropyFilter
  - Create `decimatr/filters/entropy.py`
  - Filter frames below entropy threshold
  - Default threshold: 4.0
  - _Requirements: 2.1, 2.2, 2.4_

- [x] 10. Implement stateful filters
- [x] 10.1 Create DuplicateFilter
  - Create `decimatr/filters/duplicate.py`
  - Migrate duplicate detection logic from HashGate
  - Maintain TemporalBuffer of recent frame hashes
  - Compare current frame hash against buffer
  - Add passing frames to buffer
  - Support configurable hash similarity threshold
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 10.2 Create MotionFilter
  - Create `decimatr/filters/motion.py`
  - Detect scene changes via frame differencing
  - Maintain buffer of recent frames
  - Calculate frame difference metrics
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 10.3 Create DiversityFilter
  - Create `decimatr/filters/diversity.py`
  - Select frames maximizing tag diversity
  - Maintain buffer with diversity scoring
  - Support configurable diversity metrics
  - _Requirements: 3.1, 3.2, 3.3_

---

## Phase 4: FilterStrategy Patterns

- [x] 11. Create FilterStrategy base class
  - Create `decimatr/strategies/base.py`
  - Define abstract `build_pipeline()` method
  - Return list of Taggers and Filters in order
  - _Requirements: 4.1, 5.1_

- [x] 12. Implement predefined strategies
- [x] 12.1 Create BlurRemovalStrategy
  - Create `decimatr/strategies/blur_removal.py`
  - Pipeline: [BlurTagger(), BlurFilter(threshold)]
  - Accept configurable threshold parameter
  - _Requirements: 4.1, 4.2_

- [x] 12.2 Create DuplicateDetectionStrategy
  - Create `decimatr/strategies/duplicate_detection.py`
  - Pipeline: [HashTagger(), DuplicateFilter(threshold, window_size)]
  - Accept configurable threshold and window_size
  - _Requirements: 4.1, 4.3_

- [x] 12.3 Create SmartSamplingStrategy
  - Create `decimatr/strategies/smart_sampling.py`
  - Pipeline: [BlurTagger(), HashTagger(), EntropyTagger(), BlurFilter(), DuplicateFilter(), DiversityFilter()]
  - Combine blur removal, duplicate detection, and diversity
  - _Requirements: 4.1, 4.4_

---

## Phase 5: GPU Capability Management

- [x] 13. Implement GPU detection and isolation
- [x] 13.1 Create GPUCapabilities utility
  - Create `decimatr/utils/gpu_utils.py`
  - Implement `is_available()` to check for torch and CUDA
  - Implement `get_missing_dependencies()` to list missing packages
  - Implement `get_info()` to return GPU details
  - Cache detection results
  - _Requirements: 11.1, 11.2, 11.5_

- [x] 13.2 Create GPUDependencyError exception
  - Add to `decimatr/exceptions.py`
  - Include helpful error message with installation instructions
  - _Requirements: 11.3_

- [x] 13.3 Implement CLIPTagger (optional GPU dependency)
  - Create `decimatr/taggers/clip.py`
  - Check GPU availability in `__init__`
  - Raise GPUDependencyError if GPU requested but unavailable
  - Implement `compute_tags()` for single frame
  - Implement `compute_tags_batch()` for batch processing
  - Return `{"clip_embedding": np.ndarray}` tag
  - Set `supports_gpu=True` and `requires_gpu=False`
  - _Requirements: 1.1, 1.2, 11.3, 11.4, 11.6_

---

## Phase 6: Core FrameProcessor API

- [x] 14. Implement FrameProcessor main API
- [x] 14.1 Create basic FrameProcessor class
  - Create `decimatr/core/processor.py`
  - Accept pipeline or strategy in `__init__`
  - Validate pipeline configuration at initialization
  - Store pipeline components
  - _Requirements: 5.1, 5.2, 5.3, 7.1_

- [x] 14.2 Implement single-threaded processing
  - Implement `process()` method accepting video path, iterator, or list
  - Integrate with existing `load_video_frames()` for video files
  - Apply taggers sequentially to compute tags
  - Apply filters sequentially, short-circuit on failure
  - Yield FramePackets that pass all filters
  - _Requirements: 5.4, 6.1, 7.2, 7.3_

- [x] 14.3 Add builder methods for common configurations
  - Implement `with_blur_removal(threshold)` class method
  - Implement `with_duplicate_detection(threshold)` class method
  - Implement `with_smart_sampling()` class method
  - _Requirements: 7.4_

- [x] 14.4 Add error handling and logging
  - Validate pipeline configuration, raise ConfigurationError
  - Log frame-level errors, continue processing
  - Support configurable logging levels
  - Return ProcessingResult summary with metrics
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

---

## Phase 7: Actor-Based Distributed Processing

- [x] 15. Set up xoscar actor infrastructure. To set it up, alwasy use context7 to read and search this [Xoscar Actor Model](https://github.com/xorbitsai/xoscar)
- [x] 15.1 Add xoscar dependency
  - Update `pyproject.toml` to include `xoscar>=0.3.0`
  - Add to CPU-only dependencies (default)
  - _Requirements: 6.2, 10.1_

- [x] 15.2 Create TaggerActor
  - Create `decimatr/actors/tagger_actor.py`
  - Decorate with `@xo.actor`
  - Accept Tagger instance in `__init__`
  - Implement async `process_frame()` method
  - Update packet tags and return packet
  - _Requirements: 6.2, 10.1, 10.2_

- [x] 15.3 Create FilterActor
  - Create `decimatr/actors/filter_actor.py`
  - Decorate with `@xo.actor`
  - Accept StatelessFilter instance in `__init__`
  - Implement async `process_frame()` method
  - Return packet if passes, None if filtered
  - _Requirements: 6.2, 10.1, 10.2_

- [x] 15.4 Create StatefulFilterActor
  - Create `decimatr/actors/stateful_actor.py`
  - Decorate with `@xo.actor`
  - Accept StatefulFilter instance in `__init__`
  - Maintain TemporalBuffer state
  - Implement async `process_frame()` with history comparison
  - Add passing frames to buffer
  - _Requirements: 6.2, 10.1, 10.2, 10.4_

- [-] 16. Implement ActorPipeline orchestration
- [x] 16.1 Create ActorPipeline class
  - Create `decimatr/actors/pipeline.py`
  - Accept pipeline components and n_workers in `__init__`
  - Implement `initialize()` to create actor pools
  - Create CPU-based actor pools by default (distributed across cores)
  - Create single actor for stateful filters (maintains state)
  - Create actor pools for stateless components
  - _Requirements: 6.2, 6.3, 10.2, 10.3_

- [x] 16.2 Implement frame processing through actors
  - Implement async `process_frame()` method
  - Route frame through pipeline stages sequentially
  - Handle None returns (filtered frames)
  - _Requirements: 6.3, 10.3_

- [x] 16.3 Add graceful shutdown
  - Implement async `shutdown()` method
  - Destroy all actor pools
  - Ensure in-flight frames are processed
  - _Requirements: 10.5_

- [x] 17. Integrate ActorPipeline into FrameProcessor
  - Add `n_workers` parameter to FrameProcessor
  - Create ActorPipeline when n_workers > 1
  - Use actor-based processing for parallel execution
  - Fall back to single-threaded for n_workers=1
  - _Requirements: 6.2, 6.3, 6.4_

---

## Phase 8: GPU Batch Processing (Optional)

- [ ] 18. Implement GPU batch processing actors
- [ ] 18.1 Create GPUBatchProcessor actor
  - Create `decimatr/actors/gpu_actor.py`
  - Decorate with `@xo.actor`
  - Accept GPU-enabled Tagger and batch_size
  - Implement async `add_frame()` to accumulate batch
  - Implement async `process_batch()` for GPU processing
  - Add CPU fallback on GPU failure
  - Track GPU failures and switch to CPU after threshold
  - _Requirements: 6.3, 6.7, 11.7_

- [ ] 18.2 Integrate GPU actors into ActorPipeline
  - Detect GPU-enabled taggers in pipeline
  - Create GPUBatchProcessor for taggers with `requires_gpu=True`
  - Create CPU actor pools for other taggers
  - Add `use_gpu` parameter to FrameProcessor
  - Validate GPU availability when `use_gpu=True`
  - _Requirements: 6.3, 6.7, 11.4_

---

## Phase 9: Performance Optimizations

- [x] 19. Implement lazy evaluation and memory management
  - Add lazy tag computation (compute only when required by filters)
  - Release frame_data from memory after filtering
  - Implement frame batching for throughput optimization
  - _Requirements: 6.6, 6.7_

- [ ] 20. Add monitoring and metrics
  - Create `decimatr/utils/metrics.py`
  - Track frames processed, filtered, selected
  - Track processing time per stage
  - Track actor health and throughput
  - Add metrics to ProcessingResult
  - _Requirements: 9.4, 9.5, 10.6_

---

## Phase 10: Documentation and Examples

- [x] 21. Create comprehensive API documentation
  - Document all public classes and methods
  - Add docstrings with type hints
  - Create usage examples for common scenarios
  - Document GPU installation and setup
  - _Requirements: 7.5, 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 22. Update README with new architecture
  - Explain tagging vs filtering concepts
  - Show basic usage examples
  - Document installation options (CPU-only vs GPU)
  - Add performance benchmarks
  - _Requirements: 7.5_



