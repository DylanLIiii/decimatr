# Requirements Document

## Introduction

This document specifies the requirements for refactoring the Decimatr library into a well-architected, easy-to-use infrastructure library for processing video frames and dense sequences. The refactored library will provide a clear separation between stateless frame analysis (tagging), stateful temporal processing (filtering with context), and flexible filtering strategies. The goal is to create a high-performance, intuitive API that handles both immediate frame-level decisions and time-window-based comparisons.

## Glossary

- **FrameProcessor**: The main system that orchestrates video frame processing through tagging, filtering, and selection stages
- **Tagger**: A component that analyzes a single frame and assigns metadata tags without maintaining state
- **Filter**: A component that makes pass/fail decisions on frames, potentially using temporal context from multiple frames
- **FilterStrategy**: An algorithm that determines how to apply filtering logic based on tags and temporal patterns
- **StatelessFilter**: A filter that makes decisions based solely on the current frame's tags without temporal context
- **StatefulFilter**: A filter that maintains a buffer of recent frames to make decisions based on temporal patterns
- **FramePacket**: A standardized data structure containing frame data, metadata, and tags
- **TemporalBuffer**: A data structure that maintains a sliding window of recent frames for temporal analysis
- **ProcessingPipeline**: The ordered sequence of taggers and filters applied to video frames
- **TagRegistry**: A collection of all tags assigned to a frame during processing
- **GPUDependency**: Optional library dependencies required for GPU-accelerated processing (e.g., CUDA, cuDNN, GPU-enabled frameworks)
- **CPUFallback**: The automatic mechanism that switches from GPU to CPU processing when GPU operations fail or are unavailable

## Requirements

### Requirement 1: Frame Tagging System

**User Story:** As a video processing developer, I want to tag frames with computed metrics (blur score, hash value, entropy, etc.), so that I can make filtering decisions based on these tags.

#### Acceptance Criteria

1. WHEN a frame enters THE FrameProcessor, THE FrameProcessor SHALL apply all configured taggers to compute frame metrics
2. THE Tagger SHALL accept a FramePacket as input and SHALL return tag key-value pairs without modifying the frame data
3. THE Tagger SHALL operate statelessly, computing results based solely on the input frame
4. THE FrameProcessor SHALL store all computed tags in the FramePacket's TagRegistry
5. WHERE multiple taggers are configured, THE FrameProcessor SHALL execute taggers in the configured order

### Requirement 2: Stateless Frame Filtering

**User Story:** As a video processing developer, I want to filter frames based on immediate tag values (e.g., blur threshold), so that I can quickly eliminate unsuitable frames without temporal analysis.

#### Acceptance Criteria

1. THE StatelessFilter SHALL accept a FramePacket with tags as input and SHALL return a boolean pass/fail decision
2. THE StatelessFilter SHALL make decisions based solely on the current frame's tags without accessing previous frames
3. WHEN a frame fails a StatelessFilter, THE FrameProcessor SHALL mark the frame as filtered and SHALL exclude it from further processing
4. THE StatelessFilter SHALL support configurable threshold parameters for tag-based decisions
5. WHERE multiple StatelessFilters are configured, THE FrameProcessor SHALL apply filters in sequence and SHALL stop processing if any filter fails

### Requirement 3: Stateful Temporal Filtering

**User Story:** As a video processing developer, I want to filter frames by comparing them with recent frames (e.g., detecting duplicates via hash comparison), so that I can identify temporal patterns and redundancies.

#### Acceptance Criteria

1. THE StatefulFilter SHALL maintain a TemporalBuffer containing recent frames for comparison
2. WHEN a frame enters a StatefulFilter, THE StatefulFilter SHALL compare the frame's tags against tags in the TemporalBuffer
3. THE StatefulFilter SHALL accept configurable buffer size and comparison window parameters
4. WHEN a frame passes a StatefulFilter, THE StatefulFilter SHALL add the frame to its TemporalBuffer
5. THE StatefulFilter SHALL automatically evict the oldest frames when the TemporalBuffer reaches capacity

### Requirement 4: Predefined Filtering Strategies

**User Story:** As a video processing developer, I want to use predefined filtering strategies (e.g., blur removal, duplicate detection, diversity sampling), so that I can quickly implement common video processing patterns.

#### Acceptance Criteria

1. THE FrameProcessor SHALL provide a library of predefined FilterStrategy implementations for common use cases
2. THE FrameProcessor SHALL include a BlurFilterStrategy that filters frames below a blur threshold
3. THE FrameProcessor SHALL include a DuplicateFilterStrategy that removes frames with similar hash values within a temporal window
4. THE FrameProcessor SHALL include a DiversityFilterStrategy that selects frames maximizing tag diversity over a temporal window
5. WHERE a FilterStrategy is configured, THE FrameProcessor SHALL instantiate the appropriate taggers and filters automatically

### Requirement 5: Flexible Pipeline Configuration

**User Story:** As a video processing developer, I want to configure custom processing pipelines with specific taggers and filters, so that I can implement domain-specific video processing logic.

#### Acceptance Criteria

1. THE FrameProcessor SHALL accept a pipeline configuration specifying the ordered sequence of taggers and filters
2. THE FrameProcessor SHALL support mixing stateless and stateful filters in a single pipeline
3. THE FrameProcessor SHALL validate pipeline configuration at initialization and SHALL raise errors for invalid configurations
4. THE FrameProcessor SHALL allow filters to access tags from any previously executed tagger in the pipeline
5. WHERE no pipeline is specified, THE FrameProcessor SHALL use a default pass-through configuration

### Requirement 6: High-Performance Parallel Processing

**User Story:** As a video processing developer, I want the library to leverage multi-CPU and GPU resources for parallel processing, so that I can handle high-resolution videos and real-time streams efficiently.

#### Acceptance Criteria

1. THE FrameProcessor SHALL process frames in a streaming fashion without loading entire videos into memory
2. THE FrameProcessor SHALL use the xoscar Actor Model to distribute frame processing across multiple CPU cores
3. THE Tagger SHALL support GPU acceleration for compute-intensive operations when GPU resources are available
4. THE FrameProcessor SHALL provide configurable parallelism settings for CPU worker count and GPU batch size
5. THE StatefulFilter SHALL maintain TemporalBuffers with O(1) insertion and O(k) comparison where k is the buffer size
6. THE FrameProcessor SHALL support lazy evaluation, computing tags only when required by downstream filters
7. THE FrameProcessor SHALL release frame data from memory after processing when frames are filtered out
8. WHEN multiple frames are available, THE FrameProcessor SHALL batch process frames using actor pools for maximum throughput

### Requirement 7: Simple and Intuitive API

**User Story:** As a video processing developer, I want a simple API to process videos with minimal boilerplate code, so that I can focus on my application logic rather than library mechanics.

#### Acceptance Criteria

1. THE FrameProcessor SHALL provide a single-method interface for processing video files or frame iterators
2. THE FrameProcessor SHALL accept video file paths, frame iterators, or frame lists as input
3. THE FrameProcessor SHALL return an iterator of FramePackets that passed all filters
4. THE FrameProcessor SHALL provide builder methods for common configurations (e.g., `with_blur_filter()`, `with_duplicate_detection()`)
5. THE FrameProcessor SHALL include comprehensive documentation with usage examples for all public APIs

### Requirement 8: Extensibility for Custom Components

**User Story:** As a video processing developer, I want to create custom taggers and filters by extending base classes, so that I can implement domain-specific processing logic.

#### Acceptance Criteria

1. THE FrameProcessor SHALL provide abstract base classes for Tagger, StatelessFilter, and StatefulFilter
2. THE Tagger base class SHALL define a single abstract method for computing tags from a FramePacket
3. THE StatelessFilter base class SHALL define a single abstract method for evaluating a FramePacket
4. THE StatefulFilter base class SHALL provide buffer management utilities and SHALL define an abstract comparison method
5. THE FrameProcessor SHALL accept custom tagger and filter instances in pipeline configurations

### Requirement 9: Comprehensive Error Handling and Logging

**User Story:** As a video processing developer, I want clear error messages and detailed logging, so that I can debug issues and monitor processing performance.

#### Acceptance Criteria

1. WHEN an invalid configuration is provided, THE FrameProcessor SHALL raise a descriptive exception before processing begins
2. WHEN a tagger or filter encounters an error, THE FrameProcessor SHALL log the error with frame context and SHALL continue processing subsequent frames
3. THE FrameProcessor SHALL provide configurable logging levels (DEBUG, INFO, WARNING, ERROR)
4. THE FrameProcessor SHALL log processing statistics including frames processed, filtered, and selected
5. WHERE processing completes, THE FrameProcessor SHALL return a summary dictionary with performance metrics

### Requirement 10: Actor-Based Distributed Processing

**User Story:** As a video processing developer, I want to distribute frame processing across multiple workers using the Actor Model, so that I can achieve maximum throughput and resource utilization.

#### Acceptance Criteria

1. THE FrameProcessor SHALL implement taggers and filters as xoscar actors for distributed execution
2. THE FrameProcessor SHALL provide an ActorPool for managing worker actors across CPU cores
3. WHEN processing frames, THE FrameProcessor SHALL distribute frames to available worker actors in a load-balanced manner
4. THE StatefulFilter SHALL coordinate state synchronization across actor instances using xoscar's messaging system
5. THE FrameProcessor SHALL support graceful shutdown of actor pools and SHALL ensure all in-flight frames are processed
6. THE FrameProcessor SHALL provide monitoring capabilities for actor health and processing throughput
7. WHERE GPU acceleration is enabled, THE FrameProcessor SHALL create dedicated GPU actors with batch processing capabilities

### Requirement 11: CPU and GPU Dependency Isolation

**User Story:** As a video processing developer, I want the library to work with CPU-only dependencies by default and optionally support GPU acceleration when GPU dependencies are installed, so that I can deploy the library in different environments without forcing GPU requirements.

#### Acceptance Criteria

1. THE FrameProcessor SHALL function with CPU-only dependencies without requiring GPU libraries to be installed
2. WHEN GPU dependencies are not installed, THE FrameProcessor SHALL use CPU-based implementations for all processing operations
3. WHEN a user attempts to enable GPU acceleration without GPU dependencies installed, THE FrameProcessor SHALL raise a descriptive error indicating which GPU dependencies are missing
4. WHERE GPU dependencies are installed and GPU acceleration is requested, THE FrameProcessor SHALL automatically detect and use GPU resources
5. THE FrameProcessor SHALL provide a method to check GPU availability and report which GPU dependencies are installed
6. THE Tagger SHALL implement separate CPU and GPU code paths with GPU implementations isolated in optional modules
7. WHEN GPU operations fail, THE FrameProcessor SHALL log a warning with details and SHALL fall back to CPU processing automatically
