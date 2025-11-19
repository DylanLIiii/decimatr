"""
Custom exceptions for the Decimatr library.

This module defines the exception hierarchy for the library, providing
clear error messages and context for different failure scenarios.
"""


class DecimatrError(Exception):
    """
    Base exception for all Decimatr errors.

    All custom exceptions in the Decimatr library inherit from this base
    class, making it easy to catch all library-specific errors.
    """

    pass


class ConfigurationError(DecimatrError):
    """
    Raised when pipeline configuration is invalid.

    This exception is raised during initialization when the pipeline
    configuration contains errors such as:
    - Missing required taggers for filters
    - Invalid component types
    - Conflicting configuration options

    Example:
        >>> raise ConfigurationError("Filter requires 'blur_score' tag but no BlurTagger in pipeline")
    """

    pass


class TagMissingError(DecimatrError):
    """
    Raised when a required tag is missing from a frame packet.

    This exception is raised when a filter attempts to access a tag that
    should have been computed by an earlier tagger in the pipeline but
    is not present in the frame packet's tag registry.

    Example:
        >>> raise TagMissingError("Required tag 'blur_score' not found in frame packet")
    """

    pass


class ProcessingError(DecimatrError):
    """
    Raised when frame processing fails.

    This exception is raised when an error occurs during frame processing
    that prevents the frame from being processed successfully. This could
    include errors in tagger computation or filter evaluation.

    Example:
        >>> raise ProcessingError("Failed to compute blur score: invalid frame data")
    """

    pass


class ActorError(DecimatrError):
    """
    Raised when actor operations fail.

    This exception is raised when errors occur in the actor-based
    distributed processing system, such as:
    - Actor initialization failures
    - Actor communication errors
    - Actor pool management issues

    Example:
        >>> raise ActorError("Failed to create actor pool: insufficient resources")
    """

    pass


class GPUDependencyError(DecimatrError):
    """
    Raised when GPU acceleration is requested but GPU dependencies are missing.

    This exception is raised when a user attempts to enable GPU acceleration
    or use GPU-dependent features (like CLIP embeddings) but the required
    GPU dependencies (torch, torchvision, CUDA) are not installed or available.

    The error message includes helpful installation instructions to guide
    users on how to install the missing dependencies.

    Example:
        >>> raise GPUDependencyError(
        ...     "GPU acceleration requested but dependencies are missing: torch, CUDA runtime. "
        ...     "Install with: pip install decimatr[gpu]"
        ... )
    """

    pass
