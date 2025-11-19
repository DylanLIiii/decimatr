"""
GPU capability detection and management utilities.

This module provides utilities for detecting GPU availability and managing
GPU dependencies. It allows the library to work with CPU-only dependencies
by default while optionally supporting GPU acceleration when GPU libraries
are installed.
"""

from typing import Any


class GPUCapabilities:
    """
    Detect and report GPU capabilities.

    This class provides methods to check if GPU acceleration is available,
    identify missing GPU dependencies, and retrieve GPU information. Results
    are cached to avoid repeated checks.

    Example:
        >>> if GPUCapabilities.is_available():
        ...     print("GPU acceleration is available")
        ... else:
        ...     missing = GPUCapabilities.get_missing_dependencies()
        ...     print(f"Missing dependencies: {missing}")
    """

    _gpu_available: bool | None = None
    _missing_dependencies: list[str] = []
    _gpu_info: dict[str, Any] | None = None

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if GPU acceleration is available.

        This method checks for the presence of required GPU dependencies
        (torch, torchvision) and verifies that CUDA is available. Results
        are cached after the first check.

        Returns:
            True if GPU acceleration is available, False otherwise

        Example:
            >>> GPUCapabilities.is_available()
            True
        """
        if cls._gpu_available is None:
            cls._check_gpu_dependencies()
        return cls._gpu_available

    @classmethod
    def _check_gpu_dependencies(cls) -> None:
        """
        Check for GPU dependencies and CUDA availability.

        This internal method performs the actual dependency checking and
        updates the cached results. It checks for:
        - torch library
        - torchvision library
        - CUDA runtime availability
        """
        cls._missing_dependencies = []

        # Check for torch
        try:
            import torch

            # Check if CUDA is available
            if not torch.cuda.is_available():
                cls._missing_dependencies.append("CUDA runtime")
        except ImportError:
            cls._missing_dependencies.append("torch")

        # Check for torchvision
        try:
            import torchvision  # noqa: F401
        except ImportError:
            cls._missing_dependencies.append("torchvision")

        # GPU is available only if no dependencies are missing
        cls._gpu_available = len(cls._missing_dependencies) == 0

    @classmethod
    def get_missing_dependencies(cls) -> list[str]:
        """
        Return list of missing GPU dependencies.

        This method returns a list of missing dependencies required for
        GPU acceleration. If GPU is available, returns an empty list.

        Returns:
            List of missing dependency names (e.g., ["torch", "CUDA runtime"])

        Example:
            >>> missing = GPUCapabilities.get_missing_dependencies()
            >>> if missing:
            ...     print(f"Install missing dependencies: {', '.join(missing)}")
        """
        if cls._gpu_available is None:
            cls._check_gpu_dependencies()
        return cls._missing_dependencies.copy()

    @classmethod
    def get_info(cls) -> dict[str, Any]:
        """
        Get detailed GPU information.

        Returns a dictionary containing GPU availability status and, if
        available, detailed information about the GPU hardware and CUDA
        version. Results are cached after the first call.

        Returns:
            Dictionary with GPU information including:
            - gpu_available: bool indicating if GPU is available
            - missing_dependencies: list of missing dependencies
            - cuda_version: CUDA version string (if available)
            - device_count: number of GPU devices (if available)
            - device_name: name of the first GPU device (if available)

        Example:
            >>> info = GPUCapabilities.get_info()
            >>> print(f"GPU Available: {info['gpu_available']}")
            >>> if info['gpu_available']:
            ...     print(f"Device: {info['device_name']}")
            ...     print(f"CUDA Version: {info['cuda_version']}")
        """
        # Return cached info if available
        if cls._gpu_info is not None:
            return cls._gpu_info.copy()

        # Build info dictionary
        info = {
            "gpu_available": cls.is_available(),
            "missing_dependencies": cls.get_missing_dependencies(),
        }

        # Add detailed GPU info if available
        if cls.is_available():
            try:
                import torch

                info.update(
                    {
                        "cuda_version": torch.version.cuda,
                        "device_count": torch.cuda.device_count(),
                        "device_name": (
                            torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
                        ),
                    }
                )
            except Exception:
                # If we can't get detailed info, just mark GPU as unavailable
                info["gpu_available"] = False
                info["error"] = "Failed to retrieve GPU details"

        # Cache the result
        cls._gpu_info = info
        return info.copy()

    @classmethod
    def reset_cache(cls) -> None:
        """
        Reset cached GPU detection results.

        This method clears all cached GPU detection results, forcing a
        fresh check on the next call to is_available() or get_info().
        Useful for testing or when GPU availability might change at runtime.

        Example:
            >>> GPUCapabilities.reset_cache()
            >>> # Next call will perform fresh detection
            >>> GPUCapabilities.is_available()
        """
        cls._gpu_available = None
        cls._missing_dependencies = []
        cls._gpu_info = None
