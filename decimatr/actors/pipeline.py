"""
ActorPipeline for distributed frame processing orchestration.

This module provides the ActorPipeline class that manages actor-based
distributed processing of video frames across multiple CPU cores using xoscar.
"""

import logging

import xoscar as xo

from decimatr.actors.filter_actor import FilterActor
from decimatr.actors.stateful_actor import StatefulFilterActor
from decimatr.actors.tagger_actor import TaggerActor
from decimatr.filters.base import Filter, StatefulFilter, StatelessFilter
from decimatr.scheme import VideoFramePacket
from decimatr.taggers.base import Tagger

logger = logging.getLogger(__name__)


class ActorPipeline:
    """
    Manages actor-based pipeline execution for distributed frame processing.

    ActorPipeline orchestrates the distributed processing of video frames
    through a pipeline of taggers and filters using the xoscar actor framework.
    It creates actor pools for parallel execution on CPU cores and manages
    the routing of frames through pipeline stages.

    By default, the pipeline creates CPU-based actor pools distributed across
    available cores for maximum throughput. Stateless components (taggers and
    stateless filters) are parallelized across multiple actors, while stateful
    filters use a single actor to maintain consistent temporal state.

    Attributes:
        pipeline: List of taggers and filters to execute in order
        n_workers: Number of CPU worker actors for parallel processing
        use_gpu: Whether GPU acceleration is enabled (for future GPU support)
        actor_pools: Dictionary mapping stage names to actor references
        address: xoscar actor pool address

    Example:
        >>> # Create pipeline with taggers and filters
        >>> pipeline = [
        ...     BlurTagger(),
        ...     HashTagger(),
        ...     BlurFilter(threshold=100.0),
        ...     DuplicateFilter(threshold=0.05, buffer_size=50)
        ... ]
        >>>
        >>> # Initialize actor pipeline
        >>> actor_pipeline = ActorPipeline(pipeline, n_workers=4)
        >>> await actor_pipeline.initialize()
        >>>
        >>> # Process frames
        >>> result = await actor_pipeline.process_frame(packet)
        >>>
        >>> # Shutdown gracefully
        >>> await actor_pipeline.shutdown()
    """

    def __init__(
        self,
        pipeline: list[Tagger | Filter],
        n_workers: int = 4,
        use_gpu: bool = False,
        address: str = "127.0.0.1:13527",
    ):
        """
        Initialize ActorPipeline with pipeline configuration.

        Args:
            pipeline: Ordered list of taggers and filters to execute
            n_workers: Number of CPU worker actors for parallel processing.
                      Stateless components are distributed across n_workers,
                      while stateful filters use a single actor.
            use_gpu: Enable GPU acceleration (reserved for future GPU support)
            address: xoscar actor pool address for distributed processing

        Raises:
            ValueError: If pipeline is empty or n_workers is less than 1
            TypeError: If pipeline contains invalid component types
        """
        if not pipeline:
            raise ValueError("Pipeline cannot be empty")

        if n_workers < 1:
            raise ValueError(f"n_workers must be at least 1, got {n_workers}")

        # Validate pipeline components
        for i, component in enumerate(pipeline):
            if not isinstance(component, Tagger | Filter):
                raise TypeError(
                    f"Pipeline component at index {i} must be a Tagger or Filter, "
                    f"got {type(component)}"
                )

        self.pipeline = pipeline
        self.n_workers = n_workers
        self.use_gpu = use_gpu
        self.address = address
        self.actor_pools = {}
        self._initialized = False

        logger.info(
            f"ActorPipeline created with {len(pipeline)} stages, "
            f"{n_workers} workers, GPU={'enabled' if use_gpu else 'disabled'}"
        )

    async def initialize(self) -> None:
        """
        Initialize actor pools for pipeline stages.

        Creates xoscar actor pools for distributed processing:
        - CPU-based actor pools by default (distributed across cores)
        - Actor pools for stateless components (taggers, stateless filters)
        - Single actor for stateful filters (maintains temporal state)

        This method must be called before processing frames. It sets up the
        xoscar actor system and creates all necessary actors for the pipeline.

        Raises:
            RuntimeError: If already initialized
            Exception: If actor pool creation fails

        Example:
            >>> actor_pipeline = ActorPipeline(pipeline, n_workers=4)
            >>> await actor_pipeline.initialize()
        """
        if self._initialized:
            raise RuntimeError("ActorPipeline already initialized")

        logger.info(f"Initializing ActorPipeline with address {self.address}")

        try:
            # Create main actor pool for distributed processing
            await xo.create_actor_pool(address=self.address, n_process=self.n_workers)
            logger.info(f"Created actor pool with {self.n_workers} processes")

            # Create actors for each pipeline stage
            for i, component in enumerate(self.pipeline):
                stage_name = f"stage_{i}"

                if isinstance(component, Tagger):
                    # Create actor pool for taggers (parallel execution)
                    logger.debug(
                        f"Creating TaggerActor pool for stage {i}: {component.__class__.__name__}"
                    )
                    actor_ref = await xo.create_actor(
                        TaggerActor, component, address=self.address, uid=f"tagger_{i}"
                    )
                    self.actor_pools[stage_name] = actor_ref

                elif isinstance(component, StatefulFilter):
                    # Single actor for stateful filters (maintains state)
                    logger.debug(
                        f"Creating StatefulFilterActor for stage {i}: "
                        f"{component.__class__.__name__}"
                    )
                    actor_ref = await xo.create_actor(
                        StatefulFilterActor,
                        component,
                        address=self.address,
                        uid=f"stateful_filter_{i}",
                    )
                    self.actor_pools[stage_name] = actor_ref

                elif isinstance(component, StatelessFilter):
                    # Actor pool for stateless filters (parallel execution)
                    logger.debug(
                        f"Creating FilterActor pool for stage {i}: {component.__class__.__name__}"
                    )
                    actor_ref = await xo.create_actor(
                        FilterActor, component, address=self.address, uid=f"filter_{i}"
                    )
                    self.actor_pools[stage_name] = actor_ref

                else:
                    raise TypeError(f"Unknown component type at stage {i}: {type(component)}")

            self._initialized = True
            logger.info(
                f"ActorPipeline initialized successfully with {len(self.actor_pools)} stages"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ActorPipeline: {e}")
            # Attempt cleanup on failure
            await self._cleanup_actors()
            raise

    async def process_frame(self, packet: VideoFramePacket) -> VideoFramePacket | None:
        """
        Process a single frame through the actor pipeline.

        Routes the frame through all pipeline stages sequentially:
        1. Taggers compute and add tags to the packet
        2. Filters evaluate the packet and may filter it out
        3. Returns the packet if it passes all filters, None if filtered

        The method handles None returns from filters (filtered frames) by
        short-circuiting and returning None immediately.

        Args:
            packet: VideoFramePacket to process through the pipeline

        Returns:
            VideoFramePacket if it passes all filters, None if filtered out

        Raises:
            RuntimeError: If pipeline not initialized
            TypeError: If packet is not a VideoFramePacket
            Exception: If actor processing fails

        Example:
            >>> packet = VideoFramePacket(frame_data=frame, ...)
            >>> result = await actor_pipeline.process_frame(packet)
            >>> if result is not None:
            ...     print(f"Frame {result.frame_number} passed all filters")
            ... else:
            ...     print("Frame was filtered out")
        """
        if not self._initialized:
            raise RuntimeError("ActorPipeline not initialized. Call initialize() first.")

        if not isinstance(packet, VideoFramePacket):
            raise TypeError(f"packet must be a VideoFramePacket, got {type(packet)}")

        current_packet = packet

        # Process through each pipeline stage
        for i, component in enumerate(self.pipeline):
            # Short-circuit if frame was filtered out
            if current_packet is None:
                break

            stage_name = f"stage_{i}"
            actor_ref = self.actor_pools[stage_name]

            try:
                # Route frame to appropriate actor
                if isinstance(component, Tagger):
                    # Tagger adds tags to packet
                    current_packet = await actor_ref.process_frame(current_packet)

                elif isinstance(component, Filter):
                    # Filter returns packet or None
                    current_packet = await actor_ref.process_frame(current_packet)

            except Exception as e:
                logger.error(
                    f"Error processing frame {packet.frame_number} "
                    f"at stage {i} ({component.__class__.__name__}): {e}"
                )
                raise

        return current_packet

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all actors and clean up resources.

        Destroys all actor pools and ensures in-flight frames are processed
        before shutdown. This method should be called when processing is
        complete to properly clean up xoscar resources.

        After shutdown, the pipeline cannot be used again. Create a new
        ActorPipeline instance if needed.

        Raises:
            RuntimeError: If pipeline not initialized

        Example:
            >>> await actor_pipeline.shutdown()
        """
        if not self._initialized:
            raise RuntimeError("ActorPipeline not initialized")

        logger.info("Shutting down ActorPipeline")

        await self._cleanup_actors()

        self._initialized = False
        logger.info("ActorPipeline shutdown complete")

    async def _cleanup_actors(self) -> None:
        """
        Internal method to destroy all actors and clean up resources.

        This method is called during shutdown or when initialization fails.
        It attempts to destroy all created actors gracefully.
        """
        # Destroy all actors
        for stage_name, actor_ref in self.actor_pools.items():
            try:
                await xo.destroy_actor(actor_ref)
                logger.debug(f"Destroyed actor for {stage_name}")
            except Exception as e:
                logger.warning(f"Error destroying actor for {stage_name}: {e}")

        self.actor_pools.clear()

        # Note: xoscar doesn't have a global stop() method
        # Actor pools are cleaned up when actors are destroyed
        logger.debug("Actor cleanup complete")

    def is_initialized(self) -> bool:
        """
        Check if the pipeline is initialized and ready to process frames.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized

    def get_stage_count(self) -> int:
        """
        Get the number of stages in the pipeline.

        Returns:
            Number of pipeline stages (taggers + filters)
        """
        return len(self.pipeline)

    def get_pipeline_info(self) -> dict:
        """
        Get information about the pipeline configuration.

        Returns:
            Dictionary with pipeline details including stages, workers, and status
        """
        stages = []
        for i, component in enumerate(self.pipeline):
            stage_info = {
                "index": i,
                "type": component.__class__.__name__,
                "category": "tagger" if isinstance(component, Tagger) else "filter",
            }

            if isinstance(component, StatefulFilter):
                stage_info["stateful"] = True
                stage_info["buffer_size"] = component.buffer_size
            else:
                stage_info["stateful"] = False

            stages.append(stage_info)

        return {
            "initialized": self._initialized,
            "n_workers": self.n_workers,
            "use_gpu": self.use_gpu,
            "address": self.address,
            "stage_count": len(self.pipeline),
            "stages": stages,
        }
