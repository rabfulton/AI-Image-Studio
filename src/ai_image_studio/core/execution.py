"""
Execution Engine - Async workflow execution.

This module provides the execution engine that runs node graphs
asynchronously with progress reporting and cancellation support.

Reference: architecture.md#4-execution-engine
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
from uuid import UUID, uuid4

from ai_image_studio.core.graph import NodeGraph, Node, NodeId, NodeOutput


class ExecutionStatus(Enum):
    """Status of an execution job."""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class ExecutionProgress:
    """Progress information for an execution."""
    job_id: UUID
    status: ExecutionStatus
    current_node: NodeId | None = None
    current_node_name: str = ""
    nodes_completed: int = 0
    nodes_total: int = 0
    message: str = ""
    error: str | None = None
    preview_image: Any | None = None  # ImageData, if provider supports previews
    preview_step: int | None = None
    preview_total_steps: int | None = None
    preview_is_noisy: bool = False
    
    @property
    def progress_percent(self) -> float:
        if self.nodes_total == 0:
            return 0.0
        return (self.nodes_completed / self.nodes_total) * 100


@dataclass
class ExecutionJob:
    """A queued execution job."""
    id: UUID
    graph: NodeGraph
    target_nodes: list[NodeId] | None = None  # None = execute all output nodes
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    results: dict[NodeId, Any] = field(default_factory=dict)
    error: str | None = None


class ExecutionContext:
    """
    Context passed to node executors during execution.
    
    Provides access to:
    - Provider instances for AI operations
    - Cache for intermediate results
    - Cancellation checking
    - Progress reporting
    """
    
    def __init__(
        self,
        job_id: UUID,
        on_progress: Callable[[ExecutionProgress], None] | None = None,
        external_cancelled: Callable[[], bool] | None = None,
    ):
        self.job_id = job_id
        self._on_progress = on_progress
        self._external_cancelled = external_cancelled
        self._cancelled = False
        self._cache: dict[str, Any] = {}
        self._providers: dict[str, Any] = {}
    
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def cancel(self) -> None:
        self._cancelled = True
    
    def check_cancelled(self) -> None:
        """Raise if cancelled."""
        if self._cancelled or (self._external_cancelled and self._external_cancelled()):
            raise asyncio.CancelledError("Execution cancelled")
    
    def report_progress(self, progress: ExecutionProgress) -> None:
        """Report progress to listeners."""
        if self._on_progress:
            self._on_progress(progress)
    
    def get_provider(self, provider_id: str) -> Any:
        """Get a provider instance."""
        return self._providers.get(provider_id)
    
    def set_provider(self, provider_id: str, provider: Any) -> None:
        """Register a provider."""
        self._providers[provider_id] = provider
    
    def cache_get(self, key: str) -> Any | None:
        """Get from cache."""
        return self._cache.get(key)
    
    def cache_set(self, key: str, value: Any) -> None:
        """Store in cache."""
        self._cache[key] = value


class ExecutionEngine:
    """
    Async execution engine for node graphs.
    
    Features:
    - Queued execution with priority
    - Async node execution
    - Progress reporting
    - Cancellation support
    - Result caching
    """
    
    def __init__(self, external_cancelled: Callable[[], bool] | None = None):
        self._queue: list[ExecutionJob] = []
        self._current_job: ExecutionJob | None = None
        self._current_context: ExecutionContext | None = None
        self._is_running = False
        self._lock = asyncio.Lock()
        self._external_cancelled = external_cancelled
        
        # Callbacks
        self._on_progress: Callable[[ExecutionProgress], None] | None = None
        self._on_job_complete: Callable[[ExecutionJob], None] | None = None
    
    def set_progress_callback(
        self,
        callback: Callable[[ExecutionProgress], None],
    ) -> None:
        """Set the progress callback."""
        self._on_progress = callback
    
    def set_completion_callback(
        self,
        callback: Callable[[ExecutionJob], None],
    ) -> None:
        """Set the job completion callback."""
        self._on_job_complete = callback
    
    async def submit(
        self,
        graph: NodeGraph,
        target_nodes: list[NodeId] | None = None,
    ) -> UUID:
        """
        Submit a graph for execution.
        
        Args:
            graph: The node graph to execute
            target_nodes: Specific nodes to execute, or None for all outputs
        
        Returns:
            Job ID for tracking
        """
        job = ExecutionJob(
            id=uuid4(),
            graph=graph,
            target_nodes=target_nodes,
        )
        
        async with self._lock:
            self._queue.append(job)
            job.status = ExecutionStatus.QUEUED
        
        # Start worker if not running
        if not self._is_running:
            asyncio.create_task(self._run_worker())
        
        return job.id
    
    async def cancel(self, job_id: UUID | None = None) -> bool:
        """
        Cancel a job or the current execution.
        
        Args:
            job_id: Specific job to cancel, or None for current
        
        Returns:
            True if cancelled successfully
        """
        async with self._lock:
            # Cancel current job
            if job_id is None or (
                self._current_job and self._current_job.id == job_id
            ):
                if self._current_context:
                    self._current_context.cancel()
                    return True
            
            # Cancel queued job
            for job in self._queue:
                if job.id == job_id:
                    job.status = ExecutionStatus.CANCELLED
                    self._queue.remove(job)
                    return True
        
        return False
    
    async def cancel_all(self) -> None:
        """Cancel all jobs."""
        async with self._lock:
            # Cancel current
            if self._current_context:
                self._current_context.cancel()
            
            # Clear queue
            for job in self._queue:
                job.status = ExecutionStatus.CANCELLED
            self._queue.clear()
    
    def get_queue_status(self) -> list[dict]:
        """Get status of all queued jobs."""
        return [
            {
                "id": str(job.id),
                "status": job.status.name,
                "created_at": job.created_at,
            }
            for job in self._queue
        ]
    
    async def _run_worker(self) -> None:
        """Background worker that processes the queue."""
        self._is_running = True
        
        try:
            while True:
                # Get next job
                async with self._lock:
                    if not self._queue:
                        break
                    job = self._queue.pop(0)
                    self._current_job = job
                
                # Execute
                await self._execute_job(job)
                
                # Notify completion
                if self._on_job_complete:
                    self._on_job_complete(job)
        finally:
            self._is_running = False
            self._current_job = None
            self._current_context = None
    
    async def _execute_job(self, job: ExecutionJob) -> None:
        """Execute a single job."""
        job.status = ExecutionStatus.RUNNING
        job.started_at = time.time()
        
        # Create context
        context = ExecutionContext(
            job_id=job.id,
            on_progress=self._on_progress,
            external_cancelled=self._external_cancelled,
        )
        self._current_context = context
        
        try:
            # Get execution order
            if job.target_nodes:
                # Execute specific nodes and their dependencies
                nodes_to_execute = self._get_dependencies(job.graph, job.target_nodes)
            else:
                # Execute all nodes leading to outputs
                output_nodes = job.graph.get_output_nodes()
                output_ids = [n.id for n in output_nodes]
                nodes_to_execute = self._get_dependencies(job.graph, output_ids)
            
            # Filter to dirty nodes only
            dirty_nodes = {n.id for n in job.graph.get_dirty_nodes()}
            nodes_to_execute = [nid for nid in nodes_to_execute if nid in dirty_nodes]
            
            total = len(nodes_to_execute)
            
            # Report initial progress
            context.report_progress(ExecutionProgress(
                job_id=job.id,
                status=ExecutionStatus.RUNNING,
                nodes_total=total,
                message="Starting execution",
            ))
            
            # Execute nodes in order
            for i, node_id in enumerate(nodes_to_execute):
                context.check_cancelled()
                
                node = job.graph.get_node(node_id)
                if not node:
                    continue
                
                # Report progress
                context.report_progress(ExecutionProgress(
                    job_id=job.id,
                    status=ExecutionStatus.RUNNING,
                    current_node=node_id,
                    current_node_name=node.type_id,
                    nodes_completed=i,
                    nodes_total=total,
                    message=f"Executing {node.type_id}",
                ))
                
                # Execute node
                try:
                    result = await self._execute_node(job.graph, node, context)
                    job.results[node_id] = result
                    
                    # Cache output
                    node.mark_clean(NodeOutput(
                        data=result,
                        timestamp=time.time(),
                        execution_time=0,  # TODO: measure
                    ))
                    
                except asyncio.CancelledError:
                    # Cancellation is not a node error.
                    raise
                except Exception as e:
                    node.mark_error(e)
                    raise
            
            # Success
            job.status = ExecutionStatus.COMPLETED
            job.completed_at = time.time()
            
            context.report_progress(ExecutionProgress(
                job_id=job.id,
                status=ExecutionStatus.COMPLETED,
                nodes_completed=total,
                nodes_total=total,
                message="Execution complete",
            ))
            
        except asyncio.CancelledError:
            job.status = ExecutionStatus.CANCELLED
            job.error = "Cancelled by user"
            job.completed_at = time.time()
            
            context.report_progress(ExecutionProgress(
                job_id=job.id,
                status=ExecutionStatus.CANCELLED,
                message="Execution cancelled",
            ))
            
        except Exception as e:
            job.status = ExecutionStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            
            context.report_progress(ExecutionProgress(
                job_id=job.id,
                status=ExecutionStatus.FAILED,
                error=str(e),
                message=f"Execution failed: {e}",
            ))
    
    async def _execute_node(
        self,
        graph: NodeGraph,
        node: Node,
        context: ExecutionContext,
    ) -> Any:
        """Execute a single node."""
        from ai_image_studio.core.node_types import NodeRegistry
        
        # Get node type
        registry = NodeRegistry.instance()
        node_type = registry.get(node.type_id)
        
        if not node_type:
            # Placeholder execution for demo
            await asyncio.sleep(0.1)  # Simulate work
            return {"output": f"Result from {node.type_id}"}
        
        if not node_type.executor:
            # No executor defined
            await asyncio.sleep(0.1)
            return {}
        
        # Gather inputs from connected nodes
        inputs: dict[str, Any] = {}
        for input_def in node_type.inputs:
            conn = graph.get_input_connection(node.id, input_def.name)
            if conn:
                source_node = graph.get_node(conn.source.node_id)
                if source_node and source_node.cached_output:
                    # Get the specific output
                    output_data = source_node.cached_output.data
                    if isinstance(output_data, dict):
                        inputs[input_def.name] = output_data.get(
                            conn.source.output_name
                        )
                    else:
                        inputs[input_def.name] = output_data
            elif input_def.default_value is not None:
                inputs[input_def.name] = input_def.default_value
        
        # Get parameters
        parameters = node_type.get_default_parameters()
        parameters.update(node.parameters)
        
        # Execute
        result = await node_type.executor(inputs, parameters, context)
        return result
    
    def _get_dependencies(
        self,
        graph: NodeGraph,
        target_nodes: list[NodeId],
    ) -> list[NodeId]:
        """Get all nodes needed to execute targets, in order."""
        # Get all upstream nodes
        all_nodes: set[NodeId] = set(target_nodes)
        for nid in target_nodes:
            all_nodes.update(graph.get_upstream_nodes(nid))
        
        # Get execution order and filter
        try:
            order = graph.get_execution_order()
            return [nid for nid in order if nid in all_nodes]
        except ValueError:
            # Cycle detected - return empty
            return []


# Singleton instance
_engine: ExecutionEngine | None = None


def get_engine() -> ExecutionEngine:
    """Get the singleton execution engine."""
    global _engine
    if _engine is None:
        _engine = ExecutionEngine()
    return _engine
