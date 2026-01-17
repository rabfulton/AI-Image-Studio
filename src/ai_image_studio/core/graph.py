"""
Node Graph Model - Core data structures for the node-based workflow.

This module defines the fundamental building blocks:
- Node: A single processing unit with inputs, outputs, and parameters
- Connection: A link between node outputs and inputs
- NodeGraph: The complete graph containing nodes and connections

Reference: architecture.md#3-core-data-structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NewType
from uuid import UUID, uuid4


# Type aliases for clarity
NodeId = NewType("NodeId", UUID)
ConnectionId = NewType("ConnectionId", UUID)


def new_node_id() -> NodeId:
    """Generate a new unique node ID."""
    return NodeId(uuid4())


def new_connection_id() -> ConnectionId:
    """Generate a new unique connection ID."""
    return ConnectionId(uuid4())


@dataclass
class Point2D:
    """2D point for node positioning on the canvas."""
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)


@dataclass
class Size2D:
    """2D size for node dimensions."""
    width: float = 200.0
    height: float = 100.0


@dataclass
class OutputSocket:
    """Reference to an output socket on a node."""
    node_id: NodeId
    output_name: str


@dataclass
class InputSocket:
    """Reference to an input socket on a node."""
    node_id: NodeId
    input_name: str


@dataclass
class Connection:
    """
    A connection (wire) between two nodes.
    
    Connects an output socket of one node to an input socket of another.
    """
    id: ConnectionId
    source: OutputSocket
    target: InputSocket
    
    @classmethod
    def create(
        cls,
        source_node: NodeId,
        source_output: str,
        target_node: NodeId,
        target_input: str,
    ) -> Connection:
        """Factory method to create a new connection."""
        return cls(
            id=new_connection_id(),
            source=OutputSocket(source_node, source_output),
            target=InputSocket(target_node, target_input),
        )


@dataclass
class NodeOutput:
    """Cached output from a node execution."""
    data: Any
    timestamp: float  # Unix timestamp
    execution_time: float  # Seconds
    memory_used: int = 0  # Bytes


@dataclass
class NodeError:
    """Error information from a failed node execution."""
    message: str
    details: str | None = None
    recoverable: bool = True


@dataclass
class Node:
    """
    A single node in the processing graph.
    
    Nodes have:
    - A unique ID
    - A type (references a NodeType in the registry)
    - Position and size on the canvas
    - Parameter values
    - Cached output from last execution
    - Dirty flag indicating if re-execution is needed
    """
    id: NodeId
    type_id: str  # References NodeType.id in the registry
    position: Point2D = field(default_factory=Point2D)
    size: Size2D = field(default_factory=Size2D)
    
    # User-configured parameters
    parameters: dict[str, Any] = field(default_factory=dict)
    
    # Runtime state (not serialized to project file)
    cached_output: NodeOutput | None = field(default=None, repr=False)
    is_dirty: bool = field(default=True)
    is_executing: bool = field(default=False)
    error: NodeError | None = field(default=None)
    
    @classmethod
    def create(cls, type_id: str, position: Point2D | None = None) -> Node:
        """Factory method to create a new node."""
        return cls(
            id=new_node_id(),
            type_id=type_id,
            position=position or Point2D(),
        )
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter value and mark the node as dirty."""
        if self.parameters.get(name) != value:
            self.parameters[name] = value
            self.mark_dirty()
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.parameters.get(name, default)
    
    def mark_dirty(self) -> None:
        """Mark this node as needing re-execution."""
        self.is_dirty = True
        self.cached_output = None
    
    def mark_clean(self, output: NodeOutput) -> None:
        """Mark this node as executed with the given output."""
        self.is_dirty = False
        self.cached_output = output
        self.error = None
    
    def mark_error(self, error: NodeError) -> None:
        """Mark this node as failed with the given error."""
        self.is_dirty = True
        self.error = error
        self.cached_output = None


@dataclass
class NodeGroup:
    """
    A visual grouping of nodes.
    
    Groups help organize complex workflows and can optionally
    be collapsed into subgraphs.
    """
    id: UUID
    name: str
    color: str = "#4a5568"  # Default gray color
    node_ids: list[NodeId] = field(default_factory=list)
    collapsed: bool = False
    
    @classmethod
    def create(cls, name: str, node_ids: list[NodeId] | None = None) -> NodeGroup:
        """Factory method to create a new group."""
        return cls(
            id=uuid4(),
            name=name,
            node_ids=node_ids or [],
        )


@dataclass
class GraphVariable:
    """
    A variable that can be used in batch processing.
    
    Variables are substituted during execution with values
    from the batch input source.
    """
    name: str  # e.g., "filename", "prompt"
    default_value: Any = None
    description: str = ""


class NodeGraph:
    """
    The complete node graph for a project.
    
    Contains nodes, connections between them, and optional groupings.
    Provides methods for graph manipulation and execution ordering.
    """
    
    def __init__(self, name: str = "Untitled"):
        self.id: UUID = uuid4()
        self.name: str = name
        self._nodes: dict[NodeId, Node] = {}
        self._connections: list[Connection] = []
        self._groups: list[NodeGroup] = []
        self._variables: dict[str, GraphVariable] = {}
    
    # --- Node operations ---
    
    @property
    def nodes(self) -> dict[NodeId, Node]:
        """Get all nodes (read-only view)."""
        return self._nodes.copy()
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self._nodes[node.id] = node
    
    def remove_node(self, node_id: NodeId) -> Node | None:
        """
        Remove a node and all its connections.
        
        Returns the removed node, or None if not found.
        """
        node = self._nodes.pop(node_id, None)
        if node:
            # Remove all connections involving this node
            self._connections = [
                conn for conn in self._connections
                if conn.source.node_id != node_id and conn.target.node_id != node_id
            ]
            # Remove from any groups
            for group in self._groups:
                if node_id in group.node_ids:
                    group.node_ids.remove(node_id)
        return node
    
    def get_node(self, node_id: NodeId) -> Node | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    # --- Connection operations ---
    
    @property
    def connections(self) -> list[Connection]:
        """Get all connections (read-only copy)."""
        return self._connections.copy()
    
    def add_connection(self, connection: Connection) -> bool:
        """
        Add a connection to the graph.
        
        Returns False if the connection would create a cycle or
        if either socket doesn't exist.
        """
        # Verify nodes exist
        if connection.source.node_id not in self._nodes:
            return False
        if connection.target.node_id not in self._nodes:
            return False
        
        # Check for cycles
        if self._would_create_cycle(connection):
            return False
        
        # Remove any existing connection to this input
        # (inputs can only have one connection)
        self._connections = [
            conn for conn in self._connections
            if not (conn.target.node_id == connection.target.node_id and
                    conn.target.input_name == connection.target.input_name)
        ]
        
        self._connections.append(connection)
        
        # Mark target node as dirty
        target_node = self._nodes.get(connection.target.node_id)
        if target_node:
            target_node.mark_dirty()
        
        return True
    
    def remove_connection(self, connection_id: ConnectionId) -> Connection | None:
        """Remove a connection by ID."""
        for i, conn in enumerate(self._connections):
            if conn.id == connection_id:
                removed = self._connections.pop(i)
                # Mark target node as dirty
                target_node = self._nodes.get(removed.target.node_id)
                if target_node:
                    target_node.mark_dirty()
                return removed
        return None
    
    def get_input_connection(
        self, node_id: NodeId, input_name: str
    ) -> Connection | None:
        """Get the connection feeding into a specific input."""
        for conn in self._connections:
            if conn.target.node_id == node_id and conn.target.input_name == input_name:
                return conn
        return None
    
    def get_output_connections(
        self, node_id: NodeId, output_name: str
    ) -> list[Connection]:
        """Get all connections from a specific output."""
        return [
            conn for conn in self._connections
            if conn.source.node_id == node_id and conn.source.output_name == output_name
        ]
    
    # --- Graph analysis ---
    
    def get_execution_order(self) -> list[NodeId]:
        """
        Get nodes in topological order for execution.
        
        Nodes with no dependencies come first, followed by nodes
        that depend on them, and so on.
        
        Returns:
            List of node IDs in execution order.
        
        Raises:
            ValueError: If the graph contains a cycle.
        """
        # Build adjacency list (node -> nodes it depends on)
        dependencies: dict[NodeId, set[NodeId]] = {
            node_id: set() for node_id in self._nodes
        }
        
        for conn in self._connections:
            dependencies[conn.target.node_id].add(conn.source.node_id)
        
        # Kahn's algorithm for topological sort
        result: list[NodeId] = []
        no_deps = [nid for nid, deps in dependencies.items() if not deps]
        
        while no_deps:
            node_id = no_deps.pop(0)
            result.append(node_id)
            
            # Remove this node from other nodes' dependencies
            for nid, deps in dependencies.items():
                if node_id in deps:
                    deps.remove(node_id)
                    if not deps and nid not in result and nid not in no_deps:
                        no_deps.append(nid)
        
        if len(result) != len(self._nodes):
            raise ValueError("Graph contains a cycle")
        
        return result
    
    def get_output_nodes(self) -> list[Node]:
        """
        Get nodes that produce final outputs.
        
        Output nodes are those with no outgoing connections,
        or nodes explicitly marked as outputs.
        """
        nodes_with_outputs = {
            conn.source.node_id for conn in self._connections
        }
        return [
            node for node_id, node in self._nodes.items()
            if node_id not in nodes_with_outputs
        ]
    
    def get_upstream_nodes(self, node_id: NodeId) -> set[NodeId]:
        """Get all nodes that this node depends on (directly or indirectly)."""
        upstream: set[NodeId] = set()
        to_visit = [node_id]
        
        while to_visit:
            current = to_visit.pop()
            for conn in self._connections:
                if conn.target.node_id == current:
                    source_id = conn.source.node_id
                    if source_id not in upstream:
                        upstream.add(source_id)
                        to_visit.append(source_id)
        
        return upstream
    
    def get_downstream_nodes(self, node_id: NodeId) -> set[NodeId]:
        """Get all nodes that depend on this node (directly or indirectly)."""
        downstream: set[NodeId] = set()
        to_visit = [node_id]
        
        while to_visit:
            current = to_visit.pop()
            for conn in self._connections:
                if conn.source.node_id == current:
                    target_id = conn.target.node_id
                    if target_id not in downstream:
                        downstream.add(target_id)
                        to_visit.append(target_id)
        
        return downstream
    
    def invalidate_from(self, node_id: NodeId) -> set[NodeId]:
        """
        Mark a node and all downstream nodes as dirty.
        
        Returns the set of invalidated node IDs.
        """
        to_invalidate = {node_id} | self.get_downstream_nodes(node_id)
        
        for nid in to_invalidate:
            node = self._nodes.get(nid)
            if node:
                node.mark_dirty()
        
        return to_invalidate
    
    def _would_create_cycle(self, connection: Connection) -> bool:
        """Check if adding this connection would create a cycle."""
        # Self-loop check
        if connection.source.node_id == connection.target.node_id:
            return True
        
        # Check if source is reachable from target
        # If we can get from target to source via existing connections,
        # then adding source->target would complete a cycle
        visited: set[NodeId] = set()
        to_visit = [connection.target.node_id]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            
            # Look at all nodes that current can reach (downstream)
            for conn in self._connections:
                if conn.source.node_id == current:
                    if conn.target.node_id == connection.source.node_id:
                        # We can reach the source from the target = cycle!
                        return True
                    to_visit.append(conn.target.node_id)
        
        return False
    
    # --- Dirty node tracking ---
    
    def get_dirty_nodes(self) -> list[Node]:
        """Get all nodes that need re-execution."""
        return [node for node in self._nodes.values() if node.is_dirty]
    
    def get_dirty_execution_order(self) -> list[NodeId]:
        """Get dirty nodes in execution order."""
        exec_order = self.get_execution_order()
        dirty_ids = {node.id for node in self.get_dirty_nodes()}
        return [nid for nid in exec_order if nid in dirty_ids]
    
    # --- Group operations ---
    
    @property
    def groups(self) -> list[NodeGroup]:
        """Get all groups (read-only copy)."""
        return self._groups.copy()
    
    def add_group(self, group: NodeGroup) -> None:
        """Add a group to the graph."""
        self._groups.append(group)
    
    def remove_group(self, group_id: UUID) -> NodeGroup | None:
        """Remove a group (does not remove the nodes)."""
        for i, group in enumerate(self._groups):
            if group.id == group_id:
                return self._groups.pop(i)
        return None
    
    # --- Variable operations ---
    
    @property
    def variables(self) -> dict[str, GraphVariable]:
        """Get all variables (read-only copy)."""
        return self._variables.copy()
    
    def add_variable(self, variable: GraphVariable) -> None:
        """Add or update a variable."""
        self._variables[variable.name] = variable
    
    def remove_variable(self, name: str) -> GraphVariable | None:
        """Remove a variable by name."""
        return self._variables.pop(name, None)
    
    # --- Utility ---
    
    def clear(self) -> None:
        """Remove all nodes, connections, and groups."""
        self._nodes.clear()
        self._connections.clear()
        self._groups.clear()
        self._variables.clear()
    
    def __len__(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)
    
    def __contains__(self, node_id: NodeId) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._nodes
