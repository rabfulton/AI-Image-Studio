"""
Core module - Data structures, execution engine, and project management.

This module provides the fundamental building blocks for AI Image Studio:
- Graph: Node graph data structures
- Data Types: Image, mask, and primitive types
- Node Types: Node definitions and registry
- Project: Project management
"""

from ai_image_studio.core.graph import (
    Connection,
    ConnectionId,
    Node,
    NodeError,
    NodeGraph,
    NodeGroup,
    NodeId,
    NodeOutput,
    OutputSocket,
    InputSocket,
    Point2D,
    Size2D,
    new_connection_id,
    new_node_id,
)

from ai_image_studio.core.data_types import (
    DataType,
    ImageData,
    ImageMetadata,
    MaskData,
    ParameterValue,
)

from ai_image_studio.core.node_types import (
    InputDefinition,
    NodeCategory,
    NodeExecutor,
    NodeRegistry,
    NodeType,
    OutputDefinition,
    ParameterDefinition,
    ParameterType,
    node_type,
    register_node,
)

from ai_image_studio.core.project import (
    Project,
    ProjectManager,
    ProjectSettings,
)


__all__ = [
    # graph.py
    "Connection",
    "ConnectionId",
    "Node",
    "NodeError",
    "NodeGraph",
    "NodeGroup",
    "NodeId",
    "NodeOutput",
    "OutputSocket",
    "InputSocket",
    "Point2D",
    "Size2D",
    "new_connection_id",
    "new_node_id",
    # data_types.py
    "DataType",
    "ImageData",
    "ImageMetadata",
    "MaskData",
    "ParameterValue",
    # node_types.py
    "InputDefinition",
    "NodeCategory",
    "NodeExecutor",
    "NodeRegistry",
    "NodeType",
    "OutputDefinition",
    "ParameterDefinition",
    "ParameterType",
    "node_type",
    "register_node",
    # project.py
    "Project",
    "ProjectManager",
    "ProjectSettings",
]
