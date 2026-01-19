"""
Node Type System - Definitions and registry for node types.

This module defines how node types are specified:
- InputDefinition: Describes an input socket
- OutputDefinition: Describes an output socket
- ParameterDefinition: Describes a configurable parameter
- NodeType: Complete definition of a node type
- NodeRegistry: Global registry of available node types

Reference: architecture.md#3-core-data-structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

from ai_image_studio.core.data_types import DataType, ParameterValue


class ParameterType(Enum):
    """Types of node parameters (determines UI widget)."""
    TEXT = "text"               # Single-line text input
    TEXT_MULTILINE = "text_multiline"  # Multi-line text area
    INTEGER = "integer"         # Integer spinner
    FLOAT = "float"             # Float spinner
    BOOLEAN = "boolean"         # Checkbox
    ENUM = "enum"               # Dropdown
    SLIDER = "slider"           # Slider with range
    COLOR = "color"             # Color picker
    FILE_PATH = "file_path"     # File browser
    FOLDER_PATH = "folder_path" # Folder browser
    SEED = "seed"               # Seed input with randomize button
    MODEL = "model"             # Model selector


class NodeCategory(Enum):
    """Categories for organizing nodes in the library."""
    INPUT = "input"
    OUTPUT = "output"
    GENERATION = "generation"
    ENHANCEMENT = "enhancement"
    FILTER = "filter"
    UTILITY = "utility"
    CONDITIONING = "conditioning"
    MASK = "mask"
    CUSTOM = "custom"


@dataclass
class InputDefinition:
    """
    Definition of an input socket on a node.
    
    Attributes:
        name: Socket identifier (used in code)
        label: Display label in UI
        data_type: Type of data accepted
        required: If True, node cannot execute without this input
        default_value: Value to use if not connected
    """
    name: str
    label: str
    data_type: DataType
    required: bool = True
    default_value: Any = None
    description: str = ""


@dataclass
class OutputDefinition:
    """
    Definition of an output socket on a node.
    
    Attributes:
        name: Socket identifier (used in code)
        label: Display label in UI
        data_type: Type of data produced
    """
    name: str
    label: str
    data_type: DataType
    description: str = ""


@dataclass
class EnumOption:
    """A single option in an enum parameter."""
    value: str
    label: str
    description: str = ""


@dataclass
class ParameterDefinition:
    """
    Definition of a configurable parameter on a node.
    
    Parameters are user-editable values that affect node behavior.
    Unlike inputs, they don't come from connections.
    
    Attributes:
        name: Parameter identifier
        label: Display label
        param_type: Type of parameter (determines widget)
        default: Default value
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        step: Step size (for numeric types)
        options: List of options (for enum type)
        file_filter: Filter string for file picker
        description: Tooltip/description text
    """
    name: str
    label: str
    param_type: ParameterType
    default: ParameterValue = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: list[EnumOption] = field(default_factory=list)
    file_filter: str = ""  # e.g., "Images (*.png *.jpg)"
    mode_filter: str | None = None  # For MODEL type: filter by GenerationMode (e.g., "upscale")
    description: str = ""
    
    @classmethod
    def text(
        cls,
        name: str,
        label: str,
        default: str = "",
        multiline: bool = False,
        description: str = "",
    ) -> ParameterDefinition:
        """Factory for text parameter."""
        return cls(
            name=name,
            label=label,
            param_type=ParameterType.TEXT_MULTILINE if multiline else ParameterType.TEXT,
            default=default,
            description=description,
        )
    
    @classmethod
    def integer(
        cls,
        name: str,
        label: str,
        default: int = 0,
        min_value: int | None = None,
        max_value: int | None = None,
        description: str = "",
    ) -> ParameterDefinition:
        """Factory for integer parameter."""
        return cls(
            name=name,
            label=label,
            param_type=ParameterType.INTEGER,
            default=default,
            min_value=min_value,
            max_value=max_value,
            step=1,
            description=description,
        )
    
    @classmethod
    def float_param(
        cls,
        name: str,
        label: str,
        default: float = 0.0,
        min_value: float | None = None,
        max_value: float | None = None,
        step: float = 0.1,
        description: str = "",
    ) -> ParameterDefinition:
        """Factory for float parameter."""
        return cls(
            name=name,
            label=label,
            param_type=ParameterType.FLOAT,
            default=default,
            min_value=min_value,
            max_value=max_value,
            step=step,
            description=description,
        )
    
    @classmethod
    def slider(
        cls,
        name: str,
        label: str,
        default: float = 0.5,
        min_value: float = 0.0,
        max_value: float = 1.0,
        step: float = 0.01,
        description: str = "",
    ) -> ParameterDefinition:
        """Factory for slider parameter."""
        return cls(
            name=name,
            label=label,
            param_type=ParameterType.SLIDER,
            default=default,
            min_value=min_value,
            max_value=max_value,
            step=step,
            description=description,
        )
    
    @classmethod
    def boolean(
        cls,
        name: str,
        label: str,
        default: bool = False,
        description: str = "",
    ) -> ParameterDefinition:
        """Factory for boolean parameter."""
        return cls(
            name=name,
            label=label,
            param_type=ParameterType.BOOLEAN,
            default=default,
            description=description,
        )
    
    @classmethod
    def enum(
        cls,
        name: str,
        label: str,
        options: list[tuple[str, str]],  # [(value, label), ...]
        default: str | None = None,
        description: str = "",
    ) -> ParameterDefinition:
        """Factory for enum parameter."""
        enum_options = [EnumOption(v, l) for v, l in options]
        return cls(
            name=name,
            label=label,
            param_type=ParameterType.ENUM,
            default=default or (options[0][0] if options else None),
            options=enum_options,
            description=description,
        )
    
    @classmethod
    def seed(
        cls,
        name: str = "seed",
        label: str = "Seed",
        default: int = -1,
        description: str = "Random seed (-1 for random)",
    ) -> ParameterDefinition:
        """Factory for seed parameter with randomize button."""
        return cls(
            name=name,
            label=label,
            param_type=ParameterType.SEED,
            default=default,
            min_value=-1,
            max_value=2**32 - 1,
            description=description,
        )
    
    @classmethod
    def file_path(
        cls,
        name: str,
        label: str,
        file_filter: str = "All Files (*)",
        default: str = "",
        description: str = "",
    ) -> ParameterDefinition:
        """Factory for file path parameter."""
        return cls(
            name=name,
            label=label,
            param_type=ParameterType.FILE_PATH,
            default=default,
            file_filter=file_filter,
            description=description,
        )


@runtime_checkable
class NodeExecutor(Protocol):
    """Protocol for node execution functions."""
    
    async def __call__(
        self,
        inputs: dict[str, Any],
        parameters: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        """
        Execute the node.
        
        Args:
            inputs: Connected input values by name
            parameters: Parameter values by name
            context: Execution context with access to providers, cache, etc.
        
        Returns:
            Dictionary of output values by name
        """
        ...


@dataclass
class NodeType:
    """
    Complete definition of a node type.
    
    NodeTypes are templates that define what a node does, its inputs,
    outputs, and parameters. Actual nodes in a graph reference a
    NodeType by its id.
    """
    id: str  # Unique identifier, e.g., "input.prompt"
    name: str  # Display name, e.g., "Prompt"
    category: NodeCategory
    description: str = ""
    
    inputs: list[InputDefinition] = field(default_factory=list)
    outputs: list[OutputDefinition] = field(default_factory=list)
    parameters: list[ParameterDefinition] = field(default_factory=list)
    
    # The actual execution function
    executor: NodeExecutor | None = None
    
    # UI hints
    color: str = "#4a5568"  # Header color
    icon: str = ""  # Icon name or path
    min_width: int = 200
    preview_output: str | None = None  # Output name to show as preview
    
    def get_input(self, name: str) -> InputDefinition | None:
        """Get an input definition by name."""
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return None
    
    def get_output(self, name: str) -> OutputDefinition | None:
        """Get an output definition by name."""
        for out in self.outputs:
            if out.name == name:
                return out
        return None
    
    def get_parameter(self, name: str) -> ParameterDefinition | None:
        """Get a parameter definition by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def get_default_parameters(self) -> dict[str, ParameterValue]:
        """Get default values for all parameters."""
        return {p.name: p.default for p in self.parameters}


class NodeRegistry:
    """
    Global registry of available node types.
    
    Nodes register themselves with the registry, and the UI uses
    the registry to populate the node library.
    """
    
    _instance: NodeRegistry | None = None
    
    def __new__(cls) -> NodeRegistry:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._types = {}
        return cls._instance
    
    @classmethod
    def instance(cls) -> NodeRegistry:
        """Get the singleton instance."""
        return cls()
    
    def __init__(self):
        if not hasattr(self, '_types'):
            self._types: dict[str, NodeType] = {}
    
    def register(self, node_type: NodeType) -> None:
        """Register a node type."""
        self._types[node_type.id] = node_type
    
    def unregister(self, type_id: str) -> NodeType | None:
        """Unregister a node type."""
        return self._types.pop(type_id, None)
    
    def get(self, type_id: str) -> NodeType | None:
        """Get a node type by ID."""
        return self._types.get(type_id)
    
    def get_all(self) -> list[NodeType]:
        """Get all registered node types."""
        return list(self._types.values())
    
    def list_by_category(self, category: NodeCategory) -> list[NodeType]:
        """Get all node types in a category."""
        return [t for t in self._types.values() if t.category == category]
    
    def search(self, query: str) -> list[NodeType]:
        """Search node types by name or description."""
        query = query.lower()
        return [
            t for t in self._types.values()
            if query in t.name.lower() or query in t.description.lower()
        ]
    
    def clear(self) -> None:
        """Remove all registered types (for testing)."""
        self._types.clear()
    
    def __len__(self) -> int:
        return len(self._types)
    
    def __contains__(self, type_id: str) -> bool:
        return type_id in self._types


def register_node(node_type: NodeType) -> NodeType:
    """
    Decorator/function to register a node type.
    
    Can be used as:
        register_node(my_node_type)
    
    Or as part of module initialization.
    """
    NodeRegistry.instance().register(node_type)
    return node_type


def node_type(
    id: str,
    name: str,
    category: NodeCategory,
    description: str = "",
    **kwargs,
) -> Callable[[NodeExecutor], NodeType]:
    """
    Decorator to create and register a node type from an executor function.
    
    Usage:
        @node_type("input.prompt", "Prompt", NodeCategory.INPUT)
        async def prompt_executor(inputs, parameters, context):
            return {"text": parameters["prompt"]}
    """
    def decorator(executor: NodeExecutor) -> NodeType:
        nt = NodeType(
            id=id,
            name=name,
            category=category,
            description=description,
            executor=executor,
            **kwargs,
        )
        register_node(nt)
        return nt
    return decorator
