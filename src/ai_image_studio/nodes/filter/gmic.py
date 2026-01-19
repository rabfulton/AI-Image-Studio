"""
G'MIC Filter Node - Apply G'MIC image filters to images.

This module provides a generic G'MIC filter node that can run
any filter from the filter registry with dynamic parameters.
"""

from __future__ import annotations

from typing import Any

from ai_image_studio.core.node_types import (
    NodeType,
    NodeCategory,
    InputDefinition,
    OutputDefinition,
    ParameterDefinition,
    ParameterType,
)
from ai_image_studio.core.data_types import DataType


async def gmic_filter_executor(
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """
    Execute a G'MIC filter on an input image.
    
    Args:
        inputs: Node inputs (image)
        parameters: Node parameters (filter_id and filter-specific params)
        context: Execution context
        
    Returns:
        Dict with filtered image output
    """
    from ai_image_studio.filters.gmic_runner import GmicRunner, GmicError
    from ai_image_studio.filters.filter_registry import get_filter
    
    # Get input image
    image = inputs.get("image")
    if image is None:
        raise ValueError("No input image provided")
    
    # Get selected filter
    filter_id = parameters.get("filter_id", "blur")
    filter_spec = get_filter(filter_id)
    
    if filter_spec is None:
        raise ValueError(f"Unknown filter: {filter_id}")
    
    # Build filter parameters from node parameters
    filter_params = {}
    for param in filter_spec.params:
        value = parameters.get(param.name)
        if value is not None:
            filter_params[param.name] = value
        else:
            filter_params[param.name] = param.default
    
    # Apply filter
    try:
        runner = GmicRunner()
        result = runner.apply_filter(image, filter_spec.command, filter_params)
        return {"image": result}
        
    except GmicError as e:
        raise RuntimeError(f"G'MIC filter error: {e}") from e


def get_filter_options() -> list[tuple[str, str]]:
    """Get list of available filters as (value, label) tuples."""
    from ai_image_studio.filters.filter_registry import get_filter_registry
    
    registry = get_filter_registry()
    options = []
    
    # Group by category
    by_category: dict[str, list] = {}
    for filter_spec in registry.values():
        if filter_spec.category not in by_category:
            by_category[filter_spec.category] = []
        by_category[filter_spec.category].append(filter_spec)
    
    # Build options with category grouping
    for category in sorted(by_category.keys()):
        for spec in sorted(by_category[category], key=lambda x: x.name):
            options.append((spec.id, f"[{category}] {spec.name}"))
    
    return options


# Build options for filter dropdown
try:
    _filter_options = get_filter_options()
except ImportError:
    _filter_options = [("blur", "[Blur] Gaussian Blur")]


# Register the G'MIC filter node type
GMIC_FILTER_NODE = NodeType(
    id="filter.gmic",
    name="G'MIC Filter",
    description="Apply professional image filters using G'MIC",
    category=NodeCategory.FILTER,
    inputs=[
        InputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Input image to filter",
            required=True,
        ),
    ],
    outputs=[
        OutputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Filtered image",
        ),
    ],
    parameters=[
        ParameterDefinition.enum(
            name="filter_id",
            label="Filter",
            options=_filter_options,
            default="blur",
            description="Select the filter to apply",
        ),
        # Filter-specific parameters are dynamically added by the properties panel
    ],
    executor=gmic_filter_executor,
)


def register_gmic_nodes() -> None:
    """Register all G'MIC filter nodes."""
    from ai_image_studio.core.node_types import register_node
    
    register_node(GMIC_FILTER_NODE)
