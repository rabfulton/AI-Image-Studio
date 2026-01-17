"""
Input Nodes - Nodes that provide input data to the workflow.

These include prompt input, image loading, and mask creation.
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


async def prompt_executor(
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """Execute prompt node - simply passes the text parameter to output."""
    text = parameters.get("text", "")
    return {"text": text}


# Prompt input node
PROMPT_NODE = NodeType(
    id="input.prompt",
    name="Prompt",
    description="Text prompt input for generation",
    category=NodeCategory.INPUT,
    inputs=[],
    outputs=[
        OutputDefinition(
            name="text",
            label="Text",
            data_type=DataType.TEXT,
            description="The prompt text",
        ),
    ],
    parameters=[
        ParameterDefinition.text(
            name="text",
            label="Prompt Text",
            default="",
            multiline=True,
            description="Enter your prompt here",
        ),
    ],
    executor=prompt_executor,
)


async def image_input_executor(
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """Execute image input node - loads image from file."""
    from pathlib import Path
    from ai_image_studio.core.data_types import ImageData
    
    file_path = parameters.get("file", "")
    if not file_path or not Path(file_path).exists():
        raise ValueError(f"Image file not found: {file_path}")
    
    # Load image
    image_data = ImageData.from_file(file_path)
    return {"image": image_data}


# Image input node
IMAGE_INPUT_NODE = NodeType(
    id="input.image",
    name="Load Image",
    description="Load an image from file",
    category=NodeCategory.INPUT,
    inputs=[],
    outputs=[
        OutputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Loaded image",
        ),
    ],
    parameters=[
        ParameterDefinition(
            name="file",
            label="Image File",
            param_type=ParameterType.FILE_PATH,
            default="",
            file_filter="Images (*.png *.jpg *.jpeg *.webp);;All Files (*)",
            description="Path to image file",
        ),
    ],
    executor=image_input_executor,
)


def register_input_nodes():
    """Register all input node types."""
    from ai_image_studio.core.node_types import NodeRegistry
    
    registry = NodeRegistry.instance()
    registry.register(PROMPT_NODE)
    registry.register(IMAGE_INPUT_NODE)
