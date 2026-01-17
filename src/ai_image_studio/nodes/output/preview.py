"""
Output Nodes - Nodes that display or save generated content.

These include Preview (display in Output Studio), Save Image, and similar.
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


async def preview_executor(
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """
    Execute preview node - passes image through and marks for display.
    
    The actual display is handled by the UI when it receives the result.
    This executor simply passes the image data through for the result collector.
    """
    image = inputs.get("image")
    if image is None:
        return {"displayed": False, "error": "No image input connected"}
    
    # The image is passed through - UI will handle display
    return {
        "displayed": True,
        "image": image,  # Pass through for result handling
    }


# Preview node - displays image in Output Studio
PREVIEW_NODE = NodeType(
    id="output.preview",
    name="Preview",
    description="Display image in Output Studio",
    category=NodeCategory.OUTPUT,
    inputs=[
        InputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Image to display",
            required=True,
        ),
    ],
    outputs=[],  # Terminal node - no outputs
    parameters=[
        ParameterDefinition.text(
            name="label",
            label="Label",
            default="Preview",
            description="Label for this preview",
        ),
    ],
    executor=preview_executor,
)


async def save_image_executor(
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """Execute save image node - saves image to file."""
    from pathlib import Path
    
    image = inputs.get("image")
    if image is None:
        return {"saved": False, "error": "No image input connected"}
    
    # Get save path
    output_dir = parameters.get("output_dir", "")
    filename = parameters.get("filename", "output.png")
    
    if not output_dir:
        # Use default output directory
        output_dir = str(Path.home() / "Pictures" / "AI_Image_Studio")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the image
    save_path = output_path / filename
    
    # Handle ImageData or PIL Image
    if hasattr(image, 'to_pil'):
        pil_img = image.to_pil()
    elif hasattr(image, 'save'):
        pil_img = image
    else:
        return {"saved": False, "error": "Invalid image format"}
    
    pil_img.save(str(save_path))
    
    return {
        "saved": True,
        "path": str(save_path),
        "image": image,  # Pass through
    }


# Save Image node
SAVE_IMAGE_NODE = NodeType(
    id="output.save_image",
    name="Save Image",
    description="Save image to file",
    category=NodeCategory.OUTPUT,
    inputs=[
        InputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Image to save",
            required=True,
        ),
    ],
    outputs=[],  # Terminal node
    parameters=[
        ParameterDefinition(
            name="output_dir",
            label="Output Directory",
            param_type=ParameterType.FOLDER_PATH,
            default="",
            description="Directory to save images (default: ~/Pictures/AI_Image_Studio)",
        ),
        ParameterDefinition.text(
            name="filename",
            label="Filename",
            default="output.png",
            description="Output filename",
        ),
    ],
    executor=save_image_executor,
)


def register_output_nodes():
    """Register all output node types."""
    from ai_image_studio.core.node_types import NodeRegistry
    
    registry = NodeRegistry.instance()
    registry.register(PREVIEW_NODE)
    registry.register(SAVE_IMAGE_NODE)
