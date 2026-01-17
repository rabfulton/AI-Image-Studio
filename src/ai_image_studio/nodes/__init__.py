"""
Nodes package - All node implementations.

This package contains node implementations organized by category:
- input: Prompt, Image, Mask inputs
- generation: Text-to-image, Image-to-image
- enhancement: Upscale, Face restore
- filter: G'MIC and image filters
- utility: Blend, Resize, Crop
- output: Preview, Save
"""

from ai_image_studio.nodes.generation import register_generation_nodes
from ai_image_studio.nodes.input import register_input_nodes


def register_all_nodes() -> None:
    """Register all built-in nodes."""
    register_input_nodes()
    register_generation_nodes()
    # TODO: Register other node categories


__all__ = [
    "register_all_nodes",
]
