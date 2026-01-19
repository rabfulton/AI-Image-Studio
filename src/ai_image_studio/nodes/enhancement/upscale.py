"""
Upscale Node - AI-powered image upscaling.

Supports:
- Stability AI: Conservative, Creative, Fast upscalers
- Local: Real-ESRGAN x2, x4
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
    NodeRegistry,
)
from ai_image_studio.core.data_types import DataType


async def upscale_executor(
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """Execute image upscaling."""
    from ai_image_studio.providers import get_registry, GenerationRequest
    
    registry = get_registry()
    
    model_id = parameters.get("model", "local-realesrgan-x4")
    model = registry.get_model(model_id)
    
    if not model:
        raise ValueError(f"Upscale model not found: {model_id}")
    
    provider = registry.get_provider(model.provider)
    if not provider:
        raise ValueError(f"Provider not configured: {model.provider}")
    
    if not provider.is_configured and model.provider != "upscaler":
        raise ValueError(f"Provider {model.provider} needs API key. Go to Providers → Manage Providers.")
    
    # Get input image
    input_image = inputs.get("image")
    if not input_image:
        raise ValueError("Input image required for upscaling")
    
    # Build extra params from node parameters
    extra_params = {k: v for k, v in parameters.items() if k in model.params}
    
    request = GenerationRequest(
        model=model,
        prompt=parameters.get("prompt", ""),
        reference_images=[input_image],
        extra_params=extra_params,
    )
    
    result = await provider.generate(request)
    
    if result.images:
        return {"image": result.images[0]}
    else:
        raise ValueError("Upscaling failed - no output image")


UPSCALE_NODE = NodeType(
    id="enhancement.upscale",
    name="Upscale",
    description="Increase image resolution using AI upscaling",
    category=NodeCategory.ENHANCEMENT,
    inputs=[
        InputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Image to upscale",
            required=True,
        ),
    ],
    outputs=[
        OutputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Upscaled image",
        ),
    ],
    parameters=[
        ParameterDefinition(
            name="model",
            label="Model",
            param_type=ParameterType.MODEL,
            default="local-realesrgan-x4",
            mode_filter="upscale",  # Only show upscaler models
            description="Upscaling model to use",
        ),
    ],
    executor=upscale_executor,
    color="#10B981",  # Green for enhancement
    preview_output="image",
)


def register_enhancement_nodes() -> None:
    """Register all enhancement nodes."""
    registry = NodeRegistry.instance()
    registry.register(UPSCALE_NODE)
