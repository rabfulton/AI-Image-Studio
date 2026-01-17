"""
Generation Nodes - Nodes that generate images using providers.

These nodes use the provider system to make actual API calls
for image generation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ai_image_studio.core.node_types import (
    NodeType,
    NodeCategory,
    InputDefinition,
    OutputDefinition,
    ParameterDefinition,
    ParameterType,
    register_node,
    NodeRegistry,
)
from ai_image_studio.core.data_types import DataType


async def text_to_image_executor(
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """Execute text-to-image generation."""
    from ai_image_studio.providers import get_registry, GenerationRequest
    from ai_image_studio.core.data_types import ImageData
    
    registry = get_registry()
    
    # Get model
    model_id = parameters.get("model", "dall-e-3")
    model = registry.get_model(model_id)
    
    if not model:
        raise ValueError(f"Model not found: {model_id}")
    
    # Get provider
    provider = registry.get_provider(model.provider)
    if not provider:
        raise ValueError(f"Provider not configured: {model.provider}")
    
    if not provider.is_configured:
        raise ValueError(f"Provider {model.provider} needs API key. Go to Providers → Manage Providers.")
    
    # Build request
    prompt = inputs.get("prompt", "") or parameters.get("prompt", "")
    
    request = GenerationRequest(
        model=model,
        prompt=prompt,
        width=parameters.get("width", 1024),
        height=parameters.get("height", 1024),
        num_images=parameters.get("num_images", 1),
        negative_prompt=parameters.get("negative_prompt"),
        seed=parameters.get("seed") if parameters.get("seed", -1) >= 0 else None,
        extra_params={
            k: v for k, v in parameters.items()
            if k in model.params
        },
    )
    
    # Generate
    result = await provider.generate(request)
    
    # Return first image (or all if multi-output supported)
    if result.images:
        return {
            "image": result.images[0],
            "seed": result.seed,
            "revised_prompt": result.revised_prompt,
        }
    else:
        raise ValueError("No images generated")


# Register the text-to-image node type
TEXT_TO_IMAGE_NODE = NodeType(
    id="generation.text_to_image",
    name="Text to Image",
    description="Generate an image from a text prompt using AI",
    category=NodeCategory.GENERATION,
    inputs=[
        InputDefinition(
            name="prompt",
            label="Prompt",
            data_type=DataType.TEXT,
            description="Text prompt describing the image",
            required=False,  # Can also use parameter
        ),
        InputDefinition(
            name="negative_prompt",
            label="Negative Prompt",
            data_type=DataType.TEXT,
            description="What to avoid in the image (for models that support it)",
            required=False,
        ),
    ],
    outputs=[
        OutputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Generated image",
        ),
    ],
    parameters=[
        ParameterDefinition.text(
            name="prompt",
            label="Prompt",
            default="A beautiful landscape",
            description="Text prompt when not using input connection",
        ),
        ParameterDefinition(
            name="model",
            label="Model",
            param_type=ParameterType.MODEL,
            default="dall-e-3",
            description="AI model to use",
        ),
        ParameterDefinition.integer(
            name="width",
            label="Width",
            default=1024,
            description="Output width in pixels",
            min_value=256,
            max_value=2048,
        ),
        ParameterDefinition.integer(
            name="height",
            label="Height",
            default=1024,
            description="Output height in pixels",
            min_value=256,
            max_value=2048,
        ),
        ParameterDefinition.seed(
            name="seed",
            label="Seed",
            default=-1,
        ),
    ],
    executor=text_to_image_executor,
)


async def image_to_image_executor(
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    """Execute image-to-image generation."""
    from ai_image_studio.providers import get_registry, GenerationRequest
    
    registry = get_registry()
    
    model_id = parameters.get("model", "flux-dev")
    model = registry.get_model(model_id)
    
    if not model:
        raise ValueError(f"Model not found: {model_id}")
    
    if not model.supports_img2img:
        raise ValueError(f"Model {model_id} does not support image-to-image")
    
    provider = registry.get_provider(model.provider)
    if not provider or not provider.is_configured:
        raise ValueError(f"Provider {model.provider} not configured")
    
    # Get input image
    input_image = inputs.get("image")
    if not input_image:
        raise ValueError("Input image required for image-to-image")
    
    prompt = inputs.get("prompt", "") or parameters.get("prompt", "")
    
    request = GenerationRequest(
        model=model,
        prompt=prompt,
        width=parameters.get("width", 1024),
        height=parameters.get("height", 1024),
        reference_images=[input_image],
        strength=parameters.get("strength", 0.75),
        seed=parameters.get("seed") if parameters.get("seed", -1) >= 0 else None,
        extra_params={k: v for k, v in parameters.items() if k in model.params},
    )
    
    result = await provider.generate(request)
    
    if result.images:
        return {"image": result.images[0]}
    else:
        raise ValueError("No images generated")


IMAGE_TO_IMAGE_NODE = NodeType(
    id="generation.image_to_image",
    name="Image to Image",
    description="Transform an image using AI with a text prompt",
    category=NodeCategory.GENERATION,
    inputs=[
        InputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Input image to transform",
            required=True,
        ),
        InputDefinition(
            name="prompt",
            label="Prompt",
            data_type=DataType.TEXT,
            description="Text prompt describing the transformation",
            required=False,
        ),
    ],
    outputs=[
        OutputDefinition(
            name="image",
            label="Image",
            data_type=DataType.IMAGE,
            description="Transformed image",
        ),
    ],
    parameters=[
        ParameterDefinition.text(
            name="prompt",
            label="Prompt",
            default="Transform this image",
        ),
        ParameterDefinition(
            name="model",
            label="Model",
            param_type=ParameterType.MODEL,
            default="flux-dev",
        ),
        ParameterDefinition.slider(
            name="strength",
            label="Strength",
            default=0.75,
            min_value=0.0,
            max_value=1.0,
            description="How much to transform (0=no change, 1=complete redraw)",
        ),
        ParameterDefinition.seed(name="seed", label="Seed", default=-1),
    ],
    executor=image_to_image_executor,
)


def register_generation_nodes() -> None:
    """Register all generation nodes."""
    registry = NodeRegistry.instance()
    registry.register(TEXT_TO_IMAGE_NODE)
    registry.register(IMAGE_TO_IMAGE_NODE)


def get_available_models() -> list[tuple[str, str]]:
    """Get list of available models as (value, label) tuples."""
    from ai_image_studio.providers import get_registry
    
    registry = get_registry()
    models = registry.list_available_models()
    
    if not models:
        # Return all models if none configured
        models = registry.list_models()
    
    return [(m.id, m.name) for m in models]
