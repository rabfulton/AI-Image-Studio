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


def _should_stream_previews(model) -> bool:
    return getattr(model, "provider", "") == "sd-cpp"


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
    negative_prompt = inputs.get("negative_prompt") or parameters.get("negative_prompt")
    
    # Only sd.cpp supports intermediate previews in our current implementation.
    stream_previews = bool(parameters.get("stream_previews", False)) and _should_stream_previews(model)

    preview_job_id = getattr(context, "job_id", None)
    decoding_announced = False

    def _cancel_check() -> bool:
        try:
            context.check_cancelled()
        except BaseException:
            return True
        return False

    def _emit_progress(step: int, steps: int, elapsed: float) -> None:
        nonlocal decoding_announced
        if not stream_previews:
            return
        try:
            from ai_image_studio.core.execution import ExecutionProgress, ExecutionStatus

            if preview_job_id is None:
                return

            if int(step) >= int(steps) and not decoding_announced:
                message = "Sampling complete, decoding (VAE)…"
                decoding_announced = True
            else:
                message = f"Sampling step {step}/{steps}"
            context.report_progress(
                ExecutionProgress(
                    job_id=preview_job_id,
                    status=ExecutionStatus.RUNNING,
                    current_node_name="generation.text_to_image",
                    nodes_completed=0,
                    nodes_total=1,
                    message=message,
                    preview_step=int(step),
                    preview_total_steps=int(steps),
                )
            )
        except Exception:
            return

    def _emit_preview(step: int, images: list[Any], is_noisy: bool) -> None:
        if not stream_previews:
            return
        if not images:
            return
        try:
            from ai_image_studio.core.execution import ExecutionProgress, ExecutionStatus
            from ai_image_studio.core.data_types import ImageData

            if preview_job_id is None:
                return
            img0 = images[0]
            image_data = ImageData.from_pil(img0)

            context.report_progress(
                ExecutionProgress(
                    job_id=preview_job_id,
                    status=ExecutionStatus.RUNNING,
                    current_node_name="generation.text_to_image",
                    nodes_completed=0,
                    nodes_total=1,
                    message=f"Preview step {step}",
                    preview_image=image_data,
                    preview_step=int(step),
                    preview_total_steps=int(parameters.get("steps", 20)),
                    preview_is_noisy=bool(is_noisy),
                )
            )
        except Exception:
            return

    extra_params = {k: v for k, v in parameters.items() if k in model.params}
    if _should_stream_previews(model):
        # Always allow mid-generation cancellation for sd.cpp (even if preview streaming is off).
        extra_params["_cancel_check"] = _cancel_check
    if stream_previews:
        extra_params.update(
            {
                "_progress_callback": _emit_progress,
                "_preview_callback": _emit_preview,
                "preview_method": parameters.get("preview_method", "proj"),
                "preview_interval": parameters.get("preview_interval", 2),
            }
        )

    # sd.cpp-specific decode VRAM controls
    if _should_stream_previews(model):
        extra_params.update(
            {
                "vae_tiling": bool(parameters.get("vae_tiling", False)),
                "vae_tile_overlap": float(parameters.get("vae_tile_overlap", 0.5)),
                "vae_tile_size": parameters.get("vae_tile_size", "0x0"),
                "vae_relative_tile_size": parameters.get("vae_relative_tile_size", "0x0"),
            }
        )

    request = GenerationRequest(
        model=model,
        prompt=prompt,
        width=parameters.get("width", 1024),
        height=parameters.get("height", 1024),
        num_images=parameters.get("num_images", 1),
        negative_prompt=negative_prompt,
        seed=parameters.get("seed") if parameters.get("seed", -1) >= 0 else None,
        extra_params=extra_params,
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
        ParameterDefinition.text(
            name="negative_prompt",
            label="Negative Prompt",
            default="",
            description="Negative prompt (used when not connected as an input)",
        ),
        ParameterDefinition(
            name="model",
            label="Model",
            param_type=ParameterType.MODEL,
            default="dall-e-3",
            description="AI model to use",
        ),
        ParameterDefinition.integer(
            name="steps",
            label="Steps",
            default=20,
            min_value=1,
            max_value=100,
            description="Inference steps (primarily for local models)",
        ),
        ParameterDefinition.float_param(
            name="cfg_scale",
            label="CFG Scale",
            default=7.0,
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            description="Guidance strength (primarily for local models)",
        ),
        ParameterDefinition.enum(
            name="sampler",
            label="Sampler",
            options=[
                ("default", "Default"),
                ("euler", "Euler"),
                ("euler_a", "Euler A"),
                ("heun", "Heun"),
                ("dpm2", "DPM2"),
                ("dpm++2s_a", "DPM++ 2S a"),
                ("dpm++2m", "DPM++ 2M"),
                ("dpm++2mv2", "DPM++ 2M v2"),
                ("ipndm", "IPNDM"),
                ("ipndm_v", "IPNDM v"),
                ("lcm", "LCM"),
                ("ddim_trailing", "DDIM Trailing"),
                ("tcd", "TCD"),
            ],
            default="euler_a",
            description="Sampling method (primarily for local models)",
        ),
        ParameterDefinition.enum(
            name="scheduler",
            label="Scheduler",
            options=[
                ("default", "Default"),
                ("discrete", "Discrete"),
                ("karras", "Karras"),
                ("exponential", "Exponential"),
                ("ays", "AYS"),
                ("gits", "GITS"),
                ("sgm_uniform", "SGM Uniform"),
                ("simple", "Simple"),
                ("smoothstep", "Smoothstep"),
                ("lcm", "LCM"),
            ],
            default="karras",
            description="Noise schedule (primarily for local models)",
        ),
        ParameterDefinition.boolean(
            name="stream_previews",
            label="Stream Preview",
            default=False,
            description="Show intermediate previews during sampling (local sd.cpp only)",
        ),
        ParameterDefinition.enum(
            name="preview_method",
            label="Preview Method",
            options=[
                ("proj", "Projection"),
            ],
            default="proj",
            description="Preview method used by sd.cpp (local only)",
        ),
        ParameterDefinition.integer(
            name="preview_interval",
            label="Preview Interval",
            default=2,
            min_value=1,
            max_value=20,
            description="Preview every N steps (local sd.cpp only)",
        ),
        ParameterDefinition.boolean(
            name="vae_tiling",
            label="VAE Tiling",
            default=True,
            description="Decode the final image in tiles to reduce VRAM usage (local sd.cpp only)",
        ),
        ParameterDefinition.text(
            name="vae_tile_size",
            label="VAE Tile Size",
            default="0x0",
            description="Tile size like '256x256' (0x0 = auto) (local sd.cpp only)",
        ),
        ParameterDefinition.text(
            name="vae_relative_tile_size",
            label="VAE Relative Tile Size",
            default="0x0",
            description="Relative tile size like '0.5x0.5' or '2x2' (0x0 = disabled) (local sd.cpp only)",
        ),
        ParameterDefinition.slider(
            name="vae_tile_overlap",
            label="VAE Tile Overlap",
            default=0.5,
            min_value=0.0,
            max_value=0.9,
            step=0.05,
            description="Tile overlap fraction (local sd.cpp only)",
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
