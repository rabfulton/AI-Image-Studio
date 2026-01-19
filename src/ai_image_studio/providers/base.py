"""
Provider Base - Abstract base classes and model card definitions.

This module provides the foundation for all image generation providers:
- ModelCard: Complete specification of a model's capabilities
- ImageProvider: Abstract base class for provider implementations
- GenerationRequest/Response: Request/response data structures

Note: Model capability values can be discovered from ComfyUI node definitions
at https://github.com/Comfy-Org/ComfyUI/tree/master/comfy_api_nodes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_image_studio.core.data_types import ImageData, MaskData


class GenerationMode(Enum):
    """Supported image generation modes."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    UPSCALE = "upscale"


@dataclass
class ModelCard:
    """
    Complete specification of an image generation model's capabilities.
    
    This data-driven approach allows adding new models without code changes.
    Model capabilities can be discovered from provider docs or ComfyUI nodes.
    
    Attributes:
        id: Unique model identifier (e.g., "dall-e-3", "flux-pro")
        provider: Provider ID this model belongs to (e.g., "openai", "bfl")
        name: Human-readable display name
        description: Brief description of the model
        
        modes: Supported generation modes
        
        resolutions: Fixed resolution options (e.g., ["1024x1024", "1792x1024"])
        aspect_ratios: Aspect ratio options (e.g., ["1:1", "16:9", "9:16"])
        resolution_range: Flexible resolution as (min, max, step) tuple
        
        max_images: Maximum images per generation request
        max_reference_images: Max reference images for img2img (0 = not supported)
        
        params: Set of supported parameter names
        param_options: Dict of parameter name -> valid options for dropdowns
        param_defaults: Default values for parameters
        
        pricing: Cost information for display (optional)
    """
    id: str
    provider: str
    name: str
    description: str = ""
    
    # Generation modes
    modes: set[GenerationMode] = field(default_factory=lambda: {GenerationMode.TEXT_TO_IMAGE})
    
    # Resolution handling - use ONE of these approaches:
    resolutions: list[str] | None = None        # Fixed: ["1024x1024", "1792x1024"]
    aspect_ratios: list[str] | None = None      # Flexible with ratios: ["1:1", "16:9"]
    resolution_range: tuple[int, int, int] | None = None  # Fully flexible: (min, max, step)
    
    # Limits
    max_images: int = 1
    max_reference_images: int = 0  # 0 = no image-to-image support
    
    # Supported parameters (names that can be passed to generate())
    params: set[str] = field(default_factory=set)
    
    # Valid options for dropdown parameters
    param_options: dict[str, list[str]] = field(default_factory=dict)
    
    # Default values for parameters
    param_defaults: dict[str, Any] = field(default_factory=dict)
    
    # Optional metadata
    pricing_tier: str = ""  # "free", "standard", "premium"
    tags: list[str] = field(default_factory=list)  # ["fast", "high-quality", etc.]
    
    @property
    def supports_img2img(self) -> bool:
        return GenerationMode.IMAGE_TO_IMAGE in self.modes
    
    @property
    def supports_inpainting(self) -> bool:
        return GenerationMode.INPAINTING in self.modes
    
    def get_resolution_options(self) -> list[str]:
        """Get available resolution options for UI."""
        if self.resolutions:
            return self.resolutions
        elif self.aspect_ratios:
            # Return aspect ratios as options
            return self.aspect_ratios
        elif self.resolution_range:
            # Return some preset sizes within range
            mn, mx, step = self.resolution_range
            presets = [512, 768, 1024, 1280, 1536, 1792, 2048]
            return [f"{s}x{s}" for s in presets if mn <= s <= mx]
        return ["1024x1024"]
    
    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and filter parameters against this model's capabilities."""
        validated = {}
        for key, value in params.items():
            if key in self.params:
                # Check if it has valid options
                if key in self.param_options:
                    if value in self.param_options[key]:
                        validated[key] = value
                    # else: skip invalid value
                else:
                    validated[key] = value
        return validated


@dataclass
class GenerationRequest:
    """Request for image generation."""
    model: ModelCard
    prompt: str
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    
    # Optional parameters
    negative_prompt: str | None = None
    seed: int | None = None
    guidance_scale: float | None = None
    steps: int | None = None
    
    # For image-to-image / inpainting
    reference_images: list[Any] = field(default_factory=list)  # ImageData
    mask: Any = None  # MaskData
    strength: float = 0.75  # How much to transform reference
    
    # Provider-specific extra parameters
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from image generation."""
    images: list[Any]  # list[ImageData]
    model_id: str
    prompt: str
    
    # Metadata
    seed: int | None = None
    generation_time: float = 0.0  # Seconds
    
    # Token usage (for APIs that report it)
    input_tokens: int | None = None
    output_tokens: int | None = None
    
    # Revised prompt (some APIs rewrite prompts)
    revised_prompt: str | None = None


@dataclass 
class ProviderConfig:
    """Configuration for a provider."""
    api_key: str = ""
    enabled: bool = True
    base_url: str | None = None  # Override default URL
    default_model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class AuthenticationError(ProviderError):
    """API key invalid or missing."""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    retry_after: float | None = None


class GenerationError(ProviderError):
    """Error during generation."""
    pass


class ImageProvider(ABC):
    """
    Abstract base class for image generation providers.
    
    Each provider handles communication with a specific API.
    Model capabilities are defined separately in ModelCard.
    """
    
    # Provider identification
    id: str = ""
    name: str = ""
    base_url: str = ""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @property
    def api_key(self) -> str:
        return self.config.api_key
    
    @property
    def is_configured(self) -> bool:
        """Check if provider has necessary configuration."""
        return bool(self.config.api_key)
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate images from a request.
        
        Args:
            request: Generation parameters
            
        Returns:
            GenerationResult with images and metadata
            
        Raises:
            AuthenticationError: Invalid API key
            RateLimitError: Rate limit exceeded
            GenerationError: Generation failed
        """
        ...
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Check if API credentials are valid.
        
        Returns:
            True if credentials are valid
        """
        ...
    
    async def list_models(self) -> list[ModelCard]:
        """
        List available models for this provider.
        
        Default implementation returns registered models.
        Override for dynamic model discovery.
        """
        from ai_image_studio.providers.registry import get_models_for_provider
        return get_models_for_provider(self.id)
    
    def get_headers(self) -> dict[str, str]:
        """Get default headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
