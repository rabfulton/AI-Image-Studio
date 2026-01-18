"""
Provider Registry - Central registry for providers and model cards.

This module manages:
- Registration of provider implementations
- Built-in model cards for supported providers
- Provider configuration loading/saving
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import json

from ai_image_studio.providers.base import (
    ModelCard,
    GenerationMode,
    ImageProvider,
    ProviderConfig,
)

if TYPE_CHECKING:
    pass


# ============================================================================
# Built-in Model Cards
# ============================================================================
# These define the capabilities of known models.
# Values sourced from provider docs and ComfyUI node definitions.

BUILTIN_MODEL_CARDS: dict[str, ModelCard] = {
    # -------------------------------------------------------------------------
    # OpenAI DALL-E Models
    # Source: https://platform.openai.com/docs/api-reference/images
    # -------------------------------------------------------------------------
    "dall-e-3": ModelCard(
        id="dall-e-3",
        provider="openai",
        name="DALL·E 3",
        description="OpenAI's latest image generation model with high quality and prompt following",
        modes={GenerationMode.TEXT_TO_IMAGE},
        resolutions=["1024x1024", "1024x1792", "1792x1024"],
        max_images=1,
        params={"quality", "style"},
        param_options={
            "quality": ["standard", "hd"],
            "style": ["natural", "vivid"],
        },
        param_defaults={"quality": "standard", "style": "natural"},
        pricing_tier="standard",
        tags=["high-quality", "prompt-following"],
    ),
    
    "dall-e-2": ModelCard(
        id="dall-e-2",
        provider="openai",
        name="DALL·E 2",
        description="OpenAI's previous generation model with editing support",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE, GenerationMode.INPAINTING},
        resolutions=["256x256", "512x512", "1024x1024"],
        max_images=10,
        max_reference_images=1,
        params=set(),
        pricing_tier="budget",
        tags=["editing", "inpainting"],
    ),
    
    "gpt-image-1": ModelCard(
        id="gpt-image-1",
        provider="openai",
        name="GPT Image 1",
        description="OpenAI's GPT-based image generation with editing support",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE, GenerationMode.INPAINTING},
        resolutions=["auto", "1024x1024", "1024x1536", "1536x1024"],
        max_images=8,
        max_reference_images=16,
        params={"quality", "background", "input_fidelity", "output_format", "moderation"},
        param_options={
            "quality": ["low", "medium", "high"],
            "background": ["auto", "opaque", "transparent"],
            "input_fidelity": ["low", "high"],
            "output_format": ["png", "webp", "jpeg"],
            "moderation": ["auto", "low"],
        },
        param_defaults={"quality": "low", "background": "auto", "input_fidelity": "low"},
        pricing_tier="standard",
        tags=["editing", "inpainting"],
    ),
    
    "gpt-image-1.5": ModelCard(
        id="gpt-image-1.5",
        provider="openai",
        name="GPT Image 1.5",
        description="OpenAI's most advanced image model with superior visual understanding",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE, GenerationMode.INPAINTING},
        resolutions=["auto", "1024x1024", "1024x1536", "1536x1024"],
        max_images=8,
        max_reference_images=16,
        params={"quality", "background", "input_fidelity", "output_format", "moderation"},
        param_options={
            "quality": ["low", "medium", "high"],
            "background": ["auto", "opaque", "transparent"],
            "input_fidelity": ["low", "high"],
            "output_format": ["png", "webp", "jpeg"],
            "moderation": ["auto", "low"],
        },
        param_defaults={"quality": "medium", "background": "auto", "input_fidelity": "high"},
        pricing_tier="premium",
        tags=["high-quality", "editing", "inpainting", "visual-understanding"],
    ),
    
    # -------------------------------------------------------------------------
    # Black Forest Labs FLUX Models
    # Source: https://docs.bfl.ai/
    # -------------------------------------------------------------------------
    
    # Latest FLUX 2 generation
    "flux-2-pro": ModelCard(
        id="flux-2-pro",
        provider="bfl",
        name="FLUX 2 Pro",
        description="Latest FLUX 2 Pro model with best overall quality",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
        resolution_range=(256, 1440, 32),
        max_images=1,
        params={"seed", "safety_tolerance", "prompt_upsampling"},
        param_options={
            "safety_tolerance": ["1", "2", "3", "4", "5", "6"],
        },
        param_defaults={"safety_tolerance": "2"},
        pricing_tier="premium",
        tags=["high-quality", "latest"],
    ),
    
    "flux-2-max": ModelCard(
        id="flux-2-max",
        provider="bfl",
        name="FLUX 2 Max",
        description="FLUX 2 Max - highest quality generation",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
        resolution_range=(256, 1440, 32),
        max_images=1,
        params={"seed", "safety_tolerance"},
        param_options={
            "safety_tolerance": ["1", "2", "3", "4", "5", "6"],
        },
        param_defaults={"safety_tolerance": "2"},
        pricing_tier="premium",
        tags=["high-quality", "max"],
    ),
    
    # Kontext models for image editing
    "flux-kontext-pro": ModelCard(
        id="flux-kontext-pro",
        provider="bfl",
        name="FLUX Kontext Pro",
        description="FLUX Kontext for image editing and manipulation",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "3:7", "7:3"],
        max_images=1,
        max_reference_images=1,
        params={"seed", "safety_tolerance", "prompt_upsampling", "output_format"},
        param_options={
            "safety_tolerance": ["1", "2", "3", "4", "5", "6"],
            "output_format": ["jpeg", "png"],
        },
        param_defaults={"safety_tolerance": "2", "output_format": "jpeg"},
        pricing_tier="standard",
        tags=["editing", "kontext"],
    ),
    
    "flux-kontext-max": ModelCard(
        id="flux-kontext-max",
        provider="bfl",
        name="FLUX Kontext Max",
        description="FLUX Kontext Max - highest quality editing",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "3:7", "7:3"],
        max_images=1,
        max_reference_images=1,
        params={"seed", "safety_tolerance", "prompt_upsampling", "output_format"},
        param_options={
            "safety_tolerance": ["1", "2", "3", "4", "5", "6"],
            "output_format": ["jpeg", "png"],
        },
        param_defaults={"safety_tolerance": "2", "output_format": "jpeg"},
        pricing_tier="premium",
        tags=["editing", "kontext", "high-quality"],
    ),
    
    # FLUX Pro 1.1 series
    "flux-pro-1.1-ultra": ModelCard(
        id="flux-pro-1.1-ultra",
        provider="bfl",
        name="FLUX Pro 1.1 Ultra",
        description="Ultra quality FLUX Pro 1.1 model",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
        resolution_range=(256, 1440, 32),
        max_images=1,
        params={"seed", "safety_tolerance", "prompt_upsampling"},
        param_options={
            "safety_tolerance": ["1", "2", "3", "4", "5", "6"],
        },
        param_defaults={"safety_tolerance": "2"},
        pricing_tier="premium",
        tags=["high-quality", "ultra"],
    ),
    
    "flux-pro-1.1": ModelCard(
        id="flux-pro-1.1",
        provider="bfl",
        name="FLUX Pro 1.1",
        description="FLUX Pro 1.1 with superior image quality",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
        resolution_range=(256, 1440, 32),
        max_images=1,
        params={"seed", "safety_tolerance", "prompt_upsampling"},
        param_options={
            "safety_tolerance": ["1", "2", "3", "4", "5", "6"],
        },
        param_defaults={"safety_tolerance": "2"},
        pricing_tier="standard",
        tags=["high-quality"],
    ),
    
    "flux-pro": ModelCard(
        id="flux-pro",
        provider="bfl",
        name="FLUX Pro",
        description="Professional FLUX model",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"],
        resolution_range=(256, 1440, 32),
        max_images=1,
        params={"seed", "steps", "guidance", "safety_tolerance", "prompt_upsampling"},
        param_defaults={"steps": 25, "guidance": 3.0},
        pricing_tier="standard",
    ),
    
    "flux-dev": ModelCard(
        id="flux-dev",
        provider="bfl",
        name="FLUX Dev",
        description="Development FLUX model with image-to-image support",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4"],
        resolution_range=(256, 1440, 32),
        max_images=1,
        max_reference_images=4,
        params={"seed", "steps", "guidance", "prompt_upsampling"},
        param_defaults={"steps": 28, "guidance": 3.5},
        pricing_tier="budget",
    ),
    
    "flux-schnell": ModelCard(
        id="flux-schnell",
        provider="bfl",
        name="FLUX Schnell",
        description="Fast FLUX model for rapid prototyping",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4"],
        resolution_range=(256, 1440, 32),
        max_images=1,
        params={"seed"},
        pricing_tier="budget",
        tags=["fast"],
    ),

    
    # -------------------------------------------------------------------------
    # Google Gemini/Imagen
    # Source: https://ai.google.dev/gemini-api/docs/imagen
    # Source: https://ai.google.dev/gemini-api/docs/image-generation
    # -------------------------------------------------------------------------
    
    # Imagen 4 models - text-to-image only, via :predict endpoint
    "imagen-4.0-generate-001": ModelCard(
        id="imagen-4.0-generate-001",
        provider="gemini",
        name="Imagen 4 Standard",
        description="Google's latest Imagen model for high-quality image generation",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "3:4", "4:3", "9:16", "16:9"],
        max_images=4,
        params={"numberOfImages", "aspectRatio", "personGeneration", "imageSize"},
        param_options={
            "numberOfImages": ["1", "2", "3", "4"],
            "aspectRatio": ["1:1", "3:4", "4:3", "9:16", "16:9"],
            "personGeneration": ["dont_allow", "allow_adult", "allow_all"],
            "imageSize": ["1K", "2K"],
        },
        param_defaults={"numberOfImages": "4", "aspectRatio": "1:1", "personGeneration": "allow_adult", "imageSize": "1K"},
        pricing_tier="standard",
        tags=["high-quality"],
    ),
    
    "imagen-4.0-ultra-generate-001": ModelCard(
        id="imagen-4.0-ultra-generate-001",
        provider="gemini",
        name="Imagen 4 Ultra",
        description="Google's highest quality Imagen model",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "3:4", "4:3", "9:16", "16:9"],
        max_images=4,
        params={"numberOfImages", "aspectRatio", "personGeneration", "imageSize"},
        param_options={
            "numberOfImages": ["1", "2", "3", "4"],
            "aspectRatio": ["1:1", "3:4", "4:3", "9:16", "16:9"],
            "personGeneration": ["dont_allow", "allow_adult", "allow_all"],
            "imageSize": ["1K", "2K"],
        },
        param_defaults={"numberOfImages": "4", "aspectRatio": "1:1", "personGeneration": "allow_adult", "imageSize": "2K"},
        pricing_tier="premium",
        tags=["high-quality", "ultra"],
    ),
    
    "imagen-4.0-fast-generate-001": ModelCard(
        id="imagen-4.0-fast-generate-001",
        provider="gemini",
        name="Imagen 4 Fast",
        description="Fast Imagen model optimized for speed",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "3:4", "4:3", "9:16", "16:9"],
        max_images=4,
        params={"numberOfImages", "aspectRatio", "personGeneration", "imageSize"},
        param_options={
            "numberOfImages": ["1", "2", "3", "4"],
            "aspectRatio": ["1:1", "3:4", "4:3", "9:16", "16:9"],
            "personGeneration": ["dont_allow", "allow_adult", "allow_all"],
            "imageSize": ["1K", "2K"],
        },
        param_defaults={"numberOfImages": "1", "aspectRatio": "1:1", "personGeneration": "allow_adult", "imageSize": "1K"},
        pricing_tier="budget",
        tags=["fast"],
    ),
    
    "imagen-3.0-generate-002": ModelCard(
        id="imagen-3.0-generate-002",
        provider="gemini",
        name="Imagen 3",
        description="Google's previous generation Imagen model",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "3:4", "4:3", "9:16", "16:9"],
        max_images=4,
        params={"numberOfImages", "aspectRatio", "personGeneration"},
        param_options={
            "numberOfImages": ["1", "2", "3", "4"],
            "aspectRatio": ["1:1", "3:4", "4:3", "9:16", "16:9"],
            "personGeneration": ["dont_allow", "allow_adult", "allow_all"],
        },
        param_defaults={"numberOfImages": "4", "aspectRatio": "1:1", "personGeneration": "allow_adult"},
        pricing_tier="standard",
        tags=["high-quality"],
    ),
    
    # Gemini Image models - text-to-image AND image editing, via :generateContent endpoint
    "gemini-2.5-flash-image": ModelCard(
        id="gemini-2.5-flash-image",
        provider="gemini",
        name="Gemini 2.5 Flash Image",
        description="Fast Gemini model for image generation and editing",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4"],
        max_images=1,
        max_reference_images=3,
        params={"aspectRatio", "response_modalities"},
        param_options={
            "aspectRatio": ["1:1", "16:9", "9:16", "4:3", "3:4"],
            "response_modalities": ["Image", "Text,Image"],
        },
        param_defaults={"aspectRatio": "1:1", "response_modalities": "Image"},
        pricing_tier="budget",
        tags=["fast", "editing"],
    ),
    
    "gemini-3-pro-image-preview": ModelCard(
        id="gemini-3-pro-image-preview",
        provider="gemini",
        name="Gemini 3 Pro Image",
        description="Professional Gemini model with up to 4K output and 14 reference images",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE},
        aspect_ratios=["1:1", "16:9", "9:16", "4:3", "3:4"],
        max_images=1,
        max_reference_images=14,
        params={"aspectRatio", "imageSize", "response_modalities"},
        param_options={
            "aspectRatio": ["1:1", "16:9", "9:16", "4:3", "3:4"],
            "imageSize": ["1K", "2K", "4K"],
            "response_modalities": ["Image", "Text,Image"],
        },
        param_defaults={"aspectRatio": "1:1", "imageSize": "2K", "response_modalities": "Image"},
        pricing_tier="premium",
        tags=["high-quality", "editing", "4K"],
    ),
    
    # -------------------------------------------------------------------------
    # OpenRouter (proxies various models)
    # Uses chat completions with modalities=["image", "text"]
    # Source: https://openrouter.ai/docs/guides/overview/multimodal/image-generation
    # -------------------------------------------------------------------------
    
    # Free tier model
    "openrouter/google/gemini-2.0-flash-exp:free": ModelCard(
        id="openrouter/google/gemini-2.0-flash-exp:free",
        provider="openrouter",
        name="Gemini 2.0 Flash (Free)",
        description="Google's fast multimodal model via OpenRouter - Free tier",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE},
        aspect_ratios=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        max_images=1,
        max_reference_images=3,
        params={"aspect_ratio"},
        param_options={
            "aspect_ratio": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        },
        param_defaults={"aspect_ratio": "1:1"},
        pricing_tier="free",
        tags=["free", "fast", "editing"],
    ),
    
    # Gemini Image models via OpenRouter
    "google/gemini-2.5-flash-image-preview": ModelCard(
        id="google/gemini-2.5-flash-image-preview",
        provider="openrouter",
        name="Gemini 2.5 Flash Image (OR)",
        description="Google's fast image model via OpenRouter",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE},
        aspect_ratios=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        max_images=1,
        max_reference_images=3,
        params={"aspect_ratio"},
        param_options={
            "aspect_ratio": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        },
        param_defaults={"aspect_ratio": "1:1"},
        pricing_tier="budget",
        tags=["fast", "editing"],
    ),
    
    "google/gemini-3-pro-image-preview": ModelCard(
        id="google/gemini-3-pro-image-preview",
        provider="openrouter",
        name="Gemini 3 Pro Image (OR)",
        description="Google's professional image model via OpenRouter with up to 4K output",
        modes={GenerationMode.TEXT_TO_IMAGE, GenerationMode.IMAGE_TO_IMAGE},
        aspect_ratios=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        max_images=1,
        max_reference_images=14,
        params={"aspect_ratio", "image_size"},
        param_options={
            "aspect_ratio": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
            "image_size": ["1K", "2K", "4K"],
        },
        param_defaults={"aspect_ratio": "1:1", "image_size": "2K"},
        pricing_tier="premium",
        tags=["high-quality", "editing", "4K"],
    ),
    
    # BFL FLUX models via OpenRouter
    "black-forest-labs/flux-1.1-pro": ModelCard(
        id="black-forest-labs/flux-1.1-pro",
        provider="openrouter",
        name="FLUX 1.1 Pro (OR)",
        description="BFL's FLUX 1.1 Pro model via OpenRouter",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        max_images=1,
        params={"aspect_ratio"},
        param_options={
            "aspect_ratio": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        },
        param_defaults={"aspect_ratio": "1:1"},
        pricing_tier="standard",
        tags=["high-quality"],
    ),
    
    "black-forest-labs/flux-schnell": ModelCard(
        id="black-forest-labs/flux-schnell",
        provider="openrouter",
        name="FLUX Schnell (OR)",
        description="BFL's fast FLUX model via OpenRouter",
        modes={GenerationMode.TEXT_TO_IMAGE},
        aspect_ratios=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9"],
        max_images=1,
        params={"aspect_ratio"},
        param_options={
            "aspect_ratio": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9"],
        },
        param_defaults={"aspect_ratio": "1:1"},
        pricing_tier="budget",
        tags=["fast"],
    ),
    
    # -------------------------------------------------------------------------
    # xAI (Grok)
    # Source: https://docs.x.ai/docs/guides/image-generations
    # -------------------------------------------------------------------------
    "grok-2-image": ModelCard(
        id="grok-2-image",
        provider="xai",
        name="Grok 2 Image",
        description="xAI's Grok image generation model",
        modes={GenerationMode.TEXT_TO_IMAGE},
        max_images=10,
        params=set(),  # No quality/size/style params supported
        pricing_tier="standard",
        tags=["text-to-image"],
    ),
}


class ProviderRegistry:
    """
    Central registry for providers and model cards.
    
    Handles:
    - Provider registration
    - Model card lookup
    - Configuration management
    """
    
    _instance: ProviderRegistry | None = None
    
    def __new__(cls) -> ProviderRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
    
    @classmethod
    def instance(cls) -> ProviderRegistry:
        return cls()
    
    def _init(self) -> None:
        """Initialize the registry."""
        self._providers: dict[str, type[ImageProvider]] = {}
        self._provider_instances: dict[str, ImageProvider] = {}
        self._model_cards: dict[str, ModelCard] = dict(BUILTIN_MODEL_CARDS)
        self._configs: dict[str, ProviderConfig] = {}
        self._config_path: Path | None = None
    
    # -------------------------------------------------------------------------
    # Provider Registration
    # -------------------------------------------------------------------------
    
    def register_provider(self, provider_class: type[ImageProvider]) -> None:
        """Register a provider implementation."""
        self._providers[provider_class.id] = provider_class
    
    def get_provider(self, provider_id: str) -> ImageProvider | None:
        """Get an instantiated provider."""
        if provider_id in self._provider_instances:
            return self._provider_instances[provider_id]
        
        if provider_id not in self._providers:
            return None
        
        config = self._configs.get(provider_id, ProviderConfig())
        provider = self._providers[provider_id](config)
        self._provider_instances[provider_id] = provider
        return provider
    
    def list_providers(self) -> list[str]:
        """Get list of registered provider IDs."""
        return list(self._providers.keys())
    
    def list_configured_providers(self) -> list[str]:
        """Get list of providers with API keys configured."""
        return [
            pid for pid in self._providers.keys()
            if pid in self._configs and self._configs[pid].api_key
        ]
    
    # -------------------------------------------------------------------------
    # Model Cards
    # -------------------------------------------------------------------------
    
    def register_model(self, card: ModelCard) -> None:
        """Register a model card."""
        self._model_cards[card.id] = card
    
    def get_model(self, model_id: str) -> ModelCard | None:
        """Get a model card by ID."""
        return self._model_cards.get(model_id)
    
    def list_models(self, provider_id: str | None = None) -> list[ModelCard]:
        """List all model cards, optionally filtered by provider."""
        if provider_id:
            return [m for m in self._model_cards.values() if m.provider == provider_id]
        return list(self._model_cards.values())
    
    def list_available_models(self) -> list[ModelCard]:
        """List models from configured providers only."""
        configured = set(self.list_configured_providers())
        # Also include local provider if it has models
        local_models = [m for m in self._model_cards.values() if m.id.startswith("local/")]
        api_models = [m for m in self._model_cards.values() if m.provider in configured]
        return api_models + local_models
    
    def refresh_local_models(self) -> int:
        """
        Scan local model folders and register discovered models.
        
        Returns:
            Number of models discovered and registered.
        """
        from ai_image_studio.providers.sd_cpp_models import LocalModelScanner
        from pathlib import Path
        
        config = self.get_config("sd-cpp")
        folders = config.extra.get("model_folders", [])
        
        if not folders:
            # Remove any existing local models
            self._model_cards = {
                k: v for k, v in self._model_cards.items()
                if not k.startswith("local/")
            }
            return 0
        
        # Remove old local models
        self._model_cards = {
            k: v for k, v in self._model_cards.items()
            if not k.startswith("local/")
        }
        
        # Scan and register new ones
        scanner = LocalModelScanner()
        models = scanner.scan([Path(f) for f in folders])
        
        for info in models:
            card = scanner.to_model_card(info)
            self.register_model(card)
        
        return len(models)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    def set_config(self, provider_id: str, config: ProviderConfig) -> None:
        """Set configuration for a provider."""
        self._configs[provider_id] = config
        # Invalidate cached instance
        self._provider_instances.pop(provider_id, None)
    
    def get_config(self, provider_id: str) -> ProviderConfig:
        """Get configuration for a provider."""
        return self._configs.get(provider_id, ProviderConfig())
    
    def load_config(self, path: Path | None = None) -> None:
        """Load provider configurations from file."""
        if path is None:
            path = Path.home() / ".config" / "ai_image_studio" / "providers.json"
        
        self._config_path = path
        
        if not path.exists():
            return
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            # Load provider configs
            for provider_id, cfg_data in data.get("providers", {}).items():
                self._configs[provider_id] = ProviderConfig(
                    api_key=cfg_data.get("api_key", ""),
                    enabled=cfg_data.get("enabled", True),
                    base_url=cfg_data.get("base_url"),
                    default_model=cfg_data.get("default_model"),
                    extra=cfg_data.get("extra", {}),
                )
            
            # Load custom model cards
            for card_data in data.get("custom_models", []):
                card = ModelCard(
                    id=card_data["id"],
                    provider=card_data["provider"],
                    name=card_data["name"],
                    description=card_data.get("description", ""),
                    modes={GenerationMode(m) for m in card_data.get("modes", ["text_to_image"])},
                    resolutions=card_data.get("resolutions"),
                    aspect_ratios=card_data.get("aspect_ratios"),
                    max_images=card_data.get("max_images", 1),
                    params=set(card_data.get("params", [])),
                    param_options=card_data.get("param_options", {}),
                )
                self._model_cards[card.id] = card
                
        except Exception as e:
            print(f"Warning: Failed to load provider config: {e}")
    
    def save_config(self, path: Path | None = None) -> None:
        """Save provider configurations to file."""
        if path is None:
            path = self._config_path or Path.home() / ".config" / "ai_image_studio" / "providers.json"
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "providers": {
                pid: {
                    "api_key": cfg.api_key,
                    "enabled": cfg.enabled,
                    "base_url": cfg.base_url,
                    "default_model": cfg.default_model,
                    "extra": cfg.extra,
                }
                for pid, cfg in self._configs.items()
            },
            "custom_models": [
                {
                    "id": card.id,
                    "provider": card.provider,
                    "name": card.name,
                    "description": card.description,
                    "modes": [m.value for m in card.modes],
                    "resolutions": card.resolutions,
                    "aspect_ratios": card.aspect_ratios,
                    "max_images": card.max_images,
                    "params": list(card.params),
                    "param_options": card.param_options,
                }
                for card in self._model_cards.values()
                if card.id not in BUILTIN_MODEL_CARDS  # Only save custom models
            ],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ============================================================================
# Module-level convenience functions
# ============================================================================

def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    return ProviderRegistry.instance()


def get_models_for_provider(provider_id: str) -> list[ModelCard]:
    """Get all model cards for a provider."""
    return get_registry().list_models(provider_id)


def get_model(model_id: str) -> ModelCard | None:
    """Get a model card by ID."""
    return get_registry().get_model(model_id)
