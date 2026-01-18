"""
Image Generation Providers.

This package provides integrations with various AI image generation services:
- OpenAI: DALL-E 2, DALL-E 3, GPT Image
- BFL: FLUX Pro, FLUX Dev, FLUX Schnell
- Google Gemini: Imagen 3/4, Gemini Image models
- xAI: Grok 2 Image
- OpenRouter: Multi-model proxy

Usage:
    from ai_image_studio.providers import get_registry
    
    registry = get_registry()
    registry.load_config()
    
    provider = registry.get_provider("openai")
    models = registry.list_models("openai")
"""

from ai_image_studio.providers.base import (
    AuthenticationError,
    GenerationError,
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    ImageProvider,
    ModelCard,
    ProviderConfig,
    ProviderError,
    RateLimitError,
)

from ai_image_studio.providers.registry import (
    BUILTIN_MODEL_CARDS,
    ProviderRegistry,
    get_model,
    get_models_for_provider,
    get_registry,
)

# Import providers to register them
from ai_image_studio.providers.openai import OpenAIProvider
from ai_image_studio.providers.bfl import BFLProvider
from ai_image_studio.providers.gemini import GeminiProvider
from ai_image_studio.providers.xai import XAIProvider
from ai_image_studio.providers.openrouter import OpenRouterProvider


# Auto-register providers
def _register_providers():
    registry = get_registry()
    registry.register_provider(OpenAIProvider)
    registry.register_provider(BFLProvider)
    registry.register_provider(GeminiProvider)
    registry.register_provider(XAIProvider)
    registry.register_provider(OpenRouterProvider)

_register_providers()


__all__ = [
    # Base classes
    "ImageProvider",
    "ModelCard",
    "ProviderConfig",
    "GenerationMode",
    "GenerationRequest",
    "GenerationResult",
    # Exceptions
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "GenerationError",
    # Registry
    "ProviderRegistry",
    "get_registry",
    "get_model",
    "get_models_for_provider",
    "BUILTIN_MODEL_CARDS",
    # Providers
    "OpenAIProvider",
    "BFLProvider",
    "GeminiProvider",
    "XAIProvider",
    "OpenRouterProvider",
]
