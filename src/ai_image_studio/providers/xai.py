"""
xAI Provider - Grok image generation.

Supports:
- Grok 2 Image: Text-to-image generation

API Reference: https://docs.x.ai/docs/guides/image-generations
"""

from __future__ import annotations

import base64
import aiohttp
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from ai_image_studio.providers.base import (
    ImageProvider,
    GenerationRequest,
    GenerationResult,
    ProviderConfig,
    GenerationError,
    AuthenticationError,
    RateLimitError,
)


class XAIProvider(ImageProvider):
    """
    xAI image generation provider.
    
    Uses OpenAI-compatible API at api.x.ai for Grok image generation.
    Currently supports text-to-image only.
    """
    
    id = "xai"
    name = "xAI"
    base_url = "https://api.x.ai/v1"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.base_url:
            self.base_url = config.base_url
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images using xAI API."""
        url = f"{self.base_url}/images/generations"
        
        # Build request body
        body: dict[str, Any] = {
            "model": request.model.id,
            "prompt": request.prompt,
            "n": min(request.num_images, request.model.max_images),
            "response_format": "b64_json",
        }
        
        response = await self._post(url, body)
        return self._parse_response(response, request)
    
    def _parse_response(self, data: dict, request: GenerationRequest) -> GenerationResult:
        """Parse xAI response into GenerationResult."""
        from ai_image_studio.core.data_types import ImageData
        
        images = []
        revised_prompt = None
        
        for item in data.get("data", []):
            # Get revised prompt from first image
            if not revised_prompt and "revised_prompt" in item:
                revised_prompt = item["revised_prompt"]
            
            # Decode image
            if "b64_json" in item:
                b64_data = item["b64_json"]
                # Handle data URL format
                if b64_data.startswith("data:"):
                    b64_data = b64_data.split(",", 1)[1]
                
                img_bytes = base64.b64decode(b64_data)
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                arr = np.asarray(pil_img).astype(np.float32) / 255.0
                images.append(ImageData.from_numpy(arr))
        
        return GenerationResult(
            images=images,
            model_id=request.model.id,
            prompt=request.prompt,
            revised_prompt=revised_prompt,
        )
    
    async def validate_credentials(self) -> bool:
        """Validate API key by checking models endpoint."""
        try:
            url = f"{self.base_url}/models"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def _post(self, url: str, body: dict) -> dict:
        """Make POST request with JSON body."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                data = await resp.json()
                self._check_error(resp.status, data)
                return data
    
    def _check_error(self, status: int, data: dict) -> None:
        """Check for API errors."""
        if status == 401:
            raise AuthenticationError("Invalid xAI API key")
        elif status == 429:
            error = RateLimitError("xAI rate limit exceeded")
            raise error
        elif status >= 400:
            error_msg = data.get("error", {})
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", "Unknown error")
            raise GenerationError(f"xAI error: {error_msg}")
