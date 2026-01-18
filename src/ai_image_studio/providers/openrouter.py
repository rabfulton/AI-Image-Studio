"""
OpenRouter Provider - Multi-model proxy.

OpenRouter provides access to many image generation models through a unified API.
Uses the chat completions endpoint with modalities=["image", "text"].

API Reference: https://openrouter.ai/docs#image-generation
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


class OpenRouterProvider(ImageProvider):
    """
    OpenRouter image generation provider.
    
    Uses chat completions API with modalities=["image", "text"].
    Can proxy to various backend models like Gemini, FLUX, etc.
    """
    
    id = "openrouter"
    name = "OpenRouter"
    base_url = "https://openrouter.ai/api/v1"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.base_url:
            self.base_url = config.base_url
    
    def get_headers(self) -> dict[str, str]:
        """OpenRouter headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-image-studio.local",  # Required by OpenRouter
            "X-Title": "AI Image Studio",
        }
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images using OpenRouter chat completions API."""
        url = f"{self.base_url}/chat/completions"
        
        # Build message content
        content: list[dict[str, Any]] = []
        
        # Add reference images if present
        if request.reference_images:
            for img in request.reference_images:
                b64 = self._image_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                    },
                })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": request.prompt,
        })
        
        # Build request
        body: dict[str, Any] = {
            "model": request.model.id,
            "messages": [
                {"role": "user", "content": content},
            ],
            "modalities": ["image", "text"],
        }
        
        # Build image_config from model parameters
        image_config: dict[str, Any] = {}
        extra = request.model.validate_params(request.extra_params)
        
        # Use aspect_ratio from params, or calculate from dimensions
        if "aspect_ratio" in extra:
            image_config["aspect_ratio"] = extra["aspect_ratio"]
        elif request.model.aspect_ratios:
            # Calculate closest aspect ratio from dimensions
            aspect = self._get_aspect_ratio(request.width, request.height)
            if aspect:
                image_config["aspect_ratio"] = aspect
        
        # Add image_size if present (Gemini Pro only)
        if "image_size" in extra:
            image_config["image_size"] = extra["image_size"]
        
        if image_config:
            body["image_config"] = image_config
        
        # Maximum tokens for text response
        body["max_tokens"] = 300
        
        response = await self._post(url, body)
        return self._parse_response(response, request)
    
    def _get_aspect_ratio(self, width: int, height: int) -> str | None:
        """Calculate closest standard aspect ratio."""
        ratio = width / height
        
        ratios = {
            "1:1": 1.0,
            "16:9": 16/9,
            "9:16": 9/16,
            "4:3": 4/3,
            "3:4": 3/4,
            "3:2": 3/2,
            "2:3": 2/3,
            "4:5": 4/5,
            "5:4": 5/4,
            "21:9": 21/9,
            "9:21": 9/21,
        }
        
        # Find closest
        closest = min(ratios.items(), key=lambda x: abs(x[1] - ratio))
        return closest[0]
    
    def _parse_response(self, data: dict, request: GenerationRequest) -> GenerationResult:
        """Parse OpenRouter response."""
        from ai_image_studio.core.data_types import ImageData
        
        images = []
        
        # Extract images from response
        choices = data.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            
            # Check for images array (newer format)
            for img_data in message.get("images", []):
                if isinstance(img_data, str):
                    # Base64 or data URL
                    images.extend(self._decode_image(img_data))
            
            # Check content array
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        images.extend(self._decode_image(url))
            elif isinstance(content, str) and content.startswith("data:image"):
                images.extend(self._decode_image(content))
        
        # Get usage info
        usage = data.get("usage", {})
        
        return GenerationResult(
            images=images,
            model_id=request.model.id,
            prompt=request.prompt,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
    
    def _decode_image(self, data: str) -> list:
        """Decode image from base64 or data URL."""
        from ai_image_studio.core.data_types import ImageData
        
        if not data:
            return []
        
        try:
            # Handle data URL
            if data.startswith("data:image"):
                b64_data = data.split(",", 1)[1]
            else:
                b64_data = data
            
            img_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
            arr = np.asarray(pil_img).astype(np.float32) / 255.0
            return [ImageData.from_numpy(arr)]
        except Exception:
            return []
    
    async def validate_credentials(self) -> bool:
        """Validate API key."""
        try:
            url = f"{self.base_url}/models"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.get_headers()) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def list_available_models(self) -> list[str]:
        """List models that support image generation."""
        url = f"{self.base_url}/models"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.get_headers()) as resp:
                    if resp.status != 200:
                        return []
                    
                    data = await resp.json()
                    
                    # Filter to models with image output
                    image_models = []
                    for model in data.get("data", []):
                        modalities = model.get("output_modalities", [])
                        if "image" in modalities:
                            image_models.append(model.get("id", ""))
                    
                    return image_models
        except Exception:
            return []
    
    async def _post(self, url: str, body: dict) -> dict:
        """Make POST request."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=self.get_headers(),
            ) as resp:
                data = await resp.json()
                self._check_error(resp.status, data)
                return data
    
    def _check_error(self, status: int, data: dict) -> None:
        """Check for API errors."""
        if status == 401:
            raise AuthenticationError("Invalid OpenRouter API key")
        elif status == 429:
            error = RateLimitError("OpenRouter rate limit exceeded")
            raise error
        elif status >= 400:
            error_msg = data.get("error", {})
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", "Unknown error")
            raise GenerationError(f"OpenRouter error: {error_msg}")
    
    def _image_to_base64(self, image_data) -> str:
        """Convert ImageData to base64 string."""
        pil_img = image_data.to_pil()
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
