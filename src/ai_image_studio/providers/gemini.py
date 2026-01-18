"""
Google Gemini Provider - Imagen and Gemini Image models.

Supports:
- Imagen 3/4: Text-to-image via :predict endpoint
- Gemini 2.5 Flash Image: Fast generation and editing via :generateContent
- Gemini 3 Pro Image: High-quality generation and editing up to 4K

API References:
- https://ai.google.dev/gemini-api/docs/imagen
- https://ai.google.dev/gemini-api/docs/image-generation
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


class GeminiProvider(ImageProvider):
    """
    Google Gemini/Imagen image generation provider.
    
    Handles both Imagen models (text-to-image only) and Gemini Image models
    (text-to-image + image editing).
    """
    
    id = "gemini"
    name = "Google Gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.base_url:
            self.base_url = config.base_url
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images using Google's APIs."""
        model_id = request.model.id
        
        # Route to appropriate endpoint based on model type
        if model_id.startswith("imagen"):
            return await self._generate_imagen(request)
        else:
            return await self._generate_gemini(request)
    
    async def _generate_imagen(self, request: GenerationRequest) -> GenerationResult:
        """Generate images using Imagen models via :predict endpoint.
        
        Imagen models only support text-to-image generation.
        """
        url = f"{self.base_url}/models/{request.model.id}:predict"
        
        # Build request body
        extra = request.model.validate_params(request.extra_params)
        
        # Parameters for Imagen
        parameters: dict[str, Any] = {}
        
        if "numberOfImages" in extra:
            parameters["numberOfImages"] = int(extra["numberOfImages"])
        else:
            parameters["numberOfImages"] = min(request.num_images, request.model.max_images)
        
        if "aspectRatio" in extra:
            parameters["aspectRatio"] = extra["aspectRatio"]
        
        if "personGeneration" in extra:
            parameters["personGeneration"] = extra["personGeneration"]
        
        if "imageSize" in extra:
            parameters["imageSize"] = extra["imageSize"]
        
        body = {
            "instances": [
                {"prompt": request.prompt}
            ],
            "parameters": parameters,
        }
        
        response = await self._post(url, body)
        return self._parse_imagen_response(response, request)
    
    async def _generate_gemini(self, request: GenerationRequest) -> GenerationResult:
        """Generate/edit images using Gemini models via :generateContent endpoint.
        
        Gemini Image models support both text-to-image and image editing.
        """
        url = f"{self.base_url}/models/{request.model.id}:generateContent"
        
        extra = request.model.validate_params(request.extra_params)
        
        # Build contents - can include text and images
        parts = []
        
        # Add reference images if provided (for image editing)
        if request.reference_images:
            for img in request.reference_images:
                img_bytes = self._image_to_png_bytes(img)
                b64 = base64.b64encode(img_bytes).decode()
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64,
                    }
                })
        
        # Add text prompt
        parts.append({"text": request.prompt})
        
        # Build generation config
        generation_config: dict[str, Any] = {}
        
        # Response modalities - default to Image only
        modalities = extra.get("response_modalities", "Image")
        if modalities == "Text,Image":
            generation_config["responseModalities"] = ["Text", "Image"]
        else:
            generation_config["responseModalities"] = ["Image"]
        
        # Image config
        image_config: dict[str, Any] = {}
        
        if "aspectRatio" in extra:
            image_config["aspectRatio"] = extra["aspectRatio"]
        
        if "imageSize" in extra:
            image_config["imageSize"] = extra["imageSize"]
        
        if image_config:
            generation_config["imageConfig"] = image_config
        
        body = {
            "contents": [{"parts": parts}],
        }
        
        if generation_config:
            body["generationConfig"] = generation_config
        
        response = await self._post(url, body)
        return self._parse_gemini_response(response, request)
    
    async def validate_credentials(self) -> bool:
        """Validate API key by listing models."""
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    def _parse_imagen_response(self, data: dict, request: GenerationRequest) -> GenerationResult:
        """Parse Imagen :predict response into GenerationResult."""
        from ai_image_studio.core.data_types import ImageData
        
        images = []
        
        for prediction in data.get("predictions", []):
            if "bytesBase64Encoded" in prediction:
                img_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                arr = np.asarray(pil_img).astype(np.float32) / 255.0
                images.append(ImageData.from_numpy(arr))
        
        return GenerationResult(
            images=images,
            model_id=request.model.id,
            prompt=request.prompt,
        )
    
    def _parse_gemini_response(self, data: dict, request: GenerationRequest) -> GenerationResult:
        """Parse Gemini :generateContent response into GenerationResult."""
        from ai_image_studio.core.data_types import ImageData
        
        images = []
        text_response = None
        
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                # Handle image parts
                if "inlineData" in part:
                    inline_data = part["inlineData"]
                    img_bytes = base64.b64decode(inline_data["data"])
                    pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                    arr = np.asarray(pil_img).astype(np.float32) / 255.0
                    images.append(ImageData.from_numpy(arr))
                
                # Handle text parts (for Text+Image mode)
                if "text" in part and not text_response:
                    text_response = part["text"]
        
        return GenerationResult(
            images=images,
            model_id=request.model.id,
            prompt=request.prompt,
            revised_prompt=text_response,  # Use text response as revised prompt
        )
    
    async def _post(self, url: str, body: dict) -> dict:
        """Make POST request with JSON body and API key in query string."""
        # Google API uses key in query string
        url_with_key = f"{url}?key={self.api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url_with_key,
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                data = await resp.json()
                self._check_error(resp.status, data)
                return data
    
    def _check_error(self, status: int, data: dict) -> None:
        """Check for API errors."""
        if status == 401 or status == 403:
            raise AuthenticationError("Invalid Google API key")
        elif status == 429:
            error = RateLimitError("Google API rate limit exceeded")
            error.retry_after = 60
            raise error
        elif status >= 400:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            raise GenerationError(f"Google API error: {error_msg}")
    
    def _image_to_png_bytes(self, image_data) -> bytes:
        """Convert ImageData to PNG bytes."""
        pil_img = image_data.to_pil()
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
