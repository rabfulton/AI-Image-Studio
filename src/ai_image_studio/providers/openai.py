"""
OpenAI Provider - DALL-E and GPT Image models.

Supports:
- DALL-E 2: Text-to-image, image editing, inpainting
- DALL-E 3: Text-to-image with quality/style options
- GPT Image 1: Text-to-image with multiple inputs

API Reference: https://platform.openai.com/docs/api-reference/images
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


class OpenAIProvider(ImageProvider):
    """
    OpenAI image generation provider.
    
    Handles DALL-E 2, DALL-E 3, and GPT Image models.
    """
    
    id = "openai"
    name = "OpenAI"
    base_url = "https://api.openai.com/v1"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.base_url:
            self.base_url = config.base_url
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images using OpenAI API."""
        model_id = request.model.id
        
        # Choose endpoint based on whether we have reference images
        if request.reference_images and request.mask:
            return await self._generate_edit(request)
        elif request.reference_images:
            return await self._generate_variation(request)
        else:
            return await self._generate_create(request)
    
    async def _generate_create(self, request: GenerationRequest) -> GenerationResult:
        """Standard text-to-image generation."""
        url = f"{self.base_url}/images/generations"
        
        # Build request body
        body: dict[str, Any] = {
            "model": request.model.id,
            "prompt": request.prompt,
            "n": min(request.num_images, request.model.max_images),
            "size": f"{request.width}x{request.height}",
            "response_format": "b64_json",
        }
        
        # Add model-specific parameters
        extra = request.model.validate_params(request.extra_params)
        if "quality" in extra:
            body["quality"] = extra["quality"]
        if "style" in extra:
            body["style"] = extra["style"]
        if "background" in extra:
            body["background"] = extra["background"]
        
        response = await self._post(url, body)
        return self._parse_response(response, request)
    
    async def _generate_edit(self, request: GenerationRequest) -> GenerationResult:
        """Image editing / inpainting."""
        url = f"{self.base_url}/images/edits"
        
        # Prepare multipart form data
        form = aiohttp.FormData()
        form.add_field("model", request.model.id)
        form.add_field("prompt", request.prompt)
        form.add_field("n", str(request.num_images))
        form.add_field("size", f"{request.width}x{request.height}")
        form.add_field("response_format", "b64_json")
        
        # Add image
        if request.reference_images:
            img_bytes = self._image_to_png_bytes(request.reference_images[0])
            form.add_field("image", img_bytes, filename="image.png", content_type="image/png")
        
        # Add mask
        if request.mask:
            mask_bytes = self._mask_to_png_bytes(request.mask)
            form.add_field("mask", mask_bytes, filename="mask.png", content_type="image/png")
        
        response = await self._post_multipart(url, form)
        return self._parse_response(response, request)
    
    async def _generate_variation(self, request: GenerationRequest) -> GenerationResult:
        """Generate variations of an image."""
        url = f"{self.base_url}/images/variations"
        
        form = aiohttp.FormData()
        form.add_field("model", "dall-e-2")  # Only DALL-E 2 supports variations
        form.add_field("n", str(request.num_images))
        form.add_field("size", f"{request.width}x{request.height}")
        form.add_field("response_format", "b64_json")
        
        if request.reference_images:
            img_bytes = self._image_to_png_bytes(request.reference_images[0])
            form.add_field("image", img_bytes, filename="image.png", content_type="image/png")
        
        response = await self._post_multipart(url, form)
        return self._parse_response(response, request)
    
    async def validate_credentials(self) -> bool:
        """Validate API key by listing models."""
        try:
            url = f"{self.base_url}/models"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.get_headers()) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    def _parse_response(self, data: dict, request: GenerationRequest) -> GenerationResult:
        """Parse OpenAI response into GenerationResult."""
        from ai_image_studio.core.data_types import ImageData
        
        images = []
        revised_prompt = None
        
        for item in data.get("data", []):
            # Get image data
            if "b64_json" in item:
                img_bytes = base64.b64decode(item["b64_json"])
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                arr = np.asarray(pil_img).astype(np.float32) / 255.0
                images.append(ImageData.from_numpy(arr))
            
            # Get revised prompt if present
            if "revised_prompt" in item and not revised_prompt:
                revised_prompt = item["revised_prompt"]
        
        # Get token usage if present
        usage = data.get("usage", {})
        
        return GenerationResult(
            images=images,
            model_id=request.model.id,
            prompt=request.prompt,
            revised_prompt=revised_prompt,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
    
    async def _post(self, url: str, body: dict) -> dict:
        """Make POST request with JSON body."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=self.get_headers(),
            ) as resp:
                data = await resp.json()
                self._check_error(resp.status, data)
                return data
    
    async def _post_multipart(self, url: str, form: aiohttp.FormData) -> dict:
        """Make POST request with multipart form data."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, headers=headers) as resp:
                data = await resp.json()
                self._check_error(resp.status, data)
                return data
    
    def _check_error(self, status: int, data: dict) -> None:
        """Check for API errors."""
        if status == 401:
            raise AuthenticationError("Invalid OpenAI API key")
        elif status == 429:
            error = RateLimitError("OpenAI rate limit exceeded")
            error.retry_after = 60
            raise error
        elif status >= 400:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            raise GenerationError(f"OpenAI error: {error_msg}")
    
    def _image_to_png_bytes(self, image_data) -> bytes:
        """Convert ImageData to PNG bytes."""
        pil_img = image_data.to_pil()
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    
    def _mask_to_png_bytes(self, mask_data) -> bytes:
        """Convert MaskData to PNG bytes with alpha channel."""
        # Mask as RGBA where white = transparent (to be inpainted)
        arr = mask_data.pixels
        h, w = arr.shape[:2]
        
        # Create RGBA with mask as alpha (inverted: white in mask = transparent)
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, 3] = 1.0 - arr[:, :, 0]  # Invert for OpenAI convention
        
        pil_img = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
