"""
Stability AI Provider - Stable Diffusion models.

Supports:
- SD 3.5 Large/Medium/Turbo: High quality text-to-image and image-to-image
- Stable Image Ultra/Core: Latest premium models

API Reference: https://platform.stability.ai/docs/api-reference
Note: Stability AI uses multipart/form-data for requests
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


class StabilityProvider(ImageProvider):
    """
    Stability AI image generation provider.
    
    Handles SD 3.5 and Stable Image models using the v2beta API.
    Uses multipart/form-data for requests.
    """
    
    id = "stability"
    name = "Stability AI"
    base_url = "https://api.stability.ai/v2beta"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.base_url:
            self.base_url = config.base_url
    
    def get_headers(self) -> dict[str, str]:
        """Stability AI auth headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",  # Get base64 response
        }
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images using Stability AI API."""
        model_id = request.model.id
        
        # Determine endpoint based on model and mode
        # v2beta endpoints:
        # - stable-image/generate/ultra for stable-image-ultra
        # - stable-image/generate/core for stable-image-core  
        # - stable-image/generate/sd3 for sd3.5-* models
        if model_id == "stable-image-ultra":
            endpoint = "stable-image/generate/ultra"
        elif model_id == "stable-image-core":
            endpoint = "stable-image/generate/core"
        elif request.reference_images and request.model.supports_img2img:
            # Image-to-image uses control endpoint or different mode
            endpoint = "stable-image/control/sketch"  # Fallback - img2img via sd3 is different
        else:
            endpoint = "stable-image/generate/sd3"
        
        url = f"{self.base_url}/{endpoint}"
        
        # Build form data
        form_data = self._build_form_data(request)
        
        # Headers for multipart/form-data request
        # Note: Do NOT set Content-Type here - aiohttp sets it with boundary for FormData
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        
        # Make request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=form_data,
                headers=headers,
            ) as resp:
                # Check for errors FIRST before trying to parse response
                if resp.status >= 400:
                    try:
                        error_data = await resp.json()
                    except Exception:
                        error_data = {"message": await resp.text()}
                    self._check_error(resp.status, error_data)
                
                # Handle response based on content type
                if resp.content_type.startswith("image/"):
                    # Direct image response
                    img_bytes = await resp.read()
                    return self._parse_image_response(img_bytes, request)
                else:
                    # JSON response with base64
                    data = await resp.json()
                    return self._parse_json_response(data, request)
    
    def _build_form_data(self, request: GenerationRequest) -> aiohttp.FormData:
        """Build multipart form data for the request.
        
        Note: Stability AI API requires proper multipart/form-data formatting.
        We use quote_fields=False and explicit content_type for text fields.
        """
        form = aiohttp.FormData(quote_fields=False)
        
        # Required fields - must specify content_type for Stability API compatibility
        form.add_field("prompt", request.prompt, content_type="text/plain")
        form.add_field("model", request.model.id, content_type="text/plain")
        form.add_field("output_format", "png", content_type="text/plain")
        
        # Validated extra params from model card
        extra = request.model.validate_params(request.extra_params)
        
        # Optional text fields
        if request.negative_prompt:
            form.add_field("negative_prompt", request.negative_prompt, content_type="text/plain")
        
        # Seed
        if request.seed is not None and request.seed >= 0:
            form.add_field("seed", str(request.seed), content_type="text/plain")
        
        # Aspect ratio - use from extra_params if available
        if "aspect_ratio" in extra:
            form.add_field("aspect_ratio", str(extra["aspect_ratio"]), content_type="text/plain")
        
        # CFG scale
        if "cfg_scale" in extra:
            form.add_field("cfg_scale", str(extra["cfg_scale"]), content_type="text/plain")
        
        # Style preset
        if "style_preset" in extra:
            form.add_field("style_preset", str(extra["style_preset"]), content_type="text/plain")
        
        # Output format override
        if "output_format" in extra:
            # Remove the default and add the user choice
            form._fields = [f for f in form._fields if not (isinstance(f[0], dict) and f[0].get("name") == "output_format")]
            form.add_field("output_format", str(extra["output_format"]), content_type="text/plain")
        
        # Image-to-image specific
        if request.reference_images:
            image_data = request.reference_images[0]
            pil_img = image_data.to_pil()
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            
            form.add_field(
                "image",
                buf,
                filename="input.png",
                content_type="image/png",
            )
            form.add_field("mode", "image-to-image", content_type="text/plain")
            form.add_field("strength", str(request.strength), content_type="text/plain")
        
        return form
    
    def _parse_image_response(
        self, img_bytes: bytes, request: GenerationRequest
    ) -> GenerationResult:
        """Parse direct image response."""
        from ai_image_studio.core.data_types import ImageData
        
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        
        return GenerationResult(
            images=[ImageData.from_numpy(arr)],
            model_id=request.model.id,
            prompt=request.prompt,
        )
    
    def _parse_json_response(
        self, data: dict, request: GenerationRequest
    ) -> GenerationResult:
        """Parse JSON response with base64 image."""
        from ai_image_studio.core.data_types import ImageData
        
        images = []
        seed = None
        
        # Handle different response formats
        if "image" in data:
            # Single image response
            b64_data = data["image"]
            img_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
            arr = np.asarray(pil_img).astype(np.float32) / 255.0
            images.append(ImageData.from_numpy(arr))
            seed = data.get("seed")
        elif "artifacts" in data:
            # Legacy format with artifacts array
            for artifact in data["artifacts"]:
                if artifact.get("finishReason") == "SUCCESS":
                    b64_data = artifact["base64"]
                    img_bytes = base64.b64decode(b64_data)
                    pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                    arr = np.asarray(pil_img).astype(np.float32) / 255.0
                    images.append(ImageData.from_numpy(arr))
                    if seed is None:
                        seed = artifact.get("seed")
        
        if not images:
            raise GenerationError("No image data in Stability AI response")
        
        return GenerationResult(
            images=images,
            model_id=request.model.id,
            prompt=request.prompt,
            seed=seed,
        )
    
    async def validate_credentials(self) -> bool:
        """Validate API key by checking user account."""
        try:
            url = f"{self.base_url}/user/account"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    def _check_error(self, status: int, data: dict) -> None:
        """Check for API errors."""
        if status == 401:
            raise AuthenticationError("Invalid Stability AI API key")
        elif status == 402:
            raise GenerationError("Stability AI: Insufficient credits")
        elif status == 429:
            error = RateLimitError("Stability AI rate limit exceeded")
            raise error
        elif status >= 400:
            # Try to extract error message from various formats
            # Format 1: {"message": "error text"}
            # Format 2: {"error": "error text"}
            # Format 3: {"errors": [{"message": "..."}, ...]}
            # Format 4: {"name": "error_name", "message": "..."}
            error_msg = None
            
            if "errors" in data and isinstance(data["errors"], list):
                # Array of error objects - could be strings or dicts
                msgs = []
                for e in data["errors"]:
                    if isinstance(e, dict):
                        msgs.append(e.get("message", str(e)))
                    elif e:
                        msgs.append(str(e))
                error_msg = "; ".join(msgs) if msgs else None
            
            if not error_msg:
                error_msg = data.get("message") or data.get("error")
                
            if not error_msg and "name" in data:
                error_msg = f"{data['name']}: {data.get('message', 'No details')}"
            
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
            
            if not error_msg:
                # Last resort - stringify the whole response
                error_msg = str(data) if data else f"HTTP {status}"
            
            raise GenerationError(f"Stability AI error: {error_msg}")

