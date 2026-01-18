"""
Black Forest Labs Provider - FLUX models.

Supports:
- FLUX 2 Pro/Max: Latest generation
- FLUX Kontext Pro/Max: Image editing
- FLUX Pro 1.1 Ultra/Pro: High quality text-to-image
- FLUX Dev: Development model with img2img
- FLUX Schnell: Fast inference

API Reference: https://docs.bfl.ai/
Note: BFL uses async polling for results
"""

from __future__ import annotations

import asyncio
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


class BFLProvider(ImageProvider):
    """
    Black Forest Labs image generation provider.
    
    Handles FLUX Pro, FLUX Dev, FLUX Schnell, and FLUX Kontext models.
    Uses async polling pattern for generation.
    """
    
    id = "bfl"
    name = "Black Forest Labs"
    base_url = "https://api.bfl.ai/v1"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.base_url:
            self.base_url = config.base_url
    
    def get_headers(self) -> dict[str, str]:
        """BFL uses X-Key header for auth."""
        return {
            "X-Key": self.api_key,
            "Content-Type": "application/json",
        }
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images using BFL API."""
        model_id = request.model.id
        
        # Model ID is the endpoint name
        endpoint = model_id
        url = f"{self.base_url}/{endpoint}"
        
        # Build request body
        body: dict[str, Any] = {
            "prompt": request.prompt,
        }
        
        # All BFL FLUX models use width/height parameters
        body["width"] = request.width
        body["height"] = request.height
        
        # Add optional parameters based on model
        extra = request.model.validate_params(request.extra_params)
        
        if request.seed is not None:
            body["seed"] = request.seed
        if "steps" in extra:
            body["steps"] = extra["steps"]
        if "guidance" in extra:
            body["guidance"] = extra["guidance"]
        if "safety_tolerance" in extra:
            body["safety_tolerance"] = int(extra["safety_tolerance"])
        if "prompt_upsampling" in extra:
            body["prompt_upsampling"] = extra["prompt_upsampling"]
        if "output_format" in extra:
            body["output_format"] = extra["output_format"]
        
        # Handle image input for editing
        if request.reference_images and request.model.supports_img2img:
            if model_id.startswith("flux-kontext"):
                # Kontext uses input_image parameter
                body["input_image"] = self._image_to_base64(request.reference_images[0])
            else:
                # FLUX Dev uses image_prompt parameter
                body["image_prompt"] = self._image_to_base64(request.reference_images[0])
        
        # Submit generation request
        task_id, polling_url = await self._submit_generation(url, body)
        
        # Poll for result using the returned polling URL
        result = await self._poll_result(polling_url)
        
        return await self._parse_result(result, request)
    
    async def _submit_generation(self, url: str, body: dict) -> tuple[str, str]:
        """Submit generation request and get task ID and polling URL."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=self.get_headers(),
            ) as resp:
                data = await resp.json()
                self._check_error(resp.status, data)
                
                task_id = data.get("id")
                if not task_id:
                    raise GenerationError("No task ID in BFL response")
                
                # BFL returns a polling_url to use for status checks
                polling_url = data.get("polling_url")
                if not polling_url:
                    # Fallback to the traditional get_result endpoint
                    polling_url = f"{self.base_url}/get_result?id={task_id}"
                
                return task_id, polling_url
    
    async def _poll_result(self, polling_url: str, timeout: float = 300) -> dict:
        """Poll for generation result using the provided polling URL."""
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise GenerationError("BFL generation timed out")
                
                async with session.get(
                    polling_url,
                    headers=self.get_headers(),
                ) as resp:
                    data = await resp.json()
                    
                    status = data.get("status")
                    
                    if status == "Ready":
                        return data
                    elif status == "Error":
                        raise GenerationError(f"BFL error: {data.get('error', 'Unknown')}")
                    elif status == "Task not found":
                        raise GenerationError(
                            "BFL task not found - the generation request may have failed "
                            "or expired. Please try again."
                        )
                    elif status in ("Pending", "Processing", "Queued"):
                        # Wait and retry
                        await asyncio.sleep(1.0)
                    else:
                        raise GenerationError(f"Unknown BFL status: {status}")
    
    async def _parse_result(self, data: dict, request: GenerationRequest) -> GenerationResult:
        """Parse BFL result into GenerationResult."""
        from ai_image_studio.core.data_types import ImageData
        
        images = []
        result_data = data.get("result", {})
        
        # BFL returns a URL to the generated image in result.sample
        sample = result_data.get("sample")
        
        if sample:
            if sample.startswith("http://") or sample.startswith("https://"):
                # It's a URL - download the image
                async with aiohttp.ClientSession() as session:
                    async with session.get(sample) as resp:
                        if resp.status == 200:
                            img_bytes = await resp.read()
                            pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                            arr = np.asarray(pil_img).astype(np.float32) / 255.0
                            images.append(ImageData.from_numpy(arr))
                        else:
                            raise GenerationError(f"Failed to download image: HTTP {resp.status}")
            elif sample.startswith("data:image"):
                # Data URI with base64
                b64_data = sample.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                arr = np.asarray(pil_img).astype(np.float32) / 255.0
                images.append(ImageData.from_numpy(arr))
            else:
                # Direct base64
                img_bytes = base64.b64decode(sample)
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
                arr = np.asarray(pil_img).astype(np.float32) / 255.0
                images.append(ImageData.from_numpy(arr))
        
        if not images:
            raise GenerationError("No image data in BFL response")
        
        return GenerationResult(
            images=images,
            model_id=request.model.id,
            prompt=request.prompt,
            seed=result_data.get("seed"),
        )
    
    async def validate_credentials(self) -> bool:
        """Validate API key."""
        try:
            # Try a minimal request to check auth
            url = f"{self.base_url}/flux-schnell"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={"prompt": "test", "width": 256, "height": 256},
                    headers=self.get_headers(),
                ) as resp:
                    if resp.status == 401:
                        return False
                    # Cancel the generation if it started
                    return True
        except Exception:
            return False
    
    def _check_error(self, status: int, data: dict) -> None:
        """Check for API errors."""
        if status == 401:
            raise AuthenticationError("Invalid BFL API key")
        elif status == 429:
            error = RateLimitError("BFL rate limit exceeded")
            error.retry_after = 60
            raise error
        elif status >= 400:
            error_msg = data.get("message", data.get("error", "Unknown error"))
            raise GenerationError(f"BFL error: {error_msg}")
    
    def _image_to_base64(self, image_data) -> str:
        """Convert ImageData to base64 string."""
        pil_img = image_data.to_pil()
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
