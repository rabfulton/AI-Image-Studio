"""
Local Upscaler Provider - Uses upscayl-ncnn for image upscaling.

Provides 2x and 4x upscaling using Real-ESRGAN via the upscayl-ncnn binary.
The binary is automatically downloaded on first use if not found.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ai_image_studio.providers.base import (
    ImageProvider,
    GenerationRequest,
    GenerationResult,
    ProviderConfig,
    GenerationError,
)


class UpscalerProvider(ImageProvider):
    """Local image upscaling using upscayl-ncnn."""
    
    id = "upscaler"
    name = "Local Upscaler"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._binary_path: Path | None = None
    
    @property
    def is_configured(self) -> bool:
        """Check if upscayl-ncnn binary is available."""
        from ai_image_studio.providers.upscaler_download import find_upscaler_binary
        return find_upscaler_binary() is not None
    
    def _get_binary(self) -> Path:
        """Get the upscaler binary path, downloading if needed."""
        from ai_image_studio.providers.upscaler_download import (
            find_upscaler_binary,
            download_upscaler,
        )
        
        # Check cache
        if self._binary_path and self._binary_path.exists():
            return self._binary_path
        
        # Try to find existing binary
        binary = find_upscaler_binary()
        if binary:
            self._binary_path = binary
            return binary
        
        # Download it
        try:
            binary = download_upscaler()
            self._binary_path = binary
            return binary
        except Exception as e:
            raise GenerationError(
                f"Failed to download upscaler: {e}\n\n"
                "You can manually install upscayl-ncnn from:\n"
                "https://github.com/upscayl/upscayl-ncnn/releases"
            )
    
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Upscale an image using upscayl-ncnn."""
        import asyncio
        
        if not request.reference_images:
            raise GenerationError("Input image required for upscaling")
        
        input_image = request.reference_images[0]
        model_id = request.model.id
        
        # Determine scale factor from model id
        scale = 4 if "x4" in model_id else 2
        
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._upscale_sync,
            input_image,
            scale,
        )
        
        return GenerationResult(
            images=[result],
            model_id=model_id,
            prompt="",
        )
    
    def _upscale_sync(self, input_image, scale: int):
        """Synchronous upscaling using upscayl-ncnn CLI."""
        from ai_image_studio.core.data_types import ImageData
        import numpy as np
        from PIL import Image
        
        # Get binary (may trigger download)
        binary = self._get_binary()
        
        # Create temp files for input/output
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "output.png"
            
            # Save input image
            pil_img = input_image.to_pil().convert("RGB")
            pil_img.save(input_path, format="PNG")
            
            # Build command
            cmd = [
                str(binary),
                "-i", str(input_path),
                "-o", str(output_path),
                "-s", str(scale),
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    raise GenerationError(
                        f"Upscaling failed: {result.stderr or result.stdout}"
                    )
                
                if not output_path.exists():
                    raise GenerationError("Upscaling produced no output")
                
                # Load output image
                output_pil = Image.open(output_path).convert("RGB")
                output_np = np.array(output_pil).astype(np.float32) / 255.0
                
                return ImageData.from_numpy(output_np)
                
            except subprocess.TimeoutExpired:
                raise GenerationError("Upscaling timed out after 5 minutes")
            except FileNotFoundError:
                raise GenerationError(f"Binary not found: {binary}")
    
    async def validate_credentials(self) -> bool:
        """Check if upscayl-ncnn is available or can be downloaded."""
        return True  # Always valid - we can download if needed
    
    def ensure_binary_installed(self, parent=None) -> bool:
        """
        Ensure the binary is installed, showing download dialog if needed.
        
        Args:
            parent: Optional Qt parent widget for dialog
            
        Returns:
            True if binary is available, False if user cancelled
        """
        from ai_image_studio.providers.upscaler_download import (
            find_upscaler_binary,
            download_upscaler_sync_with_dialog,
        )
        
        if find_upscaler_binary():
            return True
        
        # Show download dialog
        path = download_upscaler_sync_with_dialog(parent)
        if path:
            self._binary_path = path
            return True
        return False
