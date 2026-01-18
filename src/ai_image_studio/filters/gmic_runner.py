"""
G'MIC Runner - Core wrapper for G'MIC image processing operations.

This module provides the GmicRunner class which handles:
- Initializing the G'MIC interpreter
- Converting between ImageData and G'MIC format
- Executing filter commands with parameters
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ai_image_studio.core.data_types import ImageData

logger = logging.getLogger(__name__)


# Check if gmic is available
try:
    import gmic
    GMIC_AVAILABLE = True
except ImportError:
    GMIC_AVAILABLE = False
    logger.warning("gmic-py not installed. Install with: pip install gmic")


class GmicError(Exception):
    """Error during G'MIC filter execution."""
    pass


class GmicRunner:
    """
    Wrapper for G'MIC image processing operations.
    
    Usage:
        runner = GmicRunner()
        result = runner.apply_filter(image_data, "blur", {"sigma": 2.0})
    """
    
    def __init__(self):
        """Initialize the G'MIC interpreter."""
        if not GMIC_AVAILABLE:
            raise GmicError("gmic-py is not installed. Install with: pip install gmic")
        
        # Create interpreter instance for better performance across calls
        self._interpreter = gmic.Gmic()
    
    @property
    def is_available(self) -> bool:
        """Check if G'MIC is available."""
        return GMIC_AVAILABLE
    
    def apply_filter(
        self,
        image: "ImageData",
        command: str,
        params: dict[str, Any] | None = None,
    ) -> "ImageData":
        """
        Apply a G'MIC filter to an image.
        
        Args:
            image: Input image as ImageData
            command: G'MIC command string (e.g., "blur", "sharpen")
            params: Optional parameters to substitute into command
            
        Returns:
            Filtered image as ImageData
            
        Raises:
            GmicError: If filter execution fails
        """
        from ai_image_studio.core.data_types import ImageData
        
        try:
            # Convert ImageData to G'MIC image list
            gmic_images = self._to_gmic_images(image)
            
            # Build command with parameters
            full_command = self._build_command(command, params)
            
            logger.debug(f"Executing G'MIC command: {full_command}")
            
            # Run the filter
            self._interpreter.run(full_command, gmic_images)
            
            # Convert back to ImageData
            if len(gmic_images) == 0:
                raise GmicError("G'MIC filter produced no output")
            
            return self._from_gmic_image(gmic_images[0])
            
        except Exception as e:
            if "gmic" in str(type(e).__module__).lower():
                raise GmicError(f"G'MIC error: {e}") from e
            raise
    
    def _to_gmic_images(self, image: "ImageData") -> list:
        """Convert ImageData to G'MIC GmicImage list."""
        # Get numpy array (H, W, C) with values 0-1 as float32
        arr = image.to_numpy(dtype=np.float32)
        
        # Scale to 0-255 (G'MIC default range)
        arr = (arr * 255.0).astype(np.float32)
        
        # Ensure 3D array (H, W, C) - handle grayscale
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        
        # Only take RGB (3 channels) for G'MIC - drop alpha if present
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        
        # Use from_numpy_helper with:
        # - deinterleave=True (convert RGB,RGB,RGB -> RRR,GGG,BBB)
        # - permute='yxzc' to convert (H,W,1,C) to G'MIC's native (W,H,D,S) format
        # First expand to 4D: (H, W, C) -> (H, W, 1, C) for the z dimension
        arr = arr[:, :, np.newaxis, :]  # (H, W, 1, C)
        
        gmic_img = gmic.GmicImage.from_numpy_helper(arr, deinterleave=True, permute='yxzc')
        
        return [gmic_img]
    
    def _from_gmic_image(self, gmic_img) -> "ImageData":
        """Convert G'MIC GmicImage back to ImageData."""
        from ai_image_studio.core.data_types import ImageData
        
        # Use to_numpy_helper to:
        # - interleave=True (convert RRR,GGG,BBB -> RGB,RGB,RGB)
        # - permute='yxzc' to get back (H,W,D,C) format from G'MIC's (W,H,D,S)
        # - squeeze_shape=True to remove dimensions of size 1
        arr = gmic_img.to_numpy_helper(
            interleave=True,
            permute='yxzc',
            squeeze_shape=True,
        )
        
        # Scale back to 0-1
        arr = arr.astype(np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        
        # Handle different dimensional outputs
        if arr.ndim == 2:
            # Grayscale -> RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            # Single channel -> RGB
            arr = np.concatenate([arr, arr, arr], axis=-1)
        
        # Add alpha channel if needed
        if arr.ndim == 3 and arr.shape[2] == 3:
            alpha = np.ones((*arr.shape[:2], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] > 4:
            # Too many channels, take first 3 and add alpha
            arr = arr[:, :, :3]
            alpha = np.ones((*arr.shape[:2], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, alpha], axis=-1)
        
        return ImageData.from_numpy(arr)
    
    def _build_command(
        self,
        command: str,
        params: dict[str, Any] | None,
    ) -> str:
        """Build full G'MIC command string with parameters."""
        if not params:
            return command
        
        # Convert params dict to comma-separated values
        # Most G'MIC commands use positional parameters
        param_str = ",".join(str(v) for v in params.values())
        
        if param_str:
            return f"{command} {param_str}"
        return command
    
    def get_help(self, command: str) -> str:
        """Get help text for a G'MIC command."""
        try:
            # Capture help output
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            self._interpreter.run(f"help {command}")
            
            help_text = buffer.getvalue()
            sys.stdout = old_stdout
            
            return help_text
        except Exception:
            return f"No help available for '{command}'"
