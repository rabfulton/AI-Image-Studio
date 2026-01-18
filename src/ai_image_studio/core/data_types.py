"""
Data Types - Core data structures for image and processing data.

This module defines the data types that flow through the node graph:
- DataType: Enum of all supported data types
- ImageData: Container for image pixels and metadata
- MaskData: Container for mask/selection data
- Various utility types

Reference: architecture.md#3-core-data-structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


class DataType(Enum):
    """
    Enumeration of data types that can flow through node connections.
    
    Each input/output socket has a DataType that determines what
    kinds of connections are valid.
    """
    # Image types
    IMAGE = auto()          # RGB/RGBA image
    MASK = auto()           # Single-channel mask
    LATENT = auto()         # Latent space representation
    
    # Primitive types
    TEXT = auto()           # String/prompt
    NUMBER = auto()         # Float number
    INTEGER = auto()        # Integer number
    BOOLEAN = auto()        # True/False
    
    # Complex types
    CONDITIONING = auto()   # CLIP embeddings / conditioning
    MODEL = auto()          # AI model reference
    CONTROLNET = auto()     # ControlNet model
    VAE = auto()            # VAE model
    LORA = auto()           # LoRA weights
    
    # Collection types
    IMAGE_LIST = auto()     # List of images
    
    # Special types
    ANY = auto()            # Accepts any type (for utility nodes)
    
    def is_compatible_with(self, other: DataType) -> bool:
        """Check if this type can connect to another type."""
        if self == DataType.ANY or other == DataType.ANY:
            return True
        return self == other


# Type alias for parameter values
ParameterValue: TypeAlias = str | int | float | bool | list | dict | None


@dataclass
class ImageMetadata:
    """Metadata associated with an image."""
    
    # Source information
    source_path: Path | None = None
    source_node_id: str | None = None
    
    # Generation parameters (if AI-generated)
    prompt: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    sampler: str | None = None
    model: str | None = None
    
    # Image properties
    original_width: int | None = None
    original_height: int | None = None
    color_space: str = "sRGB"
    
    # Custom metadata
    custom: dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> ImageMetadata:
        """Create a shallow copy of this metadata."""
        return ImageMetadata(
            source_path=self.source_path,
            source_node_id=self.source_node_id,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            sampler=self.sampler,
            model=self.model,
            original_width=self.original_width,
            original_height=self.original_height,
            color_space=self.color_space,
            custom=self.custom.copy(),
        )


@dataclass
class ImageData:
    """
    Container for image data flowing through the node graph.
    
    Internally stores pixels as a numpy array in HWC format with
    float32 values in range [0, 1].
    
    Attributes:
        pixels: numpy array of shape (H, W, C) with float32 values [0, 1]
        metadata: Optional metadata about the image
    """
    pixels: NDArray[np.float32]
    metadata: ImageMetadata = field(default_factory=ImageMetadata)
    
    @classmethod
    def from_numpy(
        cls,
        array: NDArray,
        metadata: ImageMetadata | None = None
    ) -> ImageData:
        """
        Create ImageData from a numpy array.
        
        Handles various input formats:
        - uint8 [0, 255] -> float32 [0, 1]
        - float64 -> float32
        - HW (grayscale) -> HWC
        - CHW -> HWC
        """
        arr = array.copy()
        
        # Convert to float32 if needed
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        
        # Ensure HWC format
        if arr.ndim == 2:
            # Grayscale -> RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            # CHW -> HWC (if first dim is channels)
            if arr.shape[0] < arr.shape[2]:
                arr = np.transpose(arr, (1, 2, 0))
        
        return cls(pixels=arr, metadata=metadata or ImageMetadata())
    
    @classmethod
    def from_pil(cls, image, metadata: ImageMetadata | None = None) -> ImageData:
        """Create ImageData from a PIL Image."""
        from PIL import Image
        
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        
        # Convert to RGB/RGBA
        if image.mode == "L":
            image = image.convert("RGB")
        elif image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        
        arr = np.array(image, dtype=np.float32) / 255.0
        
        meta = metadata or ImageMetadata()
        meta.original_width = image.width
        meta.original_height = image.height
        
        return cls(pixels=arr, metadata=meta)
    
    @classmethod
    def from_file(cls, path: str | Path, metadata: ImageMetadata | None = None) -> ImageData:
        """
        Create ImageData by loading an image from a file.
        
        Args:
            path: Path to the image file
            metadata: Optional metadata (source_path will be set automatically)
        
        Returns:
            ImageData with the loaded image
        """
        from PIL import Image
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        image = Image.open(path)
        
        meta = metadata or ImageMetadata()
        meta.source_path = path
        
        return cls.from_pil(image, meta)
    
    @classmethod
    def empty(cls, width: int, height: int, channels: int = 3) -> ImageData:
        """Create an empty (black) image of the given size."""
        arr = np.zeros((height, width, channels), dtype=np.float32)
        return cls(pixels=arr)
    
    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.pixels.shape[1]
    
    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.pixels.shape[0]
    
    @property
    def channels(self) -> int:
        """Number of color channels (3 for RGB, 4 for RGBA)."""
        return self.pixels.shape[2] if self.pixels.ndim == 3 else 1
    
    @property
    def size(self) -> tuple[int, int]:
        """Image size as (width, height)."""
        return (self.width, self.height)
    
    @property
    def has_alpha(self) -> bool:
        """Check if image has an alpha channel."""
        return self.channels == 4
    
    def to_numpy(self, dtype: np.dtype = np.float32) -> NDArray:
        """
        Convert to numpy array.
        
        Args:
            dtype: Output dtype (float32, uint8, etc.)
        
        Returns:
            Array in HWC format
        """
        if dtype == np.uint8:
            return (self.pixels * 255).clip(0, 255).astype(np.uint8)
        return self.pixels.astype(dtype)
    
    def to_pil(self):
        """Convert to PIL Image."""
        from PIL import Image
        
        arr = self.to_numpy(np.uint8)
        mode = "RGBA" if self.has_alpha else "RGB"
        return Image.fromarray(arr, mode=mode)
    
    def to_tensor(self, normalize: bool = True):
        """
        Convert to PyTorch tensor (if available).
        
        Args:
            normalize: If True, normalize to [-1, 1] range (for diffusion models)
        
        Returns:
            Tensor in NCHW format
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        # HWC -> CHW
        arr = np.transpose(self.pixels, (2, 0, 1))
        
        if normalize:
            arr = arr * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        # Add batch dimension: CHW -> NCHW
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor
    
    @classmethod
    def from_tensor(
        cls,
        tensor,
        normalize: bool = True,
        metadata: ImageMetadata | None = None
    ) -> ImageData:
        """
        Create ImageData from a PyTorch tensor.
        
        Args:
            tensor: Tensor in NCHW or CHW format
            normalize: If True, assume tensor is in [-1, 1] range
            metadata: Optional metadata
        """
        arr = tensor.detach().cpu().numpy()
        
        # Remove batch dimension if present
        if arr.ndim == 4:
            arr = arr[0]  # NCHW -> CHW
        
        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
        
        if normalize:
            arr = (arr + 1.0) / 2.0  # [-1,1] -> [0,1]
        
        arr = arr.clip(0, 1).astype(np.float32)
        
        return cls(pixels=arr, metadata=metadata or ImageMetadata())
    
    def thumbnail(self, max_size: int = 256) -> ImageData:
        """Create a thumbnail of this image."""
        from PIL import Image
        
        pil_image = self.to_pil()
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        meta = self.metadata.copy()
        return ImageData.from_pil(pil_image, meta)
    
    def resize(self, width: int, height: int) -> ImageData:
        """Resize the image to the given dimensions."""
        from PIL import Image
        
        pil_image = self.to_pil()
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        
        meta = self.metadata.copy()
        return ImageData.from_pil(pil_image, meta)
    
    def copy(self) -> ImageData:
        """Create a copy of this image."""
        return ImageData(
            pixels=self.pixels.copy(),
            metadata=self.metadata.copy(),
        )


@dataclass
class MaskData:
    """
    Container for mask/selection data.
    
    Masks are single-channel images with values in [0, 1] where:
    - 0 = fully masked (excluded)
    - 1 = fully unmasked (included)
    """
    pixels: NDArray[np.float32]  # Shape: (H, W)
    
    @classmethod
    def from_numpy(cls, array: NDArray) -> MaskData:
        """Create mask from numpy array."""
        arr = array.copy()
        
        # Convert to float32
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        
        # Ensure 2D
        if arr.ndim == 3:
            arr = arr[:, :, 0]  # Take first channel
        
        return cls(pixels=arr)
    
    @classmethod
    def from_image(cls, image: ImageData, threshold: float = 0.5) -> MaskData:
        """
        Create mask from an image.
        
        Converts to grayscale and thresholds.
        """
        # Convert to grayscale using luminance formula
        if image.channels >= 3:
            gray = (
                0.299 * image.pixels[:, :, 0] +
                0.587 * image.pixels[:, :, 1] +
                0.114 * image.pixels[:, :, 2]
            )
        else:
            gray = image.pixels[:, :, 0]
        
        # Apply threshold
        mask = (gray > threshold).astype(np.float32)
        return cls(pixels=mask)
    
    @classmethod
    def empty(cls, width: int, height: int, value: float = 0.0) -> MaskData:
        """Create an empty mask of the given size."""
        arr = np.full((height, width), value, dtype=np.float32)
        return cls(pixels=arr)
    
    @classmethod
    def full(cls, width: int, height: int) -> MaskData:
        """Create a fully selected mask."""
        return cls.empty(width, height, value=1.0)
    
    @property
    def width(self) -> int:
        return self.pixels.shape[1]
    
    @property
    def height(self) -> int:
        return self.pixels.shape[0]
    
    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)
    
    def to_numpy(self, dtype: np.dtype = np.float32) -> NDArray:
        """Convert to numpy array."""
        if dtype == np.uint8:
            return (self.pixels * 255).clip(0, 255).astype(np.uint8)
        return self.pixels.astype(dtype)
    
    def to_image(self) -> ImageData:
        """Convert mask to grayscale image."""
        arr = np.stack([self.pixels, self.pixels, self.pixels], axis=-1)
        return ImageData(pixels=arr)
    
    def to_pil(self):
        """Convert to PIL Image (grayscale)."""
        from PIL import Image
        
        arr = self.to_numpy(np.uint8)
        return Image.fromarray(arr, mode="L")
    
    def invert(self) -> MaskData:
        """Invert the mask."""
        return MaskData(pixels=1.0 - self.pixels)
    
    def blur(self, radius: int = 5) -> MaskData:
        """Apply Gaussian blur to the mask."""
        from PIL import ImageFilter
        
        pil_mask = self.to_pil()
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=radius))
        
        arr = np.array(pil_mask, dtype=np.float32) / 255.0
        return MaskData(pixels=arr)
    
    def copy(self) -> MaskData:
        """Create a copy of this mask."""
        return MaskData(pixels=self.pixels.copy())
