"""
Filter Registry - Curated collection of G'MIC filters with parameter specs.

This module defines FilterSpec and FilterParam dataclasses for describing
filters and their parameters, along with a registry of useful filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ParamType(Enum):
    """Parameter types for filter parameters."""
    FLOAT = "float"      # Slider with float values
    INT = "int"          # Spinner with int values
    BOOL = "bool"        # Checkbox
    CHOICE = "choice"    # Dropdown with options
    COLOR = "color"      # Color picker (R,G,B)
    TEXT = "text"        # Text input


@dataclass
class FilterParam:
    """
    Specification for a filter parameter.
    
    Attributes:
        name: Parameter name (also used as label)
        param_type: Type of parameter
        default: Default value
        min_value: Minimum value (for FLOAT/INT)
        max_value: Maximum value (for FLOAT/INT)
        options: List of options (for CHOICE type)
        description: Optional description for tooltip
    """
    name: str
    param_type: ParamType
    default: any = 0
    min_value: float | None = None
    max_value: float | None = None
    options: list[str] | None = None
    description: str = ""


@dataclass
class FilterSpec:
    """
    Specification for a G'MIC filter.
    
    Attributes:
        id: Unique filter identifier
        name: Display name
        command: G'MIC command string
        category: Filter category for grouping
        params: List of parameter specifications
        description: Filter description
    """
    id: str
    name: str
    command: str
    category: str
    params: list[FilterParam] = field(default_factory=list)
    description: str = ""


# Curated filter registry
_FILTERS: dict[str, FilterSpec] = {}


def _register(spec: FilterSpec) -> FilterSpec:
    """Register a filter specification."""
    _FILTERS[spec.id] = spec
    return spec


def get_filter_registry() -> dict[str, FilterSpec]:
    """Get the complete filter registry."""
    return _FILTERS.copy()


def get_filter(filter_id: str) -> FilterSpec | None:
    """Get a specific filter by ID."""
    return _FILTERS.get(filter_id)


def get_filters_by_category(category: str) -> list[FilterSpec]:
    """Get all filters in a category."""
    return [f for f in _FILTERS.values() if f.category == category]


def get_categories() -> list[str]:
    """Get list of all filter categories."""
    return sorted(set(f.category for f in _FILTERS.values()))


# =============================================================================
# BLUR FILTERS
# =============================================================================

_register(FilterSpec(
    id="blur",
    name="Gaussian Blur",
    command="blur",
    category="Blur",
    description="Standard Gaussian blur",
    params=[
        FilterParam("sigma", ParamType.FLOAT, default=2.0, min_value=0.1, max_value=50.0,
                    description="Blur radius"),
    ],
))

_register(FilterSpec(
    id="bilateral",
    name="Bilateral Blur",
    command="bilateral",
    category="Blur",
    description="Edge-preserving blur",
    params=[
        FilterParam("spatial_sigma", ParamType.FLOAT, default=10.0, min_value=1.0, max_value=100.0,
                    description="Spatial sigma"),
        FilterParam("value_sigma", ParamType.FLOAT, default=7.0, min_value=1.0, max_value=100.0,
                    description="Value sigma"),
    ],
))

_register(FilterSpec(
    id="motion_blur",
    name="Motion Blur",
    command="blur_linear",
    category="Blur",
    description="Directional motion blur",
    params=[
        FilterParam("length", ParamType.FLOAT, default=10.0, min_value=1.0, max_value=100.0,
                    description="Blur length"),
        FilterParam("angle", ParamType.FLOAT, default=45.0, min_value=0.0, max_value=360.0,
                    description="Blur angle in degrees"),
    ],
))

# =============================================================================
# SHARPEN FILTERS
# =============================================================================

_register(FilterSpec(
    id="sharpen",
    name="Sharpen",
    command="sharpen",
    category="Sharpen",
    description="Standard sharpening",
    params=[
        FilterParam("amount", ParamType.FLOAT, default=100.0, min_value=0.0, max_value=500.0,
                    description="Sharpening amount"),
    ],
))

_register(FilterSpec(
    id="unsharp",
    name="Unsharp Mask",
    command="unsharp",
    category="Sharpen",
    description="Unsharp mask sharpening",
    params=[
        FilterParam("sigma", ParamType.FLOAT, default=1.0, min_value=0.1, max_value=10.0,
                    description="Blur sigma"),
        FilterParam("amount", ParamType.FLOAT, default=1.0, min_value=0.0, max_value=5.0,
                    description="Sharpening amount"),
    ],
))

# =============================================================================
# DENOISE FILTERS
# =============================================================================

_register(FilterSpec(
    id="denoise",
    name="Denoise",
    command="denoise",
    category="Denoise",
    description="Noise reduction",
    params=[
        FilterParam("sigma", ParamType.FLOAT, default=10.0, min_value=1.0, max_value=100.0,
                    description="Noise sigma"),
    ],
))

# =============================================================================
# COLOR ADJUSTMENTS
# =============================================================================

_register(FilterSpec(
    id="adjust_colors",
    name="Adjust Colors",
    command="adjust_colors",
    category="Color",
    description="Brightness, contrast, gamma, saturation",
    params=[
        FilterParam("brightness", ParamType.FLOAT, default=0.0, min_value=-100.0, max_value=100.0,
                    description="Brightness adjustment"),
        FilterParam("contrast", ParamType.FLOAT, default=0.0, min_value=-100.0, max_value=100.0,
                    description="Contrast adjustment"),
        FilterParam("gamma", ParamType.FLOAT, default=0.0, min_value=-100.0, max_value=100.0,
                    description="Gamma adjustment"),
        FilterParam("saturation", ParamType.FLOAT, default=0.0, min_value=-100.0, max_value=100.0,
                    description="Saturation adjustment"),
    ],
))

_register(FilterSpec(
    id="vibrance",
    name="Vibrance",
    command="fx_vibrance",
    category="Color",
    description="Vibrance adjustment (smart saturation)",
    params=[
        FilterParam("strength", ParamType.FLOAT, default=50.0, min_value=-100.0, max_value=100.0,
                    description="Vibrance strength"),
    ],
))

_register(FilterSpec(
    id="invert",
    name="Invert Colors",
    command="negate",
    category="Color",
    description="Invert all colors",
    params=[],
))

_register(FilterSpec(
    id="grayscale",
    name="Grayscale",
    command="luminance",
    category="Color",
    description="Convert to grayscale",
    params=[],
))

# =============================================================================
# ARTISTIC FILTERS
# =============================================================================

_register(FilterSpec(
    id="cartoon",
    name="Cartoon",
    command="cartoon",
    category="Artistic",
    description="Cartoon effect with edge detection",
    params=[
        FilterParam("smoothness", ParamType.FLOAT, default=3.0, min_value=0.0, max_value=10.0,
                    description="Edge smoothness"),
        FilterParam("sharpness", ParamType.FLOAT, default=10.0, min_value=0.0, max_value=100.0,
                    description="Edge sharpness"),
    ],
))

_register(FilterSpec(
    id="pencil",
    name="Pencil Sketch",
    command="pencilbw",
    category="Artistic",
    description="Black and white pencil sketch",
    params=[
        FilterParam("amplitude", ParamType.FLOAT, default=0.5, min_value=0.0, max_value=5.0,
                    description="Sketch amplitude"),
    ],
))

_register(FilterSpec(
    id="oil_painting",
    name="Oil Painting",
    command="fx_painting",
    category="Artistic",
    description="Oil painting effect",
    params=[
        FilterParam("size", ParamType.FLOAT, default=5.0, min_value=1.0, max_value=20.0,
                    description="Brush size"),
    ],
))

# =============================================================================
# DISTORT FILTERS
# =============================================================================

_register(FilterSpec(
    id="wave",
    name="Wave",
    command="wave",
    category="Distort",
    description="Wave distortion",
    params=[
        FilterParam("amplitude", ParamType.FLOAT, default=10.0, min_value=0.0, max_value=100.0,
                    description="Wave amplitude"),
        FilterParam("frequency", ParamType.FLOAT, default=0.5, min_value=0.01, max_value=2.0,
                    description="Wave frequency"),
    ],
))

_register(FilterSpec(
    id="swirl",
    name="Swirl",
    command="swirl",
    category="Distort",
    description="Swirl distortion",
    params=[
        FilterParam("angle", ParamType.FLOAT, default=45.0, min_value=-360.0, max_value=360.0,
                    description="Swirl angle"),
    ],
))

_register(FilterSpec(
    id="fisheye",
    name="Fish Eye",
    command="fisheye",
    category="Distort",
    description="Fisheye lens effect",
    params=[
        FilterParam("amount", ParamType.FLOAT, default=50.0, min_value=0.0, max_value=100.0,
                    description="Effect amount"),
    ],
))
