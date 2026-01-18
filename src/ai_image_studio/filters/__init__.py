"""
Filters module - G'MIC integration and filter system.

This package provides image filter implementations, with the primary
focus on G'MIC integration for hundreds of professional filters.
"""

from ai_image_studio.filters.gmic_runner import GmicRunner
from ai_image_studio.filters.filter_registry import (
    FilterSpec,
    FilterParam,
    get_filter_registry,
)

__all__ = [
    "GmicRunner",
    "FilterSpec", 
    "FilterParam",
    "get_filter_registry",
]
