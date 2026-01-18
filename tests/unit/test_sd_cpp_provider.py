"""
Unit tests for SDCppProvider image conversion behavior.

These tests do not require the stable-diffusion-cpp-python library; they
exercise the PIL -> ImageData conversion used by the provider.
"""

import numpy as np
from PIL import Image

from ai_image_studio.providers.base import ProviderConfig
from ai_image_studio.providers.sd_cpp import SDCppProvider


def test_sd_cpp_provider_drops_alpha_for_generated_images():
    """
    Some sd.cpp bindings can return RGBA with a zero/uninitialized alpha channel.
    The provider should convert generation outputs to RGB so they display in Qt.
    """
    provider = SDCppProvider(ProviderConfig(extra={"model_folders": ["/tmp"]}))

    # RGBA red pixel but fully transparent alpha (a common failure mode).
    rgba = Image.fromarray(np.array([[[255, 0, 0, 0]]], dtype=np.uint8), mode="RGBA")

    image_data = provider._convert_to_image_data(rgba)

    assert image_data.pixels.shape == (1, 1, 3)
    # Stored as float32 [0,1]
    assert image_data.pixels.dtype == np.float32
    assert np.allclose(image_data.pixels[0, 0], [1.0, 0.0, 0.0])

