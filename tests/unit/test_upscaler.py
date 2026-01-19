"""Unit tests for upscaler provider and upscale node."""

import pytest
from unittest.mock import MagicMock, patch


class TestUpscalerModelCards:
    """Test upscaler model cards are registered correctly."""
    
    def test_stability_upscale_models_registered(self):
        """Stability AI upscaler models should be in registry."""
        from ai_image_studio.providers import get_registry
        from ai_image_studio.providers.base import GenerationMode
        
        registry = get_registry()
        
        # Check conservative
        conservative = registry.get_model("stability-upscale-conservative")
        assert conservative is not None
        assert conservative.provider == "stability"
        assert GenerationMode.UPSCALE in conservative.modes
        
        # Check creative
        creative = registry.get_model("stability-upscale-creative")
        assert creative is not None
        assert GenerationMode.UPSCALE in creative.modes
        
        # Check fast
        fast = registry.get_model("stability-upscale-fast")
        assert fast is not None
        assert GenerationMode.UPSCALE in fast.modes
    
    def test_local_realesrgan_models_registered(self):
        """Local Real-ESRGAN models should be in registry."""
        from ai_image_studio.providers import get_registry
        from ai_image_studio.providers.base import GenerationMode
        
        registry = get_registry()
        
        # Check x4
        x4 = registry.get_model("local-realesrgan-x4")
        assert x4 is not None
        assert x4.provider == "upscaler"
        assert GenerationMode.UPSCALE in x4.modes
        
        # Check x2
        x2 = registry.get_model("local-realesrgan-x2")
        assert x2 is not None
        assert GenerationMode.UPSCALE in x2.modes


class TestUpscaleGenerationMode:
    """Test UPSCALE generation mode exists."""
    
    def test_upscale_mode_exists(self):
        """GenerationMode.UPSCALE should be defined."""
        from ai_image_studio.providers.base import GenerationMode
        
        assert hasattr(GenerationMode, "UPSCALE")
        assert GenerationMode.UPSCALE.value == "upscale"


class TestUpscalerProvider:
    """Test UpscalerProvider functionality."""
    
    def test_upscaler_provider_registered(self):
        """UpscalerProvider should be registered."""
        from ai_image_studio.providers import get_registry
        
        registry = get_registry()
        provider = registry.get_provider("upscaler")
        assert provider is not None
    
    def test_upscaler_configured_returns_boolean(self):
        """Local upscaler should return boolean for is_configured."""
        from ai_image_studio.providers import get_registry
        
        registry = get_registry()
        provider = registry.get_provider("upscaler")
        # is_configured depends on whether upscayl-ncnn binary is installed
        assert isinstance(provider.is_configured, bool)


class TestUpscaleNode:
    """Test upscale node registration."""
    
    def test_upscale_node_registered(self):
        """Upscale node should be registered in NodeRegistry."""
        from ai_image_studio.core.node_types import NodeRegistry, NodeCategory
        from ai_image_studio.nodes import register_all_nodes
        
        # Ensure nodes are registered
        register_all_nodes()
        
        registry = NodeRegistry.instance()
        node_type = registry.get("enhancement.upscale")
        
        assert node_type is not None
        assert node_type.name == "Upscale"
        assert node_type.category == NodeCategory.ENHANCEMENT
    
    def test_upscale_node_has_image_input(self):
        """Upscale node should have an image input."""
        from ai_image_studio.core.node_types import NodeRegistry
        from ai_image_studio.core.data_types import DataType
        from ai_image_studio.nodes import register_all_nodes
        
        register_all_nodes()
        
        registry = NodeRegistry.instance()
        node_type = registry.get("enhancement.upscale")
        
        image_input = node_type.get_input("image")
        assert image_input is not None
        assert image_input.data_type == DataType.IMAGE
        assert image_input.required is True
    
    def test_upscale_node_has_image_output(self):
        """Upscale node should have an image output."""
        from ai_image_studio.core.node_types import NodeRegistry
        from ai_image_studio.core.data_types import DataType
        from ai_image_studio.nodes import register_all_nodes
        
        register_all_nodes()
        
        registry = NodeRegistry.instance()
        node_type = registry.get("enhancement.upscale")
        
        image_output = node_type.get_output("image")
        assert image_output is not None
        assert image_output.data_type == DataType.IMAGE
