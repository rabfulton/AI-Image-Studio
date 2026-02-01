"""
Tests for the layers module.
"""

import pytest
import numpy as np
from uuid import UUID

from ai_image_studio.core.layers import Layer, LayerStack
from ai_image_studio.core.data_types import ImageData


class TestLayer:
    """Tests for Layer dataclass."""
    
    def test_create_default(self):
        layer = Layer.create(0)
        assert layer.index == 0
        assert layer.name == "Layer 0"
        assert isinstance(layer.id, UUID)
        assert layer.visible is True
        assert layer.opacity == 1.0
        assert layer.image_data is None
    
    def test_create_with_name(self):
        layer = Layer.create(1, name="Background")
        assert layer.index == 1
        assert layer.name == "Background"
    
    def test_create_with_image(self):
        pixels = np.zeros((64, 64, 3), dtype=np.float32)
        image = ImageData.from_numpy(pixels)
        layer = Layer.create(0, image_data=image)
        assert layer.has_image is True
        assert layer.image_data is image
    
    def test_has_image_false(self):
        layer = Layer.create(0)
        assert layer.has_image is False
    
    def test_display_name_base(self):
        layer = Layer.create(0)
        assert layer.display_name == "Layer 0 (Base)"
    
    def test_display_name_non_base(self):
        layer = Layer.create(1, name="Overlay")
        assert layer.display_name == "Overlay"
    
    def test_auto_name_from_index(self):
        layer = Layer.create(5)
        assert layer.name == "Layer 5"


class TestLayerStack:
    """Tests for LayerStack class."""
    
    def test_init_has_base_layer(self):
        stack = LayerStack()
        assert len(stack) == 1
        assert 0 in stack
        assert stack.get_layer(0) is not None
    
    def test_add_layer_auto_index(self):
        stack = LayerStack()
        layer = stack.add_layer()
        assert layer.index == 1
        assert len(stack) == 2
    
    def test_add_layer_specific_index(self):
        stack = LayerStack()
        layer = stack.add_layer(index=5, name="Custom")
        assert layer.index == 5
        assert layer.name == "Custom"
    
    def test_add_layer_duplicate_index_raises(self):
        stack = LayerStack()
        with pytest.raises(ValueError, match="already exists"):
            stack.add_layer(index=0)
    
    def test_remove_layer(self):
        stack = LayerStack()
        stack.add_layer(index=1)
        removed = stack.remove_layer(1)
        assert removed is not None
        assert 1 not in stack
    
    def test_remove_last_layer_clears(self):
        stack = LayerStack()
        pixels = np.zeros((32, 32, 3), dtype=np.float32)
        stack.set_layer_image(0, ImageData.from_numpy(pixels))
        result = stack.remove_layer(0)
        # Should not remove, just clear
        assert result is None
        assert 0 in stack
        assert stack.get_layer(0).image_data is None
    
    def test_get_layer(self):
        stack = LayerStack()
        layer = stack.get_layer(0)
        assert layer is not None
        assert layer.index == 0
    
    def test_get_by_id(self):
        stack = LayerStack()
        base_layer = stack.get_layer(0)
        found = stack.get_by_id(base_layer.id)
        assert found is base_layer
    
    def test_set_layer_image_existing(self):
        stack = LayerStack()
        pixels = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        image = ImageData.from_numpy(pixels)
        layer = stack.set_layer_image(0, image)
        assert layer.image_data is image
    
    def test_set_layer_image_creates_layer(self):
        stack = LayerStack()
        pixels = np.ones((64, 64, 3), dtype=np.float32)
        image = ImageData.from_numpy(pixels)
        layer = stack.set_layer_image(2, image)
        assert 2 in stack
        assert layer.image_data is image
    
    def test_set_layer_visibility(self):
        stack = LayerStack()
        stack.set_layer_visibility(0, False)
        assert stack.get_layer(0).visible is False
    
    def test_next_available_index_filled(self):
        stack = LayerStack()
        stack.add_layer(index=1)
        stack.add_layer(index=2)
        assert stack.next_available_index() == 3
    
    def test_next_available_index_gap(self):
        stack = LayerStack()
        stack.add_layer(index=1)
        stack.add_layer(index=3)  # Skip 2
        assert stack.next_available_index() == 2
    
    def test_layers_sorted(self):
        stack = LayerStack()
        stack.add_layer(index=3)
        stack.add_layer(index=1)
        stack.add_layer(index=2)
        layers = stack.layers
        indices = [l.index for l in layers]
        assert indices == [0, 1, 2, 3]
    
    def test_visible_layers(self):
        stack = LayerStack()
        stack.add_layer(index=1)
        stack.add_layer(index=2)
        stack.set_layer_visibility(1, False)
        visible = stack.visible_layers
        indices = [l.index for l in visible]
        assert 1 not in indices
        assert 0 in indices
        assert 2 in indices
    
    def test_selected_index(self):
        stack = LayerStack()
        stack.add_layer(index=1)
        stack.selected_index = 1
        assert stack.selected_index == 1
        assert stack.selected_layer.index == 1
    
    def test_selected_index_invalid_raises(self):
        stack = LayerStack()
        with pytest.raises(ValueError, match="does not exist"):
            stack.selected_index = 999
    
    def test_clear(self):
        stack = LayerStack()
        stack.add_layer(index=1)
        stack.add_layer(index=2)
        stack.selected_index = 1
        stack.clear()
        assert len(stack) == 1
        assert 0 in stack
        assert stack.selected_index is None
    
    def test_contains(self):
        stack = LayerStack()
        assert 0 in stack
        assert 5 not in stack
    
    def test_iter(self):
        stack = LayerStack()
        stack.add_layer(index=1)
        layers = list(stack)
        assert len(layers) == 2


class TestLayerStackComposite:
    """Tests for layer compositing."""
    
    def test_composite_empty(self):
        stack = LayerStack()
        composite = stack.get_composite()
        assert composite is None
    
    def test_composite_single_layer(self):
        stack = LayerStack()
        pixels = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        image = ImageData.from_numpy(pixels)
        stack.set_layer_image(0, image)
        composite = stack.get_composite()
        assert composite is image  # Returns same object for single layer
    
    def test_composite_excludes_hidden(self):
        stack = LayerStack()
        pixels1 = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        pixels2 = np.ones((32, 32, 3), dtype=np.float32) * 1.0
        stack.set_layer_image(0, ImageData.from_numpy(pixels1))
        stack.set_layer_image(1, ImageData.from_numpy(pixels2))
        stack.set_layer_visibility(1, False)
        composite = stack.get_composite()
        # Should only have layer 0
        assert np.allclose(composite.pixels[:, :, 0], 0.5)
    
    def test_composite_order_correct(self):
        stack = LayerStack()
        # Layer 0: red
        red = np.zeros((32, 32, 4), dtype=np.float32)
        red[:, :, 0] = 1.0  # Red
        red[:, :, 3] = 1.0  # Opaque
        # Layer 1: semi-transparent green
        green = np.zeros((32, 32, 4), dtype=np.float32)
        green[:, :, 1] = 1.0  # Green
        green[:, :, 3] = 0.5  # 50% transparent
        
        stack.set_layer_image(0, ImageData.from_numpy(red))
        stack.set_layer_image(1, ImageData.from_numpy(green))
        
        composite = stack.get_composite()
        # Should blend green over red
        assert composite is not None
        # Center pixel should have some red and some green
        center = composite.pixels[16, 16]
        assert center[0] > 0  # Some red from layer 0
        assert center[1] > 0  # Some green from layer 1
