"""
Layer Stack - Core data structures for multi-layer output.

This module provides the Layer/LayerStack model described in layers.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import numpy as np

from ai_image_studio.core.data_types import ImageData


@dataclass(slots=True)
class Layer:
    """A single image layer in the output stack."""

    id: UUID
    index: int
    name: str
    visible: bool = True
    opacity: float = 1.0
    image_data: ImageData | None = None

    @classmethod
    def create(
        cls,
        index: int,
        *,
        name: str | None = None,
        visible: bool = True,
        opacity: float = 1.0,
        image_data: ImageData | None = None,
    ) -> Layer:
        return cls(
            id=uuid4(),
            index=int(index),
            name=name if name is not None else f"Layer {int(index)}",
            visible=bool(visible),
            opacity=float(opacity),
            image_data=image_data,
        )

    @property
    def display_name(self) -> str:
        if self.index == 0:
            return f"{self.name} (Base)"
        return self.name

    @property
    def has_image(self) -> bool:
        return self.image_data is not None


class LayerStack:
    """An indexed stack of layers (0..N) with unique indices."""

    def __init__(self) -> None:
        self._layers: dict[int, Layer] = {0: Layer.create(0)}
        self._selected_index: int | None = None

    def next_available_index(self) -> int:
        idx = 0
        while idx in self._layers:
            idx += 1
        return idx

    @property
    def layers(self) -> list[Layer]:
        return [self._layers[i] for i in sorted(self._layers.keys())]

    @property
    def visible_layers(self) -> list[Layer]:
        return [l for l in self.layers if l.visible]

    @property
    def selected_index(self) -> int | None:
        return self._selected_index

    @selected_index.setter
    def selected_index(self, index: int | None) -> None:
        if index is None:
            self._selected_index = None
            return
        if int(index) not in self._layers:
            raise ValueError(f"Layer index {index} does not exist")
        self._selected_index = int(index)

    @property
    def selected_layer(self) -> Layer | None:
        if self._selected_index is None:
            return None
        return self._layers.get(self._selected_index)

    def __len__(self) -> int:
        return len(self._layers)

    def __contains__(self, index: int) -> bool:
        return int(index) in self._layers

    def __iter__(self):
        yield from self.layers

    def add_layer(self, *, index: int | None = None, name: str | None = None) -> Layer:
        if index is None:
            index = self.next_available_index()
        index = int(index)
        if index in self._layers:
            raise ValueError(f"Layer {index} already exists")
        layer = Layer.create(index, name=name)
        self._layers[index] = layer
        return layer

    def remove_layer(self, index: int) -> Layer | None:
        index = int(index)
        if index not in self._layers:
            return None
        if index == 0 and len(self._layers) == 1:
            # Base layer always exists; clearing image is the "remove" behavior.
            self._layers[0].image_data = None
            return None
        removed = self._layers.pop(index)
        if self._selected_index == index:
            self._selected_index = None
        return removed

    def clear(self) -> None:
        self._layers = {0: Layer.create(0)}
        self._selected_index = None

    def get_layer(self, index: int) -> Layer | None:
        return self._layers.get(int(index))

    def get_by_id(self, layer_id: UUID) -> Layer | None:
        for layer in self._layers.values():
            if layer.id == layer_id:
                return layer
        return None

    def set_layer_image(self, index: int, image_data: ImageData | None) -> Layer:
        index = int(index)
        layer = self._layers.get(index)
        if layer is None:
            layer = self.add_layer(index=index)
        layer.image_data = image_data
        return layer

    def set_layer_visibility(self, index: int, visible: bool) -> Layer:
        index = int(index)
        layer = self._layers.get(index)
        if layer is None:
            layer = self.add_layer(index=index)
        layer.visible = bool(visible)
        return layer

    def set_layer_name(self, index: int, name: str | None) -> Layer:
        index = int(index)
        layer = self._layers.get(index)
        if layer is None:
            layer = self.add_layer(index=index)
        layer.name = name if name is not None else f"Layer {index}"
        return layer

    def get_composite(self) -> ImageData | None:
        """
        Composite all visible layers (bottom->top).

        Notes:
        - Returns None if there are no visible images.
        - Returns the same ImageData object when there's exactly one visible image.
        - Uses simple "normal" alpha compositing when alpha is present.
        """
        ordered = [l for l in self.layers if l.visible and l.image_data is not None]
        if not ordered:
            return None

        if len(ordered) == 1:
            return ordered[0].image_data

        base_img = ordered[0].image_data
        assert base_img is not None
        height, width = base_img.pixels.shape[0], base_img.pixels.shape[1]

        out = np.zeros((height, width, 4), dtype=np.float32)

        def to_rgba(img: ImageData) -> np.ndarray:
            if img.pixels.shape[0] != height or img.pixels.shape[1] != width:
                raise ValueError("Layer image size mismatch")
            if img.channels == 4:
                return img.pixels
            rgb = img.pixels[..., :3]
            alpha = np.ones((height, width, 1), dtype=np.float32)
            return np.concatenate([rgb, alpha], axis=-1)

        for layer in ordered:
            img = layer.image_data
            if img is None:
                continue
            try:
                src = to_rgba(img)
            except ValueError:
                raise

            src_rgb = src[..., :3]
            src_a = (src[..., 3:4] * float(layer.opacity)).clip(0.0, 1.0)

            dst_rgb = out[..., :3]
            dst_a = out[..., 3:4]

            out_rgb = src_rgb * src_a + dst_rgb * (1.0 - src_a)
            out_a = src_a + dst_a * (1.0 - src_a)

            out[..., :3] = out_rgb
            out[..., 3:4] = out_a

        return ImageData.from_numpy(out)
