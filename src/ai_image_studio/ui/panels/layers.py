"""
Layers Panel - Layer stack management UI.

Provides a simple list of layers with visibility toggles and thumbnails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QLabel,
    QPushButton,
    QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QMouseEvent

if TYPE_CHECKING:
    from ai_image_studio.core.layers import Layer
    from ai_image_studio.core.data_types import ImageData


def _thumbnail_from_image_data(image_data: ImageData, size: int = 48) -> QPixmap:
    pil_img = image_data.to_pil().convert("RGBA")
    pil_img.thumbnail((size, size))
    data = pil_img.tobytes("raw", "RGBA")
    qimg = QImage(data, pil_img.width, pil_img.height, pil_img.width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class _LayerRow(QFrame):
    clicked = Signal(int)
    visibility_toggled = Signal(int, bool)

    def __init__(self, layer: Layer, parent: QWidget | None = None):
        super().__init__(parent)
        self._index = layer.index
        self._selected = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("background-color: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self._thumb = QLabel()
        self._thumb.setFixedSize(52, 52)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet("background-color: #313244; border-radius: 4px;")
        layout.addWidget(self._thumb)

        self._name = QLabel()
        self._name.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(self._name, 1)

        self._eye = QPushButton("👁")
        self._eye.setCheckable(True)
        self._eye.setFixedWidth(32)
        self._eye.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                border: none;
                border-radius: 4px;
                padding: 4px;
                color: #cdd6f4;
            }
            QPushButton:checked {
                background-color: #313244;
            }
            QPushButton:!checked {
                background-color: #1e1e2e;
                color: #6c7086;
            }
        """)
        self._eye.toggled.connect(self._on_eye_toggled)
        layout.addWidget(self._eye)

        self.update_from_layer(layer)

    @property
    def index(self) -> int:
        return self._index

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if self._selected:
            self.setStyleSheet("background-color: #313244; border-radius: 6px;")
        else:
            self.setStyleSheet("background-color: transparent;")

    def update_from_layer(self, layer: Layer) -> None:
        label = f"{layer.index}  {layer.display_name}"
        self._name.setText(label)
        self._eye.setChecked(bool(layer.visible))
        if layer.image_data is not None:
            self._thumb.setPixmap(_thumbnail_from_image_data(layer.image_data))
        else:
            self._thumb.setPixmap(QPixmap())
            self._thumb.setText(str(layer.index))
            self._thumb.setStyleSheet("background-color: #313244; border-radius: 4px; color: #6c7086;")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)
        super().mousePressEvent(event)

    def _on_eye_toggled(self, checked: bool) -> None:
        self.visibility_toggled.emit(self._index, bool(checked))


class LayersPanel(QWidget):
    """Dock panel for managing the layer stack."""

    layer_selected = Signal(int)
    layer_visibility_changed = Signal(int, bool)
    add_layer_requested = Signal()
    delete_layer_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._rows: dict[int, _LayerRow] = {}
        self._selected_index: int | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setStyleSheet("background-color: #1e1e2e; border-bottom: 1px solid #313244;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        title = QLabel("Layers")
        title.setStyleSheet("color: #a6adc8;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        add_btn = QPushButton("+")
        add_btn.setFixedWidth(28)
        add_btn.setToolTip("Add Layer")
        add_btn.clicked.connect(self.add_layer_requested.emit)
        header_layout.addWidget(add_btn)

        self._del_btn = QPushButton("🗑")
        self._del_btn.setFixedWidth(32)
        self._del_btn.setToolTip("Delete Selected Layer")
        self._del_btn.setEnabled(False)
        self._del_btn.clicked.connect(self._on_delete_clicked)
        header_layout.addWidget(self._del_btn)

        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background-color: #11111b; border: none; }")

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(8, 8, 8, 8)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch()

        scroll.setWidget(self._list_widget)
        layout.addWidget(scroll)

        self._empty = QLabel("Preview nodes will create layers")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet("color: #6c7086; padding: 12px;")
        self._list_layout.insertWidget(0, self._empty)

    def set_layers(self, layers: list[Layer], selected_index: int | None = None) -> None:
        self._selected_index = selected_index
        self._del_btn.setEnabled(selected_index is not None)

        for row in self._rows.values():
            row.deleteLater()
        self._rows.clear()

        # Remove all rows (keep stretch + empty label placeholder)
        while self._list_layout.count() > 2:
            item = self._list_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        if not layers:
            self._empty.show()
            return

        self._empty.hide()
        for layer in sorted(layers, key=lambda l: l.index, reverse=True):
            row = _LayerRow(layer)
            row.clicked.connect(self._on_row_clicked)
            row.visibility_toggled.connect(self.layer_visibility_changed.emit)
            row.set_selected(selected_index == layer.index)
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)
            self._rows[layer.index] = row

    def update_layer(self, layer: Layer) -> None:
        row = self._rows.get(layer.index)
        if row is None:
            return
        row.update_from_layer(layer)

    def _on_row_clicked(self, index: int) -> None:
        self._selected_index = index
        self._del_btn.setEnabled(True)
        for idx, row in self._rows.items():
            row.set_selected(idx == index)
        self.layer_selected.emit(index)

    def _on_delete_clicked(self) -> None:
        if self._selected_index is None:
            return
        self.delete_layer_requested.emit(self._selected_index)
