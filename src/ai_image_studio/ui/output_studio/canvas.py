"""
Output Studio - Image viewing and comparison canvas.

This module provides the Output Studio widget for displaying generated
images with zoom/pan, before/after comparison, and basic editing tools.

Reference: wireframes.md#3-output-studio-detail
Reference: features.md#2-output-studio
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QLabel,
    QSlider,
    QComboBox,
    QPushButton,
    QFrame,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer
from PySide6.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QImage,
    QPixmap,
    QMouseEvent,
    QWheelEvent,
    QPaintEvent,
    QResizeEvent,
)

if TYPE_CHECKING:
    from ai_image_studio.core.data_types import ImageData


class ViewMode(Enum):
    """Display modes for the Output Studio."""
    SINGLE = auto()       # Show single image
    COMPARISON = auto()   # Side-by-side with slider
    DIFF = auto()         # Difference visualization
    BLEND = auto()        # Blended view with opacity


@dataclass
class CanvasTransform:
    """Handles canvas pan and zoom transformations."""
    offset_x: float = 0.0
    offset_y: float = 0.0
    zoom: float = 1.0
    
    def screen_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.offset_x) / self.zoom, (y - self.offset_y) / self.zoom)
    
    def canvas_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.zoom + self.offset_x, y * self.zoom + self.offset_y)
    
    def zoom_at(self, x: float, y: float, factor: float) -> None:
        cx, cy = self.screen_to_canvas(x, y)
        self.zoom *= factor
        self.zoom = max(0.1, min(10.0, self.zoom))
        new_sx, new_sy = self.canvas_to_screen(cx, cy)
        self.offset_x += x - new_sx
        self.offset_y += y - new_sy


class ImageCanvas(QWidget):
    """
    Canvas widget for displaying an image with zoom/pan.
    
    Features:
    - Checkerboard background for transparency
    - Smooth zoom at cursor position
    - Pan with middle mouse or space+drag
    - Fit to view
    """
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        
        self._transform = CanvasTransform()
        self._image: QImage | None = None
        self._is_panning = False
        self._pan_start: QPointF | None = None
        self._pan_offset_start: tuple[float, float] = (0, 0)
        
        # Checkerboard pattern for transparency
        self._checkerboard = self._create_checkerboard(16)
        
        # Background
        self._bg_color = QColor("#16213e")
    
    def _create_checkerboard(self, size: int) -> QPixmap:
        """Create a checkerboard pattern for transparency display."""
        pixmap = QPixmap(size * 2, size * 2)
        pixmap.fill(QColor("#404040"))
        
        painter = QPainter(pixmap)
        painter.fillRect(0, 0, size, size, QColor("#505050"))
        painter.fillRect(size, size, size, size, QColor("#505050"))
        painter.end()
        
        return pixmap
    
    def set_image(self, image: QImage | None) -> None:
        """Set the image to display."""
        self._image = image
        self.fit_to_view()
        self.update()
    
    def set_image_from_data(self, image_data) -> None:
        """Set image from ImageData object."""
        if image_data is None:
            self.set_image(None)
            return
        
        # Convert to QImage
        pil_img = image_data.to_pil()
        
        # Convert PIL to QImage with explicit bytes_per_line to prevent stride issues
        if pil_img.mode == "RGBA":
            data = pil_img.tobytes("raw", "RGBA")
            bytes_per_line = pil_img.width * 4
            qimg = QImage(data, pil_img.width, pil_img.height, bytes_per_line, QImage.Format.Format_RGBA8888)
        else:
            pil_img = pil_img.convert("RGB")
            data = pil_img.tobytes("raw", "RGB")
            bytes_per_line = pil_img.width * 3
            qimg = QImage(data, pil_img.width, pil_img.height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Need to copy because the data goes out of scope
        self.set_image(qimg.copy())
    
    def fit_to_view(self) -> None:
        """Fit the image to the view."""
        if not self._image:
            return
        
        # Calculate zoom to fit
        margin = 20
        available_w = self.width() - margin * 2
        available_h = self.height() - margin * 2
        
        if available_w <= 0 or available_h <= 0:
            return
        
        zoom_w = available_w / self._image.width()
        zoom_h = available_h / self._image.height()
        self._transform.zoom = min(zoom_w, zoom_h, 1.0)
        
        # Center
        img_w = self._image.width() * self._transform.zoom
        img_h = self._image.height() * self._transform.zoom
        self._transform.offset_x = (self.width() - img_w) / 2
        self._transform.offset_y = (self.height() - img_h) / 2
        
        self.update()
    
    def zoom_in(self) -> None:
        """Zoom in centered on view."""
        cx, cy = self.width() / 2, self.height() / 2
        self._transform.zoom_at(cx, cy, 1.25)
        self.update()
    
    def zoom_out(self) -> None:
        """Zoom out centered on view."""
        cx, cy = self.width() / 2, self.height() / 2
        self._transform.zoom_at(cx, cy, 0.8)
        self.update()
    
    def actual_size(self) -> None:
        """Show image at 100% zoom."""
        if not self._image:
            return
        
        self._transform.zoom = 1.0
        img_w = self._image.width()
        img_h = self._image.height()
        self._transform.offset_x = (self.width() - img_w) / 2
        self._transform.offset_y = (self.height() - img_h) / 2
        self.update()
    
    def paintEvent(self, event: QPaintEvent) -> None:
        """Render the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Background
        painter.fillRect(self.rect(), self._bg_color)
        
        if not self._image:
            # Empty state
            painter.setPen(QColor("#6c7086"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image")
            painter.end()
            return
        
        # Calculate image rectangle
        x, y = self._transform.canvas_to_screen(0, 0)
        w = self._image.width() * self._transform.zoom
        h = self._image.height() * self._transform.zoom
        img_rect = QRectF(x, y, w, h)
        
        # Checkerboard for transparency
        painter.save()
        painter.setClipRect(img_rect)
        for ix in range(int(x), int(x + w + 32), 32):
            for iy in range(int(y), int(y + h + 32), 32):
                painter.drawPixmap(ix, iy, self._checkerboard)
        painter.restore()
        
        # Image
        painter.drawImage(img_rect, self._image)
        
        # Border
        painter.setPen(QPen(QColor("#313244"), 1))
        painter.drawRect(img_rect)
        
        painter.end()
    
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle resize."""
        super().resizeEvent(event)
        if self._image:
            self.fit_to_view()
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for panning."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._pan_start = event.position()
            self._pan_offset_start = (self._transform.offset_x, self._transform.offset_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for panning."""
        if self._is_panning and self._pan_start:
            dx = event.position().x() - self._pan_start.x()
            dy = event.position().y() - self._pan_start.y()
            self._transform.offset_x = self._pan_offset_start[0] + dx
            self._transform.offset_y = self._pan_offset_start[1] + dy
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle wheel for zooming."""
        pos = event.position()
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self._transform.zoom_at(pos.x(), pos.y(), factor)
        self.update()


class ComparisonCanvas(QWidget):
    """
    Canvas for before/after comparison with slider.
    
    Displays two images side by side with a draggable divider.
    """
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        
        self._transform = CanvasTransform()
        self._before_image: QImage | None = None
        self._after_image: QImage | None = None
        self._divider_pos: float = 0.5  # 0-1, position of divider
        self._is_dragging_divider = False
        
        self._checkerboard = self._create_checkerboard(16)
        self._bg_color = QColor("#16213e")
    
    def _create_checkerboard(self, size: int) -> QPixmap:
        pixmap = QPixmap(size * 2, size * 2)
        pixmap.fill(QColor("#404040"))
        painter = QPainter(pixmap)
        painter.fillRect(0, 0, size, size, QColor("#505050"))
        painter.fillRect(size, size, size, size, QColor("#505050"))
        painter.end()
        return pixmap
    
    def set_before_image(self, image: QImage | None) -> None:
        self._before_image = image
        self._fit_to_view()
        self.update()
    
    def set_after_image(self, image: QImage | None) -> None:
        self._after_image = image
        self._fit_to_view()
        self.update()
    
    def _fit_to_view(self) -> None:
        img = self._after_image or self._before_image
        if not img:
            return
        
        margin = 20
        available_w = self.width() - margin * 2
        available_h = self.height() - margin * 2
        
        if available_w <= 0 or available_h <= 0:
            return
        
        zoom_w = available_w / img.width()
        zoom_h = available_h / img.height()
        self._transform.zoom = min(zoom_w, zoom_h, 1.0)
        
        img_w = img.width() * self._transform.zoom
        img_h = img.height() * self._transform.zoom
        self._transform.offset_x = (self.width() - img_w) / 2
        self._transform.offset_y = (self.height() - img_h) / 2
    
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        painter.fillRect(self.rect(), self._bg_color)
        
        img = self._after_image or self._before_image
        if not img:
            painter.setPen(QColor("#6c7086"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image")
            painter.end()
            return
        
        # Image rect
        x, y = self._transform.canvas_to_screen(0, 0)
        w = img.width() * self._transform.zoom
        h = img.height() * self._transform.zoom
        img_rect = QRectF(x, y, w, h)
        
        # Divider position in screen coordinates
        divider_x = x + w * self._divider_pos
        
        # Draw "before" on left side
        if self._before_image:
            left_clip = QRectF(x, y, w * self._divider_pos, h)
            painter.save()
            painter.setClipRect(left_clip)
            
            # Checkerboard
            for ix in range(int(x), int(x + w + 32), 32):
                for iy in range(int(y), int(y + h + 32), 32):
                    painter.drawPixmap(ix, iy, self._checkerboard)
            
            painter.drawImage(img_rect, self._before_image)
            painter.restore()
        
        # Draw "after" on right side
        if self._after_image:
            right_clip = QRectF(divider_x, y, w * (1 - self._divider_pos), h)
            painter.save()
            painter.setClipRect(right_clip)
            
            # Checkerboard
            for ix in range(int(x), int(x + w + 32), 32):
                for iy in range(int(y), int(y + h + 32), 32):
                    painter.drawPixmap(ix, iy, self._checkerboard)
            
            painter.drawImage(img_rect, self._after_image)
            painter.restore()
        
        # Divider line
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawLine(int(divider_x), int(y), int(divider_x), int(y + h))
        
        # Divider handle
        handle_y = y + h / 2
        handle_rect = QRectF(divider_x - 15, handle_y - 15, 30, 30)
        painter.setBrush(QColor("#4a9eff"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(handle_rect)
        
        # Arrows
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawText(handle_rect, Qt.AlignmentFlag.AlignCenter, "◀▶")
        
        # Labels
        painter.setPen(QColor("#a6adc8"))
        if self._before_image:
            painter.drawText(int(x + 10), int(y + 20), "Before")
        if self._after_image:
            painter.drawText(int(x + w - 50), int(y + 20), "After")
        
        # Border
        painter.setPen(QPen(QColor("#313244"), 1))
        painter.drawRect(img_rect)
        
        painter.end()
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Check if clicking on divider
        img = self._after_image or self._before_image
        if img:
            x, y = self._transform.canvas_to_screen(0, 0)
            w = img.width() * self._transform.zoom
            divider_x = x + w * self._divider_pos
            
            if abs(event.position().x() - divider_x) < 20:
                self._is_dragging_divider = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_dragging_divider:
            img = self._after_image or self._before_image
            if img:
                x, y = self._transform.canvas_to_screen(0, 0)
                w = img.width() * self._transform.zoom
                self._divider_pos = max(0.0, min(1.0, (event.position().x() - x) / w))
                self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._is_dragging_divider = False
        self.setCursor(Qt.CursorShape.ArrowCursor)


class OutputStudio(QWidget):
    """
    Complete Output Studio widget with toolbar and canvas.
    
    Features:
    - Image display with zoom/pan
    - Before/after comparison mode
    - View mode switching
    - Zoom controls
    """
    
    # Signals
    image_changed = Signal()
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        self._view_mode = ViewMode.SINGLE
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        self._toolbar = QToolBar()
        self._toolbar.setStyleSheet("""
            QToolBar {
                background-color: #1e1e2e;
                border-bottom: 1px solid #313244;
                spacing: 4px;
                padding: 4px;
            }
            QPushButton {
                background-color: #313244;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: #cdd6f4;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
            QPushButton:checked {
                background-color: #4a9eff;
            }
            QComboBox {
                background-color: #313244;
                border: none;
                border-radius: 4px;
                padding: 6px;
                color: #cdd6f4;
            }
        """)
        
        # Zoom controls
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedWidth(30)
        zoom_out_btn.clicked.connect(self._on_zoom_out)
        self._toolbar.addWidget(zoom_out_btn)
        
        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(50)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom_label.setStyleSheet("color: #cdd6f4;")
        self._toolbar.addWidget(self._zoom_label)
        
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(30)
        zoom_in_btn.clicked.connect(self._on_zoom_in)
        self._toolbar.addWidget(zoom_in_btn)
        
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self._on_fit)
        self._toolbar.addWidget(fit_btn)
        
        self._toolbar.addSeparator()
        
        # View mode selector
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Single", ViewMode.SINGLE)
        self._mode_combo.addItem("Comparison", ViewMode.COMPARISON)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._toolbar.addWidget(self._mode_combo)
        
        self._toolbar.addSeparator()
        
        # Info label
        self._info_label = QLabel("No image")
        self._info_label.setStyleSheet("color: #6c7086; padding-left: 8px;")
        self._toolbar.addWidget(self._info_label)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), 
                            spacer.sizePolicy().verticalPolicy())
        self._toolbar.addWidget(spacer)
        
        layout.addWidget(self._toolbar)
        
        # Canvas stack
        self._single_canvas = ImageCanvas()
        self._comparison_canvas = ComparisonCanvas()
        self._comparison_canvas.hide()
        
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self._single_canvas)
        canvas_layout.addWidget(self._comparison_canvas)
        
        layout.addWidget(canvas_container)
        
        # Status bar - compact info strip at bottom
        self._status_bar = QFrame()
        self._status_bar.setFixedHeight(22)  # Fixed small height
        self._status_bar.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border-top: 1px solid #313244;
            }
        """)
        status_layout = QHBoxLayout(self._status_bar)
        status_layout.setContentsMargins(6, 2, 6, 2)
        status_layout.setSpacing(12)
        
        self._cursor_label = QLabel("X: - Y: -")
        self._cursor_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        status_layout.addWidget(self._cursor_label)
        
        self._color_label = QLabel("RGB: ---")
        self._color_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        status_layout.addWidget(self._color_label)
        
        status_layout.addStretch()
        
        layout.addWidget(self._status_bar)
    
    def set_image(self, image: QImage) -> None:
        """Set the main image to display (QImage)."""
        self._single_canvas.set_image(image)
        self._comparison_canvas.set_after_image(image)
        self._update_info(image)
    
    def set_image_from_data(self, image_data) -> None:
        """Set the main image from an ImageData object."""
        if image_data is None:
            self._single_canvas.set_image(None)
            self._info_label.setText("No image")
            return
        
        # Convert ImageData to QImage via PIL
        pil_img = image_data.to_pil()
        
        # Explicit bytes_per_line prevents stride/alignment issues
        if pil_img.mode == "RGBA":
            data = pil_img.tobytes("raw", "RGBA")
            bytes_per_line = pil_img.width * 4
            qimg = QImage(data, pil_img.width, pil_img.height, bytes_per_line, QImage.Format.Format_RGBA8888)
        else:
            pil_img = pil_img.convert("RGB")
            data = pil_img.tobytes("raw", "RGB")
            bytes_per_line = pil_img.width * 3
            qimg = QImage(data, pil_img.width, pil_img.height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Must copy - data goes out of scope
        self._single_canvas.set_image(qimg.copy())
        self._update_info(qimg)
    
    def set_before_image(self, image: QImage) -> None:
        """Set the 'before' image for comparison."""
        self._comparison_canvas.set_before_image(image)
    
    def set_image_from_file(self, path: str) -> bool:
        """Load and display an image from file."""
        image = QImage(path)
        if image.isNull():
            return False
        self.set_image(image)
        return True
    
    def _update_info(self, image: QImage | None) -> None:
        """Update the info label."""
        if image and not image.isNull():
            w, h = image.width(), image.height()
            self._info_label.setText(f"{w} × {h}")
        else:
            self._info_label.setText("No image")
    
    def _update_zoom_label(self) -> None:
        """Update the zoom percentage label."""
        canvas = self._single_canvas if self._view_mode == ViewMode.SINGLE else self._comparison_canvas
        if hasattr(canvas, '_transform'):
            pct = int(canvas._transform.zoom * 100)
            self._zoom_label.setText(f"{pct}%")
    
    def _on_zoom_in(self) -> None:
        self._single_canvas.zoom_in()
        self._update_zoom_label()
    
    def _on_zoom_out(self) -> None:
        self._single_canvas.zoom_out()
        self._update_zoom_label()
    
    def _on_fit(self) -> None:
        self._single_canvas.fit_to_view()
        self._update_zoom_label()
    
    def _on_mode_changed(self, index: int) -> None:
        mode = self._mode_combo.currentData()
        self._view_mode = mode
        
        if mode == ViewMode.SINGLE:
            self._single_canvas.show()
            self._comparison_canvas.hide()
        else:
            self._single_canvas.hide()
            self._comparison_canvas.show()
