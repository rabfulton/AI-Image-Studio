"""
Gallery Panel - Generated image thumbnails and history.

This panel shows thumbnails of all generated images in the current
session, with options to save, delete, and load into the Output Studio.

Reference: wireframes.md - Gallery panel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QLabel,
    QPushButton,
    QFrame,
    QGridLayout,
    QMenu,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QAction

if TYPE_CHECKING:
    from ai_image_studio.core.data_types import ImageData
    from ai_image_studio.core.workflow_metadata import WorkflowMetadata


@dataclass
class GalleryItem:
    """A single item in the gallery."""
    id: str = field(default_factory=lambda: str(uuid4()))
    image_data: object = None  # ImageData
    qimage: QImage | None = None
    thumbnail: QPixmap | None = None
    prompt: str = ""
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    saved_path: Path | None = None
    workflow_path: Path | None = None


class ThumbnailWidget(QFrame):
    """Individual thumbnail widget with click and context menu."""
    
    clicked = Signal(str)  # item_id
    double_clicked = Signal(str)  # item_id
    context_menu_requested = Signal(str, object)  # item_id, QPoint
    
    def __init__(self, item: GalleryItem, size: int = 128, parent: QWidget | None = None):
        super().__init__(parent)
        
        self.item = item
        self._size = size
        self._selected = False
        
        self.setFixedSize(size + 8, size + 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        
        self._setup_ui()
        self._apply_style()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        
        # Thumbnail image
        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(self._size, self._size)
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_label.setStyleSheet("background-color: #313244; border-radius: 4px;")
        
        if self.item.thumbnail:
            self._thumb_label.setPixmap(self.item.thumbnail)
        else:
            self._thumb_label.setText("...")
        
        layout.addWidget(self._thumb_label)
        
        # Info label
        time_str = self.item.timestamp.strftime("%H:%M")
        self._info_label = QLabel(time_str)
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #6c7086; font-size: 10px;")
        layout.addWidget(self._info_label)
    
    def _apply_style(self) -> None:
        if self._selected:
            self.setStyleSheet("""
                ThumbnailWidget {
                    background-color: #4a9eff;
                    border-radius: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                ThumbnailWidget {
                    background-color: transparent;
                    border-radius: 6px;
                }
                ThumbnailWidget:hover {
                    background-color: #313244;
                }
            """)
    
    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_style()
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.item.id)
    
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.item.id)
    
    def _on_context_menu(self, pos) -> None:
        self.context_menu_requested.emit(self.item.id, self.mapToGlobal(pos))


class GalleryPanel(QWidget):
    """
    Panel showing generated image thumbnails.
    
    Features:
    - Grid of thumbnails
    - Click to select, double-click to load
    - Right-click context menu (save, delete, copy)
    - Auto-save option
    """
    
    image_selected = Signal(str)  # item_id
    image_load_requested = Signal(str)  # item_id
    workflow_load_requested = Signal(str)  # item_id - emitted when Load button clicked
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        self._items: dict[str, GalleryItem] = {}
        self._widgets: dict[str, ThumbnailWidget] = {}
        self._selected_id: str | None = None
        self._thumbnail_size = 100
        
        # Gallery storage path
        self._gallery_path = Path.home() / ".local" / "share" / "ai_image_studio" / "gallery"
        self._gallery_path.mkdir(parents=True, exist_ok=True)
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with controls
        header = QFrame()
        header.setStyleSheet("background-color: #1e1e2e; border-bottom: 1px solid #313244;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        
        self._count_label = QLabel("0 images")
        self._count_label.setStyleSheet("color: #a6adc8;")
        header_layout.addWidget(self._count_label)
        
        header_layout.addStretch()
        
        # Load workflow button
        self._load_btn = QPushButton("Load")
        self._load_btn.setToolTip("Load image and workflow into workspace")
        self._load_btn.setFixedWidth(48)
        self._load_btn.clicked.connect(self._on_load_selected)
        self._load_btn.setEnabled(False)  # Disabled until selection
        header_layout.addWidget(self._load_btn)
        
        open_folder_btn = QPushButton("📂")
        open_folder_btn.setToolTip("Open gallery folder")
        open_folder_btn.setFixedWidth(32)
        open_folder_btn.clicked.connect(self._on_open_folder)
        header_layout.addWidget(open_folder_btn)
        
        layout.addWidget(header)
        
        # Scroll area for thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: #11111b;
                border: none;
            }
        """)
        
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(8, 8, 8, 8)
        self._grid_layout.setSpacing(4)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        scroll.setWidget(self._grid_widget)
        layout.addWidget(scroll)
        
        # Empty state
        self._empty_label = QLabel("Generated images will appear here")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #6c7086;")
        self._grid_layout.addWidget(self._empty_label, 0, 0)
    
    def add_image(
        self,
        image_data,
        prompt: str = "",
        model: str = "",
        auto_save: bool = True,
        workflow_metadata: "WorkflowMetadata | None" = None,
    ) -> str:
        """
        Add an image to the gallery.
        
        Args:
            image_data: ImageData object
            prompt: Generation prompt
            model: Model used
            auto_save: Whether to save to disk
            workflow_metadata: Optional workflow state to save alongside image
        
        Returns:
            Item ID
        """
        # Hide empty label
        self._empty_label.hide()
        
        # Create QImage from ImageData
        pil_img = image_data.to_pil()
        
        if pil_img.mode == "RGBA":
            data = pil_img.tobytes("raw", "RGBA")
            qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGBA8888)
        else:
            pil_img = pil_img.convert("RGB")
            data = pil_img.tobytes("raw", "RGB")
            qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGB888)
        
        qimg = qimg.copy()  # Copy to own the data
        
        # Create thumbnail
        thumb = QPixmap.fromImage(qimg).scaled(
            self._thumbnail_size, self._thumbnail_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        
        # Create item
        item = GalleryItem(
            image_data=image_data,
            qimage=qimg,
            thumbnail=thumb,
            prompt=prompt,
            model=model,
        )
        
        # Auto-save
        if auto_save:
            item.saved_path = self._save_image(item)
            
            # Save workflow metadata if provided
            if workflow_metadata and item.saved_path:
                from ai_image_studio.core.workflow_metadata import save_workflow_metadata
                try:
                    item.workflow_path = save_workflow_metadata(item.saved_path, workflow_metadata)
                except Exception as e:
                    print(f"Failed to save workflow metadata: {e}")
        
        self._items[item.id] = item
        
        # Create widget
        widget = ThumbnailWidget(item, self._thumbnail_size)
        widget.clicked.connect(self._on_thumbnail_clicked)
        widget.double_clicked.connect(self._on_thumbnail_double_clicked)
        widget.context_menu_requested.connect(self._on_context_menu)
        
        self._widgets[item.id] = widget
        
        # Add to grid
        self._relayout_grid()
        
        # Update count
        self._count_label.setText(f"{len(self._items)} images")
        
        return item.id
    
    def _relayout_grid(self) -> None:
        """Relayout the grid of thumbnails."""
        # Clear grid
        for i in reversed(range(self._grid_layout.count())):
            item = self._grid_layout.itemAt(i)
            if item.widget() and item.widget() != self._empty_label:
                self._grid_layout.removeItem(item)
        
        # Calculate columns based on width
        cols = max(1, (self.width() - 16) // (self._thumbnail_size + 12))
        
        # Add widgets in reverse order (newest first)
        items = sorted(self._items.values(), key=lambda x: x.timestamp, reverse=True)
        for i, item in enumerate(items):
            widget = self._widgets.get(item.id)
            if widget:
                row = i // cols
                col = i % cols
                self._grid_layout.addWidget(widget, row, col)
    
    def _save_image(self, item: GalleryItem) -> Path:
        """Save image to gallery folder."""
        timestamp = item.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{item.id[:8]}.png"
        path = self._gallery_path / filename
        
        if item.qimage:
            item.qimage.save(str(path))
        
        return path
    
    def _on_thumbnail_clicked(self, item_id: str) -> None:
        """Handle thumbnail click."""
        # Deselect previous
        if self._selected_id and self._selected_id in self._widgets:
            self._widgets[self._selected_id].set_selected(False)
        
        # Select new
        self._selected_id = item_id
        if item_id in self._widgets:
            self._widgets[item_id].set_selected(True)
        
        # Enable Load button when selection exists
        self._load_btn.setEnabled(True)
        
        self.image_selected.emit(item_id)
    
    def _on_thumbnail_double_clicked(self, item_id: str) -> None:
        """Handle thumbnail double-click - load into Output Studio."""
        self.image_load_requested.emit(item_id)
    
    def _on_context_menu(self, item_id: str, pos) -> None:
        """Show context menu for thumbnail."""
        item = self._items.get(item_id)
        if not item:
            return
        
        menu = QMenu(self)
        
        load_action = menu.addAction("Load in Output Studio")
        load_action.triggered.connect(lambda: self.image_load_requested.emit(item_id))
        
        menu.addSeparator()
        
        save_action = menu.addAction("Save As...")
        save_action.triggered.connect(lambda: self._save_as(item_id))
        
        copy_action = menu.addAction("Copy to Clipboard")
        copy_action.triggered.connect(lambda: self._copy_to_clipboard(item_id))
        
        menu.addSeparator()
        
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_item(item_id))
        
        menu.exec(pos)
    
    def _save_as(self, item_id: str) -> None:
        """Save image to user-specified location."""
        item = self._items.get(item_id)
        if not item or not item.qimage:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image",
            str(Path.home() / "generated_image.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp)"
        )
        
        if path:
            item.qimage.save(path)
            item.saved_path = Path(path)
    
    def _copy_to_clipboard(self, item_id: str) -> None:
        """Copy image to clipboard."""
        from PySide6.QtWidgets import QApplication
        
        item = self._items.get(item_id)
        if item and item.qimage:
            QApplication.clipboard().setImage(item.qimage)
    
    def _delete_item(self, item_id: str) -> None:
        """Delete an item from the gallery."""
        if item_id not in self._items:
            return
        
        # Get item for file deletion
        item = self._items.get(item_id)
        
        # Delete the saved file from disk
        if item and item.saved_path and item.saved_path.exists():
            try:
                item.saved_path.unlink()
            except Exception as e:
                print(f"Failed to delete file {item.saved_path}: {e}")
        
        # Delete associated workflow file (do NOT clear workflow from interface)
        if item and item.saved_path:
            from ai_image_studio.core.workflow_metadata import delete_workflow_metadata
            delete_workflow_metadata(item.saved_path)
        
        # Remove widget
        widget = self._widgets.pop(item_id, None)
        if widget:
            widget.deleteLater()
        
        # Remove item
        self._items.pop(item_id, None)
        
        # Relayout
        self._relayout_grid()
        
        # Update count
        self._count_label.setText(f"{len(self._items)} images")
        
        if not self._items:
            self._empty_label.show()
    
    def _on_open_folder(self) -> None:
        """Open gallery folder in system file manager."""
        import subprocess
        import platform
        
        path = str(self._gallery_path)
        system = platform.system()
        
        try:
            if system == "Linux":
                subprocess.Popen(["xdg-open", path])
            elif system == "Darwin":
                subprocess.Popen(["open", path])
            elif system == "Windows":
                subprocess.Popen(["explorer", path])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")
    
    def _on_load_selected(self) -> None:
        """Handle Load button click - load selected image's workflow."""
        if self._selected_id:
            self.workflow_load_requested.emit(self._selected_id)
    
    def get_item(self, item_id: str) -> GalleryItem | None:
        """Get a gallery item by ID."""
        return self._items.get(item_id)
    
    def resizeEvent(self, event) -> None:
        """Handle resize - relayout grid."""
        super().resizeEvent(event)
        self._relayout_grid()
    
    def load_saved_images(self) -> int:
        """
        Load previously saved images from the gallery folder.
        
        Returns:
            Number of images loaded
        """
        if not self._gallery_path.exists():
            return 0
        
        # Find all PNG files, sorted by modification time (newest first)
        png_files = sorted(
            self._gallery_path.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        count = 0
        for path in png_files:
            try:
                qimg = QImage(str(path))
                if qimg.isNull():
                    continue
                
                # Parse timestamp from filename if possible
                filename = path.stem
                parts = filename.split("_")
                if len(parts) >= 2:
                    try:
                        ts = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
                    except ValueError:
                        ts = datetime.fromtimestamp(path.stat().st_mtime)
                else:
                    ts = datetime.fromtimestamp(path.stat().st_mtime)
                
                # Create thumbnail
                thumb = QPixmap.fromImage(qimg).scaled(
                    self._thumbnail_size, self._thumbnail_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                
                # Create item (no ImageData, just QImage)
                item = GalleryItem(
                    qimage=qimg,
                    thumbnail=thumb,
                    timestamp=ts,
                    saved_path=path,
                )
                
                # Check for associated workflow file
                from ai_image_studio.core.workflow_metadata import get_workflow_path
                workflow_path = get_workflow_path(path)
                if workflow_path.exists():
                    item.workflow_path = workflow_path
                
                self._items[item.id] = item
                
                # Create widget
                widget = ThumbnailWidget(item, self._thumbnail_size)
                widget.clicked.connect(self._on_thumbnail_clicked)
                widget.double_clicked.connect(self._on_thumbnail_double_clicked)
                widget.context_menu_requested.connect(self._on_context_menu)
                
                self._widgets[item.id] = widget
                count += 1
                
            except Exception as e:
                print(f"Failed to load {path}: {e}")
        
        if count > 0:
            self._empty_label.hide()
            self._relayout_grid()
            self._count_label.setText(f"{len(self._items)} images")
        
        return count
