"""
Node Library Panel - Tree view of available nodes.

This panel displays all registered node types organized by category,
with search functionality and drag-to-add support.

Reference: features.md#12-node-library
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QPushButton,
    QLabel,
)
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QDrag, QIcon, QColor

if TYPE_CHECKING:
    from ai_image_studio.core.node_types import NodeType, NodeCategory


# Category display info
CATEGORY_INFO = {
    "input": ("Input", "#4a9eff", "Prompt, Image, Mask inputs"),
    "output": ("Output", "#22c55e", "Preview, Save nodes"),
    "generation": ("Generation", "#a855f7", "AI image generation"),
    "enhancement": ("Enhancement", "#f59e0b", "Upscale, Face restore"),
    "filter": ("Filter", "#14b8a6", "G'MIC and image filters"),
    "utility": ("Utility", "#6b7280", "Blend, Resize, Crop"),
    "conditioning": ("Conditioning", "#ec4899", "CLIP, LoRA"),
    "mask": ("Mask", "#8b5cf6", "Mask operations"),
}


class NodeLibraryPanel(QWidget):
    """
    Panel showing available node types in a tree view.
    
    Signals:
        node_requested: Emitted when user wants to add a node (type_id)
    """
    
    node_requested = Signal(str)  # type_id
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        self._setup_ui()
        self._populate_default_nodes()
    
    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Search bar
        search_layout = QHBoxLayout()
        search_layout.setSpacing(4)
        
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search nodes...")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self._search_input)
        
        layout.addLayout(search_layout)
        
        # Tree view
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(16)
        self._tree.setAnimated(True)
        self._tree.setDragEnabled(True)
        self._tree.setDragDropMode(QTreeWidget.DragDropMode.DragOnly)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        # Style
        self._tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e2e;
                border: none;
                color: #cdd6f4;
            }
            QTreeWidget::item {
                padding: 4px 8px;
            }
            QTreeWidget::item:hover {
                background-color: #313244;
            }
            QTreeWidget::item:selected {
                background-color: #45475a;
            }
        """)
        
        layout.addWidget(self._tree)
    
    def _populate_default_nodes(self) -> None:
        """Populate the tree with default node types."""
        # This would normally come from NodeRegistry
        # For now, add manual entries
        
        nodes_by_category = {
            "input": [
                ("input.prompt", "Prompt", "Text prompt input"),
                ("input.image", "Image Input", "Load image from file"),
                ("input.mask", "Mask Input", "Load or create mask"),
                ("input.model", "Model Selector", "Select AI model"),
            ],
            "generation": [
                ("generation.text_to_image", "Text to Image", "Generate from prompt"),
                ("generation.image_to_image", "Image to Image", "Transform image"),
                ("generation.inpaint", "Inpaint", "Fill masked region"),
                ("generation.controlnet", "ControlNet", "Guided generation"),
            ],
            "enhancement": [
                ("enhancement.upscale", "Upscale", "AI upscaling"),
                ("enhancement.face_restore", "Face Restore", "Enhance faces"),
                ("enhancement.background_remove", "Remove Background", "Isolate subject"),
            ],
            "filter": [
                ("filter.gmic", "G'MIC Filter", "Apply G'MIC effect"),
                ("filter.blur", "Blur", "Gaussian blur"),
                ("filter.sharpen", "Sharpen", "Enhance details"),
                ("filter.color_adjust", "Color Adjust", "Brightness/contrast"),
            ],
            "utility": [
                ("utility.blend", "Blend", "Blend two images"),
                ("utility.resize", "Resize", "Scale image"),
                ("utility.crop", "Crop", "Crop to region"),
                ("utility.composite", "Composite", "Layer images"),
            ],
            "output": [
                ("output.preview", "Preview", "Display in Output Studio"),
                ("output.save", "Save Image", "Save to file"),
            ],
        }
        
        self._tree.clear()
        
        for category_id, nodes in nodes_by_category.items():
            info = CATEGORY_INFO.get(category_id, (category_id.title(), "#6b7280", ""))
            
            # Create category item
            category_item = QTreeWidgetItem([info[0]])
            category_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "category", "id": category_id})
            category_item.setForeground(0, QColor(info[1]))
            category_item.setExpanded(True)
            
            # Add node items
            for type_id, name, description in nodes:
                node_item = QTreeWidgetItem([name])
                node_item.setData(0, Qt.ItemDataRole.UserRole, {
                    "type": "node",
                    "id": type_id,
                    "description": description,
                })
                node_item.setToolTip(0, f"{description}\n\nID: {type_id}")
                category_item.addChild(node_item)
            
            self._tree.addTopLevelItem(category_item)
    
    def refresh_from_registry(self) -> None:
        """Refresh the tree from the NodeRegistry."""
        from ai_image_studio.core.node_types import NodeRegistry
        
        registry = NodeRegistry.instance()
        self._tree.clear()
        
        # Group by category
        by_category: dict[str, list] = {}
        for node_type in registry.get_all():
            cat = node_type.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(node_type)
        
        # Add to tree
        for category_id, nodes in sorted(by_category.items()):
            info = CATEGORY_INFO.get(category_id, (category_id.title(), "#6b7280", ""))
            
            category_item = QTreeWidgetItem([info[0]])
            category_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "category", "id": category_id})
            category_item.setForeground(0, QColor(info[1]))
            category_item.setExpanded(True)
            
            for node_type in sorted(nodes, key=lambda x: x.name):
                node_item = QTreeWidgetItem([node_type.name])
                node_item.setData(0, Qt.ItemDataRole.UserRole, {
                    "type": "node",
                    "id": node_type.id,
                    "description": node_type.description,
                })
                node_item.setToolTip(0, f"{node_type.description}\n\nID: {node_type.id}")
                category_item.addChild(node_item)
            
            self._tree.addTopLevelItem(category_item)
    
    def _on_search_changed(self, text: str) -> None:
        """Filter the tree based on search text."""
        search = text.lower().strip()
        
        for i in range(self._tree.topLevelItemCount()):
            category_item = self._tree.topLevelItem(i)
            category_visible = False
            
            for j in range(category_item.childCount()):
                node_item = category_item.child(j)
                data = node_item.data(0, Qt.ItemDataRole.UserRole)
                
                # Match name or description
                name = node_item.text(0).lower()
                desc = data.get("description", "").lower()
                matches = not search or search in name or search in desc
                
                node_item.setHidden(not matches)
                if matches:
                    category_visible = True
            
            category_item.setHidden(not category_visible)
            if category_visible and search:
                category_item.setExpanded(True)
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle double-click to add node."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data.get("type") == "node":
            self.node_requested.emit(data["id"])
    
    def startDrag(self, supportedActions) -> None:
        """Start drag operation for adding nodes."""
        item = self._tree.currentItem()
        if not item:
            return
        
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or data.get("type") != "node":
            return
        
        # Create drag data
        mime_data = QMimeData()
        mime_data.setText(data["id"])
        mime_data.setData("application/x-node-type", data["id"].encode())
        
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.CopyAction)
