"""
Dock panels UI components.

This package provides the dock panels for the application:
- Node Library: Tree view of available nodes
- Properties: Parameter editor for selected node
- Console: Log and progress display
- Gallery: Generated image thumbnails
- History: Undo/redo states
"""

from ai_image_studio.ui.panels.node_library import NodeLibraryPanel
from ai_image_studio.ui.panels.properties import PropertiesPanel
from ai_image_studio.ui.panels.console import ConsolePanel
from ai_image_studio.ui.panels.gallery import GalleryPanel

__all__ = [
    "NodeLibraryPanel",
    "PropertiesPanel",
    "ConsolePanel",
    "GalleryPanel",
]
