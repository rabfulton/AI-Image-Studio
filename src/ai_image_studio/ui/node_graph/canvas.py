"""
Node Graph Canvas - OpenGL-based node graph editor.

This module provides the main canvas widget for displaying and
interacting with the node graph using OpenGL for rendering.

Reference: wireframes.md#2-node-graph-canvas-detail
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer
from PySide6.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainterPath,
    QMouseEvent,
    QWheelEvent,
    QKeyEvent,
    QPaintEvent,
)

if TYPE_CHECKING:
    from ai_image_studio.core.graph import NodeGraph, Node, NodeId, Connection


# Colors for different node categories
NODE_COLORS = {
    "input": QColor("#4a9eff"),       # Blue
    "output": QColor("#22c55e"),      # Green
    "generation": QColor("#a855f7"),  # Purple
    "enhancement": QColor("#f59e0b"), # Orange
    "filter": QColor("#14b8a6"),      # Teal
    "utility": QColor("#6b7280"),     # Gray
    "conditioning": QColor("#ec4899"), # Pink
    "mask": QColor("#8b5cf6"),        # Violet
    "default": QColor("#4a5568"),     # Dark gray
}

# Socket colors by data type
SOCKET_COLORS = {
    "IMAGE": QColor("#f59e0b"),       # Orange
    "MASK": QColor("#ffffff"),        # White
    "TEXT": QColor("#22c55e"),        # Green
    "NUMBER": QColor("#3b82f6"),      # Blue
    "INTEGER": QColor("#6366f1"),     # Indigo
    "BOOLEAN": QColor("#ef4444"),     # Red
    "CONDITIONING": QColor("#a855f7"), # Purple
    "MODEL": QColor("#14b8a6"),       # Teal
    "LATENT": QColor("#ec4899"),      # Pink
    "ANY": QColor("#9ca3af"),         # Gray
}


@dataclass
class CanvasTransform:
    """Handles canvas pan and zoom transformations."""
    offset_x: float = 0.0
    offset_y: float = 0.0
    zoom: float = 1.0
    
    def screen_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        """Convert screen coordinates to canvas coordinates."""
        return (
            (x - self.offset_x) / self.zoom,
            (y - self.offset_y) / self.zoom,
        )
    
    def canvas_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Convert canvas coordinates to screen coordinates."""
        return (
            x * self.zoom + self.offset_x,
            y * self.zoom + self.offset_y,
        )
    
    def zoom_at(self, x: float, y: float, factor: float) -> None:
        """Zoom centered on a point."""
        # Get canvas position before zoom
        cx, cy = self.screen_to_canvas(x, y)
        
        # Apply zoom
        self.zoom *= factor
        self.zoom = max(0.1, min(4.0, self.zoom))  # Clamp zoom
        
        # Adjust offset to keep point under cursor
        new_sx, new_sy = self.canvas_to_screen(cx, cy)
        self.offset_x += x - new_sx
        self.offset_y += y - new_sy


@dataclass
class VisualNode:
    """Visual representation data for a node."""
    node_id: str  # UUID as string for dict key
    x: float
    y: float
    width: float
    height: float
    title: str
    category: str
    inputs: list[tuple[str, str]]   # (name, data_type)
    outputs: list[tuple[str, str]]  # (name, data_type)
    type_id: str = ""  # Links to NodeType.id and core Node.type_id
    is_selected: bool = False
    is_hovered: bool = False
    preview_image: object = None  # PIL Image or None


class NodeGraphCanvas(QWidget):
    """
    OpenGL-accelerated canvas for displaying and editing node graphs.
    
    Signals:
        node_selected: Emitted when a node is selected (node_id or None)
        node_moved: Emitted when a node is moved (node_id, x, y)
        connection_created: Emitted when a connection is made
        connection_removed: Emitted when a connection is removed
        context_menu_requested: Emitted for context menu (position)
    """
    
    # Signals
    node_selected = Signal(object)  # NodeId or None
    node_moved = Signal(object, float, float)  # NodeId, x, y
    node_deleted = Signal(object)  # NodeId - emitted when user deletes a node
    connection_created = Signal(object, str, object, str)  # src_id, src_out, tgt_id, tgt_in
    connection_removed = Signal(object, str, object, str)  # src_id, src_out, tgt_id, tgt_in
    context_menu_requested = Signal(float, float)  # canvas x, y
    
    # Layout constants
    NODE_HEADER_HEIGHT = 28
    NODE_MIN_WIDTH = 180
    NODE_PADDING = 10
    SOCKET_RADIUS = 6
    SOCKET_SPACING = 24
    GRID_SIZE = 20
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Canvas state
        self._transform = CanvasTransform()
        self._nodes: dict[str, VisualNode] = {}
        self._connections: list[tuple[str, str, str, str]] = []  # (src_id, src_out, tgt_id, tgt_in)
        
        # Interaction state
        self._selected_nodes: set[str] = set()
        self._selected_connection: tuple[str, str, str, str] | None = None  # Selected conn for deletion
        self._hovered_node: str | None = None
        self._hovered_socket: tuple[str, str, bool] | None = None  # (node_id, socket_name, is_output)
        
        # Drag state
        self._is_panning = False
        self._is_dragging_node = False
        self._is_dragging_connection = False
        self._drag_start_pos: QPointF | None = None
        self._drag_start_offset: tuple[float, float] = (0, 0)
        self._connection_start: tuple[str, str, bool] | None = None  # (node_id, socket_name, is_output)
        self._connection_end_pos: tuple[float, float] | None = None
        
        # Selection box
        self._is_selecting = False
        self._selection_start: tuple[float, float] | None = None
        self._selection_end: tuple[float, float] | None = None
        
        # Styling
        self._background_color = QColor("#1a1a2e")
        self._grid_color = QColor("#2a2a3e")
        self._selection_color = QColor("#4a9eff")
        
        # Fonts
        self._title_font = QFont("Inter", 11, QFont.Weight.Bold)
        self._socket_font = QFont("Inter", 9)
        
        # Set minimum size
        self.setMinimumSize(400, 300)
    
    # --- Public API ---
    
    def set_graph(self, graph: NodeGraph) -> None:
        """Load a node graph for display."""
        self._nodes.clear()
        self._connections.clear()
        self._selected_nodes.clear()
        
        # Convert graph nodes to visual nodes
        for node_id, node in graph.nodes.items():
            visual = VisualNode(
                node_id=str(node_id),
                x=node.position.x,
                y=node.position.y,
                width=node.size.width,
                height=node.size.height,
                title=node.type_id.split(".")[-1].replace("_", " ").title(),
                category=node.type_id.split(".")[0] if "." in node.type_id else "default",
                inputs=[],  # TODO: Get from node type registry
                outputs=[],
            )
            self._nodes[str(node_id)] = visual
        
        # Convert connections
        for conn in graph.connections:
            self._connections.append((
                str(conn.source.node_id),
                conn.source.output_name,
                str(conn.target.node_id),
                conn.target.input_name,
            ))
        
        self.update()
    
    def add_visual_node(
        self,
        node_id: str,
        x: float,
        y: float,
        title: str,
        category: str = "default",
        inputs: list[tuple[str, str]] | None = None,
        outputs: list[tuple[str, str]] | None = None,
    ) -> None:
        """Add a visual node to the canvas."""
        height = self._calculate_node_height(inputs or [], outputs or [])
        visual = VisualNode(
            node_id=node_id,
            x=x,
            y=y,
            width=self.NODE_MIN_WIDTH,
            height=height,
            title=title,
            category=category,
            inputs=inputs or [],
            outputs=outputs or [],
        )
        self._nodes[node_id] = visual
        self.update()
    
    def remove_visual_node(self, node_id: str) -> None:
        """Remove a visual node from the canvas."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._selected_nodes.discard(node_id)
            # Remove related connections
            self._connections = [
                c for c in self._connections
                if c[0] != node_id and c[2] != node_id
            ]
            self.update()
    
    def set_node_preview(self, node_id: str, image) -> None:
        """Set the preview thumbnail for a node (PIL Image or QImage)."""
        if node_id in self._nodes:
            # Convert PIL Image to QImage if needed
            if image is not None and hasattr(image, 'tobytes'):
                # It's a PIL Image - convert to QImage
                from PySide6.QtGui import QImage
                
                if image.mode == "RGBA":
                    fmt = QImage.Format.Format_RGBA8888
                else:
                    image = image.convert("RGBA")
                    fmt = QImage.Format.Format_RGBA8888
                
                data = image.tobytes("raw", "RGBA")
                qimg = QImage(data, image.width, image.height, fmt)
                self._nodes[node_id].preview_image = qimg.copy()  # Copy to own the data
            else:
                self._nodes[node_id].preview_image = image
            
            self.update()
    
    def add_visual_connection(
        self,
        source_node: str,
        source_output: str,
        target_node: str,
        target_input: str,
    ) -> None:
        """Add a visual connection."""
        self._connections.append((source_node, source_output, target_node, target_input))
        self.update()
    
    def remove_visual_connection(
        self,
        source_node: str,
        source_output: str,
        target_node: str,
        target_input: str,
    ) -> None:
        """Remove a visual connection."""
        conn = (source_node, source_output, target_node, target_input)
        if conn in self._connections:
            self._connections.remove(conn)
            self.connection_removed.emit(source_node, source_output, target_node, target_input)
            self.update()
    
    def delete_selected_nodes(self) -> None:
        """Delete all currently selected nodes."""
        for node_id in list(self._selected_nodes):
            # Remove connections first
            conns_to_remove = [
                c for c in self._connections
                if c[0] == node_id or c[2] == node_id
            ]
            for conn in conns_to_remove:
                self._connections.remove(conn)
                self.connection_removed.emit(*conn)
            
            # Remove node
            if node_id in self._nodes:
                del self._nodes[node_id]
            self._selected_nodes.discard(node_id)
            self.node_deleted.emit(node_id)
        
        self.node_selected.emit(None)
        self.update()
    
    def select_node(self, node_id: str | None, add: bool = False) -> None:
        """Select a node (or deselect all if None)."""
        if not add:
            for nid in self._selected_nodes:
                if nid in self._nodes:
                    self._nodes[nid].is_selected = False
            self._selected_nodes.clear()
        
        if node_id and node_id in self._nodes:
            self._nodes[node_id].is_selected = True
            self._selected_nodes.add(node_id)
        
        self.node_selected.emit(node_id)
        self.update()
    
    def frame_all(self) -> None:
        """Adjust view to show all nodes."""
        if not self._nodes:
            return
        
        # Calculate bounds
        min_x = min(n.x for n in self._nodes.values())
        min_y = min(n.y for n in self._nodes.values())
        max_x = max(n.x + n.width for n in self._nodes.values())
        max_y = max(n.y + n.height for n in self._nodes.values())
        
        # Add padding
        padding = 50
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Calculate zoom to fit
        bounds_w = max_x - min_x
        bounds_h = max_y - min_y
        zoom_x = self.width() / bounds_w
        zoom_y = self.height() / bounds_h
        self._transform.zoom = min(zoom_x, zoom_y, 1.0)
        
        # Center
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        self._transform.offset_x = self.width() / 2 - center_x * self._transform.zoom
        self._transform.offset_y = self.height() / 2 - center_y * self._transform.zoom
        
        self.update()
    
    # --- Rendering ---
    
    def paintEvent(self, event: QPaintEvent) -> None:
        """Render the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self._background_color)
        
        # Grid
        self._draw_grid(painter)
        
        # Connections
        for conn in self._connections:
            is_selected = (conn == self._selected_connection)
            self._draw_connection(painter, *conn, is_selected=is_selected)
        
        # Connection being dragged
        if self._is_dragging_connection and self._connection_start and self._connection_end_pos:
            self._draw_temp_connection(painter)
        
        # Nodes
        for visual in self._nodes.values():
            self._draw_node(painter, visual)
        
        # Selection box
        if self._is_selecting and self._selection_start and self._selection_end:
            self._draw_selection_box(painter)
        
        painter.end()
    
    def _draw_grid(self, painter: QPainter) -> None:
        """Draw the background grid."""
        pen = QPen(self._grid_color, 1)
        painter.setPen(pen)
        
        # Calculate grid spacing in screen coordinates
        spacing = self.GRID_SIZE * self._transform.zoom
        if spacing < 10:
            spacing *= 5  # Show every 5th line when zoomed out
        
        # Calculate visible area
        start_x = int(-self._transform.offset_x / spacing) * spacing + self._transform.offset_x % spacing
        start_y = int(-self._transform.offset_y / spacing) * spacing + self._transform.offset_y % spacing
        
        # Draw vertical lines
        x = start_x
        while x < self.width():
            painter.drawLine(int(x), 0, int(x), self.height())
            x += spacing
        
        # Draw horizontal lines
        y = start_y
        while y < self.height():
            painter.drawLine(0, int(y), self.width(), int(y))
            y += spacing
    
    def _draw_node(self, painter: QPainter, node: VisualNode) -> None:
        """Draw a single node."""
        painter.save()  # Isolate painter state per node
        
        # Transform to screen coordinates
        sx, sy = self._transform.canvas_to_screen(node.x, node.y)
        sw = node.width * self._transform.zoom
        sh = node.height * self._transform.zoom
        header_h = self.NODE_HEADER_HEIGHT * self._transform.zoom
        
        # Node body - rounded rect
        rect = QRectF(sx, sy, sw, sh)
        path = QPainterPath()
        path.addRoundedRect(rect, 8, 8)
        
        # Background
        body_color = QColor("#2d2d3d")
        painter.fillPath(path, body_color)
        
        # Header - only top corners rounded
        # Draw as: rounded rect at top + rectangle below to fill gap
        header_path = QPainterPath()
        # Full header area with top corners rounded
        header_path.moveTo(sx + 8, sy)
        header_path.lineTo(sx + sw - 8, sy)
        header_path.arcTo(QRectF(sx + sw - 16, sy, 16, 16), 90, -90)  # top-right
        header_path.lineTo(sx + sw, sy + header_h)
        header_path.lineTo(sx, sy + header_h)
        header_path.lineTo(sx, sy + 8)
        header_path.arcTo(QRectF(sx, sy, 16, 16), 180, -90)  # top-left
        header_path.closeSubpath()
        
        header_color = NODE_COLORS.get(node.category, NODE_COLORS["default"])
        painter.fillPath(header_path, header_color)
        
        # Border
        if node.is_selected:
            pen = QPen(self._selection_color, 2)
        elif node.is_hovered:
            pen = QPen(QColor("#6b7280"), 2)
        else:
            pen = QPen(QColor("#3f3f4f"), 1)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Title
        painter.setPen(Qt.GlobalColor.white)
        title_font = QFont(self._title_font)
        title_font.setPointSizeF(title_font.pointSizeF() * self._transform.zoom)
        painter.setFont(title_font)
        title_rect = QRectF(sx + 10, sy, sw - 20, self.NODE_HEADER_HEIGHT * self._transform.zoom)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter, node.title)
        
        # Preview image thumbnail (if any)
        if node.preview_image is not None:
            from PySide6.QtGui import QImage, QPixmap
            
            # Calculate preview area (after header, before sockets)
            preview_margin = 8 * self._transform.zoom
            preview_size = min(sw - 20 * self._transform.zoom, 80 * self._transform.zoom)
            preview_x = sx + (sw - preview_size) / 2
            preview_y = sy + header_h + preview_margin
            
            # Draw the preview
            if isinstance(node.preview_image, QImage):
                pixmap = QPixmap.fromImage(node.preview_image)
                scaled = pixmap.scaled(
                    int(preview_size), int(preview_size),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                # Center the scaled image
                offset_x = (preview_size - scaled.width()) / 2
                offset_y = (preview_size - scaled.height()) / 2
                painter.drawPixmap(int(preview_x + offset_x), int(preview_y + offset_y), scaled)
        
        # Sockets
        socket_font = QFont(self._socket_font)
        socket_font.setPointSizeF(socket_font.pointSizeF() * self._transform.zoom)
        painter.setFont(socket_font)
        
        socket_y = sy + self.NODE_HEADER_HEIGHT * self._transform.zoom + 12 * self._transform.zoom
        socket_radius = self.SOCKET_RADIUS * self._transform.zoom
        socket_spacing = self.SOCKET_SPACING * self._transform.zoom
        
        # Input sockets (left side)
        for i, (name, dtype) in enumerate(node.inputs):
            y = socket_y + i * socket_spacing
            color = SOCKET_COLORS.get(dtype, SOCKET_COLORS["ANY"])
            
            # Socket circle
            painter.setBrush(color)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawEllipse(QPointF(sx, y), socket_radius, socket_radius)
            
            # Label
            painter.setPen(QColor("#cccccc"))
            label_rect = QRectF(sx + socket_radius + 4, y - socket_spacing/2, sw/2, socket_spacing)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignVCenter, name)
        
        # Output sockets (right side)
        for i, (name, dtype) in enumerate(node.outputs):
            y = socket_y + i * socket_spacing
            color = SOCKET_COLORS.get(dtype, SOCKET_COLORS["ANY"])
            
            # Socket circle
            painter.setBrush(color)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawEllipse(QPointF(sx + sw, y), socket_radius, socket_radius)
            
            # Label (right-aligned)
            painter.setPen(QColor("#cccccc"))
            label_rect = QRectF(sx + sw/2, y - socket_spacing/2, sw/2 - socket_radius - 4, socket_spacing)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, name)
        
        painter.restore()  # Restore painter state
    
    def _draw_connection(
        self,
        painter: QPainter,
        src_node: str,
        src_output: str,
        tgt_node: str,
        tgt_input: str,
        is_selected: bool = False,
    ) -> None:
        """Draw a connection between two nodes."""
        if src_node not in self._nodes or tgt_node not in self._nodes:
            return
        
        src = self._nodes[src_node]
        tgt = self._nodes[tgt_node]
        
        # Get socket positions
        src_pos = self._get_socket_position(src, src_output, is_output=True)
        tgt_pos = self._get_socket_position(tgt, tgt_input, is_output=False)
        
        if not src_pos or not tgt_pos:
            return
        
        # Transform to screen
        sx, sy = self._transform.canvas_to_screen(*src_pos)
        tx, ty = self._transform.canvas_to_screen(*tgt_pos)
        
        # Use highlight color if selected
        if is_selected:
            color = self._selection_color
            self._draw_bezier_connection(painter, sx, sy, tx, ty, color, width=3)
        else:
            self._draw_bezier_connection(painter, sx, sy, tx, ty, QColor("#888888"))
    
    def _draw_temp_connection(self, painter: QPainter) -> None:
        """Draw the connection being dragged."""
        if not self._connection_start or not self._connection_end_pos:
            return
        
        node_id, socket_name, is_output = self._connection_start
        if node_id not in self._nodes:
            return
        
        node = self._nodes[node_id]
        start_pos = self._get_socket_position(node, socket_name, is_output)
        if not start_pos:
            return
        
        sx, sy = self._transform.canvas_to_screen(*start_pos)
        ex, ey = self._connection_end_pos
        
        if not is_output:
            # Swap if dragging from input
            sx, sy, ex, ey = ex, ey, sx, sy
        
        self._draw_bezier_connection(painter, sx, sy, ex, ey, self._selection_color)
    
    def _draw_bezier_connection(
        self,
        painter: QPainter,
        x1: float, y1: float,
        x2: float, y2: float,
        color: QColor,
        width: int = 2,
    ) -> None:
        """Draw a bezier curve connection."""
        path = QPainterPath()
        path.moveTo(x1, y1)
        
        # Calculate control points
        dx = abs(x2 - x1) * 0.5
        dx = max(dx, 50 * self._transform.zoom)
        
        path.cubicTo(x1 + dx, y1, x2 - dx, y2, x2, y2)
        
        pen = QPen(color, width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
    
    def _draw_selection_box(self, painter: QPainter) -> None:
        """Draw the selection rectangle."""
        if not self._selection_start or not self._selection_end:
            return
        
        x1, y1 = self._transform.canvas_to_screen(*self._selection_start)
        x2, y2 = self._transform.canvas_to_screen(*self._selection_end)
        
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        
        # Fill
        fill_color = QColor(self._selection_color)
        fill_color.setAlpha(30)
        painter.fillRect(rect, fill_color)
        
        # Border
        painter.setPen(QPen(self._selection_color, 1, Qt.PenStyle.DashLine))
        painter.drawRect(rect)
    
    # --- Helpers ---
    
    def _calculate_node_height(
        self,
        inputs: list[tuple[str, str]],
        outputs: list[tuple[str, str]],
    ) -> float:
        """Calculate node height based on sockets."""
        num_sockets = max(len(inputs), len(outputs), 1)
        return self.NODE_HEADER_HEIGHT + self.NODE_PADDING * 2 + num_sockets * self.SOCKET_SPACING
    
    def _get_socket_position(
        self,
        node: VisualNode,
        socket_name: str,
        is_output: bool,
    ) -> tuple[float, float] | None:
        """Get the canvas position of a socket."""
        sockets = node.outputs if is_output else node.inputs
        
        for i, (name, _) in enumerate(sockets):
            if name == socket_name:
                x = node.x + (node.width if is_output else 0)
                y = node.y + self.NODE_HEADER_HEIGHT + 12 + i * self.SOCKET_SPACING
                return (x, y)
        
        return None
    
    def _node_at(self, canvas_x: float, canvas_y: float) -> str | None:
        """Get the node at canvas coordinates (topmost)."""
        # Check in reverse order (topmost first)
        for node_id, node in reversed(list(self._nodes.items())):
            if (node.x <= canvas_x <= node.x + node.width and
                node.y <= canvas_y <= node.y + node.height):
                return node_id
        return None
    
    def _socket_at(
        self,
        canvas_x: float,
        canvas_y: float,
    ) -> tuple[str, str, bool] | None:
        """Get the socket at canvas coordinates."""
        hit_radius = self.SOCKET_RADIUS * 1.5
        
        for node_id, node in self._nodes.items():
            # Check output sockets
            for i, (name, _) in enumerate(node.outputs):
                sx = node.x + node.width
                sy = node.y + self.NODE_HEADER_HEIGHT + 12 + i * self.SOCKET_SPACING
                if math.hypot(canvas_x - sx, canvas_y - sy) < hit_radius:
                    return (node_id, name, True)
            
            # Check input sockets
            for i, (name, _) in enumerate(node.inputs):
                sx = node.x
                sy = node.y + self.NODE_HEADER_HEIGHT + 12 + i * self.SOCKET_SPACING
                if math.hypot(canvas_x - sx, canvas_y - sy) < hit_radius:
                    return (node_id, name, False)
        
        return None
    
    def _connection_at(
        self,
        canvas_x: float,
        canvas_y: float,
        threshold: float = 15.0,
    ) -> tuple[str, str, str, str] | None:
        """Get the connection at canvas coordinates (for click-to-select)."""
        for conn in self._connections:
            src_id, src_out, tgt_id, tgt_in = conn
            
            if src_id not in self._nodes or tgt_id not in self._nodes:
                continue
            
            src_node = self._nodes[src_id]
            tgt_node = self._nodes[tgt_id]
            
            # Find source socket position
            src_pos = self._get_socket_position(src_node, src_out, is_output=True)
            tgt_pos = self._get_socket_position(tgt_node, tgt_in, is_output=False)
            
            if not src_pos or not tgt_pos:
                continue
            
            src_x, src_y = src_pos
            tgt_x, tgt_y = tgt_pos
            
            # Match control point calculation from _draw_bezier_connection
            dx = abs(tgt_x - src_x) * 0.5
            dx = max(dx, 50)  # Minimum curvature
            
            # Check distance to bezier curve using more sample points
            for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                # Cubic bezier point
                bx = (1-t)**3 * src_x + 3*(1-t)**2*t*(src_x + dx) + 3*(1-t)*t**2*(tgt_x - dx) + t**3*tgt_x
                by = (1-t)**3 * src_y + 3*(1-t)**2*t*src_y + 3*(1-t)*t**2*tgt_y + t**3*tgt_y
                
                if math.hypot(canvas_x - bx, canvas_y - by) < threshold:
                    return conn
        
        return None
    
    # --- Mouse Events ---
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        pos = event.position()
        cx, cy = self._transform.screen_to_canvas(pos.x(), pos.y())
        
        # Middle mouse or Space+Left for panning
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._drag_start_pos = pos
            self._drag_start_offset = (self._transform.offset_x, self._transform.offset_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        
        # Right click for context menu
        if event.button() == Qt.MouseButton.RightButton:
            self.context_menu_requested.emit(cx, cy)
            return
        
        # Left click
        if event.button() == Qt.MouseButton.LeftButton:
            # Check for socket hit first
            socket = self._socket_at(cx, cy)
            if socket:
                self._selected_connection = None  # Deselect connection
                self._is_dragging_connection = True
                self._connection_start = socket
                self._connection_end_pos = (pos.x(), pos.y())
                self.update()
                return
            
            # Check for node hit
            node_id = self._node_at(cx, cy)
            if node_id:
                self._selected_connection = None  # Deselect connection
                # Select node
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    # Toggle selection
                    if node_id in self._selected_nodes:
                        self._nodes[node_id].is_selected = False
                        self._selected_nodes.discard(node_id)
                    else:
                        self._nodes[node_id].is_selected = True
                        self._selected_nodes.add(node_id)
                else:
                    if node_id not in self._selected_nodes:
                        self.select_node(node_id)
                
                # Start drag
                self._is_dragging_node = True
                self._drag_start_pos = pos
                self.update()
                return
            
            # Check for connection hit
            conn = self._connection_at(cx, cy)
            if conn:
                self._selected_connection = conn
                self.select_node(None)  # Deselect nodes
                self.update()
                return
            
            # Click on empty space - start selection box or deselect
            self._selected_connection = None
            if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.select_node(None)
            
            self._is_selecting = True
            self._selection_start = (cx, cy)
            self._selection_end = (cx, cy)
        
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        pos = event.position()
        cx, cy = self._transform.screen_to_canvas(pos.x(), pos.y())
        
        # Panning
        if self._is_panning and self._drag_start_pos:
            dx = pos.x() - self._drag_start_pos.x()
            dy = pos.y() - self._drag_start_pos.y()
            self._transform.offset_x = self._drag_start_offset[0] + dx
            self._transform.offset_y = self._drag_start_offset[1] + dy
            self.update()
            return
        
        # Dragging nodes
        if self._is_dragging_node and self._drag_start_pos:
            dx = (pos.x() - self._drag_start_pos.x()) / self._transform.zoom
            dy = (pos.y() - self._drag_start_pos.y()) / self._transform.zoom
            
            for node_id in self._selected_nodes:
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    # Store original positions on first move
                    if not hasattr(node, '_drag_start_x'):
                        node._drag_start_x = node.x
                        node._drag_start_y = node.y
                    node.x = node._drag_start_x + dx
                    node.y = node._drag_start_y + dy
            
            self.update()
            return
        
        # Dragging connection
        if self._is_dragging_connection:
            self._connection_end_pos = (pos.x(), pos.y())
            self.update()
            return
        
        # Selection box
        if self._is_selecting:
            self._selection_end = (cx, cy)
            self.update()
            return
        
        # Hover detection
        old_hovered = self._hovered_node
        self._hovered_node = self._node_at(cx, cy)
        
        if old_hovered != self._hovered_node:
            if old_hovered and old_hovered in self._nodes:
                self._nodes[old_hovered].is_hovered = False
            if self._hovered_node and self._hovered_node in self._nodes:
                self._nodes[self._hovered_node].is_hovered = True
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        pos = event.position()
        cx, cy = self._transform.screen_to_canvas(pos.x(), pos.y())
        
        # End panning
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # End node drag
        if self._is_dragging_node:
            self._is_dragging_node = False
            # Clear drag start positions
            for node_id in self._selected_nodes:
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    if hasattr(node, '_drag_start_x'):
                        del node._drag_start_x
                        del node._drag_start_y
                    # Emit move signal
                    self.node_moved.emit(node_id, node.x, node.y)
        
        # End connection drag
        if self._is_dragging_connection and self._connection_start:
            # Check if we ended on a socket
            target = self._socket_at(cx, cy)
            if target and target[0] != self._connection_start[0]:
                src_node, src_socket, src_is_output = self._connection_start
                tgt_node, tgt_socket, tgt_is_output = target
                
                # Must connect output to input
                if src_is_output and not tgt_is_output:
                    self.connection_created.emit(src_node, src_socket, tgt_node, tgt_socket)
                elif not src_is_output and tgt_is_output:
                    self.connection_created.emit(tgt_node, tgt_socket, src_node, src_socket)
            
            self._is_dragging_connection = False
            self._connection_start = None
            self._connection_end_pos = None
        
        # End selection box
        if self._is_selecting and self._selection_start and self._selection_end:
            # Select nodes in box
            x1 = min(self._selection_start[0], self._selection_end[0])
            y1 = min(self._selection_start[1], self._selection_end[1])
            x2 = max(self._selection_start[0], self._selection_end[0])
            y2 = max(self._selection_start[1], self._selection_end[1])
            
            for node_id, node in self._nodes.items():
                if (node.x < x2 and node.x + node.width > x1 and
                    node.y < y2 and node.y + node.height > y1):
                    node.is_selected = True
                    self._selected_nodes.add(node_id)
            
            self._is_selecting = False
            self._selection_start = None
            self._selection_end = None
        
        self._drag_start_pos = None
        self.update()
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        pos = event.position()
        delta = event.angleDelta().y()
        
        factor = 1.1 if delta > 0 else 0.9
        self._transform.zoom_at(pos.x(), pos.y(), factor)
        
        self.update()
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press."""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # Delete selected connection first, then nodes
            if self._selected_connection:
                conn = self._selected_connection
                self._connections.remove(conn)
                self.connection_removed.emit(*conn)
                self._selected_connection = None
                self.update()
            elif self._selected_nodes:
                self.delete_selected_nodes()
        elif event.key() == Qt.Key.Key_F:
            # Frame all
            self.frame_all()
        elif event.key() == Qt.Key.Key_Escape:
            # Cancel current operation
            if self._is_dragging_connection:
                self._is_dragging_connection = False
                self._connection_start = None
                self._connection_end_pos = None
                self.update()
        
        super().keyPressEvent(event)
