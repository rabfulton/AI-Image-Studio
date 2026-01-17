"""
Main Window - The primary application window.

This module provides the main window for AI Image Studio, including
the menu bar, dock panels, and central workspace.
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QDockWidget,
    QMenuBar,
    QMenu,
    QStatusBar,
    QLabel,
    QMessageBox,
    QFileDialog,
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence


class MainWindow(QMainWindow):
    """
    The main application window for AI Image Studio.
    
    Contains:
    - Menu bar with File, Edit, View, Workflow, Providers, Help menus
    - Central splitter with Node Graph (left) and Output Studio (right)
    - Dock panels for Node Library, Properties, Gallery, History
    - Status bar
    """
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        self.setWindowTitle("AI Image Studio")
        self.setMinimumSize(1200, 800)
        
        # Initialize settings
        self._settings = QSettings("AIImageStudio", "AIImageStudio")
        
        # Setup UI components
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_dock_widgets()
        self._setup_status_bar()
        
        # Restore window state
        self._restore_state()
    
    def _setup_menu_bar(self) -> None:
        """Create and configure the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._on_undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._on_redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self._on_settings)
        edit_menu.addAction(settings_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        self._view_menu = view_menu  # Store for dock toggle actions
        
        # Workflow menu
        workflow_menu = menubar.addMenu("&Workflow")
        
        execute_action = QAction("&Execute", self)
        execute_action.setShortcut(QKeySequence("F5"))
        execute_action.triggered.connect(self._on_execute_workflow)
        workflow_menu.addAction(execute_action)
        
        cancel_action = QAction("&Cancel", self)
        cancel_action.setShortcut(QKeySequence("Escape"))
        cancel_action.triggered.connect(self._on_cancel_execution)
        workflow_menu.addAction(cancel_action)
        
        # Providers menu
        providers_menu = menubar.addMenu("&Providers")
        
        manage_providers_action = QAction("&Manage Providers...", self)
        manage_providers_action.triggered.connect(self._on_manage_providers)
        providers_menu.addAction(manage_providers_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _setup_central_widget(self) -> None:
        """Create the central widget with Node Graph and Output Studio."""
        from ai_image_studio.ui.node_graph import NodeGraphCanvas
        
        # Create main splitter
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Node Graph Canvas (real widget)
        self._node_graph_canvas = NodeGraphCanvas()
        self._node_graph_canvas.node_selected.connect(self._on_node_selected)
        self._node_graph_canvas.context_menu_requested.connect(self._on_canvas_context_menu)
        
        # Add some demo nodes
        self._add_demo_nodes()
        
        # Placeholder for Output Studio
        output_studio_placeholder = QWidget()
        output_studio_layout = QVBoxLayout(output_studio_placeholder)
        output_studio_label = QLabel("Output Studio")
        output_studio_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        output_studio_label.setStyleSheet("""
            QLabel {
                background-color: #16213e;
                color: #808080;
                font-size: 24px;
                border: 1px solid #333;
            }
        """)
        output_studio_layout.addWidget(output_studio_label)
        output_studio_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add to splitter
        self._main_splitter.addWidget(self._node_graph_canvas)
        self._main_splitter.addWidget(output_studio_placeholder)
        self._main_splitter.setSizes([600, 400])
        
        self.setCentralWidget(self._main_splitter)
    
    def _add_demo_nodes(self) -> None:
        """Add demo nodes to showcase the canvas."""
        # Prompt node
        self._node_graph_canvas.add_visual_node(
            node_id="node_1",
            x=50, y=100,
            title="Prompt",
            category="input",
            inputs=[],
            outputs=[("text", "TEXT")],
        )
        
        # Text-to-Image node
        self._node_graph_canvas.add_visual_node(
            node_id="node_2",
            x=300, y=80,
            title="Text to Image",
            category="generation",
            inputs=[("prompt", "TEXT"), ("negative", "TEXT"), ("model", "MODEL")],
            outputs=[("image", "IMAGE"), ("latent", "LATENT")],
        )
        
        # Preview node
        self._node_graph_canvas.add_visual_node(
            node_id="node_3",
            x=580, y=100,
            title="Preview",
            category="output",
            inputs=[("image", "IMAGE")],
            outputs=[],
        )
        
        # Connect prompt to text-to-image to preview
        self._node_graph_canvas.add_visual_connection("node_1", "text", "node_2", "prompt")
        self._node_graph_canvas.add_visual_connection("node_2", "image", "node_3", "image")
    
    def _on_node_selected(self, node_id) -> None:
        """Handle node selection."""
        if node_id:
            self.statusBar().showMessage(f"Selected: {node_id}", 2000)
    
    def _on_canvas_context_menu(self, x: float, y: float) -> None:
        """Handle canvas context menu."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QCursor
        
        menu = QMenu(self)
        
        # Add node submenu
        add_menu = menu.addMenu("Add Node")
        
        # Input nodes
        input_menu = add_menu.addMenu("Input")
        input_menu.addAction("Prompt")
        input_menu.addAction("Image Input")
        input_menu.addAction("Mask Input")
        
        # Generation nodes
        gen_menu = add_menu.addMenu("Generation")
        gen_menu.addAction("Text to Image")
        gen_menu.addAction("Image to Image")
        gen_menu.addAction("Inpaint")
        
        # Output nodes
        out_menu = add_menu.addMenu("Output")
        out_menu.addAction("Preview")
        out_menu.addAction("Save Image")
        
        menu.addSeparator()
        menu.addAction("Frame All (F)")
        
        menu.exec(QCursor.pos())
    
    def _setup_dock_widgets(self) -> None:
        """Create dock widgets for panels."""
        # Node Library dock (left)
        self._node_library_dock = QDockWidget("Node Library", self)
        self._node_library_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        node_library_placeholder = QLabel("Node Library\n\n(Tree view of nodes)")
        node_library_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        node_library_placeholder.setMinimumWidth(200)
        self._node_library_dock.setWidget(node_library_placeholder)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._node_library_dock)
        
        # Properties dock (right)
        self._properties_dock = QDockWidget("Properties", self)
        self._properties_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        properties_placeholder = QLabel("Properties\n\n(Selected node parameters)")
        properties_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        properties_placeholder.setMinimumWidth(250)
        self._properties_dock.setWidget(properties_placeholder)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._properties_dock)
        
        # Gallery dock (right, bottom)
        self._gallery_dock = QDockWidget("Gallery", self)
        self._gallery_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        gallery_placeholder = QLabel("Gallery\n\n(Generated images)")
        gallery_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gallery_dock.setWidget(gallery_placeholder)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._gallery_dock)
        
        # Stack gallery below properties
        self.tabifyDockWidget(self._properties_dock, self._gallery_dock)
        self._properties_dock.raise_()  # Show properties by default
        
        # History dock (bottom)
        self._history_dock = QDockWidget("History", self)
        self._history_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        history_placeholder = QLabel("History (Undo/Redo states)")
        history_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._history_dock.setWidget(history_placeholder)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._history_dock)
        
        # Queue dock (bottom)
        self._queue_dock = QDockWidget("Queue", self)
        self._queue_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        queue_placeholder = QLabel("Queue (Pending jobs)")
        queue_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._queue_dock.setWidget(queue_placeholder)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._queue_dock)
        
        # Tab history and queue together
        self.tabifyDockWidget(self._history_dock, self._queue_dock)
        self._history_dock.raise_()
        
        # Add toggle actions to View menu
        self._view_menu.addAction(self._node_library_dock.toggleViewAction())
        self._view_menu.addAction(self._properties_dock.toggleViewAction())
        self._view_menu.addAction(self._gallery_dock.toggleViewAction())
        self._view_menu.addAction(self._history_dock.toggleViewAction())
        self._view_menu.addAction(self._queue_dock.toggleViewAction())
    
    def _setup_status_bar(self) -> None:
        """Create and configure the status bar."""
        status_bar = self.statusBar()
        
        # Ready message
        status_bar.showMessage("Ready")
        
        # Permanent widgets on the right
        self._status_memory = QLabel("Memory: --")
        status_bar.addPermanentWidget(self._status_memory)
        
        self._status_gpu = QLabel("GPU: --")
        status_bar.addPermanentWidget(self._status_gpu)
    
    def _restore_state(self) -> None:
        """Restore window geometry and state from settings."""
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self._settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def closeEvent(self, event) -> None:
        """Save state before closing."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        super().closeEvent(event)
    
    # --- Menu action handlers ---
    
    def _on_new_project(self) -> None:
        """Create a new project."""
        # TODO: Implement new project creation
        self.statusBar().showMessage("New project created", 3000)
    
    def _on_open_project(self) -> None:
        """Open an existing project."""
        # TODO: Implement project opening
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "AI Image Studio Projects (*.aiproj);;All Files (*)"
        )
        if file_path:
            self.statusBar().showMessage(f"Opened: {file_path}", 3000)
    
    def _on_save_project(self) -> None:
        """Save the current project."""
        # TODO: Implement project saving
        self.statusBar().showMessage("Project saved", 3000)
    
    def _on_save_project_as(self) -> None:
        """Save the current project with a new name."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "AI Image Studio Projects (*.aiproj);;All Files (*)"
        )
        if file_path:
            self.statusBar().showMessage(f"Saved as: {file_path}", 3000)
    
    def _on_undo(self) -> None:
        """Undo the last action."""
        # TODO: Implement undo
        self.statusBar().showMessage("Undo", 2000)
    
    def _on_redo(self) -> None:
        """Redo the last undone action."""
        # TODO: Implement redo
        self.statusBar().showMessage("Redo", 2000)
    
    def _on_settings(self) -> None:
        """Open the settings dialog."""
        # TODO: Implement settings dialog
        QMessageBox.information(
            self,
            "Settings",
            "Settings dialog not yet implemented."
        )
    
    def _on_execute_workflow(self) -> None:
        """Execute the current workflow."""
        # TODO: Implement workflow execution
        self.statusBar().showMessage("Executing workflow...", 3000)
    
    def _on_cancel_execution(self) -> None:
        """Cancel the current execution."""
        # TODO: Implement execution cancellation
        self.statusBar().showMessage("Execution cancelled", 2000)
    
    def _on_manage_providers(self) -> None:
        """Open the provider management dialog."""
        # TODO: Implement provider management
        QMessageBox.information(
            self,
            "Providers",
            "Provider management not yet implemented."
        )
    
    def _on_about(self) -> None:
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About AI Image Studio",
            "<h2>AI Image Studio</h2>"
            "<p>Version 0.1.0</p>"
            "<p>A powerful, Linux-native AI image generation and editing application.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Node-based workflows</li>"
            "<li>Local and hosted AI models</li>"
            "<li>600+ G'MIC filters</li>"
            "<li>Batch processing and scripting</li>"
            "</ul>"
        )
