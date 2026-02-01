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
from PySide6.QtGui import QAction, QImage, QKeySequence


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
        
        # Initialize core model (NodeGraph mirrors visual canvas)
        from ai_image_studio.core.graph import NodeGraph
        self._graph = NodeGraph("Workspace")

        # Layer stack (Output Studio)
        from ai_image_studio.core.layers import LayerStack
        self._layer_stack = LayerStack()
        self._selected_layer_index: int | None = None
        
        # Node ID mapping: visual_id (str) <-> core_id (UUID)
        self._node_id_map: dict[str, object] = {}  # str -> UUID
        
        # Execution state
        self._current_job_id = None
        
        # Setup UI components
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_dock_widgets()
        self._setup_status_bar()
        
        # Connect canvas signals for syncing
        self._connect_canvas_signals()
        
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
        
        # Output Studio (real widget)
        from ai_image_studio.ui.output_studio import OutputStudio
        self._output_studio = OutputStudio()
        
        # Add to splitter
        self._main_splitter.addWidget(self._node_graph_canvas)
        self._main_splitter.addWidget(self._output_studio)
        self._main_splitter.setSizes([600, 400])
        
        self.setCentralWidget(self._main_splitter)
    
    def _add_demo_nodes(self) -> None:
        """Placeholder - no demo nodes, users add from library."""
        # Canvas starts empty - welcome message logged in _restore_state
        pass
    
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
        
        # Filter nodes
        filter_menu = add_menu.addMenu("Filter")
        filter_menu.addAction("G'MIC Filter")
        
        # Output nodes
        out_menu = add_menu.addMenu("Output")
        out_menu.addAction("Preview")
        out_menu.addAction("Save Image")
        
        menu.addSeparator()
        menu.addAction("Frame All (F)")
        
        menu.exec(QCursor.pos())
    
    def _setup_dock_widgets(self) -> None:
        """Create dock widgets for panels."""
        from ai_image_studio.ui.panels import NodeLibraryPanel, PropertiesPanel, LayersPanel
        
        # Node Library dock (left)
        self._node_library_dock = QDockWidget("Node Library", self)
        self._node_library_dock.setObjectName("node_library_dock")
        self._node_library_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._node_library_panel = NodeLibraryPanel()
        self._node_library_panel.node_requested.connect(self._on_add_node_requested)
        self._node_library_panel.setMinimumWidth(200)
        self._node_library_panel.refresh_from_registry()  # Load actual registered nodes
        self._node_library_dock.setWidget(self._node_library_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._node_library_dock)
        
        # Properties dock (right)
        self._properties_dock = QDockWidget("Properties", self)
        self._properties_dock.setObjectName("properties_dock")
        self._properties_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._properties_panel = PropertiesPanel()
        self._properties_panel.parameter_changed.connect(self._on_parameter_changed)
        self._properties_panel.setMinimumWidth(250)
        self._properties_dock.setWidget(self._properties_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._properties_dock)
        
        # Gallery dock (right, bottom)
        from ai_image_studio.ui.panels import GalleryPanel
        
        self._gallery_dock = QDockWidget("Gallery", self)
        self._gallery_dock.setObjectName("gallery_dock")
        self._gallery_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._gallery_panel = GalleryPanel()
        self._gallery_panel.image_load_requested.connect(self._on_gallery_load)
        self._gallery_panel.image_selected.connect(self._on_gallery_preview)
        self._gallery_panel.workflow_load_requested.connect(self._on_workflow_load)
        self._gallery_dock.setWidget(self._gallery_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._gallery_dock)

        # Layers dock (right)
        self._layers_dock = QDockWidget("Layers", self)
        self._layers_dock.setObjectName("layers_dock")
        self._layers_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._layers_panel = LayersPanel()
        self._layers_panel.layer_selected.connect(self._on_layer_selected)
        self._layers_panel.layer_visibility_changed.connect(self._on_layer_visibility_changed)
        self._layers_panel.add_layer_requested.connect(self._on_add_layer_requested)
        self._layers_panel.delete_layer_requested.connect(self._on_delete_layer_requested)
        self._layers_dock.setWidget(self._layers_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._layers_dock)

        # Tabify panels on the right
        self.tabifyDockWidget(self._properties_dock, self._layers_dock)
        self.tabifyDockWidget(self._properties_dock, self._gallery_dock)
        self._properties_dock.raise_()  # Show properties by default
        
        # Console dock (bottom) - replaces History placeholder
        from ai_image_studio.ui.panels import ConsolePanel
        
        self._console_dock = QDockWidget("Console", self)
        self._console_dock.setObjectName("console_dock")
        self._console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self._console_panel = ConsolePanel()
        self._console_panel.cancel_requested.connect(self._on_cancel_execution)
        self._console_dock.setWidget(self._console_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._console_dock)
        
        # Add toggle actions to View menu
        self._view_menu.addAction(self._node_library_dock.toggleViewAction())
        self._view_menu.addAction(self._properties_dock.toggleViewAction())
        self._view_menu.addAction(self._layers_dock.toggleViewAction())
        self._view_menu.addAction(self._gallery_dock.toggleViewAction())
        self._view_menu.addAction(self._console_dock.toggleViewAction())

        self._refresh_layers_panel()

    def _refresh_layers_panel(self) -> None:
        self._layers_panel.set_layers(self._layer_stack.layers, self._selected_layer_index)

    def _on_layer_selected(self, index: int) -> None:
        self._selected_layer_index = index
        self._refresh_layers_panel()

    def _on_layer_visibility_changed(self, index: int, visible: bool) -> None:
        self._layer_stack.set_layer_visibility(index, visible)
        self._refresh_layers_panel()
        self._update_output_studio_from_layers()

    def _on_add_layer_requested(self) -> None:
        index = self._layer_stack.next_available_index()
        self._layer_stack.add_layer(index=index)
        self._selected_layer_index = index
        self._refresh_layers_panel()

    def _on_delete_layer_requested(self, index: int) -> None:
        self._layer_stack.remove_layer(index)
        if self._selected_layer_index == index:
            self._selected_layer_index = None
        self._refresh_layers_panel()
        self._update_output_studio_from_layers()
    
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
        
        # Load last session workspace
        self._load_session()
        
        # Load saved gallery images
        count = self._gallery_panel.load_saved_images()
        if count > 0:
            self._console_panel.log_info(f"Loaded {count} images from gallery")
        
        # Welcome message
        self._console_panel.log_info("Welcome to AI Image Studio!")
        self._console_panel.log_info("Add nodes from the Node Library, then press F5 to execute")
    
    def _load_session(self) -> None:
        """Load the last session workspace if it exists."""
        from ai_image_studio.core.workspace import load_workspace, get_last_session_path
        from ai_image_studio.core.graph import Node, Point2D
        from ai_image_studio.core.node_types import NodeRegistry
        from uuid import UUID
        
        session_path = get_last_session_path()
        if not session_path.exists():
            return
        
        try:
            data = load_workspace(session_path)
            registry = NodeRegistry.instance()
            
            # Restore nodes
            for node_data in data.get("nodes", []):
                visual_id = node_data["id"]
                type_id = node_data["type_id"]
                x = node_data["x"]
                y = node_data["y"]
                params = node_data.get("parameters", {})
                
                # Create core node with specific ID
                core_node = Node.create(type_id, position=Point2D(x=x, y=y))
                # Set saved parameters
                for key, value in params.items():
                    core_node.set_parameter(key, value)
                
                self._graph.add_node(core_node)
                self._node_id_map[visual_id] = core_node.id
                
                # Get node type for visual info
                node_type = registry.get(type_id)
                parts = type_id.split(".")
                category = parts[0] if parts else "utility"
                
                if node_type:
                    title = node_type.name
                    inputs = [(inp.name, inp.data_type.name) for inp in node_type.inputs]
                    outputs = [(out.name, out.data_type.name) for out in node_type.outputs]
                else:
                    title = parts[-1].replace("_", " ").title()
                    inputs, outputs = [], []
                
                # Add visual node
                self._node_graph_canvas.add_visual_node(
                    node_id=visual_id,
                    x=x, y=y,
                    title=title,
                    category=category,
                    inputs=inputs,
                    outputs=outputs,
                )
                
                # Set type_id in visual node
                if visual_id in self._node_graph_canvas._nodes:
                    self._node_graph_canvas._nodes[visual_id].type_id = type_id
                
                # Reload preview images from file path parameters
                for pname in ("file", "image_path", "input_image"):
                    if pname in params and params[pname]:
                        self._update_node_image_preview(visual_id, params[pname])
                        break
            
            # Restore connections
            for conn_data in data.get("connections", []):
                src = conn_data["source"]
                src_out = conn_data["source_output"]
                tgt = conn_data["target"]
                tgt_in = conn_data["target_input"]
                
                # Add visual connection
                self._node_graph_canvas.add_visual_connection(src, src_out, tgt, tgt_in)
                
                # Add core connection
                if src in self._node_id_map and tgt in self._node_id_map:
                    from ai_image_studio.core.graph import Connection
                    conn = Connection.create(
                        source_node=self._node_id_map[src],
                        source_output=src_out,
                        target_node=self._node_id_map[tgt],
                        target_input=tgt_in,
                    )
                    self._graph.add_connection(conn)
            
            node_count = len(data.get("nodes", []))
            if node_count > 0:
                self._console_panel.log_info(f"Restored {node_count} nodes from last session")
            
            # Restore viewport state (zoom/pan)
            viewport = data.get("viewport", {})
            if viewport:
                self._node_graph_canvas._transform.zoom = viewport.get("zoom", 1.0)
                self._node_graph_canvas._transform.offset_x = viewport.get("offset_x", 0.0)
                self._node_graph_canvas._transform.offset_y = viewport.get("offset_y", 0.0)
                self._node_graph_canvas.update()
                
        except Exception as e:
            print(f"Failed to load session: {e}")
    
    def _connect_canvas_signals(self) -> None:
        """Connect canvas signals for syncing with core model."""
        # Connection created - sync to NodeGraph
        self._node_graph_canvas.connection_created.connect(self._on_connection_created)
        
        # Connection removed - sync to NodeGraph
        self._node_graph_canvas.connection_removed.connect(self._on_connection_removed)
        
        # Node deleted - sync to NodeGraph
        self._node_graph_canvas.node_deleted.connect(self._on_node_deleted)
        
        # Node moved - sync position to core Node
        self._node_graph_canvas.node_moved.connect(self._on_node_moved)
    
    def _on_node_deleted(self, node_id: str) -> None:
        """Handle node deletion from canvas - sync to core NodeGraph."""
        if node_id in self._node_id_map:
            core_id = self._node_id_map[node_id]
            self._graph.remove_node(core_id)
            del self._node_id_map[node_id]
            self._console_panel.log_info(f"Deleted node: {node_id[:8]}...")
    
    def _on_connection_removed(
        self,
        src_node_id: str,
        src_output: str,
        tgt_node_id: str,
        tgt_input: str,
    ) -> None:
        """Handle connection removal from canvas - sync to core NodeGraph."""
        src_core_id = self._node_id_map.get(src_node_id)
        tgt_core_id = self._node_id_map.get(tgt_node_id)
        
        if src_core_id and tgt_core_id:
            # Find and remove the connection
            for conn in list(self._graph.connections):
                if (conn.source.node_id == src_core_id and 
                    conn.source.output_name == src_output and
                    conn.target.node_id == tgt_core_id and
                    conn.target.input_name == tgt_input):
                    self._graph.remove_connection(conn.id)
                    self._console_panel.log_info(f"Disconnected {src_output} → {tgt_input}")
                    break
    
    def _on_connection_created(
        self,
        src_node_id: str,
        src_output: str,
        tgt_node_id: str,
        tgt_input: str,
    ) -> None:
        """Handle new connection from canvas - sync to core NodeGraph."""
        from ai_image_studio.core.graph import Connection
        
        # Get core node IDs
        src_core_id = self._node_id_map.get(src_node_id)
        tgt_core_id = self._node_id_map.get(tgt_node_id)
        
        if not src_core_id or not tgt_core_id:
            self._console_panel.log_warning(f"Connection to unknown node ignored")
            return
        
        # Create core Connection
        conn = Connection.create(
            source_node=src_core_id,
            source_output=src_output,
            target_node=tgt_core_id,
            target_input=tgt_input,
        )
        
        if self._graph.add_connection(conn):
            # Also add visual connection
            self._node_graph_canvas.add_visual_connection(
                src_node_id, src_output, tgt_node_id, tgt_input
            )
            self._console_panel.log_info(
                f"Connected {src_output} → {tgt_input}"
            )
        else:
            self._console_panel.log_warning("Connection would create cycle - rejected")
    
    def _on_node_moved(self, node_id: str, x: float, y: float) -> None:
        """Handle node movement - sync position to core Node."""
        from ai_image_studio.core.graph import Point2D
        
        if node_id in self._node_id_map:
            core_id = self._node_id_map[node_id]
            core_node = self._graph.get_node(core_id)
            if core_node:
                core_node.position = Point2D(x=x, y=y)
    
    def closeEvent(self, event) -> None:
        """Save state before closing."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        
        # Auto-save workspace for next session
        self._save_session()
        
        super().closeEvent(event)
    
    def _save_session(self) -> None:
        """Save current workspace to last session file."""
        from ai_image_studio.core.workspace import save_workspace, get_last_session_path
        
        try:
            # Get current viewport state
            transform = self._node_graph_canvas._transform
            viewport_state = {
                "zoom": transform.zoom,
                "offset_x": transform.offset_x,
                "offset_y": transform.offset_y,
            }
            
            save_workspace(
                graph=self._graph,
                node_id_map=self._node_id_map,
                visual_nodes=self._node_graph_canvas._nodes,
                connections=self._node_graph_canvas._connections,
                viewport_state=viewport_state,
                path=get_last_session_path(),
                name="_last_session",
            )
        except Exception as e:
            print(f"Failed to save session: {e}")
    
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
        """Execute the current workflow using the ExecutionEngine."""
        import asyncio
        from PySide6.QtCore import QThreadPool, QRunnable, QObject, Signal
        import threading
        from ai_image_studio.core.execution import get_engine, ExecutionStatus, ExecutionProgress
        
        # Check if we have any nodes
        if not self._graph.nodes:
            self._console_panel.log_warning("No nodes in graph. Add nodes first.")
            return
        
        self._console_panel.log_info("=" * 40)
        self._console_panel.log_info("Starting workflow execution...")
        self._console_panel.set_job("Executing Workflow")
        self._console_panel.set_status("Preparing...")
        self._console_panel.set_progress(5)
        
        # Mark all nodes as dirty so they execute
        for node in self._graph.nodes.values():
            node.mark_dirty()
        
        # Set up thread worker
        class WorkerSignals(QObject):
            progress = Signal(object)  # ExecutionProgress
            finished = Signal(object)  # ExecutionJob
            error = Signal(str)
        
        class ExecutionWorker(QRunnable):
            def __init__(worker_self, graph, window, cancel_event: threading.Event):
                super().__init__()
                worker_self.graph = graph
                worker_self.window = window
                worker_self.signals = WorkerSignals()
                worker_self.cancel_event = cancel_event
            
            def run(worker_self):
                try:
                    asyncio.run(worker_self._run_engine())
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    worker_self.signals.error.emit(str(e))
            
            async def _run_engine(worker_self):
                from ai_image_studio.core.execution import ExecutionEngine, ExecutionJob
                import time
                
                # Create a fresh engine for this execution, wired to an external cancel event
                engine = ExecutionEngine(external_cancelled=worker_self.cancel_event.is_set)
                
                # Set up progress callback
                def on_progress(p: ExecutionProgress):
                    worker_self.signals.progress.emit(p)
                
                engine.set_progress_callback(on_progress)
                
                # Submit graph - this queues the job
                job_id = await engine.submit(worker_self.graph)
                
                # Run the worker manually to execute synchronously
                await engine._run_worker()
                
                # Find the completed job in results
                # The job is stored in the engine after execution
                # We need to get it before it's cleared
                
                # Check the job we just ran
                for node in worker_self.graph.nodes.values():
                    if node.cached_output:
                        # Job completed successfully, create result
                        results = {}
                        for nid, n in worker_self.graph.nodes.items():
                            if n.cached_output:
                                results[nid] = n.cached_output.data
                        
                        from ai_image_studio.core.execution import ExecutionStatus
                        completed_job = ExecutionJob(
                            id=job_id,
                            graph=worker_self.graph,
                            status=ExecutionStatus.COMPLETED,
                            results=results,
                        )
                        worker_self.signals.finished.emit(completed_job)
                        return
                
                # No cached outputs - check for errors
                for node in worker_self.graph.nodes.values():
                    if node.error:
                        from ai_image_studio.core.execution import ExecutionStatus
                        failed_job = ExecutionJob(
                            id=job_id,
                            graph=worker_self.graph,
                            status=ExecutionStatus.FAILED,
                            error=node.error.message if hasattr(node.error, 'message') else str(node.error),
                        )
                        worker_self.signals.finished.emit(failed_job)
                        return
                
                # Fallback - emit None
                worker_self.signals.finished.emit(None)
        
        # Create worker
        cancel_event = threading.Event()
        self._active_cancel_event = cancel_event

        worker = ExecutionWorker(self._graph, self, cancel_event)
        worker.signals.progress.connect(self._on_execution_progress)
        worker.signals.finished.connect(self._on_execution_finished)
        worker.signals.error.connect(self._on_generation_error)
        
        QThreadPool.globalInstance().start(worker)
    
    def _on_execution_progress(self, progress) -> None:
        """Handle per-node progress updates from ExecutionEngine."""
        from ai_image_studio.core.execution import ExecutionStatus

        # Live preview streaming (local sd.cpp)
        preview_image = getattr(progress, "preview_image", None)
        preview_step = getattr(progress, "preview_step", None)
        preview_total = getattr(progress, "preview_total_steps", None)

        if preview_total and preview_step is not None:
            # Override node-based progress with sampling-step progress for local generation.
            pct = int((preview_step / max(1, int(preview_total))) * 100)
            self._console_panel.set_busy(False)
            self._console_panel.set_progress(pct)

            # If we're done sampling but still running, we're likely decoding.
            if progress.status == ExecutionStatus.RUNNING and int(preview_step) >= int(preview_total):
                self._console_panel.set_busy(True)
                self._console_panel.set_status("Decoding (VAE)…")
            elif progress.status == ExecutionStatus.RUNNING:
                self._console_panel.set_status(f"Sampling ({preview_step}/{preview_total})")

        if preview_image is not None:
            self._output_studio.set_image_from_data(preview_image)
            # Don't spam logs for preview frames
            return
        
        # Log per-node progress
        if progress.current_node_name:
            self._console_panel.log_info(f"[{progress.current_node_name}] {progress.message}")
        elif progress.message:
            self._console_panel.log_info(progress.message)
        
        # Update progress bar
        if not (preview_total and preview_step is not None):
            self._console_panel.set_busy(False)
            self._console_panel.set_progress(int(progress.progress_percent))

        # Update status
        if progress.status == ExecutionStatus.RUNNING:
            node_progress = f"{progress.nodes_completed}/{progress.nodes_total}"
            self._console_panel.set_status(f"Executing ({node_progress})")
    
    def _on_execution_finished(self, job) -> None:
        """Handle workflow execution completion."""
        from ai_image_studio.core.execution import ExecutionStatus

        # Clear cancel hook
        if hasattr(self, "_active_cancel_event"):
            self._active_cancel_event = None
        
        if job is None:
            self._console_panel.log_warning("Execution completed with no job result")
            self._console_panel.clear_progress()
            return
        
        if job.status == ExecutionStatus.COMPLETED:
            self._console_panel.log_success(f"Workflow complete! Executed {len(job.results)} nodes.")
            self._console_panel.set_progress(100)
            self._console_panel.set_status("Complete")

            # Apply Preview node outputs into the layer stack.
            updated_layers = False
            for node_id, result in job.results.items():
                node = self._graph.get_node(node_id)
                if not node or node.type_id != "output.preview":
                    continue
                if not (isinstance(result, dict) and "image" in result):
                    continue
                image_data = result["image"]
                try:
                    layer_index = int(node.get_parameter("layer_index", 0))
                except Exception:
                    layer_index = 0
                layer_name = node.get_parameter("layer_name", "")
                self._layer_stack.set_layer_name(layer_index, (layer_name or "").strip() or None)
                self._layer_stack.set_layer_image(layer_index, image_data)
                updated_layers = True

            if updated_layers:
                self._refresh_layers_panel()
                self._update_output_studio_from_layers()

            # Decide what to add to the gallery (composite when available).
            gallery_image = self._layer_stack.get_composite() if updated_layers else None
            gallery_result: dict | None = None

            if gallery_image is None:
                for _, result in job.results.items():
                    if isinstance(result, dict) and "image" in result:
                        gallery_image = result["image"]
                        gallery_result = result
                        break

            if gallery_image is not None:
                prompt = ""
                model_id = ""
                if gallery_result:
                    prompt = gallery_result.get("revised_prompt", "") or ""
                    model_id = gallery_result.get("model_id", "") or ""
                if not prompt:
                    for node_id, result in job.results.items():
                        if isinstance(result, dict) and "image" in result:
                            node = self._graph.get_node(node_id)
                            prompt = node.get_parameter("prompt", "") if node else ""
                            break

                workflow_metadata = self._get_current_workflow_metadata()
                self._gallery_panel.add_image(
                    gallery_image,
                    prompt=prompt,
                    model=model_id,
                    auto_save=True,
                    workflow_metadata=workflow_metadata,
                )
                self._console_panel.log_info("Image added to Gallery")
            
            self.statusBar().showMessage("Workflow complete!", 5000)
            
        elif job.status == ExecutionStatus.FAILED:
            self._console_panel.log_error(f"Workflow failed: {job.error}")
            self.statusBar().showMessage("Workflow failed", 5000)
            
        elif job.status == ExecutionStatus.CANCELLED:
            self._console_panel.log_warning("Workflow cancelled")
            self.statusBar().showMessage("Workflow cancelled", 3000)
        
        self._console_panel.clear_progress()

    def _update_output_studio_from_layers(self) -> None:
        composite = self._layer_stack.get_composite()
        self._output_studio.set_image_from_data(composite)
    
    def _on_generation_error(self, error: str) -> None:
        """Handle execution error."""
        self._console_panel.log_error(f"Execution failed: {error}")
        self._console_panel.clear_progress()
        self.statusBar().showMessage("Execution failed", 5000)
    
    def _on_cancel_execution(self) -> None:
        """Cancel the current execution."""
        cancel_event = getattr(self, "_active_cancel_event", None)
        if cancel_event is None:
            self._console_panel.log_warning("No active job to cancel.")
            return
        cancel_event.set()
        self._console_panel.log_warning("Cancellation requested...")
        self.statusBar().showMessage("Cancelling...", 2000)
    
    def _on_gallery_load(self, item_id: str) -> None:
        """Load an image from gallery into Output Studio."""
        item = self._gallery_panel.get_item(item_id)
        if not item:
            return

        image_data = self._image_data_from_gallery_item(item)
        if image_data is None:
            self._console_panel.log_warning("Image not available")
            return

        # Layer-aware behavior: load into selected layer if set; otherwise avoid
        # overwriting base layer if it already has content.
        target_index = self._selected_layer_index
        if target_index is None:
            base = self._layer_stack.get_layer(0)
            target_index = 0 if (base is None or not base.has_image) else self._layer_stack.next_available_index()

        self._layer_stack.set_layer_image(target_index, image_data)
        self._selected_layer_index = target_index
        self._refresh_layers_panel()
        self._update_output_studio_from_layers()

        self._console_panel.log_info(f"Loaded image from gallery into layer {target_index}")
        self.statusBar().showMessage(f"Loaded into layer {target_index}", 2000)

    def _image_data_from_gallery_item(self, item):
        """Normalize a GalleryItem image (ImageData/QImage) into ImageData."""
        if getattr(item, "image_data", None) is not None:
            return item.image_data

        qimage = getattr(item, "qimage", None)
        if qimage is None:
            return None

        try:
            from ai_image_studio.core.data_types import ImageData
            import numpy as np

            qimg = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
            width = qimg.width()
            height = qimg.height()
            bytes_per_line = qimg.bytesPerLine()

            ptr = qimg.bits()
            ptr.setsize(qimg.sizeInBytes())
            buf = np.frombuffer(ptr, dtype=np.uint8)
            buf = buf.reshape((height, bytes_per_line))
            buf = buf[:, : width * 4]
            rgba = buf.reshape((height, width, 4))
            return ImageData.from_numpy(rgba)
        except Exception:
            return None
    
    def _on_gallery_preview(self, item_id: str) -> None:
        """Display image in Output Studio on single-click (without loading workflow)."""
        item = self._gallery_panel.get_item(item_id)
        if not item:
            return
        
        # Show image in preview
        if item.image_data:
            self._output_studio.set_image_from_data(item.image_data)
        elif item.qimage:
            self._output_studio.set_image(item.qimage)
    
    def _on_workflow_load(self, item_id: str) -> None:
        """Load workflow from gallery item - restores full node graph."""
        item = self._gallery_panel.get_item(item_id)
        if not item:
            return
        
        # First display the image
        if item.image_data:
            self._output_studio.set_image_from_data(item.image_data)
        elif item.qimage:
            self._output_studio.set_image(item.qimage)
        
        # Check if workflow metadata exists
        if not item.workflow_path or not item.workflow_path.exists():
            self._console_panel.log_warning("No workflow data found for this image")
            return
        
        # Load the workflow metadata
        from ai_image_studio.core.workflow_metadata import load_workflow_metadata
        try:
            metadata = load_workflow_metadata(item.saved_path)
            if not metadata:
                self._console_panel.log_warning("Could not load workflow metadata")
                return
        except Exception as e:
            self._console_panel.log_error(f"Failed to load workflow: {e}")
            return
        
        # Clear current graph and canvas
        self._graph.clear()
        self._node_id_map.clear()
        self._node_graph_canvas._nodes.clear()
        self._node_graph_canvas._connections.clear()
        self._node_graph_canvas.update()
        
        # Restore nodes and connections (similar to _load_session)
        from ai_image_studio.core.graph import Node, Point2D, Connection
        from ai_image_studio.core.node_types import NodeRegistry
        
        registry = NodeRegistry.instance()
        
        for node_data in metadata.nodes:
            visual_id = node_data["id"]
            type_id = node_data["type_id"]
            x = node_data["x"]
            y = node_data["y"]
            params = node_data.get("parameters", {})
            
            # Create core node
            core_node = Node.create(type_id, position=Point2D(x=x, y=y))
            for key, value in params.items():
                core_node.set_parameter(key, value)
            
            self._graph.add_node(core_node)
            self._node_id_map[visual_id] = core_node.id
            
            # Get node type for visual info
            node_type = registry.get(type_id)
            parts = type_id.split(".")
            category = parts[0] if parts else "utility"
            
            if node_type:
                title = node_type.name
                inputs = [(inp.name, inp.data_type.name) for inp in node_type.inputs]
                outputs = [(out.name, out.data_type.name) for out in node_type.outputs]
            else:
                title = parts[-1].replace("_", " ").title()
                inputs, outputs = [], []
            
            # Add visual node
            self._node_graph_canvas.add_visual_node(
                node_id=visual_id,
                x=x, y=y,
                title=title,
                category=category,
                inputs=inputs,
                outputs=outputs,
            )
            
            if visual_id in self._node_graph_canvas._nodes:
                self._node_graph_canvas._nodes[visual_id].type_id = type_id
            
            # Reload preview images
            for pname in ("file", "image_path", "input_image"):
                if pname in params and params[pname]:
                    self._update_node_image_preview(visual_id, params[pname])
                    break
        
        # Restore connections
        for conn_data in metadata.connections:
            src = conn_data["source"]
            src_out = conn_data["source_output"]
            tgt = conn_data["target"]
            tgt_in = conn_data["target_input"]
            
            self._node_graph_canvas.add_visual_connection(src, src_out, tgt, tgt_in)
            
            if src in self._node_id_map and tgt in self._node_id_map:
                conn = Connection.create(
                    source_node=self._node_id_map[src],
                    source_output=src_out,
                    target_node=self._node_id_map[tgt],
                    target_input=tgt_in,
                )
                self._graph.add_connection(conn)
        
        # Restore viewport
        viewport = metadata.viewport
        if viewport:
            self._node_graph_canvas._transform.zoom = viewport.get("zoom", 1.0)
            self._node_graph_canvas._transform.offset_x = viewport.get("offset_x", 0.0)
            self._node_graph_canvas._transform.offset_y = viewport.get("offset_y", 0.0)
        
        self._node_graph_canvas.update()
        
        node_count = len(metadata.nodes)
        self._console_panel.log_success(f"Loaded workflow with {node_count} nodes")
        self.statusBar().showMessage("Workflow loaded from gallery", 3000)
    
    def _get_current_workflow_metadata(self):
        """
        Capture current workflow state for saving alongside gallery images.
        
        Returns:
            WorkflowMetadata with current graph state
        """
        from ai_image_studio.core.workflow_metadata import WorkflowMetadata
        
        # Serialize nodes
        nodes_data = []
        for visual_id, core_uuid in self._node_id_map.items():
            core_node = self._graph.get_node(core_uuid)
            visual = self._node_graph_canvas._nodes.get(visual_id)
            
            if core_node and visual:
                nodes_data.append({
                    "id": visual_id,
                    "type_id": core_node.type_id,
                    "x": visual.x,
                    "y": visual.y,
                    "parameters": core_node.parameters,
                })
        
        # Serialize connections
        connections_data = [
            {
                "source": src,
                "source_output": src_out,
                "target": tgt,
                "target_input": tgt_in,
            }
            for src, src_out, tgt, tgt_in in self._node_graph_canvas._connections
        ]
        
        # Get viewport state
        transform = self._node_graph_canvas._transform
        viewport_state = {
            "zoom": transform.zoom,
            "offset_x": transform.offset_x,
            "offset_y": transform.offset_y,
        }
        
        return WorkflowMetadata(
            nodes=nodes_data,
            connections=connections_data,
            viewport=viewport_state,
        )
    
    def _on_manage_providers(self) -> None:
        """Open the provider management dialog."""
        from ai_image_studio.ui.dialogs import ProviderSettingsDialog
        
        dialog = ProviderSettingsDialog(self)
        dialog.settings_changed.connect(self._on_provider_settings_changed)
        dialog.exec()
    
    def _on_provider_settings_changed(self) -> None:
        """Handle provider settings changes."""
        self.statusBar().showMessage("Provider settings saved", 3000)
    
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
    
    def _on_add_node_requested(self, type_id: str) -> None:
        """Handle request to add a node from the library."""
        import random
        from ai_image_studio.core.graph import Node, Point2D
        from ai_image_studio.core.node_types import NodeRegistry
        
        # Get node type definition
        registry = NodeRegistry.instance()
        node_type = registry.get(type_id)
        
        # Calculate position (center of view with offset)
        x = 200 + random.randint(-50, 50)
        y = 150 + random.randint(-50, 50)
        
        # Create core Node
        core_node = Node.create(type_id, position=Point2D(x=x, y=y))
        self._graph.add_node(core_node)
        
        # Map string ID to UUID
        visual_id = str(core_node.id)
        self._node_id_map[visual_id] = core_node.id
        
        # Parse type_id for display
        parts = type_id.split(".")
        category = parts[0] if parts else "utility"
        
        # Get title and IO from node type if available
        if node_type:
            title = node_type.name
            inputs = [(inp.name, inp.data_type.name) for inp in node_type.inputs]
            outputs = [(out.name, out.data_type.name) for out in node_type.outputs]
        else:
            # Fallback for unknown types
            title = parts[-1].replace("_", " ").title() if parts else type_id
            
            # Default definitions for common types
            fallback_defs = {
                "input.prompt": ([], [("text", "TEXT")]),
                "input.image": ([], [("image", "IMAGE")]),
                "input.mask": ([], [("mask", "MASK")]),
                "generation.text_to_image": (
                    [("prompt", "TEXT"), ("negative", "TEXT")],
                    [("image", "IMAGE")],
                ),
                "generation.image_to_image": (
                    [("image", "IMAGE"), ("prompt", "TEXT")],
                    [("image", "IMAGE")],
                ),
                "generation.inpaint": (
                    [("image", "IMAGE"), ("mask", "MASK"), ("prompt", "TEXT")],
                    [("image", "IMAGE")],
                ),
                "enhancement.upscale": (
                    [("image", "IMAGE")],
                    [("image", "IMAGE")],
                ),
                "output.preview": ([("image", "IMAGE")], []),
                "output.save": ([("image", "IMAGE")], []),
            }
            inputs, outputs = fallback_defs.get(type_id, ([], []))
        
        # Add visual node with same ID
        self._node_graph_canvas.add_visual_node(
            node_id=visual_id,
            x=x,
            y=y,
            title=title,
            category=category,
            inputs=inputs,
            outputs=outputs,
        )
        
        # Store type_id in visual node for later lookup
        if visual_id in self._node_graph_canvas._nodes:
            self._node_graph_canvas._nodes[visual_id].type_id = type_id

        # Preview nodes define layers
        if type_id == "output.preview":
            layer_index = self._next_available_preview_layer_index()
            core_node.set_parameter("layer_index", layer_index)
            core_node.set_parameter("layer_name", "")
            self._sync_layer_from_preview_node(core_node)
            self._selected_layer_index = layer_index
            self._refresh_layers_panel()
            self._update_preview_node_badge(visual_id)

        self._console_panel.log_info(f"Added node: {title}")
        self.statusBar().showMessage(f"Added {title} node", 2000)

    def _next_available_preview_layer_index(self) -> int:
        used: set[int] = set()
        for node in self._graph.nodes.values():
            if node.type_id != "output.preview":
                continue
            try:
                idx = int(node.get_parameter("layer_index", -1))
            except Exception:
                continue
            if idx >= 0:
                used.add(idx)

        idx = 0
        while idx in used:
            idx += 1
        return idx

    def _sync_layer_from_preview_node(self, core_node) -> None:
        try:
            layer_index = int(core_node.get_parameter("layer_index", 0))
        except Exception:
            layer_index = 0
        layer_name = core_node.get_parameter("layer_name", "")
        self._layer_stack.set_layer_name(layer_index, (layer_name or "").strip() or None)

    def _update_preview_node_badge(self, visual_id: str) -> None:
        if visual_id not in self._node_id_map:
            return
        core_id = self._node_id_map[visual_id]
        core_node = self._graph.get_node(core_id)
        if core_node is None or core_node.type_id != "output.preview":
            return
        try:
            layer_index = int(core_node.get_parameter("layer_index", 0))
        except Exception:
            layer_index = 0
        layer_name = (core_node.get_parameter("layer_name", "") or "").strip()
        badge = f"L{layer_index}" if not layer_name else f"L{layer_index} {layer_name}"

        visual = self._node_graph_canvas._nodes.get(visual_id)
        if visual is None:
            return
        visual.badge_text = badge
        self._node_graph_canvas.update()

    def _on_parameter_changed(self, node_id: str, param_name: str, value) -> None:
        """Handle parameter change from properties panel."""
        # Sync to core Node model
        if node_id in self._node_id_map:
            core_id = self._node_id_map[node_id]
            core_node = self._graph.get_node(core_id)
            if core_node:
                core_node.set_parameter(param_name, value)

                if core_node.type_id == "output.preview" and param_name in ("layer_index", "layer_name"):
                    self._sync_layer_from_preview_node(core_node)
                    self._refresh_layers_panel()
                    self._update_preview_node_badge(node_id)
                    self._update_output_studio_from_layers()
                
                # Check if this is a file path parameter - load preview
                if param_name in ("file", "image_path", "input_image") and value:
                    self._update_node_image_preview(node_id, value)
        
        self.statusBar().showMessage(f"{param_name} = {value}", 1500)
    
    def _update_node_image_preview(self, node_id: str, file_path: str) -> None:
        """Load image from path and update node's preview thumbnail."""
        from pathlib import Path
        try:
            from PIL import Image
            
            if Path(file_path).exists():
                # Load and create thumbnail
                img = Image.open(file_path)
                img.thumbnail((120, 120))  # Limit size for efficiency
                
                # Update canvas preview
                self._node_graph_canvas.set_node_preview(node_id, img)
                self._console_panel.log_info(f"Loaded preview: {Path(file_path).name}")
        except Exception as e:
            self._console_panel.log_warning(f"Could not load preview: {e}")
    
    def _on_node_selected(self, node_id) -> None:
        """Handle node selection from canvas."""
        if node_id:
            self.statusBar().showMessage(f"Selected: {node_id}", 2000)
            
            # Get node info from canvas
            if node_id in self._node_graph_canvas._nodes:
                visual_node = self._node_graph_canvas._nodes[node_id]
                
                # Try to get actual NodeType from registry
                from ai_image_studio.core.node_types import NodeRegistry
                
                # Map visual node to a type_id
                type_id = self._get_type_id_for_node(visual_node)
                node_type = NodeRegistry.instance().get(type_id) if type_id else None
                
                if node_type and node_type.parameters:
                    # Get saved parameters from core Node (if exists)
                    saved_params = {}
                    if node_id in self._node_id_map:
                        core_id = self._node_id_map[node_id]
                        core_node = self._graph.get_node(core_id)
                        if core_node:
                            saved_params = core_node.parameters.copy()
                    
                    # Merge defaults with saved values (saved takes precedence)
                    params = node_type.get_default_parameters()
                    params.update(saved_params)
                    
                    self._properties_panel.set_node(
                        node_id=node_id,
                        title=visual_node.title,
                        parameters=params,
                        definitions=node_type.parameters,
                    )
                else:
                    # Fall back to demo properties
                    props = self._get_default_properties(visual_node.category, visual_node.title)
                    self._properties_panel.set_simple_properties(
                        node_id=node_id,
                        title=visual_node.title,
                        properties=props,
                    )
        else:
            self._properties_panel.set_node(None)
    
    def _get_type_id_for_node(self, visual_node) -> str | None:
        """Map a visual node to its NodeType ID."""
        # The type_id is stored directly on the visual node
        if visual_node.type_id:
            return visual_node.type_id
        
        # Legacy fallback - map by title (for old sessions without type_id)
        title_lower = visual_node.title.lower()
        
        if "text to image" in title_lower:
            return "generation.text_to_image"
        elif "image to image" in title_lower:
            return "generation.image_to_image"
        elif "prompt" in title_lower:
            return "input.prompt"
        elif "load image" in title_lower or "image input" in title_lower:
            return "input.image"
        elif "preview" in title_lower:
            return "output.preview"
        elif "save" in title_lower:
            return "output.save_image"
        
        return None
    
    def _get_default_properties(self, category: str, title: str) -> dict:
        """Get default properties for a node based on its category."""
        if category == "input":
            if "prompt" in title.lower():
                return {
                    "prompt": ("multiline", "A beautiful landscape"),
                    "negative_prompt": ("text", ""),
                }
            elif "image" in title.lower():
                return {
                    "file_path": ("text", ""),
                }
        elif category == "generation":
            return {
                "width": ("int", 1024),
                "height": ("int", 1024),
                "steps": ("int", 30),
                "cfg_scale": ("float", 7.5),
                "sampler": ("choice:euler,euler_a,dpm++2m,ddim", "euler_a"),
                "seed": ("seed", -1),
            }
        elif category == "enhancement":
            return {
                "scale": ("choice:2x,4x", "2x"),
                "model": ("choice:RealESRGAN,ESRGAN", "RealESRGAN"),
            }
        elif category == "output":
            if "save" in title.lower():
                return {
                    "output_path": ("text", ""),
                    "format": ("choice:png,jpg,webp", "png"),
                    "quality": ("int", 95),
                }
        
        return {}
