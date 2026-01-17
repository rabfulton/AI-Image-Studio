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
        
        # Initialize core model (NodeGraph mirrors visual canvas)
        from ai_image_studio.core.graph import NodeGraph
        self._graph = NodeGraph("Workspace")
        
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
        
        # Output nodes
        out_menu = add_menu.addMenu("Output")
        out_menu.addAction("Preview")
        out_menu.addAction("Save Image")
        
        menu.addSeparator()
        menu.addAction("Frame All (F)")
        
        menu.exec(QCursor.pos())
    
    def _setup_dock_widgets(self) -> None:
        """Create dock widgets for panels."""
        from ai_image_studio.ui.panels import NodeLibraryPanel, PropertiesPanel
        
        # Node Library dock (left)
        self._node_library_dock = QDockWidget("Node Library", self)
        self._node_library_dock.setObjectName("node_library_dock")
        self._node_library_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._node_library_panel = NodeLibraryPanel()
        self._node_library_panel.node_requested.connect(self._on_add_node_requested)
        self._node_library_panel.setMinimumWidth(200)
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
        self._gallery_dock.setWidget(self._gallery_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._gallery_dock)
        
        # Stack gallery below properties
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
        
        # Queue dock (bottom)
        self._queue_dock = QDockWidget("Queue", self)
        self._queue_dock.setObjectName("queue_dock")
        self._queue_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        queue_placeholder = QLabel("Queue (Pending jobs)")
        queue_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._queue_dock.setWidget(queue_placeholder)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._queue_dock)
        
        # Tab console and queue together
        self.tabifyDockWidget(self._console_dock, self._queue_dock)
        self._console_dock.raise_()
        
        # Add toggle actions to View menu
        self._view_menu.addAction(self._node_library_dock.toggleViewAction())
        self._view_menu.addAction(self._properties_dock.toggleViewAction())
        self._view_menu.addAction(self._gallery_dock.toggleViewAction())
        self._view_menu.addAction(self._console_dock.toggleViewAction())
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
        
        # Load saved gallery images
        count = self._gallery_panel.load_saved_images()
        if count > 0:
            self._console_panel.log_info(f"Loaded {count} images from gallery")
        
        # Welcome message
        self._console_panel.log_info("Welcome to AI Image Studio!")
        self._console_panel.log_info("Add nodes from the Node Library, then press F5 to execute")
    
    def _connect_canvas_signals(self) -> None:
        """Connect canvas signals for syncing with core model."""
        # Connection created - sync to NodeGraph
        self._node_graph_canvas.connection_created.connect(self._on_connection_created)
        
        # Node moved - sync position to core Node
        self._node_graph_canvas.node_moved.connect(self._on_node_moved)
    
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
        """Execute the current workflow using the ExecutionEngine."""
        import asyncio
        from PySide6.QtCore import QThreadPool, QRunnable, QObject, Signal
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
            def __init__(worker_self, graph, window):
                super().__init__()
                worker_self.graph = graph
                worker_self.window = window
                worker_self.signals = WorkerSignals()
            
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
                
                # Create a fresh engine for this execution
                engine = ExecutionEngine()
                
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
        worker = ExecutionWorker(self._graph, self)
        worker.signals.progress.connect(self._on_execution_progress)
        worker.signals.finished.connect(self._on_execution_finished)
        worker.signals.error.connect(self._on_generation_error)
        
        QThreadPool.globalInstance().start(worker)
    
    def _on_execution_progress(self, progress) -> None:
        """Handle per-node progress updates from ExecutionEngine."""
        from ai_image_studio.core.execution import ExecutionStatus
        
        # Log per-node progress
        if progress.current_node_name:
            self._console_panel.log_info(f"[{progress.current_node_name}] {progress.message}")
        elif progress.message:
            self._console_panel.log_info(progress.message)
        
        # Update progress bar
        self._console_panel.set_progress(int(progress.progress_percent))
        
        # Update status
        if progress.status == ExecutionStatus.RUNNING:
            node_progress = f"{progress.nodes_completed}/{progress.nodes_total}"
            self._console_panel.set_status(f"Executing ({node_progress})")
    
    def _on_execution_finished(self, job) -> None:
        """Handle workflow execution completion."""
        from ai_image_studio.core.execution import ExecutionStatus
        
        if job is None:
            self._console_panel.log_warning("Execution completed with no job result")
            self._console_panel.clear_progress()
            return
        
        if job.status == ExecutionStatus.COMPLETED:
            self._console_panel.log_success(f"Workflow complete! Executed {len(job.results)} nodes.")
            self._console_panel.set_progress(100)
            self._console_panel.set_status("Complete")
            
            # Find image outputs and display them
            for node_id, result in job.results.items():
                if isinstance(result, dict) and "image" in result:
                    image_data = result["image"]
                    
                    # Show in Output Studio
                    self._output_studio.set_image_from_data(image_data)
                    
                    # Get prompt from result or node
                    prompt = result.get("revised_prompt", "")
                    if not prompt:
                        node = self._graph.get_node(node_id)
                        prompt = node.get_parameter("prompt", "") if node else ""
                    
                    # Add to gallery
                    self._gallery_panel.add_image(
                        image_data,
                        prompt=prompt,
                        model=result.get("model_id", ""),
                        auto_save=True,
                    )
                    
                    self._console_panel.log_info("Image added to Gallery")
                    break  # Display first image found
            
            self.statusBar().showMessage("Workflow complete!", 5000)
            
        elif job.status == ExecutionStatus.FAILED:
            self._console_panel.log_error(f"Workflow failed: {job.error}")
            self.statusBar().showMessage("Workflow failed", 5000)
            
        elif job.status == ExecutionStatus.CANCELLED:
            self._console_panel.log_warning("Workflow cancelled")
            self.statusBar().showMessage("Workflow cancelled", 3000)
        
        self._console_panel.clear_progress()
    
    def _on_generation_error(self, error: str) -> None:
        """Handle execution error."""
        self._console_panel.log_error(f"Execution failed: {error}")
        self._console_panel.clear_progress()
        self.statusBar().showMessage("Execution failed", 5000)
    
    def _on_cancel_execution(self) -> None:
        """Cancel the current execution."""
        import asyncio
        from ai_image_studio.core.execution import get_engine
        
        asyncio.create_task(get_engine().cancel())
        self._console_panel.log_warning("Cancellation requested...")
        self.statusBar().showMessage("Cancelling...", 2000)
    
    def _on_gallery_load(self, item_id: str) -> None:
        """Load an image from gallery into Output Studio."""
        item = self._gallery_panel.get_item(item_id)
        if not item:
            return
        
        # Items from current session have image_data, loaded from disk have qimage
        if item.image_data:
            self._output_studio.set_image_from_data(item.image_data)
        elif item.qimage:
            self._output_studio.set_image(item.qimage)
        else:
            self._console_panel.log_warning("Image not available")
            return
        
        self._console_panel.log_info(f"Loaded image from gallery")
        self.statusBar().showMessage("Image loaded from gallery", 2000)
    
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
        
        self._console_panel.log_info(f"Added node: {title}")
        self.statusBar().showMessage(f"Added {title} node", 2000)
    
    def _on_parameter_changed(self, node_id: str, param_name: str, value) -> None:
        """Handle parameter change from properties panel."""
        # Sync to core Node model
        if node_id in self._node_id_map:
            core_id = self._node_id_map[node_id]
            core_node = self._graph.get_node(core_id)
            if core_node:
                core_node.set_parameter(param_name, value)
        
        self.statusBar().showMessage(f"{param_name} = {value}", 1500)
    
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
                    # Use actual parameter definitions
                    self._properties_panel.set_node(
                        node_id=node_id,
                        title=visual_node.title,
                        parameters=node_type.get_default_parameters(),
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
        # Map by title/category to type_id
        title_lower = visual_node.title.lower()
        
        if "text to image" in title_lower:
            return "generation.text_to_image"
        elif "image to image" in title_lower:
            return "generation.image_to_image"
        
        # Could add more mappings here
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
