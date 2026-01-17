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
        
        # Output Studio (real widget)
        from ai_image_studio.ui.output_studio import OutputStudio
        self._output_studio = OutputStudio()
        
        # Add to splitter
        self._main_splitter.addWidget(self._node_graph_canvas)
        self._main_splitter.addWidget(self._output_studio)
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
        from ai_image_studio.ui.panels import NodeLibraryPanel, PropertiesPanel
        
        # Node Library dock (left)
        self._node_library_dock = QDockWidget("Node Library", self)
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
        self._properties_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._properties_panel = PropertiesPanel()
        self._properties_panel.parameter_changed.connect(self._on_parameter_changed)
        self._properties_panel.setMinimumWidth(250)
        self._properties_dock.setWidget(self._properties_panel)
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
        
        # Console dock (bottom) - replaces History placeholder
        from ai_image_studio.ui.panels import ConsolePanel
        
        self._console_dock = QDockWidget("Console", self)
        self._console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self._console_panel = ConsolePanel()
        self._console_panel.cancel_requested.connect(self._on_cancel_execution)
        self._console_dock.setWidget(self._console_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._console_dock)
        
        # Queue dock (bottom)
        self._queue_dock = QDockWidget("Queue", self)
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
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        self._console_panel.log_info("Starting workflow execution...")
        
        # Check if we have any generation nodes selected or use first generation node
        nodes = list(self._node_graph_canvas._nodes.values())
        gen_nodes = [n for n in nodes if "text to image" in n.title.lower()]
        
        if not gen_nodes:
            self._console_panel.log_warning("No Text to Image nodes found. Add one to generate images.")
            return
        
        # Get parameters from first generation node
        node = gen_nodes[0]
        type_id = self._get_type_id_for_node(node)
        
        from ai_image_studio.core.node_types import NodeRegistry
        node_type = NodeRegistry.instance().get(type_id) if type_id else None
        
        if not node_type:
            self._console_panel.log_error(f"Node type not found: {type_id}")
            return
        
        # Get default parameters (in real impl, would get actual node params)
        params = node_type.get_default_parameters()
        
        self._console_panel.set_job(f"Generating: {node.title}")
        self._console_panel.set_status("Preparing request...")
        self._console_panel.set_progress(10)
        
        # Run async generation in thread pool
        def run_generation():
            return asyncio.run(self._execute_generation(params))
        
        from PySide6.QtCore import QThreadPool, QRunnable, QObject, Signal
        
        class WorkerSignals(QObject):
            finished = Signal(object)
            error = Signal(str)
            progress = Signal(int, str)
        
        class GenerationWorker(QRunnable):
            def __init__(self, params, window):
                super().__init__()
                self.params = params
                self.window = window
                self.signals = WorkerSignals()
            
            def run(self):
                try:
                    result = asyncio.run(self.window._execute_generation(self.params))
                    self.signals.finished.emit(result)
                except Exception as e:
                    self.signals.error.emit(str(e))
        
        worker = GenerationWorker(params, self)
        worker.signals.finished.connect(self._on_generation_finished)
        worker.signals.error.connect(self._on_generation_error)
        
        QThreadPool.globalInstance().start(worker)
    
    async def _execute_generation(self, params: dict):
        """Execute the actual generation."""
        from ai_image_studio.providers import get_registry, GenerationRequest
        
        registry = get_registry()
        
        model_id = params.get("model", "dall-e-3")
        model = registry.get_model(model_id)
        
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        provider = registry.get_provider(model.provider)
        if not provider or not provider.is_configured:
            raise ValueError(f"Provider {model.provider} not configured. Go to Providers → Manage Providers.")
        
        prompt = params.get("prompt", "A beautiful landscape")
        
        request = GenerationRequest(
            model=model,
            prompt=prompt,
            width=params.get("width", 1024),
            height=params.get("height", 1024),
        )
        
        result = await provider.generate(request)
        return result
    
    def _on_generation_finished(self, result) -> None:
        """Handle successful generation."""
        self._console_panel.log_success(f"Generation complete! {len(result.images)} image(s) generated.")
        self._console_panel.set_progress(100)
        self._console_panel.set_status("Complete")
        
        if result.revised_prompt:
            self._console_panel.log_info(f"Revised prompt: {result.revised_prompt}")
        
        # Display image in output studio
        if result.images:
            self._output_studio.set_image_from_data(result.images[0])
            self._console_panel.log_info("Image displayed in Output Studio")
        
        self._console_panel.clear_progress()
        self.statusBar().showMessage("Generation complete!", 5000)
    
    def _on_generation_error(self, error: str) -> None:
        """Handle generation error."""
        self._console_panel.log_error(f"Generation failed: {error}")
        self._console_panel.clear_progress()
        self.statusBar().showMessage("Generation failed", 5000)
    
    def _on_cancel_execution(self) -> None:
        """Cancel the current execution."""
        self._console_panel.log_warning("Execution cancelled by user")
        self._console_panel.clear_progress()
        self.statusBar().showMessage("Execution cancelled", 2000)
    
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
        
        # Generate unique ID
        node_id = f"node_{random.randint(1000, 9999)}"
        
        # Parse type_id to get category and title
        parts = type_id.split(".")
        category = parts[0] if parts else "utility"
        title = parts[-1].replace("_", " ").title() if parts else type_id
        
        # Define inputs/outputs based on type
        node_defs = {
            "input.prompt": ([], [("text", "TEXT")]),
            "input.image": ([], [("image", "IMAGE")]),
            "input.mask": ([], [("mask", "MASK")]),
            "generation.text_to_image": (
                [("prompt", "TEXT"), ("negative", "TEXT"), ("model", "MODEL")],
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
        
        inputs, outputs = node_defs.get(type_id, ([], []))
        
        # Add at center of view (offset for visibility)
        self._node_graph_canvas.add_visual_node(
            node_id=node_id,
            x=200 + random.randint(-50, 50),
            y=150 + random.randint(-50, 50),
            title=title,
            category=category,
            inputs=inputs,
            outputs=outputs,
        )
        
        self.statusBar().showMessage(f"Added {title} node", 2000)
    
    def _on_parameter_changed(self, node_id: str, param_name: str, value) -> None:
        """Handle parameter change from properties panel."""
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
