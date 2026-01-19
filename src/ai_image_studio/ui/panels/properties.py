"""
Properties Panel - Node parameter editor.

This panel displays and allows editing of the selected node's
parameters, with appropriate widgets for each parameter type.

Reference: features.md#14-properties-panel
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QScrollArea,
    QLabel,
    QLineEdit,
    QTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QSlider,
    QPushButton,
    QColorDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

if TYPE_CHECKING:
    from ai_image_studio.core.node_types import ParameterDefinition, ParameterType


class PropertiesPanel(QWidget):
    """
    Panel for editing node parameters.
    
    Signals:
        parameter_changed: Emitted when a parameter value changes (node_id, param_name, value)
    """
    
    parameter_changed = Signal(str, str, object)  # node_id, param_name, value
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        self._current_node_id: str | None = None
        self._widgets: dict[str, QWidget] = {}
        self._param_containers: dict[str, QWidget] = {}  # Container widgets for visibility control
        self._model_param_widgets: dict[str, QWidget] = {}  # Dynamic model params
        self._filter_param_widgets: dict[str, QWidget] = {}  # Dynamic G'MIC filter params
        self._pending_model_id: str | None = None
        self._pending_filter_id: str | None = None  # For G'MIC filters
        self._current_model_id: str | None = None  # Track current model for visibility updates
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border-bottom: 1px solid #313244;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 8, 8, 8)
        
        self._title_label = QLabel("Properties")
        self._title_label.setStyleSheet("font-weight: bold; color: #cdd6f4;")
        header_layout.addWidget(self._title_label)
        
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: #1e1e2e;
                border: none;
            }
        """)
        
        self._params_widget = QWidget()
        self._params_layout = QVBoxLayout(self._params_widget)
        self._params_layout.setContentsMargins(8, 8, 8, 8)
        self._params_layout.setSpacing(12)
        self._params_layout.addStretch()
        
        scroll.setWidget(self._params_widget)
        layout.addWidget(scroll)
        
        # Empty state
        self._empty_label = QLabel("Select a node to view its properties")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #6c7086;")
        self._params_layout.insertWidget(0, self._empty_label)
    
    def set_node(
        self,
        node_id: str | None,
        title: str = "",
        parameters: dict[str, Any] | None = None,
        definitions: list | None = None,
    ) -> None:
        """
        Set the node to display properties for.
        
        Args:
            node_id: The node's ID, or None to clear
            title: Display title for the node
            parameters: Current parameter values
            definitions: List of ParameterDefinition objects
        """
        self._current_node_id = node_id
        self._clear_widgets()
        
        if not node_id:
            self._title_label.setText("Properties")
            self._empty_label.show()
            return
        
        self._title_label.setText(f"Properties: {title}")
        self._empty_label.hide()
        
        # If we have definitions, use them
        if definitions:
            for param_def in definitions:
                self._add_parameter_widget(param_def, parameters or {})
        else:
            # Otherwise create simple widgets from values
            if parameters:
                for name, value in parameters.items():
                    self._add_simple_parameter(name, value)
        
        # Load dynamic model params if we have a model selected
        if self._pending_model_id:
            self._update_model_params(self._pending_model_id)
            self._pending_model_id = None
        
        # Load dynamic filter params if we have a filter selected
        if self._pending_filter_id:
            self._update_filter_params(self._pending_filter_id)
            self._pending_filter_id = None
    
    def set_simple_properties(
        self,
        node_id: str | None,
        title: str = "",
        properties: dict[str, tuple[str, Any]] | None = None,
    ) -> None:
        """
        Set properties using a simplified format.
        
        Args:
            node_id: Node ID
            title: Node title
            properties: Dict of {name: (type_hint, value)}
                       type_hint: "text", "int", "float", "bool", "choice:a,b,c"
        """
        self._current_node_id = node_id
        self._clear_widgets()
        
        if not node_id:
            self._title_label.setText("Properties")
            self._empty_label.show()
            return
        
        self._title_label.setText(f"Properties: {title}")
        self._empty_label.hide()
        
        if not properties:
            return
        
        for name, (type_hint, value) in properties.items():
            widget = self._create_widget_for_type(name, type_hint, value)
            if widget:
                # Create label
                label = QLabel(self._format_label(name))
                label.setStyleSheet("color: #a6adc8;")
                
                # Container
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(4)
                container_layout.addWidget(label)
                container_layout.addWidget(widget)
                
                self._params_layout.insertWidget(
                    self._params_layout.count() - 1,  # Before stretch
                    container
                )
                self._widgets[name] = widget
    
    def _clear_widgets(self) -> None:
        """Remove all parameter widgets."""
        for widget in self._widgets.values():
            widget.deleteLater()
        self._widgets.clear()
        self._param_containers.clear()
        self._current_model_id = None
        
        # Remove all except stretch and empty label
        while self._params_layout.count() > 2:
            item = self._params_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
    
    def _add_parameter_widget(self, param_def, values: dict[str, Any]) -> None:
        """Add a widget for a parameter definition."""
        from ai_image_studio.core.node_types import ParameterType
        
        name = param_def.name
        value = values.get(name, param_def.default)
        
        # Create label
        label = QLabel(param_def.label)
        label.setStyleSheet("color: #a6adc8;")
        if param_def.description:
            label.setToolTip(param_def.description)
        
        # Create widget based on type
        widget: QWidget | None = None
        
        if param_def.param_type == ParameterType.TEXT:
            widget = QLineEdit()
            widget.setText(str(value) if value else "")
            widget.textChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        elif param_def.param_type == ParameterType.TEXT_MULTILINE:
            widget = QTextEdit()
            widget.setPlainText(str(value) if value else "")
            widget.setMaximumHeight(100)
            widget.textChanged.connect(
                lambda n=name: self._on_value_changed(n, widget.toPlainText())
            )
        
        elif param_def.param_type == ParameterType.INTEGER:
            widget = QSpinBox()
            if param_def.min_value is not None:
                widget.setMinimum(int(param_def.min_value))
            if param_def.max_value is not None:
                widget.setMaximum(int(param_def.max_value))
            widget.setValue(int(value) if value else 0)
            widget.valueChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        elif param_def.param_type == ParameterType.FLOAT:
            widget = QDoubleSpinBox()
            if param_def.min_value is not None:
                widget.setMinimum(param_def.min_value)
            if param_def.max_value is not None:
                widget.setMaximum(param_def.max_value)
            if param_def.step is not None:
                widget.setSingleStep(param_def.step)
            widget.setValue(float(value) if value else 0.0)
            widget.valueChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        elif param_def.param_type == ParameterType.BOOLEAN:
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.toggled.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        elif param_def.param_type == ParameterType.ENUM:
            widget = QComboBox()
            for option in param_def.options:
                widget.addItem(option.label, option.value)
            if value:
                idx = widget.findData(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            
            # Special handling for filter_id - update filter params dynamically
            if name == "filter_id":
                def on_filter_changed(idx, n=name, w=widget):
                    filter_id = w.currentData()
                    self._on_value_changed(n, filter_id)
                    self._update_filter_params(filter_id)
                widget.currentIndexChanged.connect(on_filter_changed)
                # Queue initial filter param load
                if value:
                    self._pending_filter_id = value
            else:
                widget.currentIndexChanged.connect(
                    lambda idx, n=name: self._on_value_changed(n, widget.currentData())
                )
        
        elif param_def.param_type == ParameterType.SLIDER:
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            
            slider = QSlider(Qt.Orientation.Horizontal)
            value_label = QLabel()
            
            # Scale to integers for slider
            scale = 1000
            slider.setMinimum(int((param_def.min_value or 0) * scale))
            slider.setMaximum(int((param_def.max_value or 1) * scale))
            slider.setValue(int((float(value) if value else 0) * scale))
            
            def update_label(v):
                real = v / scale
                value_label.setText(f"{real:.2f}")
                self._on_value_changed(name, real)
            
            slider.valueChanged.connect(update_label)
            update_label(slider.value())
            
            layout.addWidget(slider)
            layout.addWidget(value_label)
            widget = container
        
        elif param_def.param_type == ParameterType.SEED:
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            
            spin = QSpinBox()
            spin.setMinimum(-1)
            spin.setMaximum(2**31 - 1)
            spin.setValue(int(value) if value is not None else -1)
            spin.valueChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
            
            random_btn = QPushButton("🎲")
            random_btn.setFixedWidth(32)
            random_btn.setToolTip("Random seed")
            random_btn.clicked.connect(lambda: spin.setValue(-1))
            
            layout.addWidget(spin)
            layout.addWidget(random_btn)
            widget = container
        
        elif param_def.param_type == ParameterType.MODEL:
            # Dynamic model selector - populates from providers
            widget = QComboBox()
            
            # Get available models from provider registry
            try:
                from ai_image_studio.providers import get_registry
                from ai_image_studio.providers.base import GenerationMode
                registry = get_registry()
                
                # First try configured providers only
                models = registry.list_available_models()
                if not models:
                    # Fall back to all models
                    models = registry.list_models()
                
                # Filter by mode if specified (e.g., "upscale" for upscaler nodes)
                if param_def.mode_filter:
                    try:
                        mode = GenerationMode(param_def.mode_filter)
                        models = [m for m in models if mode in m.modes]
                    except ValueError:
                        pass  # Invalid mode filter, show all
                
                for model in models:
                    widget.addItem(f"{model.name} ({model.provider})", model.id)
                
            except ImportError:
                pass  # Providers not available
            
            if value:
                idx = widget.findData(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            
            # Connect to value change AND dynamic param update
            def on_model_changed(idx, n=name, w=widget):
                model_id = w.currentData()
                self._on_value_changed(n, model_id)
                self._update_model_params(model_id)
            
            widget.currentIndexChanged.connect(on_model_changed)
            
            # Initial load of model params
            if value:
                self._pending_model_id = value  # Set after widgets are added
        
        elif param_def.param_type == ParameterType.FILE_PATH:
            # File path with browse button
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            
            path_edit = QLineEdit()
            path_edit.setText(str(value) if value else "")
            path_edit.setPlaceholderText("Select file...")
            path_edit.textChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
            
            browse_btn = QPushButton("📂")
            browse_btn.setFixedWidth(32)
            browse_btn.setToolTip("Browse for file")
            
            def on_browse(n=name, edit=path_edit):
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Image",
                    "",
                    "Images (*.png *.jpg *.jpeg *.webp *.gif *.bmp);;All Files (*)"
                )
                if file_path:
                    edit.setText(file_path)
                    self._on_value_changed(n, file_path)
            
            browse_btn.clicked.connect(on_browse)
            
            layout.addWidget(path_edit, 1)
            layout.addWidget(browse_btn)
            widget = container
        
        elif param_def.param_type == ParameterType.FOLDER_PATH:
            # Folder path with browse button
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            
            path_edit = QLineEdit()
            path_edit.setText(str(value) if value else "")
            path_edit.setPlaceholderText("Select folder...")
            path_edit.textChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
            
            browse_btn = QPushButton("📁")
            browse_btn.setFixedWidth(32)
            browse_btn.setToolTip("Browse for folder")
            
            def on_browse_folder(n=name, edit=path_edit):
                folder_path = QFileDialog.getExistingDirectory(
                    self,
                    "Select Folder",
                    "",
                )
                if folder_path:
                    edit.setText(folder_path)
                    self._on_value_changed(n, folder_path)
            
            browse_btn.clicked.connect(on_browse_folder)
            
            layout.addWidget(path_edit, 1)
            layout.addWidget(browse_btn)
            widget = container
        
        else:
            # Default text input for unknown types
            widget = QLineEdit()
            widget.setText(str(value) if value else "")
            widget.textChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        if widget:
            # Container for label + widget
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(4)
            container_layout.addWidget(label)
            container_layout.addWidget(widget)
            
            self._params_layout.insertWidget(
                self._params_layout.count() - 1,
                container
            )
            self._widgets[name] = widget
            self._param_containers[name] = container  # Store container for visibility control
    
    def _add_simple_parameter(self, name: str, value: Any) -> None:
        """Add a widget for a simple parameter (auto-detect type)."""
        label = QLabel(self._format_label(name))
        label.setStyleSheet("color: #a6adc8;")
        
        widget: QWidget
        
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            widget.toggled.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setMaximum(2**31 - 1)
            widget.setMinimum(-2**31)
            widget.setValue(value)
            widget.valueChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setValue(value)
            widget.valueChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        else:
            widget = QLineEdit()
            widget.setText(str(value) if value else "")
            widget.textChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)
        container_layout.addWidget(label)
        container_layout.addWidget(widget)
        
        self._params_layout.insertWidget(
            self._params_layout.count() - 1,
            container
        )
        self._widgets[name] = widget
    
    def _create_widget_for_type(
        self,
        name: str,
        type_hint: str,
        value: Any,
    ) -> QWidget | None:
        """Create a widget based on type hint string."""
        widget: QWidget | None = None
        
        if type_hint == "text":
            widget = QLineEdit()
            widget.setText(str(value) if value else "")
            widget.textChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        elif type_hint == "multiline":
            widget = QTextEdit()
            widget.setPlainText(str(value) if value else "")
            widget.setMaximumHeight(100)
            widget.textChanged.connect(
                lambda n=name: self._on_value_changed(n, widget.toPlainText())
            )
        
        elif type_hint == "int":
            widget = QSpinBox()
            widget.setMaximum(2**31 - 1)
            widget.setMinimum(-2**31)
            widget.setValue(int(value) if value else 0)
            widget.valueChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        elif type_hint == "float":
            widget = QDoubleSpinBox()
            widget.setValue(float(value) if value else 0.0)
            widget.valueChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        elif type_hint == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.toggled.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        elif type_hint.startswith("choice:"):
            options = type_hint[7:].split(",")
            widget = QComboBox()
            for opt in options:
                widget.addItem(opt.strip())
            if value:
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            widget.currentTextChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
        
        elif type_hint == "seed":
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            
            spin = QSpinBox()
            spin.setMinimum(-1)
            spin.setMaximum(2**31 - 1)
            spin.setValue(int(value) if value else -1)
            spin.valueChanged.connect(
                lambda v, n=name: self._on_value_changed(n, v)
            )
            
            random_btn = QPushButton("🎲")
            random_btn.setFixedWidth(32)
            random_btn.clicked.connect(lambda: spin.setValue(-1))
            
            layout.addWidget(spin)
            layout.addWidget(random_btn)
            widget = container
        
        return widget
    
    def _format_label(self, name: str) -> str:
        """Convert parameter name to display label."""
        return name.replace("_", " ").title()
    
    def _on_value_changed(self, name: str, value: Any) -> None:
        """Handle parameter value change."""
        if self._current_node_id:
            self.parameter_changed.emit(self._current_node_id, name, value)
    
    def _update_model_params(self, model_id: str | None) -> None:
        """Update dynamic parameters based on selected model's ModelCard."""
        # Safely remove existing dynamic model param widgets
        for key in list(self._model_param_widgets.keys()):
            widget = self._model_param_widgets.pop(key, None)
            if widget is not None:
                try:
                    widget.deleteLater()
                except RuntimeError:
                    pass  # Widget already deleted
        
        if not model_id:
            # Show all base params when no model selected
            for container in self._param_containers.values():
                container.show()
            return
        
        self._current_model_id = model_id
        
        # Get ModelCard
        try:
            from ai_image_studio.providers import get_registry
            registry = get_registry()
            model = registry.get_model(model_id)
            
            if not model:
                return
            
            # Define which parameters are local-only (sd.cpp specific)
            local_only_params = {
                "steps", "cfg_scale", "sampler", "scheduler",
                "stream_previews", "preview_method", "preview_interval",
                "vae_tiling", "vae_tile_size", "vae_relative_tile_size", "vae_tile_overlap",
                "negative_prompt",  # Most cloud models don't use negative prompts
            }
            
            # Determine if this is a local model
            is_local = model.provider == "sd-cpp" or model.id.startswith("local/")
            
            # Hide/show parameters based on whether they're relevant
            for param_name, container in self._param_containers.items():
                if param_name in local_only_params:
                    # Show only for local models
                    container.setVisible(is_local)
                else:
                    # Always show other parameters (prompt, model, seed, width, height)
                    container.show()
            
            # Now add model-specific parameter options from the ModelCard
            if model.param_options:
                # Create separator
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.HLine)
                separator.setStyleSheet("background-color: #313244;")
                self._params_layout.insertWidget(
                    self._params_layout.count() - 1,
                    separator
                )
                self._model_param_widgets["_separator"] = separator
                
                # Create header
                header = QLabel(f"🎨 {model.name} Options")
                header.setStyleSheet("color: #89b4fa; font-weight: bold; margin-top: 8px;")
                self._params_layout.insertWidget(
                    self._params_layout.count() - 1,
                    header
                )
                self._model_param_widgets["_header"] = header
                
                # Create widgets for each param option
                for param_name, options in model.param_options.items():
                    label = QLabel(self._format_label(param_name))
                    label.setStyleSheet("color: #a6adc8;")
                    
                    combo = QComboBox()
                    for opt in options:
                        combo.addItem(str(opt), opt)
                    
                    # Set default if available
                    if model.param_defaults and param_name in model.param_defaults:
                        default = model.param_defaults[param_name]
                        idx = combo.findData(default)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                    
                    combo.currentIndexChanged.connect(
                        lambda idx, n=param_name, c=combo: self._on_value_changed(n, c.currentData())
                    )
                    
                    # Container
                    container = QWidget()
                    container_layout = QVBoxLayout(container)
                    container_layout.setContentsMargins(0, 0, 0, 0)
                    container_layout.setSpacing(4)
                    container_layout.addWidget(label)
                    container_layout.addWidget(combo)
                    
                    self._params_layout.insertWidget(
                        self._params_layout.count() - 1,
                        container
                    )
                    self._model_param_widgets[param_name] = container
                
        except ImportError:
            pass
        except Exception as e:
            print(f"Error loading model params: {e}")
    
    def _update_filter_params(self, filter_id: str | None) -> None:
        """Update dynamic G'MIC filter parameters based on selected filter."""
        from PySide6.QtWidgets import QLabel, QSlider, QSpinBox, QComboBox, QCheckBox, QVBoxLayout, QHBoxLayout, QWidget
        from PySide6.QtCore import Qt
        
        # Clear existing filter param widgets
        for widget in self._filter_param_widgets.values():
            widget.setParent(None)
            widget.deleteLater()
        self._filter_param_widgets.clear()
        
        if not filter_id:
            return
        
        try:
            from ai_image_studio.filters.filter_registry import get_filter, ParamType
            
            filter_spec = get_filter(filter_id)
            if not filter_spec or not filter_spec.params:
                return
            
            # Create widgets for each filter parameter
            for param in filter_spec.params:
                label = QLabel(param.name.replace("_", " ").title())
                label.setStyleSheet("color: #a6adc8;")
                
                widget = None
                scale = 100  # For slider precision
                
                if param.param_type == ParamType.FLOAT:
                    # Slider for float params
                    container_inner = QWidget()
                    inner_layout = QHBoxLayout(container_inner)
                    inner_layout.setContentsMargins(0, 0, 0, 0)
                    
                    slider = QSlider(Qt.Orientation.Horizontal)
                    value_label = QLabel()
                    
                    min_val = param.min_value if param.min_value is not None else 0
                    max_val = param.max_value if param.max_value is not None else 100
                    default = param.default if param.default is not None else min_val
                    
                    slider.setMinimum(int(min_val * scale))
                    slider.setMaximum(int(max_val * scale))
                    slider.setValue(int(default * scale))
                    
                    def update_slider(v, s=scale, n=param.name, lbl=value_label):
                        real = v / s
                        lbl.setText(f"{real:.2f}")
                        self._on_value_changed(n, real)
                    
                    slider.valueChanged.connect(update_slider)
                    update_slider(slider.value())
                    
                    inner_layout.addWidget(slider)
                    inner_layout.addWidget(value_label)
                    widget = container_inner
                    
                elif param.param_type == ParamType.INT:
                    spin = QSpinBox()
                    min_val = int(param.min_value) if param.min_value is not None else 0
                    max_val = int(param.max_value) if param.max_value is not None else 1000
                    spin.setMinimum(min_val)
                    spin.setMaximum(max_val)
                    spin.setValue(int(param.default) if param.default is not None else min_val)
                    spin.valueChanged.connect(
                        lambda v, n=param.name: self._on_value_changed(n, v)
                    )
                    widget = spin
                    
                elif param.param_type == ParamType.BOOL:
                    check = QCheckBox()
                    check.setChecked(bool(param.default) if param.default is not None else False)
                    check.toggled.connect(
                        lambda v, n=param.name: self._on_value_changed(n, v)
                    )
                    widget = check
                    
                elif param.param_type == ParamType.CHOICE:
                    combo = QComboBox()
                    if param.options:
                        for opt in param.options:
                            combo.addItem(str(opt), opt)
                    if param.default is not None:
                        idx = combo.findData(param.default)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                    combo.currentIndexChanged.connect(
                        lambda idx, n=param.name, c=combo: self._on_value_changed(n, c.currentData())
                    )
                    widget = combo
                
                if widget:
                    # Container for label + widget
                    container = QWidget()
                    container_layout = QVBoxLayout(container)
                    container_layout.setContentsMargins(0, 0, 0, 0)
                    container_layout.setSpacing(4)
                    container_layout.addWidget(label)
                    container_layout.addWidget(widget)
                    
                    # Insert before the stretch
                    self._params_layout.insertWidget(
                        self._params_layout.count() - 1,
                        container
                    )
                    self._filter_param_widgets[param.name] = container
                    
        except ImportError:
            pass
        except Exception as e:
            print(f"Error loading filter params: {e}")
