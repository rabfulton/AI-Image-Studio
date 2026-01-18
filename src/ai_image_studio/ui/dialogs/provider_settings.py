"""
Provider Settings Dialog - Configure API keys and provider settings.

This dialog allows users to configure API keys and settings for
each image generation provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QSpinBox,
    QComboBox,
    QFileDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon

if TYPE_CHECKING:
    pass


class ProviderSettingsDialog(QDialog):
    """
    Dialog for configuring image generation providers.
    
    Allows users to:
    - Enter API keys for each provider
    - Enable/disable providers
    - Test API connections
    - View available models
    """
    
    settings_changed = Signal()
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        self.setWindowTitle("Provider Settings")
        self.setMinimumSize(600, 450)
        
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QHBoxLayout(self)
        
        # Provider list on left
        list_layout = QVBoxLayout()
        
        self._provider_list = QListWidget()
        self._provider_list.setMaximumWidth(150)
        self._provider_list.currentRowChanged.connect(self._on_provider_changed)
        list_layout.addWidget(self._provider_list)
        
        layout.addLayout(list_layout)
        
        # Settings panel on right
        right_layout = QVBoxLayout()
        
        self._settings_stack = QStackedWidget()
        right_layout.addWidget(self._settings_stack)
        
        # Add provider pages
        self._add_provider_page("openai", "OpenAI", [
            ("api_key", "API Key", "sk-...", True),
        ], "https://platform.openai.com/api-keys")
        
        self._add_provider_page("bfl", "Black Forest Labs", [
            ("api_key", "API Key", "Your BFL API key", True),
        ], "https://api.bfl.ml/")
        
        self._add_provider_page("openrouter", "OpenRouter", [
            ("api_key", "API Key", "Your OpenRouter key", True),
        ], "https://openrouter.ai/keys")
        
        self._add_provider_page("gemini", "Google Gemini", [
            ("api_key", "API Key", "Your Gemini API key", True),
        ], "https://aistudio.google.com/app/apikey")
        
        self._add_provider_page("xai", "xAI (Grok)", [
            ("api_key", "API Key", "Your xAI API key", True),
        ], "https://console.x.ai/")
        
        # Local provider (special handling)
        self._add_local_provider_page()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._on_test_connection)
        button_layout.addWidget(self._test_btn)
        
        button_layout.addStretch()
        
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self._on_save)
        self._button_box.rejected.connect(self.reject)
        button_layout.addWidget(self._button_box)
        
        right_layout.addLayout(button_layout)
        layout.addLayout(right_layout, 1)
        
        # Select first provider
        if self._provider_list.count() > 0:
            self._provider_list.setCurrentRow(0)
    
    def _add_provider_page(
        self,
        provider_id: str,
        name: str,
        fields: list[tuple[str, str, str, bool]],  # (id, label, placeholder, is_password)
        url: str,
    ) -> None:
        """Add a settings page for a provider."""
        # Add to list
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, provider_id)
        self._provider_list.addItem(item)
        
        # Create page
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel(f"<h2>{name}</h2>")
        page_layout.addWidget(header)
        
        # Enable checkbox
        enable_cb = QCheckBox("Enable this provider")
        enable_cb.setChecked(True)
        enable_cb.setObjectName(f"{provider_id}_enabled")
        page_layout.addWidget(enable_cb)
        
        # Settings group
        group = QGroupBox("Settings")
        form = QFormLayout(group)
        
        for field_id, label, placeholder, is_password in fields:
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setObjectName(f"{provider_id}_{field_id}")
            if is_password:
                edit.setEchoMode(QLineEdit.EchoMode.Password)
                
                # Add show/hide button
                container = QWidget()
                container_layout = QHBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.addWidget(edit)
                
                show_btn = QPushButton("👁")
                show_btn.setFixedWidth(30)
                show_btn.setCheckable(True)
                show_btn.toggled.connect(
                    lambda checked, e=edit: e.setEchoMode(
                        QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
                    )
                )
                container_layout.addWidget(show_btn)
                
                form.addRow(label + ":", container)
            else:
                form.addRow(label + ":", edit)
        
        page_layout.addWidget(group)
        
        # Link to get API key
        link = QLabel(f'<a href="{url}">Get API key →</a>')
        link.setOpenExternalLinks(True)
        link.setStyleSheet("color: #4a9eff;")
        page_layout.addWidget(link)
        
        # Models info
        models_group = QGroupBox("Available Models")
        models_layout = QVBoxLayout(models_group)
        
        from ai_image_studio.providers import get_registry
        registry = get_registry()
        models = registry.list_models(provider_id)
        
        if models:
            for model in models[:5]:  # Show first 5
                model_label = QLabel(f"• {model.name}")
                model_label.setStyleSheet("color: #a6adc8;")
                models_layout.addWidget(model_label)
            if len(models) > 5:
                more = QLabel(f"... and {len(models) - 5} more")
                more.setStyleSheet("color: #6c7086; font-style: italic;")
                models_layout.addWidget(more)
        else:
            no_models = QLabel("No models registered")
            no_models.setStyleSheet("color: #6c7086;")
            models_layout.addWidget(no_models)
        
        page_layout.addWidget(models_group)
        
        page_layout.addStretch()
        
        self._settings_stack.addWidget(page)
    
    def _add_local_provider_page(self) -> None:
        """Add settings page for local sd.cpp provider."""
        # Add to list
        item = QListWidgetItem("Local (sd.cpp)")
        item.setData(Qt.ItemDataRole.UserRole, "sd-cpp")
        self._provider_list.addItem(item)
        
        # Create page
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("<h2>Local (sd.cpp)</h2>")
        page_layout.addWidget(header)
        
        desc = QLabel("Run Stable Diffusion locally using stable-diffusion.cpp")
        desc.setStyleSheet("color: #a6adc8;")
        page_layout.addWidget(desc)
        
        # Enable checkbox
        enable_cb = QCheckBox("Enable this provider")
        enable_cb.setChecked(True)
        enable_cb.setObjectName("sd-cpp_enabled")
        page_layout.addWidget(enable_cb)
        
        # Model folders group
        folders_group = QGroupBox("Model Folders")
        folders_layout = QVBoxLayout(folders_group)
        
        self._local_folders_list = QListWidget()
        self._local_folders_list.setMaximumHeight(100)
        folders_layout.addWidget(self._local_folders_list)
        
        folder_buttons = QHBoxLayout()
        
        add_folder_btn = QPushButton("Add Folder...")
        add_folder_btn.clicked.connect(self._on_add_model_folder)
        folder_buttons.addWidget(add_folder_btn)
        
        remove_folder_btn = QPushButton("Remove")
        remove_folder_btn.clicked.connect(self._on_remove_model_folder)
        folder_buttons.addWidget(remove_folder_btn)
        
        folder_buttons.addStretch()
        
        download_btn = QPushButton("Download Models...")
        download_btn.clicked.connect(self._on_download_models)
        folder_buttons.addWidget(download_btn)
        
        scan_btn = QPushButton("Scan Models")
        scan_btn.clicked.connect(self._on_scan_models)
        folder_buttons.addWidget(scan_btn)
        
        folders_layout.addLayout(folder_buttons)
        page_layout.addWidget(folders_group)
        
        # Device settings group
        device_group = QGroupBox("Device Settings")
        device_form = QFormLayout(device_group)
        
        self._device_combo = QComboBox()
        self._device_combo.addItems(["Auto", "CPU", "CUDA", "Vulkan"])
        self._device_combo.setObjectName("sd-cpp_device")
        device_form.addRow("Device:", self._device_combo)
        
        self._threads_spin = QSpinBox()
        self._threads_spin.setRange(-1, 128)
        self._threads_spin.setValue(-1)
        self._threads_spin.setSpecialValueText("Auto")
        self._threads_spin.setObjectName("sd-cpp_threads")
        device_form.addRow("Threads:", self._threads_spin)

        self._keep_vae_on_cpu_cb = QCheckBox(
            "Keep VAE on CPU (reduces VRAM usage; helps avoid grey outputs)"
        )
        self._keep_vae_on_cpu_cb.setObjectName("sd-cpp_keep_vae_on_cpu")
        self._keep_vae_on_cpu_cb.setChecked(False)
        device_form.addRow("", self._keep_vae_on_cpu_cb)

        self._keep_clip_on_cpu_cb = QCheckBox("Keep CLIP on CPU (reduces VRAM usage)")
        self._keep_clip_on_cpu_cb.setObjectName("sd-cpp_keep_clip_on_cpu")
        self._keep_clip_on_cpu_cb.setChecked(False)
        device_form.addRow("", self._keep_clip_on_cpu_cb)
        
        page_layout.addWidget(device_group)
        
        # Library status and capabilities
        status_group = QGroupBox("Library Status")
        status_layout = QVBoxLayout(status_group)
        
        try:
            import stable_diffusion_cpp as sd_cpp
            status_text = "✓ stable-diffusion-cpp-python installed"
            status_style = "color: #a6e3a1;"
            
            # Check backend
            info = sd_cpp.sd_get_system_info()
            if isinstance(info, bytes):
                info = info.decode('utf-8', errors='ignore')
            
            # Determine backend (case-insensitive; different builds format this differently)
            info_upper = info.upper()
            if "CUBLAS" in info_upper or "CUDA" in info_upper:
                backend = "CUDA"
            elif "VULKAN" in info_upper:
                backend = "Vulkan"
            elif "METAL" in info_upper:
                backend = "Metal"
            else:
                backend = "CPU"
            
            backend_label = QLabel(f"Backend: {backend}")
            backend_label.setStyleSheet("color: #89b4fa;")
            status_layout.addWidget(backend_label)
            
            if backend == "CPU":
                hint = QLabel(
                    "💡 For GPU acceleration, reinstall with:\n"
                    "CMAKE_ARGS=\"-DSD_VULKAN=ON\" pip install --force-reinstall stable-diffusion-cpp-python"
                )
                hint.setStyleSheet("color: #fab387; font-size: 10px;")
                hint.setWordWrap(True)
                status_layout.addWidget(hint)
                
        except ImportError:
            status_text = "⚠ stable-diffusion-cpp-python not installed"
            status_style = "color: #f38ba8;"
        
        status = QLabel(status_text)
        status.setStyleSheet(status_style)
        status_layout.insertWidget(0, status)
        
        page_layout.addWidget(status_group)
        
        # Discovered models
        self._local_models_group = QGroupBox("Discovered Models")
        self._local_models_layout = QVBoxLayout(self._local_models_group)
        
        no_models = QLabel("Click 'Scan Models' to discover local models")
        no_models.setStyleSheet("color: #6c7086;")
        no_models.setObjectName("sd-cpp_no_models_label")
        self._local_models_layout.addWidget(no_models)
        
        page_layout.addWidget(self._local_models_group)
        
        page_layout.addStretch()
        
        self._settings_stack.addWidget(page)
    
    def _on_add_model_folder(self) -> None:
        """Add a model folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Model Folder",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if folder:
            # Check if already in list
            for i in range(self._local_folders_list.count()):
                if self._local_folders_list.item(i).text() == folder:
                    return
            self._local_folders_list.addItem(folder)
    
    def _on_remove_model_folder(self) -> None:
        """Remove selected model folder."""
        current = self._local_folders_list.currentRow()
        if current >= 0:
            self._local_folders_list.takeItem(current)
    
    def _on_scan_models(self) -> None:
        """Scan configured folders for models."""
        from pathlib import Path
        from ai_image_studio.providers.sd_cpp_models import LocalModelScanner
        
        # Gather folders
        folders = []
        for i in range(self._local_folders_list.count()):
            folders.append(Path(self._local_folders_list.item(i).text()))
        
        if not folders:
            QMessageBox.information(self, "No Folders", "Add model folders first.")
            return
        
        # Scan
        scanner = LocalModelScanner()
        models = scanner.scan(folders)
        
        # Clear existing labels
        while self._local_models_layout.count():
            item = self._local_models_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Show results
        if models:
            for info in models[:10]:  # Show first 10
                size_str = f"{info.size_gb:.1f}GB"
                label = QLabel(f"• {info.display_name} ({size_str})")
                label.setStyleSheet("color: #a6adc8;")
                self._local_models_layout.addWidget(label)
            if len(models) > 10:
                more = QLabel(f"... and {len(models) - 10} more")
                more.setStyleSheet("color: #6c7086; font-style: italic;")
                self._local_models_layout.addWidget(more)
            
            QMessageBox.information(
                self, "Scan Complete", f"Found {len(models)} model(s)."
            )
        else:
            no_models = QLabel("No models found in configured folders")
            no_models.setStyleSheet("color: #f38ba8;")
            self._local_models_layout.addWidget(no_models)
    
    def _on_download_models(self) -> None:
        """Open the model download dialog."""
        from pathlib import Path
        from ai_image_studio.ui.dialogs.model_download import ModelDownloadDialog
        
        # Determine destination folder
        if self._local_folders_list.count() > 0:
            dest = Path(self._local_folders_list.item(0).text())
        else:
            dest = Path.home() / ".local" / "share" / "ai-models"
        
        dialog = ModelDownloadDialog(dest_folder=dest, parent=self)
        
        # When a model is downloaded, refresh the list
        def on_model_downloaded(path: str):
            # Add folder if not already in list
            folder = str(Path(path).parent)
            exists = False
            for i in range(self._local_folders_list.count()):
                if self._local_folders_list.item(i).text() == folder:
                    exists = True
                    break
            if not exists:
                self._local_folders_list.addItem(folder)
            # Trigger scan
            self._on_scan_models()
        
        dialog.model_downloaded.connect(on_model_downloaded)
        dialog.exec()
    
    def _on_provider_changed(self, row: int) -> None:
        """Handle provider selection change."""
        self._settings_stack.setCurrentIndex(row)
    
    def _load_settings(self) -> None:
        """Load current settings into the UI."""
        from ai_image_studio.providers import get_registry
        registry = get_registry()
        
        for provider_id in ["openai", "bfl", "openrouter", "gemini", "xai"]:
            config = registry.get_config(provider_id)
            
            # Set enabled
            enabled_cb = self.findChild(QCheckBox, f"{provider_id}_enabled")
            if enabled_cb:
                enabled_cb.setChecked(config.enabled)
            
            # Set API key
            key_edit = self.findChild(QLineEdit, f"{provider_id}_api_key")
            if key_edit:
                key_edit.setText(config.api_key)
        
        # Load local provider settings
        local_config = registry.get_config("sd-cpp")
        
        # Model folders
        folders = local_config.extra.get("model_folders", [])
        self._local_folders_list.clear()
        for folder in folders:
            self._local_folders_list.addItem(folder)
        
        # Device
        device = local_config.extra.get("device", "auto").lower()
        device_map = {"auto": 0, "cpu": 1, "cuda": 2, "vulkan": 3}
        self._device_combo.setCurrentIndex(device_map.get(device, 0))
        
        # Threads
        threads = local_config.extra.get("n_threads", -1)
        self._threads_spin.setValue(threads)

        # Runtime flags
        keep_vae_on_cpu = bool(local_config.extra.get("keep_vae_on_cpu", True))
        keep_clip_on_cpu = bool(local_config.extra.get("keep_clip_on_cpu", False))
        self._keep_vae_on_cpu_cb.setChecked(keep_vae_on_cpu)
        self._keep_clip_on_cpu_cb.setChecked(keep_clip_on_cpu)
        
        # Enabled
        enabled_cb = self.findChild(QCheckBox, "sd-cpp_enabled")
        if enabled_cb:
            enabled_cb.setChecked(local_config.enabled)
    
    def _on_save(self) -> None:
        """Save settings and close."""
        from ai_image_studio.providers import get_registry, ProviderConfig
        registry = get_registry()
        
        for provider_id in ["openai", "bfl", "openrouter", "gemini", "xai"]:
            enabled_cb = self.findChild(QCheckBox, f"{provider_id}_enabled")
            key_edit = self.findChild(QLineEdit, f"{provider_id}_api_key")
            
            config = ProviderConfig(
                api_key=key_edit.text() if key_edit else "",
                enabled=enabled_cb.isChecked() if enabled_cb else True,
            )
            registry.set_config(provider_id, config)
        
        # Save local provider settings
        enabled_cb = self.findChild(QCheckBox, "sd-cpp_enabled")
        
        # Gather folders
        folders = []
        for i in range(self._local_folders_list.count()):
            folders.append(self._local_folders_list.item(i).text())
        
        # Device
        device_map = {0: "auto", 1: "cpu", 2: "cuda", 3: "vulkan"}
        device = device_map.get(self._device_combo.currentIndex(), "auto")
        
        local_config = ProviderConfig(
            api_key="",  # Local provider doesn't use API keys
            enabled=enabled_cb.isChecked() if enabled_cb else True,
            extra={
                "model_folders": folders,
                "device": device,
                "n_threads": self._threads_spin.value(),
                "keep_vae_on_cpu": self._keep_vae_on_cpu_cb.isChecked(),
                "keep_clip_on_cpu": self._keep_clip_on_cpu_cb.isChecked(),
            },
        )
        registry.set_config("sd-cpp", local_config)
        
        # Refresh local models in registry
        registry.refresh_local_models()
        
        # Save to disk
        registry.save_config()
        
        self.settings_changed.emit()
        self.accept()
    
    def _on_test_connection(self) -> None:
        """Test connection to the current provider."""
        import asyncio
        
        item = self._provider_list.currentItem()
        if not item:
            return
        
        provider_id = item.data(Qt.ItemDataRole.UserRole)
        key_edit = self.findChild(QLineEdit, f"{provider_id}_api_key")
        
        if not key_edit or not key_edit.text():
            QMessageBox.warning(self, "Test Failed", "Please enter an API key first.")
            return
        
        # Create temporary config
        from ai_image_studio.providers import get_registry, ProviderConfig
        registry = get_registry()
        
        old_config = registry.get_config(provider_id)
        temp_config = ProviderConfig(api_key=key_edit.text())
        registry.set_config(provider_id, temp_config)
        
        provider = registry.get_provider(provider_id)
        if not provider:
            QMessageBox.warning(self, "Test Failed", "Provider not found.")
            return
        
        # Test connection
        self._test_btn.setEnabled(False)
        self._test_btn.setText("Testing...")
        
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(provider.validate_credentials())
            loop.close()
            
            if result:
                QMessageBox.information(self, "Success", "Connection successful! ✓")
            else:
                QMessageBox.warning(self, "Test Failed", "Invalid API key or connection failed.")
        except Exception as e:
            QMessageBox.warning(self, "Test Failed", f"Error: {str(e)}")
        finally:
            self._test_btn.setEnabled(True)
            self._test_btn.setText("Test Connection")
            # Restore old config if not saving
            registry.set_config(provider_id, old_config)
