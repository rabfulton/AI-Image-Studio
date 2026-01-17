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
    
    def _on_provider_changed(self, row: int) -> None:
        """Handle provider selection change."""
        self._settings_stack.setCurrentIndex(row)
    
    def _load_settings(self) -> None:
        """Load current settings into the UI."""
        from ai_image_studio.providers import get_registry
        registry = get_registry()
        
        for provider_id in ["openai", "bfl", "openrouter"]:
            config = registry.get_config(provider_id)
            
            # Set enabled
            enabled_cb = self.findChild(QCheckBox, f"{provider_id}_enabled")
            if enabled_cb:
                enabled_cb.setChecked(config.enabled)
            
            # Set API key
            key_edit = self.findChild(QLineEdit, f"{provider_id}_api_key")
            if key_edit:
                key_edit.setText(config.api_key)
    
    def _on_save(self) -> None:
        """Save settings and close."""
        from ai_image_studio.providers import get_registry, ProviderConfig
        registry = get_registry()
        
        for provider_id in ["openai", "bfl", "openrouter"]:
            enabled_cb = self.findChild(QCheckBox, f"{provider_id}_enabled")
            key_edit = self.findChild(QLineEdit, f"{provider_id}_api_key")
            
            config = ProviderConfig(
                api_key=key_edit.text() if key_edit else "",
                enabled=enabled_cb.isChecked() if enabled_cb else True,
            )
            registry.set_config(provider_id, config)
        
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
