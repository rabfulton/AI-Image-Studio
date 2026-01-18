"""
Console Panel - Application log and progress display.

This panel shows:
- Execution progress and status
- Error messages
- Job queue status
- General application logs

Reference: wireframes.md - Bottom dock panels
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QProgressBar,
    QFrame,
    QTabWidget,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QTextCursor, QColor, QTextCharFormat

if TYPE_CHECKING:
    pass


class ConsolePanel(QWidget):
    """
    Console panel for displaying application logs and progress.
    
    Features:
    - Logs tab: Scrolling text log with timestamps
    - Progress tab: Current job progress with status
    - Clear and copy functionality
    """
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Tab widget for different views
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #11111b;
            }
            QTabBar::tab {
                background-color: #1e1e2e;
                color: #a6adc8;
                padding: 6px 16px;
                border: none;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected {
                color: #cdd6f4;
                border-bottom-color: #89b4fa;
            }
        """)
        
        # Logs tab
        logs_widget = QWidget()
        logs_layout = QVBoxLayout(logs_widget)
        logs_layout.setContentsMargins(0, 0, 0, 0)
        
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet("""
            QTextEdit {
                background-color: #11111b;
                color: #cdd6f4;
                font-family: 'Fira Code', 'Consolas', monospace;
                font-size: 12px;
                border: none;
                padding: 8px;
            }
        """)
        logs_layout.addWidget(self._log_text)
        
        # Log controls
        log_controls = QFrame()
        log_controls.setStyleSheet("background-color: #1e1e2e; border-top: 1px solid #313244;")
        log_controls_layout = QHBoxLayout(log_controls)
        log_controls_layout.setContentsMargins(8, 4, 8, 4)
        
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._log_text.clear)
        log_controls_layout.addWidget(clear_btn)
        
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(60)
        copy_btn.clicked.connect(self._copy_logs)
        log_controls_layout.addWidget(copy_btn)
        
        log_controls_layout.addStretch()
        
        self._log_count_label = QLabel("0 messages")
        self._log_count_label.setStyleSheet("color: #6c7086;")
        log_controls_layout.addWidget(self._log_count_label)
        
        logs_layout.addWidget(log_controls)
        
        self._tabs.addTab(logs_widget, "Console")
        
        # Progress tab
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(16, 16, 16, 16)
        progress_layout.setSpacing(12)
        
        # Current job
        self._job_label = QLabel("No active job")
        self._job_label.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        progress_layout.addWidget(self._job_label)
        
        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet("color: #a6adc8;")
        progress_layout.addWidget(self._status_label)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #313244;
                border: none;
                border-radius: 4px;
                height: 20px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 4px;
            }
        """)
        progress_layout.addWidget(self._progress_bar)
        
        # Node progress
        self._node_label = QLabel("")
        self._node_label.setStyleSheet("color: #6c7086;")
        progress_layout.addWidget(self._node_label)
        
        progress_layout.addStretch()
        
        # Cancel button
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        progress_layout.addWidget(self._cancel_btn)
        
        self._tabs.addTab(progress_widget, "Progress")
        
        # Queue tab
        queue_widget = QWidget()
        queue_layout = QVBoxLayout(queue_widget)
        queue_layout.setContentsMargins(16, 16, 16, 16)
        queue_layout.setSpacing(12)
        
        self._queue_label = QLabel("Pending Jobs")
        self._queue_label.setStyleSheet("color: #cdd6f4; font-weight: bold;")
        queue_layout.addWidget(self._queue_label)
        
        self._queue_list = QTextEdit()
        self._queue_list.setReadOnly(True)
        self._queue_list.setStyleSheet("""
            QTextEdit {
                background-color: #11111b;
                color: #cdd6f4;
                font-family: 'Fira Code', 'Consolas', monospace;
                font-size: 12px;
                border: none;
                padding: 8px;
            }
        """)
        self._queue_list.setPlainText("No pending jobs")
        queue_layout.addWidget(self._queue_list)
        
        self._tabs.addTab(queue_widget, "Queue")
        
        layout.addWidget(self._tabs)
        
        # Message counter
        self._message_count = 0
    
    # -------------------------------------------------------------------------
    # Logging Methods
    # -------------------------------------------------------------------------
    
    def log(self, message: str, level: str = "info") -> None:
        """
        Add a log message.
        
        Args:
            message: The message to log
            level: One of "info", "warning", "error", "success", "debug"
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color based on level
        colors = {
            "info": "#cdd6f4",
            "warning": "#f9e2af",
            "error": "#f38ba8",
            "success": "#a6e3a1",
            "debug": "#6c7086",
        }
        color = colors.get(level, colors["info"])
        
        # Level prefix
        prefixes = {
            "info": "ℹ",
            "warning": "⚠",
            "error": "✗",
            "success": "✓",
            "debug": "•",
        }
        prefix = prefixes.get(level, "•")
        
        # Format and append
        html = f'<span style="color:#6c7086">[{timestamp}]</span> <span style="color:{color}">{prefix} {message}</span><br>'
        
        cursor = self._log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_text.setTextCursor(cursor)
        self._log_text.insertHtml(html)
        
        # Auto-scroll to bottom
        self._log_text.verticalScrollBar().setValue(
            self._log_text.verticalScrollBar().maximum()
        )
        
        self._message_count += 1
        self._log_count_label.setText(f"{self._message_count} messages")
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, "info")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, "warning")
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, "error")
    
    def log_success(self, message: str) -> None:
        """Log a success message."""
        self.log(message, "success")
    
    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(message, "debug")
    
    def _copy_logs(self) -> None:
        """Copy log text to clipboard."""
        from PySide6.QtWidgets import QApplication
        
        text = self._log_text.toPlainText()
        QApplication.clipboard().setText(text)
        self.log_info("Logs copied to clipboard")
    
    # -------------------------------------------------------------------------
    # Progress Methods
    # -------------------------------------------------------------------------
    
    def set_job(self, job_name: str) -> None:
        """Set the current job name."""
        self._job_label.setText(job_name)
        self._cancel_btn.setEnabled(True)
        self._tabs.setCurrentIndex(1)  # Switch to progress tab
    
    def set_status(self, status: str) -> None:
        """Set the current status message."""
        self._status_label.setText(status)
    
    def set_progress(self, value: int, maximum: int = 100) -> None:
        """Set progress bar value."""
        self._progress_bar.setMaximum(maximum)
        self._progress_bar.setValue(value)
    
    def set_node_progress(self, node_name: str, current: int, total: int) -> None:
        """Set current node being executed."""
        self._node_label.setText(f"Node {current}/{total}: {node_name}")
        self.set_progress(int(current / total * 100) if total > 0 else 0)
    
    def clear_progress(self) -> None:
        """Clear progress display."""
        self._job_label.setText("No active job")
        self._status_label.setText("Idle")
        self._node_label.setText("")
        self._progress_bar.setValue(0)
        self._cancel_btn.setEnabled(False)
    
    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self.log_warning("Cancellation requested...")
        # Emit signal or call callback here
        self.cancel_requested.emit()
    
    # Signal for cancel button
    cancel_requested = Signal()
