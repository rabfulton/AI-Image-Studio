"""
Model Download Dialog - Download models from HuggingFace.

This dialog provides a curated list of popular Stable Diffusion models
in GGUF format that can be downloaded directly from HuggingFace.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable
import fnmatch
import os
import subprocess
import sys
import time

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QGroupBox,
    QProgressBar,
    QMessageBox,
    QDialogButtonBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QFileDialog,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


from ai_image_studio.ui.dialogs.model_catalog import MODEL_CATALOG, CatalogModel, ModelArtifact


# ============================================================================
# Download Worker
# ============================================================================

class DownloadWorker(QObject):
    """Worker thread for downloading models."""
    
    progress = Signal(int, int)  # completed_files, total_files
    finished = Signal(bool, str)  # success, message
    
    def __init__(self, model: CatalogModel, dest_folder: Path):
        super().__init__()
        self._model = model
        self._dest_folder = dest_folder
        self._cancelled = False
        self._proc: subprocess.Popen[str] | None = None
    
    def cancel(self):
        self._cancelled = True
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    
    def run(self):
        """Execute the download."""
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()

            resolved_artifacts: list[tuple[str, str]] = []  # (repo, filename)

            for artifact in self._model.artifacts:
                repo = artifact.repo

                if artifact.filename:
                    resolved_artifacts.append((repo, artifact.filename))
                    continue

                if not artifact.patterns:
                    raise RuntimeError(f"No filename or patterns provided for repo: {repo}")

                repo_files = api.list_repo_files(repo)
                candidates = [f for f in repo_files if f.lower().endswith((".safetensors", ".gguf", ".ckpt", ".pth"))]
                matched: list[str] = []
                for pattern in artifact.patterns:
                    matched.extend([f for f in candidates if fnmatch.fnmatch(f, pattern)])

                # Prefer main/root files over nested artifacts (tokenizers, etc.) unless explicitly asked.
                matched = [f for f in matched if "/" not in f] or matched

                if not matched:
                    raise RuntimeError(f"No files matched patterns {artifact.patterns} in {repo}")

                # Prefer smaller/fast variants for "auto-select" where possible.
                # For safetensors, prefer names containing turbo if available.
                def score(name: str) -> tuple[int, int]:
                    lname = name.lower()
                    return (
                        0 if "turbo" in lname else 1,
                        0 if lname.endswith(".safetensors") else 1,
                    )

                matched.sort(key=score)
                resolved_artifacts.append((repo, matched[0]))

            # Expand GGUF component downloads (VAE/CLIP/T5/etc.) when enabled.
            downloads: list[tuple[str, str]] = list(resolved_artifacts)
            if self._model.include_components:
                for repo, filename in resolved_artifacts:
                    if not filename.lower().endswith(".gguf"):
                        continue
                    repo_files = api.list_repo_files(repo)

                    def is_component_file(path: str) -> bool:
                        name = path.lower()
                        if not (name.endswith(".gguf") or name.endswith(".safetensors")):
                            return False
                        # Restrict to known component keywords.
                        return any(key in name for key in ("vae", "taesd", "clip", "t5"))

                    for f in repo_files:
                        if f == filename:
                            continue
                        if is_component_file(f):
                            downloads.append((repo, f))

            # De-duplicate while preserving order.
            seen: set[tuple[str, str]] = set()
            downloads = [(r, f) for (r, f) in downloads if not ((r, f) in seen or seen.add((r, f)))]

            total = len(downloads)
            completed = 0
            self.progress.emit(completed, total)

            for repo, filename in downloads:
                if self._cancelled:
                    raise InterruptedError("Download cancelled")

                logger.info("Downloading %s/%s", repo, filename)

                # Use a subprocess for the actual download so we can cancel mid-file
                # (hf_hub_download does not provide an abort mechanism).
                cmd = [
                    sys.executable,
                    "-c",
                    (
                        "from huggingface_hub import hf_hub_download; "
                        "hf_hub_download("
                        f"repo_id={repo!r}, filename={filename!r}, local_dir={str(self._dest_folder)!r}"
                        ")"
                    ),
                ]
                env = dict(os.environ)
                env["HF_HUB_DISABLE_TELEMETRY"] = "1"
                self._proc = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    env=env,
                )

                # Poll for completion/cancel.
                while True:
                    if self._cancelled:
                        try:
                            self._proc.terminate()
                            self._proc.wait(timeout=5)
                        except Exception:
                            try:
                                self._proc.kill()
                            except Exception:
                                pass
                        raise InterruptedError("Download cancelled")

                    ret = self._proc.poll()
                    if ret is not None:
                        if ret != 0:
                            raise RuntimeError(f"Download failed for {repo}/{filename}")
                        break
                    time.sleep(0.1)

                self._proc = None

                completed += 1
                self.progress.emit(completed, total)

            # Assume the first artifact is the "primary model file".
            primary_repo, primary_filename = resolved_artifacts[0]
            dest_path = self._dest_folder / primary_filename
            if dest_path.exists():
                extra_count = max(0, total - 1)
                suffix = f" (+{extra_count} component file(s))" if extra_count else ""
                self.finished.emit(True, f"Downloaded to: {dest_path}{suffix}")
            else:
                self.finished.emit(False, "Download completed but primary model file not found")

        except ImportError:
            self.finished.emit(False, "huggingface-hub is not installed. Run: pip install huggingface-hub")
        except InterruptedError:
            self.finished.emit(False, "Download cancelled")
        except Exception as e:
            self.finished.emit(False, f"Download failed: {e}")


# ============================================================================
# Download Dialog
# ============================================================================

class ModelDownloadDialog(QDialog):
    """
    Dialog for downloading models from HuggingFace.
    
    Presents a curated list of popular models and allows
    one-click download to the configured model folder.
    """
    
    model_downloaded = Signal(str)  # path to downloaded model
    
    def __init__(self, dest_folder: Path | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        
        self._dest_folder = dest_folder or Path.home() / ".local" / "share" / "ai-models"
        self._download_thread: QThread | None = None
        self._download_worker: DownloadWorker | None = None
        
        self.setWindowTitle("Download Models")
        self.setMinimumSize(650, 450)
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("<h2>Download Models</h2>")
        layout.addWidget(header)
        
        desc = QLabel(
            "Download popular Stable Diffusion models in GGUF format. "
            "Models will be saved to the configured model folder."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a6adc8;")
        layout.addWidget(desc)
        
        # Destination folder
        dest_layout = QHBoxLayout()
        
        dest_label = QLabel("Destination:")
        dest_layout.addWidget(dest_label)
        
        self._dest_label = QLabel(str(self._dest_folder))
        self._dest_label.setStyleSheet("color: #89b4fa;")
        dest_layout.addWidget(self._dest_label, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        dest_layout.addWidget(browse_btn)
        
        layout.addLayout(dest_layout)
        
        # Model table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Model", "Description", "Size", "Architecture"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        
        # Populate table
        self._table.setRowCount(len(MODEL_CATALOG))
        for row, model in enumerate(MODEL_CATALOG):
            self._table.setItem(row, 0, QTableWidgetItem(model.name))
            self._table.setItem(row, 1, QTableWidgetItem(model.description))
            self._table.setItem(row, 2, QTableWidgetItem("—"))
            self._table.setItem(row, 3, QTableWidgetItem(model.architecture.upper()))
        
        self._table.selectRow(0)
        layout.addWidget(self._table)
        
        # Progress section
        progress_group = QGroupBox("Download Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self._progress_label = QLabel("Select a model and click Download")
        self._progress_label.setStyleSheet("color: #6c7086;")
        progress_layout.addWidget(self._progress_label)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        progress_layout.addWidget(self._progress_bar)
        
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self._download_btn = QPushButton("Download")
        self._download_btn.clicked.connect(self._on_download)
        button_layout.addWidget(self._download_btn)
        
        self._cancel_btn = QPushButton("Cancel Download")
        self._cancel_btn.clicked.connect(self._on_cancel_download)
        self._cancel_btn.setEnabled(False)
        button_layout.addWidget(self._cancel_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _on_browse(self) -> None:
        """Browse for destination folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Model Folder",
            str(self._dest_folder),
            QFileDialog.Option.ShowDirsOnly,
        )
        if folder:
            self._dest_folder = Path(folder)
            self._dest_label.setText(str(self._dest_folder))
    
    def _on_download(self) -> None:
        """Start downloading selected model."""
        row = self._table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a model to download.")
            return
        
        model = MODEL_CATALOG[row]
        
        # Ensure destination exists
        self._dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        # Best-effort primary destination check. Primary file is first artifact.
        primary = model.artifacts[0]
        dest_path = None
        if primary.filename:
            dest_path = self._dest_folder / primary.filename

        if dest_path and dest_path.exists():
            reply = QMessageBox.question(
                self,
                "Model Exists",
                f"Model already exists at:\n{dest_path}\n\nDownload anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Start download
        self._progress_label.setText(f"Downloading {model.name}...")
        self._progress_label.setStyleSheet("color: #89b4fa;")
        self._progress_bar.setValue(0)
        self._progress_bar.setRange(0, 0)  # Indeterminate
        self._progress_bar.setVisible(True)
        self._download_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._table.setEnabled(False)
        
        # Create worker
        self._download_worker = DownloadWorker(model, self._dest_folder)
        self._download_thread = QThread()
        self._download_worker.moveToThread(self._download_thread)
        
        # Connect signals
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(self._on_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        
        # Start
        self._download_thread.start()
    
    def _on_cancel_download(self) -> None:
        """Cancel the current download."""
        if self._download_worker:
            self._progress_label.setText("Cancelling download…")
            self._progress_label.setStyleSheet("color: #fab387;")
            self._download_worker.cancel()
    
    def _on_progress(self, current: int, total: int) -> None:
        """Update progress bar."""
        if total > 0:
            self._progress_bar.setRange(0, 100)
            percent = int(current * 100 / total)
            self._progress_bar.setValue(percent)
            self._progress_label.setText(f"Downloading... ({current}/{total} files)")
    
    def _on_download_finished(self, success: bool, message: str) -> None:
        """Handle download completion."""
        # Clean up thread
        if self._download_thread:
            self._download_thread.quit()
            self._download_thread.wait()
            self._download_thread = None
        self._download_worker = None
        
        # Update UI
        self._progress_bar.setRange(0, 100)
        self._download_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._table.setEnabled(True)
        
        if success:
            self._progress_bar.setValue(100)
            self._progress_label.setText(message)
            self._progress_label.setStyleSheet("color: #a6e3a1;")
            
            # Emit signal
            row = self._table.currentRow()
            if row >= 0:
                model = MODEL_CATALOG[row]
                primary = model.artifacts[0]
                if primary.filename:
                    path = self._dest_folder / primary.filename
                    self.model_downloaded.emit(str(path))
            
            QMessageBox.information(self, "Download Complete", message)
        else:
            self._progress_bar.setValue(0)
            self._progress_label.setText(message)
            self._progress_label.setStyleSheet("color: #f38ba8;")
            
            if "cancelled" not in message.lower():
                QMessageBox.warning(self, "Download Failed", message)
    
    def closeEvent(self, event):
        """Handle dialog close."""
        if self._download_thread and self._download_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Download in Progress",
                "A download is in progress. Please cancel and wait for it to stop before closing.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._on_cancel_download()
            event.ignore()
            return
        
        super().closeEvent(event)
