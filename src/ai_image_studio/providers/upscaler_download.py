"""
Upscaler Binary Manager - Downloads and manages Real-ESRGAN ncnn-vulkan.

Downloads the realesrgan-ncnn-vulkan binary and models from GitHub releases.
Uses the official xinntao/Real-ESRGAN project which includes model files.
"""

from __future__ import annotations

import logging
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable
import urllib.request
import json
import tempfile

logger = logging.getLogger(__name__)


# Use xinntao/Real-ESRGAN official release (includes models)
DOWNLOAD_URLS = {
    "linux": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
    "windows": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip",
    "darwin": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip",
}


def get_upscaler_bin_dir() -> Path:
    """Get the directory where the upscaler binary should be stored."""
    return Path.home() / ".local" / "share" / "ai_image_studio" / "upscaler"


def get_upscaler_binary_path() -> Path:
    """Get the path to the realesrgan-ncnn-vulkan binary."""
    bin_dir = get_upscaler_bin_dir()
    if platform.system() == "Windows":
        return bin_dir / "realesrgan-ncnn-vulkan.exe"
    else:
        return bin_dir / "realesrgan-ncnn-vulkan"


def get_platform_key() -> str:
    """Get the platform key for download URL."""
    system = platform.system().lower()
    if system == "darwin":
        return "darwin"
    elif system == "windows":
        return "windows"
    else:
        return "linux"


def find_upscaler_binary() -> Path | None:
    """
    Find the realesrgan-ncnn-vulkan binary.
    
    Checks:
    1. Our managed location (~/.local/share/ai_image_studio/upscaler/)
    2. System PATH
    """
    # Check our managed location first
    managed_path = get_upscaler_binary_path()
    if managed_path.exists():
        return managed_path
    
    # Check PATH
    for name in ["realesrgan-ncnn-vulkan", "upscayl-bin", "upscayl-ncnn"]:
        path = shutil.which(name)
        if path:
            return Path(path)
    
    return None


def is_upscaler_installed() -> bool:
    """Check if realesrgan-ncnn-vulkan is available."""
    return find_upscaler_binary() is not None


def download_upscaler(
    progress_callback: Callable[[int, int], None] | None = None
) -> Path:
    """
    Download and install the realesrgan-ncnn-vulkan binary with models.
    
    Args:
        progress_callback: Optional callback(bytes_downloaded, total_bytes)
        
    Returns:
        Path to the installed binary
        
    Raises:
        RuntimeError: If download or installation fails
    """
    platform_key = get_platform_key()
    download_url = DOWNLOAD_URLS.get(platform_key)
    
    if not download_url:
        raise RuntimeError(f"No binary available for platform: {platform_key}")
    
    logger.info(f"Downloading Real-ESRGAN from {download_url}")
    
    # Create destination directory
    bin_dir = get_upscaler_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    try:
        # Use curl for reliable GitHub downloads (handles redirects properly)
        import subprocess
        
        result = subprocess.run(
            ["curl", "-L", "-o", str(tmp_path), download_url],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")
        
        # Verify we got a zip file
        if not tmp_path.exists() or tmp_path.stat().st_size < 1000:
            raise RuntimeError("Download incomplete or failed")
        
        downloaded = tmp_path.stat().st_size
        logger.info(f"Downloaded {downloaded} bytes")
        
        # Extract zip - get all files including models/
        with zipfile.ZipFile(tmp_path, "r") as zf:
            # Find the root folder name in the zip
            root_folder = None
            for name in zf.namelist():
                if "/" in name:
                    root_folder = name.split("/")[0]
                    break
            
            for member in zf.namelist():
                # Skip directories
                if member.endswith("/"):
                    continue
                
                # Get relative path (strip root folder if exists)
                if root_folder and member.startswith(root_folder + "/"):
                    rel_path = member[len(root_folder) + 1:]
                else:
                    rel_path = member
                
                if not rel_path:
                    continue
                
                # Skip non-essential files
                if rel_path.endswith((".jpg", ".mp4", ".md")):
                    continue
                
                # Extract to destination
                dest_path = bin_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                
                # Make binary executable on Unix
                if rel_path == "realesrgan-ncnn-vulkan" and platform.system() != "Windows":
                    dest_path.chmod(dest_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
                
                logger.debug(f"Extracted: {rel_path}")
        
        binary_path = get_upscaler_binary_path()
        if not binary_path.exists():
            raise RuntimeError(f"Binary not found after extraction: {binary_path}")
        
        logger.info(f"Installed Real-ESRGAN to {bin_dir}")
        return binary_path
    
    finally:
        # Clean up temp file
        try:
            tmp_path.unlink()
        except Exception:
            pass


def download_upscaler_sync_with_dialog(parent=None) -> Path | None:
    """
    Download upscaler with a progress dialog.
    
    Args:
        parent: Optional Qt parent widget
        
    Returns:
        Path to binary or None if cancelled/failed
    """
    from PySide6.QtWidgets import QProgressDialog, QMessageBox
    from PySide6.QtCore import Qt
    
    progress = QProgressDialog(
        "Downloading Real-ESRGAN upscaler (~55MB)...",
        "Cancel",
        0, 100,
        parent
    )
    progress.setWindowTitle("Downloading Upscaler")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)
    
    cancelled = False
    
    def on_progress(downloaded: int, total: int):
        nonlocal cancelled
        if progress.wasCanceled():
            cancelled = True
            raise InterruptedError("Download cancelled")
        if total > 0:
            percent = int(downloaded * 100 / total)
            progress.setValue(percent)
            progress.setLabelText(f"Downloading... {downloaded // 1024} / {total // 1024} KB")
    
    try:
        path = download_upscaler(progress_callback=on_progress)
        progress.close()
        
        QMessageBox.information(
            parent,
            "Download Complete",
            f"Real-ESRGAN upscaler installed to:\n{path.parent}"
        )
        return path
        
    except InterruptedError:
        progress.close()
        return None
    except Exception as e:
        progress.close()
        QMessageBox.critical(
            parent,
            "Download Failed",
            f"Failed to download upscaler:\n{e}"
        )
        return None
