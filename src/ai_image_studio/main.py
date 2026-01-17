"""
AI Image Studio - Main Entry Point

This module provides the main entry point for the application.
"""

import sys
from pathlib import Path


def main() -> int:
    """
    Main entry point for AI Image Studio.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Ensure we're running Python 3.11+
    if sys.version_info < (3, 11):
        print("Error: AI Image Studio requires Python 3.11 or later")
        return 1
    
    # Import Qt here to avoid import overhead if just checking version
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    
    # Import our main window
    from ai_image_studio.ui.main_window import MainWindow
    
    # Create application instance
    app = QApplication(sys.argv)
    app.setApplicationName("AI Image Studio")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("AI Image Studio")
    
    # Enable high DPI scaling
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
