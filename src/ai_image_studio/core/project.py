"""
Project Model - Project structure and settings.

This module defines the project data structure that contains
all the information needed to save/load a complete project.

Reference: architecture.md#3-core-data-structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ai_image_studio.core.graph import NodeGraph


@dataclass
class ProjectSettings:
    """
    Project-level settings.
    
    These settings affect the entire project and are saved with it.
    """
    # Output settings
    default_output_format: str = "png"
    default_output_quality: int = 95
    output_directory: Path | None = None
    
    # Generation defaults
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 30
    default_cfg_scale: float = 7.5
    
    # Metadata settings
    embed_metadata: bool = True
    metadata_format: str = "exif"  # "exif", "xmp", "png_chunk"
    
    # Cache settings
    enable_cache: bool = True
    cache_size_mb: int = 2048
    
    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        return {
            "default_output_format": self.default_output_format,
            "default_output_quality": self.default_output_quality,
            "output_directory": str(self.output_directory) if self.output_directory else None,
            "default_width": self.default_width,
            "default_height": self.default_height,
            "default_steps": self.default_steps,
            "default_cfg_scale": self.default_cfg_scale,
            "embed_metadata": self.embed_metadata,
            "metadata_format": self.metadata_format,
            "enable_cache": self.enable_cache,
            "cache_size_mb": self.cache_size_mb,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectSettings:
        """Create settings from dictionary."""
        return cls(
            default_output_format=data.get("default_output_format", "png"),
            default_output_quality=data.get("default_output_quality", 95),
            output_directory=Path(data["output_directory"]) if data.get("output_directory") else None,
            default_width=data.get("default_width", 1024),
            default_height=data.get("default_height", 1024),
            default_steps=data.get("default_steps", 30),
            default_cfg_scale=data.get("default_cfg_scale", 7.5),
            embed_metadata=data.get("embed_metadata", True),
            metadata_format=data.get("metadata_format", "exif"),
            enable_cache=data.get("enable_cache", True),
            cache_size_mb=data.get("cache_size_mb", 2048),
        )


@dataclass
class Project:
    """
    A complete project containing the node graph and settings.
    
    Projects can be saved to and loaded from disk.
    """
    id: UUID
    name: str
    graph: NodeGraph
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    
    # File location (None for unsaved projects)
    path: Path | None = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    # State
    is_modified: bool = False
    
    @classmethod
    def create(cls, name: str = "Untitled") -> Project:
        """Create a new empty project."""
        return cls(
            id=uuid4(),
            name=name,
            graph=NodeGraph(name=name),
        )
    
    def mark_modified(self) -> None:
        """Mark the project as having unsaved changes."""
        self.is_modified = True
        self.modified_at = datetime.now()
    
    def mark_saved(self, path: Path | None = None) -> None:
        """Mark the project as saved."""
        self.is_modified = False
        if path:
            self.path = path
    
    @property
    def display_name(self) -> str:
        """Get the display name with modified indicator."""
        modified = "* " if self.is_modified else ""
        return f"{modified}{self.name}"
    
    @property
    def is_saved(self) -> bool:
        """Check if this project has been saved to disk."""
        return self.path is not None
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert project to dictionary for serialization.
        
        Note: This is a simplified version. Full serialization
        would include the complete graph structure.
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "settings": self.settings.to_dict(),
            # TODO: Add graph serialization
            "graph": {
                "id": str(self.graph.id),
                "name": self.graph.name,
                # Nodes and connections will be serialized separately
            },
        }


class ProjectManager:
    """
    Manages project lifecycle: creation, loading, saving.
    
    This is a singleton that holds the current project and
    handles recent project history.
    """
    
    _instance: ProjectManager | None = None
    
    def __new__(cls) -> ProjectManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._current = None
            cls._instance._recent = []
        return cls._instance
    
    @classmethod
    def instance(cls) -> ProjectManager:
        return cls()
    
    def __init__(self):
        if not hasattr(self, '_current'):
            self._current: Project | None = None
            self._recent: list[Path] = []
    
    @property
    def current(self) -> Project | None:
        """Get the current project."""
        return self._current
    
    @current.setter
    def current(self, project: Project | None) -> None:
        """Set the current project."""
        self._current = project
    
    @property
    def recent_projects(self) -> list[Path]:
        """Get list of recent project paths."""
        return self._recent.copy()
    
    def new_project(self, name: str = "Untitled") -> Project:
        """Create and set a new project as current."""
        project = Project.create(name)
        self._current = project
        return project
    
    def add_recent(self, path: Path) -> None:
        """Add a path to recent projects."""
        if path in self._recent:
            self._recent.remove(path)
        self._recent.insert(0, path)
        # Keep only last 10
        self._recent = self._recent[:10]
    
    def close_current(self) -> bool:
        """
        Close the current project.
        
        Returns True if closed, False if cancelled (e.g., unsaved changes).
        """
        # TODO: Check for unsaved changes and prompt
        self._current = None
        return True
