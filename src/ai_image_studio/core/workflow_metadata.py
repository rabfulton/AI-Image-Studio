"""
Workflow Metadata - Save and load workflow state for gallery images.

This module provides functions to serialize and deserialize the complete
workflow state (node graph, parameters, connections) alongside gallery images,
enabling full workflow restoration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class WorkflowMetadata:
    """Complete workflow state for an image."""
    
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    image_filename: str = ""
    nodes: list[dict[str, Any]] = field(default_factory=list)
    connections: list[dict[str, str]] = field(default_factory=list)
    viewport: dict[str, float] = field(default_factory=lambda: {
        "zoom": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
    })


def get_workflow_path(image_path: Path) -> Path:
    """
    Get the workflow file path for an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Path to the corresponding workflow JSON file
    """
    return image_path.with_suffix(".workflow.json")


def save_workflow_metadata(image_path: Path, metadata: WorkflowMetadata) -> Path:
    """
    Save workflow metadata alongside an image.
    
    Args:
        image_path: Path to the associated image file
        metadata: WorkflowMetadata to save
    
    Returns:
        Path where workflow was saved
    """
    workflow_path = get_workflow_path(image_path)
    
    # Ensure image filename is set
    metadata.image_filename = image_path.name
    
    # Serialize to JSON
    data = asdict(metadata)
    
    with open(workflow_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    return workflow_path


def load_workflow_metadata(image_path: Path) -> WorkflowMetadata | None:
    """
    Load workflow metadata for an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        WorkflowMetadata if found, None if no workflow file exists
    
    Raises:
        ValueError: If workflow format is invalid
    """
    workflow_path = get_workflow_path(image_path)
    
    if not workflow_path.exists():
        return None
    
    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate minimal structure
        if "version" not in data or "nodes" not in data:
            raise ValueError(f"Invalid workflow format: {workflow_path}")
        
        return WorkflowMetadata(
            version=data.get("version", 1),
            created_at=data.get("created_at", ""),
            image_filename=data.get("image_filename", ""),
            nodes=data.get("nodes", []),
            connections=data.get("connections", []),
            viewport=data.get("viewport", {"zoom": 1.0, "offset_x": 0.0, "offset_y": 0.0}),
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse workflow: {workflow_path}: {e}")


def delete_workflow_metadata(image_path: Path) -> bool:
    """
    Delete workflow metadata for an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        True if file was deleted, False if it didn't exist
    """
    workflow_path = get_workflow_path(image_path)
    
    if workflow_path.exists():
        try:
            workflow_path.unlink()
            return True
        except Exception as e:
            print(f"Failed to delete workflow file {workflow_path}: {e}")
    
    return False
