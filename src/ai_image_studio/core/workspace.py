"""
Workspace Persistence - Save and load node graphs to/from disk.

This module provides functions to serialize and deserialize node graphs
to a JSON format for workspace persistence.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from ai_image_studio.core.graph import NodeGraph, Node, Connection, Point2D


# Workspace storage directory
WORKSPACE_DIR = Path.home() / ".local" / "share" / "ai_image_studio" / "workspaces"


def get_workspace_dir() -> Path:
    """Get the workspace storage directory, creating if needed."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR


def save_workspace(
    graph: NodeGraph,
    node_id_map: dict[str, UUID],
    visual_nodes: dict[str, Any],
    connections: list[tuple[str, str, str, str]],
    viewport_state: dict[str, float] | None = None,
    path: Path | None = None,
    name: str = "workspace",
) -> Path:
    """
    Save a workspace to disk.
    
    Args:
        graph: The core NodeGraph model
        node_id_map: Mapping of visual IDs to core UUIDs
        visual_nodes: Visual node data from canvas
        connections: Visual connections list
        path: Optional specific path, otherwise uses default location
        name: Workspace name (used for filename if path not specified)
    
    Returns:
        Path where workspace was saved
    """
    # Build serializable data
    nodes_data = []
    for visual_id, core_uuid in node_id_map.items():
        core_node = graph.get_node(core_uuid)
        visual = visual_nodes.get(visual_id)
        
        if core_node and visual:
            nodes_data.append({
                "id": visual_id,
                "type_id": core_node.type_id,
                "x": visual.x,
                "y": visual.y,
                "parameters": core_node.parameters,
            })
    
    connections_data = [
        {
            "source": src,
            "source_output": src_out,
            "target": tgt,
            "target_input": tgt_in,
        }
        for src, src_out, tgt, tgt_in in connections
    ]
    
    workspace_data = {
        "version": 1,
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "nodes": nodes_data,
        "connections": connections_data,
        "viewport": viewport_state or {"zoom": 1.0, "offset_x": 0.0, "offset_y": 0.0},
    }
    
    # Determine save path
    if path is None:
        path = get_workspace_dir() / f"{name}.json"
    
    # Write to file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workspace_data, f, indent=2)
    
    return path


def load_workspace(path: Path) -> dict[str, Any]:
    """
    Load a workspace from disk.
    
    Args:
        path: Path to workspace JSON file
    
    Returns:
        Workspace data dict with 'nodes' and 'connections'
    
    Raises:
        FileNotFoundError: If workspace file doesn't exist
        ValueError: If workspace format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Workspace not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Validate
    if "version" not in data or "nodes" not in data:
        raise ValueError(f"Invalid workspace format: {path}")
    
    return data


def list_workspaces() -> list[dict[str, Any]]:
    """
    List all saved workspaces.
    
    Returns:
        List of workspace metadata dicts with 'name', 'path', 'saved_at'
    """
    workspaces = []
    workspace_dir = get_workspace_dir()
    
    for path in workspace_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            workspaces.append({
                "name": data.get("name", path.stem),
                "path": path,
                "saved_at": data.get("saved_at"),
                "node_count": len(data.get("nodes", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    
    # Sort by most recent
    workspaces.sort(key=lambda w: w.get("saved_at", ""), reverse=True)
    return workspaces


def get_last_session_path() -> Path:
    """Get the path for the auto-saved last session."""
    return get_workspace_dir() / "_last_session.json"
