"""
Tests for the graph module.
"""

import pytest
from uuid import UUID

from ai_image_studio.core.graph import (
    Connection,
    Node,
    NodeGraph,
    NodeGroup,
    Point2D,
    Size2D,
    new_node_id,
)


class TestPoint2D:
    """Tests for Point2D dataclass."""
    
    def test_default_values(self):
        p = Point2D()
        assert p.x == 0.0
        assert p.y == 0.0
    
    def test_custom_values(self):
        p = Point2D(100, 200)
        assert p.x == 100
        assert p.y == 200
    
    def test_addition(self):
        p1 = Point2D(10, 20)
        p2 = Point2D(5, 10)
        result = p1 + p2
        assert result.x == 15
        assert result.y == 30
    
    def test_subtraction(self):
        p1 = Point2D(10, 20)
        p2 = Point2D(5, 10)
        result = p1 - p2
        assert result.x == 5
        assert result.y == 10


class TestNode:
    """Tests for Node dataclass."""
    
    def test_create_node(self):
        node = Node.create("input.prompt")
        assert node.type_id == "input.prompt"
        assert isinstance(node.id, UUID)
        assert node.is_dirty is True
    
    def test_set_parameter(self):
        node = Node.create("input.prompt")
        node.is_dirty = False
        
        node.set_parameter("text", "Hello")
        
        assert node.get_parameter("text") == "Hello"
        assert node.is_dirty is True  # Should be marked dirty
    
    def test_get_parameter_default(self):
        node = Node.create("input.prompt")
        assert node.get_parameter("missing", "default") == "default"
    
    def test_mark_dirty(self):
        node = Node.create("input.prompt")
        node.is_dirty = False
        node.cached_output = "something"
        
        node.mark_dirty()
        
        assert node.is_dirty is True
        assert node.cached_output is None


class TestNodeGraph:
    """Tests for NodeGraph class."""
    
    def test_create_empty_graph(self):
        graph = NodeGraph("Test Graph")
        assert graph.name == "Test Graph"
        assert len(graph) == 0
    
    def test_add_node(self):
        graph = NodeGraph()
        node = Node.create("input.prompt")
        
        graph.add_node(node)
        
        assert len(graph) == 1
        assert node.id in graph
    
    def test_remove_node(self):
        graph = NodeGraph()
        node = Node.create("input.prompt")
        graph.add_node(node)
        
        removed = graph.remove_node(node.id)
        
        assert removed == node
        assert len(graph) == 0
    
    def test_get_node(self):
        graph = NodeGraph()
        node = Node.create("input.prompt")
        graph.add_node(node)
        
        retrieved = graph.get_node(node.id)
        
        assert retrieved == node
    
    def test_add_connection(self):
        graph = NodeGraph()
        node1 = Node.create("input.prompt")
        node2 = Node.create("generation.text_to_image")
        graph.add_node(node1)
        graph.add_node(node2)
        
        conn = Connection.create(node1.id, "text", node2.id, "prompt")
        result = graph.add_connection(conn)
        
        assert result is True
        assert len(graph.connections) == 1
    
    def test_add_connection_invalid_source(self):
        graph = NodeGraph()
        node = Node.create("input.prompt")
        graph.add_node(node)
        
        fake_id = new_node_id()
        conn = Connection.create(fake_id, "text", node.id, "prompt")
        result = graph.add_connection(conn)
        
        assert result is False
    
    def test_remove_node_removes_connections(self):
        graph = NodeGraph()
        node1 = Node.create("input.prompt")
        node2 = Node.create("generation.text_to_image")
        graph.add_node(node1)
        graph.add_node(node2)
        
        conn = Connection.create(node1.id, "text", node2.id, "prompt")
        graph.add_connection(conn)
        
        graph.remove_node(node1.id)
        
        assert len(graph.connections) == 0
    
    def test_execution_order_simple(self):
        graph = NodeGraph()
        node1 = Node.create("input.prompt")
        node2 = Node.create("generation.text_to_image")
        node3 = Node.create("output.preview")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        # node1 -> node2 -> node3
        graph.add_connection(Connection.create(node1.id, "text", node2.id, "prompt"))
        graph.add_connection(Connection.create(node2.id, "image", node3.id, "image"))
        
        order = graph.get_execution_order()
        
        # node1 should come before node2, node2 before node3
        assert order.index(node1.id) < order.index(node2.id)
        assert order.index(node2.id) < order.index(node3.id)
    
    def test_execution_order_parallel(self):
        graph = NodeGraph()
        node1 = Node.create("input.prompt")
        node2 = Node.create("input.image")
        node3 = Node.create("generation.img2img")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        # Both node1 and node2 feed into node3
        graph.add_connection(Connection.create(node1.id, "text", node3.id, "prompt"))
        graph.add_connection(Connection.create(node2.id, "image", node3.id, "image"))
        
        order = graph.get_execution_order()
        
        # node1 and node2 should both come before node3
        assert order.index(node1.id) < order.index(node3.id)
        assert order.index(node2.id) < order.index(node3.id)
    
    def test_cycle_detection(self):
        graph = NodeGraph()
        node1 = Node.create("a")
        node2 = Node.create("b")
        graph.add_node(node1)
        graph.add_node(node2)
        
        # Create valid connection
        conn1 = Connection.create(node1.id, "out", node2.id, "in")
        assert graph.add_connection(conn1) is True
        
        # Try to create cycle
        conn2 = Connection.create(node2.id, "out", node1.id, "in")
        assert graph.add_connection(conn2) is False
    
    def test_self_connection_prevented(self):
        graph = NodeGraph()
        node = Node.create("utility.blend")
        graph.add_node(node)
        
        conn = Connection.create(node.id, "output", node.id, "input")
        result = graph.add_connection(conn)
        
        assert result is False
    
    def test_get_upstream_nodes(self):
        graph = NodeGraph()
        node1 = Node.create("a")
        node2 = Node.create("b")
        node3 = Node.create("c")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        graph.add_connection(Connection.create(node1.id, "out", node2.id, "in"))
        graph.add_connection(Connection.create(node2.id, "out", node3.id, "in"))
        
        upstream = graph.get_upstream_nodes(node3.id)
        
        assert node1.id in upstream
        assert node2.id in upstream
        assert node3.id not in upstream
    
    def test_get_downstream_nodes(self):
        graph = NodeGraph()
        node1 = Node.create("a")
        node2 = Node.create("b")
        node3 = Node.create("c")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        graph.add_connection(Connection.create(node1.id, "out", node2.id, "in"))
        graph.add_connection(Connection.create(node2.id, "out", node3.id, "in"))
        
        downstream = graph.get_downstream_nodes(node1.id)
        
        assert node2.id in downstream
        assert node3.id in downstream
        assert node1.id not in downstream
    
    def test_invalidate_from(self):
        graph = NodeGraph()
        node1 = Node.create("a")
        node2 = Node.create("b")
        node3 = Node.create("c")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        graph.add_connection(Connection.create(node1.id, "out", node2.id, "in"))
        graph.add_connection(Connection.create(node2.id, "out", node3.id, "in"))
        
        # Mark all clean
        node1.is_dirty = False
        node2.is_dirty = False
        node3.is_dirty = False
        
        # Invalidate from node1
        invalidated = graph.invalidate_from(node1.id)
        
        assert node1.id in invalidated
        assert node2.id in invalidated
        assert node3.id in invalidated
        assert node1.is_dirty is True
        assert node2.is_dirty is True
        assert node3.is_dirty is True
    
    def test_get_output_nodes(self):
        graph = NodeGraph()
        node1 = Node.create("input.prompt")
        node2 = Node.create("generation.text_to_image")
        node3 = Node.create("output.preview")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        graph.add_connection(Connection.create(node1.id, "text", node2.id, "prompt"))
        graph.add_connection(Connection.create(node2.id, "image", node3.id, "image"))
        
        outputs = graph.get_output_nodes()
        
        # Only node3 has no outgoing connections
        assert len(outputs) == 1
        assert outputs[0].id == node3.id


class TestNodeGroup:
    """Tests for NodeGroup class."""
    
    def test_create_group(self):
        group = NodeGroup.create("My Group")
        assert group.name == "My Group"
        assert len(group.node_ids) == 0
    
    def test_create_group_with_nodes(self):
        node_ids = [new_node_id(), new_node_id()]
        group = NodeGroup.create("My Group", node_ids)
        assert len(group.node_ids) == 2
