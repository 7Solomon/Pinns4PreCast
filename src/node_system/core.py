"""
Core node-based architecture for flexible ML pipeline construction.

This provides:
1. Base node abstraction with plugin support
2. Dynamic node registry
3. Graph execution engine
4. Type-safe connections
"""

from abc import ABC, abstractmethod
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect
from pydantic import BaseModel, Field as PydanticField


class PortType(str, Enum):
    """Data types that can flow through node connections."""
    MODEL = "model"
    PROBLEM = "problem"
    DATASET = "dataset"
    DATALOADER = "dataloader"
    SOLVER = "solver"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    CALLBACK = 'callback'
    LOGGER = 'logger'
    CONFIG = "config"
    MATERIAL = "material"
    DOMAIN = "domain"
    TENSOR = "tensor"
    ANY = "any"

    RUN_ID = "run_id"  # maybe?
    SPEC = "spec"


@dataclass
class Port:
    """Represents an input or output port on a node."""
    name: str
    port_type: PortType
    required: bool = True
    default: Any = None
    description: str = ""
    
    def validate(self, value: Any) -> bool:
        """Check if value matches port type (simplified)."""
        if self.port_type == PortType.ANY:
            return True
        return True


@dataclass
class NodeMetadata:
    """Metadata describing a node type."""
    category: str  # e.g., "model", "dataset", "training", "loss"
    display_name: str
    description: str
    color: str = "#4A90E2"  # For frontend visualization
    icon: Optional[str] = None


class Node(ABC):
    """
    Base class for all nodes in the system.
    
    Each node:
    - Declares input/output ports
    - Implements execute() to perform its function
    - Can expose configuration parameters
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {} 
        self._executed = False

    
    @classmethod
    @abstractmethod
    def get_input_ports(cls) -> List[Port]:
        """Define what inputs this node accepts."""
        pass
    
    @classmethod
    @abstractmethod
    def get_output_ports(cls) -> List[Port]:
        """Define what outputs this node produces."""
        pass
    
    @classmethod
    @abstractmethod
    def get_metadata(cls) -> NodeMetadata:
        """Provide metadata for UI display."""
        pass
    
    @classmethod
    def get_config_schema(cls) -> Optional[Type[BaseModel]]:
        """
        Optional: Return a Pydantic model for node configuration.
        This allows nodes to have adjustable parameters.
        """
        return None
    
    def set_config(self, config: Dict[str, Any]):
        """Apply configuration to this node instance."""
        schema = self.get_config_schema()
        if schema:
            self.config = schema(**config)
        else:
            self.config = None
    
    def set_input(self, port_name: str, value: Any):
        """Set an input value."""
        self.inputs[port_name] = value
        self._executed = False
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the node's logic.
        
        Returns:
            Dict mapping output port names to their values
        """
        pass
    
    def get_output(self, port_name: str) -> Any:
        """Retrieve an output value, executing if necessary."""
        if not self._executed:
            self.outputs = self.execute()
            self._executed = True
        return self.outputs.get(port_name)
    
    def reset(self):
        """Clear execution state."""
        self._executed = False
        self.outputs = {}


class NodeRegistry:
    """
    Central registry for all available node types.
    Supports dynamic plugin loading.
    """
    
    _nodes: Dict[str, Type[Node]] = {}
    
    @classmethod
    def register(cls, node_class: Type[Node], node_type: str = None):
        """Register a node class."""
        node_type = node_type or node_class.__name__
        cls._nodes[node_type] = node_class
        #print(f"Registered node: {node_type}")
    
    @classmethod
    def get(cls, node_type: str) -> Type[Node]:
        """Retrieve a node class by type."""
        if node_type not in cls._nodes:
            raise ValueError(f"Unknown node type: {node_type}")
        return cls._nodes[node_type]
    
    @classmethod
    def list_all(cls) -> Dict[str, NodeMetadata]:
        """Get metadata for all registered nodes."""
        return {
            node_type: node_cls.get_metadata()
            for node_type, node_cls in cls._nodes.items()
        }
    
    @classmethod
    def load_plugin(cls, module_path: str):
        """
        Load nodes from a plugin module.
        
        Plugin modules should define node classes that inherit from Node
        and are decorated with @register_node.
        """
        import importlib
        module = importlib.import_module(module_path)
        # The decorators will handle registration


def register_node(node_type: str = None):
    """
    Decorator to register a node class.
    
    Usage:
        @register_node("custom_model")
        class CustomModelNode(Node):
            ...
    """
    def decorator(cls: Type[Node]):
        NodeRegistry.register(cls, node_type)
        return cls
    return decorator


@dataclass
class Connection:
    """Represents a connection between two nodes."""
    from_node: str  # node_id
    from_port: str
    to_node: str
    to_port: str


class NodeGraph:
    """
    Manages a graph of connected nodes and executes them.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.connections: List[Connection] = []
    
    def add_node(self, node_type: str, node_id: str, config: Dict[str, Any] = None) -> Node:
        """Create and add a node to the graph."""
        node_class = NodeRegistry.get(node_type)
        node = node_class(node_id)
        
        if config is not None:
            node.set_config(config)
        
        self.nodes[node_id] = node
        return node
    
    def connect(self, from_node: str, from_port: str, to_node: str, to_port: str):
        """Connect an output port to an input port."""
        # Validate connection
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Invalid node IDs in connection")
        
        connection = Connection(from_node, from_port, to_node, to_port)
        self.connections.append(connection)
    
    def _build_execution_order(self) -> List[str]:
        """
        Topologically sort nodes for execution.
        Returns list of node IDs in execution order.
        """
        # Build dependency graph
        dependencies = {node_id: set() for node_id in self.nodes}
        
        for conn in self.connections:
            dependencies[conn.to_node].add(conn.from_node)
        
        # Kahn's algorithm for topological sort
        in_degree = {node_id: len(deps) for node_id, deps in dependencies.items()}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            
            # Reduce in-degree for dependent nodes
            for other_id in self.nodes:
                if node_id in dependencies[other_id]:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)
        
        if len(order) != len(self.nodes):
            raise ValueError("Graph contains cycles")
        
        return order
    
    def execute(self, output_node: str = None, output_port: str = None, context: Dict[str, Any] = None) -> Any:
        """
        Execute the graph.
        
        Args:
            output_node: If specified, return output from this node
            output_port: Port to retrieve from output_node
        """
        execution_order = self._build_execution_order()
        
        # Execute each node in order
        for node_id in execution_order:
            node = self.nodes[node_id]

            # Global context
            if context:
                node.context = context
            
            # Set inputs from connections
            for conn in self.connections:
                if conn.to_node == node_id:
                    from_node = self.nodes[conn.from_node]
                    value = from_node.get_output(conn.from_port)
                    node.set_input(conn.to_port, value)
            
            # Execute node
            node.execute()
        
        # Return requested output
        if output_node:
            return self.nodes[output_node].get_output(output_port)
        
        return None
    
    def save_to_file(self, filepath: str, metadata: dict = None):
        """
        Save graph to JSON file.
        
        Args:
            filepath: Path to save graph.json
            metadata: Optional metadata dict (description, author, tags)
        """
        graph_dict = self.to_dict()
        
        # Add metadata
        graph_dict["version"] = "1.0"
        graph_dict["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if metadata:
            graph_dict["metadata"] = metadata
        
        with open(filepath, 'w') as f:
            json.dump(graph_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'NodeGraph':
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        version = data.get("version", "1.0")
        if version != "1.0":
            raise ValueError(f"Unsupported graph version: {version}")
        
        return cls.from_dict(data)

    def get_config_snapshot(self) -> dict:
        """
        Get fully resolved configuration for all nodes.
        Useful for reproducing exact settings later.
        """
        snapshot = {}
        for node_id, node in self.nodes.items():
            snapshot[node_id] = {
                "type": node.__class__.__name__,
                "config_class": node.get_config_schema().__name__ if node.get_config_schema() else None,
                "config": node.config.model_dump() if hasattr(node, 'config') and node.config else {}
            }
        return snapshot
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to JSON-compatible dict."""
        return {
            "nodes": [
                {
                    "id": node_id,
                    "type": node.__class__.__name__,
                    "config": node.config.model_dump() if hasattr(node, 'config') and node.config else {}
                }
                for node_id, node in self.nodes.items()
            ],
            "connections": [
                {
                    "from": {"node": c.from_node, "port": c.from_port},
                    "to": {"node": c.to_node, "port": c.to_port}
                }
                for c in self.connections
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeGraph':
        """Deserialize graph from dict."""
        graph = cls()
        
        # Create nodes
        for node_data in data["nodes"]:
            graph.add_node(
                node_type=node_data["type"],
                node_id=node_data["id"],
                config=node_data.get("config")
            )
        
        # Create connections
        for conn_data in data["connections"]:
            graph.connect(
                from_node=conn_data["from"]["node"],
                from_port=conn_data["from"]["port"],
                to_node=conn_data["to"]["node"],
                to_port=conn_data["to"]["port"]
            )
        
        return graph

