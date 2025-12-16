
from typing import Any, Dict, List
from pydantic import BaseModel


class NodeConfigPayload(BaseModel):
    id: str
    type: str
    config: Dict[str, Any] = {}
    position: Dict[str, float] = {"x": 0, "y": 0}

class ConnectionPayload(BaseModel):
    source_node: str
    source_port: str
    target_node: str
    target_port: str

class GraphExecutionPayload(BaseModel):
    nodes: List[NodeConfigPayload]
    connections: List[ConnectionPayload]
    target_node_id: str



class GraphTemplate(BaseModel):
    """Represents a saved graph template."""
    name: str
    description: str = ""
    tags: List[str] = []
    nodes: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    created_at: str = ""
    updated_at: str = ""

class SaveGraphPayload(BaseModel):
    name: str
    description: str = ""
    tags: List[str] = []
    nodes: List[NodeConfigPayload]
    connections: List[ConnectionPayload]
    overwrite: bool = False  # Whether to overwrite if exists

class GraphMetadata(BaseModel):
    """Metadata about a saved graph."""
    name: str
    filename: str
    description: str
    tags: List[str]
    created_at: str
    updated_at: str
    node_count: int
    connection_count: int
