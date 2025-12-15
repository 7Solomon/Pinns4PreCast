from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.node_system.core import NodeGraph, NodeRegistry, Port

import src.node_system.registry
import pydantic

app = FastAPI(title="PINNs Node Editor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API MODELS ---
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

# --- HELPER FUNCTIONS ---

def port_to_dict(ports: Any) -> Dict[str, str]:
    """
    Robust converter that handles both List[Port] and Dict[str, Port].
    """
    result = {}
    
    # CASE 1: node returns a Dictionary (DONT DO THIS HERE BUT THATS WHAT I want )
    if isinstance(ports, dict):
        for name, port_obj in ports.items():
            if hasattr(port_obj, "port_type"):
                result[name] = port_obj.port_type.value
            else:
                result[name] = "any"

    # CASE 2: Node returns a List
    elif isinstance(ports, list):
        for p in ports:
            if hasattr(p, "name") and hasattr(p, "port_type"):
                result[p.name] = p.port_type.value
            elif hasattr(p, "port_type"):
                # SHOULDNT GO IN HERE BUT T O BE SURE
                fallback_name = f"port_{len(result)}"
                result[fallback_name] = p.port_type.value
                
    return result


def get_config_schema_json(node_cls):
    """Extracts the JSON Schema from the Pydantic config model."""
    schema_model = node_cls.get_config_schema()
    if schema_model:
        if hasattr(schema_model, "model_json_schema"):
            return schema_model.model_json_schema()
        elif hasattr(schema_model, "schema"):
            return schema_model.schema()
    return {}

# --- ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "running", "nodes_registered": len(NodeRegistry._nodes)}

@app.get("/registry")
def get_node_registry():
    registry_data = []
    
    for node_type, node_cls in NodeRegistry._nodes.items():
        meta = node_cls.get_metadata()
        
        # Helper correctly handles the Dict returned by get_input_ports
        inputs = port_to_dict(node_cls.get_input_ports())
        outputs = port_to_dict(node_cls.get_output_ports())
        
        config_schema = get_config_schema_json(node_cls)

        registry_data.append({
            "type": node_type,
            "label": meta.display_name,
            "category": meta.category,
            "description": meta.description,
            "inputs": inputs,
            "outputs": outputs,
            "default_config": config_schema,
            "color": meta.color,
            "icon": meta.icon
        })
        
    return registry_data

@app.post("/execute")
async def execute_graph(payload: GraphExecutionPayload):
    print(f"Received execution request for target: {payload.target_node_id}")
    try:
        graph = NodeGraph()
        
        for n in payload.nodes:
            try:
                graph.add_node(node_type=n.type, node_id=n.id, config=n.config)
            except ValueError as e:
                print(f"Warning: Skipping unknown node type '{n.type}': {e}")
                continue

        for c in payload.connections:
            try:
                graph.connect(c.source_node, c.source_port, c.target_node, c.target_port)
            except ValueError as e:
                print(f"Warning: Connection failed: {e}")

        result = graph.execute(output_node=payload.target_node_id)
        
        return {
            "status": "success", 
            "message": "Graph executed successfully",
            "result_summary": str(result)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import pydantic

    uvicorn.run(app, host="0.0.0.0", port=8000)
