from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import pandas as pd
from datetime import datetime

from src.node_system.core import NodeGraph, NodeRegistry, Port

import src.node_system.registry

app = FastAPI(title="PINNs Node Editor API", version="1.0")
TEMPLATES_DIR = "content/tree_templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

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

# --- HELPER FUNCTIONS ---

def port_to_dict(ports: Any) -> Dict[str, str]:
    """Robust converter that handles both List[Port] and Dict[str, Port]."""
    result = {}
    
    if isinstance(ports, dict):
        for name, port_obj in ports.items():
            if hasattr(port_obj, "port_type"):
                result[name] = port_obj.port_type.value
            else:
                result[name] = "any"
    elif isinstance(ports, list):
        for p in ports:
            if hasattr(p, "name") and hasattr(p, "port_type"):
                result[p.name] = p.port_type.value
            elif hasattr(p, "port_type"):
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

# --- BASIC ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "running", "nodes_registered": len(NodeRegistry._nodes)}

@app.get("/registry")
def get_node_registry():
    registry_data = []
    
    for node_type, node_cls in NodeRegistry._nodes.items():
        meta = node_cls.get_metadata()
        
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


# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@app.get("/monitor/status/{run_id}")
def get_training_status(run_id: str):
    """
    Returns current training status for a specific run.
    Frontend polls this endpoint every second for live updates.
    """
    status_path = f"content/runs/{run_id}/status.json"
    
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    
    try:
        with open(status_path, 'r') as f:
            status = json.load(f)
        return status
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Corrupted status file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitor/metrics/{run_id}")
def get_training_metrics(run_id: str, limit: int = 100):
    """
    Returns training metrics (loss curves, etc.)
    
    Args:
        run_id: The run identifier
        limit: Maximum number of recent records to return
    """
    metrics_path = f"content/runs/{run_id}/metrics.csv"
    
    if not os.path.exists(metrics_path):
        return {
            "metrics": [],
            "total_epochs": 0,
            "latest_epoch": 0,
            "message": "No metrics available yet"
        }
    
    try:
        df = pd.read_csv(metrics_path)
        
        # Handle empty CSV
        if len(df) == 0:
            return {
                "metrics": [],
                "total_epochs": 0,
                "latest_epoch": 0
            }
        
        # Get last N rows
        recent = df.tail(limit)
        
        # Convert to list of dicts, replacing NaN with None
        metrics = recent.fillna('').to_dict('records')
        
        # Get summary stats
        latest_epoch = int(df['epoch'].max()) if 'epoch' in df else 0
        total_records = len(df)
        
        return {
            "metrics": metrics,
            "total_records": total_records,
            "latest_epoch": latest_epoch,
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")


@app.get("/monitor/runs")
def list_active_runs():
    """
    Lists all runs with their current status.
    Useful for showing a dashboard of all training runs.
    """
    runs_dir = "content/runs"
    
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir, exist_ok=True)
        return {"runs": [], "total": 0}
    
    runs = []
    for run_id in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_id)
        
        # Skip files
        if not os.path.isdir(run_path):
            continue
            
        status_path = os.path.join(run_path, "status.json")
        
        if os.path.exists(status_path):
            try:
                with open(status_path, 'r') as f:
                    status = json.load(f)
                    status['run_id'] = run_id  # Ensure run_id is included
                    runs.append(status)
            except Exception as e:
                print(f"Warning: Could not read status for {run_id}: {e}")
                runs.append({
                    "run_id": run_id,
                    "status": "unknown",
                    "error": str(e)
                })
    
    # Sort by start time (most recent first)
    runs.sort(key=lambda x: x.get('start_time', ''), reverse=True)
    
    return {
        "runs": runs,
        "total": len(runs)
    }


@app.delete("/monitor/runs/{run_id}")
def delete_run(run_id: str):
    """Delete a training run and all its files."""
    import shutil
    
    run_path = f"content/runs/{run_id}"
    
    if not os.path.exists(run_path):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    
    try:
        shutil.rmtree(run_path)
        return {"message": f"Run '{run_id}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete run: {str(e)}")


@app.get("/monitor/visualizations/{run_id}")
def get_visualization_files(run_id: str):
    """
    Lists available visualization files (CSV exports) for a run.
    """
    run_path = f"content/runs/{run_id}"
    
    if not os.path.exists(run_path):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    
    visualizations = {
        "sensors_temp": [],
        "sensors_alpha": [],
        "vtk_files": []
    }
    
    # Check for sensor CSV files
    temp_dir = os.path.join(run_path, "sensors_temp")
    alpha_dir = os.path.join(run_path, "sensors_alpha")
    vtk_dir = os.path.join(run_path, "vtk_output")
    
    if os.path.exists(temp_dir):
        visualizations["sensors_temp"] = sorted([
            f for f in os.listdir(temp_dir) if f.endswith('.csv')
        ])
    
    if os.path.exists(alpha_dir):
        visualizations["sensors_alpha"] = sorted([
            f for f in os.listdir(alpha_dir) if f.endswith('.csv')
        ])
    
    if os.path.exists(vtk_dir):
        visualizations["vtk_files"] = sorted([
            f for f in os.listdir(vtk_dir) if f.endswith('.vtk')
        ])
    
    return visualizations


@app.post("/graphs/save")
async def save_graph(payload: SaveGraphPayload):
    """
    Save a graph as a template.
    
    Args:
        payload: Graph data with name, nodes, connections, metadata
    
    Returns:
        Success message with filename
    """
    # Sanitize filename
    safe_name = "".join(c for c in payload.name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    filename = f"{safe_name}.json"
    filepath = os.path.join(TEMPLATES_DIR, filename)
    
    # Check if file exists
    if os.path.exists(filepath) and not payload.overwrite:
        raise HTTPException(
            status_code=409, 
            detail=f"Graph '{payload.name}' already exists. Set overwrite=true to replace it."
        )
    
    # Prepare graph data
    now = datetime.now().isoformat()
    
    # Load existing data to preserve created_at if updating
    created_at = now
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                existing = json.load(f)
                created_at = existing.get('created_at', now)
        except:
            pass
    
    graph_data = {
        "name": payload.name,
        "description": payload.description,
        "tags": payload.tags,
        "nodes": [
            {
                "id": n.id,
                "type": n.type,
                "config": n.config,
                "position": n.position
            }
            for n in payload.nodes
        ],
        "connections": [
            {
                "source_node": c.source_node,
                "source_port": c.source_port,
                "target_node": c.target_node,
                "target_port": c.target_port
            }
            for c in payload.connections
        ],
        "created_at": created_at,
        "updated_at": now,
        "version": "1.0"
    }
    
    # Save to file
    try:
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Graph '{payload.name}' saved successfully",
            "filename": filename,
            "filepath": filepath
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save graph: {str(e)}")


@app.get("/graphs/list")
def list_saved_graphs():
    """
    List all saved graph templates with metadata.
    
    Returns:
        List of graph metadata objects
    """
    if not os.path.exists(TEMPLATES_DIR):
        return {"graphs": [], "total": 0}
    
    graphs = []
    
    for filename in os.listdir(TEMPLATES_DIR):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(TEMPLATES_DIR, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            graphs.append({
                "name": data.get("name", filename[:-5]),
                "filename": filename,
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "node_count": len(data.get("nodes", [])),
                "connection_count": len(data.get("connections", [])),
            })
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")
            continue
    
    # Sort by updated_at (most recent first)
    graphs.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    
    return {
        "graphs": graphs,
        "total": len(graphs)
    }


@app.get("/graphs/load/{filename}")
def load_graph(filename: str):
    """
    Load a saved graph template.
    
    Args:
        filename: Name of the graph file (with or without .json)
    
    Returns:
        Complete graph data
    """
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    filepath = os.path.join(TEMPLATES_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Graph '{filename}' not found")
    
    try:
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        return {
            "status": "success",
            "graph": graph_data
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid graph file format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load graph: {str(e)}")


@app.delete("/graphs/delete/{filename}")
def delete_graph(filename: str):
    """
    Delete a saved graph template.
    
    Args:
        filename: Name of the graph file to delete
    
    Returns:
        Success message
    """
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    filepath = os.path.join(TEMPLATES_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Graph '{filename}' not found")
    
    try:
        os.remove(filepath)
        return {
            "status": "success",
            "message": f"Graph '{filename}' deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete graph: {str(e)}")


@app.post("/graphs/duplicate/{filename}")
def duplicate_graph(filename: str, new_name: str):
    """
    Create a copy of an existing graph with a new name.
    
    Args:
        filename: Source graph filename
        new_name: Name for the duplicate
    
    Returns:
        Success message with new filename
    """
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    source_path = os.path.join(TEMPLATES_DIR, filename)
    
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail=f"Source graph '{filename}' not found")
    
    # Create new filename
    safe_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    new_filename = f"{safe_name}.json"
    dest_path = os.path.join(TEMPLATES_DIR, new_filename)
    
    if os.path.exists(dest_path):
        raise HTTPException(status_code=409, detail=f"Graph '{new_name}' already exists")
    
    try:
        # Load original
        with open(source_path, 'r') as f:
            graph_data = json.load(f)
        
        # Update metadata
        now = datetime.now().isoformat()
        graph_data['name'] = new_name
        graph_data['description'] = f"Copy of {graph_data.get('name', filename[:-5])}"
        graph_data['created_at'] = now
        graph_data['updated_at'] = now
        
        # Save copy
        with open(dest_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Graph duplicated as '{new_name}'",
            "filename": new_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to duplicate graph: {str(e)}")


# ============================================================================
# HELPER ENDPOINT: Export graph config for debugging
# ============================================================================

@app.post("/graphs/export-config")
async def export_graph_config(payload: GraphExecutionPayload):
    """
    Export the full configuration snapshot of a graph.
    Useful for debugging and reproducing exact settings.
    """
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

        # Get resolved configuration
        config_snapshot = graph.get_config_snapshot()
        
        return {
            "status": "success",
            "config": config_snapshot
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)