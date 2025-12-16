import os
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException
from src.api.models import SaveGraphPayload, GraphExecutionPayload
from src.node_system.core import NodeGraph

router = APIRouter(prefix="/graphs", tags=["graphs"])

TEMPLATES_DIR = "content/tree_templates"

@router.post("/save")
async def save_graph(payload: SaveGraphPayload):
    # Sanitize filename
    safe_name = "".join(c for c in payload.name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    filename = f"{safe_name}.json"
    filepath = os.path.join(TEMPLATES_DIR, filename)
    
    if os.path.exists(filepath) and not payload.overwrite:
        raise HTTPException(
            status_code=409, 
            detail=f"Graph '{payload.name}' already exists. Set overwrite=true to replace it."
        )
    
    now = datetime.now().isoformat()
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
        "nodes": [n.model_dump() for n in payload.nodes],
        "connections": [c.model_dump() for c in payload.connections],
        "created_at": created_at,
        "updated_at": now,
        "version": "1.0"
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Graph '{payload.name}' saved successfully",
            "filename": filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save graph: {str(e)}")

@router.get("/list")
def list_saved_graphs():
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
    
    graphs.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return {"graphs": graphs, "total": len(graphs)}

@router.get("/load/{filename}")
def load_graph(filename: str):
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    filepath = os.path.join(TEMPLATES_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Graph '{filename}' not found")
    
    try:
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        return {"status": "success", "graph": graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load graph: {str(e)}")

@router.delete("/delete/{filename}")
def delete_graph(filename: str):
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    filepath = os.path.join(TEMPLATES_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Graph '{filename}' not found")
    
    try:
        os.remove(filepath)
        return {"status": "success", "message": f"Graph '{filename}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete graph: {str(e)}")

@router.post("/duplicate/{filename}")
def duplicate_graph(filename: str, new_name: str):
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    source_path = os.path.join(TEMPLATES_DIR, filename)
    
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail=f"Source graph '{filename}' not found")
    
    safe_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
    new_filename = f"{safe_name}.json"
    dest_path = os.path.join(TEMPLATES_DIR, new_filename)
    
    if os.path.exists(dest_path):
        raise HTTPException(status_code=409, detail=f"Graph '{new_name}' already exists")
    
    try:
        with open(source_path, 'r') as f:
            graph_data = json.load(f)
        
        now = datetime.now().isoformat()
        graph_data['name'] = new_name
        graph_data['description'] = f"Copy of {graph_data.get('name', filename[:-5])}"
        graph_data['created_at'] = now
        graph_data['updated_at'] = now
        
        with open(dest_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
            
        return {"status": "success", "message": f"Graph duplicated as '{new_name}'", "filename": new_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to duplicate graph: {str(e)}")

@router.post("/export-config")
async def export_graph_config(payload: GraphExecutionPayload):
    try:
        graph = NodeGraph()
        for n in payload.nodes:
            try:
                graph.add_node(node_type=n.type, node_id=n.id, config=n.config)
            except ValueError:
                continue
        for c in payload.connections:
            try:
                graph.connect(c.source_node, c.source_port, c.target_node, c.target_port)
            except ValueError:
                continue
                
        return {"status": "success", "config": graph.get_config_snapshot()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
