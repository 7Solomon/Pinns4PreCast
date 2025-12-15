import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from src.node_system.core import NodeGraph

class GraphLibrary:
    """Manages saved graph templates and run graphs."""
    
    def __init__(self, library_path: str = "content/graphs", runs_path: str = "content/runs"):
        self.library_path = library_path
        self.runs_path = runs_path
        os.makedirs(library_path, exist_ok=True)
    
    # === Template Graphs ===
    
    def save_template(self, graph: NodeGraph, name: str, description: str = "", tags: List[str] = None):
        """Save a graph as a reusable template."""
        filepath = os.path.join(self.library_path, f"{name}.json")
        
        metadata = {
            "description": description,
            "tags": tags or [],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        graph.save_to_file(filepath, metadata=metadata)
        return filepath
    
    def load_template(self, name: str) -> NodeGraph:
        """Load a graph template by name."""
        filepath = os.path.join(self.library_path, f"{name}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Template '{name}' not found")
        return NodeGraph.load_from_file(filepath)
    
    def list_templates(self) -> List[Dict]:
        """List all available graph templates."""
        templates = []
        for filename in os.listdir(self.library_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.library_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    templates.append({
                        "name": filename[:-5],  # Remove .json
                        "description": data.get("metadata", {}).get("description", ""),
                        "tags": data.get("metadata", {}).get("tags", []),
                        "created_at": data.get("created_at", "Unknown"),
                        "node_count": len(data.get("nodes", []))
                    })
                except:
                    pass
        return templates
    
    def delete_template(self, name: str):
        """Delete a graph template."""
        filepath = os.path.join(self.library_path, f"{name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
    
    # === Run Graphs ===
    
    def get_run_graph(self, run_id: str) -> Optional[NodeGraph]:
        """Load the graph that produced a specific run."""
        graph_path = os.path.join(self.runs_path, run_id, "graph.json")
        if not os.path.exists(graph_path):
            return None
        return NodeGraph.load_from_file(graph_path)
    
    def get_run_config(self, run_id: str) -> Optional[Dict]:
        """Get the configuration snapshot for a run."""
        config_path = os.path.join(self.runs_path, run_id, "graph_config.json")
        if not os.path.exists(config_path):
            return None
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def list_runs_with_graphs(self) -> List[Dict]:
        """List all runs that have saved graphs."""
        runs = []
        for run_id in os.listdir(self.runs_path):
            graph_path = os.path.join(self.runs_path, run_id, "graph.json")
            status_path = os.path.join(self.runs_path, run_id, "status.json")
            
            if os.path.exists(graph_path):
                run_info = {"id": run_id, "has_graph": True}
                
                # Load status if available
                if os.path.exists(status_path):
                    with open(status_path, 'r') as f:
                        status = json.load(f)
                        run_info.update(status)
                
                runs.append(run_info)
        
        return sorted(runs, key=lambda x: x['id'], reverse=True)
    
    # === Utilities ===
    
    def clone_graph(self, source: str, dest: str, is_template: bool = True):
        """Clone a graph (template or run) to a new template."""
        if is_template:
            source_path = os.path.join(self.library_path, f"{source}.json")
        else:
            source_path = os.path.join(self.runs_path, source, "graph.json")
        
        dest_path = os.path.join(self.library_path, f"{dest}.json")
        
        # Load and re-save with new metadata
        graph = NodeGraph.load_from_file(source_path)
        graph.save_to_file(dest_path, metadata={
            "description": f"Cloned from {source}",
            "cloned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })