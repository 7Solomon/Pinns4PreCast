from flask import Blueprint, request, jsonify
from src.node_system.graph_library import GraphLibrary
from src.node_system.core import NodeGraph

graphs_bp = Blueprint('graphs', __name__, url_prefix='/api/graphs')
library = GraphLibrary()


@graphs_bp.route('/templates', methods=['GET'])
def list_templates():
    """Get all saved graph templates."""
    return jsonify(library.list_templates())

@graphs_bp.route('/templates/<name>', methods=['GET'])
def get_template(name):
    """Get a specific template."""
    try:
        graph = library.load_template(name)
        return jsonify(graph.to_dict())
    except FileNotFoundError:
        return jsonify({"error": "Template not found"}), 404

@graphs_bp.route('/templates/<name>', methods=['POST'])
def save_template(name):
    """Save a new template."""
    data = request.get_json()
    
    graph = NodeGraph.from_dict(data['graph'])
    filepath = library.save_template(
        graph, 
        name, 
        description=data.get('description', ''),
        tags=data.get('tags', [])
    )
    
    return jsonify({"message": "Template saved", "path": filepath})

@graphs_bp.route('/templates/<name>', methods=['DELETE'])
def delete_template(name):
    """Delete a template."""
    library.delete_template(name)
    return jsonify({"message": "Template deleted"})

# === Run Graphs ===

@graphs_bp.route('/runs/<run_id>/graph', methods=['GET'])
def get_run_graph(run_id):
    """Get the graph that produced a run."""
    graph = library.get_run_graph(run_id)
    if not graph:
        return jsonify({"error": "No graph found for this run"}), 404
    return jsonify(graph.to_dict())

@graphs_bp.route('/runs/<run_id>/config', methods=['GET'])
def get_run_config(run_id):
    """Get the configuration snapshot for a run."""
    config = library.get_run_config(run_id)
    if not config:
        return jsonify({"error": "No config found for this run"}), 404
    return jsonify(config)

@graphs_bp.route('/runs', methods=['GET'])
def list_runs_with_graphs():
    """List all runs that have graphs."""
    return jsonify(library.list_runs_with_graphs())

# === Utilities ===

@graphs_bp.route('/clone', methods=['POST'])
def clone_graph():
    """Clone a graph to create a new template."""
    data = request.get_json()
    library.clone_graph(
        source=data['source'],
        dest=data['dest'],
        is_template=data.get('is_template', True)
    )
    return jsonify({"message": "Graph cloned"})