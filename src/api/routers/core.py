from fastapi import APIRouter, HTTPException
from src.node_system.core import NodeGraph, NodeRegistry
from src.api.models import GraphExecutionPayload
from src.api.utils import port_to_dict, get_config_schema_json

router = APIRouter(tags=["core"])

@router.get("/")
def health_check():
    return {"status": "running", "nodes_registered": len(NodeRegistry._nodes)}

@router.get("/registry")
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

@router.post("/execute")
async def execute_graph(payload: GraphExecutionPayload):
    print(f"Received execution request for target: {payload.target_node_id}")
    try:
        graph = NodeGraph()
        # Build graph
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

        print(graph.to_dict())
        result = graph.execute(output_node=payload.target_node_id)
        
        run_id = None
        widget_specs = []
        
        for node_id, node in graph.nodes.items():
            if hasattr(node, 'outputs') and 'run_id' in node.outputs:
                run_id = node.outputs['run_id']
            
            if hasattr(node, 'outputs') and 'widget_spec' in node.outputs:
                widget_specs.append(node.outputs['widget_spec'])
        
        return {
            "status": "success",
            "message": "Graph executed successfully",
            "run_id": run_id,
            "widgets": widget_specs,
            "result_summary": str(result)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
