from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
import src.node_system.session as session_state 
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
async def execute_graph(payload: GraphExecutionPayload, background_tasks: BackgroundTasks):
    
    if session_state.EXECUTION_LOCK:
        raise HTTPException(status_code=409, detail="A training session is already running/queued.")
    print("🔒 Acquiring Execution Lock")
    session_state.EXECUTION_LOCK = True

    try:
        print(f"Received execution request...")
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Assigning Run ID: {run_id}")

        global_context = {
            "run_id": run_id,
        }

        # BUILD GRAPH
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

        print(graph.to_dict())
        # DEFINE WRAPPER
        def protected_execute(*args, **kwargs):
            try:
                graph.execute(*args, **kwargs)
            finally:
                print("🔓 Releasing Execution Lock (Task Finished)")
                session_state.EXECUTION_LOCK = False

        # FIRE AND FORGET
        background_tasks.add_task(
            protected_execute, 
            output_node=payload.target_node_id, 
            context=global_context
        )

        return {
            "status": "started",
            "message": "Graph started in background",
            "run_id": run_id, 
            "widgets": []
        }

    except Exception as e:
        print("🔓 Releasing Execution Lock (Error during setup)")
        session_state.EXECUTION_LOCK = False
        
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/execute/stop/{run_id}")
def stop_execution(run_id: str):
    success = session_state.stop_session(run_id)
    if success:
        return {"message": f"Stop signal sent to run {run_id}"}
    else:
        return {"message": "Run not found or already finished", "warning": True}
