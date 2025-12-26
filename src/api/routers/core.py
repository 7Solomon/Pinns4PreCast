from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.state import app_state
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

@router.get("/execute/status")
def get_execution_status():
    return {
        "is_running": app_state.is_running,
        "run_id": app_state.current_run_id
    }

@router.post("/execute")
async def execute_graph(payload: GraphExecutionPayload, background_tasks: BackgroundTasks):
    
    if app_state.is_running:
        raise HTTPException(
            status_code=409, 
            detail=f"A training session ({app_state.current_run_id}) is already active."
        )

    try:
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        app_state.register_session(run_id, trainer=None)

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

        # DEFINE WRAPPER
        def protected_execute(*args, **kwargs):
            try:
                graph.execute(*args, **kwargs)
            except Exception as e:
                print(f"❌ Execution failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print(f"🔓 Execution finished. Clearing session {run_id}")
                app_state.clear_session(run_id)

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
        print("🔓 Setup failed, releasing state")
        app_state.clear_session(run_id) 
        
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/execute/stop/{run_id}")
async def stop_execution(run_id: str):
    if app_state.current_run_id != run_id:
        return {"message": "Run already finished or ID mismatch"}

    success = app_state.stop_current_session()
    if success:
        return {"message": "Stop signal sent to trainer"}
    else:
        raise HTTPException(status_code=404, detail="No active trainer found")