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
    
    # 1. Quick Check (Fast fail)
    if session_state.EXECUTION_LOCK:
        raise HTTPException(status_code=409, detail="Session running.")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    global_context = {"run_id": run_id}

    def run_pipeline_safely():
        try:
            with session_state.execution_lock_guard():
                
                graph = NodeGraph()
                
                # Callback to update global state
                def status_cb(node_id, status, error=None):
                    session_state.update_node_status(run_id, node_id, status, error)

                # Add nodes/connections (same as before)
                for n in payload.nodes:
                    graph.add_node(n.type, n.id, n.config)
                for c in payload.connections:
                    graph.connect(c.source_node, c.source_port, c.target_node, c.target_port)

                # Execute
                graph.execute(
                    output_node=payload.target_node_id,
                    context=global_context,
                    status_callback=status_cb
                )
        
        except RuntimeError as lock_err:
            print(f"Skipped run: {lock_err}")
        except Exception as e:
            print(f"CRITICAL GRAPH FAILURE: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up stop flags
            session_state.clear_stop_request(run_id)

    # 3. Dispatch
    background_tasks.add_task(run_pipeline_safely)

    return {
        "status": "started",
        "run_id": run_id, 
        "message": "Graph started in background"
    }

@router.post("/execute/stop/{run_id}")
def stop_execution(run_id: str):
    session_state.request_stop(run_id)
    return {"message": f"Stop requested for {run_id}"}