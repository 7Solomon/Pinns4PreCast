from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.node_system.event_bus import get_event_bus
import json

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/monitor/{run_id}")
async def websocket_monitor(websocket: WebSocket, run_id: str):
    """
    Real-time monitoring WebSocket for a specific run.
    
    Usage from frontend:
    ```javascript
    const ws = new WebSocket(`ws://localhost:8000/ws/monitor/${runId}`);
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // Update UI based on event.type
    };
    ```
    """
    await websocket.accept()
    event_bus = get_event_bus()
    
    # Register this WebSocket for the run
    event_bus.register_websocket(run_id, websocket)
    
    try:
        # Send initial confirmation
        await websocket.send_json({
            "type": "connection_established",
            "run_id": run_id,
            "message": "Connected to real-time monitoring"
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (e.g., ping, or subscription updates)
                data = await websocket.receive_text()
                
                # Handle client requests
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass
                    
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        print(f"WebSocket error for run {run_id}: {e}")
    finally:
        # Cleanup on disconnect
        event_bus.unregister_websocket(run_id, websocket)
        print(f"WebSocket disconnected for run: {run_id}")


@router.websocket("/monitor-all")
async def websocket_monitor_all(websocket: WebSocket):
    """
    Monitor ALL active runs.
    Useful for a dashboard showing multiple runs.
    """
    await websocket.accept()
    event_bus = get_event_bus()
    
    # Register for all runs using wildcard
    event_bus.register_websocket("*", websocket)
    
    try:
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to all runs monitoring"
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except WebSocketDisconnect:
                break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        event_bus.unregister_websocket("*", websocket)