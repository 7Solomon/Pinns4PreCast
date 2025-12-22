import asyncio
import os

import pandas as pd
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.node_system.event_bus import get_event_bus
import json

router = APIRouter(prefix="/ws", tags=["websocket"])

@router.websocket("/monitor/{run_id}")
async def websocket_monitor(websocket: WebSocket, run_id: str):
    try:
        await websocket.accept()
        event_bus = get_event_bus()
        event_bus.register_websocket(run_id, websocket)

        await websocket.send_json({
            "type": "connection_established",
            "run_id": run_id,
            "message": "Connected to real-time monitoring"
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif message.get("type") == "request_history_since":
                    last_step = message.get("last_step", 0)
                    
                    # MAYBE HERE SMARTER
                    metrics_path = f"content/runs/{run_id}/metrics.csv"
                    if os.path.exists(metrics_path):
                        df = pd.read_csv(metrics_path)
                        # Filter step > last_step
                        recent_df = df[df['step'] > last_step]
                        metrics = recent_df.fillna('').to_dict('records')
                    else:
                        metrics = []
                    
                    await websocket.send_json({
                        "type": "metrics_updates_since",
                        "run_id": run_id,
                        "data": metrics
                    })
                
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        print(f"WebSocket error for run {run_id}: {e}")
    finally:
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