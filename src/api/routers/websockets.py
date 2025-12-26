import asyncio
import os
import numpy as np
import pandas as pd
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.node_system.event_bus import get_event_bus
import json


router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/monitor/{run_id}")
async def websocket_monitor(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for monitoring a specific training run.
    Handles history requests + live event bus updates.
    """
    event_bus = get_event_bus()
    
    try:
        await websocket.accept()
        event_bus.register_websocket(run_id, websocket)

        # Send connection confirmation
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
                    
                    metrics_path = f"content/runs/{run_id}/metrics.csv"
                    if os.path.exists(metrics_path):
                        df = pd.read_csv(metrics_path)
                        recent_df = df[df['step'] > last_step]
                        metrics = recent_df.replace({np.nan: None}).to_dict('records')
                        
                        await websocket.send_json({
                            "type": "metrics_updates_since",
                            "run_id": run_id,
                            "data": metrics
                        })
                    else:
                        await websocket.send_json({
                            "type": "metrics_updates_since",
                            "run_id": run_id,
                            "data": []
                        })
                
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                print(f"[WS] Invalid JSON from {run_id}")
            except Exception as e:
                print(f"[WS] Error processing message for {run_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except Exception as e:
        print(f"WebSocket setup failed for {run_id}: {e}")
    finally:
        event_bus.unregister_websocket(run_id, websocket)
        print(f"WebSocket disconnected for run: {run_id}")


@router.websocket("/monitor-all")
async def websocket_monitor_all(websocket: WebSocket):
    """
    Monitor ALL active runs using wildcard subscription.
    Useful for dashboard showing multiple runs.
    """
    event_bus = get_event_bus()
    
    try:
        await websocket.accept()
        event_bus.register_websocket("*", websocket)
        
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
            except json.JSONDecodeError:
                print("[WS-ALL] Invalid JSON")
            except Exception as e:
                print(f"[WS-ALL] Error: {e}")
                
    except Exception as e:
        print(f"[WS-ALL] Setup failed: {e}")
    finally:
        event_bus.unregister_websocket("*", websocket)
        print("WebSocket monitor-all disconnected")


@router.get("/ws/test/{run_id}")
async def test_history_endpoint(run_id: str, last_step: int = 0):
    """
    HTTP endpoint to test the WebSocket history logic.
    Returns same data as WebSocket request_history_since.
    """
    metrics_path = f"content/runs/{run_id}/metrics.csv"
    if not os.path.exists(metrics_path):
        return {"error": "Metrics file not found", "data": []}
    
    df = pd.read_csv(metrics_path)
    recent_df = df[df['step'] > last_step]
    metrics = recent_df.fillna('').to_dict('records')
    
    return {
        "success": True,
        "total_records": len(df),
        "records_since": len(metrics),
        "last_step": last_step,
        "data": metrics
    }
