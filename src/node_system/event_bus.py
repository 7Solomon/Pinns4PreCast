import asyncio
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class EventType(str, Enum):
    """Types of events in the system"""
    TRAINING_STARTED = "training_started"
    TRAINING_EPOCH_END = "training_epoch_end"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_STOPPED = "training_stopped"
    METRICS_UPDATED = "metrics_updated"
    SENSOR_DATA_UPDATED = "sensor_data_updated"
    CHECKPOINT_SAVED = "checkpoint_saved"
    RUN_STATUS_CHANGED = "run_status_changed"


@dataclass
class Event:
    """Represents a system event"""
    type: EventType
    run_id: str
    data: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()
    
    def to_dict(self):
        return {
            "type": self.type.value,
            "run_id": self.run_id,
            "data": self.data,
            "timestamp": self.timestamp
        }


class EventBus:
    """
    Central event bus for system-wide event distribution.
    Supports both synchronous callbacks and WebSocket broadcasting.
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._websocket_clients: Dict[str, List[Any]] = {}  # run_id -> [websockets]
        
    def subscribe(self, event_type: EventType, callback: Callable):
        """Register a callback for an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Remove a callback"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)
    
    async def publish(self, event: Event):
        """
        Publish an event to all subscribers and WebSocket clients.
        This is the core method that replaces polling.
        """
        # 1. Notify synchronous callbacks
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    print(f"Error in event callback: {e}")
        
        # 2. Broadcast to WebSocket clients
        await self._broadcast_to_websockets(event)
    
    async def _broadcast_to_websockets(self, event: Event):
        """Send event to all WebSocket clients subscribed to this run"""
        # Send to run-specific subscribers
        if event.run_id in self._websocket_clients:
            dead_clients = []
            for ws in self._websocket_clients[event.run_id]:
                try:
                    await ws.send_json(event.to_dict())
                except Exception:
                    dead_clients.append(ws)
            
            # Clean up dead connections
            for ws in dead_clients:
                self._websocket_clients[event.run_id].remove(ws)
        
        # Send to clients subscribed to ALL runs
        if "*" in self._websocket_clients:
            for ws in self._websocket_clients["*"]:
                try:
                    await ws.send_json(event.to_dict())
                except Exception:
                    pass
    
    def register_websocket(self, run_id: str, websocket):
        """Register a WebSocket client for a specific run"""
        if run_id not in self._websocket_clients:
            self._websocket_clients[run_id] = []
        self._websocket_clients[run_id].append(websocket)
        print(f"WebSocket registered for run: {run_id}")
    
    def unregister_websocket(self, run_id: str, websocket):
        """Unregister a WebSocket client"""
        if run_id in self._websocket_clients:
            if websocket in self._websocket_clients[run_id]:
                self._websocket_clients[run_id].remove(websocket)
                print(f"WebSocket unregistered for run: {run_id}")


# Global singleton instance
_event_bus = EventBus()

def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    return _event_bus