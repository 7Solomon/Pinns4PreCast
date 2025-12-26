import math
import os
import json
import time
import csv
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from src.node_system.configs.logger import LoggerConfig
from src.node_system.core import Node, NodeMetadata, Port, PortType, register_node

from src.node_system.event_bus import get_event_bus, Event, EventType
import asyncio




@register_node("dashboard_logger")
class DashboardLoggerNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            #Port("graph", PortType.ANY, description="NodeGraph instance to save"),
            Port("logger_config", PortType.CONFIG, description="Config for logger so save dir and stuff"),

        ]

    @classmethod
    def get_output_ports(cls):
        return [
            Port("logger", PortType.LOGGER),
        ]
    
    @classmethod
    def get_metadata(cls):
        return NodeMetadata(
            category="Logger",
            display_name="Dashboard Logger",
            description="This crestaes the run ID",
            icon="network-wired"
        )


    @classmethod
    def get_config_schema(cls):
        return LoggerConfig

    def execute(self):
        cfg = self.inputs.get("logger_config") or self.config
        
        # 1. Create/Get Run Directory
        version_name = self.context.get("run_id")
        
        run_path = os.path.join(cfg.save_dir, version_name)
        
        # 2. Save Graph if provided
        if cfg.save_graph and "graph" in self.inputs:
            graph = self.inputs["graph"]
            if graph:
                graph_path = os.path.join(run_path, "graph.json")
                config_path = os.path.join(run_path, "graph_config.json")
                
                graph.save_to_file(graph_path, metadata={
                    "run_id": version_name,
                    "purpose": "training"
                })
                
                config_snapshot = graph.get_config_snapshot()
                with open(config_path, 'w') as f:
                    json.dump(config_snapshot, f, indent=2)
        
        # 3. Create Logger
        logger = DashboardLogger(save_dir=cfg.save_dir, version=version_name)
        
        return {
            "logger": logger,
        }


class DashboardLogger(Logger):
    """
    Logger that publishes real-time events to the event bus.
    Frontend subscribes via WebSocket instead of polling.
    """
    
    def __init__(self, save_dir, version=None):
        super().__init__()
        self._save_dir = save_dir
        self._version = version or "0"
        self._experiment_name = "dashboard_logs"
        
        # Create Directory
        os.makedirs(self.log_dir, exist_ok=True)
        self.status_file = os.path.join(self.log_dir, 'status.json')
        self.metrics_file = os.path.join(self.log_dir, 'metrics.csv')
        
        self.fieldnames = [
            'step', 'epoch', 'timestamp',
            'loss_physics', 'loss_bc', 'loss_ic',
            'loss', 'loss_step', 'loss_epoch',
            'loss_phys_temperature', 'loss_phys_alpha',
            'loss_bc_temperature', 
            'loss_ic_temperature', 'loss_ic_alpha',
            'val_loss_physics', 'val_loss_bc', 'val_loss_ic',
            'val_loss' 
        ]
        
        self.event_bus = get_event_bus()
        self._update_status("initialized")
        
        # Publish initialization event
        self._publish_event(EventType.TRAINING_STARTED, {
            "status": "initialized",
            "log_dir": self.log_dir
        })

    @property
    def name(self):
        return self._experiment_name

    @property
    def version(self):
        return self._version

    @property
    def log_dir(self):
        return os.path.join(self._save_dir, self.version)

    @rank_zero_only
    def log_hyperparams(self, params):
        params_file = os.path.join(self.log_dir, 'hparams.json')
        with open(params_file, 'w') as f:
            json.dump(params, f, default=str, indent=4)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """
        Log metrics to CSV AND publish event for real-time updates.
        """
        # Filter for scalars only
        row = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    row[k] = None
                else:
                    row[k] = float(v)

        row['step'] = step
        row['timestamp'] = time.time()

        current_epoch = metrics.get('epoch', None)        
        current_loss = metrics.get('loss_step', metrics.get('loss_epoch', metrics.get('loss', None)))

        # Write to CSV (for historical data)
        file_exists = os.path.isfile(self.metrics_file)
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=self.fieldnames, 
                extrasaction='ignore', 
                restval=''
            )
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)

        # Update status file
        self._update_status("running", epoch=current_epoch, loss=current_loss)
        
        # 🚀 PUBLISH EVENT - This replaces polling!
        self._publish_event(EventType.METRICS_UPDATED, {
            "epoch": current_epoch,
            "step": step,
            "metrics": row
        })

    @rank_zero_only
    def finalize(self, status):
        self._update_status(status)
        print("completed? :", status)
        # Publish completion event
        event_type = EventType.TRAINING_COMPLETED if status == "completed" else EventType.TRAINING_STOPPED
        self._publish_event(event_type, {"status": status})

    def _update_status(self, status, epoch=None, loss=None):
        """Update status file (kept for historical compatibility)"""
        data = {}
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
            except:
                pass

        data['id'] = self.version
        data['status'] = status
        data['last_update'] = time.time()
        
        if epoch is not None:
            data['epoch'] = int(epoch)
        if loss is not None:
            data['loss'] = float(loss)

        temp = self.status_file + '.tmp'
        try:
            with open(temp, 'w') as f:
                json.dump(data, f, indent=4)
            os.replace(temp, self.status_file)
        except Exception as e:
            print(f"Status update failed: {e}")
    
    def _publish_event(self, event_type: EventType, data: dict):
        """
        Publish an event to the event bus.
        Uses asyncio to handle async event publishing from sync context.
        """
        event = Event(
            type=event_type,
            run_id=self.version,
            data=data
        )
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule coroutine
                asyncio.ensure_future(self.event_bus.publish(event))
            else:
                # If loop is not running, run until complete
                loop.run_until_complete(self.event_bus.publish(event))
        except RuntimeError:
            # No event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.event_bus.publish(event))