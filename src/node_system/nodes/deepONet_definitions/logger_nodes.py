import os
import json
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from src.node_system.core import Node, Port, PortType, register_node
from src.DeepONet.logger import DashboardLogger

def get_new_run(save_dir: str, status_file_name: str = "status.json") -> str:
    """Generates a timestamp ID and initializes the run directory."""
    now = datetime.now()
    timestamp_id = now.strftime("%Y-%m-%d_%H-%M-%S")
    pretty_date = now.strftime("%Y-%m-%d %H:%M:%S")

    run_path = os.path.join(save_dir, timestamp_id)

    os.makedirs(run_path, exist_ok=True)
    #os.makedirs(os.path.join(run_path, 'checkpoints'), exist_ok=True)
    #os.makedirs(os.path.join(run_path, 'vtk'), exist_ok=True)

    # 2. Initialize Metadata
    initial_status = {
        "id": timestamp_id,
        "status": "initializing",
        "start_time": pretty_date,
        "epoch": 0,
        "loss": None
    }
        
    with open(os.path.join(run_path, status_file_name), 'w') as f:
        json.dump(initial_status, f, indent=4)

    return timestamp_id

class LoggerConfig(BaseModel):
    save_dir: str = Field(default="content/runs", title="Runs Directory")
    version: str | None = Field(default=None, title="Run Name (auto-generate if empty)")
    save_graph: bool = Field(default=True, title="Save graph.json to run directory")

@register_node("dashboard_logger")
class DashboardLoggerNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("graph", PortType.ANY, required=False, description="NodeGraph instance to save")
        ]

    @classmethod
    def get_output_ports(cls):
        return [
            Port("logger", "logger"),
            Port("run_path", PortType.CONFIG, description="Path to run directory")
        ]

    @classmethod
    def get_config_schema(cls):
        return LoggerConfig

    def execute(self):
        cfg = self.config
        
        # 1. Create/Get Run Directory
        version_name = cfg.version
        if not version_name:
            version_name = get_new_run(cfg.save_dir)
        
        run_path = os.path.join(cfg.save_dir, version_name)
        
        # 2. Save Graph if provided
        if cfg.save_graph and "graph" in self.inputs:
            graph = self.inputs["graph"]
            if graph:
                graph_path = os.path.join(run_path, "graph.json")
                config_path = os.path.join(run_path, "graph_config.json")
                
                # Save structure
                graph.save_to_file(graph_path, metadata={
                    "run_id": version_name,
                    "purpose": "training"
                })
                
                # Save resolved configs
                config_snapshot = graph.get_config_snapshot()
                with open(config_path, 'w') as f:
                    json.dump(config_snapshot, f, indent=2)
        
        # 3. Create Logger
        logger = DashboardLogger(save_dir=cfg.save_dir, version=version_name)
        
        return {
            "logger": logger,
            "run_path": run_path
        }