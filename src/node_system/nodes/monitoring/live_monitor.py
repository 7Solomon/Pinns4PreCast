# src/node_system/nodes/monitoring/live_monitor.py

import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node


class MonitorConfig(BaseModel):
    """Configuration for live monitoring."""
    update_interval: float = Field(
        default=1.0,
        title="Update Interval (seconds)",
        description="How often to check for updates"
    )
    show_metrics: list[str] = Field(
        default_factory=lambda: ["loss", "epoch", "loss_physics", "loss_bc", "loss_ic"],
        title="Metrics to Display",
        description="Which metrics to show in the monitor"
    )


@register_node("live_training_monitor")
class LiveTrainingMonitorNode(Node):
    """
    A node that provides a live data feed during training.
    Frontend can poll the /monitor/{run_id} endpoint to get updates.
    """
    
    @classmethod
    def get_input_ports(cls) -> Dict[str, Port]:
        return {
            "run_path": Port(
                "run_path",
                PortType.RUN_ID,
                required=True,
                description="Path to the run directory (from logger)"
            ),
            "logger": Port(
                "logger",
                PortType.LOGGER,
                required=False,
                description="Optional logger reference"
            )
        }
    
    @classmethod
    def get_output_ports(cls) -> Dict[str, Port]:
        return {
            "monitor_config": Port(
                "monitor_config",
                PortType.CONFIG,
                description="Monitor configuration for frontend polling"
            )
        }
    
    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            category="Monitoring",
            display_name="Live Training Monitor",
            description="Provides real-time training metrics for visualization",
            icon="activity"
        )
    
    @classmethod
    def get_config_schema(cls):
        return MonitorConfig
    
    def execute(self) -> Dict[str, Any]:
        run_path = self.inputs["run_path"]
        config = self.config
        
        # Extract run_id from path
        run_id = os.path.basename(run_path)
        
        # Return monitor configuration
        monitor_info = {
            "run_id": run_id,
            "run_path": run_path,
            "status_file": os.path.join(run_path, getattr(config, "status_file_name", "status.json")),
            "metrics_file": os.path.join(run_path, getattr(config, "metric_file_name", "metrics.csv")), 
            "update_interval": config.update_interval,
            "show_metrics": config.show_metrics
        }
        
        return {"monitor_config": monitor_info}
