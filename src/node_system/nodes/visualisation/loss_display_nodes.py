from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.node_system.core import Node, Port, PortType, register_node, NodeMetadata

class LossCurveConfig(BaseModel):
    title: str = Field(default="Training Loss", title="Chart Title")
    smoothing: float = Field(default=0.0, ge=0.0, le=1.0, title="Smoothing Factor")
    refresh_rate: int = Field(default=1000, title="Poll Interval (ms)")

@register_node("loss_curve")
class LossCurveNode(Node):
    """
    Does not 'run' training. 
    Instead, it generates a UI specification for the frontend to render.
    """

    @classmethod
    def get_input_ports(cls):
        return [
            Port("run_id", PortType.RUN_ID, description="Output from DashboardLogger"),
            Port("loss_curve_config", PortType.CONFIG, description="Config for Vis"),
        ]

    @classmethod
    def get_output_ports(cls):
        return [
            Port("widget_spec", PortType.SPEC, description="JSON spec for the frontend renderer"),
        ]

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            category="Visualization",
            display_name="Loss Curve",
            description="Real-time plot of metrics from CSV",
            icon="activity",
            color="#E64A19" # Orange for Vis
        )

    @classmethod
    def get_config_schema(cls):
        return LossCurveConfig

    def execute(self) -> Dict[str, Any]:
        run_id = self.inputs["run_id"]

        cfg = self.inputs.get("loss_curve_config") or self.config
        
        widget_spec = {
            "type": "line_chart",
            "id": self.node_id,
            "data_source": {
                "type": "csv_poll",
                "run_id": run_id,
                "file": "metrics.csv"
            },
            "visualization": {
                "title": cfg.title,
                "smoothing": cfg.smoothing
            },
        }

        return {
            "widget_spec": widget_spec
        }
