#from typing import List, Optional, Dict, Any
#from pydantic import BaseModel, Field
#from src.node_system.core import Node, Port, PortType, register_node, NodeMetadata
#
#
#class LossCurveConfig(BaseModel):
#    title: str = Field(default="Training Loss", title="Chart Title")
#    metrics: List[str] = Field(
#        default_factory=lambda: ["loss", "loss_physics", "loss_bc", "loss_ic"],
#        title="Metrics to Display"
#    )
#    refresh_rate: int = Field(default=2000, title="Poll Interval (ms)", ge=1000)
#
#@register_node("loss_curve")
#class LossCurveNode(Node):
#    """
#    Visualization specification node.
#    Does NOT execute during training - just declares WHAT to visualize.
#    """
#
#    @classmethod
#    def get_input_ports(cls):
#        return [
#            Port("logger", PortType.LOGGER, required=True, description="Just so that it connnects to tree dont know if necessary")
#            #Port("run_id", PortType.RUN_ID, required=True, description="Run ID from logger")
#        ]
#
#    @classmethod
#    def get_output_ports(cls):
#        return [
#            Port("widget_spec", PortType.SPEC, description="UI specification for frontend")
#        ]
#
#    @classmethod
#    def get_metadata(cls):
#        return NodeMetadata(
#            category="Visualization",
#            display_name="Loss Curve Monitor",
#            description="Real-time loss visualization",
#            icon="activity",
#            color="#E64A19"
#        )
#
#    @classmethod
#    def get_config_schema(cls):
#        return LossCurveConfig
#
#    def execute(self):
#        #run_id = self.inputs["run_id"]
#        cfg = self.config
#        
#        # Generate a widget specification for the frontend
#        #widget_spec = {
#        #    "type": "loss_curve",
#        #    "node_id": self.node_id,
#        #    "run_id": run_id,
#        #    "title": cfg.title,
#        #    "metrics": cfg.metrics,
#        #    "refresh_rate": cfg.refresh_rate,
#        #    "data_endpoint": f"/monitor/metrics/{run_id}"
#        #}
#        widget_spec = {} # kinda useless because not needed
#        
#        return {"widget_spec": widget_spec}
