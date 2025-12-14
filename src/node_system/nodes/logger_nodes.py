from src.node_system.core import Node, Port, register_node
from src.DeepONet.logger import DashboardLogger

@register_node("dashboard_logger")
class DashboardLoggerNode(Node):
    @classmethod
    def get_input_ports(cls):
        return []

    @classmethod
    def get_output_ports(cls):
        return [Port("logger", "logger")]

    @classmethod
    def get_config_schema(cls):
        return None 

    def execute(self):
        cfg = self.config
        logger = DashboardLogger(save_dir=cfg.save_dir, version=cfg.version)
        return {"logger": logger}
