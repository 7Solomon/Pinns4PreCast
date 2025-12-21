from src.node_system.configs.infrence import RunIdChooserConfig
from src.node_system.core import NodeMetadata, register_node, Node, PortType, Port


@register_node("run_id_chooser")
class RunIdChooserNode(Node):
    @classmethod
    def get_input_ports(cls):
        return []

    @classmethod
    def get_output_ports(cls):
        return {
            "run_id": Port("run_id", PortType.RUN_ID)
            }
        

    @classmethod
    def get_metadata(cls):
        return NodeMetadata(
            category="Run IDs",
            display_name="Run ID Selector",
            description="Selects an existing run ID for inference",
            icon="database" 
        )

    @classmethod
    def get_config_schema(cls):
        return RunIdChooserConfig

    def execute(self):
        cfg: RunIdChooserConfig = self.config
        return {
            "run_id": cfg.run_id,
        }
