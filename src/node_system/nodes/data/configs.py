
from node_system.core import register_node, Node, Port, PortType, NodeMetadata
from src.state_management.config import DatasetConfig


@register_node("dataset_config")
class DatasetConfigNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [] # Pure data source

    @classmethod
    def get_output_ports(cls):
        return [Port("config", PortType.CONFIG)]

    @classmethod
    def get_metadata(cls):
        return NodeMetadata("Configs", "Dataset Params", "Sample counts, batch size", icon="list")

    @classmethod
    def get_config_schema(cls):
        return DatasetConfig

    def execute(self):
        return {"config": self.config}