from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from src.node_system.nodes.data.function_definitions import deeponet_collate_fn
from src.node_system.configs.dataloader import DataLoaderConfig
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node



@register_node("deeponet_dataloader")
class DeepONetDataLoaderNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("dataset", PortType.DATASET),
            Port("config", PortType.CONFIG, required=False) # Optional external DataLoaderConfig
        ]

    @classmethod
    def get_output_ports(cls):
        return [Port("dataloader", PortType.DATALOADER)]

    @classmethod
    def get_metadata(cls):
        return NodeMetadata("Dataset", "DeepONet Loader", "Batches and collates data", icon="truck")

    @classmethod
    def get_config_schema(cls):
        return DataLoaderConfig

    def execute(self):
        dataset = self.inputs["dataset"]
        
        # 1. Config Fallback
        cfg = self.inputs.get("config")
        if not cfg: cfg = self.config

        
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            collate_fn=deeponet_collate_fn
        )

        return {"dataloader": loader}
