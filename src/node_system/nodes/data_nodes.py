from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node

from src.DeepONet.dataset import DeepONetDataset, deeponet_collate_fn 
from src.state_management.config import CompositeDatasetConfig, DatasetConfig
from src.state_management.config import DataLoaderConfig


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

@register_node("deeponet_dataset")
class DeepONetDatasetNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("problem", PortType.PROBLEM),
            Port("domain", PortType.DOMAIN),
            Port("material", PortType.MATERIAL),
            Port("input_config", PortType.CONFIG, required=False),
            Port("dataset_config", PortType.CONFIG, required=False) # Now just generation params
        ]

    @classmethod
    def get_output_ports(cls):
        # CHANGED: Outputs the raw dataset object
        return [Port("dataset", PortType.DATASET)] 

    @classmethod
    def get_config_schema(cls):
        return CompositeDatasetConfig 
    
    @classmethod
    def get_metadata(cls):
        return NodeMetadata(
            category="Dataset",
            display_name="DeepONet Dataset Generator",
            description="Generates samples using random fields",
            icon="database"
        )

    def execute(self):
        problem = self.inputs["problem"]
        domain = self.inputs["domain"]
        material = self.inputs["material"]
        problem = self.inputs["problem"]
        
        i_cfg = self.inputs.get("input_config")
        d_cfg = self.inputs.get("dataset_config")
        
        if not i_cfg: i_cfg = self.config.input_config
        if not d_cfg: d_cfg = self.config.data_config

        dataset = DeepONetDataset(
            problem=problem,
            domain=domain,
            material=material,
            n_pde=d_cfg.n_pde,
            n_ic=d_cfg.n_ic,
            n_bc_face=d_cfg.n_bc_face,
            num_samples=d_cfg.num_samples,
            num_sensors_bc=i_cfg.num_sensors_bc, 
            num_sensors_ic=i_cfg.num_sensors_ic
        )
        
        return {"dataset": dataset}
