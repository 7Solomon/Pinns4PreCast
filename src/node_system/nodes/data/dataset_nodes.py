
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node

from src.DeepONet.dataset import DeepONetDataset 
from src.state_management.config import CompositeDatasetConfig

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
