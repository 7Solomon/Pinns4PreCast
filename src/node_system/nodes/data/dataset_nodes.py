
from src.node_system.nodes.data.function_definitions import DeepONetDataset
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
from src.node_system.configs.dataset import CompositeDatasetConfig

@register_node("deeponet_dataset")
class DeepONetDatasetNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("problem", PortType.PROBLEM),
            Port("material", PortType.MATERIAL),
            Port("domain", PortType.DOMAIN),
            Port("composite_dataset_config", PortType.CONFIG, required=False)
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
        
        cfg = self.inputs.get("composite_dataset_config") or self.config
        
        dataset = DeepONetDataset(
            problem=problem,
            domain=domain,
            material=material,
            n_pde=cfg.data_config.n_pde,
            n_ic=cfg.data_config.n_ic,
            n_bc_face=cfg.data_config.n_bc_face,
            num_samples=cfg.data_config.num_samples,
            num_sensors_bc=cfg.input_config.num_sensors_bc, 
            num_sensors_ic=cfg.input_config.num_sensors_ic
        )
        
        return {"dataset": dataset}