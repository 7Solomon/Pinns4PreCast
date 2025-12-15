import torch
import os
from pydantic import BaseModel, Field

from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
from src.DeepONet.infrence_pipline import create_test_grid
from src.DeepONet.vis import export_sensors_to_csv
from src.DeepONet.dataset import DeepONetDataset

# Config for Resolution
class InferenceConfig(BaseModel):
    n_spatial: int = Field(15, description="Grid points per spatial axis")
    n_time: int = Field(15, description="Grid points for time")
    save_dir: str = Field("./results", description="Output directory")

@register_node("deeponet_inference")
class DeepONetInferenceNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("model", PortType.MODEL),
            Port("domain", PortType.DOMAIN),
            Port("material", PortType.MATERIAL),                        
            Port("problem", PortType.PROBLEM),

            Port("infrence_config", PortType.CONFIG, required=False),
            Port("input_config", PortType.CONFIG, required=False)

        ]

    @classmethod
    def get_output_ports(cls):
        return [Port("status", PortType.CONFIG)] # Just a success message/path

    @classmethod
    def get_metadata(cls):
        return NodeMetadata("Inference", "Visualizer", "Run prediction & export CSV", icon="eye")

    @classmethod
    def get_config_schema(cls):
        return InferenceConfig

    def execute(self):
        # 1. Unpack Inputs
        model = self.inputs["model"]
        domain = self.inputs["domain"]
        material = self.inputs["material"]
        problem = self.inputs["problem"]
        
        inf_cfg = self.inputs.get("infrence_config")
        inp_cfg = self.inputs.get("input_config")
        
        if not inf_cfg: inf_cfg = self.config.infrence_config
        if not inp_cfg: inp_cfg = self.config.input_config
        raise NotImplementedError("is missing here schere")
        # 2. Load Weights
        # Handle the Lightning checkpoint wrapper
        checkpoint = torch.load(os.path.join(inf_cfg.ckpt_path, 'checkpoints', dir[-1]), map_location='cpu')
        solver.load_state_dict(checkpoint['state_dict'])

        
        # Set to eval mode
        solver.eval()
        
        # Get model device
        device = next(solver.model.parameters()).device
        print(f"Running inference on device: {device}")
        
        with torch.no_grad():
            ds = DeepONetDataset(
                problem=solver.problem,
                n_pde=1,      # NOT NEEDED beause 
                n_ic=1,       # NOT NEEDED
                n_bc_face=1,  # SAME
                num_samples=1,
                num_sensors_bc=num_sensors_bc,
                num_sensors_ic= num_sensors_ic
            )
            
            sample = ds[0]
            bc_target_temperature = sample['bc_sensor_values'].unsqueeze(0)    # DELETET CHANGED TO USE SOMEHTING ELSE
            ic_target_temperature = sample['ic_sensor_values'].unsqueeze(0)

            # BECAUSE WE LIKE STRUTURED GRID FOR CHECKING
            test_coords = create_test_grid(
                spatial_domain=[State().domain.x, State().domain.y, State().domain.z], 
                time_domain=State().domain.t,
                n_spatial=n_spatial,
                n_time=n_time
            ).to(device)

            # batch dimension to test_coords [1, num_points, 4]
            test_coords_batched = test_coords.unsqueeze(0)
            
            # DICT
            test_batch = {
                'bc_sensor_values': bc_target_temperature,  # [1, num_sensors]
                'ic_sensor_values': ic_target_temperature,  # [1, num_sensors]
                'query_coords': test_coords_batched  # [1, num_points, 4]
            }
            
            print("\nForward...")
            predictions = solver.forward(test_batch)  # [1, num_points, 2]
            predictions = predictions.squeeze(0)  # [num_points, 2]

            # Unscale predictions
            predictions_unscaled = predictions.clone()
            predictions_unscaled[:, 0] = unscale_T(predictions[:, 0]) - 273.15
            predictions_unscaled[:, 1] = predictions[:, 1]#unscale_alpha(predictions[:, 1])

            print(f"\nPredictions shape: {predictions_unscaled.shape}")
            print(f"Temperature range: [{predictions_unscaled[:, 0].min():.4f}, {predictions_unscaled[:, 0].max():.4f}] C")
            print(f"Alpha range: [{predictions_unscaled[:, 1].min():.4f}, {predictions_unscaled[:, 1].max():.4f}]")
            
            # Visualize
            print("\nCreating visualizations...")
            export_to_vtk_series(
                predictions_unscaled.cpu(), 
                test_coords.cpu(), 
                idx_path=idx_path
            )
            export_sensors_to_csv(
                predictions_unscaled.cpu(), 
                test_coords.cpu(),
                idx_path=idx_path
            )

            return {"status": f"Saved to {cfg.save_dir}"}
