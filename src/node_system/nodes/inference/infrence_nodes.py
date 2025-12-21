import torch
import os
from pydantic import BaseModel, Field

from src.node_system.configs.infrence import CompositeInferenceConfig
from src.node_system.nodes.visualisation.export_sensors import export_sensors_to_csv
from src.node_system.nodes.visualisation.export_to_vtk import export_to_vtk_series
from src.node_system.configs.model_input import InputConfig
from src.node_system.nodes.data.function_definitions import DeepONetDataset
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node

    
def create_test_grid(spatial_domain=[(0, 1),(0, 1),(0, 1)], time_domain=(0, 1), 
                     n_spatial=15, n_time=30):
    """
    Returns:
        test_coords: tensor of shape [n_spatial^3 * n_time, 4]
    """
    # Create 1D grids
    x = torch.linspace(spatial_domain[0][0], spatial_domain[0][1], n_spatial)
    y = torch.linspace(spatial_domain[1][0], spatial_domain[1][1], n_spatial)
    z = torch.linspace(spatial_domain[2][0], spatial_domain[2][1], n_spatial)
    t = torch.linspace(time_domain[0], time_domain[1], n_time)
    
    # Create 4D meshgrid
    X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
    
    # Flatten and stack
    test_coords = torch.stack([X.flatten(), Y.flatten(), 
                               Z.flatten(), T.flatten()], dim=1)
    
    #print(f"Created test grid:")
    #print(f"  X range: [{test_coords[:, 0].min():.3f}, {test_coords[:, 0].max():.3f}]")
    #print(f"  Y range: [{test_coords[:, 1].min():.3f}, {test_coords[:, 1].max():.3f}]")
    #print(f"  Z range: [{test_coords[:, 2].min():.3f}, {test_coords[:, 2].max():.3f}]")
    #print(f"  T range: [{test_coords[:, 3].min():.3f}, {test_coords[:, 3].max():.3f}]")
    
    return test_coords



@register_node("deeponet_inference")
class DeepONetInferenceNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("solver", PortType.SOLVER, description="Initialized LightningModule"),
            Port("domain", PortType.DOMAIN),
            Port("material", PortType.MATERIAL),                        
            Port("problem", PortType.PROBLEM),
            Port("run_id", PortType.RUN_ID),

            Port("infrence_config", PortType.CONFIG, required=False),
            Port("input_config", PortType.CONFIG, required=False)

        ]

    @classmethod
    def get_output_ports(cls):
        return [] # Just a success message/path

    @classmethod
    def get_metadata(cls):
        return NodeMetadata("Inference", "Visualizer", "Run prediction & export CSV", icon="eye")

    @classmethod
    def get_config_schema(cls):
        return CompositeInferenceConfig

    def execute(self):
        # 1. Unpack Inputs
        solver = self.inputs["solver"]
        domain = self.inputs["domain"]
        material = self.inputs["material"]
        problem = self.inputs["problem"]
        run_id = self.inputs["run_id"]
        
        inf_cfg = self.inputs.get("infrence_config")
        inp_cfg = self.inputs.get("input_config")
        
        if not inf_cfg: inf_cfg = self.config.inf_cfg
        if not inp_cfg: inp_cfg = self.config.inp_cfg


        ckpts_dir = os.path.join("content", "runs", run_id, "checkpoints")
        result_dir = os.path.join("content", "runs", run_id, "results")
        vtk_dir = os.path.join(result_dir, "vtk")
        temp_sensor_dir =  os.path.join(result_dir, "temp_sensor")
        alpha_sensor_dir =  os.path.join(result_dir, "alpha_sensor")
        
        os.makedirs(ckpts_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(vtk_dir, exist_ok=True)
        os.makedirs(temp_sensor_dir, exist_ok=True)
        os.makedirs(alpha_sensor_dir, exist_ok=True)

        idx_path = os.path.join(vtk_dir, str(len(vtk_dir)))
        temp_sensor_path = os.path.join(temp_sensor_dir, str(len(temp_sensor_dir)))
        alpha_sensor_path = os.path.join(alpha_sensor_dir, str(len(alpha_sensor_dir)))

        latest_ckpt = max(os.listdir(ckpts_dir), key=lambda x: os.path.getctime(os.path.join(ckpts_dir, x)))
        checkpoint = torch.load(os.path.join(ckpts_dir, latest_ckpt), map_location='cpu')
        solver.load_state_dict(checkpoint['state_dict'])

        
        # Set to eval mode
        solver.eval()
        
        # Get model device
        device = next(solver.model.parameters()).device
        print(f"Running inference on device: {device}")
        
        with torch.no_grad():
            ds = DeepONetDataset(
                problem=solver.problem,
                domain=domain,
                material=material,
                n_pde=1,      # NOT NEEDED beause 
                n_ic=1,       # NOT NEEDED
                n_bc_face=1,  # SAME
                num_samples=1,
                num_sensors_bc=inp_cfg.num_sensors_bc,
                num_sensors_ic= inp_cfg.num_sensors_ic
            )
            
            sample = ds[0]
            bc_target_temperature = sample['bc_sensor_values'].unsqueeze(0)    # DELETET CHANGED TO USE SOMEHTING ELSE
            ic_target_temperature = sample['ic_sensor_values'].unsqueeze(0)

            # BECAUSE WE LIKE STRUTURED GRID FOR CHECKING
            test_coords = create_test_grid(
                spatial_domain=[domain.x, domain.y, domain.z], 
                time_domain=domain.t,
                n_spatial=inf_cfg.n_spatial,
                n_time=inf_cfg.n_time
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
            predictions_unscaled[:, 0] = domain.unscale_T(predictions[:, 0], material.Temp_ref) - 273.15
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
                domain,
                test_coords.cpu(),
                sensor_temp_path=temp_sensor_path,
                sensor_alpha_path=alpha_sensor_path
            )

            return []
