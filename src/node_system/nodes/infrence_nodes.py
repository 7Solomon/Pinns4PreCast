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
            # The Architecture (from FlexDeepONetNode)
            Port("model", PortType.MODEL),
            
            # The Weights (Path string from SolverNode or CheckpointSelector)
            Port("checkpoint_path", PortType.CONFIG), 
            
            # Context for Grids & Scaling
            Port("domain", PortType.DOMAIN),
            Port("material", PortType.MATERIAL),
            
            # Context for Branch Inputs (reusing your input config node)
            Port("input_config", PortType.CONFIG), 
            
            # Helper for constructing the problem if needed (optional)
            Port("problem", PortType.PROBLEM),

            # Inference Settings
            Port("config", PortType.CONFIG, required=False)
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
        ckpt_path = self.inputs["checkpoint_path"]
        domain = self.inputs["domain"]
        material = self.inputs["material"]
        input_cfg = self.inputs["input_config"]
        problem = self.inputs["problem"]
        
        cfg = self.inputs.get("config")
        if not cfg: cfg = self.config

        print(f"[Inference] Loading weights from {ckpt_path}...")
        raise NotImplementedError("Schere!")
        # 2. Load Weights
        # Handle the Lightning checkpoint wrapper
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Clean keys (Lightning adds 'model.' prefix, our model doesn't have it)
        # Or if your Solver saved the inner model directly, keys might be fine.
        # This is a robust cleaner:
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_state_dict[k[6:]] = v # Remove 'model.'
            elif k.startswith('_pina_models.0.'):
                 # PINA sometimes wraps models like this depending on version
                 clean_state_dict[k.replace('_pina_models.0.', '')] = v
            else:
                clean_state_dict[k] = v
                
        try:
            model.load_state_dict(clean_state_dict, strict=False)
        except Exception as e:
            print(f"[Inference] Warning loading state dict: {e}")

        model.eval()

        # 3. Create Scaling Manager
        # We assume you implemented ScalingManager as discussed previously
        #scaler = ScalingManager(domain, material)

        # 4. Prepare Inputs (Branch & Trunk)
        
        # A. Branch Inputs: We need valid sensor values.
        # We can generate a dummy sample using the Dataset class logic
        # strictly for getting the sensor locations/values of a "standard" sample.
        ds = DeepONetDataset(
            problem=problem,
            domain=domain,
            material=material,
            n_pde=10, n_ic=10, n_bc_face=10, num_samples=1, # Minimal
            num_sensors_bc=input_cfg.num_sensors_bc,
            num_sensors_ic=input_cfg.num_sensors_ic
        )
        sample = ds[0]
        bc_sensors = sample['bc_sensor_values'].unsqueeze(0) # [1, n_sensors]
        ic_sensors = sample['ic_sensor_values'].unsqueeze(0) # [1, n_sensors]

        # B. Trunk Inputs: The Grid
        grid_coords = create_test_grid(
            spatial_domain_list=[domain.x, domain.y, domain.z],
            time_domain=domain.t,
            n_spatial=cfg.n_spatial,
            n_time=cfg.n_time
        ) # [N_points, 4]
        
        trunk_input = grid_coords.unsqueeze(0) # [1, N_points, 4]

        # 5. Run Prediction
        inputs_map = {
            'bc_sensor_values': bc_sensors,
            'ic_sensor_values': ic_sensors,
            'query_coords': trunk_input
        }
        
        print("[Inference] Running forward pass...")
        with torch.no_grad():
            # Model returns [1, N_points, 2] usually
            preds_raw = model(inputs_map)
            preds = preds_raw.squeeze(0) # [N_points, 2]

        # 6. Unscale Results
        # Col 0 = Temp, Col 1 = Alpha
        preds_unscaled = preds.clone()
        preds_unscaled[:, 0] = scaler.unscale_T(preds[:, 0]) - 273.15 # Kelvin to Celsius
        # Alpha usually 0-1, verify if scaling needed:
        preds_unscaled[:, 1] = scaler.unscale_alpha(preds[:, 1])

        # 7. Export
        print(f"[Inference] Saving results to {cfg.save_dir}")
        temp_path = os.path.join(cfg.save_dir, "inference_temp.csv")
        alpha_path = os.path.join(cfg.save_dir, "inference_alpha.csv")
        
        export_sensors_to_csv(
            preds_unscaled, 
            grid_coords, 
            temp_path, 
            alpha_path
        )

        return {"status": f"Saved to {cfg.save_dir}"}
