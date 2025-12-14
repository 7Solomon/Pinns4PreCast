from lightning.pytorch.callbacks import Callback
import torch
import os
from src.DeepONet.dataset import DeepONetDataset
from src.DeepONet.infrence_pipline import create_test_grid
from src.DeepONet.vis import export_sensors_to_csv

from src.node_system.core import Node, NodeMetadata, PortType, Port, register_node


@register_node("visualization_callback")
class VisualizationCallbackNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("domain", PortType.DOMAIN),
            Port("material", PortType.MATERIAL),
            Port("dataset_config", PortType.CONFIG, required=False),
            Port("input_config", PortType.CONFIG, required=False),
            Port("config", PortType.CONFIG, required=False)
        ]

    @classmethod
    def get_output_ports(cls):
        # We define a new PortType for Callbacks
        return [Port("callback", "callback")] 

    @classmethod
    def get_metadata(cls):
        return NodeMetadata("Training", "Vis Callback", "Exports CSVs during training", icon="eye")

    @classmethod
    def get_config_schema(cls):
        return None
        #return VisualizationConfig 

    def execute(self):
        dom = self.inputs["domain"]
        mat = self.inputs["material"]
        d_cfg = self.inputs.get("dataset_config")
        i_cfg = self.inputs.get("input_config")
        v_cfg =  self.inputs.get("vis_config")

        if d_cfg is None: d_cfg = self.config.dataset_config 
        if i_cfg is None: i_cfg = self.config.input_config 
        if v_cfg is None: v_cfg = self.config.vis_config 

        # Instantiate the Callback
        cb = VisualizationCallback(
            domain=dom,
            material=mat,
            dataset_config=d_cfg,
            model_config=i_cfg,
            save_dir=v_cfg.save_dir,
            plot_every_n_epochs=v_cfg.plot_every_n_epochs
        )
        
        return {"callback": cb}

class VisualizationCallback(Callback):
    def __init__(self, domain, material, dataset_config, model_config, save_dir, plot_every_n_epochs=10):
        self.domain = domain
        self.material = material
        self.d_cfg = dataset_config
        self.m_cfg = model_config
        self.save_dir = save_dir
        self.every_n = plot_every_n_epochs
        
        # Create output directories immediately
        self.sensor_temp_path = os.path.join(save_dir, "sensors_temp")
        self.sensor_alpha_path = os.path.join(save_dir, "sensors_alpha")
        os.makedirs(self.sensor_temp_path, exist_ok=True)
        os.makedirs(self.sensor_alpha_path, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n != 0:
            return

        device = pl_module.device
        print(f"[Vis Callback] Visualizing epoch {epoch}")
        pl_module.eval()

        with torch.no_grad():
            # 1. Create a mini-dataset for visualization
            # Note: We use the params passed in __init__, not State()
            ds = DeepONetDataset(
                problem=pl_module.problem,
                domain=self.domain,
                material=self.material,
                n_pde=self.d_cfg.n_pde,
                n_ic=self.d_cfg.n_ic,
                n_bc_face=self.d_cfg.n_bc_face,
                num_samples=1, # We only need 1 sample for vis
                num_sensors_bc=self.m_cfg.num_sensors_bc,
                num_sensors_ic=self.m_cfg.num_sensors_ic
            )
            
            sample = ds[0]
            bc_target = sample['bc_sensor_values'].unsqueeze(0).to(device)
            ic_target = sample['ic_sensor_values'].unsqueeze(0).to(device)

            # 2. Create Grid
            test_coords = create_test_grid(
                spatial_domain=[self.domain.x, self.domain.y, self.domain.z], 
                time_domain=self.domain.t,
                n_spatial=15, n_time=15
            ).to(device)

            # 3. Predict
            test_batch = {
                'bc_sensor_values': bc_target,
                'ic_sensor_values': ic_target,
                'query_coords': test_coords.unsqueeze(0)
            }
            predictions = pl_module(test_batch).squeeze(0)
            
            # 4. Unscale (Using scaler attached to problem via previous steps)
            # If problem has scaler attached as discussed previously:
            scaler = pl_module.problem.scaler
            
            preds_unscaled = predictions.clone()
            # T is index 0
            preds_unscaled[:, 0] = scaler.unscale_T(predictions[:, 0]) - 273.15
            # Alpha is index 1 (already scaled 0-1 or needs unscaling depending on your logic)
            preds_unscaled[:, 1] = predictions[:, 1] 

            # 5. Export
            export_sensors_to_csv(
                preds_unscaled.cpu(), 
                test_coords.cpu(),
                sensor_temp_path=os.path.join(self.sensor_temp_path, f"epoch_{epoch}.csv"),
                sensor_alpha_path=os.path.join(self.sensor_alpha_path, f"epoch_{epoch}.csv")
            )
            
        pl_module.train()

