from lightning.pytorch.callbacks import Callback
from pydantic import BaseModel, Field
import torch
import pandas as pd
import os
from src.node_system.configs.vis import CompositeVisualizationConfig
from src.node_system.nodes.visualisation.export_sensors import export_sensors_to_csv
from src.node_system.nodes.data.function_definitions import DeepONetDataset
from src.node_system.core import Node, NodeMetadata, PortType, Port, register_node

from src.node_system.event_bus import get_event_bus, Event, EventType
import asyncio


@register_node("visualization_callback")
class VisualizationCallbackNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("material", PortType.MATERIAL),
            Port("domain", PortType.DOMAIN),
            Port("dataset_config", PortType.CONFIG, required=False),
            Port("input_config", PortType.CONFIG, required=False),
            Port("vis_config", PortType.CONFIG, required=False)
        ]

    @classmethod
    def get_output_ports(cls):
        return [Port("callback", PortType.CALLBACK)] 

    @classmethod
    def get_metadata(cls):
        return NodeMetadata("Training", "Vis Callback", "Exports CSVs during training", icon="eye")

    @classmethod
    def get_config_schema(cls):
        return CompositeVisualizationConfig 

    def execute(self):
        dom = self.inputs["domain"]
        mat = self.inputs["material"]
        d_cfg = self.inputs.get("data_config") or self.config.data_config 
        i_cfg = self.inputs.get("input_config") or  self.config.input_config
        v_cfg =  self.inputs.get("vis_config") or  self.config.vis_config 

        run_id = self.context.get("run_id")


        # Instantiate the Callback  
        cb = VisualizationCallback(
            domain=dom,
            material=mat,
            dataset_config=d_cfg,
            model_config=i_cfg,
            save_dir=v_cfg.save_dir,
            run_id=run_id,
            plot_every_n_epochs=v_cfg.plot_every_n_epochs
        )
        
        return {"callback": cb, "run_id": None}


class VisualizationCallback(Callback):
    """
    Visualization callback that publishes events instead of relying on polling.
    """
    
    def __init__(self, domain, material, dataset_config, model_config, 
                 save_dir, run_id, plot_every_n_epochs=10):
        self.domain = domain
        self.material = material
        self.d_cfg = dataset_config
        self.m_cfg = model_config
        self.save_dir = save_dir
        self.run_id = run_id
        self.every_n = plot_every_n_epochs
        
        self.event_bus = get_event_bus()
        
        # Create output directories
        self.sensor_temp_path = os.path.join(save_dir, run_id, "sensors_temp")
        self.sensor_alpha_path = os.path.join(save_dir, run_id, "sensors_alpha")
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
            from src.node_system.nodes.data.function_definitions import DeepONetDataset
            from src.node_system.nodes.visualisation.export_sensors import export_sensors_to_csv
            
            ds = DeepONetDataset(
                problem=pl_module.problem,
                domain=self.domain,
                material=self.material,
                n_pde=self.d_cfg.n_pde,
                n_ic=self.d_cfg.n_ic,
                n_bc_face=self.d_cfg.n_bc_face,
                num_samples=1,
                num_sensors_bc=self.m_cfg.num_sensors_bc,
                num_sensors_ic=self.m_cfg.num_sensors_ic
            )
            
            sample = ds[0]
            bc_target = sample['bc_sensor_values'].unsqueeze(0).to(device)
            ic_target = sample['ic_sensor_values'].unsqueeze(0).to(device)

            test_coords = create_test_grid(
                spatial_domain=[self.domain.x, self.domain.y, self.domain.z],
                time_domain=self.domain.t
            ).to(device)

            test_batch = {
                'bc_sensor_values': bc_target,
                'ic_sensor_values': ic_target,
                'query_coords': test_coords.unsqueeze(0)
            }
            predictions = pl_module(test_batch).squeeze(0)
            
            # Unscale
            preds_unscaled = predictions.clone()
            preds_unscaled[:, 0] = self.domain.unscale_T(predictions[:, 0], self.material.Temp_ref) - 273.15
            preds_unscaled[:, 1] = predictions[:, 1]
            
            temp_file = os.path.join(self.sensor_temp_path, f"epoch_{epoch}.csv")
            alpha_file = os.path.join(self.sensor_alpha_path, f"epoch_{epoch}.csv")
            
            export_sensors_to_csv(
                preds_unscaled.cpu().numpy(),
                self.domain,
                test_coords=test_coords.cpu().numpy(),
                sensor_temp_path=temp_file,
                sensor_alpha_path=alpha_file
            )
            
            sensor_data = self._csv_to_recharts_format(temp_file, alpha_file)
            
            self._publish_event({
                "epoch": epoch,
                "data": sensor_data,  # Full array: [{"step": 0, "T_sensor1": 23.5, "alpha_sensor1": 0.1}, ...]
                "message": f"Sensor data for epoch {epoch}"
            })
            
        pl_module.train()

        
    def _csv_to_recharts_format(self, temp_file: str, alpha_file: str) -> list:
        """Convert your CSV format to Recharts time-series"""
        # Read your CSV headers
        df_temp = pd.read_csv(temp_file)
        df_alpha = pd.read_csv(alpha_file)
        
        # Merge on Time_s
        df = pd.merge(df_temp, df_alpha, on=['Time_s', 'Time_h'], suffixes=('_temp', '_alpha'))
        
        # Convert to Recharts format
        recharts_data = []
        for _, row in df.iterrows():
            step_data = {
                "step": int(row['Time_s']),  # Use Time_s as step
                "time_hours": float(row['Time_h'])
            }
            
            # Add all sensors
            for col in df.columns:
                if col.endswith('_Temp') or col.endswith('_Alpha'):
                    step_data[col] = float(row[col])
            
            recharts_data.append(step_data)
        
        return recharts_data

        

    
    def _publish_event(self, data: dict):
        """Publish sensor data update event"""
        print(f"publish SENSOR: {data}")
        event = Event(
            type=EventType.SENSOR_DATA_UPDATED,
            run_id=self.run_id,
            data=data
        )
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.event_bus.publish(event))
            else:
                loop.run_until_complete(self.event_bus.publish(event))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.event_bus.publish(event))


def create_test_grid(spatial_domain=[(0, 1),(0, 1),(0, 1)], time_domain=(0, 1), 
                     n_spatial=15, n_time=30):
    """
    Create structured test grid for evaluation.
    
    Args:
        spatial_domain: tuple (min, max) for x, y, z
        time_domain: tuple (t_start, t_end)
        n_spatial: number of points per spatial dimension
        n_time: number of time points
    
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
    
    print(f"Created test grid:")
    print(f"  X range: [{test_coords[:, 0].min():.3f}, {test_coords[:, 0].max():.3f}]")
    print(f"  Y range: [{test_coords[:, 1].min():.3f}, {test_coords[:, 1].max():.3f}]")
    print(f"  Z range: [{test_coords[:, 2].min():.3f}, {test_coords[:, 2].max():.3f}]")
    print(f"  T range: [{test_coords[:, 3].min():.3f}, {test_coords[:, 3].max():.3f}]")
    
    return test_coords

