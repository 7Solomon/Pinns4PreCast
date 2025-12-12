from lightning.pytorch.callbacks import Callback
import torch
import os


from src.DeepONet.dataset import DeepONetDataset
from src.DeepONet.infrence_pipline import create_test_grid
from src.DeepONet.vis import export_sensors_to_csv, export_to_vtk_series
from src.state_management.state import State
from src.utils import unscale_T


class VisualizationCallback(Callback):
    """
        THIS is so that every 10 EPOCHS a csv file is exported for visualization of the SENSOR POINTs
    """
    def __init__(self, vtk_path, sensor_temp_path, sensor_alpha_path, plot_every_n_epochs=10):
        self.vtk_path = vtk_path
        self.sensor_temp_path = sensor_temp_path
        self.sensor_alpha_path = sensor_alpha_path
        
        self.every_n = plot_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        """
        trainer: The PyTorch Lightning Trainer object
        pl_module: Your LightningModule (i.e., your 'solver' or model)
        """
        
        if getattr(State(), 'kill_training_signal', False):
            return
        
        epoch = trainer.current_epoch
        
        if epoch % self.every_n == 0:
            
            device = pl_module.device
            print(f"Visualizing epoch {epoch} on device: {device}")
            pl_module.eval()

            idx_path = State().directory_manager.run_idx_path
            with torch.no_grad():
                ds = DeepONetDataset(
                    problem=pl_module.problem,
                    n_pde=State().config.dataset.n_pde,
                    n_ic=State().config.dataset.n_ic,
                    n_bc_face=State().config.dataset.n_bc_face,
                    num_samples=State().config.dataset.num_samples,
                    num_sensors_bc=State().config.model.num_sensors_bc,
                    num_sensors_ic=State().config.model.num_sensors_ic
                )
                
                sample = ds[0]
                bc_target_temperature = sample['bc_sensor_values'].unsqueeze(0).to(device)
                ic_target_temperature = sample['ic_sensor_values'].unsqueeze(0).to(device)

                # Create grid
                test_coords = create_test_grid(
                    spatial_domain=[State().domain.x, State().domain.y, State().domain.z], 
                    time_domain=State().domain.t,
                    n_spatial=15,
                    n_time=15
                ).to(device)

                test_coords_batched = test_coords.unsqueeze(0) # [1, num_points, 4]

                test_batch = {
                    'bc_sensor_values': bc_target_temperature,
                    'ic_sensor_values': ic_target_temperature,
                    'query_coords': test_coords_batched
                }
                
                # Directly call the module
                predictions = pl_module(test_batch) # [1, num_points, 2]
                predictions = predictions.squeeze(0)
                
                # Unscale (Operations on GPU)
                predictions_unscaled = predictions.clone()
                predictions_unscaled[:, 0] = unscale_T(predictions[:, 0]) - 273.15
                predictions_unscaled[:, 1] = predictions[:, 1]

                # Move to CPU for plotting/saving to avoid GPU memory leaks in loops
                preds_cpu = predictions_unscaled.cpu()
                coords_cpu = test_coords.cpu()

                print(f"Temp range: [{preds_cpu[:, 0].min():.4f}, {preds_cpu[:, 0].max():.4f}] C")

                # 5. VISUALIZE
                #export_to_vtk_series(
                #    preds_cpu, 
                #    coords_cpu, 
                #    idx_path=idx_path,
                #    epoch=epoch # Good practice to append epoch to filename
                #)
                sensor_temp_path = os.path.join(self.sensor_temp_path, f"epoch_{epoch}.csv")
                sensor_alpha_path = os.path.join(self.sensor_alpha_path, f"epoch_{epoch}.csv")
                export_sensors_to_csv(
                    preds_cpu, 
                    coords_cpu,
                    sensor_temp_path=sensor_temp_path,
                    sensor_alpha_path=sensor_alpha_path
                )
            pl_module.train()
