from lightning.pytorch.callbacks import Callback
import torch
import os
import numpy as np
from src.node_system.core import Node, NodeMetadata, PortType, Port, register_node



def export_to_vtk_series(predictions, test_coords, idx_path=None):
    """
    Exports predictions to a series of VTK files (one per timestep) 
    and a .pvd file for ParaView time handling.
    """
    if idx_path is None:
        content_dir = os.listdir(os.path.join('content'))
        if not content_dir:
            raise ValueError("No content directory found for exporting VTK files.")
        idx_path = os.path.join('content', content_dir[-1])
    output_folder = os.path.join(idx_path, "vtk_output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    unique_times = np.unique(test_coords[:, 3])
    unique_times.sort()
    
    print(f"Exporting {len(unique_times)} timesteps to '{output_folder}'...")

    # PVD file header (links time steps to files)
    pvd_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>'
    ]

    for step, t in enumerate(unique_times):
        # 1. Filter data for this time step
        mask = np.isclose(test_coords[:, 3], t)
        points = test_coords[mask, :3]  # x, y, z
        temp = predictions[mask, 0]
        alpha = predictions[mask, 1]
        
        filename = f"result_{step:04d}.vtk"
        filepath = os.path.join(output_folder, filename)
        
        # 2. Write Legacy VTK file (ASCII) - Robust and dependency-free
        with open(filepath, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"PINN prediction t={t}\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Write Points
            n_points = len(points)
            f.write(f"POINTS {n_points} float\n")
            np.savetxt(f, points, fmt="%.6f")
            
            # Write Cells (We treat points as VERTEX cells so they render)
            f.write(f"\nCELLS {n_points} {2 * n_points}\n")
            # Format: 1 (num_points_in_cell) id
            cell_data = np.column_stack((np.ones(n_points, dtype=int), np.arange(n_points, dtype=int)))
            np.savetxt(f, cell_data, fmt="%d")
            
            f.write(f"\nCELL_TYPES {n_points}\n")
            # Type 1 is VTK_VERTEX
            np.savetxt(f, np.ones(n_points, dtype=int), fmt="%d")
            
            # Write Data
            f.write(f"\nPOINT_DATA {n_points}\n")
            
            f.write("SCALARS Temperature float 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, temp, fmt="%.6f")
            
            f.write("\nSCALARS Alpha float 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, alpha, fmt="%.6f")

        # Add entry to PVD file
        pvd_lines.append(f'    <DataSet timestep="{t}" group="" part="0" file="{filename}"/>')

    # Close PVD file
    pvd_lines.append('  </Collection>')
    pvd_lines.append('</VTKFile>')
    
    with open(os.path.join(output_folder, "results.pvd"), "w") as f:
        f.write("\n".join(pvd_lines))
        
    print(f"Done. Open '{os.path.join(output_folder, 'results.pvd')}' in ParaView.")


def export_sensors_to_csv(predictions, test_coords, sensor_temp_path, sensor_alpha_path):
    """
    Interpolates data at sensor locations for all time steps and saves two CSVs:
    - sensors_temperature.csv (Time_s, Time_h, <sensor>_Temp...)
    - sensors_alpha.csv       (Time_s, Time_h, <sensor>_Alpha...)
    """
    #if idx_path is None:
    #    content_dir = os.listdir(os.path.join('content'))
    #    if not content_dir:
    #        raise ValueError("No content directory found for exporting VTK files.")
    #    idx_path = os.path.join('content', content_dir[-1])
#
    #output_temp = os.path.join(idx_path, "sensors_temperature.csv")
    #output_alpha = os.path.join(idx_path, "sensors_alpha.csv")

    all_times = np.unique(test_coords[:, 3])
    all_times.sort()

    sensor_ids = list(domain.TEMP_SENS_POINTS.keys())

    header_temp = ["Time_s", "Time_h"] + [f"{sid}_Temp" for sid in sensor_ids]
    header_alpha = ["Time_s", "Time_h"] + [f"{sid}_Alpha" for sid in sensor_ids]

    rows_temp = []
    rows_alpha = []

    #print(f"Interpolating sensors for CSV export ({len(all_times)} timesteps, {len(sensor_ids)} sensors)...")

    for t in all_times:
        mask_t = np.isclose(test_coords[:, 3], t)
        coords_t = test_coords[mask_t, :3]
        pred_t = predictions[mask_t]

        temp_row = [t, t / 3600.0]
        alpha_row = [t, t / 3600.0]

        for sid in sensor_ids:
            point = domain.TEMP_SENS_POINTS[sid]

            temp_val = griddata(coords_t, pred_t[:, 0], point, method='linear')
            alpha_val = griddata(coords_t, pred_t[:, 1], point, method='linear')

            if np.isnan(temp_val):
                temp_val = griddata(coords_t, pred_t[:, 0], point, method='nearest')
            if np.isnan(alpha_val):
                alpha_val = griddata(coords_t, pred_t[:, 1], point, method='nearest')

            temp_row.append(float(temp_val))
            alpha_row.append(float(alpha_val))

        rows_temp.append(temp_row)
        rows_alpha.append(alpha_row)

    # Save CSVs
    np.savetxt(sensor_temp_path, np.array(rows_temp), delimiter=",", header=",".join(header_temp), comments="", fmt="%.6f")
    np.savetxt(sensor_alpha_path, np.array(rows_alpha), delimiter=",", header=",".join(header_alpha), comments="", fmt="%.6f")

    #print(f"Temperature sensor data exported to '{sensor_temp_path}'")
    #print(f"Alpha sensor data exported to '{sensor_alpha_path}'")
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
        return [Port("callback", PortType.CALLBACK)] 

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

