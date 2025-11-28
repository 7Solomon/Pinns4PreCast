import os
import numpy as np
import plotly.graph_objects as go
import torch
from scipy.interpolate import griddata

from src.state_management.state import State

"""
    Functions to export the predictions to VTK files for ParaView as well as CSV files for sensor data
"""


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

    sensor_ids = list(State().domain.TEMP_SENS_POINTS.keys())

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
            point = State().domain.TEMP_SENS_POINTS[sid]

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