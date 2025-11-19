import os
import numpy as np
import plotly.graph_objects as go
import torch
from scipy.interpolate import griddata

from domain import DomainVariables
domain_vars = DomainVariables()

def create_unified_interactive_viz_v2(predictions, test_coords, bc_samples=None, ic_samples=None, output_html="viz.html"):
    all_times = np.unique(test_coords[:, 3])
    N_times = len(all_times)

    # --- Create one Temperature + Alpha trace per timestep ---
    data = []
    for k, t in enumerate(all_times):
        mask_t = np.isclose(test_coords[:, 3], t)
        X, Y, Z = test_coords[mask_t, :3].T
        temp, alpha = predictions[mask_t, 0], predictions[mask_t, 1]

        data.append(go.Scatter3d(
            x=X, y=Y, z=Z, mode="markers",
            marker=dict(size=3, color=temp, colorscale="Viridis", 
                       colorbar=dict(title="Temperature", x=1.1)),
            name=f"Temperature t={t:.2f}",
            visible=(k == 0)
        ))
        data.append(go.Scatter3d(
            x=X, y=Y, z=Z, mode="markers",
            marker=dict(size=3, color=alpha, colorscale="Cividis", 
                       colorbar=dict(title="Alpha", x=1.1)),
            name=f"Alpha t={t:.2f}",
            visible=False
        ))

    # --- Create frames that switch which timestep is visible ---
    # We need TWO sets of frames: one for temperature mode, one for alpha mode
    frames = []
    
    # Temperature frames (show temp traces, hide alpha traces)
    for k in range(N_times):
        frame_data = []
        for i in range(2 * N_times):
            trace = data[i]
            new_trace = go.Scatter3d(
                x=trace.x, y=trace.y, z=trace.z,
                mode=trace.mode,
                marker=trace.marker,
                name=trace.name,
                visible=(i == 2*k)  # Show only temp trace for timestep k
            )
            frame_data.append(new_trace)
        frames.append(go.Frame(name=f"temp_{k}", data=frame_data))
    
    # Alpha frames (show alpha traces, hide temp traces)
    for k in range(N_times):
        frame_data = []
        for i in range(2 * N_times):
            trace = data[i]
            new_trace = go.Scatter3d(
                x=trace.x, y=trace.y, z=trace.z,
                mode=trace.mode,
                marker=trace.marker,
                name=trace.name,
                visible=(i == 2*k + 1)  # Show only alpha trace for timestep k
            )
            frame_data.append(new_trace)
        frames.append(go.Frame(name=f"alpha_{k}", data=frame_data))

    # --- Temperature/Alpha toggle buttons ---
    # For temperature: show first timestep (index 0), hide all others
    temp_initial = [False] * (2 * N_times)
    temp_initial[0] = True  # Show only first temperature trace
    
    # For alpha: show first timestep (index 1), hide all others
    alpha_initial = [False] * (2 * N_times)
    alpha_initial[1] = True  # Show only first alpha trace

    # Create slider steps for temperature mode
    temp_slider_steps = [
        dict(method='animate',
             args=[[f"temp_{i}"],
                   {"frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}}],
             label=f"{all_times[i]/3600:.1f}h")
        for i in range(N_times)
    ]
    
    # Create slider steps for alpha mode
    alpha_slider_steps = [
        dict(method='animate',
             args=[[f"alpha_{i}"],
                   {"frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}}],
             label=f"{all_times[i]/3600:.1f}h")
        for i in range(N_times)
    ]

    updatemenus = [
        dict(type="buttons", showactive=True,
             buttons=[
                 dict(label="Temperature", method="update",
                      args=[{"visible": temp_initial},
                            {"sliders": [dict(
                                steps=temp_slider_steps,
                                active=0,
                                transition={'duration': 0},
                                x=0.15, y=-0.05, len=0.7,
                                xanchor='left', yanchor='top',
                                pad=dict(t=50, b=10),
                                currentvalue=dict(
                                    visible=True,
                                    prefix="Time: ",
                                    xanchor="center",
                                    font=dict(size=14)
                                )
                            )]}]),
                 dict(label="Alpha", method="update",
                      args=[{"visible": alpha_initial},
                            {"sliders": [dict(
                                steps=alpha_slider_steps,
                                active=0,
                                transition={'duration': 0},
                                x=0.15, y=-0.05, len=0.7,
                                xanchor='left', yanchor='top',
                                pad=dict(t=50, b=10),
                                currentvalue=dict(
                                    visible=True,
                                    prefix="Time: ",
                                    xanchor="center",
                                    font=dict(size=14)
                                )
                            )]}]),
             ], 
             x=0.5, y=1.12, xanchor='center', yanchor='top',
             bgcolor='rgba(255, 255, 255, 0.8)',
             bordercolor='#666',
             borderwidth=1)
    ]

    # --- Time slider (initial state for temperature) ---
    sliders = [dict(
        steps=temp_slider_steps,
        active=0,
        transition={'duration': 0},
        x=0.15, y=-0.05, len=0.7,
        xanchor='left', yanchor='top',
        pad=dict(t=50, b=10),
        currentvalue=dict(
            visible=True,
            prefix="Time: ",
            xanchor="center",
            font=dict(size=14)
        )
    )]

    layout = go.Layout(
        #title=dict(
        #    text="DeepONet: Temperature & Alpha Evolution",
        #    x=0.5,
        #    xanchor='center',
        #    font=dict(size=18)
        #),
        updatemenus=updatemenus,
        sliders=sliders,
        margin=dict(l=0, r=0, t=100, b=80),
        scene=dict(
            xaxis_title="X", 
            yaxis_title="Y", 
            zaxis_title="Z", 
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=850
    )

    fig = go.Figure(data=data, frames=frames, layout=layout)
    fig.write_html(output_html, auto_play=False)
    print(f"Visualization exported to {output_html}")
    print(f"Time steps: {N_times}, ranging from {all_times[0]/3600:.1f}h to {all_times[-1]/3600:.1f}h")


def create_vis_on_sensor_points(predictions, test_coords, output_html="sensor_viz.html"):
    data = []
    for idx, (sensor_id, point) in enumerate(domain_vars.TEMP_SENS_POINTS.items()):
        mask_point = np.isclose(test_coords[:, :3], point).all(axis=1)
        times = test_coords[mask_point, 3]
        temp = predictions[mask_point, 0]
        alpha = predictions[mask_point, 1]

        data.append(go.Scatter(
            x=times / 3600, y=temp,
            mode='lines+markers',
            name=f"Sensor {idx+1} Temperature",
            yaxis='y1'
        ))
        data.append(go.Scatter(
            x=times / 3600, y=alpha,
            mode='lines+markers',
            name=f"Sensor {idx+1} Alpha",
            yaxis='y2'
        ))

    layout = go.Layout(
        title="Sensor Point Predictions Over Time",
        xaxis=dict(title="Time (hours)"),
        yaxis=dict(title="Temperature", side='left'),
        yaxis2=dict(title="Alpha", overlaying='y', side='right'),
        height=600
    )

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(output_html)
    print(f"Sensor point visualization exported to {output_html}")

def create_vis_loss_history(checkpoint_path="checkpoints/best_model.pth", output_html="loss_history.html"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    if 'loss_history' not in checkpoint:
        print(f"Error: 'loss_history' dictionary not found in '{checkpoint_path}'")
        return
    loss_history = checkpoint['loss_history']

    epochs = np.arange(1, len(loss_history['total']) + 1)
    fig = go.Figure()

    for component, values in loss_history.items():
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=values, 
            mode='lines+markers', 
            name=component.replace("_", " ").title() # e.g., 'T_ic' -> 'T Ic'
        ))

    fig.update_layout(
        title=f"Training Loss History from {os.path.basename(checkpoint_path)}",
        xaxis_title="Epoch",
        yaxis_title="Loss (Log Scale)",
        yaxis_type="log",
        height=600,
        legend_title="Loss Components"
    )
    
    fig.write_html(output_html)
    print(f"Loss history visualization exported to '{output_html}'")



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


def export_sensors_to_csv(predictions, test_coords, idx_path=None):
    """
    Interpolates data at sensor locations for all time steps and saves to CSV.
    """
    if idx_path is None:
        content_dir = os.listdir(os.path.join('content'))
        if not content_dir:
            raise ValueError("No content directory found for exporting VTK files.")
        idx_path = os.path.join('content', content_dir[-1])
    output_file = os.path.join(idx_path, "sensors_interpolated.csv")

    all_times = np.unique(test_coords[:, 3])
    all_times.sort()
    
    # Prepare header
    header = ["Time_s", "Time_h"]
    sensor_ids = list(domain_vars.TEMP_SENS_POINTS.keys())
    for sid in sensor_ids:
        header.append(f"{sid}_Temp")
        header.append(f"{sid}_Alpha")
    
    # Collect data rows
    rows = []
    
    print(f"Interpolating sensors for CSV export...")
    
    for t in all_times:
        row = [t, t/3600.0]
        
        # Get field data for this timestep
        mask_t = np.isclose(test_coords[:, 3], t)
        coords_t = test_coords[mask_t, :3]
        pred_t = predictions[mask_t]
        
        for sid in sensor_ids:
            point = domain_vars.TEMP_SENS_POINTS[sid]
            
            # Interpolate
            temp_val = griddata(coords_t, pred_t[:, 0], point, method='linear')
            alpha_val = griddata(coords_t, pred_t[:, 1], point, method='linear')
            
            # Fallback to nearest if linear fails (e.g. point on boundary)
            if np.isnan(temp_val):
                temp_val = griddata(coords_t, pred_t[:, 0], point, method='nearest')
            if np.isnan(alpha_val):
                alpha_val = griddata(coords_t, pred_t[:, 1], point, method='nearest')
                
            row.append(float(temp_val))
            row.append(float(alpha_val))
            
        rows.append(row)
        
    # Write to CSV
    np.savetxt(output_file, np.array(rows), delimiter=",", header=",".join(header), comments="")
    print(f"Sensor data exported to '{output_file}'")