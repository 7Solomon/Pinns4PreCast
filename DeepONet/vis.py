import numpy as np
import plotly.graph_objects as go

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