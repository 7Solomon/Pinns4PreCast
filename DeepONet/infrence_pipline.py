import os
import torch
from scipy.stats import qmc


from DeepONet.data_loader import eval_temperature_bc, eval_temperature_ic, sample_temperature_bc_params, sample_temperature_ic_params
from DeepONet.vis import create_unified_interactive_viz_v2, create_vis_loss_history, create_vis_on_sensor_points, export_to_vtk_series, export_sensors_to_csv
from domain import DomainVariables
from material import ConcreteData
from utils import scale_domain, unscale_T, unscale_alpha
material_data = ConcreteData()
domain_vars = DomainVariables()


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
def testFlexDeepONet(solver, idx_path=None, 
                     n_spatial=10, n_time=10, num_sensors=100):
    """
    Run inference with the trained DeepONet model.
    
    Args:
        solver: DeepONetSolver instance (can be already trained or will load from checkpoint)
        checkpoint_path: Path to saved Lightning checkpoint (optional if solver already trained)
        n_spatial: Number of spatial grid points per dimension
        n_time: Number of time points
        num_sensors: Number of sensor points for BC/IC
    """
    # Load checkpoint if provided
    if idx_path is None:
        content_dir = os.listdir(os.path.join('content', 'checkpoints'))
        idx_path = os.path.join('content', 'checkpoints', content_dir[-1])

    dir = os.listdir(os.path.join(idx_path, 'checkpoints'))
    print(f"Loading checkpoint from: {os.path.join(idx_path, 'checkpoints', dir[-1])}")
    checkpoint = torch.load(os.path.join(idx_path, 'checkpoints', dir[-1]), map_location='cpu')
    solver.load_state_dict(checkpoint['state_dict'])

    
    # Set to eval mode
    solver.eval()
    
    # Get model device
    device = next(solver.model.parameters()).device
    print(f"Running inference on device: {device}")
    
    with torch.no_grad():
        # Generate BC and IC sensor data (1 sample for testing)
        amp, phase, offset = sample_temperature_bc_params(1, device=device)
        T0 = sample_temperature_ic_params(1, device=device)
        
        sensor_locations = torch.linspace(0, 1, num_sensors, device=device).unsqueeze(0)
        bc_sensors = eval_temperature_bc(sensor_locations, amp, phase, offset)  # [1, num_sensors]
        ic_sensors = eval_temperature_ic(sensor_locations, T0)  # [1, num_sensors]

        print(f"\nBC sensors shape: {bc_sensors.shape}")
        print(f"IC sensors shape: {ic_sensors.shape}")
        print(f"BC sample values: {bc_sensors[0, :5]} ... {bc_sensors[0, -5:]}")
        print(f"IC sample values: {ic_sensors[0, :5]} ... {ic_sensors[0, -5:]}")
        
        # Create structured test grid
        test_coords = create_test_grid(
            spatial_domain=[domain_vars.x, domain_vars.y, domain_vars.z], 
            time_domain=domain_vars.t,
            n_spatial=n_spatial,
            n_time=n_time
        ).to(device)
        
        print(f"\nTest coordinates shape: {test_coords.shape}")
        print(f"Total points: {test_coords.shape[0]}")
        print(f"Sample coordinates:\n{test_coords[:5]}\n...\n{test_coords[-5:]}")

        # Add batch dimension to test_coords: [1, num_points, 4]
        test_coords_batched = test_coords.unsqueeze(0)
        
        # Create batch dict for inference
        test_batch = {
            'bc_sensors': bc_sensors,  # [1, num_sensors]
            'ic_sensors': ic_sensors,  # [1, num_sensors]
            'query_coords': test_coords_batched  # [1, num_points, 4]
        }
        
        # Make predictions using the solver's forward method
        print("\nRunning forward pass...")
        predictions = solver.forward(test_batch)  # [1, num_points, 2]
        predictions = predictions.squeeze(0)  # [num_points, 2]

        # Unscale predictions
        predictions_unscaled = predictions.clone()
        predictions_unscaled[:, 0] = unscale_T(predictions[:, 0]) - 273.15
        predictions_unscaled[:, 1] = unscale_alpha(predictions[:, 1])

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
        #create_unified_interactive_viz_v2(
        #    predictions_unscaled.cpu(), 
        #    test_coords.cpu(), 
        #    bc_samples=bc_sensors.cpu(), 
        #    ic_samples=ic_sensors.cpu()
        #)
        #create_vis_on_sensor_points(predictions_unscaled.cpu(), test_coords.cpu())
        #create_vis_loss_history()
        
        
        return predictions_unscaled, test_coords