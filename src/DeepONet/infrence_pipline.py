import os
import torch
from scipy.stats import qmc


from src.DeepONet.dataset import DeepONetDataset
from src.DeepONet.vis import export_to_vtk_series, export_sensors_to_csv

from src.utils import unscale_T


""""

    Functions for the infrence Call of a model so Point grid creation and then the Visulasation of the output

"""

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
                     n_spatial=10, n_time=10, num_sensors_bc=100, num_sensors_ic=100):
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
        content_dir = os.listdir(os.path.join('content'))
        idx_path = os.path.join('content', content_dir[-1])

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
        ds = DeepONetDataset(
            problem=solver.problem,
            n_pde=1,      # NOT NEEDED beause 
            n_ic=1,       # NOT NEEDED
            n_bc_face=1,  # SAME
            num_samples=1,
            num_sensors_bc=num_sensors_bc,
            num_sensors_ic= num_sensors_ic
        )
        
        sample = ds[0]
        bc_target_temperature = sample['bc_sensor_values'].unsqueeze(0)    # DELETET CHANGED TO USE SOMEHTING ELSE
        ic_target_temperature = sample['ic_sensor_values'].unsqueeze(0)

        # BECAUSE WE LIKE STRUTURED GRID FOR CHECKING
        test_coords = create_test_grid(
            spatial_domain=[State().domain.x, State().domain.y, State().domain.z], 
            time_domain=State().domain.t,
            n_spatial=n_spatial,
            n_time=n_time
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
        predictions_unscaled[:, 0] = unscale_T(predictions[:, 0]) - 273.15
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
            test_coords.cpu(),
            idx_path=idx_path
        )
        return predictions_unscaled, test_coords