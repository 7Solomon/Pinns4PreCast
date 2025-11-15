import torch
from scipy.stats import qmc

from DeepONet.trainer import sample_temperature_bc, sample_temperature_ic

from DeepONet.vis import create_unified_interactive_viz_v2
from domain import DomainVariables
from material import ConcreteData
material_data = ConcreteData()
domain_vars = DomainVariables()


def create_test_grid(spatial_domain=[(0, 1),(0, 1),(0, 1)], time_domain=(0, 1), 
                     n_spatial=15, n_time=10):
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

def testFlexDeepONet(model, problem):
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Generate BC and IC
        bc_test = sample_temperature_bc(1, num_sensors=100).to(device)
        ic_test = sample_temperature_ic(1, num_sensors=100).to(device)

        print(f"\nBC test shape: {bc_test.shape}, BC sample:\n{bc_test[:, :5]} \n ... {bc_test[:, -5:]}")
        print(f"IC test shape: {ic_test.shape}, IC sample:\n{ic_test[:, :5]} \n ... {ic_test[:, -5:]}")
        
        # Structured grid
        test_coords = create_test_grid(
            spatial_domain=[domain_vars.x, domain_vars.y, domain_vars.z], 
            time_domain=domain_vars.t,
            n_spatial=10,
            n_time=10
        ).to(device)

        test_coords_scaled = test_coords.clone()
        test_coords_scaled[:, 0] = test_coords[:, 0] / domain_vars.L_c
        test_coords_scaled[:, 1] = test_coords[:, 1] / domain_vars.L_c
        test_coords_scaled[:, 2] = test_coords[:, 2] / domain_vars.L_c
        test_coords_scaled[:, 3] = test_coords[:, 3] / domain_vars.t_c

        print(f"\nTest coordinates shape: {test_coords.shape}")
        print(f"Sample coordinates:\n{test_coords[:5]}, ...\n{test_coords[-5:]}")
        
        # Make predictions
        branch_inputs = [bc_test, ic_test]
        trunk_inputs = [test_coords_scaled, test_coords_scaled]
        
        predictions = model(branch_inputs, trunk_inputs)
        predictions[:, 0] = predictions[:, 0] * domain_vars.T_c + (material_data.Temp_ref - 273.15)  # Scale back Temperature
        predictions[:, 1] = predictions[:, 1]

        print(f"\nPredictions shape: {predictions.shape}")
        print(f"Temperature range: [{predictions[:, 0].min():.4f}, {predictions[:, 0].max():.4f}]")
        print(f"Alpha range: [{predictions[:, 1].min():.4f}, {predictions[:, 1].max():.4f}]")
        
        # Visualize
        #visualize_all(predictions, test_coords)
        create_unified_interactive_viz_v2(predictions, test_coords, bc_samples=bc_test, ic_samples=ic_test)    
