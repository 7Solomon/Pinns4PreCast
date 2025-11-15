import pyvista as pv
import numpy as np

def export_to_vtk(predictions, test_coords, output_name='results'):
    """
    predictions: shape [N, num_outputs] - your model output
    test_coords: shape [N, 4] - your (x, y, z, t) coordinates
    output_name: name for the saved file
    """
    
    # Move to numpy if on GPU
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
        test_coords = test_coords.cpu().numpy()
    
    # Extract spatial coordinates only (x, y, z)
    points = test_coords[:, :3]
    
    # Create a PyVista point cloud from your coordinates
    cloud = pv.PolyData(points)
    
    # Attach your model outputs as "arrays" (data at each point)
    cloud['Temperature'] = predictions[:, 0]  # First output (Temperature)
    cloud['Alpha'] = predictions[:, 1]         # Second output (Alpha)
    cloud['Time'] = test_coords[:, 3]          # Also save time coordinate
    
    # Save to file
    cloud.save(f'{output_name}.vtp')
    print(f"✓ Saved: {output_name}.vtp")
