import torch
from torch.utils.data import Dataset
from pina import LabelTensor

from src.utils import scale_T, scale_domain

from src.state_management.state import State


def sample_random_field_params(num_features=10, device='cpu'):
    """
    Generates random frequencies and phases for a smooth random field.
    f(x) = sum(A * cos(k*x + phi))
    """
    # Random frequencies (k) determine how "bumpy" the noise is
    # Lower scale = smoother blobs. Higher scale = high freq noise.
    k = torch.randn(num_features, 3, device=device) * 3.0 
    
    # Random phases
    phi = torch.rand(num_features, device=device) * 2 * torch.pi
    
    # Random amplitudes
    A = torch.randn(num_features, device=device)
    
    # Base offset (Mean temperature)
    offset = State().material.Temp_ref + torch.randn(1, device=device) * 5.0 # +/- 5 deg variation
    
    return k, phi, A, offset

def eval_random_field(points, k, phi, A, offset):
    """
    Evaluates the random field at specific points.
    points: (N, 4) or (N, 3) [x, y, z, (t)]
    """
    # 1. Extract Spatial Coordinates (x, y, z)
    # We ignore Time (t) because BCs are stationary!
    if isinstance(points, LabelTensor):
        coords = points.extract(['x', 'y', 'z']) # (N, 3)
    else:
        coords = points[:, :3] # Assume first 3 are x,y,z
        
    # 2. Compute Random Fourier Features
    # (N, 3) @ (3, num_features) -> (N, num_features)
    projection = torch.matmul(coords, k.T) 
    
    # cos(k*x + phi)
    features = torch.cos(projection + phi)
    
    noise = torch.matmul(features, A) / torch.sqrt(torch.tensor(len(A), dtype=torch.float64))
    
    return offset + noise

class DeepONetDataset(Dataset):
    """
    A map-style dataset for Physics-Informed DeepONet.
    """
    def __init__(self, problem, n_pde, n_ic, n_bc_face, num_samples, num_sensors_bc, num_sensors_ic):
        self.problem = problem
        self.n_pde = n_pde
        self.n_ic = n_ic
        self.n_bc_face = n_bc_face
        self.num_samples = num_samples
        self.num_sensors_bc = num_sensors_bc
        self.num_sensors_ic = num_sensors_ic
        self.boundary_faces = ["left", "right", "bottom", "top", "front", "back"]


        sensors_per_face = (num_sensors_bc // 6) + 1
        bc_sensor_list = [self.problem.domain[face].sample(n=sensors_per_face) for face in self.boundary_faces]
        self.bc_sensor_coords = LabelTensor.cat(bc_sensor_list)[:num_sensors_bc]    # IMPORTANT TO SLICE TO EXACT NUMBER
        self.ic_sensor_coords = self.problem.domain["D"].sample(n=num_sensors_ic)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        k_bc, phi_bc, A_bc, offset_bc = sample_random_field_params(num_features=20)   # BC MORE FEATURES
        k_ic, phi_ic, A_ic, offset_ic = sample_random_field_params(num_features=5)  # IC LESS FEATURES
        A_ic = A_ic * 0.2  # BEcause IC in concrete GOOD MIXED 

        pde_pts = self.problem.domain["D"].sample(n=self.n_pde)
        
        ic_pts = self.ic_sensor_coords
        bc_pts = self.bc_sensor_coords

        # Calculate Targets
        # IC
        ic_T_vals = scale_T(eval_random_field(ic_pts, k_ic, phi_ic, A_ic, offset_ic)).squeeze(-1)
        ic_alpha_vals = torch.ones_like(ic_T_vals) * 1e-6 
        
        # BC
        bc_T_vals = scale_T(eval_random_field(bc_pts, k_bc, phi_bc, A_bc, offset_bc)).squeeze(-1)

        # 4. Scaling & Formatting
        pde_pts = scale_domain(pde_pts).as_subclass(torch.Tensor)
        ic_pts = scale_domain(ic_pts).as_subclass(torch.Tensor)
        bc_pts = scale_domain(bc_pts).as_subclass(torch.Tensor)
        
        ic_T_vals = ic_T_vals.as_subclass(torch.Tensor)
        ic_alpha_vals = ic_alpha_vals.as_subclass(torch.Tensor)
        bc_T_vals = bc_T_vals.as_subclass(torch.Tensor)

        return {
            "pde_coords": pde_pts,             
            "ic_coords": ic_pts,               
            "bc_coords": bc_pts,               
            "ic_target_temperature": ic_T_vals, 
            "ic_target_alpha": ic_alpha_vals,
            "bc_target_temperature": bc_T_vals,
        }
    

def deeponet_collate_fn(batch):
    """Custom collate function."""
    return {
        "pde_coords": torch.stack([item["pde_coords"] for item in batch]),
        "ic_coords": torch.stack([item["ic_coords"] for item in batch]),
        "bc_coords": torch.stack([item["bc_coords"] for item in batch]),
        "ic_target_temperature": torch.stack([item["ic_target_temperature"] for item in batch]),
        "ic_target_alpha": torch.stack([item["ic_target_alpha"] for item in batch]),
        "bc_target_temperature": torch.stack([item["bc_target_temperature"] for item in batch]),
    }