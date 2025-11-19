import torch
from torch.utils.data import Dataset
from pina import LabelTensor


from material import ConcreteData
material_data = ConcreteData()

## BC
def sample_temperature_bc_params(num_samples, device='cpu'):
    amplitude = 4 + torch.rand(num_samples, 1, device=device) * 2
    phase = torch.rand(num_samples, 1, device=device) * 2 * torch.pi
    offset = material_data.Temp_ref
    return amplitude, phase, offset

def eval_temperature_bc(points, amplitude, phase, offset):
    return offset + amplitude * torch.sin(0.5 * torch.pi * points + phase)

## IC
def sample_temperature_ic_params(num_samples, device='cpu'):
    T0 = material_data.Temp_ref + 2 * torch.randn(num_samples, 1, device=device)
    return T0
def eval_temperature_ic(points, T0):
    return T0.expand_as(points)

class DeepONetDataset(Dataset):
    """
    A map-style dataset for Physics-Informed DeepONet.
    """
    def __init__(self, problem, n_pde, n_ic, n_bc_face, num_samples, num_sensors=100, device='cpu'):
        self.problem = problem
        self.n_pde = n_pde
        self.n_ic = n_ic
        self.n_bc_face = n_bc_face
        self.num_samples = num_samples
        self.num_sensors = num_sensors
        self.boundary_faces = ["left", "right", "bottom", "top", "front", "back"]
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        amp, phase, offset = sample_temperature_bc_params(1, device='cpu')  # FIRST CPU THEN CUDA AT THE END, bECAUSE PINAD WEIRD ON CPU
        T0 = sample_temperature_ic_params(1, device='cpu')
        
        sensor_locations = torch.linspace(0, 1, self.num_sensors, device='cpu').unsqueeze(0)
        bc_sensors = eval_temperature_bc(sensor_locations, amp, phase, offset).squeeze(0)
        ic_sensors = eval_temperature_ic(sensor_locations, T0).squeeze(0)

        pde_pts = self.problem.domain["D"].sample(n=self.n_pde)
        ic_pts = self.problem.domain["initial"].sample(n=self.n_ic)
        bc_pts_list = [self.problem.domain[face].sample(n=self.n_bc_face) for face in self.boundary_faces]
        bc_pts = LabelTensor.cat(bc_pts_list)

        x_ic = ic_pts.extract(['x']).unsqueeze(0)
        ic_T_vals = eval_temperature_ic(x_ic, T0).squeeze(0)
        ic_alpha_vals = torch.ones_like(ic_T_vals) * 1e-6
        ic_vals = torch.stack([ic_T_vals, ic_alpha_vals], dim=-1)  # Stack T and alpha

        x_bc = bc_pts.extract(['x']).unsqueeze(0)
        bc_T_vals = eval_temperature_bc(x_bc, amp, phase, offset).squeeze(0)

        # Convert target values to regular tensors
        ic_vals = ic_vals.as_subclass(torch.Tensor) if isinstance(ic_vals, LabelTensor) else ic_vals
        bc_T_vals = bc_T_vals.as_subclass(torch.Tensor) if isinstance(bc_T_vals, LabelTensor) else bc_T_vals

        # MOVE TO DEVICE IF NEEDED
        if self.device != 'cpu':
            bc_sensors = bc_sensors.to(self.device)
            ic_sensors = ic_sensors.to(self.device)
            pde_pts = pde_pts.to(self.device)
            ic_pts = ic_pts.to(self.device)
            bc_pts = bc_pts.to(self.device)
            ic_vals = ic_vals.to(self.device)
            bc_T_vals = bc_T_vals.to(self.device)


        return {
            "bc_sensors": bc_sensors,  # (num_sensors,)
            "ic_sensors": ic_sensors,  # (num_sensors,)
            "pde_coords": pde_pts,     # (n_pde, 4) - LabelTensor
            "ic_coords": ic_pts,       # (n_ic, 4) - LabelTensor
            "bc_coords": bc_pts,       # (n_bc_total, 4) - LabelTensor
            "ic_target": ic_vals,    # (n_ic, 2) - Temperature and alpha values
            "bc_target": bc_T_vals,    # (n_bc_total,) - Temperature values
        }
    

###
##
##

def deeponet_collate_fn(batch):
    """Custom collate function to batch LabelTensor objects."""
    
    bc_sensors = torch.stack([item["bc_sensors"] for item in batch])
    ic_sensors = torch.stack([item["ic_sensors"] for item in batch])
    
    # Stack coordinates
    pde_coords = torch.stack([item["pde_coords"].as_subclass(torch.Tensor) for item in batch])
    ic_coords = torch.stack([item["ic_coords"].as_subclass(torch.Tensor) for item in batch])
    bc_coords = torch.stack([item["bc_coords"].as_subclass(torch.Tensor) for item in batch])
    
    # Stack targets
    ic_target = torch.stack([item["ic_target"] for item in batch])
    bc_target = torch.stack([item["bc_target"] for item in batch])
    
    return {
        "bc_sensors": bc_sensors,
        "ic_sensors": ic_sensors,
        "pde_coords": pde_coords,
        "ic_coords": ic_coords,
        "bc_coords": bc_coords,
        "ic_target": ic_target,
        "bc_target": bc_target
    }