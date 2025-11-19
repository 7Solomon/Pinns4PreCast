import torch

from material import ConcreteData
from domain import DomainVariables
material_data = ConcreteData()
domain_vars = DomainVariables()

def unscale_T(T_scaled):
    return (T_scaled * domain_vars.T_c) + material_data.Temp_ref

def scale_T(T_actual):
    return (T_actual - material_data.Temp_ref) / domain_vars.T_c

def unscale_alpha(alpha_scaled):
    return alpha_scaled * material_data.deg_hydr_max

def scale_alpha(alpha_actual):
    return alpha_actual / material_data.deg_hydr_max

def scale_domain(coords):
    coords_scaled = coords.clone()
    coords_scaled[:, 0] = coords[:, 0] / domain_vars.L_c
    coords_scaled[:, 1] = coords[:, 1] / domain_vars.L_c
    coords_scaled[:, 2] = coords[:, 2] / domain_vars.L_c
    coords_scaled[:, 3] = coords[:, 3] / domain_vars.t_c
    return coords_scaled

def unscale_domain(coords_scaled):
    coords = coords_scaled.clone()
    coords[:, 0] = coords_scaled[:, 0] * domain_vars.L_c
    coords[:, 1] = coords_scaled[:, 1] * domain_vars.L_c
    coords[:, 2] = coords_scaled[:, 2] * domain_vars.L_c
    coords[:, 3] = coords_scaled[:, 3] * domain_vars.t_c
    return coords


def torch_interp1d(x_new, x_old, y_old):
    """Performs 1D linear interpolation using pure PyTorch on the GPU."""
    right_indices = torch.searchsorted(x_old, x_new).clamp(min=1, max=len(x_old) - 1)
    left_indices = right_indices - 1
    
    x_left, x_right = x_old[left_indices], x_old[right_indices]
    y_left, y_right = y_old[left_indices], y_old[right_indices]

    denom = x_right - x_left
    weight = (x_new - x_left) / torch.where(denom == 0, torch.ones_like(denom), denom)

    return y_left + weight * (y_right - y_left)
