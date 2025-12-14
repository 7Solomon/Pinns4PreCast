import os
import torch
from pina import LabelTensor # Ensure LabelTensor is imported if checking isinstance

from dataclasses import fields, is_dataclass
import typing


def unscale_T(T_scaled):
    return (T_scaled * State().domain.T_c) + State().material.Temp_ref

def scale_T(T_actual):
    return (T_actual - State().material.Temp_ref) / State().domain.T_c

def unscale_alpha(alpha_scaled):
    return alpha_scaled * State().material.deg_hydr_max

def scale_alpha(alpha_actual):
    return alpha_actual / State().material.deg_hydr_max

def scale_domain(coords):
    if isinstance(coords, LabelTensor):
        coords_scaled = coords.clone()
        data = coords_scaled.as_subclass(torch.Tensor)


        idx_x = coords.labels.index('x')
        idx_y = coords.labels.index('y')
        idx_z = coords.labels.index('z')
        idx_t = coords.labels.index('t')
        
        data[:, idx_x] /= State().domain.L_c
        data[:, idx_y] /= State().domain.L_c
        data[:, idx_z] /= State().domain.L_c
        data[:, idx_t] /= State().domain.t_c
        return coords_scaled
    else:
        coords_scaled = coords.clone()
        coords_scaled[:, 0] /= State().domain.L_c
        coords_scaled[:, 1] /= State().domain.L_c
        coords_scaled[:, 2] /= State().domain.L_c
        coords_scaled[:, 3] /= State().domain.t_c
        return coords_scaled

def unscale_domain(coords_scaled):
    if isinstance(coords_scaled, LabelTensor):
        coords = coords_scaled.clone()
        data = coords.as_subclass(torch.Tensor)
        
        idx_x = coords.labels.index('x')
        idx_y = coords.labels.index('y')
        idx_z = coords.labels.index('z')
        idx_t = coords.labels.index('t')
        
        data[:, idx_x] *= State().domain.L_c
        data[:, idx_y] *= State().domain.L_c
        data[:, idx_z] *= State().domain.L_c
        data[:, idx_t] *= State().domain.t_c

        return coords
    else:
        coords = coords_scaled.clone()
        coords[:, 0] *= State().domain.L_c
        coords[:, 1] *= State().domain.L_c
        coords[:, 2] *= State().domain.L_c
        coords[:, 3] *= State().domain.t_c
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


def read_files_to_map(directory):
    files_map = {}
    if not os.path.exists(directory):
        return files_map
        
    try:
        sorted_files = sorted(
            [f for f in os.listdir(directory) if f.endswith('.csv')],
            key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x else 0
        )
    except:
        sorted_files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])
    for filename in sorted_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            files_map[filename] = f.read() # Read content as string
    return files_map



def get_pydantic_schema(cls):
    """
    Extracts JSON Schema from a Pydantic Dataclass.
    Also fixes the 'label' vs 'title' issue since you used metadata={'label':...}
    """
    schema = cls.__pydantic_model__.model_json_schema()
    
    def fix_labels(props):
        for key, value in props.items():
            if 'metadata' in value and 'label' in value['metadata']:
                value['title'] = value['metadata']['label']
            if 'properties' in value:
                fix_labels(value['properties'])
                
    if 'properties' in schema:
        fix_labels(schema['properties'])
        
    return schema