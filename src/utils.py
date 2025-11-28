import os
import torch
from pina import LabelTensor # Ensure LabelTensor is imported if checking isinstance

from dataclasses import fields, is_dataclass
import typing

from src.state_management.state import State

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


def get_field_type(ty):
    """Determines the input type based on the Python type hint."""
    # Handle Optional[T]
    if typing.get_origin(ty) is typing.Union and type(None) in typing.get_args(ty):
         ty = typing.get_args(ty)[0]

    if is_dataclass(ty):
        return 'group'
    
    # Handle Lists and Dicts -> JSON Editor
    origin = typing.get_origin(ty)
    if origin in (list, dict, typing.List, typing.Dict):
        return 'json'
        
    if ty is int: return 'integer'
    if ty is float: return 'number'
    if ty is bool: return 'boolean'
    return 'text'

def generate_schema(cls):
    """Recursively builds a dictionary describing the dataclass structure."""
    schema = {}
    for f in fields(cls):
        field_type = get_field_type(f.type)
        
        item = {
            'label': f.metadata.get('label', f.name.replace('_', ' ').title()),
            'type': f.metadata.get('type', field_type), # Allow metadata to override
            'unit': f.metadata.get('unit', ''),
        }
        
        if field_type == 'group':
            # Recursively generate schema for nested dataclass
            item['sub_schema'] = generate_schema(f.type)
            
        schema[f.name] = item
    return schema
