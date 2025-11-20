import json
from typing import List
import torch

from dataclasses import asdict, dataclass, field
from pyparsing import Any, Dict
import os

@dataclass
class DatasetConfig:
    n_pde: int = 500
    n_ic: int = 100
    n_bc_face: int = 50
    batch_size: int = 32
    num_samples: int = 10000  # Total training samples in the dataset


@dataclass
class ModelConfig:
    num_sensors_bc: int = 100
    num_sensors_ic: int = 100 

    num_outputs: int = 2 
    latent_dim: int = 100
    branch_configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'input_size': 100, 'hidden_layers': [256, 256, 256]},
        {'input_size': 100, 'hidden_layers': [256, 256, 256]}
    ])
    
    trunk_config: Dict[str, Any] = field(default_factory=lambda: {
        'input_size': 4, 'hidden_layers': [256, 256, 256] # x,y,z,t 
    })
    

@dataclass
class TrainingConfig:
    max_epochs: int = 100
    learning_rate: float = 1e-4
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'physics': 1.0, 'bc': 1.0, 'ic': 1.0})
    
    optimizer_type: str = 'adam'
    scheduler_type : str = 'ReduceLROnPlateau'
    mode: str ='min'
    factor: float = 0.5
    patience: int = 15

    @property
    def optimizer(self):
        return getattr(torch.optim, self.optimizer_type)
    @property
    def scheduler(self):
        return getattr(torch.optim.lr_scheduler, self.scheduler_type)
    
@dataclass
class DirectoryConfig:
    content_path: str = os.path.join('content')

    idx_path: str = None
    @property
    def checkpoint_path(self):
        return os.path.join(self.idx_path, 'checkpoints')
    @property
    def log_path(self):
        return os.path.join(self.idx_path, 'logs')
    @property
    def vtk_path(self):
        return os.path.join(self.idx_path, 'vtk')
    @property
    def sensor_alpha_path(self):
        return os.path.join(self.idx_path, 'sensor_alpha.csv')
    @property
    def sensor_temp_path(self):
        return os.path.join(self.idx_path, 'sensor_temperature.csv')
    


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    directory: DirectoryConfig = field(default_factory=DirectoryConfig)

    def save(self, path: str):
        """Saves the config to a JSON file."""
        config_dict = asdict(self)        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, path: str):
        """Loads the config from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            dataset=DatasetConfig(**data.get('dataset', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            directory=DirectoryConfig(**data.get('directory', {}))
        )