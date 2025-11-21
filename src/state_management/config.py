import json
from typing import List, Dict, Any
import torch
from pina.optim import TorchOptimizer
from pina.optim import TorchScheduler

from dataclasses import dataclass, field
import os

from src.class_definition.base_state import BaseState


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

    activation_types: Dict[str, str] = field(default_factory=lambda: {
        'branch': 'Tanh',
        'trunk': 'Tanh'
    })
    @property
    def activation(self):
        import torch.nn as nn
        return {
            'branch': getattr(nn, self.activation_types['branch']),
            'trunk': getattr(nn, self.activation_types['trunk'])
        }

    

@dataclass
class TrainingConfig:
    max_epochs: int = 100
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'physics': 1.0, 'bc': 1.0, 'ic': 1.0})
    
    optimizer_type: str = 'Adam'
    optimizer_learning_rate: float = 1e-4


    scheduler_type : str = 'ReduceLROnPlateau'
    scheduler_mode: str ='min'
    scheduler_factor: float = 0.5
    scheduler_patience: int = 15

    @property
    def optimizer(self):
        return TorchOptimizer(getattr(torch.optim, self.optimizer_type), lr=self.optimizer_learning_rate)
    @property
    def scheduler(self):
        return TorchScheduler(getattr(torch.optim.lr_scheduler, self.scheduler_type),  mode=self.scheduler_mode, factor=self.scheduler_factor, patience=self.scheduler_patience)
    
@dataclass
class Config(BaseState):
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


    @classmethod
    def load(cls, path: str):
        """Loads the config from a JSON file."""
        if not os.path.exists(path):
            return cls.create_default(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            dataset=DatasetConfig(**data.get('dataset', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
        )
