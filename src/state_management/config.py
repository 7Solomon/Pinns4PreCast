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
    n_pde: int = field(default=500, metadata={"label": "PDE Samples", "type": "integer"})
    n_ic: int = field(default=100, metadata={"label": "IC Samples", "type": "integer"})
    n_bc_face: int = field(default=50, metadata={"label": "BC Samples per face", "type": "integer"})
    batch_size: int = field(default=32, metadata={"label": "Batch Size", "type": "integer"})
    num_samples: int = field(default=10000, metadata={"label": "Total Samples", "type": "integer"})  # Total training samples in the dataset

    num_workers: int = field(default=91, metadata={"label": "DataLoader Workers", "type": "integer"})

@dataclass
class ModelConfig:
    num_sensors_bc: int = field(default=100, metadata={"label": "BC Sensor Count", "type": "integer"})
    num_sensors_ic: int = field(default=100, metadata={"label": "IC Sensor Count", "type": "integer"})

    num_outputs: int = field(default=2, metadata={"label": "Model Outputs", "type": "integer"})
    latent_dim: int = field(default=256, metadata={"label": "Latent Dimension", "type": "integer"})
    branch_configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'input_size': 100, 'hidden_layers': [256, 256, 256]},
        {'input_size': 100, 'hidden_layers': [256, 256, 256]}
    ], metadata={"label": "Branch Configs", "type": "list"})
    
    trunk_config: Dict[str, Any] = field(default_factory=lambda: {
        'input_size': 4, 'hidden_layers': [256, 256, 256] # x,y,z,t 
    }, metadata={"label": "Trunk Config", "type": "dict"})

    fourier_features: Dict[str, Any] = field(default_factory=lambda: {
        'input_dim': 4,       # x, y, z, t
        'mapping_size': 64,   # 128 features (sin + cos)
        'scale': 5.0          # frequency RANGE
    }, metadata={"label": "Fourier Features Config", "type": "dict"})

    activation_types: Dict[str, str] = field(default_factory=lambda: {
        'branch': 'Tanh',
        'trunk': 'SiLU'
    }, metadata={"label": "Activation Types", "type": "dict"})
    @property
    def activation(self):
        import torch.nn as nn
        return {
            'branch': getattr(nn, self.activation_types['branch']),
            'trunk': getattr(nn, self.activation_types['trunk'])
        }

    

@dataclass
class TrainingConfig:
    max_epochs: int = field(default=100, metadata={"label": "Max Epochs", "type": "integer"})
    loss_weights: Dict[str, float] = field(default_factory=lambda: {'physics': 100.0, 'bc': 1.0, 'ic': 10.0}, metadata={"label": "Loss Weights", "type": "dict"})
    time_weighted_loss: Dict[str, Any] = field(default_factory=lambda: {'time_decay_rate': 5.0}, metadata={"label": "Time Weighted Loss", "type": "boolean"})
    
    optimizer_type: str = field(default='Adam', metadata={"label": "Optimizer", "type": "text"})
    optimizer_learning_rate: float = field(default=1e-4, metadata={"label": "Learning Rate", "type": "number"})


    scheduler_type : str = field(default='ReduceLROnPlateau', metadata={"label": "Scheduler", "type": "text"})
    scheduler_mode: str = field(default='min', metadata={"label": "Scheduler Mode", "type": "text"})
    scheduler_factor: float = field(default=0.5, metadata={"label": "Scheduler Factor", "type": "number"})
    scheduler_patience: int = field(default=15, metadata={"label": "Scheduler Patience", "type": "integer"})

    @property
    def optimizer(self):
        return TorchOptimizer(getattr(torch.optim, self.optimizer_type), lr=self.optimizer_learning_rate)
    @property
    def scheduler(self):
        return TorchScheduler(getattr(torch.optim.lr_scheduler, self.scheduler_type),  mode=self.scheduler_mode, factor=self.scheduler_factor, patience=self.scheduler_patience)
    
@dataclass
class Config(BaseState):
    dataset: DatasetConfig = field(default_factory=DatasetConfig, metadata={"label": "Dataset Config", "type": "group"})
    model: ModelConfig = field(default_factory=ModelConfig, metadata={"label": "Model Config", "type": "group"})
    training: TrainingConfig = field(default_factory=TrainingConfig, metadata={"label": "Training Config", "type": "group"})


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
