import json
from typing import List, Dict, Any
import torch
from pina.optim import TorchOptimizer
from pina.optim import TorchScheduler

from pydantic import BaseModel, Field
import os

from src.class_definition.base_state import BaseState

class DatasetConfig(BaseModel):
    n_pde: int = Field(default=500, title="PDE Samples")
    n_ic: int = Field(default=100, title="IC Samples")
    n_bc_face: int = Field(default=50, title="BC Samples per face")
    batch_size: int = Field(default=32, title="Batch Size")
    num_samples: int = Field(default=10000, title="Total Samples")
    num_workers: int = Field(default=0, title="DataLoader Workers") # Changed to 0, 91 is unusually high

class ModelConfig(BaseModel):
    num_sensors_bc: int = Field(default=100, title="BC Sensor Count")
    num_sensors_ic: int = Field(default=100, title="IC Sensor Count")
    num_outputs: int = Field(default=2, title="Model Outputs")
    latent_dim: int = Field(default=256, title="Latent Dimension")
    
    branch_configs: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {'input_size': 100, 'hidden_layers': [256, 256, 256]},
        {'input_size': 100, 'hidden_layers': [256, 256, 256]}
    ], title="Branch Configs")
    
    trunk_config: Dict[str, Any] = Field(default_factory=lambda: {
        'input_size': 4, 'hidden_layers': [256, 256, 256]
    }, title="Trunk Config")

    fourier_features: Dict[str, Any] = Field(default_factory=lambda: {
        'input_dim': 4,
        'mapping_size': 64,
        'scale': 5.0
    }, title="Fourier Features Config")

    activation_types: Dict[str, str] = Field(default_factory=lambda: {
        'branch': 'Tanh',
        'trunk': 'SiLU'
    }, title="Activation Types")

    @property
    def activation(self):
        import torch.nn as nn
        return {
            'branch': getattr(nn, self.activation_types['branch']),
            'trunk': getattr(nn, self.activation_types['trunk'])
        }

class TrainingConfig(BaseModel):
    max_epochs: int = Field(default=100, title="Max Epochs")
    loss_weights: Dict[str, float] = Field(default_factory=lambda: {'physics': 100.0, 'bc': 1.0, 'ic': 10.0}, title="Loss Weights")
    time_weighted_loss: Dict[str, Any] = Field(default_factory=lambda: {'time_decay_rate': 5.0}, title="Time Weighted Loss")
    
    optimizer_type: str = Field(default='Adam', title="Optimizer")
    optimizer_learning_rate: float = Field(default=1e-4, title="Learning Rate")

    scheduler_type: str = Field(default='ReduceLROnPlateau', title="Scheduler")
    scheduler_mode: str = Field(default='min', title="Scheduler Mode")
    scheduler_factor: float = Field(default=0.5, title="Scheduler Factor")
    scheduler_patience: int = Field(default=15, title="Scheduler Patience")

    @property
    def optimizer(self):
        return TorchOptimizer(getattr(torch.optim, self.optimizer_type), lr=self.optimizer_learning_rate)
    
    @property
    def scheduler(self):
        return TorchScheduler(getattr(torch.optim.lr_scheduler, self.scheduler_type), mode=self.scheduler_mode, factor=self.scheduler_factor, patience=self.scheduler_patience)

class Config(BaseModel, BaseState):
    # FOR FRONTEDND
    model_config = {
        "json_schema_extra": {
            "format": "tabs"
        }
    }
    dataset: DatasetConfig = Field(default_factory=DatasetConfig, title="Dataset Config")
    model: ModelConfig = Field(default_factory=ModelConfig, title="Model Config")
    training: TrainingConfig = Field(default_factory=TrainingConfig, title="Training Config")

    @classmethod
    def load(cls, path: str):
        """Loads the config from a JSON file using Pydantic's built-in validation."""
        if not os.path.exists(path):
            # Assumes your BaseState.create_default method works as intended
            return cls.create_default(path)
        
        with open(path, 'r') as f:
            # This reads the JSON, validates it against the schema, and creates all nested models.
            return cls.model_validate_json(f.read())
