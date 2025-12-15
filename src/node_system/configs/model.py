from pydantic import BaseModel, Field
from typing import Any, Dict, List

from src.node_system.configs.model_input import InputConfig

class ModelConfig(BaseModel):
    num_outputs: int = Field(default=2, title="Model Outputs")
    latent_dim: int = Field(default=256, title="Latent Dimension")
    
    branch_hidden_layers: List[int] = Field(default_factory=lambda: [256, 256, 256], title="Branch Configs")
    
    trunk_config: Dict[str, Any] = Field(default_factory=lambda: {'input_size': 4, 'hidden_layers': [256, 256, 256]}, title="Trunk Config")

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

class CompositeModelConfig(BaseModel):
    model_hyperparameter: ModelConfig = Field(default_factory=ModelConfig)
    input_config: InputConfig = Field(default_factory=InputConfig) 


    