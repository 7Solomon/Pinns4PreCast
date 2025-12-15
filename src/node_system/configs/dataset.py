from pydantic import BaseModel, Field
from typing import Any, Dict, List

from src.node_system.configs.model_input import InputConfig
    

class DatasetConfig(BaseModel):
    n_pde: int = Field(default=500, title="Number of PDE Samples")
    n_ic: int = Field(default=100, title="Number of IC Samples")
    n_bc_face: int = Field(default=50, title="Number of BC Samples per face")
    num_samples: int = Field(default=10000, title="Number of Total Samples")

class CompositeDatasetConfig(BaseModel):
    data_config: DatasetConfig = Field(default_factory=DatasetConfig)
    input_config: InputConfig = Field(default_factory=InputConfig) 