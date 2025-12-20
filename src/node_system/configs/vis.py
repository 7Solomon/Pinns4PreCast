from pydantic import BaseModel, Field

from src.node_system.configs.dataset import DatasetConfig
from src.node_system.configs.model_input import InputConfig


class VisualizationConfig(BaseModel):
    save_dir: str = Field(default="content/runs", title="Save Directory")
    plot_every_n_epochs: int = Field(default=10, title="Plot Every n Epoch ")

class CompositeVisualizationConfig(BaseModel):
    input_config: InputConfig = Field(default_factory=InputConfig) 
    data_config: DatasetConfig = Field(default_factory=DatasetConfig)
    vis_config: VisualizationConfig = Field(default_factory=VisualizationConfig)