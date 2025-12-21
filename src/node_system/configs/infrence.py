from pydantic import BaseModel, Field

from src.node_system.configs.model_input import InputConfig


class InferenceConfig(BaseModel):
    n_spatial: int = Field(15, title="Grid points per spatial axis")
    n_time: int = Field(15, title="Grid points for time")
    save_dir: str = Field("./results", title="Output directory")


class CompositeInferenceConfig(BaseModel):
    inf_cfg: InferenceConfig = Field(default_factory=InferenceConfig, title="Infrerence")
    inp_cfg: InputConfig = Field(default_factory=InputConfig, title="Input")

class RunIdChooserConfig(BaseModel):
    run_id: str = Field(..., title="Run ID", description="Selected run ID for inference")