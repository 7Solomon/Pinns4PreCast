from pydantic import BaseModel, Field


class CheckpointConfig(BaseModel):
    monitor: str = Field(default="loss_epoch", description="Metric to monitor")
    save_top_k: int = Field(default=3, description="Number of top models to save")
    save_last: bool = Field(default=True, description="Save last epoch")
    every_n_epochs: int = Field(default=1, description="Save every N epochs")