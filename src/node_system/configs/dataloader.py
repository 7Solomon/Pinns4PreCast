from typing import Any, Dict, List
from pydantic import BaseModel, Field

class DataLoaderConfig(BaseModel):
    batch_size: int = Field(default=32, title="Batch Size")
    num_workers: int = Field(default=91, title="DataLoader Workers")
    shuffle: bool = Field(default=True, title="Shuffle Data")
