from typing import Any, Dict, List
from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    num_sensors_bc: int = Field(default=100, title="BC Sensor Count")
    num_sensors_ic: int = Field(default=100, title="IC Sensor Count")

