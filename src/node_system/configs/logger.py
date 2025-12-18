from typing import Optional
from pydantic import BaseModel, Field


class LoggerConfig(BaseModel):
    save_dir: str = Field(default="content/runs", title="Runs Directory")
    version: Optional[str] = Field(default=None, title="Run Name (auto-generate if empty)")
    save_graph: bool = Field(default=True, title="Save graph.json to run directory")