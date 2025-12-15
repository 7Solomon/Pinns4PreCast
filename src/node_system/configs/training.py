from typing import Any, Dict, List
from pydantic import BaseModel, Field

import torch
from pina.optim import TorchOptimizer
from pina.optim import TorchScheduler

class TrainingConfig(BaseModel):
    max_epochs: int = Field(default=100, title="Max Epochs")
    loss_weights: Dict[str, float] = Field(default_factory=lambda: {'physics': 100.0, 'bc': 1.0, 'ic': 10.0}, title="Loss Weights")
    time_weighted_loss: Dict[str, Any] = Field(default_factory=lambda: {'time_decay_rate': 5.0}, title="Time Weighted Loss")
    
    optimizer_type: str = Field(default='Adam', title="Optimizer")
    optimizer_learning_rate: float = Field(default=1e-4, title="Learning Rate")

    accelerator: str = Field(default='auto', title="Accelerator")

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
