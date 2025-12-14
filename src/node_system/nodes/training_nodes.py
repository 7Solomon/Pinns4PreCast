import torch
from pydantic import BaseModel, Field
from typing import Dict, Optional

import lightning.pytorch as pl

from pina.optim import TorchOptimizer 
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
from src.DeepONet.training_pipline import DeepONetSolver
from src.state_management.config import TrainingConfig

@register_node("deeponet_solver")
class DeepONetSolverNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("model", PortType.MODEL),
            Port("problem", PortType.PROBLEM),
            Port("dataloader", PortType.DATALOADER),
            Port("training_config", PortType.CONFIG, required=False),

            Port("callbacks", "callback", required=False),
            Port("logger", "logger", required=False)
        ]

    @classmethod
    def get_output_ports(cls):
        return [
            Port("trained_solver", PortType.SOLVER),
            Port("checkpoint_path", PortType.CONFIG)
        ]

    @classmethod
    def get_metadata(cls):
        return NodeMetadata(
            category="Training",
            display_name="DeepONet Solver",
            description="Training loop with Configurable Loss Weights"
        )

    @classmethod
    def get_config_schema(cls):
        return TrainingConfig

    def execute(self):
        # 1. Unpack Inputs
        model = self.inputs["model"]
        problem = self.inputs["problem"]
        dataloader = self.inputs["dataloader"]
        t_cfg = self.inputs.get("training_config")

        if not t_cfg: t_cfg = self.config

        callbacks_input = self.inputs.get("callbacks")
        callbacks_list = []
        if callbacks_input:
            if isinstance(callbacks_input, list):
                callbacks_list.extend(callbacks_input)
            else:
                callbacks_list.append(callbacks_input)
                
        # 2. Handle Logger
        logger_input = self.inputs.get("logger")
        
        #  Construct Loss Weights Dict
        weights = {
            'physics': t_cfg.loss_weights["physics"],
            'bc': t_cfg.loss_weights["bc"],
            'ic': t_cfg.loss_weights["ic"]
        }
        if not hasattr(t_cfg, "time_weighted_loss"):
            time_weighted_loss = None
        else:
            time_weighted_loss = t_cfg.time_weighted_loss

        #  Instantiate Solver
        solver = DeepONetSolver(
            problem=problem,
            model=model,
            optimizer=t_cfg.optimizer,
            loss_weights=weights,
            time_weighted_loss=time_weighted_loss
        )
        
        # 5. Run Training with the MATCHING Trainer
        trainer = pl.Trainer(
            max_epochs=t_cfg.max_epochs,
            callbacks=callbacks_list,
            logger=logger_input if logger_input else False,
            accelerator=t_cfg.accelerator,
            enable_checkpointing=False,
        )
        print(f"DEBUG: DataLoader type: {type(dataloader)}")
        
        print(f"\n[Node System] Starting Training (Epochs: {t_cfg.max_epochs})...")
        trainer.fit(solver, dataloader)
        print("[Node System] Training Finished.\n")
        
        ckpt_path = trainer.checkpoint_callback.best_model_path
        
        return {
            "trained_solver": solver,
            "checkpoint_path": ckpt_path
        }
