# src/node_system/nodes/solver_nodes.py

import torch
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
from src.DeepONet.training_pipline import DeepONetSolver
from src.state_management.config import TrainingConfig

@register_node("deeponet_solver")
class DeepONetSolverNode(Node):
    """
    Constructs the DeepONetSolver (LightningModule).
    This defines the physics, loss functions, and optimization strategy.
    It does NOT start training.
    """
    
    @classmethod
    def get_input_ports(cls) -> Dict[str, Port]:
        return {
            "model": Port(PortType.MODEL),
            "problem": Port(PortType.PROBLEM),
            "training_config": Port(PortType.CONFIG, required=False), # Config needed for optimizer/loss params
        }

    @classmethod
    def get_output_ports(cls) -> Dict[str, Port]:
        return {
            "solver": Port(PortType.SOLVER, description="Initialized DeepONetSolver instance")
        }

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            category="Solver",
            display_name="Solver Builder",
            description="Initializes the physics solver and optimizer settings",
            icon="function"
        )

    @classmethod
    def get_config_schema(cls):
        return TrainingConfig

    def execute(self) -> Dict[str, Any]:
        model = self.inputs["model"]
        problem = self.inputs["problem"]
        
        t_cfg = self.inputs.get("training_config")
        if not t_cfg: t_cfg = self.config

        # Construct Loss Weights
        weights = {
            'physics': t_cfg.loss_weights.get("physics", 1.0),
            'bc': t_cfg.loss_weights.get("bc", 1.0),
            'ic': t_cfg.loss_weights.get("ic", 1.0)
        }
        
        time_weighted_loss = getattr(t_cfg, "time_weighted_loss", None)

        solver = DeepONetSolver(
            problem=problem,
            model=model,
            optimizer=t_cfg.optimizer,
            loss_weights=weights,
            time_weighted_loss=time_weighted_loss
        )
        
        return {"solver": solver}
