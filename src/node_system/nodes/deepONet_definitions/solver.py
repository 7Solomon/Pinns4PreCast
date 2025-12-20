import torch
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from src.node_system.configs.training import TrainingConfig
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
from pina.condition import Condition
from pina.condition.domain_equation_condition import DomainEquationCondition
from pina.solver import SingleSolverInterface
from pina.label_tensor import LabelTensor
import torch
from pina.optim import TorchOptimizer


class DeepONetSolver(SingleSolverInterface):
    """
    Custom solver for Physics-Informed DeepONet training.

    SO THAT WE CAN HAVE CUSTOM LOSS FUNCTIONS FOR BC, IC AND PDE RESIDUAL
    """

    accepted_conditions_types = (DomainEquationCondition, Condition)

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        loss_weights={'physics': 1.0, 'bc': 1.0, 'ic': 1.0},
        time_weighted_loss=None
    ):
        """
        Args:
            problem: PINA problem instance
            model: FlexDeepONet model
            optimizer: PINA optimizer (if None, uses default Adam)
            scheduler: PINA scheduler (if None, uses default ConstantLR)
            loss_weights: Dict with 'data' and 'physics' loss weights
        """

        if optimizer is None:
            optimizer = TorchOptimizer(torch.optim.Adam, lr=0.001)
        
        super().__init__(
            problem=problem,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            use_lt=False  # FOR LabelTensor conversion but we do manually
        )
        self.loss_weights = loss_weights
        self.time_weighted_loss = time_weighted_loss

    def setup(self, stage):
        """Override setup to avoid PINA-specific trainer attributes."""  # JUST PASS BECAUSE 
        pass
    def on_train_start(self):
        """
        OVERRIDE: This method replaces the buggy method in PINA's library.
        The original method tries to access 'self.trainer.compile', which 
        does not exist in my version of Lightning.
        """
        pass

    
    def forward(self, batch):
        """
        Forward pass for DeepONet.
        
        Args:
            batch: Dict containing:
                - 'bc_sensors': [batch_size, num_bc_sensors]
                - 'ic_sensors': [batch_size, num_ic_sensors]
                - 'query_coords': [batch_size, num_points, 4] (x,y,z,t)
        """
        bc_sensor_values = batch['bc_sensor_values']
        ic_sensor_values = batch['ic_sensor_values']
        query_coords = batch['query_coords']
        
        predictions = self.model([bc_sensor_values, ic_sensor_values], query_coords)
        
        return predictions
    
    def loss_phys(self, batch):
        """
        Compute physics-informed loss (PDE residuals) - only on interior points.
        """
        # Get PDE points
        pde_coords = batch['pde_coords']  # [batch_size, n_pde, 4]
        batch_size, n_pde, _ = pde_coords.shape
        
        # Flatten for forward pass
        pde_coords_flat = pde_coords.reshape(-1, 4)
        pde_coords_labeled = LabelTensor(pde_coords_flat, labels=['x', 'y', 'z', 't'])
        pde_coords_labeled.requires_grad_(True)

        pde_coords_reshaped = pde_coords_labeled.reshape(batch_size, n_pde, 4)
        
        # Create batch dict for forward pass
        pde_batch = {
            'bc_sensor_values': batch['bc_sensor_values'],  # [batch_size, num_sensors]
            'ic_sensor_values': batch['ic_sensor_values'],  # [batch_size, num_sensors, 2]
            'query_coords': pde_coords_reshaped
        }
        
        # Get predictions
        predictions = self.forward(pde_batch)  # [batch_size, n_pde, 2]
        
        
        pred_flat = predictions.reshape(-1, 2)
        pred_labeled = LabelTensor(pred_flat, labels=['T', 'alpha'])
        
        # Compute PDE residuals
        heat_residual = self.problem.conditions['physi_T'].equation.residual(
            pde_coords_labeled, pred_labeled
        )
        alpha_residual = self.problem.conditions['physi_alpha'].equation.residual(
            pde_coords_labeled, pred_labeled
        )

        time_weight = 1.0
        if self.time_weighted_loss and 'time_decay_rate' in self.time_weighted_loss:
            t = pde_coords_flat[..., 3]
            time_weight = torch.exp(-self.time_weighted_loss['time_decay_rate'] * t)

        loss_heat = torch.mean((heat_residual.as_subclass(torch.Tensor)**2) * time_weight)
        loss_alpha = torch.mean((alpha_residual.as_subclass(torch.Tensor)**2) * time_weight)
        
        self.log('loss_phys_temperature', loss_heat)
        self.log('loss_phys_alpha', loss_alpha)
        
        return loss_heat + loss_alpha

    def loss_bc(self, batch):
        """
        Compute boundary condition loss.
        """
        bc_coords = batch['bc_coords']  # [batch_size, n_bc, 4]
        bc_target_temperature = batch['bc_target_temperature']  # [batch_size, n_bc]
        
        batch_size, n_bc, _ = bc_coords.shape
        
        # Create batch dict for forward pass
        bc_batch = {
            'bc_sensor_values': batch['bc_sensor_values'],
            'ic_sensor_values': batch['ic_sensor_values'],
            'query_coords': bc_coords
        }
        
        # Get predictions
        predictions = self.forward(bc_batch)  # [batch_size, n_bc, 2]
        pred_T = predictions[..., 0]  # [batch_size, n_bc] - Temperature predictions
        
        # MSE between predicted and target temperature
        loss = torch.nn.functional.mse_loss(pred_T, bc_target_temperature)

        self.log('loss_bc_temperature', loss)
        
        return loss

    def loss_ic(self, batch):
        """
        Compute initial condition loss.
        """
        ic_coords = batch['ic_coords']  # [batch_size, n_ic, 4]
        ic_target_temperature = batch['ic_target_temperature']  # [batch_size, n_ic]
        ic_target_alpha = batch['ic_target_alpha']  # [batch_size, n_ic]
        
        batch_size, n_ic, _ = ic_coords.shape
        
        # Create batch dict for forward pass
        ic_batch = {
            'bc_sensor_values': batch['bc_sensor_values'],
            'ic_sensor_values': batch['ic_sensor_values'],
            'query_coords': ic_coords
        }
        
        # Get predictions
        predictions = self.forward(ic_batch)  # [batch_size, n_ic, 2]
        pred_T = predictions[..., 0]  # Temperature
        pred_alpha = predictions[..., 1]  # Degree of hydration
        
        # IC losses
        loss_T = torch.nn.functional.mse_loss(pred_T, ic_target_temperature)        
        loss_alpha = torch.nn.functional.mse_loss(pred_alpha, ic_target_alpha)

        self.log('loss_ic_temperature', loss_T)
        self.log('loss_ic_alpha', loss_alpha)
    
        return loss_T + loss_alpha
    
    def optimization_cycle(self, batch):
        """
        Required by PINA: compute all losses for the batch.
        """
        loss_pde = self.loss_phys(batch)
        loss_bc = self.loss_bc(batch)
        loss_ic = self.loss_ic(batch)
        
        return {
            'pde': loss_pde,
            'bc': loss_bc,
            'ic': loss_ic
        }
    
    #def loss_data(self, samples):
    #    """
    #    Compute supervised data loss (if training data available).
    #    """
    #    if 'target' not in samples:
    #        return torch.tensor(0.0)
    #    
    #    predictions = self.forward(samples)
    #    target = samples['target']
    #    
    #    return torch.nn.functional.mse_loss(predictions, target)
    
    def training_step(self, batch, batch_idx):
        """
        Single training step combining data + physics losses.
        """
        # Calculate all losses
        loss_p = self.loss_phys(batch)
        loss_b = self.loss_bc(batch)
        loss_i = self.loss_ic(batch)
        
        # Combine losses
        total_loss = (
            self.loss_weights.get('physics', 1.0) * loss_p + 
            self.loss_weights.get('bc', 1.0) * loss_b + 
            self.loss_weights.get('ic', 1.0) * loss_i
        )
        
        # Log losses
        #self.log('loss_physics', loss_p)
        #self.log('loss_bc', loss_b)
        #self.log('loss_ic', loss_i)
        
        self.log('loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        loss_p = self.loss_phys(batch)
        loss_b = self.loss_bc(batch)
        loss_i = self.loss_ic(batch)
        
        total_loss = (
            self.loss_weights['physics'] * loss_p + 
            loss_b + 
            loss_i
        )
        
        self.log('val_loss_physics', loss_p)
        self.log('val_loss_bc', loss_b)
        self.log('val_loss_ic', loss_i)

        self.log('loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """
        Override configure_optimizers to support ReduceLROnPlateau.
        """
        optimizer_class = self.optimizer.optimizer_class
        optimizer_kwargs = self.optimizer.kwargs
        
        opt = optimizer_class(self.model.parameters(), **optimizer_kwargs)       
        
        if self.scheduler is None:
            return opt
        
        scheduler_class = self.scheduler.scheduler_class
        scheduler_kwargs = self.scheduler.kwargs
        
        sched = scheduler_class(opt, **scheduler_kwargs)
        
        # Check if we are using ReduceLROnPlateau
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "loss_epoch",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
            
        return [opt], [sched]
    


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
            "model": Port("model", PortType.MODEL),
            "problem": Port("problem", PortType.PROBLEM),
            "training_config": Port("training_config", PortType.CONFIG, required=False), 
        }

    @classmethod
    def get_output_ports(cls) -> Dict[str, Port]:
        return {
            "solver": Port("solver", PortType.SOLVER, description="Initialized DeepONetSolver instance")
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
