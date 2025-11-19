from pina.condition import Condition
from pina.condition.domain_equation_condition import DomainEquationCondition
from pina.solver import SingleSolverInterface
from pina.label_tensor import LabelTensor
import torch
from pina.optim import TorchOptimizer


class DeepONetSolver(SingleSolverInterface):
    """
    Custom solver for Physics-Informed DeepONet training.
    """

    accepted_conditions_types = (DomainEquationCondition, Condition)

    def __init__(
        self,
        problem,
        model,
        optimizer=None,
        scheduler=None,
        loss_weights={'data': 1.0, 'physics': 1.0}
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

    def setup(self, stage):
        """Override setup to avoid PINA-specific trainer attributes."""  # JUST PASS BECAUSE 
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
        bc_sensors = batch['bc_sensors']
        ic_sensors = batch['ic_sensors']
        query_coords = batch['query_coords']
        
        predictions = self.model([bc_sensors, ic_sensors], query_coords)
        
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
            'bc_sensors': batch['bc_sensors'],  # [batch_size, num_sensors]
            'ic_sensors': batch['ic_sensors'],  # [batch_size, num_sensors]
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
        
        loss_heat = torch.mean(heat_residual.as_subclass(torch.Tensor) ** 2)
        loss_alpha = torch.mean(alpha_residual.as_subclass(torch.Tensor) ** 2)
        
        return loss_heat + loss_alpha

    def loss_bc(self, batch):
        """
        Compute boundary condition loss.
        """
        bc_coords = batch['bc_coords']  # [batch_size, n_bc, 4]
        bc_target = batch['bc_target']  # [batch_size, n_bc]
        
        batch_size, n_bc, _ = bc_coords.shape
        
        # Create batch dict for forward pass
        bc_batch = {
            'bc_sensors': batch['bc_sensors'],
            'ic_sensors': batch['ic_sensors'],
            'query_coords': bc_coords
        }
        
        # Get predictions
        predictions = self.forward(bc_batch)  # [batch_size, n_bc, 2]
        pred_T = predictions[..., 0]  # [batch_size, n_bc] - Temperature predictions
        
        # MSE between predicted and target temperature
        loss = torch.nn.functional.mse_loss(pred_T, bc_target)
        
        return loss

    def loss_ic(self, batch):
        """
        Compute initial condition loss.
        """
        ic_coords = batch['ic_coords']  # [batch_size, n_ic, 4]
        ic_target = batch['ic_target']  # [batch_size, n_ic, 2]
        
        batch_size, n_ic, _ = ic_coords.shape
        
        # Create batch dict for forward pass
        ic_batch = {
            'bc_sensors': batch['bc_sensors'],
            'ic_sensors': batch['ic_sensors'],
            'query_coords': ic_coords
        }
        
        # Get predictions
        predictions = self.forward(ic_batch)  # [batch_size, n_ic, 2]
        pred_T = predictions[..., 0]  # Temperature
        pred_alpha = predictions[..., 1]  # Degree of hydration
        
        # IC losses
        loss_T = torch.nn.functional.mse_loss(pred_T, ic_target)
        loss_alpha = torch.nn.functional.mse_loss(pred_alpha, torch.zeros_like(pred_alpha))
        
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
        #loss_d = self.loss_data(batch)
        loss_p = self.loss_phys(batch)
        
        total_loss = (
        #    self.loss_weights['data'] * loss_d +
            self.loss_weights['physics'] * loss_p
        )
        
        # Log losses
        #self.log('loss_data', loss_d)
        self.log('loss_physics', loss_p)
        self.log('loss', total_loss)
        #if batch_idx % 10 == 0:
        #    print(f'Train Step {batch_idx}, Loss: {total_loss.item():.6f}')
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        #loss_d = self.loss_data(batch)
        loss_p = self.loss_phys(batch)
        
        total_loss = (
        #    self.loss_weights['data'] * loss_d +
            self.loss_weights['physics'] * loss_p
        )
        
        #self.log('val_loss_data', loss_d)
        self.log('val_loss_physics', loss_p)
        self.log('val_loss', total_loss)
        
        return total_loss
    