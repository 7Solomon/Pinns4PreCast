from pina import LabelTensor
from pina.trainer import Trainer
import tqdm
import torch

from material import ConcreteData
material_data = ConcreteData()

def sample_temperature_bc(num, num_sensors=100):
    xs = torch.linspace(0, 1, num_sensors)
    samples = []
    for _ in range(num):
        amplitude = 4 + torch.rand(1) * 2        # [4, 6]
        phase     = torch.rand(1) * 2 * torch.pi # random phase
        offset    = material_data.Temp_ref

        T_ambient = offset + amplitude * torch.sin(0.5 * torch.pi * xs + phase)
        samples.append(T_ambient)
    return torch.stack(samples)

def sample_temperature_ic(num, num_sensors=100):
    samples = []
    for _ in range(num):
        T0 = material_data.Temp_ref + 2 * torch.randn(1)      # small variability
        samples.append(T0 * torch.ones(num_sensors))
    return torch.stack(samples)



class DeepONetTrainer(Trainer):
    def train(self):
        model = self.solver.model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = next(model.parameters()).device
        
        # Create checkpoint directory
        import os
        os.makedirs('checkpoints', exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in tqdm.tqdm(range(self.max_epochs)):
            model.train()
            batch_size = 32
            
            # Generate branch inputs
            bc_input = sample_temperature_bc(batch_size, num_sensors=100).to(device)
            ic_input = sample_temperature_ic(batch_size, num_sensors=100).to(device)
            
            # Sample trunk input
            trunk_input_sample = self.solver.problem.domain["D"].sample(n=batch_size)
            trunk_input_sample = trunk_input_sample.to(device)
            trunk_input_sample.requires_grad_(True) 
            
            # Extract tensor for model input
            if hasattr(trunk_input_sample, 'tensor'):
                trunk_coords = trunk_input_sample.tensor
            else:
                trunk_coords = trunk_input_sample.as_subclass(torch.Tensor)
            
            # Forward pass
            branch_inputs = [bc_input, ic_input]
            trunk_inputs = [trunk_coords, trunk_coords]
            pred = model(branch_inputs, trunk_inputs)
            
            # Create output LabelTensor
            output = LabelTensor(pred, ["T", "alpha"])
            
            # Compute residual using LabelTensor
            condition = self.solver.problem.conditions["physi"]
            residual = condition.equation.residual(trunk_input_sample, output)
            
            # Compute loss
            loss = (residual ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'checkpoints/best_model.pth')
            
            # Save checkpoint every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}, "
                      f"Residual = {residual.mean().item():.6f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, f'checkpoints/checkpoint_epoch_{epoch}.pth')
        
        # Save final model
        torch.save({
            'epoch': self.max_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, 'checkpoints/final_model.pth')
        
        print(f"Best model saved with loss: {best_loss:.6f}")
