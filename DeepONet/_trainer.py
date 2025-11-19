from pina import LabelTensor
from pina.trainer import Trainer
from torch.utils.data import DataLoader
import tqdm
import torch

from DeepONet.data_loader import PINNDataset
from material import ConcreteData
from utils import scale_T, torch_interp1d, unscale_T, unscale_alpha

material_data = ConcreteData()

#class DeepONetTrainer(Trainer):
#    def train(self):
#        model = self.solver.model
#        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#        device = next(model.parameters()).device
#        problem = self.solver.problem
#
#        self.loss_history = { "total": [], "pde": [], "T_ic": [], "alpha_ic": [], "bc": [] }
#        self.weights = {"pde": 1.0, "T_ic": 10.0, "alpha_ic": 10.0, "bc": 10.0}
#        best_loss = float('inf')
#
#        n_pde, n_ic, n_bc_face = 500, 100, 50
#        batch_size = 32
#
#        dataset = PINNDataset(problem, n_pde, n_ic, n_bc_face, batch_size)
#        dataloader = DataLoader(dataset, batch_size=None, num_workers=4, pin_memory=True)
#
#        progress_bar = tqdm.tqdm(enumerate(dataloader), total=self.max_epochs, desc="Training")
#        for epoch, batch in progress_bar:
#            
#            if epoch >= self.max_epochs:
#                break
#
#            model.train()
#
#            # --- FINALIZED DATA FLOW ---
#
#            # 1. Get branch inputs
#            bc_input_unscaled = batch["bc_input"].to(device, non_blocking=True)
#            ic_input_unscaled = batch["ic_input"].to(device, non_blocking=True)
#            branch_inputs = [bc_input_unscaled, ic_input_unscaled]
#            
#            # 2. Reconstruct LabelTensors (on CPU) and set gradient tracking
#            pde_pts_lt = LabelTensor(batch["pde_pts_tensor"], problem.domain['D'].variables)
#            pde_pts_lt.requires_grad_(True)
#            
#            ic_pts_lt = LabelTensor(batch["ic_pts_tensor"], problem.domain['initial'].variables)
#            bc_pts_lt = LabelTensor(batch["bc_pts_tensor"], problem.domain['left'].variables)
#
#            sizes = {'pde': len(pde_pts_lt), 'ic': len(ic_pts_lt), 'bc': len(bc_pts_lt)}
#            
#            trunk_pts = LabelTensor.cat([pde_pts_lt, ic_pts_lt, bc_pts_lt])
#            trunk_tensor = trunk_pts.tensor.to(device, non_blocking=True)
#            trunk_tensor = trunk_tensor.clone().detach().requires_grad_(True)  
#            trunk_batched = trunk_tensor.unsqueeze(0).expand(batch_size, -1, -1)
#            trunk_inputs = [trunk_batched, trunk_batched]
#            
#
#            print('branch_inputs shapes:', [bi.shape for bi in branch_inputs])
#            print('trunk_inputs shapes:', [ti.shape for ti in trunk_inputs])
#            predictions = model(branch_inputs, trunk_inputs)
#            print('predictions shape:', predictions.shape)            
#            pde_pred, ic_pred, bc_pred = torch.split(predictions, list(sizes.values()), dim=1)
#            
#            # GET
#            #pde_pts_tensor = trunk_tensor[:sizes['pde'], :].clone().detach().requires_grad_(True)  
#            #pde_pts_expanded_tensor = pde_pts_tensor  # make so that points are BATCH axis size
#            #pde_pts_expanded = LabelTensor(pde_pts_expanded_tensor, pde_pts_lt.labels)
#
#
#            # --- Vectorized Loss Calculation ---
#            output_pde = LabelTensor(pde_pred, problem.output_variables)
#            input_pde = [LabelTensor(ti, problem.domain['D'].variables) for ti in trunk_inputs]
#            res_T = problem.conditions["physi_T"].equation.residual(input_pde, output_pde)
#            res_alpha = problem.conditions["physi_alpha"].equation.residual(input_pde, output_pde)
#            loss_pde = (res_T**2).mean() + (res_alpha**2).mean()
#
#            output_ic = LabelTensor(ic_pred, problem.output_variables)
#            T_pred_ic_scaled = output_ic.extract(["T"])
#            T_target_unscaled = ic_input_unscaled[:, 0].unsqueeze(-1).unsqueeze(-1)
#            T_target_scaled = scale_T(T_target_unscaled)
#            loss_T_ic = ((T_pred_ic_scaled - T_target_scaled) ** 2).mean()
#            loss_alpha_ic = (output_ic.extract(['alpha']) ** 2).mean()
#
#            output_bc = LabelTensor(bc_pred, problem.output_variables)
#            T_pred_bc_scaled = output_bc.extract(["T"])
#            t_boundary = bc_pts_lt.extract(['t']).to(device, non_blocking=True).squeeze(-1).contiguous()
#            xp = torch.linspace(0, 1, 100, device=device)
#            fp = bc_input_unscaled
#            T_target_bc_unscaled = torch_interp1d(t_boundary, xp, fp).unsqueeze(-1)
#            T_target_bc_scaled = scale_T(T_target_bc_unscaled)
#            loss_bc = ((T_pred_bc_scaled - T_target_bc_scaled) ** 2).mean()
#            
#            # --- Backpropagation and Logging ---
#            loss = (self.weights["pde"] * loss_pde + self.weights["T_ic"] * loss_T_ic + self.weights["alpha_ic"] * loss_alpha_ic + self.weights["bc"] * loss_bc)
#            
#            optimizer.zero_grad(set_to_none=True)
#            loss.backward()
#            optimizer.step()
#
#            self.loss_history["total"].append(loss.item())
#            self.loss_history["pde"].append(loss_pde.item())
#            self.loss_history["T_ic"].append(loss_T_ic.item())
#            self.loss_history["alpha_ic"].append(loss_alpha_ic.item())
#            self.loss_history["bc"].append(loss_bc.item())
#
#            progress_bar.set_postfix(loss=loss.item())
#            
#            if loss.item() < best_loss:
#                best_loss = loss.item()
#                if epoch % 100 == 0:
#                    torch.save({
#                        'epoch': epoch,
#                        'model_state_dict': model.state_dict(),
#                        'optimizer_state_dict': optimizer.state_dict(),
#                        'loss': best_loss,
#                    }, 'checkpoints/best_model.pth')
#        
#        # Save final model
#        torch.save({
#            'epoch': self.max_epochs,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': loss.item(),
#        }, 'checkpoints/final_model.pth')
#        
#        print(f"Best model saved with loss: {best_loss:.6f}")
