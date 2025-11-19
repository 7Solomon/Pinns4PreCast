from pina import Trainer
from pina.optim import TorchOptimizer
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import os

from DeepONet.deepONet import FlexDeepONet
from DeepONet.infrence_pipline import testFlexDeepONet
from DeepONet.training_pipline import DeepONetSolver
from DeepONet.data_loader import DeepONetDataset, deeponet_collate_fn
from problem_definition import HeatODE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def defineFlexDeepONet(n_bc_sensor_points, n_ic_sensor_points): 
    branch_configs = [
        {'input_size': n_bc_sensor_points, 'hidden_layers': [128, 128]},
        {'input_size': n_ic_sensor_points, 'hidden_layers': [128, 128]}
    ]
    
    trunk_config = {'input_size': 4, 'hidden_layers': [128, 128]}  # x,y,z,t
   
    model = FlexDeepONet(
        branch_configs=branch_configs,
        trunk_config=trunk_config,
        num_outputs=2, # temp and alpha
        latent_dim=100,
        activation={'branch': nn.Tanh, 'trunk': nn.Tanh}
    )
    
    problem = HeatODE()
    
    return model, problem

def trainFlexDeepONet(model, problem):   
    #n_pde, n_ic, n_bc_face = 500, 100, 50
   # batch_size = 32

    optimizer = TorchOptimizer(torch.optim.Adam, lr=1e-3)
    solver = DeepONetSolver(
        problem=problem,
        model=model,
        optimizer=optimizer,
    )
    
    trainer = Trainer(solver, max_epochs=1000, accelerator='gpu')
    trainer.train()


def define_trainFlexDeepONet_with_lightning(model, problem):  # benutzt nicht Trainer von PINA
    n_pde, n_ic, n_bc_face = 500, 100, 50
    batch_size = 32
    num_samples = 10000  # Total training samples

    train_dataset = DeepONetDataset(
        problem=problem,
        n_pde=n_pde,
        n_ic=n_ic,
        n_bc_face=n_bc_face,
        num_samples=num_samples,
        num_sensors=100,
        device=device
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=deeponet_collate_fn,
        num_workers=0  # CAN BE CHAGNED JUST FOR DEBUGGING
    ) 
    optimizer = TorchOptimizer(torch.optim.Adam, lr=1e-3)
    solver = DeepONetSolver(
        problem=problem,
        model=model,
        optimizer=optimizer,
    )
    return solver, train_loader, train_dataset


def train(solver, train_loader):
    ## CALLBACK
    CONTENT_PATH = os.path.join('content')
    os.makedirs(CONTENT_PATH, exist_ok=True)

    entries = os.listdir(CONTENT_PATH)
    idx = len(entries) + 1
    idx_path = os.path.join(CONTENT_PATH, str(idx))
    os.makedirs(idx_path, exist_ok=True)

    checkpoints_dir = os.path.join(idx_path, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)


    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='deeponet-{epoch:02d}',#-{oss:.4f}',
        monitor='loss',
        mode='min',
        save_top_k=3,  # Keep top 3 models
        save_last=True,  # Also save the last checkpoint
        verbose=True
    )
    logger = TensorBoardLogger(
        save_dir=idx_path,
        name='logs'
    )
    
    trainer = pl.Trainer(
        max_epochs=6,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    trainer.fit(solver, train_dataloaders=train_loader)
    #best_checkpoint = checkpoint_callback.best_model_path
    return trainer, idx_path



if __name__ == "__main__":
    n_bc_sensor_points = 100
    n_ic_sensor_points = 100

    model, problem = defineFlexDeepONet(n_bc_sensor_points, n_ic_sensor_points)
    solver, train_loader, train_dataset = define_trainFlexDeepONet_with_lightning(model, problem)
    trainer, idx_path = train(solver, train_loader)

    predictions, coords = testFlexDeepONet(
        solver, 
        idx_path=idx_path,
        n_spatial=10,
        n_time=30
    )