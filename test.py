from pina.solver import PINN
import torch.nn as nn

from DeepONet.deepONet import FlexDeepONet
from DeepONet.infrence_pipiline import testFlexDeepONet
from problem_definition import HeatODE
from DeepONet.trainer import DeepONetTrainer
from lightning.pytorch.callbacks import ModelCheckpoint



def defineFlexDeepONet(n_bc_features, n_ic_features): 
    latent_dim = 50
    
    # Define configurations for branch networks
    branch_configs = [
        # TEMP
        {
            'input_size': n_bc_features,
            'hidden_layers': [128, 128]  # optional hidden layers
        },
        {
            'input_size': n_ic_features,
            'hidden_layers': [128, 128]
        }
    ]
    
    # Define configurations for trunk networks (one per output)
    trunk_configs = [
        {
            'input_size': 4,  # (x, y, z, t) for Temperature
            'hidden_layers': [128, 128]
        },
        {
            'input_size': 4,  # (x, y, z, t) for alpha
            'hidden_layers': [128, 128]
        }
    ]

    activation = {
        'branch': nn.Tanh,
        'trunk': [nn.Tanh, nn.Sigmoid]
    }
    
    # Create the model
    model = FlexDeepONet(
        branch_configs=branch_configs,
        trunk_configs=trunk_configs,
        latent_dim=latent_dim,
        activation=activation
    )

    problem = HeatODE()
    problem.discretise_domain(n=100, mode="random")


    return model, problem

def trainFlexDeepONet(model, problem):   
    solver = PINN(problem, model)
    trainer = DeepONetTrainer(
        solver, 
        max_epochs=1000, 
        accelerator='gpu',
        callbacks=[]
    )
    trainer.train()


if __name__ == "__main__":
    n_bc_features = 100  # features for BC branch
    n_ic_features = 100  # features for IC branch  

    model, problem = defineFlexDeepONet(n_bc_features, n_ic_features)
    trainFlexDeepONet(model, problem)
    testFlexDeepONet(model, problem)