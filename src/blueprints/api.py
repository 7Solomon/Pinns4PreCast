import json
import pandas as pd
import torch
from flask import Blueprint, redirect, request, jsonify, url_for
import torch.nn as nn
from torch.utils.data import DataLoader
from pina.optim import TorchOptimizer
from pina.model import DeepONet, FeedForward
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import os
import threading
import numpy as np
from src.DeepONet.callback import VisualizationCallback
from src.DeepONet.infrence_pipline import testFlexDeepONet
from src.DeepONet.logger import DashboardLogger
from src.DeepONet.model_definition import FlexDeepONet
from problem_definition import HeatODE
from src.DeepONet.dataset import DeepONetDataset, deeponet_collate_fn
from src.DeepONet.training_pipline import DeepONetSolver

from src.state_management.state import State

api_bp = Blueprint('api', __name__, url_prefix='/api')
CONSOLE_OUTPUT = False 


#@api_bp.route('/define_model', methods=['POST'])
#def define_model_basic():
#    return redirect(url_for('api.define_model_flexnet'))

@api_bp.route('/define_model', methods=['POST'])
def define_flex_model():
    """
    Define the FlexDeepONet model and problem instance using data from the request.
    """
    if State().config is None:
        return jsonify({"error": "State not initialized. Please go to /setup first."}), 400

    model = FlexDeepONet(
        branch_configs=State().config.model.branch_configs,
        trunk_config=State().config.model.trunk_config,
        num_outputs=State().config.model.num_outputs,
        latent_dim=State().config.model.latent_dim,
        activation=State().config.model.activation,
        fourier_features=State().config.model.fourier_features
    )
    problem = HeatODE()
    
    # STATE SETTING
    State().model = model
    State().problem = problem
    return jsonify({"message": "Model and problem defined successfully"})

@api_bp.route('/define_model_pina', methods=['POST'])
def define_model_pina():
    total_sensors = State().config.model.num_sensors_bc + State().config.model.num_sensors_ic 

    raise NotImplementedError("To Use PINA DeepONet model, the dataloader aswell as the output pipiline needs an change.")
    model = DeepONet(
        net_branch = FeedForward(
            input_dimension=total_sensors, 
            output_dimension=State().config.model.latent_dim,
            layers=[256, 256, 256],
            func=torch.nn.Tanh
        ),
        net_trunk = FeedForward(
            input_dimension=4, # x,y,z,t
            output_dimension=State().config.model.latent_dim,
            layers=[256, 256, 256],
            func=torch.nn.SiLU
        ),
        reduction='sum',
    )
    problem = HeatODE()
    
    # STATE SETTING
    State().model = model
    State().problem = problem
    return jsonify({"message": "Model and problem defined successfully"})

@api_bp.route('/define_training_pipeline', methods=['POST'])
def define_training_pipeline():
    """
    Define the training pipeline using the model and problem stored in state.
    """
    if State().model is None or State().problem is None:
        return jsonify({"error": "Model or Problem not defined in state"}), 400

    
    train_dataset = DeepONetDataset(
        problem=State().problem,
        n_pde=State().config.dataset.n_pde,
        n_ic=State().config.dataset.n_ic,
        n_bc_face=State().config.dataset.n_bc_face,
        num_samples=State().config.dataset.num_samples,
        num_sensors_bc=State().config.model.num_sensors_bc,
        num_sensors_ic=State().config.model.num_sensors_ic
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=State().config.dataset.batch_size,
        shuffle=True,
        collate_fn=deeponet_collate_fn,
        num_workers=State().config.dataset.num_workers
    )
    solver = DeepONetSolver(
        model=State().model,
        problem=State().problem,
        optimizer=State().config.training.optimizer,
        scheduler=State().config.training.scheduler,
        loss_weights=State().config.training.loss_weights,
        time_weighted_loss=State().config.training.time_weighted_loss
    )
    
    # STATE
    State().solver = solver
    State().dataloader = train_loader
    
    return jsonify({"message": "Training pipeline created"})

def run_training_background(trainer: pl.Trainer, solver, dataloader):
    try:
        trainer.fit(solver, train_dataloaders=dataloader)
    except Exception as e:
        print(f"Training failed: {e}")

@api_bp.route('/train', methods=['POST'])
def train():
    if State().solver is None or State().dataloader is None:
        return jsonify({"error": "Solver or Dataloader not defined"}), 400
    
    State().kill_training_signal = False

    run_id = State().directory_manager.create_new_run()

    checkpoint_callback = ModelCheckpoint(
        dirpath=State().directory_manager.checkpoint_path,
        filename='deeponet-{epoch:02d}',
        monitor='loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=CONSOLE_OUTPUT
    )
    visualization_callback = VisualizationCallback(
        vtk_path=State().directory_manager.vtk_path, 
        sensor_temp_path=State().directory_manager.sensor_temp_path, 
        sensor_alpha_path=State().directory_manager.sensor_alpha_path,
        plot_every_n_epochs=10
    )

    ## DEPRECAED LOGGER
    #logger = TensorBoardLogger(
    #    save_dir=State().directory_manager.runs_path,
    #    name=State().directory_manager.log_name
    #)
    dash_logger = DashboardLogger(
        save_dir=State().directory_manager.runs_path, 
        version=State().directory_manager.run_idx_path
    )
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        logger=[dash_logger],  # CAN be both loggers
        callbacks=[checkpoint_callback, visualization_callback],
        log_every_n_steps=10,
        enable_progress_bar=CONSOLE_OUTPUT,
        enable_model_summary=CONSOLE_OUTPUT
    )
    State().trainer = trainer

    thread = threading.Thread(
        target=run_training_background, 
        args=(trainer, State().solver, State().dataloader),
        daemon=True
    )
    thread.start()
    
    #trainer.fit(State().solver, train_dataloaders=State().dataloader)
    
    return jsonify({
        "message": "Training started in background", 
        "run_id": State().directory_manager.run_idx_path
    })


@api_bp.route('/stop_training', methods=['POST'])
def stop_training():
    """Gracefully stops the PyTorch Lightning training."""
    try:
        trainer = getattr(State(), 'trainer', None)
        if trainer:
            trainer.should_stop = True
            return jsonify({"message": "Signal sent to stop training. It will finish the current batch."})
        else:
            return jsonify({"error": "No active trainer found in State."}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/kill_training', methods=['POST'])
def kill_training():
    """
    Stops training and sets a flag to skip post-training visualizations.
    """
    try:
        trainer = getattr(State(), 'trainer', None)
        if trainer:
            State().kill_training_signal = True             
            trainer.should_stop = True
            return jsonify({"message": "Training killed. Visualization will be skipped."})
        else:
            return jsonify({"error": "No active trainer found."}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
