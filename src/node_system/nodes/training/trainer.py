import lightning.pytorch as pl

from typing import Dict, Any, List

from src.node_system.session import register_session, unregister_session
from src.node_system.configs.training import TrainingConfig
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node



@register_node("lightning_trainer")
class LightningTrainerNode(Node):
    """
    Wraps the PyTorch Lightning Trainer.
    Takes a solver and data, runs training, and returns the trained solver.
    """

    @classmethod
    def get_input_ports(cls) -> Dict[str, Port]:
        return {
            "solver": Port("solver", PortType.SOLVER, required=True, description="Initialized LightningModule"),
            "dataloader": Port("dataloader", PortType.DATALOADER, required=True, description="Train DataLoader"),
            "val_dataloader": Port("val_dataloader", PortType.DATALOADER, required=False, description="Validation DataLoader"),
            
            "callbacks": Port("callbacks", PortType.CALLBACK, required=False, multi_input=True),
            "logger": Port("logger", PortType.LOGGER, required=False),
            
            "training_config": Port("training_config", PortType.CONFIG, required=False)
        }

    @classmethod
    def get_output_ports(cls) -> Dict[str, Port]:
        return {
            "trained_solver": Port("trained_solver", PortType.SOLVER),
            "checkpoint_path": Port("checkpoint_path", PortType.CONFIG)
        }

    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            category="Training",
            display_name="Lightning Trainer",
            description="Executes the training loop",
            icon="play"
        )

    @classmethod
    def get_config_schema(cls):
        return TrainingConfig

    def execute(self) -> Dict[str, Any]:
        print("Traininer is NOw executing")
        solver = self.inputs["solver"]
        train_loader = self.inputs["dataloader"]
        val_loader = self.inputs.get("val_dataloader")
        
        t_cfg = self.inputs.get("training_config") or self.config
        run_id = self.context.get("run_id")

        # Handle Callbacks
        callbacks_input = self.inputs.get("callbacks")
        callbacks_list = []
        if callbacks_input:
            if isinstance(callbacks_input, list):
                callbacks_list.extend(callbacks_input)
            else:
                callbacks_list.append(callbacks_input)

        has_checkpoint_callback = any(isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in callbacks_list)
        logger = self.inputs.get("logger") or False
        
        # Create Trainer
        trainer = pl.Trainer(
            max_epochs=t_cfg.max_epochs,
            callbacks=callbacks_list,
            logger=logger,
            accelerator=t_cfg.accelerator,
            enable_checkpointing=has_checkpoint_callback,
            enable_progress_bar=False,
        )

        register_session(run_id, trainer)

        print(f"[Trainer] Starting fit for {t_cfg.max_epochs} epochs...")
        try:
            # Execute Training
            trainer.fit(
                model=solver,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )
        finally:
            unregister_session(run_id)

        # Retrieve best checkpoint if available
        best_path = ""
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                best_path = cb.best_model_path
                break

        return {
            "trained_solver": solver,
            "checkpoint_path": best_path
        }
