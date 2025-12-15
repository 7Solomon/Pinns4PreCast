import lightning.pytorch as pl
from typing import Dict, Any, List

from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
from src.state_management.config import TrainingConfig

@register_node("lightning_trainer")
class LightningTrainerNode(Node):
    """
    Wraps the PyTorch Lightning Trainer.
    Takes a solver and data, runs training, and returns the trained solver.
    """

    @classmethod
    def get_input_ports(cls) -> Dict[str, Port]:
        return {
            "solver": Port(PortType.SOLVER, required=True, description="Initialized LightningModule"),
            "dataloader": Port(PortType.DATALOADER, required=True, description="Train DataLoader"),
            "val_dataloader": Port(PortType.DATALOADER, required=False, description="Validation DataLoader"),
            "callbacks": Port("callback", required=False),
            "logger": Port("logger", required=False),
            "training_config": Port(PortType.CONFIG, required=False)
        }

    @classmethod
    def get_output_ports(cls) -> Dict[str, Port]:
        return {
            "trained_solver": Port(PortType.SOLVER),
            "checkpoint_path": Port(PortType.CONFIG)
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
        solver = self.inputs["solver"]
        train_loader = self.inputs["dataloader"]
        val_loader = self.inputs.get("val_dataloader")
        
        t_cfg = self.inputs.get("training_config")
        if not t_cfg:
            t_cfg = self.config

        # Handle Callbacks
        callbacks_input = self.inputs.get("callbacks")
        callbacks_list = []
        if callbacks_input:
            if isinstance(callbacks_input, list):
                callbacks_list.extend(callbacks_input)
            else:
                callbacks_list.append(callbacks_input)

        logger = self.inputs.get("logger")

        # Create Trainer
        trainer = pl.Trainer(
            max_epochs=t_cfg.max_epochs,
            callbacks=callbacks_list,
            logger=logger if logger else False,
            accelerator=t_cfg.accelerator,
            enable_checkpointing=bool(callbacks_list), # Only enable if we have callbacks (like CheckpointSaver)
        )

        print(f"[Trainer] Starting fit for {t_cfg.max_epochs} epochs...")
        
        # Execute Training
        trainer.fit(
            model=solver,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

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
