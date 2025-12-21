from lightning.pytorch.callbacks import ModelCheckpoint
from pydantic import BaseModel, Field
import os
from src.node_system.configs.checkpoint import CheckpointConfig
from src.node_system.core import Node, NodeMetadata, PortType, Port, register_node


@register_node("checkpoint_callback")
class CheckpointCallbackNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
        ]

    @classmethod
    def get_output_ports(cls):
        return [Port("callback", PortType.CALLBACK)] 

    @classmethod
    def get_metadata(cls):
        return NodeMetadata("Training", "Checkpoint Callback", "Saves model checkpoints", icon="save")

    @classmethod
    def get_config_schema(cls):        
        return CheckpointConfig

    def execute(self):
        run_id = self.context.get("run_id")
        
        ckpt_dir = os.path.join("content/runs", run_id, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Create ModelCheckpoint
        cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch={epoch:03d}-step={step:08d}-{monitor:.4f}",
            monitor=self.config.monitor,
            mode="min",
            save_top_k=self.config.save_top_k,
            save_last=self.config.save_last,
            every_n_epochs=self.config.every_n_epochs,
            verbose=True
        )
        
        return {"callback": cb}
