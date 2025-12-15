# src/node_system/nodes/inference_nodes.py (add to existing file)

import torch
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node


class CheckpointLoaderConfig(BaseModel):
    """Configuration for checkpoint loading behavior."""
    strict_loading: bool = Field(
        default=False, 
        title="Whether to enforce strict key matching when loading state dict"
    )
    map_location: str = Field(
        default="cpu",
        title="Device to map tensors to during loading ('cpu', 'cuda', 'cuda:0', etc.)"
    )
    strip_prefix: str = Field(
        default="model.",
        title="Prefix to strip from checkpoint keys (Lightning often adds 'model.' prefix)"
    )


@register_node("checkpoint_loader")
class CheckpointLoaderNode(Node):
    """
    Loads trained model weights from a PyTorch Lightning checkpoint file.
    
    This node enables inference on previously trained models by loading
    saved weights into a model architecture instance.
    """
    
    @classmethod
    def get_input_ports(cls) -> Dict[str, Port]:
        return {
            "model": Port(
                port_type=PortType.MODEL,
                required=True,
                description="The model architecture instance to load weights into"
            ),
            "checkpoint_path": Port(
                port_type=PortType.CONFIG,
                required=True,
                description="File path to the .ckpt file"
            )
        }
    
    @classmethod
    def get_output_ports(cls) -> Dict[str, Port]:
        return {
            "loaded_model": Port(
                port_type=PortType.MODEL,
                required=False,
                description="Model instance with loaded weights"
            )
        }
    
    @classmethod
    def get_metadata(cls) -> NodeMetadata:
        return NodeMetadata(
            display_name="Checkpoint Loader",
            category="Inference",
            description="Loads trained weights from .ckpt file into model",
            icon="download"
        )
    
    @classmethod
    def get_config_schema(cls) -> type[BaseModel] | None:
        return CheckpointLoaderConfig
    
    def execute(self) -> Dict[str, Any]:
        """
        Loads model weights from checkpoint file.
        
        Returns:
            Dict with 'loaded_model' key containing model with loaded weights
            
        Raises:
            FileNotFoundError: If checkpoint path doesn't exist
            RuntimeError: If state_dict loading fails
        """
        model = self.inputs.get("model")
        checkpoint_path = self.inputs.get("checkpoint_path")
        config: CheckpointLoaderConfig = self.config
        
        # Validate inputs
        if model is None:
            raise ValueError("Model input is required")
        if checkpoint_path is None:
            raise ValueError("Checkpoint path input is required")
        
        # Validate checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            # Load checkpoint from file
            print(f"[CheckpointLoader] Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=config.map_location)
            
            # Extract state_dict (handle both wrapped and raw checkpoints)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"[CheckpointLoader] Loaded wrapped checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
            else:
                state_dict = checkpoint
                print("[CheckpointLoader] Loaded raw state dict")
            
            # Clean state_dict keys
            cleaned_state_dict = {}
            stripped_keys = []
            
            for key, value in state_dict.items():
                new_key = key
                
                # Strip configured prefix (e.g., "model.")
                if config.strip_prefix and key.startswith(config.strip_prefix):
                    new_key = key[len(config.strip_prefix):]
                    stripped_keys.append(key)
                
                # Handle PINA's potential prefix
                if new_key.startswith("_pina_models.0."):
                    new_key = new_key[len("_pina_models.0."):]
                    stripped_keys.append(key)
                
                cleaned_state_dict[new_key] = value
            
            if stripped_keys:
                print(f"[CheckpointLoader] Stripped prefix from {len(stripped_keys)} keys")
            
            # Load state dict into model
            missing_keys, unexpected_keys = model.load_state_dict(
                cleaned_state_dict, 
                strict=config.strict_loading
            )
            
            # Report key matching status
            if missing_keys:
                print(f"[CheckpointLoader] Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"[CheckpointLoader] Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
            if not missing_keys and not unexpected_keys:
                print("[CheckpointLoader] All keys matched successfully")
            
            # Set model to evaluation mode
            model.eval()
            print("[CheckpointLoader] Model set to eval mode")
            
            return {"loaded_model": model}
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}") from e
