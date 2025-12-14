import torch.nn as nn
from pydantic import BaseModel, Field
from typing import List, Optional

from src.DeepONet.model_definition import FlexDeepONet 
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
from src.state_management.config import CompositeModelConfig


@register_node("flex_deeponet")
class FlexDeepONetNode(Node):
    @classmethod
    def get_input_ports(cls):
        return [
            Port("input_config", PortType.CONFIG, required=False),
            Port("model_config", PortType.CONFIG, required=False)
        ] 

    @classmethod
    def get_output_ports(cls):
        return [Port("model_instance", PortType.MODEL)]

    @classmethod
    def get_metadata(cls):
        return NodeMetadata(
            category="Model",
            display_name="Flex DeepONet",
            description="Multi-branch DeepONet with optional Fourier features",
            icon="network-wired" # You can use generic icon names for UI later
        )

    @classmethod
    def get_config_schema(cls):
        return CompositeModelConfig 

    def execute(self):
        i_cfg = self.inputs.get("input_config")
        m_cfg = self.inputs.get("model_config") 
        
        if not i_cfg: i_cfg = self.config.input_config
        if not m_cfg: m_cfg = self.config.model_hyperparameter
        
        branch_configs = [
            # Branch 0 (BC)
            {'input_size': i_cfg.num_sensors_bc, 'hidden_layers': m_cfg.branch_hidden_layers},
            # Branch 1 (IC)
            {'input_size': i_cfg.num_sensors_ic, 'hidden_layers': m_cfg.branch_hidden_layers}
        ]
        
        # 2. Handle Activation (String vs Dict)
        act_map = {
            "Tanh": nn.Tanh, "ReLU": nn.ReLU, "SiLU": nn.SiLU, "Sigmoid": nn.Sigmoid
        }

        # Check if activation is a Dict (from ModelConfig) or String (from FlexDeepONetConfig)
        activation_arg = None
        
        # Access the value safely (Pydantic models access fields as attributes)
        act_val = m_cfg.activation
        
        if isinstance(act_val, dict):
            # It's already a dict like {'branch': 'Tanh', ...}
            # We need to map the strings inside it to classes
            activation_arg = {
                'branch': act_map.get(act_val.get('branch', 'Tanh'), nn.Tanh),
                'trunk': act_map.get(act_val.get('trunk', 'SiLU'), nn.SiLU)
            }
        elif isinstance(act_val, str):
            # It's a single string like "Tanh"
            activation_arg = act_map.get(act_val, nn.Tanh)
        else:
            # Fallback
            activation_arg = nn.Tanh

        # 3. Helper to dump Pydantic to Dict
        def to_dict(obj):
            if isinstance(obj, dict): return obj
            return obj.model_dump() if hasattr(obj, 'model_dump') else obj.dict()

        # 4. Prepare Args
        trunk_dict = to_dict(m_cfg.trunk_config)
        
        fourier_args = None
        if m_cfg.fourier_features:
            fourier_args = to_dict(m_cfg.fourier_features)

        # 5. Instantiate
        model = FlexDeepONet(
            branch_configs=branch_configs,
            trunk_config=trunk_dict,
            num_outputs=m_cfg.num_outputs,
            latent_dim=m_cfg.latent_dim,
            activation=activation_arg, # Pass the processed arg
            fourier_features=fourier_args
        )

        return {"model_instance": model}


