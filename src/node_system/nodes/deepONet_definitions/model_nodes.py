import torch.nn as nn
from pydantic import BaseModel, Field
from typing import List, Optional

from src.node_system.configs.model import CompositeModelConfig
from src.node_system.core import Node, Port, PortType, NodeMetadata, register_node
import torch.nn as nn
import torch

class FourierFeatureEncoding(nn.Module):
    """
        Fourier Feature Mapping as described in "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains, THIS GOOD for High Freqeuncy changes"
    """
    def __init__(self, input_dim, mapping_size, scale=10.0):
        super().__init__()
        # B * inputs -> features
        #self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x: [batch, points, input_dim]
        # proj: [batch, points, mapping_size]
        proj = torch.matmul(x, self.B) * 2 * torch.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class FlexDeepONet(nn.Module):

    """
        This is a custome Neural Network class for a FlexDeepONet architecture, that allows multiple branch networks with different input sizes
        as well as 2 outputs (Temperature and Hydration Degree).
    """

    def __init__(self, branch_configs, trunk_config, num_outputs=2, latent_dim=256, activation=None, fourier_features=None):
        super().__init__()
        self.num_outputs = num_outputs
        self.latent_dim = latent_dim

        if isinstance(activation, dict):
            branch_act = activation.get('branch', nn.Tanh)
            trunk_act = activation.get('trunk', nn.SiLU)
        else:
            branch_act = activation or nn.Tanh
            trunk_act = activation or nn.SiLU

        # ========== BRANCH NETWORKS ==========
        self.branch_nets = nn.ModuleList()
        self.num_branches = len(branch_configs)
        
        for config in branch_configs:
            layers = []
            in_dim = config['input_size']
            hidden = config.get('hidden_layers', [128, 128])
            
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(branch_act())
                in_dim = h
            
            layers.append(nn.Linear(in_dim, latent_dim))
            self.branch_nets.append(nn.Sequential(*layers))
        
        # ========== FOURIER ENCODING (TRUNK INPUT) ==========
        if fourier_features is not None:
            self.fourier_encoding = FourierFeatureEncoding(**fourier_features)
            # Fourier features: [sin(2πZx), cos(2πZx)] → 2 * mapping_size
            trunk_input_size = 2 * fourier_features['mapping_size']
        else:
            self.fourier_encoding = None
            trunk_input_size = trunk_config['input_size']

        # ========== TRUNK NETWORK ==========
        layers = []
        in_dim = trunk_input_size
        hidden = trunk_config.get('hidden_layers', [128, 128])
        
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(trunk_act())
            in_dim = h
        
        self.trunk_output_dim = (latent_dim * self.num_branches) * num_outputs
        layers.append(nn.Linear(in_dim, self.trunk_output_dim))
        
        self.trunk_net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, branch_inputs, trunk_input):
        batch_size, num_points, coord_dim = trunk_input.shape
        
        # ========== BRANCHES ==========
        branch_outputs = []
        for i, branch_net in enumerate(self.branch_nets):
            branch_outputs.append(branch_net(branch_inputs[i]))
        
        branch_combined = torch.cat(branch_outputs, dim=1)
        
        # ========== FOURIER ENCODING (TRUNK) ==========
        if self.fourier_encoding is not None:
            trunk_input = self.fourier_encoding(trunk_input)  # [batch, points, 128]
        
        # ========== TRUNK FORWARD ==========
        # coord_dim either 4 or 128 (if FOURIER)
        trunk_input_flat = trunk_input.reshape(batch_size * num_points, -1)
        trunk_output = self.trunk_net(trunk_input_flat)
        
        trunk_output = trunk_output.reshape(
            batch_size, 
            num_points, 
            self.latent_dim * self.num_branches, 
            self.num_outputs
        )

        # ========== BRANCH-TRUNK INTERACTION ==========
        branch_combined = branch_combined.unsqueeze(1).unsqueeze(-1)
        product = trunk_output * branch_combined
        output = product.sum(dim=2)
        
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)



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
        return [Port("model", PortType.MODEL)]

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
        if hasattr(m_cfg, "fourier_features"):
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

        return {"model": model}


