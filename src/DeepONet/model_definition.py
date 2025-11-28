import torch.nn as nn
import torch

from src.DeepONet.fourier_features import FourierFeatureEncoding

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
