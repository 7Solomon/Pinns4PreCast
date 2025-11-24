import torch.nn as nn
import torch

class FlexDeepONet(nn.Module):
    def __init__(self, branch_configs, trunk_config, num_outputs=2, latent_dim=256, activation=None):
        """
        Args:
            branch_configs (list[dict]): Configs for each branch (BC, IC).
            trunk_config (dict): Config for trunk.
            num_outputs (int): Output fields (T, alpha).
            latent_dim (int): Dimension of the latent space.
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.latent_dim = latent_dim

        if isinstance(activation, dict):
            branch_act = activation.get('branch', nn.Tanh)
            trunk_act = activation.get('trunk', nn.SiLU)
        else:
            branch_act = activation or nn.Tanh
            trunk_act = activation or nn.SiLU

        # BRANCH NETWORKS
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
            
            # OUTPUT LAYER: Project to 'latent_dim'
            layers.append(nn.Linear(in_dim, latent_dim))
            self.branch_nets.append(nn.Sequential(*layers))

        # TRUNK NETWORK
        # CONCAT branches, so effective latent size is latent_dim * num_branches
        layers = []
        in_dim = trunk_config['input_size']
        hidden = trunk_config.get('hidden_layers', [128, 128])
        
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(trunk_act())
            in_dim = h
            
        # TRUNK OUTPUT:
        # Trunk Output Size is (latent_dim * num_branches) * num_outputs
        self.trunk_output_dim = (latent_dim * self.num_branches) * num_outputs
        layers.append(nn.Linear(in_dim, self.trunk_output_dim))
        
        self.trunk_net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, branch_inputs, trunk_input):
        batch_size, num_points, coord_dim = trunk_input.shape
        
        # --- A. Process Branches ---
        branch_outputs = []
        for i, branch_net in enumerate(self.branch_nets):
            # Shape: [batch_size, latent_dim]
            branch_outputs.append(branch_net(branch_inputs[i]))
            
        # Shape: [batch_size, latent_dim * num_branches]
        branch_combined = torch.cat(branch_outputs, dim=1)
        
        # TRUNK
        # [batch_size * num_points, coord_dim]
        trunk_input_flat = trunk_input.reshape(batch_size * num_points, coord_dim)
        trunk_output = self.trunk_net(trunk_input_flat)
        
        # [batch_size, num_points, (latent_dim * num_branches), num_outputs]
        trunk_output = trunk_output.reshape(
            batch_size, 
            num_points, 
            self.latent_dim * self.num_branches, 
            self.num_outputs
        )

        # Branch: [batch_size, latent_dim_total] -> [batch_size, 1, latent_dim_total, 1]
        branch_combined = branch_combined.unsqueeze(1).unsqueeze(-1)
        
        # Multiply: [batch_size, num_points, latent_dim_total, num_outputs]
        product = trunk_output * branch_combined
        
        # Sum over the latent dimension to get final prediction
        # Shape: [batch_size, num_points, num_outputs]
        output = product.sum(dim=2)
        
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
