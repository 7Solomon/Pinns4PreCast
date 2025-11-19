import torch.nn as nn
import torch


class FlexDeepONet(nn.Module):
    def __init__(self, branch_configs, trunk_config, num_outputs=2, latent_dim=100, activation=None):
        """
        Args:
            branch_configs (list[dict]): List of configs for each branch network.
                Each dict should have 'input_size' and optional 'hidden_layers'.
            trunk_config (dict): Config for the single trunk network.
                Should have 'input_size' (e.g., 4 for x,y,z,t) and optional 'hidden_layers'.
            num_outputs (int): Number of output fields (e.g., 2 for temperature + hydration).
            latent_dim (int): Latent dimension for branch-trunk interaction.
            activation: Activation function or dict with 'branch' and 'trunk' keys.
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.latent_dim = latent_dim

        # Get activations
        if isinstance(activation, dict):
            branch_activation = activation.get('branch', nn.Tanh)
            trunk_activation = activation.get('trunk', nn.Tanh)
        else:
            branch_activation = activation or nn.Tanh
            trunk_activation = activation or nn.Tanh

        # Build branch networks
        self.branch_nets = nn.ModuleList()
        for config in branch_configs:
            layers = []
            in_dim = config['input_size']
            hidden = config.get('hidden_layers', [128, 128])
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(branch_activation())
                in_dim = h
            layers.append(nn.Linear(in_dim, latent_dim))
            self.branch_nets.append(nn.Sequential(*layers))

        # Build ONE shared trunk network
        layers = []
        in_dim = trunk_config['input_size']
        hidden = trunk_config.get('hidden_layers', [128, 128])
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(trunk_activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim * num_outputs))  # Output for all fields
        self.trunk_net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def forward(self, branch_inputs, trunk_input):
        """
        Args:
            branch_inputs (list[torch.Tensor]): List of tensors, each [batch_size, sensor_count_i].
            trunk_input (torch.Tensor): Tensor of shape [batch_size, num_points, 4] (x,y,z,t).
        
        Returns:
            torch.Tensor: Predictions of shape [batch_size, num_points, num_outputs].
        """
        batch_size, num_points, coord_dim = trunk_input.shape
        
        # Process branch networks: [batch_size, sensor_count] -> [batch_size, latent_dim]
        branch_outputs = []
        for i, branch_net in enumerate(self.branch_nets):
            b = branch_net(branch_inputs[i])  # [batch_size, latent_dim]
            branch_outputs.append(b)
        
        # Combine all branch outputs via element-wise product
        branch_combined = branch_outputs[0]
        for b in branch_outputs[1:]:
            branch_combined = branch_combined * b  # [batch_size, latent_dim]
        
        # Process trunk network: [batch_size, num_points, 4] -> [batch_size, num_points, latent_dim * num_outputs]
        trunk_input_reshaped = trunk_input.reshape(batch_size * num_points, coord_dim)
        trunk_output = self.trunk_net(trunk_input_reshaped)  # [batch_size * num_points, latent_dim * num_outputs]
        trunk_output = trunk_output.reshape(batch_size, num_points, self.latent_dim, self.num_outputs)
        
        # Aggregate: branch-trunk product + sum reduction
        # branch_combined: [batch_size, latent_dim] -> [batch_size, 1, latent_dim, 1]
        branch_combined = branch_combined.unsqueeze(1).unsqueeze(-1)
        
        # Element-wise product: [batch_size, num_points, latent_dim, num_outputs]
        product = trunk_output * branch_combined
        
        # Sum over latent dimension: [batch_size, num_points, num_outputs]
        output = product.sum(dim=2)
        
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
