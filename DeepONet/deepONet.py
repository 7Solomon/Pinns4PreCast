import torch.nn as nn
import torch

class FlexDeepONet(nn.Module):
    def __init__(self, branch_configs, trunk_configs, latent_dim=100, activation=None):
        super().__init__()
        self.num_outputs = len(trunk_configs)

        # Get branch/trunk activation
        branch_activation = activation['branch'] if isinstance(activation, dict) else activation
        trunk_activations = activation.get('trunk', [nn.Identity]*self.num_outputs) if isinstance(activation, dict) else [activation]*self.num_outputs

        # Build branch nets
        self.branch_nets = nn.ModuleList()
        for config in branch_configs:
            layers = []
            in_dim = config['input_size']
            hidden = config.get('hidden_layers', [])
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(branch_activation())
                in_dim = h
            layers.append(nn.Linear(in_dim, latent_dim))
            self.branch_nets.append(nn.Sequential(*layers))

        # Build trunk nets
        self.trunk_nets = nn.ModuleList()
        for i, config in enumerate(trunk_configs):
            layers = []
            in_dim = config['input_size']
            hidden = config.get('hidden_layers', [])
            for h in hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(trunk_activations[i]())
                in_dim = h
            layers.append(nn.Linear(in_dim, latent_dim))
            self.trunk_nets.append(nn.Sequential(*layers))
        self.apply(self._init_weights) 

    def forward(self, branch_inputs, trunk_inputs):
        """
        Args:
            branch_inputs: list of tensors, each [batch, branch_input_size_i]
            trunk_inputs: list of tensors, each [batch, trunk_input_size_i]
            
        Returns:
            output: [batch, num_trunks]
                output[:, j] = sum(branch_1 * trunk_j * branch_2 * ... * branch_n)
        """
        batch_size = branch_inputs[0].shape[0]
        
        # Process all branch networks
        branch_outputs = []
        for i, branch_net in enumerate(self.branch_nets):
            b = branch_net(branch_inputs[i])  # [batch, latent_dim]
            branch_outputs.append(b)
        
        # Process all trunk networks
        trunk_outputs = []
        for i, trunk_net in enumerate(self.trunk_nets):
            t = trunk_net(trunk_inputs[i])  # [batch, latent_dim]
            trunk_outputs.append(t)
        
        # For each trunk, compute product of all branches and that trunk
        outputs = []
        for trunk_out in trunk_outputs:
            # Start with the trunk
            product = trunk_out  # [batch, latent_dim]
            
            # Multiply by all branches element-wise
            for branch_out in branch_outputs:
                product = product * branch_out  # [batch, latent_dim]
            
            # Sum over latent dimension to get scalar output
            output = torch.sum(product, dim=1, keepdim=True)  # [batch, 1]
            outputs.append(output)
        
        # Concatenate all outputs
        result = torch.cat(outputs, dim=1) 
        result = torch.stack([
            torch.clamp(result[:, 0], min=0.0, max=1),
            torch.clamp(result[:, 1], min=0.0, max=1.1)
        ], dim=1)
        return result
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavivier init
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
