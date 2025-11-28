import torch
import torch.nn as nn

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
