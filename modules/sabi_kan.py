import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvGNAct


class RegionSABIKAN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 4, num_bases: int = 6, state_dim: int = 64):
        super().__init__()
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        self.groups = groups
        self.num_bases = num_bases
        self.group_channels = out_channels // groups
        self.state_dim = state_dim

        self.pre = ConvGNAct(in_channels, out_channels, kernel_size=1)
        self.state_proj = nn.Sequential(
            nn.Linear(out_channels, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(state_dim, groups * num_bases)
        self.mu_head = nn.Linear(state_dim, groups * num_bases)
        self.sigma_head = nn.Linear(state_dim, groups * num_bases)
        self.mix_head = nn.Linear(state_dim, groups * num_bases)
        self.gate_head = nn.Linear(state_dim, groups)

        self.post = ConvGNAct(out_channels, out_channels, kernel_size=1)

    def forward(self, x, state_hint=None):
        z = self.pre(x)
        b, c, h, w = z.shape

        pooled = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
        if state_hint is not None:
            pooled = pooled + F.adaptive_avg_pool2d(state_hint, output_size=1).flatten(1)[:, :pooled.shape[1]]

        s = self.state_proj(pooled)

        alpha = self.alpha_head(s).view(b, self.groups, self.num_bases)
        alpha = torch.softmax(alpha, dim=-1)

        mu = self.mu_head(s).view(b, self.groups, self.num_bases)
        sigma = F.softplus(self.sigma_head(s)).view(b, self.groups, self.num_bases) + 1e-3
        mix = torch.sigmoid(self.mix_head(s)).view(b, self.groups, self.num_bases)
        gate = torch.sigmoid(self.gate_head(s)).view(b, self.groups, 1, 1, 1)

        z_group = z.view(b, self.groups, self.group_channels, h, w)
        x_exp = z_group.unsqueeze(3)  # [B,G,Cg,1,H,W]

        alpha = alpha[:, :, None, :, None, None]
        mu = mu[:, :, None, :, None, None]
        sigma = sigma[:, :, None, :, None, None]
        mix = mix[:, :, None, :, None, None]

        u = (x_exp - mu) / sigma
        gaussian = torch.exp(-(u ** 2))
        odd_edge = u * torch.exp(-(u ** 2))
        basis = mix * gaussian + (1.0 - mix) * odd_edge

        response = (alpha * basis).sum(dim=3)
        y = z_group + gate * response
        y = y.reshape(b, c, h, w)
        y = self.post(y)
        return y
