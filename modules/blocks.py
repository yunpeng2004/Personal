import torch.nn as nn


def _gn_groups(channels: int):
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_ch, bias=False),
            nn.GroupNorm(_gn_groups(in_ch), in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
