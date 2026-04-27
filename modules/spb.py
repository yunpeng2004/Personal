import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DepthwiseSeparableConv
from .sabi_kan import RegionSABIKAN
from .wavelet import dwt2d


class WaveletRegionKANSPB(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, state_channels: int, kan_groups: int, kan_bases: int):
        super().__init__()
        self.pre = ConvGNAct(in_channels, hidden_channels, kernel_size=1)

        self.ll_refine = DepthwiseSeparableConv(hidden_channels, hidden_channels)
        self.hf_fuse = ConvGNAct(hidden_channels * 3, hidden_channels, kernel_size=1)
        self.hf_refine = DepthwiseSeparableConv(hidden_channels, hidden_channels)

        self.region_kan = RegionSABIKAN(hidden_channels, state_channels, groups=kan_groups, num_bases=kan_bases)
        self.boundary_head = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1),
        )
        self.confidence_head = nn.Sequential(
            ConvGNAct(hidden_channels * 2, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1),
        )

        self.region_logit = nn.Conv2d(state_channels, 1, kernel_size=1)
        self.boundary_logit = nn.Conv2d(state_channels, 1, kernel_size=1)
        self.confidence_logit = nn.Conv2d(state_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h0, w0 = x.shape[-2:]
        z = self.pre(x)

        ll, lh, hl, hh = dwt2d(z)
        ll = self.ll_refine(ll)
        hf = self.hf_refine(self.hf_fuse(torch.cat([lh, hl, hh], dim=1)))

        region_half = self.region_kan(ll)
        boundary_half = self.boundary_head(hf)
        confidence_half = self.confidence_head(torch.cat([ll, hf], dim=1))

        region = F.interpolate(region_half, size=(h0, w0), mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary_half, size=(h0, w0), mode="bilinear", align_corners=False)
        confidence = F.interpolate(confidence_half, size=(h0, w0), mode="bilinear", align_corners=False)

        region_logit = self.region_logit(region)
        boundary_logit = self.boundary_logit(boundary)
        confidence_logit = self.confidence_logit(confidence)

        boundary_prob = torch.sigmoid(boundary_logit)
        confidence_prob = torch.sigmoid(confidence_logit)

        confidence_gate = 0.5 + 0.5 * confidence_prob
        boundary_gate = 1.0 - 0.5 * boundary_prob
        region = region * confidence_gate * boundary_gate
        boundary = boundary * (0.5 + 0.5 * boundary_prob)
        confidence = confidence * confidence_gate

        tensor = torch.cat([region, boundary, confidence], dim=1)
        return {
            "tensor": tensor,
            "components": {
                "region": region,
                "boundary": boundary,
                "confidence": confidence,
            },
            "maps": {
                "region_logit": region_logit,
                "boundary_logit": boundary_logit,
                "confidence_logit": confidence_logit,
            },
        }
